#!/usr/bin/env python3
"""Matched-seed CA/MCE and upsampling comparisons. Reuse source checkpoints; train only missing variants."""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import statistics
import tempfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.nn.modules import C2fAxisResidual, GatedInterpolationUpsample
from ultralytics.nn.modules.comparisons import CoordinateAttention, CARAFEUpsample
from ultralytics.utils import YAML

CONFIGS = {
    "ca_giu": "yolov8n-fabric-ca-interp.yaml",
    "mce_bilinear": "yolov8n-fabric-axis-bilinear.yaml",
    "mce_carafe": "yolov8n-fabric-axis-carafe-reference.yaml",
}
REFERENCE = {"mce_nearest": "ablation-axis-gate/axis-only", "mce_giu": "axis-interp"}
PROTOCOL_KEYS = "epochs patience batch imgsz workers optimizer lr0 lrf momentum weight_decay warmup_epochs warmup_momentum warmup_bias_lr box cls dfl nbs hsv_h hsv_s hsv_v degrees translate scale shear perspective flipud fliplr bgr mosaic mixup cutmix copy_paste copy_paste_mode close_mosaic amp deterministic cos_lr fraction freeze multi_scale rect single_cls classes agnostic_nms max_det".split()
EVAL = dict(split="test", imgsz=640, batch=16, workers=8, conf=0.001, iou=0.7, max_det=16, half=False, augment=False, rect=True, plots=False)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
        f.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        tmp = Path(f.name)
    tmp.replace(path)


def dataset_digest(data):
    from ultralytics.data.utils import check_det_dataset
    d = check_det_dataset(str(data), autodownload=False)
    h = hashlib.sha256(Path(data).read_bytes())
    count = {}
    for split in ("train", "val", "test"):
        values = d[split] if isinstance(d[split], list) else [d[split]]
        files = set()
        for value in values:
            p = Path(value)
            if not p.is_dir():
                raise ValueError("This dataset audit requires directory-based splits.")
            files.update(x for x in p.rglob("*") if x.is_file())
            label = Path(str(p).replace("/images/", "/labels/"))
            files.update(label.rglob("*.txt"))
        for p in sorted(files):
            h.update(str(p).encode())
            h.update(bytes.fromhex(digest(p)))
        count[split] = len(files)
    return h.hexdigest(), count


def protocol(args, seeds):
    source = ROOT / "runs/detect/axis-interp/1/args.yaml"
    base = YAML.load(source)
    effective = torch.load(source.parent / "weights/best.pt", map_location="cpu", weights_only=False)["train_args"]
    common = {k: effective[k] for k in PROTOCOL_KEYS if k in effective}
    if base.get("pretrained") != "yolov8n.pt":
        raise ValueError("Reference initialization is not the expected explicit yolov8n.pt.")
    for group in REFERENCE.values():
        for seed in seeds:
            p = ROOT / "runs/detect" / group / str(seed)
            saved = YAML.load(p / "args.yaml")
            checkpoint_args = torch.load(p / "weights/best.pt", map_location="cpu", weights_only=False)["train_args"]
            differences = {k: [common[k], checkpoint_args.get(k)] for k in common if common[k] != checkpoint_args.get(k)}
            for k in [*common, "seed", "pretrained"]:
                if checkpoint_args.get(k) != saved.get(k):
                    if k == "warmup_bias_lr" and saved.get("optimizer") == "auto" and checkpoint_args[k] == 0:
                        continue  # build_optimizer normalizes this value when optimizer=auto.
                    raise ValueError(f"Checkpoint/args.yaml mismatch at {p}: {k}")
            if differences or saved["seed"] != seed or not (p / "weights/best.pt").is_file():
                raise ValueError(f"Reference mismatch: {p}: {differences}")
    data = Path(base["data"])
    signature, counts = dataset_digest(data)
    return common, data, signature, counts


def complexity(network):
    """Count Conv2d MACs at actual 640 input plus explicitly defined comparison-operator arithmetic.

    FLOPs = 2*conv MACs + pooling/gating/interpolation/reassembly arithmetic in comparison modules.
    Generic nonlinear activations, backbone pooling and memory copies are excluded for every variant.
    """
    params_unfused = sum(p.numel() for p in network.parameters())
    model = copy.deepcopy(network).cpu().float().eval().fuse(verbose=False)
    for layer in model.modules():
        if isinstance(layer, CoordinateAttention) and isinstance(layer.bn, nn.BatchNorm2d):
            layer.reduce = torch.nn.utils.fuse_conv_bn_eval(layer.reduce, layer.bn)
            layer.bn = nn.Identity()
    core, extra, hooks = [0], [0], []
    def conv(m, xs, out):
        core[0] += 2 * out.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
    def custom(m, xs, out):
        x = xs[0]; b,c,h,w = x.shape
        if isinstance(m, nn.Upsample) and m.mode == "bilinear":
            extra[0] += 7 * out.numel()
        elif isinstance(m, GatedInterpolationUpsample):
            extra[0] += 10 * out.numel()  # bilinear: 4 multiply + 3 add; blend: subtract/multiply/add
        elif isinstance(m, CoordinateAttention):
            extra[0] += 4 * x.numel()  # two full-axis means and two spatial multiplications
        elif isinstance(m, C2fAxisResidual):
            extra[0] += 8 * b * m.active * h * w  # two means, broadcast sum, centered residual modulation
        elif isinstance(m, CARAFEUpsample):
            extra[0] += (2 * m.up_kernel**2 - 1) * out.numel()
    for m in model.modules():
        if isinstance(m, nn.Conv2d): hooks.append(m.register_forward_hook(conv))
        if isinstance(m, (nn.Upsample, GatedInterpolationUpsample, CoordinateAttention, C2fAxisResidual, CARAFEUpsample)):
            hooks.append(m.register_forward_hook(custom))
    with torch.inference_mode(): model(torch.zeros(1,3,640,640))
    for h in hooks: h.remove()
    return dict(params=params_unfused, params_fused=sum(p.numel() for p in model.parameters()), gflops=(core[0]+extra[0])/1e9,
                conv_gflops=core[0]/1e9, comparison_arithmetic_gflops=extra[0]/1e9)


def evaluate(weight, output, seed, variant, data, signature, common):
    result_file = output / variant / str(seed) / "metrics.json"
    checkpoint_hash = digest(weight)
    if result_file.exists():
        existing = json.loads(result_file.read_text())
        if existing["checkpoint_sha256"] != checkpoint_hash or existing["dataset_sha256"] != signature or existing["eval_protocol"] != EVAL:
            raise ValueError(f"Cached evaluation does not match: {result_file}")
        return
    detector = YOLO(str(weight))
    metrics = detector.val(data=str(data), device=OPTIONS.device, project=str(result_file.parent), name="evaluation", exist_ok=True, **EVAL)
    classes = {str(metrics.names[int(c)]): dict(zip(("P","R","AP50","AP50_95"), map(float,metrics.class_result(i)))) for i,c in enumerate(metrics.ap_class_index)}
    if set(classes) != {"row","col","hole"}: raise ValueError(f"Unexpected classes: {classes.keys()}")
    record = dict(variant=variant, seed=seed, checkpoint=str(weight), checkpoint_sha256=checkpoint_hash,
                  dataset_sha256=signature, eval_protocol=EVAL, train_protocol=common,
                  overall=dict(zip(("P","R","mAP50","mAP50_95"),map(float,metrics.mean_results()))), classes=classes,
                  complexity=complexity(detector.model))
    save(result_file, record)


def run(options):
    common,data,signature,counts = protocol(options,options.seeds)
    output = options.output.resolve()
    save(output/("protocol-" + options.variant + "-" + "-".join(map(str,options.seeds)) + ".json"),dict(train=common,evaluation=EVAL,data=str(data),dataset_sha256=signature,
                                   dataset_files=counts,seeds=options.seeds,pretrained_sha256=digest(ROOT/"yolov8n.pt"),
                                   comparison_design="CA vs MCE: GIU fixed; upsampling: MCE backbone fixed."))
    variants = REFERENCE if options.variant=="references" else {options.variant: CONFIGS[options.variant]}
    for variant in variants:
        for seed in options.seeds:
            if variant in REFERENCE:
                weight = ROOT/"runs/detect"/REFERENCE[variant]/str(seed)/"weights/best.pt"
            else:
                config = ROOT/"ultralytics/cfg/models/v8"/CONFIGS[variant]
                run_dir = output/variant/str(seed)/"train"
                weight = run_dir/"weights/best.pt"
                if not weight.exists():
                    if run_dir.exists(): raise FileExistsError(f"Incomplete training directory; inspect before retry: {run_dir}")
                    detector = YOLO(str(config))
                    detector.train(data=str(data), pretrained=str(ROOT/"yolov8n.pt"), seed=seed, device=options.device,
                                   project=str(run_dir.parent),name="train",exist_ok=False,**common)
                else:
                    a=YAML.load(run_dir/"args.yaml")
                    if a["seed"]!=seed or Path(a["model"]).name!=config.name or any(a.get(k)!=v for k,v in common.items()):
                        raise ValueError(f"Existing run protocol differs: {run_dir}")
            evaluate(weight,output,seed,variant,data,signature,common)
    summarize(output)


def summarize(output):
    records=[json.loads(p.read_text()) for p in sorted(output.glob("*/*/metrics.json"))]
    if not records: return
    signatures={r["dataset_sha256"] for r in records}
    if len(signatures)!=1: raise ValueError("Mixed dataset signatures")
    fields=["variant","seed","mAP50","mAP50_95","params","gflops"]+[f"{c}_{m}" for c in ("row","col","hole") for m in ("AP50","AP50_95")]
    rows=[]
    for r in records:
        row={"variant":r["variant"],"seed":r["seed"],**{k:100*r["overall"][k] for k in ("mAP50","mAP50_95")},"params":r["complexity"]["params"],"gflops":r["complexity"]["gflops"]}
        row.update({f"{c}_{m}":100*r["classes"][c][m] for c in ("row","col","hole") for m in ("AP50","AP50_95")})
        rows.append(row)
    with (output/"per_seed.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    summary={}
    for variant in sorted({r["variant"] for r in rows}):
        selected=[r for r in rows if r["variant"]==variant]
        summary[variant]={"seeds":[r["seed"] for r in selected],"metrics":{k:{"mean":statistics.mean(r[k] for r in selected),"std":statistics.stdev(r[k] for r in selected) if len(selected)>1 else None} for k in fields[2:]}}
    save(output/"summary.json",summary)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant",choices=[*CONFIGS,"references","summarize"],required=True)
    parser.add_argument("--device",default="0")
    parser.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
    parser.add_argument("--output",type=Path,default=ROOT/"runs/detect/targeted-mcgi")
    OPTIONS=parser.parse_args()
    torch.set_num_threads(2)
    if OPTIONS.variant=="summarize": summarize(OPTIONS.output)
    else: run(OPTIONS)
