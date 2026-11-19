#!/usr/bin/env python3
"""Export original YOLO annotations and native detector predictions with one drawing style."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
COLORS = ("#ff4444", "#0099ff", "#00bb66")


def files(root):
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not paths:
        raise ValueError(f"No images: {root}")
    return paths


def gt_boxes(path, size, names):
    """Read detection labels, failing on missing annotations rather than silently inventing empty GT."""
    width, height = size
    records = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        values = list(map(float, line.split()))
        if len(values) != 5 or not all(math.isfinite(v) for v in values):
            raise ValueError(f"Expected five finite YOLO detection values: {path}: {line}")
        cls, x, y, w, h = values
        if (cls != int(cls) or not 0 <= cls < len(names) or min(w, h) <= 0
                or not all(0 <= v <= 1 for v in (x, y, w, h))):
            raise ValueError(f"Invalid YOLO class or box: {path}: {line}")
        records.append({"class_id": int(cls), "xyxy": [(x-w/2)*width, (y-h/2)*height,
                                                         (x+w/2)*width, (y+h/2)*height], "score": None})
    return records


def class_mapping(source, target):
    if isinstance(source, dict):
        source = [source[i] for i in range(len(source))]
    if len(source) != len(target) or set(source) != set(target):
        raise ValueError(f"Checkpoint classes {source} do not match dataset classes {target}")
    return [target.index(name) for name in source]


def predictions(boxes, scores, labels, mapping, threshold, max_det):
    rows = []
    for box, score, label in zip(boxes, scores, labels):
        label = int(label)
        if not 0 <= label < len(mapping):
            raise ValueError(f"Invalid predicted class {label}")
        if not math.isfinite(float(score)) or not all(math.isfinite(float(v)) for v in box):
            raise ValueError("Non-finite predicted box/score")
        if float(score) >= threshold:
            rows.append({"class_id": mapping[label], "xyxy": list(map(float, box)), "score": float(score)})
    return sorted(rows, key=lambda row: row["score"], reverse=True)[:max_det]


def font(size):
    for name in ("DejaVuSans.ttf", "Arial.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw(image, records, names, args):
    result = image.copy()
    painter = ImageDraw.Draw(result)
    text_font = font(args.font_size)
    for row in records:
        color = COLORS[row["class_id"] % len(COLORS)]
        x1, y1, x2, y2 = row["xyxy"]
        x1, x2 = (max(0, min(image.width-1, v)) for v in (x1, x2))
        y1, y2 = (max(0, min(image.height-1, v)) for v in (y1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        painter.rectangle((x1, y1, x2, y2), outline=color, width=args.line_width)
        if not args.hide_labels:
            label = names[row["class_id"]]
            if args.show_conf and row["score"] is not None:
                label += f" {row['score']:.2f}"
            bounds = painter.textbbox((0, 0), label, font=text_font)
            tw, th = bounds[2]-bounds[0]+6, bounds[3]-bounds[1]+6
            tx, ty = min(x1, max(0, image.width-tw)), max(0, y1-th)
            painter.rectangle((tx, ty, tx+tw, ty+th), fill=color)
            painter.text((tx+3-bounds[0], ty+3-bounds[1]), label, font=text_font, fill="white")
    return result


def build_predictor(spec, names, args):
    device = "cpu" if args.device == "cpu" else f"cuda:{args.device}"
    if spec["backend"] == "ultralytics":
        from ultralytics import YOLO

        model = YOLO(spec["weights"])
        mapping = class_mapping(model.names, names)

        def predict(path):
            result = model.predict(str(path), imgsz=args.imgsz, device=args.device, conf=args.conf,
                                   iou=args.iou, max_det=args.max_det, rect=False, half=False,
                                   save=False, verbose=False)[0]
            boxes = result.boxes.cpu()
            return predictions(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist(),
                               mapping, args.conf, args.max_det)
        return predict, "Ultralytics native prediction; square letterbox input"

    if spec["backend"] == "mmdet":
        from mmdet.apis import init_detector, inference_detector
        from mmdet.utils import get_test_pipeline_cfg
        from mmengine.dataset import Compose

        model = init_detector(spec["config"], spec["weights"], device=device)
        mapping = class_mapping(model.dataset_meta["classes"], names)
        # Standalone images have no dataset annotation metadata. Keep native resize/normalization.
        pipeline_cfg = [t for t in copy.deepcopy(get_test_pipeline_cfg(model.cfg))
                        if str(t["type"]).split(".")[-1] != "LoadAnnotations"]
        pipeline = Compose(pipeline_cfg)

        def predict(path):
            result = inference_detector(model, str(path), test_pipeline=pipeline).pred_instances.cpu()
            return predictions(result.bboxes.tolist(), result.scores.tolist(), result.labels.tolist(),
                               mapping, args.conf, args.max_det)
        return predict, repr(pipeline_cfg)

    import numpy as np
    import torch
    sys.path.insert(0, spec["repo"])
    from src.core import YAMLConfig

    cfg = YAMLConfig(spec["config"], resume=spec["weights"])
    for value in cfg.yaml_cfg.values():
        if isinstance(value, dict) and "pretrained" in value:
            value["pretrained"] = False
    if cfg.yaml_cfg.get("remap_mscoco_category", False):
        raise ValueError("This manifest requires custom contiguous RT-DETR class ids, not MS-COCO remapping")
    mapping = class_mapping(spec["class_names"], names)
    checkpoint = torch.load(spec["weights"], map_location="cpu", weights_only=False)
    state = checkpoint["ema"]["module"] if checkpoint.get("ema") is not None else checkpoint["model"]
    model = cfg.model
    model.load_state_dict(state, strict=True)
    model = model.deploy().float().to(device).eval()
    postprocessor = cfg.postprocessor.deploy().to(device).eval()

    def predict(path):
        with Image.open(path) as original:
            original = original.convert("RGB")
            size = torch.tensor([original.size], device=device)  # RT-DETR expects [width, height].
            array = np.array(original.resize((args.imgsz, args.imgsz), Image.Resampling.BILINEAR), copy=True)
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float().to(device)/255
        with torch.inference_mode():
            labels, boxes, scores = postprocessor(model(tensor), size)
        return predictions(boxes[0].cpu().tolist(), scores[0].cpu().tolist(), labels[0].cpu().tolist(),
                           mapping, args.conf, args.max_det)
    return predict, "RT-DETRv2 native deploy postprocessor; EMA when present; RGB square resize, values 0..1"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            value.update(chunk)
    return value.hexdigest()


def export(spec, names, args):
    name = spec["name"]
    predict, preprocessing = (None, "Original YOLO annotations") if name == "GT" else build_predictor(spec, names, args)
    root = args.output / name
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"model": spec, "classes": names, "arguments": {k: str(v) for k, v in vars(args).items()},
                "preprocessing": preprocessing, "weights_sha256": digest(spec["weights"]) if predict else None,
                "config_sha256": digest(spec["config"]) if spec.get("config") else None, "images": []}
    for split in args.splits:
        image_root = args.dataset / "images" / split
        for path in files(image_root):
            relative = path.relative_to(image_root)
            with Image.open(path) as original:
                image = original.convert("RGB")
            records = predict(path) if predict else gt_boxes(
                args.dataset / "labels" / split / relative.with_suffix(".txt"), image.size, names)
            output = root / split / relative.parent / (relative.name + ".png")
            output.parent.mkdir(parents=True, exist_ok=True)
            draw(image, records, names, args).save(output)
            source_hash = digest(path)
            output.with_suffix(".json").write_text(json.dumps({"image": str(path.resolve()), "size": image.size,
                                                              "image_sha256": source_hash,
                                                              "detections": records}, indent=2), encoding="utf-8")
            manifest["images"].append({"split": split, "file": relative.as_posix(), "sha256": source_hash,
                                       "detections": len(records)})
            print(f"{name}/{split}: {relative} ({len(records)} boxes)", flush=True)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def compose(names, args):
    """One labeled comparison strip per image, with a strict check that every model used that image."""
    title_font = font(22)
    for split in args.splits:
        image_root = args.dataset / "images" / split
        for source in files(image_root):
            relative = source.relative_to(image_root)
            panels = []
            for name in names:
                path = args.output / name / split / relative.parent / (relative.name + ".png")
                if not path.is_file():
                    raise FileNotFoundError(f"Export this model first: {path}")
                metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
                if metadata["image"] != str(source.resolve()) or metadata["image_sha256"] != digest(source):
                    raise ValueError(f"Source image mismatch: {path}")
                with Image.open(path) as image:
                    h = round(image.height * args.panel_width / image.width)
                    panels.append(image.convert("RGB").resize((args.panel_width, h), Image.Resampling.LANCZOS))
            if len({p.size for p in panels}) != 1:
                raise ValueError(f"Panel sizes differ: {source}")
            width, height = panels[0].size
            canvas = Image.new("RGB", (len(panels)*width, height+40), "white")
            painter = ImageDraw.Draw(canvas)
            for index, (name, panel) in enumerate(zip(names, panels)):
                canvas.paste(panel, (index*width, 40))
                painter.text((index*width+8, 8), name, fill="black", font=title_font)
            output = args.output / "comparison" / split / relative.parent / (relative.name+".png")
            output.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(output)
            print(f"Saved {output}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("gt", "ultralytics", "mmdet", "rtdetrv2", "compose"))
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("detection_comparison_models.json"))
    parser.add_argument("--dataset", type=Path, default=Path("/home/tkz/datasets/pingwen_yolo"))
    parser.add_argument("--output", type=Path, default=Path("runs/detection_comparison"))
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=("train", "val", "test"))
    parser.add_argument("--models", nargs="+", help="Optional model-name subset; compose accepts GT as well.")
    parser.add_argument("--device", default="0", help="CUDA index or cpu")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO and RT-DETR input size; MMDet uses its test config")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7, help="Ultralytics NMS only; other models keep native postprocessing")
    parser.add_argument("--max-det", type=int, default=100, help="Display at most this many detections above --conf")
    parser.add_argument("--line-width", type=int, default=3)
    parser.add_argument("--font-size", type=int, default=22)
    parser.add_argument("--show-conf", action="store_true")
    parser.add_argument("--hide-labels", action="store_true")
    parser.add_argument("--panel-width", type=int, default=480)
    args = parser.parse_args(argv)
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1 or min(
            args.imgsz, args.max_det, args.line_width, args.font_size, args.panel_width) <= 0:
        parser.error("Invalid threshold or non-positive size/count")
    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    names, specs = data["classes"], data["models"]
    known = ["GT"] + [s["name"] for s in specs]
    if len(set(known)) != len(known) or any(Path(n).name != n or n in {".", ".."} for n in known):
        parser.error("Model names must be unique single directory names")
    if args.models and set(args.models) - set(known):
        parser.error(f"Unknown names: {set(args.models)-set(known)}")
    if args.backend == "compose":
        compose(args.models or known, args)
    elif args.backend == "gt":
        export({"name": "GT"}, names, args)
    else:
        selected = [s for s in specs if s["backend"] == args.backend and (not args.models or s["name"] in args.models)]
        if not selected:
            parser.error("No models match the backend/name filters")
        for spec in selected:
            for key in ("weights", "config"):
                if key in spec and not Path(spec[key]).is_file():
                    raise FileNotFoundError(f"Missing {key}: {spec[key]}")
            export(spec, names, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
