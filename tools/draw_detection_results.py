#!/usr/bin/env python3
"""Draw PNGs for a local YOLO checkpoint, the comparison baselines, or ground truth."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import yaml
from PIL import Image

if __package__ in (None, ""):
    # Prefer this checkout when loading a checkpoint with custom Ultralytics modules.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.export_detection_comparison import build_predictor, build_ultralytics_predictor, draw, files, gt_boxes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")
BACKENDS = ("ultralytics", "mmdet", "rtdetrv2")


def image_jobs(dataset: Path, output: Path) -> list[tuple[Path, Path]]:
    """Keep the split, nested path, and original suffix to avoid filename collisions."""
    root = dataset / "images"
    jobs = []
    for split in SPLITS:
        for source in files(root / split):
            relative = source.relative_to(root)
            jobs.append((source, output / relative.parent / f"{relative.name}.png"))
    return jobs


def draw_images(jobs, read_records, names, args):
    """Commit each PNG atomically so an interrupted baseline run can be resumed."""
    for source, destination in jobs:
        with Image.open(source) as image:
            records = read_records(source, image.size)
            result = draw(image.convert("RGB"), records, names, args)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_path = None
        try:
            with NamedTemporaryFile(dir=destination.parent, prefix=".draw-", delete=False) as tmp:
                temp_path = Path(tmp.name)
                result.save(tmp, format="PNG")
            temp_path.replace(destination)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)


def own_output(weights: Path) -> Path:
    """Place results beside the training run or use the checkpoint stem for standalone files."""
    if weights.parent.name == "weights" and weights.stem == "best":
        return weights.parent.parent / "detection-results"
    return weights.with_name(f"{weights.stem}-detections")


def draw_ground_truth(args):
    output = args.output / "GT"
    jobs = image_jobs(args.dataset, output)
    pending = jobs if args.overwrite else [(src, dest) for src, dest in jobs if not dest.is_file()]
    if pending:
        data_file = args.dataset / "data.yaml"
        data = yaml.safe_load(data_file.read_text(encoding="utf-8-sig"))
        names = data.get("names") if isinstance(data, dict) else None
        if isinstance(names, dict):
            names = {int(key): value for key, value in names.items()}
            if set(names) != set(range(len(names))):
                raise ValueError(f"Expected consecutive class IDs in {data_file}")
            names = [names[index] for index in range(len(names))]
        if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
            raise ValueError(f"Expected class names in {data_file}")

        def read_labels(source, size):
            relative = source.relative_to(args.dataset / "images")
            label = args.dataset / "labels" / relative.with_suffix(".txt")
            return gt_boxes(label, size, names)

        print(f"GT: drawing {len(pending)} images", flush=True)
        draw_images(pending, read_labels, names, args)
    print(f"GT: {len(jobs)} images in {output}", flush=True)


def draw_baseline(spec, names, args):
    output = args.output / spec["name"]
    jobs = image_jobs(args.dataset, output)
    pending = jobs if args.overwrite else [(src, dest) for src, dest in jobs if not dest.is_file()]
    if pending:
        for key in ("weights", "config"):
            if key in spec and not Path(spec[key]).is_file():
                raise FileNotFoundError(f"{key}: {spec[key]}")
        print(f"{spec['name']}: drawing {len(pending)} images", flush=True)
        predictor, _ = build_predictor(spec, names, args)
        draw_images(pending, lambda source, _size: predictor(source), names, args)
    print(f"{spec['name']}: {len(jobs)} images in {output}", flush=True)


def draw_in_environment(backend, environment, args):
    command = [
        os.environ.get("CONDA_EXE") or "conda", "run", "--no-capture-output", "-n", environment, "python",
        str(Path(__file__).resolve()), "--baselines", "--backend", backend,
        "--dataset", str(args.dataset), "--output", str(args.output), "--manifest", str(args.manifest),
        "--device", args.device, "--imgsz", str(args.imgsz), "--conf", str(args.conf),
        "--iou", str(args.iou), "--max-det", str(args.max_det),
    ]
    if args.overwrite:
        command.append("--overwrite")
    return subprocess.run(command, check=False).returncode


def run_baselines(args, environments, parser):
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    specs = [spec for spec in manifest["models"] if spec.get("baseline", True)]
    if args.backend:
        specs = [spec for spec in specs if spec["backend"] == args.backend]
    names = [spec["name"] for spec in specs]
    if not specs or len(set(names)) != len(names) or any(Path(n).name != n or n.lower() in {".", "..", "gt"} for n in names):
        parser.error("Expected at least one baseline with a unique model directory name other than GT.")
    if any(spec["backend"] not in BACKENDS for spec in specs):
        parser.error("A baseline uses an unsupported backend.")

    failed = []
    for backend in BACKENDS:
        group = [spec for spec in specs if spec["backend"] == backend]
        if not group:
            continue
        if backend in environments:
            try:
                if draw_in_environment(backend, environments[backend], args):
                    failed.append(backend)
            except OSError as exc:
                print(f"{backend}: {exc}", file=sys.stderr)
                failed.append(backend)
            continue
        for spec in group:
            try:
                draw_baseline(spec, manifest["classes"], args)
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                print(f"{spec['name']}: {exc}", file=sys.stderr)
                failed.append(spec["name"])
    return failed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--weights", type=Path, help="Draw all train/val/test images using this local YOLO .pt file.")
    mode.add_argument("--baselines", action="store_true", help="Draw the public baseline models, reusing completed PNGs.")
    parser.add_argument("--gt", action="store_true", help="Draw ground-truth labels. Can be combined with --baselines.")
    parser.add_argument("--dataset", type=Path, default=Path("/home/tkz/datasets/pingwen_yolo"))
    parser.add_argument("--output", type=Path, help="Destination for the rendered PNGs.")
    parser.add_argument("--device", default="0", help="CUDA index or cpu.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=16)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("detection_comparison_models.json"))
    parser.add_argument("--backend", choices=BACKENDS, help="Limit baseline drawing to this framework.")
    parser.add_argument("--backend-env", action="append", default=[], metavar="BACKEND=ENV",
                        help="Run a baseline framework in this Conda environment. Repeat as needed.")
    parser.add_argument("--overwrite", action="store_true", help="Redraw existing baseline or ground-truth PNGs.")
    parser.set_defaults(line_width=3, font_size=22, hide_labels=False, show_conf=True)
    args = parser.parse_args(argv)
    if args.weights is None and not (args.baselines or args.gt):
        parser.error("Choose --weights, --baselines, or --gt.")
    if not args.baselines and (args.backend or args.backend_env):
        parser.error("--backend and --backend-env require --baselines.")
    if min(args.imgsz, args.max_det) <= 0 or not (0 <= args.conf <= 1 and 0 <= args.iou <= 1):
        parser.error("Image size and max detections must be positive, and thresholds must be in [0, 1].")

    if args.weights is not None:
        if args.gt or args.overwrite:
            parser.error("Use --gt separately from --weights. Checkpoint drawing already overwrites its PNGs.")
        args.weights = args.weights.resolve()
        if args.weights.suffix.lower() != ".pt" or not args.weights.is_file():
            parser.error(f"Expected a local .pt checkpoint: {args.weights}")
        output = (args.output or own_output(args.weights)).resolve()
        jobs = image_jobs(args.dataset, output)
        predictor, names = build_ultralytics_predictor(args.weights, args)
        print(f"Drawing {len(jobs)} images to {output}", flush=True)
        draw_images(jobs, lambda source, _size: predictor(source), names, args)
        print(f"Saved {len(jobs)} images to {output}", flush=True)
        return 0

    environments = {}
    for entry in args.backend_env:
        backend, _, environment = entry.partition("=")
        if backend not in BACKENDS or not environment or backend in environments:
            parser.error(f"Expected one BACKEND=ENV entry per framework: {entry}")
        environments[backend] = environment
    args.output = (args.output or ROOT / "runs" / "detection-baselines").resolve()
    failed = []
    if args.gt:
        try:
            draw_ground_truth(args)
        except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
            print(f"GT: {exc}", file=sys.stderr)
            failed.append("GT")
    if args.baselines:
        failed.extend(run_baselines(args, environments, parser))
    if failed:
        print(f"Drawing incomplete: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
