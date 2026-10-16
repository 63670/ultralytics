#!/usr/bin/env python3
"""Export trained custom YOLOv8 detectors with Rockchip's raw nine-output ONNX layout."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Prefer this checkout, including its custom modules, when called as python tools/export_rkopt_onnx.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="Trained baseline or custom .pt checkpoint.")
    parser.add_argument("--output", type=Path, help="ONNX path; default filename includes -rkopt and the CAFR mode.")
    parser.add_argument(
        "--imgsz", type=int, nargs="+", default=[640], help="One size or height width (multiples of 32)."
    )
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument(
        "--cafr-mode", choices=["original", "streaming"], default="original",
        help="streaming reuses CAFR weights with 4-D shifted-neighbor accumulation; benchmark after RKNN conversion.",
    )
    args = parser.parse_args()
    if not args.weights.is_file() or args.weights.suffix.lower() != ".pt":
        parser.error("--weights must be an existing .pt file.")
    if len(args.imgsz) not in (1, 2) or any(size < 32 or size % 32 for size in args.imgsz):
        parser.error("--imgsz requires one or two positive multiples of 32.")
    if not 12 <= args.opset <= 19:
        parser.error("--opset must be from 12 to 19.")
    suffix = "-rkopt-cafr4d" if args.cafr_mode == "streaming" else "-rkopt"
    output = args.output or args.weights.with_name(f"{args.weights.stem}{suffix}.onnx")
    if output.suffix.lower() != ".onnx":
        parser.error("--output must end with .onnx.")
    from ultralytics import YOLO
    from ultralytics.utils.export.rkopt import export_rkopt_onnx

    model = YOLO(str(args.weights)).model
    size = args.imgsz * 2 if len(args.imgsz) == 1 else args.imgsz
    info = export_rkopt_onnx(model, output, tuple(size), args.opset, cafr_mode=args.cafr_mode)
    digest = hashlib.sha256()
    with args.weights.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    info.update(weights=str(args.weights.resolve()), weights_sha256=digest.hexdigest())
    report = output.with_suffix(".rkopt.json")
    report.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved ONNX: {output}")
    print(f"CAFR mode: {info['cafr_mode']}; rewritten modules: {info['rewritten_cafr_modules']}")
    for item in info["outputs"]:
        print(f"  {item['name']}: {item['shape']}")
    print(f"Saved output contract: {report}")
    print("Convert with the same RKNN settings as the baseline, then use tools/benchmark_rknn.py on the .rknn file.")


if __name__ == "__main__":
    main()
