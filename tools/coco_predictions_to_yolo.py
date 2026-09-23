#!/usr/bin/env python3
"""Convert COCO detection JSON into per-image YOLO prediction text files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--images-file", type=Path,
                        help="Optional image-name allowlist, one per line.")
    parser.add_argument("--category-offset", type=int, default=1,
                        help="COCO category ID corresponding to YOLO class 0.")
    parser.add_argument("--score-threshold", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotation = json.loads(args.annotations.read_text(encoding="utf-8"))
    images = {item["id"]: item for item in annotation["images"]}
    allowed: set[str] | None = None
    if args.images_file:
        allowed = {line.strip() for line in args.images_file.read_text(encoding="utf-8").splitlines()
                   if line.strip() and not line.lstrip().startswith("#")}
    grouped: dict[int, list[str]] = {}
    for detection in json.loads(args.predictions.read_text(encoding="utf-8")):
        image = images.get(detection["image_id"])
        if image is None or (allowed is not None and image["file_name"] not in allowed):
            continue
        score = float(detection.get("score", 1.0))
        if score < args.score_threshold:
            continue
        x, y, w, h = map(float, detection["bbox"])
        iw, ih = float(image["width"]), float(image["height"])
        cls = int(detection["category_id"]) - args.category_offset
        line = f"{cls} {(x + w / 2) / iw:.8f} {(y + h / 2) / ih:.8f} {w / iw:.8f} {h / ih:.8f} {score:.8f}"
        grouped.setdefault(detection["image_id"], []).append(line)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for image_id, image in images.items():
        if allowed is not None and image["file_name"] not in allowed:
            continue
        (args.output_dir / f"{Path(image['file_name']).stem}.txt").write_text(
            "\n".join(grouped.get(image_id, [])) + ("\n" if grouped.get(image_id) else ""),
            encoding="utf-8")
    print(f"Saved YOLO prediction labels to {args.output_dir}")


if __name__ == "__main__":
    main()
