"""Convert X-AnyLabeling JSON annotations into a deterministically split COCO detection dataset."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Directory containing paired image and LabelMe JSON files.")
    parser.add_argument("--output", type=Path, required=True, help="Destination COCO dataset directory.")
    parser.add_argument("--classes", type=Path, required=True, help="One class name per line, in desired category order.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the 7:1:2 image split.")
    return parser.parse_args()


def load_records(source: Path, classes: list[str]) -> list[dict]:
    class_to_id = {name: index + 1 for index, name in enumerate(classes)}
    records = []
    for image_path in sorted(path for path in source.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
        json_path = image_path.with_suffix(".json")
        if not json_path.is_file():
            raise FileNotFoundError(f"Missing annotation for image: {image_path}")
        annotation = json.loads(json_path.read_text(encoding="utf-8"))
        shapes = []
        for shape in annotation.get("shapes", []):
            label = shape.get("label")
            if label not in class_to_id:
                raise ValueError(f"Unknown label {label!r} in {json_path}; expected one of {classes}.")
            points = shape.get("points", [])
            if len(points) < 2:
                raise ValueError(f"Invalid shape with fewer than two points in {json_path}")
            xs, ys = zip(*points, strict=True)
            x1, y1 = max(0.0, min(xs)), max(0.0, min(ys))
            x2, y2 = min(float(annotation["imageWidth"]), max(xs)), min(float(annotation["imageHeight"]), max(ys))
            width, height = x2 - x1, y2 - y1
            if width <= 0 or height <= 0:
                continue
            segmentation = [[coordinate for point in points for coordinate in point]] if len(points) >= 3 else []
            shapes.append(
                {
                    "category_id": class_to_id[label],
                    "bbox": [round(x1, 4), round(y1, 4), round(width, 4), round(height, 4)],
                    "area": round(width * height, 4),
                    "segmentation": segmentation,
                }
            )
        records.append(
            {
                "image_path": image_path,
                "width": int(annotation["imageWidth"]),
                "height": int(annotation["imageHeight"]),
                "annotations": shapes,
            }
        )
    if not records:
        raise ValueError(f"No images found in {source}")
    return records


def split_records(records: list[dict], seed: int) -> dict[str, list[dict]]:
    shuffled = records.copy()
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    train_end = round(total * 0.7)
    val_end = train_end + round(total * 0.1)
    return {"train": shuffled[:train_end], "val": shuffled[train_end:val_end], "test": shuffled[val_end:]}


def write_coco_split(records: list[dict], split: str, output: Path, categories: list[dict]) -> dict:
    image_dir = output / f"{split}2017"
    image_dir.mkdir(parents=True, exist_ok=False)
    images, annotations = [], []
    annotation_id = 1
    for image_id, record in enumerate(records, start=1):
        destination = image_dir / record["image_path"].name
        shutil.copy2(record["image_path"], destination)
        images.append(
            {
                "id": image_id,
                "file_name": f"{split}2017/{destination.name}",
                "width": record["width"],
                "height": record["height"],
            }
        )
        for shape in record["annotations"]:
            annotations.append({"id": annotation_id, "image_id": image_id, "iscrowd": 0, **shape})
            annotation_id += 1
    coco = {"info": {"description": "Fabric defects converted from X-AnyLabeling"}, "licenses": [], "images": images,
            "annotations": annotations, "categories": categories}
    annotation_path = output / "annotations" / f"instances_{split}2017.json"
    annotation_path.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"images": len(images), "annotations": len(annotations), "classes": dict(Counter(a["category_id"] for a in annotations))}


def main() -> None:
    args = parse_args()
    classes = [line.strip() for line in args.classes.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not classes:
        raise ValueError("Classes file is empty.")
    generated = [args.output / "annotations", *(args.output / f"{split}2017" for split in ("train", "val", "test"))]
    existing = [path for path in generated if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing generated directories: {existing}")
    records = load_records(args.source, classes)
    categories = [{"id": index + 1, "name": name, "supercategory": "defect"} for index, name in enumerate(classes)]
    (args.output / "annotations").mkdir(parents=True)
    summary = {split: write_coco_split(items, split, args.output, categories) for split, items in split_records(records, args.seed).items()}
    summary.update({"seed": args.seed, "categories": categories, "source_images": len(records)})
    (args.output / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
