"""Split paired YOLO/COCO detection labels into matching 7:1:2 datasets.

Usage:
    python scripts/split_hybrid_dataset.py --source C:/path/to/pingwen_hybrid --output datasets/pingwen_hybrid_split

Input: images/, yolo_labels/, coco_labels/coco_detection.json, classes.txt.
Empty YOLO labels absent from COCO are retained as background images. Requires Pillow.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")
RATIOS = (7, 1, 2)


def read_json(path: Path) -> dict:
    """Read JSON including exports with a UTF-8 byte order mark."""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_yolo(path: Path, class_count: int) -> list[list[float]]:
    """Read detection rows and reject missing, malformed or non-finite labels."""
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = [float(value) for value in line.split()]
        if len(row) != 5 or not all(math.isfinite(value) for value in row):
            raise ValueError(f"{path}:{line_number}: expected class_id cx cy width height")
        class_id, cx, cy, width, height = row
        if not class_id.is_integer() or not 0 <= class_id < class_count:
            raise ValueError(f"{path}:{line_number}: invalid class ID {class_id}")
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"{path}:{line_number}: invalid normalized bounding box")
        rows.append(row)
    return rows


def yolo_row(annotation: dict, width: int, height: int, class_ids: dict) -> list[float]:
    """Convert a COCO box to a YOLO detection row using the classes.txt order."""
    x, y, box_width, box_height = annotation["bbox"]
    return [
        class_ids[annotation["category_id"]],
        (x + box_width / 2) / width,
        (y + box_height / 2) / height,
        box_width / width,
        box_height / height,
    ]


def load_source(source: Path) -> tuple[list[str], dict, list[dict]]:
    """Validate both exports before selecting COCO boxes as the common annotation source."""
    classes = [line.strip() for line in (source / "classes.txt").read_text(encoding="utf-8-sig").splitlines()
               if line.strip()]
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("classes.txt must contain unique, nonempty class names")
    coco = read_json(source / "coco_labels" / "coco_detection.json")
    categories = coco["categories"]
    category_names = {category["name"]: category["id"] for category in categories}
    if (len(category_names) != len(categories) or set(category_names) != set(classes)
            or len({category["id"] for category in categories}) != len(categories)):
        raise ValueError("COCO categories must match classes.txt and have unique IDs and names")
    class_ids = {category_names[name]: index for index, name in enumerate(classes)}
    # Order COCO categories identically to YOLO classes, including non-contiguous source IDs.
    coco["categories"] = sorted(categories, key=lambda category: class_ids[category["id"]])
    image_dir = source / "images"
    paths = sorted(path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise ValueError(f"No images found in {image_dir}")
    paths_by_name = {path.relative_to(image_dir).as_posix(): path for path in paths}
    label_names = [Path(name).with_suffix(".txt").as_posix().casefold() for name in paths_by_name]
    if len(set(label_names)) != len(label_names):
        raise ValueError("Multiple images map to the same YOLO label filename")

    images_by_name, images_by_id = {}, {}
    for entry in coco["images"]:
        name = entry["file_name"].replace("\\", "/")
        if name not in paths_by_name:
            raise ValueError(f"COCO image is absent from images/: {name}")
        if name in images_by_name or entry["id"] in images_by_id:
            raise ValueError(f"Duplicate COCO image name or ID: {name}")
        images_by_name[name] = entry
        images_by_id[entry["id"]] = entry
    annotations_by_id = defaultdict(list)
    for annotation in coco["annotations"]:
        if annotation["image_id"] not in images_by_id or annotation["category_id"] not in class_ids:
            raise ValueError(f"COCO annotation references an unknown image/category: {annotation}")
        if annotation.get("iscrowd", 0) or annotation.get("ignore", 0):
            raise ValueError("Crowd/ignored COCO annotations cannot be represented as matching YOLO detection labels")
        annotations_by_id[annotation["image_id"]].append(annotation)

    records = []
    for name, path in paths_by_name.items():
        label_path = source / "yolo_labels" / Path(name).with_suffix(".txt")
        rows = read_yolo(label_path, len(classes))
        with Image.open(path) as image:
            width, height = image.size
        entry = images_by_name.get(name)
        if entry is None:
            if rows:
                raise ValueError(f"Nonempty YOLO label has no matching COCO image: {name}")
            entry = {"file_name": name, "width": width, "height": height}
            annotations = []
        else:
            if (entry["width"], entry["height"]) != (width, height):
                raise ValueError(f"COCO dimensions disagree with the image: {name}")
            annotations = annotations_by_id[entry["id"]]
        for annotation in annotations:
            box = annotation["bbox"]
            if len(box) != 4 or not all(math.isfinite(value) for value in box):
                raise ValueError(f"Invalid COCO bounding box: {name}")
            x, y, box_width, box_height = box
            if (min(x, y) < -1e-6 or min(box_width, box_height) <= 0
                    or x + box_width > width + 1e-6 or y + box_height > height + 1e-6):
                raise ValueError(f"COCO bounding box extends outside the image: {name}")
        expected = [yolo_row(annotation, width, height, class_ids) for annotation in annotations]
        # Source YOLO exports use six decimal places; compare independently of row order.
        if len(rows) != len(expected) or any(
            actual[0] != wanted[0] or any(abs(a - b) > 1e-6 for a, b in zip(actual[1:], wanted[1:]))
            for actual, wanted in zip(sorted(rows), sorted(expected))
        ):
            raise ValueError(f"YOLO and COCO labels disagree: {name}")
        records.append({"path": path, "name": name, "image": entry, "annotations": annotations, "rows": expected})
    return classes, coco, records


def split_records(records: list[dict], seed: int) -> dict[str, list[dict]]:
    """Shuffle once and use largest remainders to allocate every image in a 7:1:2 ratio."""
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    counts = [len(records) * ratio // 10 for ratio in RATIOS]
    remainder_order = sorted(range(3), key=lambda index: -(len(records) * RATIOS[index] % 10))
    for index in remainder_order[:len(records) - sum(counts)]:
        counts[index] += 1
    result, offset = {}, 0
    for split, count in zip(SPLITS, counts):
        result[split] = shuffled[offset:offset + count]
        offset += count
    return result


def write_json(path: Path, value: dict) -> None:
    """Write readable, standards-compliant JSON."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_datasets(output: Path, classes: list[str], coco: dict, splits: dict, seed: int, source: Path) -> dict:
    """Export both formats from the same records and save the exact shared split manifest."""
    summary = {"source": str(source), "seed": seed, "ratio": list(RATIOS), "classes": classes, "splits": {}}
    image_id, annotation_id = 1, 1
    (output / "coco" / "annotations").mkdir(parents=True)
    for split, records in splits.items():
        yolo_images = output / "yolo" / "images" / split
        yolo_labels = output / "yolo" / "labels" / split
        coco_images = output / "coco" / f"{split}2017"
        for directory in (yolo_images, yolo_labels, coco_images):
            directory.mkdir(parents=True)
        images, annotations = [], []
        for record in records:
            name = record["name"]
            for directory in (yolo_images, coco_images):
                destination = directory / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(record["path"], destination)
            label = yolo_labels / Path(name).with_suffix(".txt")
            label.parent.mkdir(parents=True, exist_ok=True)
            text = "".join(f"{int(row[0])} " + " ".join(f"{value:.10f}" for value in row[1:]) + "\n"
                           for row in record["rows"])
            label.write_text(text, encoding="utf-8")
            images.append({**record["image"], "id": image_id, "file_name": name})
            for annotation in record["annotations"]:
                x, y, width, height = annotation["bbox"]
                annotations.append({**annotation, "id": annotation_id, "image_id": image_id,
                                    "area": width * height, "iscrowd": 0})
                annotation_id += 1
            image_id += 1
        subset = {key: value for key, value in coco.items() if key not in {"images", "annotations"}}
        subset.update({"images": images, "annotations": annotations})
        write_json(output / "coco" / "annotations" / f"instances_{split}2017.json", subset)
        counts = Counter(classes[int(row[0])] for record in records for row in record["rows"])
        summary["splits"][split] = {
            "images": len(records), "annotations": len(annotations),
            "background_images": sum(not record["rows"] for record in records),
            "class_counts": {name: counts[name] for name in classes},
            "files": [record["name"] for record in records],
        }
    # A relative root keeps data.yaml usable after moving the dataset to another machine.
    yaml_text = "train: images/train\nval: images/val\ntest: images/test\nnames:\n"
    yaml_text += "".join(f"  {index}: {json.dumps(name, ensure_ascii=False)}\n" for index, name in enumerate(classes))
    (output / "yolo" / "data.yaml").write_text(yaml_text, encoding="utf-8")
    (output / "yolo" / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    write_json(output / "split_manifest.json", summary)
    return summary


def prepare_dataset(source: Path, output: Path, seed: int = 42) -> dict:
    """Validate before exporting, refuse overwrites, and publish only a complete dataset."""
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new --output directory: {output}")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and output directories must not contain each other")
    classes, coco, records = load_source(source)
    splits = split_records(records, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hybrid-split-", dir=output.parent) as temporary:
        staged = Path(temporary) / "dataset"
        summary = write_datasets(staged, classes, coco, splits, seed, source)
        staged.rename(output)
    return summary


def main() -> None:
    """Run the dataset export and print per-split counts."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=Path.home() / "Desktop" / "pingwen_hybrid")
    parser.add_argument("--output", type=Path, default=Path("datasets/pingwen_hybrid_split"))
    parser.add_argument("--seed", type=int, default=42, help="Reproducible random split seed (default: 42).")
    args = parser.parse_args()
    try:
        summary = prepare_dataset(args.source, args.output, args.seed)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Error: {error}\n")
    for split, counts in summary["splits"].items():
        print(f"{split}: {counts['images']} images, {counts['annotations']} boxes, "
              f"{counts['background_images']} backgrounds")
    print(f"Saved matching YOLO and COCO datasets to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
