"""Split an exported YOLO dataset into train, validation, and test sets."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args():
    """Parse script arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Directory containing exported images.")
    parser.add_argument("--labels", type=Path, required=True, help="Directory containing exported YOLO labels.")
    parser.add_argument("--classes", type=Path, required=True, help="Class names file, one name per line.")
    parser.add_argument("--output", type=Path, required=True, help="Empty destination directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random split seed.")
    return parser.parse_args()


def validate_label(label_file: Path, class_count: int):
    """Validate one YOLO detection label file."""
    for line_number, line in enumerate(label_file.read_text(encoding="utf-8").splitlines(), 1):
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"{label_file}:{line_number} must have 5 values.")
        class_id = int(values[0])
        if not 0 <= class_id < class_count:
            raise ValueError(f"{label_file}:{line_number} has invalid class ID {class_id}.")
        coordinates = [float(value) for value in values[1:]]
        if not all(0 <= value <= 1 for value in coordinates):
            raise ValueError(f"{label_file}:{line_number} has coordinates outside [0, 1].")


def collect_pairs(images_dir: Path, labels_dir: Path, class_count: int) -> list[tuple[Path, Path]]:
    """Return all image-label pairs, rejecting incomplete exports."""
    pairs = []
    for image_file in sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
        label_file = labels_dir / f"{image_file.stem}.txt"
        if not label_file.is_file():
            raise FileNotFoundError(f"Missing label for {image_file.name}: {label_file}")
        validate_label(label_file, class_count)
        pairs.append((image_file, label_file))
    if not pairs:
        raise ValueError(f"No images found in {images_dir}")
    return pairs


def write_dataset(output_dir: Path, pairs: list[tuple[Path, Path]], class_names: list[str], seed: int):
    """Copy a deterministic 7:1:2 split and write its data configuration."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Destination must be empty: {output_dir}")

    random.Random(seed).shuffle(pairs)
    train_end = len(pairs) * 7 // 10
    val_end = train_end + len(pairs) // 10
    splits = {"train": pairs[:train_end], "val": pairs[train_end:val_end], "test": pairs[val_end:]}
    for split, items in splits.items():
        image_dir, label_dir = output_dir / "images" / split, output_dir / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image_file, label_file in items:
            shutil.copy2(image_file, image_dir / image_file.name)
            shutil.copy2(label_file, label_dir / label_file.name)

    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(class_names))
    (output_dir / "data.yaml").write_text(
        f"path: {output_dir.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n{names}\n",
        encoding="utf-8",
    )
    print(
        f"Created {output_dir} with "
        + ", ".join(f"{split}={len(items)}" for split, items in splits.items())
        + f" (seed={seed})."
    )


def main():
    """Run the dataset splitter."""
    args = parse_args()
    class_names = [name.strip() for name in args.classes.read_text(encoding="utf-8").splitlines() if name.strip()]
    if not class_names or len(class_names) != len(set(class_names)):
        raise ValueError("Class names must be present and unique.")
    write_dataset(args.output, collect_pairs(args.images, args.labels, len(class_names)), class_names, args.seed)


if __name__ == "__main__":
    main()
