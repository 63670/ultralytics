"""Compose labeled sources, YOLOv8n, and MCGI-YOLO heatmaps into a paper figure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLUMNS = (
    ("src", "", "Label"),
    ("yolov8n", "_cam", "YOLOv8n"),
    ("ours", "_cam", "MCGI-YOLO"),
)


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in (
        "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/Times New Roman.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def panel_path(root: Path, directory: str, suffix: str, stem: str) -> Path:
    candidates = [p for p in (root / directory).glob(f"{stem}{suffix}.*")
                  if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one image for {directory}/{stem}{suffix}, found {candidates}")
    return candidates[0]


def collect_rows(root: Path, selected: list[str] | None = None) -> list[str]:
    names = ([Path(name).stem for name in selected] if selected else
             [p.stem for p in sorted((root / "src").iterdir())
              if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if not names or len(set(names)) != len(names):
        raise ValueError("Select at least one unique source image")
    for stem in names:
        for directory, suffix, _ in COLUMNS:
            panel_path(root, directory, suffix, stem)
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Directory containing src/, yolov8n/, and ours/.")
    parser.add_argument("--output", type=Path, required=True, help="PNG output path.")
    parser.add_argument("--images", nargs="+", help="Source filenames in desired row order.")
    parser.add_argument("--gap", type=int, default=12, help="White spacing between panels in pixels.")
    parser.add_argument("--label-font-size", type=int, default=64,
                        help="Column-label font size in pixels (default: 64).")
    args = parser.parse_args()
    if args.label_font_size <= 0 or args.gap < 0:
        parser.error("Label font size must be positive and gap must be nonnegative.")

    root = args.input.expanduser().resolve()
    stems = collect_rows(root, args.images)
    first = Image.open(panel_path(root, "src", "", stems[0])).convert("RGB")
    panel_w, panel_h = first.size
    for directory, suffix, _ in COLUMNS:
        for stem in stems:
            path = panel_path(root, directory, suffix, stem)
            with Image.open(path) as image:
                if image.size != (panel_w, panel_h):
                    raise ValueError(f"Unexpected image size: {path} is {image.size}, expected {(panel_w, panel_h)}")

    label_h = args.label_font_size + 40
    width = panel_w * len(COLUMNS) + args.gap * (len(COLUMNS) - 1)
    height = panel_h * len(stems) + args.gap * (len(stems) - 1) + label_h
    canvas = Image.new("RGB", (width, height), "white")
    metadata = {"columns": [label for _, _, label in COLUMNS], "rows": stems,
                "panel_size": [panel_w, panel_h], "label_font_size": args.label_font_size,
                "dpi": 300, "sources": []}

    for row, stem in enumerate(stems):
        y = row * (panel_h + args.gap)
        for col, (directory, suffix, _) in enumerate(COLUMNS):
            x = col * (panel_w + args.gap)
            path = panel_path(root, directory, suffix, stem)
            with Image.open(path) as image:
                canvas.paste(image.convert("RGB"), (x, y))
            metadata["sources"].append({"column": directory, "file": path.name,
                                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    draw = ImageDraw.Draw(canvas)
    font = load_font(args.label_font_size)
    for col, (_, _, label) in enumerate(COLUMNS):
        x_center = col * (panel_w + args.gap) + panel_w / 2
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if text_w > panel_w:
            raise ValueError(f"Column label is wider than its panel: {label}")
        label_y = height - label_h + (label_h - text_h) / 2 - bbox[1]
        draw.text((x_center - text_w / 2 - bbox[0], label_y), label, fill="black", font=font)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, dpi=(300, 300))
    args.output.with_suffix(".layout.json").write_bytes((json.dumps(metadata, indent=2) + "\n").encode())
    print(f"Saved {args.output} ({width}x{height}) with {len(stems)} rows.")


if __name__ == "__main__":
    main()
