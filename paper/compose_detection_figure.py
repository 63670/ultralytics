#!/usr/bin/env python3
"""Compose matched detection images into a publication figure without changing annotations.

Example:
    python paper/compose_detection_figure.py --input C:/Users/tankz/Desktop/compare \
        --output paper/figures/qualitative_comparison.png
    python paper/compose_detection_figure.py --input C:/Users/tankz/Desktop/compare \
        --columns GT yolov8n ours --output paper/figures/qualitative_baseline_comparison.png
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TITLES = {"GT": "GT", "yolov8n": "YOLOv8n", "RT-DETR-R18": "RT-DETR-R18",
          "Faster-R-CNN-R50": "Faster R-CNN-R50", "ours": "PAGI-YOLO (Ours)"}
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def collect_rows(root, columns, selected=None):
    inventories = {}
    for name in columns:
        directory = root / name
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing model directory: {directory}")
        inventories[name] = {p.name for p in directory.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS}
    rows = selected or sorted(inventories[columns[0]])
    if not rows or len(rows) != len(set(rows)):
        raise ValueError("Select at least one image, without duplicate filenames")
    for name, inventory in inventories.items():
        missing = set(rows) - inventory
        if missing:
            raise ValueError(f"Missing images in {name}: {sorted(missing)}")
        if selected is None and inventory != set(rows):
            raise ValueError(f"Different image sets in {name}; use --images to explicitly select common filenames")
    return rows


def load_font(size, custom=None):
    candidates = [str(custom)] if custom else ["C:/Windows/Fonts/times.ttf", "Times New Roman.ttf",
                                                "LiberationSerif-Regular.ttf", "DejaVuSerif.ttf"]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    raise FileNotFoundError("No suitable font found; specify a TrueType font using --font")


def compose(args):
    root = args.input.expanduser().resolve()
    rows = collect_rows(root, args.columns, args.images)
    with Image.open(root/args.columns[0]/rows[0]) as first:
        original_size = first.size
    for name in args.columns:
        for filename in rows:
            with Image.open(root/name/filename) as image:
                if image.size != original_size:
                    raise ValueError(f"Different image size at {name}/{filename}: {image.size} != {original_size}")
    width = args.panel_width or original_size[0]
    height = round(original_size[1] * width / original_size[0])
    label_height = args.font_size + 24
    canvas_size = (width*len(args.columns) + args.gap*(len(args.columns)-1),
                   height*len(rows) + args.gap*(len(rows)-1) + label_height)
    canvas = Image.new("RGB", canvas_size, "white")
    painter = ImageDraw.Draw(canvas)
    text_font = load_font(args.font_size, args.font)
    # Keep all annotation pixels intact unless the caller explicitly requests scaling.
    metadata = {"columns": args.columns, "rows": rows, "source_size": original_size,
                "panel_size": [width, height], "dpi": args.dpi, "sources": []}
    for row, filename in enumerate(rows):
        for col, name in enumerate(args.columns):
            path = root/name/filename
            if path.resolve() == args.output.resolve():
                raise ValueError("Output must not overwrite a source image")
            with Image.open(path) as source:
                panel = source.convert("RGB")
                if panel.size != (width, height):
                    panel = panel.resize((width, height), Image.Resampling.LANCZOS)
                canvas.paste(panel, (col*(width+args.gap), row*(height+args.gap)))
            metadata["sources"].append({"column": name, "file": filename,
                                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    for col, name in enumerate(args.columns):
        label = TITLES.get(name, name)
        if not args.no_panel_letters:
            label = f"({chr(97+col)}) {label}"
        box = painter.textbbox((0, 0), label, font=text_font)
        tw, th = box[2]-box[0], box[3]-box[1]
        if tw > width:
            raise ValueError(f"Label too wide for its panel: {label}; reduce --font-size")
        x = col*(width+args.gap) + (width-tw)/2 - box[0]
        y = canvas.height-label_height + (label_height-th)/2 - box[1]
        painter.text((x, y), label, font=text_font, fill="black")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, dpi=(args.dpi, args.dpi))
    args.output.with_suffix(".layout.json").write_bytes((json.dumps(metadata, indent=2) + "\n").encode("utf-8"))
    print(f"Saved {args.output}: {canvas.width} x {canvas.height}; {len(rows)} rows x {len(args.columns)} columns")
    for index, filename in enumerate(rows, 1):
        print(f"  Row {index}: {filename}")
    return canvas.size


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Lossless PNG destination")
    parser.add_argument("--columns", nargs="+", default=list(TITLES), help="Input directories, in display order")
    parser.add_argument("--images", nargs="+", help="Exact filenames in desired row order, otherwise sorted")
    parser.add_argument("--panel-width", type=int, default=0, help="0 preserves source resolution")
    parser.add_argument("--gap", type=int, default=12)
    parser.add_argument("--font-size", type=int, default=44)
    parser.add_argument("--font", type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-panel-letters", action="store_true")
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != ".png":
        parser.error("Use a .png output to preserve annotation quality")
    if (min(args.font_size, args.dpi) <= 0 or min(args.gap, args.panel_width) < 0
            or len(args.columns) != len(set(args.columns))):
        parser.error("Invalid sizes or duplicate columns")
    compose(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
