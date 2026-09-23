#!/usr/bin/env python3
"""Render a qualitative detection comparison from YOLO-format labels.

Ground-truth labels use ``class cx cy w h``; predictions may additionally
contain a confidence score as the sixth field. Coordinates are normalized.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CLASS_NAMES = ("row", "col", "hole")
COLORS = ((33, 150, 243), (0, 188, 212), (255, 255, 255))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True,
                        help="Ground-truth YOLO label directory.")
    parser.add_argument("--images-file", type=Path, required=True)
    parser.add_argument("--prediction", action="append", default=[], metavar="NAME=DIR",
                        help="Prediction label directory, repeat for each model.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--title-font-size", type=int, default=18)
    parser.add_argument("--show-score", action="store_true")
    return parser.parse_args()


def parse_prediction(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"--prediction must be NAME=DIR, got: {spec}")
    name, directory = spec.split("=", 1)
    return name, Path(directory)


def read_yolo(path: Path) -> list[tuple[int, float, float, float, float, float | None]]:
    if not path.exists():
        return []
    objects = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) not in (5, 6):
            raise ValueError(f"{path}:{line_no}: expected 5 or 6 fields")
        cls, cx, cy, width, height = map(float, fields[:5])
        score = float(fields[5]) if len(fields) == 6 else None
        objects.append((int(cls), cx, cy, width, height, score))
    return objects


def draw_tile(image_path: Path, label_path: Path, tile: int, threshold: float,
              show_score: bool) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    canvas = image.resize((tile, tile), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(canvas)
    for cls, cx, cy, box_w, box_h, score in read_yolo(label_path):
        if score is not None and score < threshold:
            continue
        if not 0 <= cls < len(CLASS_NAMES):
            continue
        x1 = (cx - box_w / 2) * tile
        y1 = (cy - box_h / 2) * tile
        x2 = (cx + box_w / 2) * tile
        y2 = (cy + box_h / 2) * tile
        color = COLORS[cls]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        text = CLASS_NAMES[cls]
        if show_score and score is not None:
            text += f" {score:.2f}"
        draw.text((x1 + 2, max(0, y1 - 12)), text, fill=color, stroke_width=1,
                  stroke_fill=(0, 0, 0))
    return canvas


def main() -> None:
    args = parse_args()
    models = [parse_prediction(spec) for spec in args.prediction]
    names = [line.strip() for line in args.images_file.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if not names:
        raise ValueError("No image names found in --images-file")
    columns = [("Labels", args.labels_dir), *models]
    footer_h, pad = args.title_font_size + 14, 3
    try:
        title_font = ImageFont.truetype("DejaVuSans.ttf", args.title_font_size)
    except OSError:
        title_font = ImageFont.load_default()
    grid = Image.new("RGB", (len(columns) * (args.tile + pad) + pad,
                              len(names) * (args.tile + pad) + footer_h + pad), "white")
    draw = ImageDraw.Draw(grid)
    for col, (title, _) in enumerate(columns):
        x = pad + col * (args.tile + pad)
        draw.text((x + args.tile / 2, len(names) * (args.tile + pad) + pad + 5),
                  title, fill="black", font=title_font, anchor="mt")
    for row, image_name in enumerate(names):
        image_path = args.images_dir / image_name
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        stem = image_path.stem
        for col, (_, label_dir) in enumerate(columns):
            label_path = label_dir / f"{stem}.txt"
            tile = draw_tile(image_path, label_path, args.tile, args.threshold,
                             args.show_score)
            x = pad + col * (args.tile + pad)
            y = pad + row * (args.tile + pad)
            grid.paste(tile, (x, y))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.output, dpi=(300, 300))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
