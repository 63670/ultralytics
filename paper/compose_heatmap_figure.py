"""Compose source, YOLOv8n, and TFRA-YOLO heatmaps into a paper figure."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLUMNS = (
    ("src", "", "Original image"),
    ("yolov8n", "_cam", "YOLOv8n"),
    ("ours", "_cam", "Ours"),
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


def collect_rows(root: Path) -> list[str]:
    names = []
    for source in sorted((root / "src").glob("*.jpg")):
        stem = source.stem
        required = [
            root / "yolov8n" / f"{stem}_cam.jpg",
            root / "ours" / f"{stem}_cam.jpg",
        ]
        if all(path.is_file() for path in required):
            names.append(stem)
    if not names:
        raise FileNotFoundError("No complete source/YOLOv8n/Ours image triplets were found.")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Directory containing src/, yolov8n/, and ours/.")
    parser.add_argument("--output", type=Path, required=True, help="PNG output path.")
    parser.add_argument("--gap", type=int, default=12, help="White spacing between panels in pixels.")
    args = parser.parse_args()

    root = args.input.expanduser().resolve()
    stems = collect_rows(root)
    first = Image.open(root / "src" / f"{stems[0]}.jpg").convert("RGB")
    panel_w, panel_h = first.size
    for directory, suffix, _ in COLUMNS:
        for stem in stems:
            path = root / directory / f"{stem}{suffix}.jpg"
            with Image.open(path) as image:
                if image.size != (panel_w, panel_h):
                    raise ValueError(f"Unexpected image size: {path} is {image.size}, expected {(panel_w, panel_h)}")

    label_h = 74
    width = panel_w * len(COLUMNS) + args.gap * (len(COLUMNS) - 1)
    height = panel_h * len(stems) + args.gap * (len(stems) - 1) + label_h
    canvas = Image.new("RGB", (width, height), "white")

    for row, stem in enumerate(stems):
        y = row * (panel_h + args.gap)
        for col, (directory, suffix, _) in enumerate(COLUMNS):
            x = col * (panel_w + args.gap)
            with Image.open(root / directory / f"{stem}{suffix}.jpg") as image:
                canvas.paste(image.convert("RGB"), (x, y))

    draw = ImageDraw.Draw(canvas)
    font = load_font(30)
    label_y = height - label_h + 19
    for col, (_, _, label) in enumerate(COLUMNS):
        x_center = col * (panel_w + args.gap) + panel_w / 2
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text((x_center - (bbox[2] - bbox[0]) / 2, label_y), label, fill="black", font=font)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=95)
    print(f"Saved {args.output} ({width}x{height}) with {len(stems)} rows.")


if __name__ == "__main__":
    main()
