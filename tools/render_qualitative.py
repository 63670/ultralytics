"""Render GT labels and COCO-format predictions with a unified paper style."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PALETTE = {1: "#00B8D9", 2: "#FF9F1C", 3: "#E71D36"}


def parse_model(value):
    name, path = value.split("=", 1)
    return name, Path(path)


def draw_panel(image, detections, categories, threshold, tile):
    image = image.convert("RGB").resize((tile, tile))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    scale = tile / 640
    for det in detections:
        if det.get("score", 1.0) < threshold:
            continue
        x, y, w, h = det["bbox"]
        x1, y1, x2, y2 = [round(v * scale) for v in (x, y, x + w, y + h)]
        color = PALETTE.get(det["category_id"], "#FFFFFF")
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        label = categories[det["category_id"]]
        draw.text((x1 + 2, max(0, y1 - 12)), label, fill=color, font=font, stroke_width=1, stroke_fill="#000000")
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--images-root", required=True, type=Path)
    parser.add_argument("--images-file", required=True, type=Path)
    parser.add_argument("--prediction", action="append", default=[], type=parse_model, metavar="NAME=JSON")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--tile", type=int, default=256)
    args = parser.parse_args()

    coco = json.loads(args.annotations.read_text(encoding="utf-8"))
    categories = {item["id"]: item["name"] for item in coco["categories"]}
    images = {Path(item["file_name"]).name: item for item in coco["images"]}
    gt = {}
    for ann in coco["annotations"]:
        gt.setdefault(ann["image_id"], []).append(ann)
    selected = [line.strip() for line in args.images_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    models = [(name, json.loads(path.read_text(encoding="utf-8"))) for name, path in args.prediction]
    predictions = []
    for _, records in models:
        by_id = {}
        for item in records:
            by_id.setdefault(item["image_id"], []).append(item)
        predictions.append(by_id)

    title_h, gap, row_label_w = 32, 5, 46
    width = row_label_w + (len(models) + 1) * (args.tile + gap)
    height = title_h + len(selected) * (args.tile + gap)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    headers = ["Labels"] + [name for name, _ in models]
    for col, header in enumerate(headers):
        x = row_label_w + col * (args.tile + gap) + args.tile // 2
        draw.text((x, 10), header, anchor="ma", fill="black", font=font)
    for row, filename in enumerate(selected):
        if filename not in images:
            raise KeyError(f"{filename} is absent from COCO annotations")
        item = images[filename]
        image_path = args.images_root / item["file_name"]
        y = title_h + row * (args.tile + gap)
        draw.text((4, y + args.tile // 2), str(row + 1), anchor="lm", fill="black", font=font)
        source = Image.open(image_path)
        panels = [draw_panel(source, gt.get(item["id"], []), categories, 0.0, args.tile)]
        panels += [draw_panel(source, pred.get(item["id"], []), categories, args.threshold, args.tile) for pred in predictions]
        for col, panel in enumerate(panels):
            canvas.paste(panel, (row_label_w + col * (args.tile + gap), y))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, dpi=(300, 300))


if __name__ == "__main__":
    main()
