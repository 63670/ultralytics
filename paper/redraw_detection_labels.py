#!/usr/bin/env python3
"""Fetch selected original images and saved predictions, then redraw large detection labels.

Run on the server (requires Pillow, no model/GPU):
    python paper/redraw_detection_labels.py --font-size 44 --model-font-size 48
The five default rows follow the author's selected order. Footer model names have no letter prefixes.
Use --ssh only when reading the server from another computer.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import inspect
import io
import json
import math
import shlex
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Allow both direct execution and importing this module in tests.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compose_detection_figure import compose

PAPER = Path(__file__).resolve().parent
MODELS = {"GT": "GT", "yolov8n": "YOLOv8n", "RT-DETR-R18": "RT-DETR-R18",
          "Faster-R-CNN-R50": "Faster-R-CNN-R50", "ours": "PAGI-YOLO"}
COLORS = ("#ff4444", "#0099ff", "#00bb66")
DEFAULT_IMAGES = [
    "VFDS66RK26040002_20260801_195949765.jpg.png",
    "VFDS66RK26040002_20260809_022619641.jpg.png",
    "VFDS66RK26040002_20260727_193808156.jpg.png",
    "VFDS66RK26040002_20260805_002859637.jpg.png",
    "VFDS66RK26040002_20260727_200936881.jpg.png",
]


def read_bundle(request):
    """Read only the selected metadata and clean sources; also runs verbatim on the SSH host."""
    root = Path(request["source"])
    bundle = {"images": {}, "models": {}}
    for alias, model in request["models"].items():
        model_root = root/model
        manifest = json.loads((model_root/"manifest.json").read_text(encoding="utf-8"))
        items = {}
        for filename in request["images"]:
            if Path(filename).name != filename:
                raise ValueError("Image selections must be filenames, without directory components")
            splits = ("train", "val", "test") if request["split"] == "auto" else (request["split"],)
            paths = [model_root/split/Path(filename).with_suffix(".json") for split in splits]
            paths = [p for p in paths if p.is_file()]
            if len(paths) != 1:
                raise ValueError(f"Expected one saved prediction for {model}/{filename}, found {paths}")
            path = paths[0]
            record = json.loads(path.read_text(encoding="utf-8"))
            record["export_split"] = path.parent.name
            source = Path(record["image"])
            if source.name + ".png" != filename:
                raise ValueError(f"Saved source filename does not match selection: {path}")
            content = source.read_bytes()
            source_hash = hashlib.sha256(content).hexdigest()
            if record["image_sha256"] != source_hash:
                raise ValueError(f"Source hash differs from saved predictions: {path}")
            if filename in bundle["images"]:
                if bundle["images"][filename]["sha256"] != source_hash:
                    raise ValueError(f"Models used different original images: {filename}")
            else:
                bundle["images"][filename] = {"sha256": source_hash, "source": str(source),
                                               "split": record["export_split"],
                                               "content_base64": base64.b64encode(content).decode("ascii")}
            items[filename] = record
        bundle["models"][alias] = {"name": model, "classes": manifest["classes"], "items": items}
    return bundle


def fetch(request, host=None, port=13022):
    if host is None:
        return read_bundle(request)
    # Send the request through stdin, and quote the fixed Python reader for the remote POSIX shell.
    code = ("from pathlib import Path\nimport json, base64, hashlib, sys\n"
            + inspect.getsource(read_bundle) + "\nprint(json.dumps(read_bundle(json.load(sys.stdin))))\n")
    result = subprocess.run(["ssh", "-p", str(port), "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                             host, "python3 -c " + shlex.quote(code)],
                            input=json.dumps(request), capture_output=True, text=True, encoding="utf-8", timeout=120)
    if result.returncode:
        raise RuntimeError(f"SSH read failed:\n{result.stderr}")
    return json.loads(result.stdout)


def load_font(size, custom=None):
    for path in ([str(custom)] if custom else ["DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf",
                                               "LiberationSans-Regular.ttf"]):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    raise FileNotFoundError("Specify a scalable TrueType font with --font")


def overlap(a, b):
    return max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))


def label_rect(box, text_size, image_size, occupied, boxes):
    """Choose an in-image label position that favors clear pixels and avoids other labels."""
    x1, y1, x2, y2 = box
    tw, th = text_size
    iw, ih = image_size
    if tw > iw or th > ih:
        raise ValueError("Label exceeds image size; reduce --font-size")
    candidates = [(x1, y1-th-3), (x2+4, y1), (x1-tw-4, y1), (x1, y2+4), (x1, y1+4)]
    for offset in range(1, ih//max(1, th)+1):
        candidates.extend([(x1, y1-offset*(th+4)), (x1, y1+offset*(th+4))])
    options = []
    for rank, (x, y) in enumerate(candidates):
        x, y = max(0, min(iw-tw, x)), max(0, min(ih-th, y))
        rect = (x, y, x+tw, y+th)
        # Preserve every detection; only text placement may move.
        penalty = 1000*sum(overlap(rect, r) for r in occupied) + sum(overlap(rect, b) for b in boxes)
        options.append((penalty, rank, rect))
    return min(options)[2]


def redraw(image, detections, names, text_font, line_width=4, show_conf=True):
    result = image.copy()
    painter = ImageDraw.Draw(result)
    boxes, descriptions = [], []
    for record in detections:
        cls = record["class_id"]
        coords = record["xyxy"]
        score = record["score"]
        if (not isinstance(cls, int) or not 0 <= cls < len(names) or len(coords) != 4
                or not all(math.isfinite(v) for v in coords)
                or (score is not None and not math.isfinite(score))):
            raise ValueError("Invalid detection metadata")
        x1, y1, x2, y2 = coords
        box = (max(0, min(image.width-1, x1)), max(0, min(image.height-1, y1)),
               max(0, min(image.width-1, x2)), max(0, min(image.height-1, y2)))
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Non-positive detection box")
        text = names[cls] + (f" {score:.2f}" if show_conf and score is not None else "")
        boxes.append(box)
        descriptions.append((text, COLORS[cls % len(COLORS)]))
        painter.rectangle(box, outline=COLORS[cls % len(COLORS)], width=line_width)
    occupied, label_metadata = [], []
    for box, (text, color) in zip(boxes, descriptions):
        bounds = painter.textbbox((0, 0), text, font=text_font)
        size = (bounds[2]-bounds[0]+10, bounds[3]-bounds[1]+10)
        rect = label_rect(box, size, image.size, occupied, boxes)
        occupied.append(rect)
        painter.rectangle(rect, fill=color)
        painter.text((rect[0]+5-bounds[0], rect[1]+5-bounds[1]), text, font=text_font, fill="white")
        label_metadata.append({"text": text, "rectangle": rect})
    return result, label_metadata


def render_bundle(bundle, filenames, args):
    text_font = load_font(args.font_size, args.font)
    panel_root = args.work/"panels"
    metadata = {"font_size": args.font_size, "model_font_size": args.model_font_size,
                "panel_letters": False, "rows": filenames, "show_conf": not args.no_conf, "models": {},
                "images": {name: {key: source[key] for key in ("source", "sha256", "split")}
                           for name, source in bundle["images"].items()}}
    for alias, model in bundle["models"].items():
        target = panel_root/alias
        target.mkdir(parents=True, exist_ok=True)
        metadata["models"][alias] = {}
        for filename in filenames:
            source = bundle["images"][filename]
            content = base64.b64decode(source["content_base64"], validate=True)
            if hashlib.sha256(content).hexdigest() != source["sha256"]:
                raise ValueError(f"Downloaded image hash mismatch: {filename}")
            with Image.open(io.BytesIO(content)) as original:
                image = original.convert("RGB")
            record = model["items"][filename]
            if list(image.size) != record["size"]:
                raise ValueError(f"Image dimensions differ from saved box coordinates: {filename}")
            rendered, labels = redraw(image, record["detections"], model["classes"], text_font,
                                      args.line_width, not args.no_conf)
            rendered.save(target/filename, dpi=(300, 300))
            # The original detection list is copied verbatim for auditing; no filtering or new inference.
            metadata["models"][alias][filename] = {"source_sha256": source["sha256"], "split": source["split"],
                                                     "detections": record["detections"], "labels": labels}
    args.work.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/"qualitative_large_labels.sources.json").write_bytes((json.dumps(metadata, indent=2)+"\n").encode())
    for suffix, columns in (("", list(MODELS)), ("_baseline", ["GT", "yolov8n", "ours"])):
        compose(Namespace(input=panel_root, output=args.output_dir/f"qualitative_large_labels{suffix}.png",
                          columns=columns, images=filenames, panel_width=0, gap=args.gap,
                          font_size=args.model_font_size, font=args.model_font, dpi=300, no_panel_letters=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="/home/tkz/code/github/ultralytics/runs/detection_comparison")
    parser.add_argument("--ssh", help="Read-only SSH host, e.g. tkz@8.138.122.96; omit for local/server input")
    parser.add_argument("--port", type=int, default=13022)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--selection", type=Path, help="Optional layout JSON overriding the five default rows")
    selection.add_argument("--images", nargs="+", help="Optional exact filenames, in desired row order")
    parser.add_argument("--split", default="auto", choices=("auto", "train", "val", "test"))
    parser.add_argument("--font-size", type=int, default=44, help="Detection class/confidence font size")
    parser.add_argument("--font", type=Path, help="Detection class/confidence TrueType font")
    parser.add_argument("--model-font-size", type=int, default=48, help="Footer model-name font size")
    parser.add_argument("--model-font", type=Path, help="Footer model-name TrueType font")
    parser.add_argument("--gap", type=int, default=12, help="White spacing between panels")
    parser.add_argument("--line-width", type=int, default=4)
    parser.add_argument("--no-conf", action="store_true", help="Hide scores; keep class names")
    parser.add_argument("--work", type=Path, help="Cache/panel directory; defaults to OUTPUT/work")
    parser.add_argument("--output-dir", type=Path, help="Defaults to SOURCE/paper_figures in server mode")
    args = parser.parse_args(argv)
    if min(args.font_size, args.model_font_size, args.line_width, args.port) <= 0 or args.gap < 0:
        parser.error("Font size, line width, and SSH port must be positive")
    if args.output_dir is None:
        args.output_dir = PAPER/"figures" if args.ssh else Path(args.source)/"paper_figures"
    if args.work is None:
        args.work = args.output_dir/"work"
    filenames = (json.loads(args.selection.read_text(encoding="utf-8"))["rows"] if args.selection
                 else args.images if args.images else list(DEFAULT_IMAGES))
    if not filenames or len(filenames) != len(set(filenames)):
        parser.error("Selection must contain unique image filenames")
    request = {"source": args.source, "models": MODELS, "images": filenames, "split": args.split}
    bundle = fetch(request, args.ssh, args.port)
    for filename, source in bundle["images"].items():
        print(f"Source split: {source['split']} / {filename}")
    # Cache exact source bytes/metadata to allow inspection without a second network read.
    args.work.mkdir(parents=True, exist_ok=True)
    (args.work/"source_bundle.json").write_bytes((json.dumps(bundle)+"\n").encode())
    render_bundle(bundle, filenames, args)
    print(f"Redrew {len(filenames)} images for {len(MODELS)} columns, font size {args.font_size}; no inference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
