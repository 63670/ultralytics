#!/usr/bin/env python3
"""Export post-Detect-head class-confidence heatmaps for an Ultralytics detector."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image
from ultralytics import YOLO


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--level", choices=("p3", "p4", "p5", "fuse"), default="fuse")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--colormap", choices=("jet", "turbo"), default="jet")
    parser.add_argument("--low-alpha", type=float, default=0.35)
    parser.add_argument("--blur-sigma", type=float, default=1.5)
    return parser.parse_args()


def source_images(path: Path) -> list[Path]:
    return [path] if path.is_file() else sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def letterbox(image: Image.Image, size: int) -> tuple[Image.Image, tuple[int, int, int, int]]:
    width, height = image.size
    scale = min(size / width, size / height)
    resized_w, resized_h = round(width * scale), round(height * scale)
    pad_x, pad_y = (size - resized_w) // 2, (size - resized_h) // 2
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(image.resize((resized_w, resized_h), Image.Resampling.BILINEAR), (pad_x, pad_y))
    return canvas, (pad_x, pad_y, resized_w, resized_h)


def normalize(map_: np.ndarray) -> np.ndarray:
    low, high = np.percentile(map_, (2, 99.5))
    return np.zeros_like(map_, dtype=np.float32) if high <= low else np.clip((map_ - low) / (high - low), 0, 1)


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.model)).model.eval()
    device_name = args.device if args.device == "cpu" or args.device.startswith("cuda:") else f"cuda:{args.device}"
    model.to(device_name)
    args.output.mkdir(parents=True, exist_ok=True)
    cmap = colormaps[args.colormap]
    for index, path in enumerate(source_images(args.source), 1):
        original = Image.open(path).convert("RGB")
        network_image, (pad_x, pad_y, resized_w, resized_h) = letterbox(original, args.imgsz)
        array = np.asarray(network_image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device_name)
        with torch.no_grad():
            raw = model(tensor)[1]
        scores = raw["scores"].sigmoid().amax(dim=1)[0]  # strongest class score per anchor
        level_shapes = [(args.imgsz // 8, args.imgsz // 8), (args.imgsz // 16, args.imgsz // 16),
                        (args.imgsz // 32, args.imgsz // 32)]
        offset, maps = 0, []
        for height, width in level_shapes:
            score_map = scores[offset:offset + height * width].reshape(height, width).float().cpu().numpy()
            offset += height * width
            score_map = np.asarray(Image.fromarray(score_map).resize((args.imgsz, args.imgsz), Image.Resampling.BILINEAR))
            score_map = score_map[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w]
            score_map = np.asarray(Image.fromarray(score_map).resize(original.size, Image.Resampling.BILINEAR))
            maps.append(normalize(score_map))
        selected = maps[{"p3": 0, "p4": 1, "p5": 2}[args.level]] if args.level != "fuse" else np.mean(maps, axis=0)
        if args.blur_sigma > 0:
            selected = cv2.GaussianBlur(selected.astype(np.float32), (0, 0), args.blur_sigma)
        selected = normalize(selected)
        heatmap = (cmap(selected)[..., :3] * 255).astype(np.uint8)
        alpha = args.low_alpha + (1 - args.low_alpha) * selected
        overlay = (np.asarray(original, dtype=np.float32) * (1 - alpha[..., None]) + heatmap * alpha[..., None]).astype(np.uint8)
        Image.fromarray(overlay).save(args.output / f"{path.stem}_head_{args.level}.jpg", quality=95)
        print(f"[{index}] {path.name}")


if __name__ == "__main__":
    main()
