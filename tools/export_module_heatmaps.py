#!/usr/bin/env python3
"""Export activation heatmaps from a selected Ultralytics model module.

The hook captures the module output, averages absolute responses across
channels, and overlays the normalized map on each source image.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image
from ultralytics import YOLO


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True,
                        help="One image or a directory of images.")
    parser.add_argument("--layer", help="Module path, e.g. model.4.direction.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--list-layers", action="store_true")
    return parser.parse_args()


def source_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(item for item in path.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES)


def letterbox(image: Image.Image, size: int) -> tuple[Image.Image, tuple[int, int, int, int]]:
    width, height = image.size
    scale = min(size / width, size / height)
    resized_w, resized_h = round(width * scale), round(height * scale)
    pad_x, pad_y = (size - resized_w) // 2, (size - resized_h) // 2
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(image.resize((resized_w, resized_h), Image.Resampling.BILINEAR), (pad_x, pad_y))
    return canvas, (pad_x, pad_y, resized_w, resized_h)


def normalize(feature: np.ndarray) -> np.ndarray:
    low, high = np.percentile(feature, (1, 99))
    if high <= low:
        return np.zeros_like(feature, dtype=np.float32)
    return np.clip((feature - low) / (high - low), 0, 1).astype(np.float32)


def main() -> None:
    args = parse_args()
    yolo = YOLO(str(args.model))
    model = yolo.model.eval()
    if args.list_layers:
        for name, module in model.named_modules():
            if name:
                print(f"{name}: {module.__class__.__name__}")
        return
    if not args.layer:
        raise ValueError("--layer is required unless --list-layers is used")

    modules = dict(model.named_modules())
    if args.layer not in modules:
        raise ValueError(f"Unknown layer {args.layer!r}. Use --list-layers to inspect available names.")

    device_name = args.device if args.device == "cpu" or args.device.startswith("cuda:") else f"cuda:{args.device}"
    device = torch.device(device_name)
    model.to(device)
    captured: dict[str, torch.Tensor] = {}

    def save_output(_, __, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise TypeError(f"{args.layer} returned {type(output).__name__}, not a 4-D feature tensor")
        captured["feature"] = output.detach()

    handle = modules[args.layer].register_forward_hook(save_output)
    images = source_images(args.source)
    if not images:
        raise FileNotFoundError(f"No images found in {args.source}")
    args.output.mkdir(parents=True, exist_ok=True)
    color_map = colormaps["turbo"]

    try:
        for index, path in enumerate(images, 1):
            original = Image.open(path).convert("RGB")
            network_image, (pad_x, pad_y, resized_w, resized_h) = letterbox(original, args.imgsz)
            array = np.asarray(network_image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
            captured.clear()
            with torch.no_grad():
                model(tensor)
            feature = captured["feature"].abs().mean(dim=1)[0].float().cpu().numpy()
            feature = np.asarray(Image.fromarray(feature).resize((args.imgsz, args.imgsz), Image.Resampling.BILINEAR))
            feature = feature[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w]
            feature = np.asarray(Image.fromarray(feature).resize(original.size, Image.Resampling.BILINEAR))
            heatmap = (color_map(normalize(feature))[..., :3] * 255).astype(np.uint8)
            overlay = (np.asarray(original, dtype=np.float32) * (1 - args.alpha) + heatmap * args.alpha).astype(np.uint8)
            Image.fromarray(overlay).save(args.output / f"{path.stem}_activation.jpg", quality=95)
            print(f"[{index}/{len(images)}] {path.name}")
    finally:
        handle.remove()


if __name__ == "__main__":
    main()
