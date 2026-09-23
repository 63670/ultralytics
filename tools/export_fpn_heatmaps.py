#!/usr/bin/env python3
"""Export fused feature-pyramid activation heatmaps from an Ultralytics detector."""
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
    parser.add_argument("--layers", nargs="+", default=["model.15", "model.18", "model.21"],
                        help="Final FPN feature layers passed to Detect.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--fusion", choices=("mean", "max"), default="mean")
    parser.add_argument("--colormap", choices=("jet", "turbo"), default="jet")
    parser.add_argument("--low-alpha", type=float, default=0.35)
    parser.add_argument("--blur-sigma", type=float, default=1.5)
    return parser.parse_args()


def images(path: Path) -> list[Path]:
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
    modules = dict(model.named_modules())
    missing = set(args.layers) - set(modules)
    if missing:
        raise ValueError(f"Unknown FPN layers: {sorted(missing)}")
    device_name = args.device if args.device == "cpu" or args.device.startswith("cuda:") else f"cuda:{args.device}"
    model.to(device_name)
    outputs: dict[str, torch.Tensor] = {}
    handles = [modules[name].register_forward_hook(lambda _, __, out, n=name: outputs.__setitem__(n, out))
               for name in args.layers]
    args.output.mkdir(parents=True, exist_ok=True)
    cmap = colormaps[args.colormap]
    try:
        for index, path in enumerate(images(args.source), 1):
            original = Image.open(path).convert("RGB")
            network_image, (pad_x, pad_y, resized_w, resized_h) = letterbox(original, args.imgsz)
            array = np.asarray(network_image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device_name)
            outputs.clear()
            with torch.no_grad():
                model(tensor)
            maps = []
            for name in args.layers:
                # Positive channel energy is stable across differently sized FPN levels.
                feature = outputs[name].relu().square().mean(dim=1)[0].float().cpu().numpy()
                feature = np.asarray(Image.fromarray(feature).resize((args.imgsz, args.imgsz), Image.Resampling.BILINEAR))
                feature = feature[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w]
                feature = np.asarray(Image.fromarray(feature).resize(original.size, Image.Resampling.BILINEAR))
                maps.append(normalize(feature))
            fused = np.mean(maps, axis=0) if args.fusion == "mean" else np.maximum.reduce(maps)
            if args.blur_sigma > 0:
                fused = cv2.GaussianBlur(fused.astype(np.float32), (0, 0), args.blur_sigma)
            fused = normalize(fused)
            heatmap = (cmap(fused)[..., :3] * 255).astype(np.uint8)
            alpha = args.low_alpha + (1 - args.low_alpha) * fused
            overlay = (np.asarray(original, dtype=np.float32) * (1 - alpha[..., None]) + heatmap * alpha[..., None]).astype(np.uint8)
            Image.fromarray(overlay).save(args.output / f"{path.stem}_fpn.jpg", quality=95)
            print(f"[{index}] {path.name}")
    finally:
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    main()
