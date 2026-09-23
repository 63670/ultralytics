#!/usr/bin/env python3
"""Export detection-conditioned Grad-CAM or LayerCAM maps for an Ultralytics detector.

For every image, the script selects the model's highest-confidence raw
detection candidate and backpropagates that candidate's class score to a
chosen feature module.  The resulting map is therefore tied to one predicted
object rather than being a channel-averaged feature response.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image, ImageDraw
from ultralytics import YOLO


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--layer", required=True, help="Module path, e.g. model.4.direction")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--alpha", type=float, default=0.42)
    parser.add_argument("--method", choices=("gradcam", "layercam"), default="layercam",
                        help="CAM variant; LayerCAM is sharper for intermediate feature maps.")
    parser.add_argument("--no-box", action="store_true", help="Do not draw the target prediction box.")
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


def normalize(cam: np.ndarray) -> np.ndarray:
    cam = np.maximum(cam, 0)
    low, high = np.percentile(cam, (2, 99.5))
    if high <= low:
        return np.zeros_like(cam, dtype=np.float32)
    return np.clip((cam - low) / (high - low), 0, 1).astype(np.float32)


def original_box(xywh: np.ndarray, pad: tuple[int, int, int, int], original_size: tuple[int, int]) -> tuple[float, float, float, float]:
    pad_x, pad_y, resized_w, resized_h = pad
    original_w, original_h = original_size
    scale_x, scale_y = original_w / resized_w, original_h / resized_h
    x, y, width, height = xywh
    x1 = (x - width / 2 - pad_x) * scale_x
    y1 = (y - height / 2 - pad_y) * scale_y
    x2 = (x + width / 2 - pad_x) * scale_x
    y2 = (y + height / 2 - pad_y) * scale_y
    return (
        float(np.clip(x1, 0, original_w - 1)), float(np.clip(y1, 0, original_h - 1)),
        float(np.clip(x2, 0, original_w - 1)), float(np.clip(y2, 0, original_h - 1)),
    )


def main() -> None:
    args = parse_args()
    yolo = YOLO(str(args.model))
    model = yolo.model.eval()
    modules = dict(model.named_modules())
    if args.layer not in modules:
        raise ValueError(f"Unknown layer {args.layer!r}.")
    device_name = args.device if args.device == "cpu" or args.device.startswith("cuda:") else f"cuda:{args.device}"
    device = torch.device(device_name)
    model.to(device)
    # Ultralytics checkpoints are loaded for inference and may have all
    # parameter gradients disabled. Grad-CAM needs a backward graph.
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    captured: dict[str, torch.Tensor] = {}

    def forward_hook(_, __, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise TypeError(f"{args.layer} must return a 4-D feature tensor")
        output.retain_grad()
        captured["feature"] = output

    handle = modules[args.layer].register_forward_hook(forward_hook)
    images = source_images(args.source)
    if not images:
        raise FileNotFoundError(f"No images found in {args.source}")
    args.output.mkdir(parents=True, exist_ok=True)
    color_map = colormaps["turbo"]
    names = yolo.names
    metadata: list[dict[str, object]] = []

    try:
        for index, path in enumerate(images, 1):
            original = Image.open(path).convert("RGB")
            network_image, pad = letterbox(original, args.imgsz)
            array = np.asarray(network_image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
            captured.clear()
            model.zero_grad(set_to_none=True)
            prediction = model(tensor)[0]
            class_scores = prediction[0, 4:, :]
            flat_index = int(class_scores.reshape(-1).argmax())
            candidate_count = class_scores.shape[1]
            class_id, candidate_id = divmod(flat_index, candidate_count)
            score = class_scores[class_id, candidate_id]
            score.backward()
            feature = captured["feature"]
            gradient = feature.grad
            if args.method == "gradcam":
                weights = gradient.mean(dim=(2, 3), keepdim=True)
                cam_tensor = torch.relu((weights * feature).sum(dim=1))
            else:
                # Preserve spatial gradients instead of averaging them. This is
                # particularly useful for compact fabric defects at P3.
                cam_tensor = torch.relu((torch.relu(gradient) * feature).sum(dim=1))
            cam = cam_tensor[0].detach().float().cpu().numpy()
            cam = np.asarray(Image.fromarray(cam).resize((args.imgsz, args.imgsz), Image.Resampling.BILINEAR))
            pad_x, pad_y, resized_w, resized_h = pad
            cam = cam[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w]
            cam = np.asarray(Image.fromarray(cam).resize(original.size, Image.Resampling.BILINEAR))
            heatmap = (color_map(normalize(cam))[..., :3] * 255).astype(np.uint8)
            overlay = (np.asarray(original, dtype=np.float32) * (1 - args.alpha) + heatmap * args.alpha).astype(np.uint8)
            box = original_box(prediction[0, :4, candidate_id].detach().cpu().numpy(), pad, original.size)
            rendered = Image.fromarray(overlay)
            if not args.no_box:
                ImageDraw.Draw(rendered).rectangle(box, outline="white", width=3)
            rendered.save(args.output / f"{path.stem}_gradcam.jpg", quality=95)
            label = names[int(class_id)] if isinstance(names, dict) else names[int(class_id)]
            metadata.append({"image": path.name, "class_id": class_id, "class_name": label,
                             "confidence": f"{float(score.detach()):.6f}",
                             "x1": f"{box[0]:.2f}", "y1": f"{box[1]:.2f}",
                             "x2": f"{box[2]:.2f}", "y2": f"{box[3]:.2f}"})
            print(f"[{index}/{len(images)}] {path.name}: {label} {float(score.detach()):.3f}")
    finally:
        handle.remove()

    with (args.output / "targets.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])
        writer.writeheader()
        writer.writerows(metadata)


if __name__ == "__main__":
    main()
