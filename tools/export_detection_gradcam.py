#!/usr/bin/env python3
"""Export detection-conditioned Grad-CAM or LayerCAM maps for an Ultralytics detector.

For every image, the script applies class-aware NMS, then backpropagates each
retained detection's class score to a chosen feature module. The individual
maps are combined, so every displayed detection contributes to the heatmap.
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
    parser.add_argument("--layer", default="detect",
                        help="Module path, or 'detect' to automatically use the matched detection-head scale.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--alpha", type=float, default=0.42)
    parser.add_argument("--method", choices=("gradcam", "layercam"), default="layercam",
                        help="CAM variant; LayerCAM is sharper for intermediate feature maps.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--max-det", type=int, default=16)
    parser.add_argument("--no-box", action="store_true", help="Do not draw retained detection boxes.")
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


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x, y, width, height = boxes.unbind(dim=1)
    return torch.stack((x - width / 2, y - height / 2, x + width / 2, y + height / 2), dim=1)


def box_iou(one_box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(one_box[:2], boxes[:, :2])
    bottom_right = torch.minimum(one_box[2:], boxes[:, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=1)
    one_area = (one_box[2] - one_box[0]) * (one_box[3] - one_box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return intersection / (one_area + areas - intersection + 1e-7)


def select_detections(prediction: torch.Tensor, conf: float, iou: float, max_det: int) -> list[tuple[int, int, float, np.ndarray]]:
    """Run lightweight class-aware NMS while retaining raw candidate indices."""
    class_scores = prediction[4:, :]
    scores, class_ids = class_scores.max(dim=0)
    candidate_ids = torch.nonzero(scores >= conf, as_tuple=False).flatten()
    if not len(candidate_ids):
        return []
    boxes = xywh_to_xyxy(prediction[:4, :].T)
    order = candidate_ids[scores[candidate_ids].argsort(descending=True)]
    retained: list[int] = []
    while len(order) and len(retained) < max_det:
        selected = int(order[0])
        retained.append(selected)
        remaining = order[1:]
        if not len(remaining):
            break
        overlap = box_iou(boxes[selected], boxes[remaining])
        same_class = class_ids[remaining] == class_ids[selected]
        order = remaining[~(same_class & (overlap > iou))]
    return [(candidate, int(class_ids[candidate]), float(scores[candidate]),
             prediction[:4, candidate].cpu().numpy()) for candidate in retained]


def main() -> None:
    args = parse_args()
    yolo = YOLO(str(args.model))
    model = yolo.model.eval()
    modules = dict(model.named_modules())
    device_name = args.device if args.device == "cpu" or args.device.startswith("cuda:") else f"cuda:{args.device}"
    device = torch.device(device_name)
    model.to(device)
    # Ultralytics checkpoints are loaded for inference and may have all
    # parameter gradients disabled. Grad-CAM needs a backward graph.
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    captured: dict[str, object] = {}

    def forward_hook(_, __, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise TypeError(f"{args.layer} must return a 4-D feature tensor")
        output.retain_grad()
        captured["feature"] = output

    if args.layer == "detect":
        detect_candidates = [(name, module) for name, module in modules.items()
                             if module.__class__.__name__ == "Detect"]
        if not detect_candidates:
            raise ValueError("Could not find a Detect module in this model.")

        def detect_pre_hook(_, inputs):
            features = list(inputs[0])
            for feature in features:
                feature.retain_grad()
            captured["features"] = features

        handle = detect_candidates[-1][1].register_forward_pre_hook(detect_pre_hook)
    else:
        if args.layer not in modules:
            raise ValueError(f"Unknown layer {args.layer!r}.")
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
            detections = select_detections(prediction[0].detach(), args.conf, args.iou, args.max_det)
            if args.layer == "detect":
                features = captured["features"]
                assert isinstance(features, list)
                candidate_counts = [feature.shape[2] * feature.shape[3] for feature in features]

                def feature_for(candidate_id: int) -> tuple[torch.Tensor, int]:
                    offset = 0
                    for level, (feature, count) in enumerate(zip(features, candidate_counts)):
                        if candidate_id < offset + count:
                            return feature, level
                        offset += count
                    raise IndexError(f"Candidate {candidate_id} is outside detection-head outputs")
            else:
                feature = captured["feature"]
                assert isinstance(feature, torch.Tensor)

            pad_x, pad_y, resized_w, resized_h = pad
            cams: list[np.ndarray] = []
            for target_index, (candidate_id, class_id, _, _) in enumerate(detections):
                model.zero_grad(set_to_none=True)
                if args.layer == "detect":
                    feature, feature_level = feature_for(candidate_id)
                else:
                    feature_level = -1
                feature.grad = None
                score = prediction[0, 4 + class_id, candidate_id]
                score.backward(retain_graph=target_index + 1 < len(detections))
                gradient = feature.grad
                if args.method == "gradcam":
                    weights = gradient.mean(dim=(2, 3), keepdim=True)
                    cam_tensor = torch.relu((weights * feature).sum(dim=1))
                else:
                    cam_tensor = torch.relu((torch.relu(gradient) * feature).sum(dim=1))
                cam = cam_tensor[0].detach().float().cpu().numpy()
                cam = np.asarray(Image.fromarray(cam).resize((args.imgsz, args.imgsz), Image.Resampling.BILINEAR))
                cam = cam[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w]
                cam = np.asarray(Image.fromarray(cam).resize(original.size, Image.Resampling.BILINEAR))
                cams.append(normalize(cam))
            combined_cam = np.maximum.reduce(cams) if cams else np.zeros((original.height, original.width), dtype=np.float32)
            heatmap = (color_map(combined_cam)[..., :3] * 255).astype(np.uint8)
            overlay = (np.asarray(original, dtype=np.float32) * (1 - args.alpha) + heatmap * args.alpha).astype(np.uint8)
            rendered = Image.fromarray(overlay)
            drawer = ImageDraw.Draw(rendered)
            labels: list[str] = []
            for candidate_id, class_id, confidence, xywh in detections:
                box = original_box(xywh, pad, original.size)
                label = names[int(class_id)] if isinstance(names, dict) else names[int(class_id)]
                labels.append(f"{label} {confidence:.3f}")
                if not args.no_box:
                    drawer.rectangle(box, outline="white", width=3)
                level = feature_for(candidate_id)[1] if args.layer == "detect" else -1
                metadata.append({"image": path.name, "feature_level": level, "class_id": class_id, "class_name": label,
                                 "confidence": f"{confidence:.6f}",
                                 "x1": f"{box[0]:.2f}", "y1": f"{box[1]:.2f}",
                                 "x2": f"{box[2]:.2f}", "y2": f"{box[3]:.2f}"})
            rendered.save(args.output / f"{path.stem}_gradcam.jpg", quality=95)
            print(f"[{index}/{len(images)}] {path.name}: {', '.join(labels) or 'no detections'}")
    finally:
        handle.remove()

    with (args.output / "targets.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image", "feature_level", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])
        writer.writeheader()
        writer.writerows(metadata)


if __name__ == "__main__":
    main()
