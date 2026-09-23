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
from ultralytics.utils.nms import non_max_suppression


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


def original_box(xyxy: np.ndarray, pad: tuple[int, int, int, int], original_size: tuple[int, int]) -> tuple[float, float, float, float]:
    pad_x, pad_y, resized_w, resized_h = pad
    original_w, original_h = original_size
    scale_x, scale_y = original_w / resized_w, original_h / resized_h
    x1, y1, x2, y2 = xyxy
    x1 = (x1 - pad_x) * scale_x
    y1 = (y1 - pad_y) * scale_y
    x2 = (x2 - pad_x) * scale_x
    y2 = (y2 - pad_y) * scale_y
    return (
        float(np.clip(x1, 0, original_w - 1)), float(np.clip(y1, 0, original_h - 1)),
        float(np.clip(x2, 0, original_w - 1)), float(np.clip(y2, 0, original_h - 1)),
    )


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
            nms_detections, kept_indices = non_max_suppression(
                prediction.detach(), conf_thres=args.conf, iou_thres=args.iou, max_det=args.max_det, return_idxs=True
            )
            detections = [
                (int(candidate), int(detection[5]), float(detection[4]), detection[:4].cpu().numpy())
                for detection, candidate in zip(nms_detections[0], kept_indices[0])
            ]
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
            for detection_index, (candidate_id, class_id, confidence, xyxy) in enumerate(detections, 1):
                box = original_box(xyxy, pad, original.size)
                label = names[int(class_id)] if isinstance(names, dict) else names[int(class_id)]
                labels.append(f"{label} {confidence:.3f}")
                if not args.no_box:
                    drawer.rectangle(box, outline="white", width=3)
                level = feature_for(candidate_id)[1] if args.layer == "detect" else -1
                metadata.append({"image": path.name, "feature_level": level, "class_id": class_id, "class_name": label,
                                 "confidence": f"{confidence:.6f}",
                                 "x1": f"{box[0]:.2f}", "y1": f"{box[1]:.2f}",
                                 "x2": f"{box[2]:.2f}", "y2": f"{box[3]:.2f}"})
                single_heatmap = (color_map(cams[detection_index - 1])[..., :3] * 255).astype(np.uint8)
                single_overlay = (np.asarray(original, dtype=np.float32) * (1 - args.alpha)
                                  + single_heatmap * args.alpha).astype(np.uint8)
                single_rendered = Image.fromarray(single_overlay)
                if not args.no_box:
                    ImageDraw.Draw(single_rendered).rectangle(box, outline="white", width=3)
                single_rendered.save(
                    args.output / f"{path.stem}__det{detection_index:02d}_{label}_{confidence:.2f}_cam.jpg", quality=95
                )
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
