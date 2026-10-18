#!/usr/bin/env python3
"""Benchmark batch-1 FP32 model forwards on CUDA, excluding image I/O and external postprocessing."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import sys
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True, help="Shared directory of real test images.")
    parser.add_argument("--name", help="Display name, e.g. PAGI-YOLO.")
    parser.add_argument("--backend", choices=("ultralytics", "mmdet", "rtdetrv2"), default="ultralytics")
    parser.add_argument("--config", type=Path, help="Training configuration for MMDetection or RT-DETRv2.")
    parser.add_argument("--repo", type=Path, help="RT-DETR/rtdetrv2_pytorch directory.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index, default 0.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--limit", type=int, default=32, help="Cache the first N sorted test images (default 32).")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200, help="Forwards per timed round.")
    parser.add_argument("--repeat", type=int, default=5, help="Timed rounds, default 5.")
    parser.add_argument("--fuse", action=argparse.BooleanOptionalAction, default=True,
                        help="Apply each backend's inference fusion/deployment conversion (default enabled).")
    parser.add_argument("--output", type=Path, help="Save protocol metadata and measurements as JSON.")
    args = parser.parse_args(argv)
    if not args.weights.is_file() or not args.images.is_dir():
        parser.error("--weights must be a checkpoint file and --images must be an image directory.")
    if min(args.imgsz, args.limit, args.iterations, args.repeat) < 1 or args.warmup < 0 or args.device < 0:
        parser.error("Sizes/counts must be positive; warmup/device must be non-negative.")
    if args.backend != "ultralytics" and (not args.config or not args.config.is_file()):
        parser.error("MMDetection and RT-DETRv2 require an existing --config.")
    if args.backend == "rtdetrv2" and (not args.repo or not (args.repo / "src/core").is_dir()):
        parser.error("--repo must point to RT-DETR/rtdetrv2_pytorch.")
    if args.output and args.output.suffix.lower() != ".json":
        parser.error("--output must use a .json extension.")
    if args.output and args.output.resolve() in {args.weights.resolve(), args.config.resolve() if args.config else None}:
        parser.error("--output must not overwrite the checkpoint or configuration.")
    return args


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def image_files(directory, limit):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    paths = sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in extensions)
    if not paths:
        raise ValueError(f"No images in {directory}")
    return paths[:limit]


def build_backend(args, torch, device):
    """Return an eval-mode model, a preprocessing callable, and its timed forward callable."""
    if args.backend == "ultralytics":
        from ultralytics import YOLO

        model = YOLO(str(args.weights)).model.float().to(device).eval()
        if args.fuse:
            model.fuse(verbose=False)
        return model, lambda image: image, lambda image: model(image)

    if args.backend == "mmdet":
        from mmdet.apis import init_detector
        from mmdet.structures import DetDataSample

        model = init_detector(str(args.config), str(args.weights), device=str(device)).float().eval()
        if args.fuse:
            from mmcv.cnn import fuse_conv_bn

            model = fuse_conv_bn(model)

        def prepare(image):
            # MMDetection's image loader normally supplies BGR values on the 0..255 scale.
            sample = DetDataSample(metainfo={
                "img_shape": (args.imgsz, args.imgsz), "ori_shape": (args.imgsz, args.imgsz),
                "batch_input_shape": (args.imgsz, args.imgsz), "scale_factor": (1.0, 1.0),
            })
            batch = model.data_preprocessor({"inputs": [image[0, [2, 1, 0]] * 255.0],
                                             "data_samples": [sample]}, training=False)
            return batch["inputs"], batch["data_samples"]

        # For Faster R-CNN this includes RPN proposal selection required by the RoI head.
        return model, prepare, lambda batch: model(batch[0], batch[1], mode="tensor")

    sys.path.insert(0, str(args.repo.resolve()))
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(args.config), resume=str(args.weights))
    # The full checkpoint supplies backbone weights; avoid a separate ImageNet download.
    for value in cfg.yaml_cfg.values():
        if isinstance(value, dict) and "pretrained" in value:
            value["pretrained"] = False
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    state = checkpoint["ema"]["module"] if checkpoint.get("ema") is not None else checkpoint["model"]
    model = cfg.model
    model.load_state_dict(state, strict=True)
    model = model.float().to(device).eval()
    if args.fuse:
        model.deploy()
        model.to(device).eval()
    # Match the repository's deployment reference: Resize + RGB ToTensor, no postprocessor.
    return model, lambda image: image, lambda image: model(image)


def cache_images(paths, size, device, prepare, torch):
    import numpy as np
    from PIL import Image

    batches = []
    for path in paths:
        with Image.open(path) as image:
            array = np.array(image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR),
                             dtype=np.float32, copy=True)
        rgb = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device).div_(255.0)
        batch = prepare(rgb)
        tensor = batch[0] if isinstance(batch, tuple) else batch
        if tuple(tensor.shape) != (1, 3, size, size):
            raise ValueError(f"Preprocessed input shape {tuple(tensor.shape)} does not match batch=1, 3x{size}x{size}.")
        batches.append(batch)
    return batches


def measure(forward, batches, warmup, iterations, repeat, torch):
    """Measure CUDA event time for complete forward calls, with no input preparation in the timed loop."""
    if not batches or min(iterations, repeat) < 1 or warmup < 0:
        raise ValueError("Need inputs, positive iterations/repeat and non-negative warmup.")
    means = []
    cursor = 0
    with torch.inference_mode():
        for _ in range(warmup):
            forward(batches[cursor % len(batches)])
            cursor += 1
        torch.cuda.synchronize()
        for _ in range(repeat):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                forward(batches[cursor % len(batches)])
                cursor += 1
            end.record()
            torch.cuda.synchronize()
            means.append(start.elapsed_time(end) / iterations)
    if not all(math.isfinite(value) and value > 0 for value in means):
        raise RuntimeError(f"Invalid timing report: {means}")
    mean = statistics.mean(means)
    return {"round_mean_ms": means, "mean_ms": mean, "std_of_round_means_ms": statistics.pstdev(means),
            "fps": 1000.0 / mean, "timed_forwards": iterations * repeat}


def run(args):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("This protocol requires CUDA. Run it in the RTX 3090 environment.")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    paths = image_files(args.images, args.limit)
    model, prepare, forward = build_backend(args, torch, device)
    with torch.inference_mode():
        batches = cache_images(paths, args.imgsz, device, prepare, torch)
    result = measure(forward, batches, args.warmup, args.iterations, args.repeat, torch)
    versions = {}
    for package in ("ultralytics", "torchvision", "mmcv", "mmcv-lite", "mmengine", "mmdet"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    result.update({
        "name": args.name or args.weights.stem, "backend": args.backend,
        "weights": str(args.weights.resolve()), "weights_sha256": digest(args.weights),
        "config": str(args.config.resolve()) if args.config else None,
        "config_sha256": digest(args.config) if args.config else None,
        "gpu": torch.cuda.get_device_name(device), "device": args.device,
        "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "packages": versions,
        "python": platform.python_version(), "precision": "FP32", "tf32": False,
        "batch": 1, "imgsz": [args.imgsz, args.imgsz], "fuse": args.fuse,
        "warmup": args.warmup, "iterations": args.iterations, "repeat": args.repeat,
        "parameters_after_conversion": sum(p.numel() for p in model.parameters()),
        "images": [{"path": str(p.resolve()), "sha256": digest(p)} for p in paths],
        "scope": "CUDA-event model forward; excludes image I/O, preprocessing, transfers and external postprocessing. "
                 "Native forward decoding/selection, including Faster R-CNN RPN proposals, remains included.",
    })
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"{result['name']}: {result['mean_ms']:.3f} ms/image, {result['fps']:.2f} FPS")
    return result


def main():
    args = parse_args()
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
