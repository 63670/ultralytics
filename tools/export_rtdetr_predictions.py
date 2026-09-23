#!/usr/bin/env python3
"""Export RT-DETR predictions to COCO JSON for a configured validation dataset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="RT-DETR PyTorch repository root.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--ann-file", type=Path,
                        help="Override val_dataloader.dataset.ann_file (for a subset).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.repo))
    import torch
    from src.core import YAMLConfig
    from src.misc import dist_utils
    from src.solver import TASKS

    dist_utils.setup_distributed(0, "builtin")
    overrides = {"resume": str(args.checkpoint), "device": args.device}
    if args.ann_file:
        overrides["val_dataloader"] = {"dataset": {"ann_file": str(args.ann_file)}}
    cfg = YAMLConfig(str(args.config), **overrides)
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.eval()
    model = solver.ema.module if solver.ema else solver.model
    records = []
    with torch.no_grad():
        for samples, targets in solver.val_dataloader:
            samples = samples.to(solver.device)
            outputs = model(samples)
            sizes = torch.stack([target["orig_size"] for target in targets]).to(solver.device)
            results = solver.postprocessor(outputs, sizes)
            for target, result in zip(targets, results):
                image_id = int(target["image_id"].item())
                for label, box, score in zip(result["labels"], result["boxes"], result["scores"]):
                    score_value = float(score.item())
                    if score_value < args.score_threshold:
                        continue
                    x1, y1, x2, y2 = [float(value) for value in box.cpu().tolist()]
                    records.append({
                        "image_id": image_id,
                        "category_id": int(label.item()),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": score_value,
                    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records), encoding="utf-8")
    print(f"Saved {len(records)} predictions to {args.output}")
    dist_utils.cleanup()


if __name__ == "__main__":
    main()
