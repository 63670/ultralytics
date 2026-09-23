#!/usr/bin/env python3
"""Plot mAP--parameter and mAP--FLOPs trade-off scatter plots from a CSV file."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("tools/efficiency_results.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/figures/efficiency_tradeoff.pdf"))
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


LABEL_OFFSETS = {
    "params_m": {
        "YOLOv5n": (7, -10), "YOLOv8n": (8, 4), "YOLO11n": (8, -13),
        "YOLO12n": (8, 6), "YOLO26n": (8, 7),
        "Deformable-DETR": (-80, 5), "Faster-RCNN": (8, 8),
    },
    "flops_g": {
        "YOLOv5n": (8, -11), "YOLOv8n": (8, 4), "YOLO11n": (8, -13),
        "YOLO12n": (8, 6), "YOLO26n": (8, 7),
        "Deformable-DETR": (8, 7), "Faster-RCNN": (8, 8),
    },
}


def draw_panel(ax, rows: list[dict[str, str]], x_key: str, x_label: str, panel: str) -> None:
    colors = plt.get_cmap("tab10").colors
    for index, row in enumerate(rows):
        x, y = float(row[x_key]), float(row["map50_95"])
        highlight = row["highlight"].lower() == "true"
        color = "#d62728" if highlight else colors[index % len(colors)]
        ax.scatter(x, y, s=115 if highlight else 90, c=[color], edgecolors="black",
                   linewidths=0.7, zorder=3)
        offset = LABEL_OFFSETS.get(x_key, {}).get(row["model"], (5, 5))
        ax.annotate(row["model"], (x, y), xytext=offset, textcoords="offset points",
                    fontsize=8, fontweight="bold" if highlight else "normal")
    ax.set_xscale("log")
    ax.set_xlabel(x_label, fontweight="bold")
    ax.set_ylabel("mAP@0.5:0.95 (%)", fontweight="bold")
    ax.set_title(panel, loc="left", fontsize=11, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.45, zorder=0)
    ax.set_ylim(50, 70)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9})
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    draw_panel(axes[0], rows, "params_m", "Parameters (M)",
               "(a) Performance-Efficiency Trade-off (mAP vs Parameters)")
    draw_panel(axes[1], rows, "flops_g", "FLOPs (G)",
               "(b) Performance-Efficiency Trade-off (mAP vs FLOPs)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    print(f"Saved {args.output} and {args.output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
