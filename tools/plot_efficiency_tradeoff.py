#!/usr/bin/env python3
"""Plot mAP--parameter and mAP--FLOPs trade-off scatter plots from a CSV file."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("tools/efficiency_results.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/figures/efficiency_tradeoff.pdf"))
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--no-text", action="store_true",
                        help="Hide model annotations, titles, and axis titles for manual typesetting.")
    parser.add_argument("--compress-gap", action="store_true",
                        help="Use visibly broken x-axes to compress empty parameter/FLOPs ranges.")
    parser.add_argument("--soft-compress", action="store_true",
                        help="Continuously compress empty x-axis ranges without a visible axis break.")
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


def draw_panel(ax, rows: list[dict[str, str]], x_key: str, x_label: str, panel: str,
               no_text: bool, xlim: tuple[float, float] | None = None,
               ticks: tuple[float, ...] | None = None, show_y: bool = True,
               soft_compress: bool = False) -> None:
    colors = plt.get_cmap("tab10").colors
    for index, row in enumerate(rows):
        x, y = float(row[x_key]), float(row["map50_95"])
        if xlim is not None and not xlim[0] <= x <= xlim[1]:
            continue
        highlight = row["highlight"].lower() == "true"
        color = "#d62728" if highlight else colors[index % len(colors)]
        ax.scatter(x, y, s=185 if highlight else 150, c=[color], edgecolors="black",
                   linewidths=0.8, alpha=0.72, zorder=3)
        if not no_text:
            offset = LABEL_OFFSETS.get(x_key, {}).get(row["model"], (5, 5))
            ax.annotate(row["model"], (x, y), xytext=offset, textcoords="offset points",
                        fontsize=8, fontweight="bold" if highlight else "normal")
    if soft_compress:
        ax.set_xscale("symlog", linthresh=5 if x_key == "params_m" else 10,
                      linscale=0.55, base=10)
    else:
        ax.set_xscale("log")
    if xlim is None:
        if x_key == "params_m":
            xlim, ticks = ((0.8, 50), (1, 2, 5, 10, 20, 50)) if soft_compress else ((1.2, 50), (2, 5, 10, 20, 50))
        else:
            xlim, ticks = ((1.5, 250), (2, 5, 10, 20, 50, 100, 200)) if soft_compress else ((3, 250), (5, 10, 20, 50, 100, 200))
    ax.set_xlim(*xlim)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    if not show_y:
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.spines["left"].set_visible(False)
    if not no_text:
        ax.set_xlabel(x_label, fontweight="bold")
        ax.set_ylabel("mAP@0.5:0.95 (%)", fontweight="bold")
        ax.set_title(panel, loc="left", fontsize=11, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.45, zorder=0)
    ax.set_ylim(50, 70)


def add_break_marks(left, right) -> None:
    """Draw diagonal marks that explicitly indicate a removed x-axis interval."""
    size = 0.014
    kwargs = dict(color="black", clip_on=False, linewidth=1.0)
    left.plot((1 - size, 1 + size), (-size, +size), transform=left.transAxes, **kwargs)
    left.plot((1 - size, 1 + size), (1 - size, 1 + size), transform=left.transAxes, **kwargs)
    right.plot((-size, +size), (-size, +size), transform=right.transAxes, **kwargs)
    right.plot((-size, +size), (1 - size, 1 + size), transform=right.transAxes, **kwargs)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9})
    if not args.compress_gap:
        figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
        draw_panel(axes[0], rows, "params_m", "Parameters (M)",
                   "(a) Performance-Efficiency Trade-off (mAP vs Parameters)", args.no_text,
                   soft_compress=args.soft_compress)
        draw_panel(axes[1], rows, "flops_g", "FLOPs (G)",
                   "(b) Performance-Efficiency Trade-off (mAP vs FLOPs)", args.no_text,
                   soft_compress=args.soft_compress)
    else:
        figure = plt.figure(figsize=(10.5, 4.1), constrained_layout=True)
        grid = figure.add_gridspec(1, 5, width_ratios=(1.05, 1.3, 0.18, 1.05, 1.3))
        param_left = figure.add_subplot(grid[0, 0])
        param_right = figure.add_subplot(grid[0, 1], sharey=param_left)
        flops_left = figure.add_subplot(grid[0, 3])
        flops_right = figure.add_subplot(grid[0, 4], sharey=flops_left)
        draw_panel(param_left, rows, "params_m", "", "", args.no_text, (1.2, 6), (2, 5))
        draw_panel(param_right, rows, "params_m", "", "", args.no_text, (18, 50), (20, 50), False)
        draw_panel(flops_left, rows, "flops_g", "", "", args.no_text, (3, 12), (5, 10))
        draw_panel(flops_right, rows, "flops_g", "", "", args.no_text, (35, 230), (50, 100, 200), False)
        add_break_marks(param_left, param_right)
        add_break_marks(flops_left, flops_right)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    print(f"Saved {args.output} and {args.output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
