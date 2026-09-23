#!/usr/bin/env python3
"""Plot mAP--parameter and mAP--FLOPs trade-off scatter plots from a CSV file."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from adjustText import adjust_text
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


def place_labels(ax, labels: list[tuple[str, float, float, bool]], figure) -> None:
    """Automatically repel labels and draw leader arrows to their points."""
    if not labels:
        return
    texts = [ax.text(x, y, name, fontsize=8.5,
                     fontweight="bold" if highlight else "normal",
                     color="#111111", zorder=5)
             for name, x, y, highlight in labels]
    adjust_text(
        texts,
        x=[x for _, x, _, _ in labels],
        y=[y for _, _, y, _ in labels],
        ax=ax,
        force_text=(0.35, 0.65), force_static=(0.45, 0.8),
        force_pull=(0.02, 0.03), expand=(1.15, 1.25),
        max_move=(12, 14), iter_lim=500,
        ensure_inside_axes=True, prevent_crossings=True,
        arrowprops=dict(arrowstyle="->", color="#666666", lw=0.65,
                       shrinkA=3, shrinkB=5),
    )


def draw_panel(ax, rows: list[dict[str, str]], x_key: str, x_label: str, panel: str,
               no_text: bool, xlim: tuple[float, float] | None = None,
               ticks: tuple[float, ...] | None = None, show_y: bool = True,
               soft_compress: bool = False) -> list[tuple[str, float, float, bool]]:
    colors = plt.get_cmap("tab10").colors
    labels = []
    for index, row in enumerate(rows):
        x, y = float(row[x_key]), float(row["map50_95"])
        if xlim is not None and not xlim[0] <= x <= xlim[1]:
            continue
        highlight = row["highlight"].lower() == "true"
        color = "#d62728" if highlight else colors[index % len(colors)]
        display_x = math.sqrt(math.log10(max(x, 1))) if soft_compress else x
        ax.scatter(display_x, y, s=185 if highlight else 150, c=[color], edgecolors="black",
                   linewidths=0.8, alpha=0.72, zorder=3)
        labels.append((row["model"], display_x, y, highlight))
    if not soft_compress:
        ax.set_xscale("log")
    if soft_compress:
        values = [float(row[x_key]) for row in rows]
        minimum, maximum = min(values), max(values)
        compress = lambda value: math.sqrt(math.log10(max(value, 1)))
        ax.set_xlim(compress(minimum) - 0.03, compress(maximum) + 0.22)
        candidate_ticks = ((2.5, 5, 10, 20, 40) if x_key == "params_m"
                           else (5, 10, 20, 50, 100, 200))
        visible_ticks = [value for value in candidate_ticks
                         if minimum * 0.94 <= value <= maximum * 1.06]
        ax.set_xticks([compress(value) for value in visible_ticks])
        ax.set_xticklabels([f"{value:g}" for value in visible_ticks])
    elif xlim is None:
        if x_key == "params_m":
            xlim, ticks = (1.2, 50), (2, 5, 10, 20, 50)
        else:
            xlim, ticks = (3, 250), (5, 10, 20, 50, 100, 200)
    if not soft_compress:
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
    return [] if no_text else labels


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
        figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
        labels_left = draw_panel(axes[0], rows, "params_m", "Parameters (M)",
                   "(a) Performance-Efficiency Trade-off (mAP vs Parameters)", args.no_text,
                   soft_compress=args.soft_compress)
        labels_right = draw_panel(axes[1], rows, "flops_g", "FLOPs (G)",
                   "(b) Performance-Efficiency Trade-off (mAP vs FLOPs)", args.no_text,
                   soft_compress=args.soft_compress)
        if not args.no_text:
            figure.canvas.draw()
            figure.set_layout_engine(None)
            place_labels(axes[0], labels_left, figure)
            place_labels(axes[1], labels_right, figure)
    else:
        figure = plt.figure(figsize=(10.5, 4.1), constrained_layout=True)
        grid = figure.add_gridspec(1, 5, width_ratios=(1.05, 1.3, 0.18, 1.05, 1.3))
        param_left = figure.add_subplot(grid[0, 0])
        param_right = figure.add_subplot(grid[0, 1], sharey=param_left)
        flops_left = figure.add_subplot(grid[0, 3])
        flops_right = figure.add_subplot(grid[0, 4], sharey=flops_left)
        label_sets = [
            (param_left, draw_panel(param_left, rows, "params_m", "", "", args.no_text, (1.2, 6), (2, 5))),
            (param_right, draw_panel(param_right, rows, "params_m", "", "", args.no_text, (18, 50), (20, 50), False)),
            (flops_left, draw_panel(flops_left, rows, "flops_g", "", "", args.no_text, (3, 12), (5, 10))),
            (flops_right, draw_panel(flops_right, rows, "flops_g", "", "", args.no_text, (35, 230), (50, 100, 200), False)),
        ]
        add_break_marks(param_left, param_right)
        add_break_marks(flops_left, flops_right)
        if not args.no_text:
            for axis, panel_labels in label_sets:
                place_labels(axis, panel_labels, figure)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    print(f"Saved {args.output} and {args.output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
