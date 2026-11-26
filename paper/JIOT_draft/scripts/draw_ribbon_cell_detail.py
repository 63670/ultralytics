"""Draw the detector and its two modules for the JIOT manuscript."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "figures"
INK = "#203348"
BLUE, GREEN, AMBER, PURPLE = "#E6EFF8", "#E6F2EC", "#FBEBD9", "#EFE8F7"
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "pdf.fonttype": 42,
                     "svg.fonttype": "none"})


def box(ax, x, y, w, h, label, color=BLUE, size=15):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.025,rounding_size=0.08",
                               linewidth=1.25, edgecolor="#788DA3", facecolor=color))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", color=INK, fontsize=size)


def arrow(ax, *points, dashed=False):
    for i, (start, end) in enumerate(zip(points, points[1:])):
        if i == len(points) - 2:
            ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                                         color=INK, linewidth=1.4, shrinkA=1.0, shrinkB=1.0,
                                         linestyle="--" if dashed else "-"))
        else:
            ax.plot((start[0], end[0]), (start[1], end[1]), color=INK, lw=1.4, ls="--" if dashed else "-")


def panel(ax, title, height):
    ax.set(xlim=(0, 14), ylim=(0, height))
    ax.axis("off")
    ax.text(0.18, height - 0.20, title, ha="left", va="top", fontsize=19, fontweight="bold", color=INK)


def architecture(ax):
    panel(ax, "(a) Detector architecture", 4.15)
    stages = ((0.25, "Stem and P2", BLUE), (3.05, "P3 C2f +\nribbon attention", GREEN),
              (5.85, "P4 C2f +\nribbon attention", GREEN), (8.65, "P5 C2f + SPPF", BLUE))
    for x, label, color in stages:
        box(ax, x, 2.60, 2.26, 0.80, label, color)
    for x, _, _ in stages[:-1]:
        arrow(ax, (x + 2.26, 3), (x + 2.80, 3))

    box(ax, 8.65, 1.15, 2.26, 0.82, "Cell-detail\nP5 to P4", AMBER)
    box(ax, 5.85, 1.15, 2.26, 0.82, "Fuse P4\nand refine", BLUE)
    box(ax, 3.05, 1.15, 2.26, 0.82, "Cell-detail\nP4 to P3", AMBER)
    box(ax, 0.25, 1.15, 2.26, 0.82, "Fuse P3\nand refine", BLUE)
    box(ax, 11.45, 1.15, 2.26, 0.82, "Bottom-up PAN\nDetect P3/P4/P5", PURPLE)
    arrow(ax, (9.78, 2.60), (9.78, 1.97))
    for x in (8.65, 5.85, 3.05):
        arrow(ax, (x, 1.56), (x - 0.54, 1.56))
    arrow(ax, (4.18, 2.60), (4.18, 1.97), dashed=True)
    arrow(ax, (4.18, 2.30), (1.38, 2.30), (1.38, 1.97), dashed=True)
    arrow(ax, (6.98, 2.60), (6.98, 1.97), dashed=True)
    arrow(ax, (6.98, 2.30), (9.19, 2.30), (9.19, 1.97), dashed=True)
    arrow(ax, (1.38, 1.15), (1.38, 0.63), (12.58, 0.63), (12.58, 1.15))
    ax.text(6.98, 0.28, "The bottom-up path uses the fused pyramid features. Dashed lines carry backbone features.",
            color=INK, fontsize=13.5, ha="center", va="center")


def attention(ax):
    panel(ax, "(b) Interleaved ribbon attention", 2.60)
    steps = (
        (0.25, 1.62, "C2f output", BLUE), (2.32, 2.05, "Q, K, V projection\nand depthwise mix", BLUE),
        (4.82, 2.00, "Horizontal and\nvertical groups", GREEN),
        (7.27, 1.86, "Attention\nwithin groups", GREEN),
        (9.58, 1.78, "Restore the\nfeature grid", GREEN), (11.81, 1.91, "Project and\nadd to input", BLUE),
    )
    for x, width, label, color in steps:
        box(ax, x, 0.91, width, 0.88, label, color, size=14)
    for (x, width, _, _), (next_x, _, _, _) in zip(steps, steps[1:]):
        arrow(ax, (x + width, 1.35), (next_x, 1.35))
    ax.text(7, 0.47, "Group membership is set by a two-position band and an eight-position sampling step.",
            color=INK, fontsize=14, ha="center", va="center")


def upsampling(ax):
    panel(ax, "(c) Mean-preserving cell-detail upsampling", 3.64)
    box(ax, 0.25, 2.27, 1.62, 0.70, "Coarse $X$")
    box(ax, 0.25, 0.84, 1.62, 0.70, "Lateral $L$", GREEN)
    box(ax, 3.00, 2.27, 2.15, 0.70, "Nearest-neighbor\nbase", BLUE, size=14)
    box(ax, 2.38, 0.84, 2.18, 0.70, "Project lateral\nfeatures", GREEN, size=14)
    box(ax, 5.07, 0.84, 2.34, 0.70, "Subtract each\ncell mean", AMBER, size=14)
    box(ax, 5.75, 1.88, 2.26, 0.78, "Cell gate from $X$\nand lateral mean", PURPLE, size=13.5)
    box(ax, 8.52, 0.84, 1.82, 0.70, "Scale detail\nby cell gate", AMBER, size=14)
    box(ax, 10.85, 0.84, 1.54, 0.70, "Project\ndetail", AMBER, size=14)
    box(ax, 12.90, 1.65, 0.82, 0.94, "Add", BLUE, size=14)
    arrow(ax, (1.87, 2.62), (3.00, 2.62))
    arrow(ax, (5.15, 2.62), (5.40, 2.62), (5.40, 2.93), (13.31, 2.93), (13.31, 2.59))
    arrow(ax, (2.13, 2.62), (2.13, 1.96), (5.75, 1.96))
    arrow(ax, (1.87, 1.19), (2.38, 1.19))
    arrow(ax, (4.56, 1.19), (5.07, 1.19))
    arrow(ax, (5.60, 1.54), (5.60, 1.65), (6.27, 1.65), (6.27, 1.88))
    arrow(ax, (7.41, 1.19), (8.52, 1.19))
    arrow(ax, (8.01, 2.27), (9.43, 2.27), (9.43, 1.54))
    arrow(ax, (10.34, 1.19), (10.85, 1.19))
    arrow(ax, (12.39, 1.19), (13.31, 1.19), (13.31, 1.65))
    ax.text(7, 0.39, "Each gate is shared by the four spatial positions of its cell.", color=INK,
            fontsize=14, ha="center", va="center")


if __name__ == "__main__":
    fig, axes = plt.subplots(3, 1, figsize=(14, 8.6), gridspec_kw={"height_ratios": (4.15, 2.60, 3.64)})
    architecture(axes[0])
    attention(axes[1])
    upsampling(axes[2])
    fig.subplots_adjust(left=0.008, right=0.992, top=0.992, bottom=0.012, hspace=0.09)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"ribbon-cell-detail.{suffix}", dpi=220)
    plt.close(fig)
