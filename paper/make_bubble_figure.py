"""Create a single-column AP/Params/FLOPs figure without changing source data.

Bubble area (not radius) is proportional to FLOPs. Label positions are moved
for readability; all data coordinates remain those in efficiency_results.csv.
The existing hand-edited efficiency.png and manuscript are left untouched.
"""

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "tools" / "efficiency_results.csv"
AREA_PER_GFLOP = 5.5  # Matplotlib scatter area in points squared.
STYLE = {
    "YOLOv5n": ("#4E79A7", "YOLOv5n", (4.7, 57.0)),
    "YOLOv8n": ("#F28E2B", "YOLOv8n", (5.2, 58.8)),
    "YOLO11n": ("#59A14F", "YOLO11n", (4.7, 60.1)),
    "YOLO12n": ("#B07AA1", "YOLO12n", (4.7, 61.8)),
    "YOLO26n": ("#369A9D", "YOLO26n", (2.7, 63.3)),
    "RT-DETR-R18": ("#9C755F", "RT-DETR-R18", (14.0, 69.0)),
    "DETR-R50": ("#C3A52F", "DETR-R50", (18.5, 53.6)),
    "Deformable-DETR": ("#777777", "Deformable\nDETR-R50", (17.0, 61.2)),
    "Faster-RCNN": ("#B56F9C", "Faster R-CNN-R50", (16.0, 65.1)),
    "Ours": ("#D43D3D", "DMFR-YOLO", (3.39, 68.6)),
}


def main():
    with SOURCE.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 10 and len({row["model"] for row in rows}) == 10
    for row in rows:
        assert row["model"] in STYLE
        values = [float(row[key]) for key in ("params_m", "flops_g", "map50_95")]
        assert all(math.isfinite(value) and value > 0 for value in values)
        assert values[2] <= 100

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8, "svg.fonttype": "none", "axes.linewidth": 0.7,
    })
    fig, ax = plt.subplots(figsize=(3.5, 3.25))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.97, bottom=0.25)
    ax.set_xscale("log")
    ax.set_xlim(1.85, 57)
    ax.set_ylim(51, 70.5)
    ax.xaxis.set_major_locator(FixedLocator([2, 3, 5, 10, 20, 50]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([52, 56, 60, 64, 68])
    ax.set_xlabel("Params(M)", labelpad=3)
    ax.set_ylabel("AP (%)", labelpad=3)
    ax.tick_params(length=3, width=0.6, pad=3)
    ax.set_axisbelow(True)
    ax.grid(color="#E4E4E4", linewidth=0.5, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    # Largest bubbles first, allowing smaller marks to remain visible.
    for row in sorted(rows, key=lambda item: float(item["flops_g"]), reverse=True):
        color, label, label_position = STYLE[row["model"]]
        x, y, flops = (float(row[key]) for key in ("params_m", "map50_95", "flops_g"))
        ours = row["highlight"].lower() == "true"
        ax.scatter(x, y, s=AREA_PER_GFLOP * flops, facecolor=color,
                   edgecolor=color, alpha=0.55, linewidth=0.9, zorder=3)
        # The center dot locates overlapping observations precisely.
        ax.plot(x, y, ".", color=color, markersize=2.3, zorder=4)
        ax.annotate(label, xy=(x, y), xytext=label_position,
                    ha="center", va="center", fontsize=7.2,
                    fontweight="bold" if ours else "normal",
                    color=color if ours else "#222222", zorder=5,
                    arrowprops={"arrowstyle": "-", "color": color,
                                "lw": 0.55, "shrinkA": 2, "shrinkB": 3})

    # Use the same area mapping in the legend, with no min-size inflation.
    legend_values = [10, 60, 180]
    legend_ax = fig.add_axes([0.12, 0.017, 0.85, 0.14])
    legend_ax.set(xlim=(0, 1), ylim=(0, 1))
    legend_ax.set_axis_off()
    legend_ax.text(0, 0.5, "FLOPs (G)", fontsize=7.3, va="center")
    for value, x, text_x in zip(legend_values, [0.35, 0.56, 0.79], [0.40, 0.65, 0.90]):
        legend_ax.scatter(x, 0.5, s=AREA_PER_GFLOP * value, facecolor="#C8CED5",
                          edgecolor="#718096", alpha=0.55, linewidth=0.7,
                          clip_on=False)
        legend_ax.text(text_x, 0.5, str(value), fontsize=7.3, va="center")
    output = ROOT / "figures" / "efficiency_bubble"
    for suffix in ("svg", "png"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=600, facecolor="white")
    plt.close(fig)
    print(f"Validated and plotted {len(rows)} models; bubble area = {AREA_PER_GFLOP} * FLOPs.")
    print(f"Saved {output.with_suffix('.svg')} and {output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
