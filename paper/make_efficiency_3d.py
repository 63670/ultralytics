"""Render a compact 3-D AP/Params/FLOPs figure from retained results.

Params and FLOPs use explicit log10 coordinate transforms and original-unit
tick labels. AP is linear. Only annotation positions are offset for clarity.
No existing manuscript or hand-edited figure is overwritten.
"""

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "tools" / "efficiency_results.csv"
STYLE = {
    "YOLOv5n": ("#4E79A7", "YOLOv5n", (-35, -12)),
    "YOLOv8n": ("#F28E2B", "YOLOv8n", (32, -9)),
    "YOLO11n": ("#59A14F", "YOLO11n", (-35, 2)),
    "YOLO12n": ("#B07AA1", "YOLO12n", (32, 5)),
    "YOLO26n": ("#369A9D", "YOLO26n", (-35, 17)),
    "RT-DETR-R18": ("#9C755F", "RT-DETR-R18", (16, 12)),
    "DETR-R50": ("#B29B27", "DETR-R50", (-30, -4)),
    "Deformable-DETR": ("#777777", "Deformable\nDETR-R50", (-32, -21)),
    "Faster-RCNN": ("#B56F9C", "Faster R-CNN-R50", (-24, 40)),
    "Ours": ("#D43D3D", "DMFR-YOLO", (-5, 23)),
}


def main():
    with SOURCE.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(STYLE) and {row["model"] for row in rows} == set(STYLE)
    points = []
    for row in rows:
        params, flops, ap = (float(row[key]) for key in ("params_m", "flops_g", "map50_95"))
        assert all(math.isfinite(value) and value > 0 for value in (params, flops, ap))
        assert ap <= 100
        points.append((row, math.log10(params), math.log10(flops), ap))

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8, "svg.fonttype": "none", "axes.linewidth": 0.65,
    })
    fig = plt.figure(figsize=(3.5, 3.15))
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.12)
    ax.set_box_aspect((1.1, 1.1, 1.05))
    ax.view_init(elev=24, azim=-58)
    ax.set_proj_type("ortho")
    ax.set_xlim(math.log10(1.8), math.log10(60))
    ax.set_ylim(math.log10(4), math.log10(240))
    ax.set_zlim(51, 70)
    for axis, values in ((ax.xaxis, [2, 5, 10, 20, 50]),
                         (ax.yaxis, [5, 10, 50, 100, 200])):
        axis.set_ticks([math.log10(value) for value in values])
        axis.set_ticklabels([str(value) for value in values])
        axis.pane.fill = False
        axis.pane.set_edgecolor("#DDDDDD")
    ax.zaxis.pane.fill = False
    ax.zaxis.pane.set_edgecolor("#DDDDDD")
    ax.set_zticks([52, 56, 60, 64, 68])
    ax.set_xlabel("Params(M)", labelpad=0)
    ax.set_ylabel("FLOPs(G)", labelpad=0)
    # A 2-D label avoids mplot3d clipping its outward-facing z-axis title.
    ax.text2D(1.10, 0.67, "AP (%)", transform=ax.transAxes,
              rotation=90, ha="center", va="center", fontsize=8)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"].update({"color": "#DADDE2", "linewidth": 0.45})
    ax.tick_params(labelsize=7, pad=0)

    for row, x, y, z in points:
        color, _, _ = STYLE[row["model"]]
        ours = row["highlight"].lower() == "true"
        ax.scatter(x, y, z, s=27, c=color, edgecolors="#222222",
                   linewidths=0.8 if ours else 0.35, depthshade=False, zorder=4)
        # A subtle guide anchors the AP height without adding a fourth encoding.
        ax.plot([x, x], [y, y], [51, z], color=color, linestyle=":",
                linewidth=0.5, alpha=0.4, zorder=2)

    fig.canvas.draw()
    for row, x, y, z in points:
        color, label, offset = STYLE[row["model"]]
        px, py, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
        ours = row["highlight"].lower() == "true"
        ax.annotate(label, xy=(px, py), xytext=offset, textcoords="offset points",
                    ha="center", va="center", fontsize=6.8,
                    fontweight="bold" if ours else "normal",
                    color=color if ours else "#222222", zorder=10,
                    annotation_clip=False,
                    arrowprops={"arrowstyle": "-", "lw": 0.5, "color": color,
                                "shrinkA": 2, "shrinkB": 3})
    fig.text(0.5, 0.025, "Params and FLOPs use logarithmic scales.",
             ha="center", fontsize=7, color="#444444")

    output = ROOT / "figures" / "efficiency_3d"
    for suffix in ("svg", "png"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=600, facecolor="white",
                    bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Validated {len(points)} models; Params/FLOPs log10, AP linear.")
    print(f"Saved {output.with_suffix('.svg')} and {output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
