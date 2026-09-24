"""Rebuild the manuscript's scientific plot from the retained experiment record.

No data coordinates are shifted or compressed; both x axes are logarithmic.
Run from any directory. Outputs stay in paper/figures/.
"""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "tools" / "efficiency_results.csv"


def main():
    with SOURCE.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "axes.linewidth": 0.7})
    colors = ["#4E79A7", "#F28E2B", "#59A14F", "#B07AA1", "#76B7B2",
              "#9C755F", "#EDC948", "#888888", "#FF9DA7", "#E15759"]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.95), sharey=True)
    for ax, key, label, ticks, limits in zip(
        axes, ["params_m", "flops_g"], ["Param.(M)", "FLOPs(G)"],
        [[2, 5, 10, 20, 50], [5, 10, 20, 50, 100, 200]], [(1.8, 56), (3.9, 235)]
    ):
        for i, row in enumerate(rows):
            ours = row["highlight"].lower() == "true"
            ax.scatter(float(row[key]), float(row["map50_95"]), s=78 if ours else 47,
                       color=colors[i], marker="*" if ours else "o", alpha=.8,
                       edgecolor="#333333", linewidth=.55, zorder=3,
                       label=row["model"].replace("YOLOv5n", "YOLOv5n*") )
        ax.set_xscale("log")
        ax.set_xlim(*limits)
        ax.set_ylim(51, 70)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel(label)
        ax.grid(color="#dddddd", linewidth=.5, zorder=0)
    axes[0].set_ylabel("AP (%)")
    axes[0].set_title("(a) Parameter--accuracy trade-off", fontsize=9)
    axes[1].set_title("(b) Operation--accuracy trade-off", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               fontsize=6.7, columnspacing=1, handletextpad=.3)
    fig.subplots_adjust(left=.075, right=.99, top=.9, bottom=.27, wspace=.1)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(ROOT / "figures" / f"tradeoff.{suffix}", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
