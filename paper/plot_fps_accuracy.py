"""Draw FPS versus accuracy directly from the manuscript's detector-comparison table."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PAPER = Path(__file__).resolve().parent / "TIE_submission"
MANUSCRIPT = PAPER / "A Defect Detection and Closed-Loop Control System with MCGI-YOLO for Circular Knitting Machines.tex"
# Tableau categorical colors, extended with a dark indigo for the eleventh model.
MODEL_COLORS = {
    "YOLOv9t": "#1F77B4",
    "YOLOv10n": "#BCBD22",
    "YOLO11n": "#2CA02C",
    "YOLO12n": "#9467BD",
    "YOLO26n": "#17BECF",
    "RT-DETR-R18": "#8C564B",
    "DETR-R50": "#E377C2",
    "Deformable DETR-R50": "#7F7F7F",
    "Faster R-CNN-R50": "#393B79",
    "YOLOv8n (Baseline)": "#FF7F0E",
    "MCGI-YOLO": "#D62728",
}
OFFSETS = (
    {"YOLOv9t": (-8, 15), "YOLOv10n": (-4, -20), "YOLO11n": (12, -14),
     "YOLO12n": (8, -16), "YOLO26n": (-8, 15), "RT-DETR-R18": (40, 0),
     "DETR-R50": (48, 6), "Deformable DETR-R50": (8, 21),
     "Faster R-CNN-R50": (10, -48), "YOLOv8n (Baseline)": (-5, 12), "MCGI-YOLO": (-8, 18)},
    {"YOLOv9t": (-8, 15), "YOLOv10n": (-38, 19), "YOLO11n": (12, -17),
     "YOLO12n": (-8, -18), "YOLO26n": (-8, 16), "RT-DETR-R18": (8, 15),
     "DETR-R50": (10, 12), "Deformable DETR-R50": (20, -28),
     "Faster R-CNN-R50": (28, 10), "YOLOv8n (Baseline)": (0, -23), "MCGI-YOLO": (8, 15)},
)


def read_comparison(path: Path):
    source = path.read_text(encoding="utf-8")
    table = source.split(r"\label{tab:comparison}", 1)[1].split(r"\bottomrule", 1)[0]
    rows = []
    for line in table.splitlines():
        cells = line.split("&")
        if len(cells) != 8 or cells[0].strip() == "Model":
            continue
        name = re.sub(r"~?\\cite\{[^}]+\}", "", cells[0]).strip().replace(r"\model", "MCGI-YOLO")
        values = [float(re.search(r"\d+(?:\.\d+)?", cell).group()) for cell in cells[1:]]
        rows.append({"model": name, "fps": values[-1], "map50": values[-3], "map50_95": values[-2]})
    if not rows or len({row["model"] for row in rows}) != len(rows):
        raise ValueError("Missing comparison rows or duplicate model names")
    if any(row["fps"] <= 0 or not (0 <= row["map50_95"] <= row["map50"] <= 100) for row in rows):
        raise ValueError("Invalid FPS or mAP values in comparison table")
    return rows


def draw_panel(ax, rows, metric, panel):
    for row in rows:
        name = row["model"]
        ours = name == "MCGI-YOLO"
        baseline = "Baseline" in name
        color = MODEL_COLORS[name]
        ax.scatter(row["fps"], row[metric], s=320 if ours else 150,
                   marker="*" if ours else "D" if baseline else "o", c=color,
                   edgecolor="white", linewidth=1.0, zorder=4)
        dx, dy = OFFSETS[panel].get(name, (8, 10))
        label = name.replace(" (Baseline)", "\n(Baseline)")
        label = label.replace("Deformable DETR-R50", "Deformable\nDETR-R50").replace("Faster R-CNN-R50", "Faster R-CNN\n(R50)")
        ax.annotate(label, (row["fps"], row[metric]), xytext=(dx, dy), textcoords="offset points",
                    ha="right" if dx < 0 else "left", va="center", fontsize=9,
                    fontweight="bold" if ours else "normal", color=color if ours or baseline else "#333333",
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5, "alpha": 0.9},
                    arrowprops={"arrowstyle": "-", "color": color, "lw": 0.55, "alpha": 0.65}, zorder=5)
    ax.set_xlim(0, 650)
    ax.set_ylim((87, 99.2) if panel == 0 else (51, 70))
    ax.set_xticks(range(0, 651, 100))
    ax.set_xlabel("Inference throughput (FPS)")
    ax.set_ylabel("mAP@50 (%)" if panel == 0 else "mAP@50:95 (%)")
    ax.set_title("(a) FPS vs. mAP@50" if panel == 0 else "(b) FPS vs. mAP@50:95", pad=12)
    ax.grid(color="#E4E7EB", linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#777777")
    ax.tick_params(color="#777777")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript", type=Path, default=MANUSCRIPT)
    parser.add_argument("--output", type=Path, default=PAPER / "figures" / "fps_accuracy")
    args = parser.parse_args()
    rows = read_comparison(args.manuscript)
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.labelsize": 11,
                         "axes.titlesize": 12, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    for panel, metric in enumerate(("map50", "map50_95")):
        draw_panel(axes[panel], rows, metric, panel)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.13, top=0.9, wspace=0.22)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(args.output.with_suffix("." + extension), dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    provenance = {"manuscript": str(args.manuscript.resolve()),
                  "manuscript_sha256": hashlib.sha256(args.manuscript.read_bytes()).hexdigest(),
                  "source_table": "tab:comparison", "platform": "RTX 3090", "precision": "FP32",
                  "batch_size": 1, "input_size": [640, 640], "fps_scope": "model forward only",
                  "accuracy_unit": "percent", "models": rows}
    args.output.with_suffix(".json").write_bytes((json.dumps(provenance, indent=2) + "\n").encode())
    print(json.dumps(rows, indent=2))
    print(f"Saved {args.output}.png and {args.output}.pdf")


if __name__ == "__main__":
    main()
