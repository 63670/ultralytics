"""Plot every YOLO annotation at a common center, retaining normalized width and height."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


NAMES = ("row", "col", "hole")
COLORS = ("#D55E00", "#0072B2", "#009E73")


def read_annotations(root: Path):
    files = sorted(root.rglob("*.txt"))
    if not files:
        raise ValueError(f"No annotation files found in {root}")
    digest = hashlib.sha256()
    boxes, splits = [], Counter()
    for path in files:
        raw = path.read_bytes()
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + raw + b"\0")
        for line_number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            values = np.asarray([float(v) for v in line.split()])
            if len(values) != 5 or not np.isfinite(values).all():
                raise ValueError(f"Invalid detection annotation: {path}:{line_number}")
            cls, x, y, w, h = values
            if cls not in range(len(NAMES)) or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError(f"Invalid class or normalized box: {path}:{line_number}")
            boxes.append((int(cls), w, h))
            splits[path.parent.name] += 1
    if not boxes:
        raise ValueError("Annotations contain no boxes")
    return np.asarray(boxes), len(files), dict(splits), digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path, help="Label root, e.g. labels containing train/val/test")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    boxes, image_count, splits, digest = read_annotations(args.labels)
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})
    fig, axes = plt.subplots(1, 4, figsize=(7.3, 2.35), sharex=True, sharey=True)
    for panel, ax in enumerate(axes):
        selected = boxes if panel == 0 else boxes[boxes[:, 0] == panel - 1]
        for cls, w, h in sorted(selected, key=lambda box: box[1] * box[2], reverse=True):
            ax.add_patch(Rectangle(((1 - w) / 2, (1 - h) / 2), w, h,
                                   fill=False, edgecolor=COLORS[int(cls)], linewidth=0.35, alpha=0.22))
        title = "All defects" if panel == 0 else NAMES[panel - 1].capitalize()
        ax.set_title(f"({chr(97 + panel)}) {title} (n={len(selected)})", pad=5)
        ax.set(xlim=(0, 1), ylim=(1, 0), aspect="equal", xlabel="Normalized x")
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(length=2, pad=2, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#777777")
            spine.set_linewidth(0.5)
    axes[0].set_ylabel("Normalized y")
    fig.legend([Line2D([0], [0], color=c, lw=1.5) for c in COLORS], NAMES,
               loc="lower center", bbox_to_anchor=(0.5, 0), ncol=3, frameon=False)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.88, bottom=0.23, wspace=0.16)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    stats = {"label_root": str(args.labels.resolve()), "annotation_sha256": digest,
             "images": image_count, "instances": len(boxes), "split_instances": splits,
             "geometry": "All boxes centered at (0.5, 0.5), preserving normalized width and height.",
             "classes": {}}
    for cls, name in enumerate(NAMES):
        wh = boxes[boxes[:, 0] == cls, 1:]
        stats["classes"][name] = {
            "instances": len(wh), "median_width_height_ratio": float(np.median(wh[:, 0] / wh[:, 1])),
            "box_area_percentiles_percent": dict(zip(("p10", "p50", "p90"),
                                                        np.percentile(100 * wh[:, 0] * wh[:, 1], [10, 50, 90]).tolist()))}
    args.output.with_suffix(".stats.json").write_bytes((json.dumps(stats, indent=2) + "\n").encode("utf-8"))
    print(f"Saved {args.output}: {image_count} images, {len(boxes)} boxes; "
          + ", ".join(f"{name}={stats['classes'][name]['instances']}" for name in NAMES))


if __name__ == "__main__":
    main()
