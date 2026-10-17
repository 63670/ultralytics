"""Draw implementation-matched, editable diagrams for the axis/interpolation candidates.

Run: python paper/draw_axis_gate_modules.py
Requires matplotlib. Outputs SVG (editable text) and PNG to paper/figures/.
These are candidate-model illustrations; this script does not modify manuscript.tex.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figures"
INK = "#243746"
MUTED = "#566876"
BLUE = "#E8F1FA"
TEAL = "#E5F3ED"
GOLD = "#FFF1D9"
LILAC = "#F0EBF8"
GRAY = "#F2F4F6"


class Diagram:
    def __init__(self, width, height, title, subtitle):
        self.fig, self.ax = plt.subplots(figsize=(width / 120, height / 120), dpi=150)
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.ax.set(xlim=(0, width), ylim=(height, 0))
        self.ax.axis("off")
        self.width, self.height = width, height
        self.labels = []
        self.text(40, 48, title, 22, weight="bold", ha="left")
        self.text(40, 91, subtitle, 12, color=MUTED, ha="left")

    def text(self, x, y, text, size=12, color=INK, ha="center", weight="normal"):
        item = self.ax.text(x, y, text, fontsize=size, color=color, ha=ha, va="center",
                            fontweight=weight, linespacing=1.45, zorder=5)
        self.labels.append(item)
        return item

    def box(self, x, y, w, h, text, color=BLUE, size=12):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=9",
                                        facecolor=color, edgecolor=INK, linewidth=1.15, zorder=3))
        self.text(x + w / 2, y + h / 2, text, size)

    def panel(self, x, y, w, h, title):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=14",
                                        facecolor="white", edgecolor="#CCD5DD", linewidth=1, zorder=0))
        self.text(x + 22, y + 28, title, 14, ha="left", weight="bold")

    def line(self, points, arrow=True, dashed=False, color=INK):
        xs, ys = zip(*points)
        self.ax.plot(xs, ys, color=color, linewidth=1.35, linestyle="--" if dashed else "-", zorder=1)
        if arrow:
            self.ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                             mutation_scale=13, linewidth=1.35, color=color, zorder=2))

    def dot(self, x, y):
        self.ax.add_patch(Circle((x, y), 3.2, color=INK, zorder=4))

    def op(self, x, y, text, color=GRAY, radius=27):
        self.ax.add_patch(Circle((x, y), radius, facecolor=color, edgecolor=INK,
                                 linewidth=1.25, zorder=3))
        self.text(x, y, text, 21)

    def save(self, stem):
        # Check that labels fit within the canvas before writing publication assets.
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        extent = self.fig.bbox
        for item in self.labels:
            bounds = item.get_window_extent(renderer)
            assert bounds.x0 >= extent.x0 and bounds.x1 <= extent.x1, item.get_text()
            assert bounds.y0 >= extent.y0 and bounds.y1 <= extent.y1, item.get_text()
        OUT.mkdir(parents=True, exist_ok=True)
        for extension in ("svg", "png"):
            path = OUT / f"{stem}.{extension}"
            kwargs = {"metadata": {"Date": None}} if extension == "svg" else {}
            self.fig.savefig(path, dpi=180, facecolor="white", **kwargs)
            if extension == "svg":
                path.write_bytes(b"\n".join(line.rstrip() for line in path.read_bytes().splitlines()) + b"\n")
        plt.close(self.fig)


def c2f_location(d, bounded):
    d.panel(40, 130, 1720, 285, "A   Placement inside C2f (nano configuration: n = 2)")
    d.box(65, 245, 125, 60, "Input X", GRAY)
    d.box(230, 245, 150, 60, "1 × 1 Conv")
    d.box(420, 245, 130, 60, "Split\n(c, c)", GRAY)
    d.box(600, 245, 140, 60, "Bottleneck 1")
    d.box(790, 245, 140, 60, "Bottleneck 2")
    d.box(990, 245, 200, 60, "Partial axial\nmodulation", TEAL)
    d.box(1260, 210, 170, 130, "Concat\n[z₀, z₁, z₂, z′₃]", LILAC)
    d.box(1480, 245, 140, 60, "1 × 1 Conv")
    d.text(1690, 275, "Y", 16, weight="bold")
    for a, b in ((190, 230), (380, 420), (550, 600), (740, 790), (930, 990), (1190, 1260)):
        d.line([(a, 275), (b, 275)])
    d.line([(1430, 275), (1480, 275)])
    d.line([(1620, 275), (1660, 275)])
    d.line([(485, 245), (485, 195), (1345, 195), (1345, 210)])
    d.text(880, 183, "z₀ (unchanged)", 10)
    d.dot(570, 275)
    d.line([(570, 275), (570, 365), (1230, 365), (1230, 320), (1260, 320)])
    d.text(870, 354, "z₁ (unchanged)", 10)
    d.dot(765, 275)
    d.line([(765, 275), (765, 330), (1210, 330), (1210, 300), (1260, 300)])
    d.text(970, 320, "z₂ (unchanged)", 10)
    d.text(960, 257, "F", 12)
    d.text(1225, 258, "F′", 12)
    name = "C2fAxisBounded" if bounded else "C2fAxisResidual"
    d.text(65, 392, f"{name}: only z₃ is modified; all preceding C2f branches are retained.",
           11, ha="left", color=MUTED)


def axis_diagram(bounded=False):
    name = "C2fAxisBounded" if bounded else "C2fAxisResidual"
    d = Diagram(1800, 1140, f"{'Bounded axial modulation' if bounded else 'Residual axial context'}  |  {name}",
                "Backbone P3 and P4  •  Partial-channel long-range context  •  Batch dimension omitted")
    c2f_location(d, bounded)
    d.panel(40, 440, 1720, 600, "B   Detail of the partial axial modulation block")
    d.box(80, 690, 150, 80, "F\nc × H × W", GRAY)
    d.box(270, 690, 130, 80, "Channel\nsplit", GRAY)
    d.box(450, 540, 170, 60, "Fₐ: q × H × W", TEAL)
    d.line([(230, 730), (270, 730)])
    d.line([(335, 690), (335, 570), (450, 570)])
    d.text(65, 518, "c = C_out / 2", 11, ha="left", color=MUTED)
    d.text(65, 545, "q = max(1, floor(c / 4))", 11, ha="left", color=MUTED)
    d.box(690, 500, 180, 72, "Mean over W\nR: q × H × 1", BLUE)
    d.box(690, 710, 180, 72, "Mean over H\nC: q × 1 × W", BLUE)
    row_text = "DWConv 3 × 1\n(R − M)" if bounded else "DWConv 3 × 1\n(R)"
    col_text = "DWConv 1 × 3\n(C − M)" if bounded else "DWConv 1 × 3\n(C)"
    d.box(950, 500, 180, 72, row_text, TEAL)
    d.box(950, 710, 180, 72, col_text, TEAL)
    d.line([(620, 570), (650, 570), (650, 536), (690, 536)])
    d.line([(650, 570), (650, 746), (690, 746)])
    d.dot(650, 570)
    d.line([(870, 536), (950, 536)])
    d.line([(870, 746), (950, 746)])
    d.op(1220, 660, "+", TEAL, radius=30)
    d.line([(1130, 536), (1220, 536), (1220, 630)])
    d.line([(1130, 746), (1220, 746), (1220, 690)])
    d.text(1170, 608, "Broadcast", 10, color=MUTED)
    d.text(1170, 625, "to H × W", 10, color=MUTED)
    d.box(1300, 625, 160, 70, "g =\nHardSigmoid", GOLD, size=11)
    d.line([(1250, 660), (1300, 660)])
    d.box(1500, 625, 200, 70,
          "S = 1 + 0.25 ×\n(2g − 1)" if bounded else "A = 2g − 1", GOLD, size=11)
    d.line([(1460, 660), (1500, 660)])
    if bounded:
        d.box(690, 610, 280, 60, "Shared center\nM = Mean over H,W (Fₐ)", GRAY, size=11)
        d.line([(650, 630), (690, 630)])
        d.dot(650, 630)
        d.line([(970, 630), (1040, 630), (1040, 572)], dashed=True)
        d.line([(1040, 630), (1040, 710)], dashed=True)
        d.text(1080, 646, "M broadcast", 9, color=MUTED)
        formula = "F′ₐ = Fₐ ⊙ S\nS ∈ [0.75, 1.25]"
    else:
        formula = "F′ₐ = Fₐ + γ ⊙ Fₐ ⊙ A\nγ: learnable per-channel scale"
    d.box(1180, 840, 510, 80, formula, TEAL, size=14)
    d.line([(1600, 695), (1600, 840)])
    d.text(1630, 780, "S" if bounded else "A", 12)
    d.line([(535, 600), (535, 820), (1280, 820), (1280, 840)])
    d.text(820, 808, "Fₐ (feature path)", 11)
    d.dot(535, 600)
    d.line([(335, 770), (335, 980), (1180, 980)])
    d.text(690, 967, "Fᵦ: (c − q) × H × W  —  unchanged", 11)
    d.box(1180, 950, 200, 60, "Concat [F′ₐ, Fᵦ]", LILAC)
    d.line([(1435, 920), (1435, 935), (1280, 935), (1280, 950)])
    d.box(1480, 950, 210, 60, "F′: c × H × W", GRAY)
    d.line([(1380, 980), (1480, 980)])
    d.text(65, 1082,
           "Initialization: bias-free directional filters = 0 → S = 1. No learnable γ is used."
           if bounded else "Initialization: γ = 0 → F′ₐ = Fₐ. Directional convolutions include a bias; γ is unconstrained.",
           12, ha="left")
    d.text(65, 1110, "DWConv: depthwise convolution. ⊙: elementwise product. Axis means retain full extent along the other axis.",
           11, color=MUTED, ha="left")
    d.save("axis_bounded_module" if bounded else "axis_residual_module")


def gate_diagram():
    d = Diagram(1640, 900, "Gated interpolation upsampling  |  GatedInterpolationUpsample",
                "Neck P5 → P4 and P4 → P3  •  Upsampling factor 2  •  Batch dimension omitted")
    d.panel(35, 130, 1570, 610, "Three paths: nearest features, bilinear features, and one shared spatial gate")
    d.box(65, 375, 140, 85, "Input X\nC × H × W", GRAY)
    d.box(360, 195, 260, 85, "Nearest upsample ×2\nN: C × 2H × 2W", BLUE)
    d.box(360, 375, 260, 85, "Bilinear upsample ×2\nB: C × 2H × 2W", TEAL)
    d.box(360, 600, 240, 85, "Conv 3 × 3\nC → 1; padding = 1", GOLD)
    d.box(650, 600, 200, 85, "HardSigmoid\ng₀: 1 × H × W", GOLD)
    d.box(900, 600, 210, 85, "Nearest upsample ×2\nG: 1 × 2H × 2W", GOLD, size=11)
    d.line([(205, 417), (270, 417), (270, 237), (360, 237)])
    d.line([(270, 417), (360, 417)])
    d.line([(270, 417), (270, 642), (360, 642)])
    d.dot(270, 417)
    d.op(850, 417, "−")
    d.line([(620, 417), (823, 417)])
    d.line([(620, 237), (1380, 237)])
    d.dot(850, 237)
    d.line([(850, 237), (850, 390)])
    d.text(885, 355, "N", 12)
    d.text(735, 400, "B", 12)
    d.text(1000, 222, "N (baseline path)", 11)
    d.op(1140, 417, "×", GOLD)
    d.line([(877, 417), (1113, 417)])
    d.text(992, 399, "B − N", 12)
    d.line([(600, 642), (650, 642)])
    d.line([(850, 642), (900, 642)])
    d.line([(1110, 642), (1140, 642), (1140, 444)])
    d.text(1245, 570, "G ∈ [0, 1]", 12)
    d.text(1275, 600, "Shared across C channels", 10, color=MUTED)
    d.text(490, 484, "align_corners = False", 10, color=MUTED)
    d.op(1407, 237, "+", LILAC)
    d.line([(1167, 417), (1407, 417), (1407, 264)])
    d.text(1325, 451, "G ⊙ (B − N)", 12)
    d.box(1460, 195, 135, 85, "Output Y\nC × 2H × 2W", GRAY, size=9)
    d.line([(1434, 237), (1460, 237)])
    d.text(820, 787, "Y = N + G ⊙ (B − N) = (1 − G) ⊙ N + G ⊙ B", 18, weight="bold")
    d.text(820, 831, "Initialization: Conv weights = 0; bias = −2.7 → G = 0.05 → Y = 0.95N + 0.05B.", 12)
    d.text(820, 865, "One learned gate per spatial location. Channels are preserved; the output is a pointwise convex combination.",
           11, color=MUTED)
    d.save("gated_interpolation_module")


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "svg.hashsalt": "axis-gate"})
    axis_diagram()
    axis_diagram(bounded=True)
    gate_diagram()
    print("Saved 3 SVG diagrams and 3 PNG previews to", OUT)


if __name__ == "__main__":
    main()
