"""Draw implementation-matched v3 diagrams as editable SVG, vector PDF and PNG."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "figures"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15,
                     "pdf.fonttype": 42, "svg.fonttype": "none", "mathtext.fontset": "dejavusans"})
INK = "#26384A"
BLUE, GREEN, ORANGE, PURPLE = "#E4EDF8", "#E4F2EB", "#FBEAD9", "#EEE7F7"


def text(ax, x, y, value, size=15, **kwargs):
    kwargs.setdefault("ha", "center")
    kwargs.setdefault("va", "center")
    ax.text(x, y, value, fontsize=size, color=INK, **kwargs)


def box(ax, x, y, w, h, label, color=BLUE, size=15, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.025,rounding_size=0.07",
                              facecolor=color, edgecolor="#7B91A4", linewidth=1.3,
                              linestyle="--" if dashed else "-"))
    text(ax, x+w/2, y+h/2, label, size)


def path(ax, points, color=INK, arrow=True, dashed=False):
    for i, (a, b) in enumerate(zip(points, points[1:])):
        if arrow and i == len(points)-2:
            ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13,
                                        linewidth=1.4, color=color, linestyle="--" if dashed else "-"))
        else:
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=1.4,
                    linestyle="--" if dashed else "-")


def op(ax, x, y, value, radius=.18):
    ax.add_patch(Circle((x,y),radius,fc="white",ec=INK,lw=1.4))
    text(ax,x,y,value,17)


def canvas(height):
    fig, ax = plt.subplots(figsize=(14,height))
    ax.set(xlim=(0,14),ylim=(0,height))
    ax.axis("off")
    fig.subplots_adjust(left=.015,right=.985,top=.985,bottom=.015)
    return fig, ax


def sgre(ax, offset=0):
    def t(x,y,s,size=15,**kw): text(ax,x,y+offset,s,size,**kw)
    def b(x,y,w,h,s,c=BLUE,size=15,dashed=False): box(ax,x,y+offset,w,h,s,c,size,dashed)
    def p(points,**kw): path(ax,[(x,y+offset) for x,y in points],**kw)
    def o(x,y,s): op(ax,x,y+offset,s)
    t(.22,5.65,"(a) SGRE | Selective Gradient Residual Enhancement",20,ha="left")
    t(7,5.15,"After every bottleneck in backbone P3 / P4; all hidden channels",15)
    # The original bottleneck includes its own shortcut before entering SGRE.
    b(.3,3.05,1.65,.9,"Bottleneck\noutput",BLUE)
    t(1.1,2.7,r"$F: B\times C\times H\times W$",14)
    p([(1.95,3.5),(2.25,3.5),(2.25,4.65),(12.6,4.65),(12.6,3.5)])
    t(8.7,4.87,"Identity",14)
    p([(2.25,3.5),(2.25,1.0),(3.0,1.0)])
    b(3,0.55,2.15,.9,"DWConv 3 × 3\nC → C",ORANGE)
    p([(5.15,1),(5.65,1)])
    b(5.65,.55,1.7,.9,"Hardsigmoid",ORANGE,14)
    p([(7.35,1),(10.8,1),(10.8,3.31)])
    t(9.1,1.28,r"$G: B\times C\times H\times W$",14)
    p([(2.25,3.5),(3.0,3.5)])
    b(3,3.05,2.15,.9,"DW Sobel x\n3 × 3 (fixed)",GREEN,14,True)
    p([(2.25,3.5),(2.65,3.5),(2.65,2.05),(3,2.05)])
    b(3,1.6,2.15,.9,"DW Sobel y\n3 × 3 (fixed)",GREEN,14,True)
    b(5.65,3.15,.8,.7,"Abs",GREEN,14)
    b(5.65,1.7,.8,.7,"Abs",GREEN,14)
    p([(5.15,3.5),(5.65,3.5)])
    p([(5.15,2.05),(5.65,2.05)])
    o(7.15,3.5,"+")
    p([(6.45,3.5),(6.95,3.5)])
    p([(6.45,2.05),(7.15,2.05),(7.15,3.3)])
    t(7.5,3.87,r"$E$",15)
    p([(7.35,3.5),(7.9,3.5)])
    b(7.9,3.05,2,.9,"Conv 1 × 1\nC → C",PURPLE)
    t(8.9,2.8,"No bias",13)
    p([(9.9,3.5),(10.6,3.5)])
    o(10.8,3.5,"×")
    p([(11,3.5),(12.4,3.5)])
    o(12.6,3.5,"+")
    p([(12.8,3.5),(13.7,3.5)])
    t(13.35,3.95,r"$Z$",17)
    t(7,.12,r"$Z=F+\mathrm{HS}(\mathrm{DWConv}_{3\times3}(F))\odot"
      r"\mathrm{Conv}_{1\times1}(|K_x*_{\rm dw}F|+|K_y*_{\rm dw}F|)$",17)


def saru(ax, offset=0):
    def t(x,y,s,size=15,**kw): text(ax,x,y+offset,s,size,**kw)
    def b(x,y,w,h,s,c=BLUE,size=15): box(ax,x,y+offset,w,h,s,c,size)
    def p(points,**kw): path(ax,[(x,y+offset) for x,y in points],**kw)
    def o(x,y,s): op(ax,x,y+offset,s)
    t(.22,5.65,"(b) SARU | Subpixel-Adaptive Residual Upsampling",20,ha="left")
    t(7,5.15,"Low-resolution filtering + four subpixel-position weights",15)
    t(.5,3.25,r"$X$",19)
    p([(.8,3.25),(1.2,3.25),(1.2,4.2),(2,4.2)])
    b(2,3.75,2.1,.9,"Nearest ×2",BLUE)
    p([(4.1,4.2),(12.55,4.2),(12.55,3.25)])
    t(8.3,4.5,r"$N: B\times C\times2H\times2W$",14)
    p([(1.2,3.25),(2,3.25)])
    b(2,2.8,2.1,.9,"DWConv 3 × 3\nC → C",GREEN)
    t(3.0,2.45,r"$X: B\times C\times H\times W$",14)
    p([(4.1,3.25),(5.25,3.25)])
    b(5.25,2.8,2,.9,"Nearest ×2",GREEN)
    p([(7.25,3.25),(10.65,3.25)])
    t(8.7,3.55,r"$R: B\times C\times2H\times2W$",14)
    p([(1.2,3.25),(1.2,1.35),(2,1.35)])
    b(2,.9,2.1,.9,"Conv 1 × 1\nC → 4",ORANGE)
    p([(4.1,1.35),(4.55,1.35)])
    b(4.55,.9,1.75,.9,"Hardsigmoid",ORANGE,14)
    t(4.95,.55,r"$B\times4\times H\times W$",14)
    p([(6.3,1.35),(6.8,1.35)])
    b(6.8,.9,2,.9,"PixelShuffle ×2",PURPLE,14)
    t(7.8,.55,r"$B\times1\times2H\times2W$",14)
    p([(8.8,1.35),(9.25,1.35)])
    b(9.25,.9,2.1,.9,"Broadcast\n1 → C channels",PURPLE,14)
    p([(11.35,1.35),(11.65,1.35),(11.65,2.5),(10.85,2.5),(10.85,3.05)])
    o(10.85,3.25,"×")
    p([(11.05,3.25),(12.35,3.25)])
    o(12.55,3.25,"+")
    p([(12.75,3.25),(13.7,3.25)])
    t(13.35,3.65,r"$Y$",19)
    t(12.1,2.8,r"$B\times C\times2H\times2W$",14)
    t(7,.12,r"$Y=\mathrm{Nearest}_{2}(X)+"
      r"\mathrm{PS}_{2}(\mathrm{HS}(\mathrm{Conv}_{1\times1}(X)))"
      r"\odot\mathrm{Nearest}_{2}(\mathrm{DWConv}_{3\times3}(X))$",17)


def export(fig, name):
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf","svg","png"):
        output = OUT/f"{name}.{ext}"
        fig.savefig(output,dpi=220,facecolor="white")
        if ext == "svg":
            lines = output.read_text(encoding="utf-8").splitlines()
            output.write_bytes(("\n".join(line.rstrip() for line in lines)+"\n").encode("utf-8"))
    plt.close(fig)


def architecture():
    fig, ax=canvas(7.5)
    text(ax,.2,7.15,"Network architecture | Backbone P3/P4 refinement and top-down reconstruction",19,ha="left")
    text(ax,2.0,6.65,"Backbone",17,weight="bold")
    text(ax,7.55,6.65,"PAN-FPN",17,weight="bold")
    text(ax,12.4,6.65,"Detect",17,weight="bold")
    stages=[("Stem + P2 C2f",5.55,BLUE),("P3 downsample +\nC2f-SGRE",4.15,ORANGE),
            ("P4 downsample +\nC2f-SGRE",2.75,ORANGE),("P5 downsample +\nC2f + SPPF",1.35,BLUE)]
    for label,y,color in stages: box(ax,.5,y,3,.7,label,color,14)
    for hi,lo in zip(stages,stages[1:]): path(ax,[(2,hi[1]),(2,lo[1]+.7)])
    # P5 -> SARU -> lateral P4 fusion -> SARU -> lateral P3 fusion.
    box(ax,4.25,1.35,1.35,.7,"SARU",PURPLE)
    path(ax,[(3.5,1.7),(4.25,1.7)])
    op(ax,6,3.1,"C")
    path(ax,[(5.6,1.7),(6,1.7),(6,2.9)])
    path(ax,[(3.5,3.1),(5.8,3.1)])
    box(ax,6.55,2.75,1.45,.7,"C2f",BLUE)
    path(ax,[(6.2,3.1),(6.55,3.1)])
    box(ax,6.55,4.15,1.45,.7,"SARU",PURPLE)
    path(ax,[(7.275,3.45),(7.275,4.15)])
    op(ax,8.6,5.9,"C")
    path(ax,[(7.275,4.85),(7.275,5.9),(8.4,5.9)])
    path(ax,[(3.5,4.5),(4,4.5),(4,5.9),(8.4,5.9)])
    box(ax,9.15,5.55,1.55,.7,"C2f / P3",BLUE,14)
    path(ax,[(8.8,5.9),(9.15,5.9)])
    box(ax,11.7,5.55,1.5,.7,"Head P3",GREEN,14)
    path(ax,[(10.7,5.9),(11.7,5.9)])
    box(ax,9.15,3.65,1.55,.7,"Conv s=2",BLUE,14)
    path(ax,[(9.925,5.55),(9.925,4.35)])
    op(ax,9.925,3.1,"C")
    path(ax,[(9.925,3.65),(9.925,3.3)])
    path(ax,[(8,3.1),(9.725,3.1)])
    box(ax,9.15,1.95,1.55,.7,"C2f / P4",BLUE,14)
    path(ax,[(9.925,2.9),(9.925,2.65)])
    box(ax,11.7,1.95,1.5,.7,"Head P4",GREEN,14)
    path(ax,[(10.7,2.3),(11.7,2.3)])
    box(ax,9.15,.75,1.55,.7,"Conv s=2",BLUE,14)
    path(ax,[(9.925,1.95),(9.925,1.45)])
    op(ax,9.925,.2,"C")
    path(ax,[(9.925,.75),(9.925,.4)])
    path(ax,[(3.5,1.7),(3.8,1.7),(3.8,.2),(9.725,.2)])
    box(ax,9.15,-.95,1.55,.7,"C2f / P5",BLUE,14)
    path(ax,[(9.925,0),(9.925,-.25)])
    box(ax,11.7,-.95,1.5,.7,"Head P5",GREEN,14)
    path(ax,[(10.7,-.6),(11.7,-.6)])
    ax.set_ylim(-1.25,7.5)
    export(fig,"network-v3")


if __name__ == "__main__":
    for name,draw in (("sgre",sgre),("saru",saru)):
        fig,ax=canvas(6)
        draw(ax)
        export(fig,name)
    fig,ax=canvas(12.4)
    sgre(ax,6.4)
    ax.plot([.2,13.8],[6.2,6.2],color="#C5CED6",lw=1)
    saru(ax)
    export(fig,"modules-v3")
    architecture()
