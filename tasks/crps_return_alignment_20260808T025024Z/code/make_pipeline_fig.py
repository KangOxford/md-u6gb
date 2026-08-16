#!/usr/bin/env python3
"""Paper pipeline figure for the mid-training method, emitted twice from one
layout: an editable draw.io XML and a publication PDF (matplotlib).

Keeping a single BOXES/EDGES table is what keeps the two artifacts in sync;
the draw.io file is the editable source of truth for later revisions, the PDF
is what the TeX includes today.  v2 layout mirrors the paper's mathematics
subsection: the middle row is weight -> tilted target q* (I-projection) ->
distillation, with the IPF loop and its fixed point on the return edge.
"""

import html
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = "/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z/figs"
OVL = "/projects/public/u6gb/overleaf/mid-train-kang-neurips-workshop2026/figures"

# fill, stroke
GRAY = ("#F5F5F5", "#666666")
BLUE = ("#DAE8FC", "#6C8EBF")
YELL = ("#FFF2CC", "#D6B656")
RED = ("#F8CECC", "#B85450")
GREEN = ("#D5E8D4", "#82B366")
PURP = ("#E1D5E7", "#9673A6")

# id: (x, y, w, h, text, colors)   -- draw.io coordinates, y grows downward
BOXES = {
    "data":   (20,  50, 150, 66, "Real LOB stream\n(LOBSTER, event time)", GRAY),
    "ctx":    (205, 50, 160, 66, "Context $C$\n250 real messages", GRAY),
    "model":  (400, 50, 185, 66, "Frozen policy $\\pi_0$\n(pretrained AR model)", BLUE),
    "gen":    (620, 50, 175, 66, "$K$ rollouts $x_i$\n250 generated messages", BLUE),
    "engine": (830, 50, 155, 66, "Matching engine $g$\n(deterministic replay)", GRAY),
    "ret":    (1020, 50, 160, 66, "Terminal return\n$r(x)=\\log m_{500}/m_{250}$", GRAY),
    "kde":    (875, 230, 305, 96, "Density-ratio weight\n$w(r)=\\mathrm{clip}\\,\\hat p_{\\mathrm{real}}(r)/\\hat p_{0}(r)$\nlog-KDE per ticker, self-normalise\nESS audit", YELL),
    "qstar":  (570, 230, 270, 96, "Tilted target (I-projection)\n$q^*(x)=\\pi_0(x)\\,w(r(x))$\nterminal law $=p_{\\mathrm{real}}$; closest\nto $\\pi_0$ in KL with that property", PURP),
    "train":  (205, 230, 330, 96, "Distil $q^*$: weighted MLE\n$\\max_\\theta\\sum_i w_i\\log\\pi_\\theta(x_i)=\\min_\\theta\\mathrm{KL}(q^*\\Vert\\pi_\\theta)$\n($\\equiv$ REINFORCE, sparse terminal reward $w$)\nalternating real-CE anchor, $\\lambda=1$", RED),
    "update": (30,  245, 140, 66, "Full-parameter\nupdate $\\theta\\to\\theta'$", BLUE),
    "eval":   (20,  480, 560, 76, "Primary: fair CRPS, shape $qL_1$, sd ratio, raw + standardised tails\nheld-out contexts, day-block bootstrap", GREEN),
    "ceil":   (570, 480, 270, 64, "Computable ceiling: score the\n$w$-reweighted pool (sd $0.98$, $qL_1$ $0.040$)", GRAY),
    "safety": (880, 480, 300, 76, "Safety (non-inferiority):\nheld-out CE, LOB-Bench 21 features", GREEN),
}

# (src, dst, style, label)  style: solid | dashed | loop
EDGES = [
    ("data", "ctx", "solid", ""),
    ("ctx", "model", "solid", ""),
    ("model", "gen", "solid", "sample"),
    ("gen", "engine", "solid", ""),
    ("engine", "ret", "solid", ""),
    ("ret", "kde", "solid", ""),
    ("kde", "qstar", "solid", "tilt"),
    ("qstar", "train", "solid", "distil"),
    ("ctx", "train", "dashed", "real continuation (anchor)"),
    ("train", "update", "solid", ""),
    ("update", "model", "loop", "IPF round: re-sample from $\\pi_{\\theta'}$, re-estimate $\\hat p$;   fixed point $w\\equiv 1$ (matched law);   stop: dispersion stall or safety gate"),
    ("update", "eval", "solid", ""),
    ("qstar", "ceil", "dashed", ""),
]


def emit_drawio(path):
    # LaTeX in labels stays literal; draw.io renders $...$ via its MathJax option.
    cells = []
    for bid, (x, y, w, h, text, (fill, stroke)) in BOXES.items():
        v = html.escape(text.replace("\n", "&#10;"), quote=True).replace("&amp;#10;", "&#10;")
        cells.append(
            f'<mxCell id="{bid}" value="{v}" style="rounded=1;whiteSpace=wrap;html=1;'
            f'fillColor={fill};strokeColor={stroke};fontSize=12;" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    for i, (s, d, style, label) in enumerate(EDGES):
        dash = "dashed=1;" if style == "dashed" else ""
        extra = "exitX=0.5;exitY=0;entryX=0;entryY=0.5;" if style == "loop" else ""
        v = html.escape(label.replace("\n", "&#10;"), quote=True).replace("&amp;#10;", "&#10;")
        cells.append(
            f'<mxCell id="e{i}" value="{v}" style="edgeStyle=orthogonalEdgeStyle;rounded=0;'
            f'html=1;{dash}{extra}fontSize=11;" edge="1" parent="1" source="{s}" target="{d}">'
            f'<mxGeometry relative="1" as="geometry"/></mxCell>')
    xml = ('<mxfile host="app.diagrams.net"><diagram id="midtrain" name="pipeline">'
           '<mxGraphModel dx="1400" dy="800" grid="1" gridSize="10" guides="1" tooltips="1" '
           'connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1200" '
           'pageHeight="620" math="1" shadow="0"><root><mxCell id="0"/>'
           '<mxCell id="1" parent="0"/>' + "".join(cells) +
           '</root></mxGraphModel></diagram></mxfile>')
    with open(path, "w") as f:
        f.write(xml)
    print("wrote", path)


def _c(bid):
    x, y, w, h, *_ = BOXES[bid]
    return x + w / 2, y + h / 2


def _edge_pts(s, d):
    """Anchor on facing sides, matplotlib coords (y already flipped by axis)."""
    xs, ys, ws, hs, *_ = BOXES[s]
    xd, yd, wd, hd, *_ = BOXES[d]
    cxs, cys = _c(s)
    cxd, cyd = _c(d)
    if abs(cys - cyd) < 20:            # same row -> horizontal
        if cxs < cxd:
            return (xs + ws, cys), (xd, cyd)
        return (xs, cys), (xd + wd, cyd)
    # vertical: drop straight from the source centre when the target spans it,
    # so long strips receive a plumb line instead of a diagonal into their centre
    xt = cxs if xd <= cxs <= xd + wd else cxd
    if cys < cyd:                       # downward
        return (cxs, ys + hs), (xt, yd)
    return (cxs, ys), (xt, yd + hd)     # upward


def emit_pdf(path):
    fig, ax = plt.subplots(figsize=(12.0, 5.9))
    ax.set_xlim(0, 1200)
    ax.set_ylim(620, 0)                 # draw.io y grows down
    ax.axis("off")
    for bid, (x, y, w, h, text, (fill, stroke)) in BOXES.items():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=8",
                                    facecolor=fill, edgecolor=stroke, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.3)
    for s, d, style, label in EDGES:
        if style == "loop":
            # update.left -> left margin -> up along x=10 -> right at y=22 -> model.top
            xs, ys, ws, hs, *_ = BOXES[s]
            xd, yd, wd, hd, *_ = BOXES[d]
            cy = ys + hs / 2
            pts = [(xs, cy), (10, cy), (10, 22), (xd + wd / 2, 22)]
            ax.plot(*zip(*pts), color="#B85450", lw=1.3, ls=(0, (5, 3)))
            ax.annotate("", xy=(xd + wd / 2, yd), xytext=pts[-1],
                        arrowprops=dict(arrowstyle="-|>", color="#B85450",
                                        lw=1.3, linestyle=(0, (5, 3))))
            ax.text(600, 12, label.replace("\n", "   "),
                    ha="center", va="center", fontsize=8.0, color="#B85450")
            continue
        (x0, y0), (x1, y1) = _edge_pts(s, d)
        ls = (0, (4, 3)) if style == "dashed" else "-"
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.1,
                                    linestyle=ls, shrinkA=2, shrinkB=2))
        if label:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 - 8, label.replace("\n", " "),
                    ha="center", va="center", fontsize=7.8, color="#444444",
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.2))
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=150)
    print("wrote", path)


if __name__ == "__main__":
    import os
    for base in (OUT, OVL):
        os.makedirs(base, exist_ok=True)
    emit_drawio(f"{OUT}/fig_midtrain_pipeline.drawio")
    emit_drawio(f"{OVL}/fig_midtrain_pipeline.drawio")
    emit_pdf(f"{OUT}/fig_midtrain_pipeline.pdf")
    emit_pdf(f"{OVL}/fig_midtrain_pipeline.pdf")
    emit_pdf(f"{OUT}/fig_midtrain_pipeline_preview.png")
