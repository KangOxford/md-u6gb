#!/usr/bin/env python3
"""A03 出图：复合误差曲线，两种看法并排。

上排是测到的原样（水平 + 斜率），下排把每条曲线平移到起点为 0（只剩累积）。
这两排是同一份数据，差别就是论点：修正臂的**水平**更高，但**累积**几乎没有。
斜率表把这两件事压成一个数，于是看不出来。
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.lines import Line2D                  # noqa: E402

FIELDS = ["event_type", "direction", "price_rel", "size", "log10_dt"]
NICE = {"event_type": "event_type", "direction": "direction",
        "price_rel": "price_rel  (Δprice / tick)", "size": "size",
        "log10_dt": "log10_dt"}
C = {"draft": "#8b95a1", "corr": "#1f4e79", "rand": "#b0562e"}
LBL = {"draft": "draft  (pretrained Mamba-3, no DFM)",
       "corr": "corrected  (+ DFM residual, learned)",
       "rand": "control  (+ residual, random direction)"}
INK, GRID, BAND = "#16191d", "#e6e9ec", "#f5f7f8"


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.edgecolor": "#c9ced3", "axes.linewidth": 0.7,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": "#5b6570", "ytick.color": "#5b6570",
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "legend.frameon": False,
    })


def panel(ax, g, z, f, shift, arms):
    for a in arms:
        m = z[f"{f}__{a}__mean"]
        lo, hi = z[f"{f}__{a}__lo"], z[f"{f}__{a}__hi"]
        off = m[0] if shift else 0.0
        ax.fill_between(g, lo - off, hi - off, color=C[a], alpha=0.16, lw=0)
        ax.plot(g, m - off, color=C[a], lw=1.9,
                zorder=3 if a == "corr" else 2)
    ax.axhline(0, color=INK, lw=0.9, ls=(0, (4, 3)), alpha=0.55, zorder=1)
    ax.set_xlim(g[0], g[-1])
    ax.set_xticks([50, 150, 250, 350, 440])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--sub", default="")
    a = ap.parse_args()
    style()
    z = np.load(a.npz)
    g = z["grid"].astype(float)
    n_tk = len(z["tickers"])
    arms = [k for k in ("draft", "corr", "rand") if f"{FIELDS[0]}__{k}__mean" in z]

    fig, AX = plt.subplots(2, 5, figsize=(15.6, 6.9),
                           gridspec_kw=dict(hspace=0.44, wspace=0.29,
                                            top=0.795, bottom=0.135,
                                            left=0.088, right=0.988))
    for j, f in enumerate(FIELDS):
        for i, shift in enumerate((False, True)):
            ax = AX[i, j]
            panel(ax, g, z, f, shift, arms)
            if i == 0:
                ax.set_title(NICE[f], fontsize=10, pad=7, color=INK,
                             fontweight="semibold")
            else:
                # 累积量：末端相对起点涨了多少（每条曲线自己的起点）
                rise = {k: float(z[f"{f}__{k}__mean"][-1]
                                 - z[f"{f}__{k}__mean"][0]) for k in arms}
                txt = "\n".join(f"{k:<6s}{rise[k]:+.3f}" for k in arms)
                ax.text(0.035, 0.965, txt, transform=ax.transAxes, va="top",
                        ha="left", fontsize=7.6, family="DejaVu Sans Mono",
                        color="#3d4650",
                        bbox=dict(fc="white", ec=GRID, lw=0.6, pad=3.2))
                ax.set_xlabel("generated message index  m", fontsize=8.5)
    AX[0, 0].set_ylabel("excess divergence over floor", fontsize=8.8,
                        color=INK, labelpad=6)
    AX[1, 0].set_ylabel("same curves, shifted to start at 0", fontsize=8.8,
                        color=INK, labelpad=6)

    # 左侧色带 + 竖排大字：两排是同一份数据的两种读法，这是全图的论点
    for i, tag in enumerate(("AS MEASURED\nlevel + accumulation",
                             "ACCUMULATION ONLY\nlevel removed")):
        bb = AX[i, 0].get_position()
        col = "#1f4e79" if i else "#5b6570"
        fig.patches.append(plt.Rectangle(
            (0.010, bb.y0), 0.006, bb.height, transform=fig.transFigure,
            fc=col, ec="none", alpha=0.9, zorder=5))
        fig.text(0.026, bb.y0 + bb.height / 2, tag, rotation=90, va="center",
                 ha="center", fontsize=8.1, color=col, fontweight="bold",
                 linespacing=1.5)

    fig.legend(handles=[Line2D([], [], color=C[k], lw=2.4, label=LBL[k])
                        for k in arms]
               + [Line2D([], [], color=INK, lw=1.0, ls=(0, (4, 3)), alpha=.6,
                         label="floor  (real data vs itself) = 0 by construction")],
               loc="upper left", bbox_to_anchor=(0.048, 0.900), ncol=4,
               fontsize=8.6, handlelength=2.4, columnspacing=2.6,
               labelspacing=0.42)
    fig.text(0.052, 0.982, a.title, fontsize=15, fontweight="bold", color=INK,
             ha="left", va="top")
    fig.text(0.052, 0.940, a.sub, fontsize=9.2, color="#5b6570", ha="left",
             va="top")
    fig.text(0.052, 0.038,
             f"{n_tk} held-out tickers, 8 sequences each, 500 generated "
             f"messages · shaded = 95% CI from ticker-level bootstrap "
             f"(2000 draws) · symmetric cross-half estimator, "
             f"entropy-normalised",
             fontsize=7.8, color="#7b858f", ha="left", va="bottom")
    fig.savefig(a.out, dpi=170)
    fig.savefig(a.out.replace(".png", ".pdf"))
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
