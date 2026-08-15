#!/usr/bin/env python3
"""目标区图：水平 vs 累积，度量 A/B 的三条臂 + 3500 步的旧点。

与 `plot_frontier.py` 的区别是**臂不同**：那张图扫的是推理旋钮（说明 keep-tau
只在一维前沿上滑），这张图比的是**训练**（3500 步 vs 14000 步 x 两条度量），
而这一组里目标真的被达成了。

阴影框 = 两个坐标同时低于 draft = 目标。
"""
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402

sys.path.insert(0, '/lus/lfs1aip2/projects/public/u6gb/tasks/'
                   'ce_orderflow_20260812T200352Z/A01_does_ce_exist/code')
from a01_ce_existence import (FIELDS, action_fields, build_counts,   # noqa: E402
                              codes, make_edges, D_curve)

R = '/lus/lfs1aip2/projects/public/u6gb/tasks/dfm_largescale_20260813'
INK, GRID = "#16191d", "#e6e9ec"
MO = sys.argv[1] if len(sys.argv) > 1 else "2026-01"

ARMS = [
    ("draft (Mamba-3, no DFM)", f"{R}/rollouts_metric/met_field_*_{MO}_learned.npz",
     "draft", "#8b95a1", "o"),
    ("DFM 3,500 steps",         f"{R}/rollouts_tau/tau_t0_*_{MO}_learned.npz",
     "corr",  "#b0562e", "D"),
    ("DFM 14,000 steps",        f"{R}/rollouts_metric/met_field_*_{MO}_learned.npz",
     "corr",  "#3d7fb8", "^"),
    ("DFM 14,000 + value metric", f"{R}/rollouts_metric/met_value_*_{MO}_learned.npz",
     "corr",  "#2f7d5a", "s"),
]


def curves(pat, arm):
    per = {}
    for p in sorted(glob.glob(pat)):
        tk = p.split('/')[-1].replace('.npz', '').split('_')[-3]
        try:
            z = np.load(p, allow_pickle=True)
            F = action_fields(z, 100)
        except Exception:
            continue
        N, Rr = z['real_msgs'].shape[0], z['real_msgs'].shape[1]
        if N < 4 or Rr < 160:
            continue
        g = np.arange(40, Rr - 40, 20)
        perm = np.random.default_rng(20260813).permutation(N)
        G = [perm[:N // 2], perm[N // 2:]]
        d = {}
        for nm in FIELDS:
            try:
                e = make_edges(F['true'][nm], F['true']['_ok'], nm, 40)
                cT, K = codes(F['true'][nm], F['true']['_ok'], nm, e)
                CT = build_counts(cT, G, g, 40, Rr, K)
                fl = D_curve(CT[0], CT[1], CT[0], CT[1], K)
                cG, _ = codes(F[arm][nm], F[arm]['_ok'], nm, e)
                CG = build_counts(cG, G, g, 40, Rr, K)
                d[nm] = D_curve(CT[0], CT[1], CG[0], CG[1], K) - fl
            except Exception:
                pass
        per[tk] = d
    return per


if __name__ == "__main__":
    raw = {}
    for label, pat, arm, _c, _m in ARMS:
        raw[label] = curves(pat, arm)
        print(f"  {label:<28} {len(raw[label])} ticker")
    common = None
    for v in raw.values():
        common = set(v) if common is None else (common & set(v))
    common = sorted(common or [])
    print(f"  共同 ticker: {len(common)}")
    if len(common) < 5:
        sys.exit("共同 ticker 太少")

    pts = {}
    for label in raw:
        pts[label] = {}
        for nm in FIELDS:
            c = np.vstack([raw[label][t][nm] for t in common if nm in raw[label][t]])
            m = np.nanmean(c, 0)
            pts[label][nm] = (float(m[-1]), float(m[-1] - m[0]))

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.edgecolor": "#c9ced3",
        "axes.linewidth": 0.7, "figure.facecolor": "white",
        "savefig.facecolor": "white", "legend.frameon": False,
    })
    fig, AX = plt.subplots(1, 5, figsize=(15.6, 3.9),
                           gridspec_kw=dict(wspace=0.32, top=0.70,
                                            bottom=0.19, left=0.058,
                                            right=0.988))
    dl_lab = ARMS[0][0]
    n_in = {a[0]: 0 for a in ARMS[1:]}
    for j, f in enumerate(FIELDS):
        ax = AX[j]
        dx, dy = pts[dl_lab][f]
        xs = [pts[l][f][0] for l in pts]
        ys = [pts[l][f][1] for l in pts]
        mx = 0.20 * (max(xs) - min(xs) + 1e-9)
        my = 0.20 * (max(ys) - min(ys) + 1e-9)
        x0, x1 = min(xs) - mx, max(xs) + mx
        y0, y1 = min(ys) - my, max(ys) + my
        ax.add_patch(plt.Rectangle((x0, y0), dx - x0, dy - y0,
                                   color="#e3efe8", zorder=0))
        ax.axvline(dx, color="#c9ced3", lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.axhline(dy, color="#c9ced3", lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.axhline(0.0, color="#b9c0c7", lw=0.9, zorder=1)
        for label, _p, _a, col, mk in ARMS:
            x, y = pts[label][f]
            ax.scatter([x], [y], s=70, c=col, marker=mk, zorder=4,
                       edgecolor="white", linewidth=0.9, label=label)
            if label != dl_lab and x < dx and y < dy:
                n_in[label] += 1
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_title(f, fontsize=10, pad=6, fontweight="semibold")
        ax.set_xlabel("level  @ m=440", fontsize=8.5)
        if j == 0:
            ax.set_ylabel("accumulation  (m=40 → 440)", fontsize=8.5)
        inbox = [l for l in pts if l != dl_lab
                 and pts[l][f][0] < dx and pts[l][f][1] < dy]
        ax.text(0.035, 0.05, f"{len(inbox)} arm(s) in the box" if inbox
                else "none in the box", transform=ax.transAxes, fontsize=8,
                color="#2f7d5a" if inbox else "#9aa4ad",
                fontweight="bold" if inbox else "normal")

    h, l = AX[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.055, 0.875), ncol=4,
               fontsize=8.8, handletextpad=0.4, columnspacing=2.0)
    fig.text(0.058, 0.965,
             "Two fields now clear BOTH bars: lower level AND lower accumulation "
             "than the draft", fontsize=14, fontweight="bold", color=INK, va="top")
    fig.text(0.058, 0.916,
             "Shaded box = both coordinates below the draft's, which is the goal. "
             "At 3,500 steps only `direction` was inside; 14,000 steps adds "
             "`price_rel` under BOTH metrics, and its accumulation goes negative "
             "(solid line = 0), i.e. the error SHRINKS with depth.",
             fontsize=9, color="#5b6570", va="top")
    fig.text(0.058, 0.035,
             f"{len(common)} held-out tickers, {MO}. Level = excess divergence "
             f"over the real-vs-real floor at m=440; accumulation = its rise from "
             f"m=40. Verdicts identical under JS (gate G2).",
             fontsize=7.8, color="#7b858f", va="bottom")
    os.makedirs(f"{R}/figs", exist_ok=True)
    out = f"{R}/figs/goalbox_{MO}.png"
    fig.savefig(out, dpi=170)
    fig.savefig(out.replace(".png", ".pdf"))
    print("\n每条臂进目标区的字段数: " +
          "  ".join(f"{k}={v}/5" for k, v in n_in.items()))
    print("wrote", out)
