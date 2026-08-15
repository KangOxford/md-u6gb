#!/usr/bin/env python3
"""目标的两半画在一张图上：水平 vs 累积，每个字段一格。

这张图要回答的问题是「DFM 后训练到底做到了什么」，而它有两个坐标：

    x = 水平 @ m=440    分布现在有多错     越小越好
    y = 累积涨幅        错得多快在增长     越小越好

draft 在右下（水平好、会累积），纯修正器在左上（不累积、水平差）。keep-tau
把点沿着两者之间移动。**如果所有点都落在 draft 与修正器的连线附近，那就是
在前沿上插值，没有压过任何一端** —— 而压过去才叫达成目标。

左下角（同时更低）是目标区，图上用阴影标出。
"""
import glob
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
NICE = {"event_type": "event_type", "direction": "direction",
        "price_rel": "price_rel", "size": "size", "log10_dt": "log10_dt"}


def curves(pat, month='2026-01'):
    fs = sorted(glob.glob(pat))
    acc = {f: {'draft': [], 'corr': []} for f in FIELDS}
    for p in fs:
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
        for nm in FIELDS:
            try:
                e = make_edges(F['true'][nm], F['true']['_ok'], nm, 40)
                cT, K = codes(F['true'][nm], F['true']['_ok'], nm, e)
                CT = build_counts(cT, G, g, 40, Rr, K)
                fl = D_curve(CT[0], CT[1], CT[0], CT[1], K)
                for arm in ('draft', 'corr'):
                    cG, _ = codes(F[arm][nm], F[arm]['_ok'], nm, e)
                    CG = build_counts(cG, G, g, 40, Rr, K)
                    acc[nm][arm].append(
                        D_curve(CT[0], CT[1], CG[0], CG[1], K) - fl)
            except Exception:
                pass
    n = len(acc[FIELDS[0]]['corr'])
    return n, {f: {a: np.nanmean(np.vstack(v), 0)
                   for a, v in d.items() if v} for f, d in acc.items()}


ARMS = [
    ("draft",       f"{R}/rollouts_tau/tau_t0_*_2026-01_learned.npz", "draft",
     "#8b95a1", "o"),
    ("tau 0.01",    f"{R}/rollouts_tau/tau_t001_*_2026-01_learned.npz", "corr",
     "#2f7d5a", "s"),
    ("tau 0.1",     f"{R}/rollouts_tau/tau_t01_*_2026-01_learned.npz", "corr",
     "#3d7fb8", "^"),
    ("corrector",   f"{R}/rollouts_tau/tau_t0_*_2026-01_learned.npz", "corr",
     "#b0562e", "D"),
]

if __name__ == "__main__":
    pts = {}
    for label, pat, arm, col, mk in ARMS:
        n, c = curves(pat)
        if n == 0:
            print(f"  {label}: 无产物"); continue
        pts[label] = {f: (float(c[f][arm][-1]),
                          float(c[f][arm][-1] - c[f][arm][0])) for f in FIELDS}
        print(f"  {label:<11} n={n}")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.edgecolor": "#c9ced3",
        "axes.linewidth": 0.7, "figure.facecolor": "white",
        "savefig.facecolor": "white", "legend.frameon": False,
    })
    fig, AX = plt.subplots(1, 5, figsize=(15.2, 3.8),
                           gridspec_kw=dict(wspace=0.30, top=0.72,
                                            bottom=0.18, left=0.055,
                                            right=0.988))
    for j, f in enumerate(FIELDS):
        ax = AX[j]
        dx, dy = pts["draft"][f]
        # 目标区 = 左下矩形：两个坐标**同时**低于 draft
        ax.axvline(dx, color="#c9ced3", lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.axhline(dy, color="#c9ced3", lw=0.8, ls=(0, (4, 3)), zorder=1)
        # draft 与纯修正器之间的连线 = 前沿
        cx, cy = pts["corrector"][f]
        ax.plot([dx, cx], [dy, cy], color="#c9ced3", lw=1.4, zorder=2)
        for label, _p, _a, col, mk in ARMS:
            if label not in pts:
                continue
            x, y = pts[label][f]
            ax.scatter([x], [y], s=64, c=col, marker=mk, zorder=4,
                       edgecolor="white", linewidth=0.8, label=label)
        ax.set_title(NICE[f], fontsize=10, pad=6, fontweight="semibold")
        ax.set_xlabel("level @ m=440", fontsize=8.5)
        if j == 0:
            ax.set_ylabel("accumulation  (rise over m=40..440)", fontsize=8.5)
        ys = [pts[l][f][1] for l in pts]
        xs = [pts[l][f][0] for l in pts]
        my = 0.18 * (max(ys) - min(ys) + 1e-9)
        mx = 0.18 * (max(xs) - min(xs) + 1e-9)
        y0, y1 = min(ys) - my, max(ys) + my
        x0, x1 = min(xs) - mx, max(xs) + mx
        ax.add_patch(plt.Rectangle((x0, y0), dx - x0, dy - y0,
                                   color="#e3efe8", zorder=0))
        ax.set_ylim(y0, y1); ax.set_xlim(x0, x1)
        # 这一格里有没有臂落进目标区
        win = [l for l in pts if l != "draft"
               and pts[l][f][0] < dx and pts[l][f][1] < dy]
        ax.text(0.03, 0.04, ("GOAL: " + win[0]) if win else "no arm in the box",
                transform=ax.transAxes, fontsize=8,
                color="#2f7d5a" if win else "#9aa4ad",
                fontweight="bold" if win else "normal")

    h, l = AX[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.055, 0.885), ncol=4,
               fontsize=9, handletextpad=0.4, columnspacing=2.2)
    fig.text(0.055, 0.965, "One field is genuinely won; three are a trade-off; one is lost",
             fontsize=14, fontweight="bold", color=INK, va="top")
    fig.text(0.055, 0.912, "Shaded box = BOTH coordinates below the draft's, "
             "which is the goal. `direction` has the corrector inside it; "
             "`event_type`, `price_rel` and `size` only trade one for the "
             "other; on `log10_dt` the pure CORRECTOR loses both (the keep-tau "
             "arms there are still a trade-off).",
             fontsize=9, color="#5b6570", va="top")
    fig.text(0.055, 0.035, "60-ticker sweep, same checkpoint lg488b_g2, "
             "2026-01 held out. Grey line joins the draft to the pure "
             "corrector; keep-tau arms sit on it.",
             fontsize=7.8, color="#7b858f", va="bottom")
    out = f"{R}/figs/frontier_2026-01.png"
    import os
    os.makedirs(f"{R}/figs", exist_ok=True)
    fig.savefig(out, dpi=170)
    fig.savefig(out.replace(".png", ".pdf"))
    print("wrote", out)
