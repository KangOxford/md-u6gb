#!/usr/bin/env python3
"""The paper's distribution figure, from real evaluation data.

Three panels answer the user's three questions in order: what the terminal
return distribution looked like before mid-training (base vs real), what it
looks like after (wm_ft_b vs real), and what produced the change (the terminal
density-ratio weight w(r) that the weighted MLE consumed in round 1).
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = Path("/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/crps-return-alignment-20260808")
T = Path("/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z")
sys.path.insert(0, str(W / "run" / "mid_training"))
sys.path.insert(0, str(W / "src"))
from compare_arms import load_arm, load_real  # noqa: E402

BP = 1e4  # log-return -> basis points


def pooled(roots):
    out = []
    for r in roots:
        out.extend(load_arm(T / "data" / r).values())
    return np.asarray(out) * BP


def kde(x, grid, bw):
    # plain gaussian KDE; bandwidth shared across arms so shapes are comparable
    d = (grid[:, None] - x[None, :]) / bw
    return np.exp(-0.5 * d * d).sum(axis=1) / (x.size * bw * np.sqrt(2 * np.pi))


def main():
    real = np.asarray(list(load_real(T / "data" / "hp_mdbase_s91001").values())) * BP
    base = pooled([f"hp_mdbase_s9100{i}" for i in (1, 2, 3)])
    ft = pooled([f"hp_mdwmftb_s9100{i}" for i in (1, 2, 3)])

    # round-1 weights on the base policy's own rollouts: (root, rid) -> w,
    # joined with each rollout's terminal return from the same collection
    z = np.load(T / "v5w_weights.npz", allow_pickle=True)
    wmap = {}
    for root, rid, w in zip(z["seed"], z["rid"], z["w"]):
        wmap[(str(root), str(int(rid)))] = float(w)
    roots = sorted({k[0] for k in wmap})
    wr, ww = [], []
    for root in roots:
        arm = load_arm(T / "data" / root)
        for rid, r in arm.items():
            w = wmap.get((root, str(int(rid))))
            if w is not None:
                wr.append(r * BP)
                ww.append(w)
    wr, ww = np.asarray(wr), np.asarray(ww)

    lim = np.quantile(np.abs(real), 0.998) * 1.6
    grid = np.linspace(-lim, lim, 481)
    # Silverman on the real sample, doubled: at n=1984 the raw rule leaves
    # log-scale notches wherever a 0.3 bp window is empty, and the same
    # bandwidth must smooth all three curves for the shapes to be comparable
    bw = 2.0 * 1.06 * real.std() * real.size ** (-1 / 5)

    kr, kb, kf = kde(real, grid, bw), kde(base, grid, bw), kde(ft, grid, bw)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7))

    for ax, kg, col, name, sd in (
            (axes[0], kb, "#4878b0", "base (pretrained)", base.std() / real.std()),
            (axes[1], kf, "#d1495b", "wm_ft_b (mid-trained)", ft.std() / real.std())):
        ax.plot(grid, kr, color="black", lw=1.6, label=f"real  (n={real.size})")
        ax.fill_between(grid, kr, color="black", alpha=0.06)
        ax.plot(grid, kg, color=col, lw=1.6, label=name)
        ax.set_yscale("log")
        ax.set_ylim(max(kr[kr > 0].min() * 0.5, kr.max() * 3e-6), kr.max() * 3)
        ax.set_xlabel("terminal return $r_{250}$ (bp)")
        ax.legend(fontsize=8, loc="upper right")
        ax.text(0.03, 0.95, f"sd ratio {sd:.2f}", transform=ax.transAxes,
                fontsize=9, va="top")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("density (log scale)")
    axes[0].set_title("(a) before: too narrow, tails missing", fontsize=10)
    axes[1].set_title("(b) after: dispersion and tails recovered", fontsize=10)

    ax = axes[2]
    ax.scatter(wr, ww, s=5, alpha=0.18, color="#7a5195", edgecolors="none")
    med, ctr = [], []
    for lo, hi in zip(np.quantile(wr, np.linspace(0, 0.98, 25)),
                      np.quantile(wr, np.linspace(0.02, 1.0, 25))):
        m = (wr >= lo) & (wr <= hi)
        if m.sum() >= 20:
            med.append(np.median(ww[m]))
            ctr.append(np.median(wr[m]))
    ax.plot(ctr, med, color="#7a5195", lw=2.0, label="binned median $w$")
    ax.axhline(1.0, color="black", lw=0.9, ls="--")
    # the base policy's own support barely reaches |r|=2 bp; that narrowness
    # is exactly what w>1 corrects, so frame the panel on the support
    ax.set_xlim(-3.2, 3.2)
    ax.set_yscale("log")
    ax.set_xlabel("rollout terminal return $r_{250}$ (bp)")
    ax.set_ylabel("weight $w=\\hat p_{\\mathrm{real}}/\\hat p_\\theta$")
    ax.set_title("(c) how: reweight the model's own rollouts", fontsize=10)
    ax.text(0.5, 0.06, "$w<1$: overproduced near zero", transform=ax.transAxes,
            ha="center", fontsize=8)
    ax.text(0.03, 0.9, "$w>1$: underproduced\nlarge moves", transform=ax.transAxes,
            fontsize=8)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)

    fig.suptitle("Terminal-return law before and after mid-training "
                 "(GOOG 2026-01, 20 days $\\times$ 1984 contexts $\\times$ 3 seeds)",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for out in (T / "figs" / "fig_midtrain_return_dist.pdf",
                Path("/projects/public/u6gb/overleaf/mid-train-kang-neurips-workshop2026/figures/fig_midtrain_return_dist.pdf"),
                T / "figs" / "fig_midtrain_return_dist_preview.png"):
        fig.savefig(out, dpi=150)
        print("wrote", out)


if __name__ == "__main__":
    main()
