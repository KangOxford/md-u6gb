#!/usr/bin/env python3
"""The one-glance version of the distribution result.

Two visual idioms that need no statistics to read: a Q-Q panel where the
dashed diagonal is perfection and whichever curve hugs it is better, and a
paired bar panel where every bar is the gap to the real quantile, so shorter
bars mean improvement. Same pooled data and quantile convention as the
reported qL1.
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

BP = 1e4
BLUE, RED = "#4878b0", "#d1495b"


def pooled(roots):
    out = []
    for r in roots:
        out.extend(load_arm(T / "data" / r).values())
    return np.asarray(out) * BP


def main():
    real = np.asarray(list(load_real(T / "data" / "hp_mdbase_s91001").values())) * BP
    base = pooled([f"hp_mdbase_s9100{i}" for i in (1, 2, 3)])
    ft = pooled([f"hp_mdwmftb_s9100{i}" for i in (1, 2, 3)])

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4),
                             gridspec_kw={"width_ratios": [1, 1.05]})
    axes = axes[::-1]           # bars lead on the left, the curve is detail

    # ---- signed gap per quantile: the zero line is perfection ------------
    # Subtracting the diagonal is what makes the verdict visible: on a raw
    # Q-Q both curves sit near the diagonal and the eye cannot rank them.
    # A 2% grid also steps over the price-tick discreteness of r_250, which
    # at a fine grid renders as distracting saw-teeth.
    q = np.linspace(0.02, 0.98, 49)
    qr = np.quantile(real, q)
    dev_b = np.quantile(base, q) - qr
    dev_f = np.quantile(ft, q) - qr
    ax = axes[0]
    ax.axhline(0, color="black", lw=1.6, ls="--")
    ax.text(50, 0.045, "0 = perfect match", ha="center", fontsize=9.5)
    ax.fill_between(q * 100, dev_b, dev_f, color="#2E7D32", alpha=0.15,
                    where=np.abs(dev_f) <= np.abs(dev_b), interpolate=True,
                    label="gap recovered by mid-training")
    ax.plot(q * 100, dev_b, color=BLUE, lw=2.2, label="before (base)")
    ax.plot(q * 100, dev_f, color=RED, lw=2.2, label="after (wm_ft_b)")
    for p in (0.05, 0.95):
        i = np.argmin(np.abs(q - p))
        ax.annotate("", xy=(q[i] * 100, dev_f[i]), xytext=(q[i] * 100, dev_b[i]),
                    arrowprops=dict(arrowstyle="-|>", color="#2E7D32", lw=1.8))
    ax.set_xlabel("quantile of the real return law (%)")
    ax.set_ylabel("generated $-$ real quantile (bp)")
    ax.set_title("(b) signed quantile gap, 0 = perfect", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(alpha=0.25)

    # ---- (b) gap to the real quantile: shorter bar = better --------------
    ax = axes[1]
    ps = [0.01, 0.05, 0.10, 0.25, 0.75, 0.90, 0.95, 0.99]
    labels = ["1%", "5%", "10%", "25%", "75%", "90%", "95%", "99%"]
    gap_b = [abs(np.quantile(base, p) - np.quantile(real, p)) for p in ps]
    gap_f = [abs(np.quantile(ft, p) - np.quantile(real, p)) for p in ps]
    x = np.arange(len(ps))
    ax.bar(x - 0.19, gap_b, 0.38, color=BLUE, label="before (base)")
    ax.bar(x + 0.19, gap_f, 0.38, color=RED, label="after (wm_ft_b)")
    for i, (gb, gf) in enumerate(zip(gap_b, gap_f)):
        if gb > 0:
            ax.text(i, max(gb, gf) + 0.015, f"$-${(1 - gf / gb) * 100:.0f}%",
                    ha="center", fontsize=8.6, color="#2E7D32")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("quantile of the real return law")
    ax.set_ylabel("|generated $-$ real| quantile gap (bp)")
    ax.set_title("(a) gap to the real quantile, shorter = better", fontsize=10.5)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle("Did the generated return distribution get better? "
                 "(GOOG 2026-01, 20 days $\\times$ 1984 contexts $\\times$ 3 seeds)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for out in (T / "figs" / "fig_midtrain_return_qq.pdf",
                Path("/projects/public/u6gb/overleaf/mid-train-kang-neurips-workshop2026/figures/fig_midtrain_return_qq.pdf"),
                T / "figs" / "fig_midtrain_return_qq_preview.png"):
        fig.savefig(out, dpi=150)
        print("wrote", out)


if __name__ == "__main__":
    main()
