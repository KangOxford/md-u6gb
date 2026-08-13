#!/usr/bin/env python3
"""The wide-format distribution atlas: every ticker at three horizons, plus
the per-ticker dispersion-decay profile that summarises it in one panel.

Base policy vs real only for now; the multi-ticker arm joins each panel once
its checkpoint and evaluation rollouts exist.
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

T = Path("/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z")
OVL = Path("/projects/public/u6gb/overleaf/mid-train-kang-neurips-workshop2026/figures")
BP = 1e4
HS = [10, 25, 50, 100, 150, 200, 250]
SHOW = [25, 100, 250]
TICKERS = ["GOOG", "MSFT", "AMZN", "META", "AMD", "INTC", "NFLX", "JPM"]
BLUE = "#4878b0"


def multih(seed, kind):
    z = np.load(T / "data" / f"v5m_train_s{seed}" / "member_0"
                / f".returns_multih_{kind}.npz", allow_pickle=False)
    return z["vals"] * BP


def ticker_pools(t):
    gen = np.vstack([multih(96000 + t * 10 + k, "gen") for k in range(6)])
    real = multih(96000 + t * 10 + 0, "real")
    return real, gen


def kde(x, grid, bw):
    d = (grid[:, None] - x[None, :]) / bw
    return np.exp(-0.5 * d * d).sum(axis=1) / (x.size * bw * np.sqrt(2 * np.pi))


def main():
    pools = [ticker_pools(t) for t in range(8)]

    # ---- atlas: 8 tickers x 3 horizons --------------------------------
    fig, axes = plt.subplots(8, 3, figsize=(11.5, 19.5))
    for row, (tk, (real, gen)) in enumerate(zip(TICKERS, pools)):
        for col, h in enumerate(SHOW):
            j = HS.index(h)
            r = real[:, j][np.isfinite(real[:, j])]
            g = gen[:, j][np.isfinite(gen[:, j])]
            ax = axes[row, col]
            lim = np.quantile(np.abs(r), 0.998) * 1.5
            grid = np.linspace(-lim, lim, 241)
            bw = 2.0 * 1.06 * r.std() * r.size ** (-1 / 5)
            kr = kde(r, grid, bw)
            ax.plot(grid, kr, color="black", lw=1.3)
            ax.fill_between(grid, kr, color="black", alpha=0.06)
            ax.plot(grid, kde(g, grid, bw), color=BLUE, lw=1.3)
            ax.set_yscale("log")
            ax.set_ylim(max(kr[kr > 0].min() * 0.5, kr.max() * 1e-5),
                        kr.max() * 3)
            ax.text(0.03, 0.93, f"sd {g.std() / r.std():.2f}",
                    transform=ax.transAxes, fontsize=8, va="top")
            if row == 0:
                ax.set_title(f"$h={h}$", fontsize=10)
            if col == 0:
                ax.set_ylabel(tk, fontsize=10)
            ax.set_yticklabels([])
            ax.grid(alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("$r_h$ (bp)")
    fig.suptitle("Return-law atlas: real (black) vs pretrained (blue), "
                 "8 tickers $\\times$ 3 horizons (2026-01)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for out in (T / "figs" / "fig_v5m_atlas.pdf", OVL / "fig_v5m_atlas.pdf",
                T / "figs" / "fig_v5m_atlas_preview.png"):
        fig.savefig(out, dpi=120)
        print("wrote", out)

    # ---- dispersion decay per ticker ----------------------------------
    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    cmap = plt.get_cmap("tab10")
    for t, (tk, (real, gen)) in enumerate(zip(TICKERS, pools)):
        sd, t99 = [], []
        for j, h in enumerate(HS):
            r = real[:, j][np.isfinite(real[:, j])]
            g = gen[:, j][np.isfinite(gen[:, j])]
            sd.append(g.std() / r.std())
            tau = np.quantile(np.abs(r), 0.99)
            pr = max((np.abs(r) > tau).mean(), 1e-12)
            t99.append((np.abs(g) > tau).mean() / pr)
        a1.plot(HS, sd, "o-", color=cmap(t), label=tk, lw=1.6, ms=3.5)
        a2.plot(HS, t99, "o-", color=cmap(t), lw=1.6, ms=3.5)
    for ax, title in ((a1, "sd ratio vs horizon (1 = match)"),
                      (a2, "tail mass at real $q_{99}$ (1 = match)")):
        ax.axhline(1.0, color="green", ls="--", lw=1.1)
        ax.set_xscale("log")
        ax.set_xticks(HS)
        ax.set_xticklabels(map(str, HS), fontsize=8)
        ax.minorticks_off()
        ax.set_xlabel("horizon $h$ (messages)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
    a1.legend(fontsize=7.5, ncol=2)
    fig2.suptitle("Pretrained dispersion deficit by ticker and horizon",
                  fontsize=11)
    fig2.tight_layout(rect=(0, 0, 1, 0.93))
    for out in (T / "figs" / "fig_v5m_decay.pdf", OVL / "fig_v5m_decay.pdf",
                T / "figs" / "fig_v5m_decay_preview.png"):
        fig2.savefig(out, dpi=140)
        print("wrote", out)


if __name__ == "__main__":
    main()
