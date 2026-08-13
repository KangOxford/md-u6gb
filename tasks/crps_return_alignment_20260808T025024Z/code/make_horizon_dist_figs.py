#!/usr/bin/env python3
"""Return-distribution figures across horizons and tickers, from caches only.

Figure 1: the GOOG law at six horizons, real vs pretrained vs mid-trained --
the distribution-level companion of the horizon-profile curves.
Figure 2: the terminal law for all eight tickers, real vs pretrained, from the
v5m training material (600 real + 3600 generated per ticker).  Regenerated
with the multi-ticker arm once its checkpoint lands.
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
BLUE, RED = "#4878b0", "#d1495b"
TICKERS = ["GOOG", "MSFT", "AMZN", "META", "AMD", "INTC", "NFLX", "JPM"]


def multih(root, kind):
    z = np.load(T / "data" / root / "member_0" / f".returns_multih_{kind}.npz",
                allow_pickle=False)
    return z["vals"] * BP                     # (n, len(HS))


def kde(x, grid, bw):
    d = (grid[:, None] - x[None, :]) / bw
    return np.exp(-0.5 * d * d).sum(axis=1) / (x.size * bw * np.sqrt(2 * np.pi))


def panel(ax, real, base, ft=None, title=""):
    real = real[np.isfinite(real)]
    base = base[np.isfinite(base)]
    lim = np.quantile(np.abs(real), 0.998) * 1.5
    grid = np.linspace(-lim, lim, 301)
    bw = 2.0 * 1.06 * real.std() * real.size ** (-1 / 5)
    kr = kde(real, grid, bw)
    ax.plot(grid, kr, color="black", lw=1.5, label="real")
    ax.fill_between(grid, kr, color="black", alpha=0.06)
    ax.plot(grid, kde(base, grid, bw), color=BLUE, lw=1.5, label="base")
    note = f"sd {base.std() / real.std():.2f}"
    if ft is not None:
        ft = ft[np.isfinite(ft)]
        ax.plot(grid, kde(ft, grid, bw), color=RED, lw=1.5, label="mid-trained")
        note += f" $\\to$ {ft.std() / real.std():.2f}"
    ax.set_yscale("log")
    ax.set_ylim(max(kr[kr > 0].min() * 0.5, kr.max() * 3e-6), kr.max() * 3)
    ax.set_title(title, fontsize=9.5)
    ax.text(0.03, 0.95, note, transform=ax.transAxes, fontsize=8.5, va="top")
    ax.grid(alpha=0.25)


def fig_horizons():
    real = multih("hp_mdbase_s91001", "real")
    base = np.vstack([multih(f"hp_mdbase_s9100{i}", "gen") for i in (1, 2, 3)])
    ft = np.vstack([multih(f"hp_mdwmftb_s9100{i}", "gen") for i in (1, 2, 3)])
    show = [10, 25, 50, 100, 150, 250]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.0))
    for ax, h in zip(axes.flat, show):
        j = HS.index(h)
        panel(ax, real[:, j], base[:, j], ft[:, j],
              title=f"$h={h}$ messages")
    for ax in axes[:, 0]:
        ax.set_ylabel("density (log)")
    for ax in axes[1]:
        ax.set_xlabel("return $r_h$ (bp)")
    axes[0, 0].legend(fontsize=8.5, loc="upper right")
    fig.suptitle("Return law by horizon: real vs pretrained vs mid-trained "
                 "(GOOG 2026-01, 20 days $\\times$ 1984 contexts $\\times$ 3 seeds)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for out in (T / "figs" / "fig_v5m_horizon_dists.pdf",
                OVL / "fig_v5m_horizon_dists.pdf",
                T / "figs" / "fig_v5m_horizon_dists_preview.png"):
        fig.savefig(out, dpi=140)
        print("wrote", out)


def load_terminal(seed):
    m = T / "data" / f"v5m_train_s{seed}" / "member_0"
    g = np.load(m / ".returns_gen.npz", allow_pickle=False)["vals"] * BP
    r = np.load(m / ".returns_real.npz", allow_pickle=False)["vals"] * BP
    return g, r


def fig_tickers():
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 6.6))
    for t, (tk, ax) in enumerate(zip(TICKERS, axes.flat)):
        gs, rs = [], None
        for k in range(6):
            g, r = load_terminal(96000 + t * 10 + k)
            gs.append(g)
            rs = r if rs is None else rs
        panel(ax, rs, np.concatenate(gs),
              title=f"{tk}  (real n={rs.size}, gen n={sum(map(len, gs))})")
    for ax in axes[:, 0]:
        ax.set_ylabel("density (log)")
    for ax in axes[1]:
        ax.set_xlabel("terminal return $r_{250}$ (bp)")
    axes[0, 0].legend(fontsize=8.5, loc="upper right")
    fig.suptitle("Terminal return law per ticker: real vs pretrained "
                 "(2026-01, 600 contexts $\\times$ 6 seeds each)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for out in (T / "figs" / "fig_v5m_ticker_dists.pdf",
                OVL / "fig_v5m_ticker_dists.pdf",
                T / "figs" / "fig_v5m_ticker_dists_preview.png"):
        fig.savefig(out, dpi=140)
        print("wrote", out)


if __name__ == "__main__":
    fig_horizons()
    fig_tickers()
