#!/usr/bin/env python3
"""Render every (ticker, horizon) two-part element as its own standalone
figure, exactly in the approved style, then stitch the standalone images into
the 8x3 matrix.  Standalone-first is deliberate: each cell is usable alone in
the paper, margins stay consistent, and the composite is a lossless paste.

The mid-trained red line appears wherever that ticker has trained-arm
rollouts (GOOG today; every ticker once wm_ft_multi evaluation lands and
ARM_DIRS is extended).
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

T = Path("/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z")
OVL = Path("/projects/public/u6gb/overleaf/mid-train-kang-neurips-workshop2026/figures")
CELLS = T / "figs" / "cells"
BP = 1e4
HS = [10, 25, 50, 100, 150, 200, 250]
SHOW = [25, 100, 250]
TICKERS = ["GOOG", "MSFT", "AMZN", "META", "AMD", "INTC", "NFLX", "JPM"]
BLUE, RED, GREEN = "#4878b0", "#d1495b", "#2E7D32"
DPI = 110

# Uniform protocol across every row: real, base and the mid-trained arm all
# on the ticker's held-out evaluation index, so the three curves in a cell
# are paired on identical contexts.
import os

ARM_TAG = os.environ.get("ARM_TAG", "v5mem")
ARM_SEEDS = os.environ.get("ARM_SEEDS", "97101,97102").split(",")
BASE_SEEDS = ("97001", "97002")
ARM_LABEL = os.environ.get("ARM_LABEL", "wm_ft_multi (mid-trained)")


def multih(root, kind):
    z = np.load(T / "data" / root / "member_0" / f".returns_multih_{kind}.npz")
    return z["vals"] * BP


def pools(tk):
    base = np.vstack([multih(f"hp_v5meb_{tk}_s{s}", "gen") for s in BASE_SEEDS])
    real = multih(f"hp_v5meb_{tk}_s{BASE_SEEDS[0]}", "real")
    try:
        ft = np.vstack([multih(f"hp_{ARM_TAG}_{tk}_s{s}", "gen")
                        for s in ARM_SEEDS])
    except FileNotFoundError:
        ft = None
    return real, base, ft


def kde(x, grid, bw):
    d = (grid[:, None] - x[None, :]) / bw
    return np.exp(-0.5 * d * d).sum(axis=1) / (x.size * bw * np.sqrt(2 * np.pi))


def render_cell(tk, h, r, b, f, out_png):
    fig, (top, bot) = plt.subplots(
        2, 1, figsize=(8.2, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.07})
    lim = np.quantile(np.abs(r), 0.998) * 1.4
    grid = np.linspace(-lim, lim, 321)
    bw = 2.0 * 1.06 * r.std() * r.size ** (-1 / 5)
    kr, kb = kde(r, grid, bw), kde(b, grid, bw)
    top.plot(grid, kr, color="black", lw=1.8, label="real")
    top.fill_between(grid, kr, color="black", alpha=0.06)
    top.plot(grid, kb, color=BLUE, lw=1.8, label="base (pretrained)")
    note = f"sd ratio {b.std() / r.std():.2f}"
    floor = kr.max() * 1e-6
    db = np.log10(np.maximum(kb, floor)) - np.log10(np.maximum(kr, floor))
    bot.axhline(0, color="black", lw=1.2, ls="--")
    bot.text(0.02, 0.9, "0 = perfect match", transform=bot.transAxes,
             fontsize=8.5, va="top")
    bot.plot(grid, db, color=BLUE, lw=1.8)
    if f is not None:
        kf = kde(f, grid, bw)
        top.plot(grid, kf, color=RED, lw=1.8, label=ARM_LABEL)
        df = np.log10(np.maximum(kf, floor)) - np.log10(np.maximum(kr, floor))
        bot.plot(grid, df, color=RED, lw=1.8)
        bot.fill_between(grid, db, df, where=np.abs(df) <= np.abs(db),
                         color=GREEN, alpha=0.15, interpolate=True,
                         label="gap recovered")
        bot.legend(fontsize=8.5, loc="lower center")
        note += f" $\\to$ {f.std() / r.std():.2f}"
    top.set_yscale("log")
    top.set_ylim(max(kr[kr > 0].min() * 0.5, kr.max() * 3e-6), kr.max() * 3)
    top.set_ylabel("density (log)")
    top.legend(fontsize=9)
    top.text(0.02, 0.95, note, transform=top.transAxes, fontsize=9.5, va="top")
    top.set_title(f"{tk} return $r_{{{h}}}$: law (top) and gap to real (bottom)",
                  fontsize=11)
    bot.set_ylim(-2.6, 1.2)
    bot.set_ylabel("$\\log_{10}$ density gap")
    bot.set_xlabel(f"return $r_{{{h}}}$ (bp)")
    for ax in (top, bot):
        ax.grid(alpha=0.25)
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    CELLS.mkdir(parents=True, exist_ok=True)
    paths = {}
    for tk in TICKERS:
        real, base, ft = pools(tk)
        for h in SHOW:
            j = HS.index(h)
            r = real[:, j][np.isfinite(real[:, j])]
            b = base[:, j][np.isfinite(base[:, j])]
            f = ft[:, j][np.isfinite(ft[:, j])] if ft is not None else None
            p = CELLS / f"cell_{tk}_h{h}.png"
            render_cell(tk, h, r, b, f, p)
            paths[(tk, h)] = p
            print("cell", tk, h)

    # ---- stitch: rows = tickers, cols = horizons -----------------------
    imgs = {k: Image.open(p) for k, p in paths.items()}
    cw = max(im.width for im in imgs.values())
    ch = max(im.height for im in imgs.values())
    W, H = cw * len(SHOW), ch * len(TICKERS)
    canvas = Image.new("RGB", (W, H), "white")
    for i, tk in enumerate(TICKERS):
        for c, h in enumerate(SHOW):
            im = imgs[(tk, h)]
            canvas.paste(im, (c * cw + (cw - im.width) // 2,
                              i * ch + (ch - im.height) // 2))
    out = T / "figs" / "fig_v5m_matrix.png"
    canvas.save(out)
    canvas.save(OVL / "fig_v5m_matrix.png")
    small = canvas.resize((W // 2, H // 2), Image.LANCZOS)
    small.save(T / "figs" / "fig_v5m_matrix_preview.png")
    print("stitched", out, canvas.size)


if __name__ == "__main__":
    main()
