#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""valset CE 结果图（图表文字英文，正文中文规范）。
fig1: val CE vs N（terminal，micro+macro，对照 Jan-2026 macro）
fig2: Δ(val_macro − Jan_macro) vs N —— 时移分布漂移的代价
fig3: 132 点全景：CE vs D 按 size 分组（@step 点提供 run 内 D 变异）
用法: python valset_ce_figs.py <results_dir>"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rd = sys.argv[1]
fig_dir = os.path.join(rd, "figures"); os.makedirs(fig_dir, exist_ok=True)
TICKER_NPY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ticker_per_sample_30720.npy")
tickers = np.load(TICKER_NPY)
order = np.argsort(tickers, kind="stable")
st_sorted = tickers[order]
cuts = np.flatnonzero(np.r_[True, st_sorted[1:] != st_sorted[:-1]])
bounds = list(zip(cuts, np.r_[cuts[1:], len(st_sorted)]))

def macro_ce(sample_loss):
    ss = sample_loss[order]
    return float(np.mean([ss[a:b].mean() for a, b in bounds]))

rows = []
for f in sorted(glob.glob(os.path.join(rd, "valce_*.json"))):
    r = json.load(open(f))
    sl = f.replace(".json", "_sampleloss.npy")
    if os.path.exists(sl):
        r["val_ce_macro"] = macro_ce(np.load(sl))
    r["is_terminal"] = "@" not in r["label"]
    rows.append(r)
if not rows:
    sys.exit("no results yet")
print(f"{len(rows)} results ({sum(r['is_terminal'] for r in rows)} terminal)")

SIZES = sorted({r["size"] for r in rows}, key=lambda s: [r["num_params"] for r in rows if r["size"] == s][0])
cmap = plt.get_cmap("viridis")
size_color = {s: cmap(i / max(1, len(SIZES) - 1)) for i, s in enumerate(SIZES)}

# fig1: terminal CE vs N
term = [r for r in rows if r["is_terminal"]]
if term:
    fig, ax = plt.subplots(figsize=(7, 5))
    N = [r["num_params"] for r in term]
    ax.scatter(N, [r["val_ce_mean"] for r in term], marker="o", s=42, label="valset_v1 CE (micro)", zorder=3)
    if all("val_ce_macro" in r for r in term):
        ax.scatter(N, [r["val_ce_macro"] for r in term], marker="s", s=42, label="valset_v1 CE (macro, 487 tickers)", zorder=3)
    ax.scatter(N, [r["jan2026_ce"] for r in term], marker="^", s=42, label="Jan-2026 test CE (macro)", zorder=3)
    ax.set_xscale("log"); ax.set_xlabel("Model parameters N"); ax.set_ylabel("Cross-entropy (nats/token)")
    ax.set_title("Terminal checkpoints: in-distribution valset vs forward-time test CE")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(f"{fig_dir}/fig1_terminal_ce_vs_N.png", dpi=150); plt.close(fig)
    print("fig1 saved")

# fig2: Δ(val macro − Jan macro) vs N
term_m = [r for r in term if "val_ce_macro" in r]
if term_m:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for r in term_m:
        ax.scatter(r["num_params"], r["val_ce_macro"] - r["jan2026_ce"],
                   color=size_color[r["size"]], s=40)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xscale("log"); ax.set_xlabel("Model parameters N")
    ax.set_ylabel("CE difference (nats/token)")
    ax.set_title("Forward-time shift cost: Jan-2026 CE minus in-distribution valset CE (macro)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{fig_dir}/fig2_timeshift_delta_vs_N.png", dpi=150); plt.close(fig)
    print("fig2 saved")

# fig3: CE vs D per size（132 全景）
fig, ax = plt.subplots(figsize=(7.5, 5))
for s in SIZES:
    rs = sorted([r for r in rows if r["size"] == s], key=lambda r: r["D_tokens"])
    for seed in sorted({r["seed"] for r in rs}):
        rr = [r for r in rs if r["seed"] == seed]
        ax.plot([r["D_tokens"] for r in rr], [r["val_ce_mean"] for r in rr],
                marker="o", ms=3.5, lw=0.9, color=size_color[s], alpha=0.85)
    ax.plot([], [], marker="o", ms=4, lw=1, color=size_color[s], label=s)
ax.set_xscale("log"); ax.set_xlabel("Training tokens D"); ax.set_ylabel("valset_v1 CE (micro, nats/token)")
ax.set_title("Final-25% window: valset CE vs D across sizes (run-level D variation)")
ax.grid(alpha=0.3); ax.legend(ncol=3, fontsize=8)
fig.tight_layout(); fig.savefig(f"{fig_dir}/fig3_ce_vs_D_all132.png", dpi=150); plt.close(fig)
print("fig3 saved")
