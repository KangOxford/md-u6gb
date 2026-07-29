#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""valset_v1 报告统计 + matplotlib 图（图内文字全英文，避免 CJK 字体缺失）。

跑在训练 env（torch 2.8 复现排列位置分布）。产出：
  figures/fig1_ticker_representativeness.png
  figures/fig2_monthly_distribution.png
  figures/fig3_seed_positions.png
  figures/fig4_top30_tickers.png
  stats.json
"""
import json, csv
from pathlib import Path
import numpy as np

BASE = Path("/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set")
ART = BASE / "artifacts_valset_v1_j5790795"
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)

# ── palette (validated: dataviz reference, light mode) ──
BLUE, GREEN, MAGENTA = "#2a78d6", "#008300", "#e87ba4"
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
font_manager.fontManager.addfont(str(BASE / "fonts/NotoSansCJKsc-Regular.otf"))
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "sans-serif", "font.size": 10,
})

log = lambda *a: print(*a, flush=True)

# ── load ──
man = json.load(open(ART / "manifest.json"))
N48 = man["domain"]["N48"]
dec = np.load(ART / "val_pool_decode.npz")
V, fidx, flag = dec["global_idx"], dec["file_idx"], dec["flag_v1_8ticker"]
log("V:", len(V))

tk_l, dt_l, seqs_l = [], [], []
with open(ART / "files_48mo.csv") as f:
    for row in csv.DictReader(f):
        tk_l.append(row["ticker"]); dt_l.append(row["date"]); seqs_l.append(int(row["seqs"]))
tk = np.array(tk_l); months = np.array([d[:7] for d in dt_l]); seqs = np.array(seqs_l, dtype=np.int64)
nfiles = len(tk); log("files:", nfiles, "domain seqs sum:", seqs.sum())
assert seqs.sum() == N48

# per-file val counts
val_per_file = np.bincount(fidx, minlength=nfiles)

# ── ticker stats ──
tickers_u, tk_id = np.unique(tk, return_inverse=True)
dom_by_tk = np.bincount(tk_id, weights=seqs).astype(np.int64)
val_by_tk = np.bincount(tk_id, weights=val_per_file).astype(np.int64)
dom_share, val_share = dom_by_tk / N48, val_by_tk / len(V)
covered = int((val_by_tk > 0).sum())
V1SET = set(man["flags"]["v1_8ticker"]["tickers"])
is_v1 = np.array([t in V1SET for t in tickers_u])
order = np.argsort(val_by_tk)[::-1]
goog_i = int(np.where(tickers_u == "GOOG")[0][0])

# GOOG Dec-2025 excision check
goog_dec_files = (tk == "GOOG") & np.char.startswith(months, "2025-12")
assert val_per_file[goog_dec_files].sum() == 0, "GOOG 2025-12 leak!"

# ── monthly stats ──
months_u, m_id = np.unique(months, return_inverse=True)
dom_by_m = np.bincount(m_id, weights=seqs).astype(np.int64)
val_by_m = np.bincount(m_id, weights=val_per_file).astype(np.int64)
ratio_m = val_by_m / dom_by_m

# ── seed position distributions (training-env torch) ──
import torch
pos_stats, hists = {}, {}
edges = np.linspace(0, 1, 101)
for s in [5, 42, 137]:
    g = torch.Generator(); g.manual_seed(s)
    p = torch.randperm(N48, generator=g).numpy()
    inv = np.empty(N48, dtype=np.int64); inv[p] = np.arange(N48)
    pos = inv[V] / N48
    hists[s], _ = np.histogram(pos, bins=edges, density=True)
    pos_stats[s] = dict(min=float(pos.min()), p50=float(np.median(pos)),
                        frac_in_tail=float((pos >= 0.98).mean()))
    del p, inv, pos
    log(f"seed {s}: min pos {pos_stats[s]['min']:.4f}, in-tail {pos_stats[s]['frac_in_tail']:.3f}")
    assert pos_stats[s]["min"] >= 0.20

# ══ fig1: representativeness scatter ══
fig, ax = plt.subplots(figsize=(6.4, 5.4))
lo = min(dom_share[dom_share > 0].min(), val_share[val_share > 0].min()) * 0.7
hi = max(dom_share.max(), val_share.max()) * 1.7
lims = [lo, hi]
ax.plot(lims, lims, ls="--", lw=1, color=MUTED, zorder=1)
m0 = ~is_v1
ax.scatter(dom_share[m0], val_share[m0], s=14, c=BLUE, alpha=0.55, lw=0, zorder=2,
           label=f"ticker ({int(m0.sum())})")
ax.scatter(dom_share[is_v1], val_share[is_v1], s=34, c=GREEN, lw=0, zorder=3,
           label="v1 8-ticker flag")
for i in np.where(is_v1)[0]:
    ax.annotate(tickers_u[i], (dom_share[i], val_share[i]), textcoords="offset points",
                xytext=(5, 4), fontsize=7.5, color=INK2)
ax.annotate("GOOG below diagonal:\n(GOOG, 2025-12) excised", (dom_share[goog_i], val_share[goog_i]),
            textcoords="offset points", xytext=(14, -26), fontsize=8, color=INK2,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("ticker share of training domain (log)")
ax.set_ylabel("ticker share of valset_v1 (log)")
ax.set_title("valset_v1 preserves the activity-weighted ticker distribution", color=INK)
ax.legend(frameon=False, loc="upper left", fontsize=8.5)
fig.tight_layout(); fig.savefig(FIG / "fig1_ticker_representativeness.png", dpi=150); plt.close(fig)

# ══ fig2: monthly distribution + coverage ratio ══
x = np.arange(len(months_u))
fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.6, 6.2), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.25, 1]))
a1.plot(x, dom_by_m / N48 * 100, color=BLUE, lw=2, label="training domain")
a1.plot(x, val_by_m / len(V) * 100, color=GREEN, lw=2, label="valset_v1")
a1.set_ylabel("share of its own total (%)")
a1.set_title("Monthly distribution: valset_v1 vs training domain", color=INK)
a1.legend(frameon=False, fontsize=8.5)
a2.plot(x, ratio_m * 100, color=BLUE, lw=2)
a2.axvline(11.5, color=MUTED, ls="--", lw=1)
a2.annotate("2023-01 onward: the 36-month domain's\nthree 20% zones also excluded", (12.2, ratio_m[:12].mean() * 100 * 0.93),
            fontsize=8.5, color=INK2)
i_goog = int(np.where(months_u == "2025-12")[0][0])
a2.annotate("2025-12 dip:\n(GOOG, 2025-12) excised", (i_goog - 8.5, ratio_m[i_goog] * 100 - 0.05),
            fontsize=8, color=INK2)
a2.set_ylabel("coverage ratio (% of month kept)")
a2.set_xticks(x[::6]); a2.set_xticklabels(months_u[::6], rotation=0, fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "fig2_monthly_distribution.png", dpi=150); plt.close(fig)

# ══ fig3: seed position histograms ══
fig, ax = plt.subplots(figsize=(8.8, 4.6))
centers = (edges[:-1] + edges[1:]) / 2
cols = {5: BLUE, 42: GREEN, 137: MAGENTA}
for s in [5, 42, 137]:
    ax.step(centers, hists[s], where="mid", color=cols[s], lw=2, label=f"seed {s}")
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.235, 0.80), fontsize=9)
spike = hists[5][-1]
ax.annotate(f"spike density ≈ {spike:.1f} (34.2% of V per seed)", (0.99, spike),
            xytext=(0.62, spike * 0.45), fontsize=8.5, color=INK2,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.text(0.42, 0.30, "3 curves nearly coincide:\nseeds are interchangeable by construction",
        fontsize=8.5, color=INK2, transform=ax.transAxes)
ax.axvspan(0, 0.20, color=GRID, alpha=0.55, lw=0)
ax.axvline(0.20, color=MUTED, ls="--", lw=1)
ax.axvline(0.98, color=MUTED, ls=":", lw=1)
ax.set_yscale("log")
ax.text(0.10, 0.5, "20% exclusion zone\n(0 samples)", ha="center", fontsize=8.5, color=INK2,
        transform=ax.get_xaxis_transform())
ax.text(0.60, 0.88, "uniform background: positions in the\nOTHER seeds' permutations", ha="center",
        fontsize=8.5, color=INK2, transform=ax.transAxes)
ax.set_xlim(0, 1.04); ax.set_xlabel("position of val sample in perm_s (fraction of N)")
ax.set_ylabel("density (log)")
ax.set_title("Every val sample sits past 20% in every seed's permutation (zero-leakage, per-sample)", color=INK)
fig.tight_layout(); fig.savefig(FIG / "fig3_seed_positions.png", dpi=150); plt.close(fig)

# ══ fig4: top-30 tickers ══
top30 = order[:30]
fig, ax = plt.subplots(figsize=(7.6, 7.6))
y = np.arange(len(top30))[::-1]
colors = [GREEN if is_v1[i] else BLUE for i in top30]
ax.barh(y, val_by_tk[top30], color=colors, height=0.72)
ax.set_yticks(y); ax.set_yticklabels(tickers_u[top30], fontsize=8.5)
for r, i in enumerate(top30):
    if r < 3 or tickers_u[i] == "GOOG":
        ax.annotate(f"{val_by_tk[i]:,}", (val_by_tk[i], y[r]), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK2)
import matplotlib.patches as mpatches
ax.legend(handles=[mpatches.Patch(color=BLUE, label="ticker"),
                   mpatches.Patch(color=GREEN, label="v1 8-ticker flag")],
          frameon=False, fontsize=8.5, loc="lower right")
ax.set_xlabel("valset_v1 samples")
ax.set_title("Top-30 tickers by valset_v1 sample count", color=INK)
ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(FIG / "fig4_top30_tickers.png", dpi=150); plt.close(fig)

# ── stats.json ──
top10 = [dict(ticker=str(tickers_u[i]), val_n=int(val_by_tk[i]),
              val_share=round(float(val_share[i]) * 100, 3),
              dom_share=round(float(dom_share[i]) * 100, 3),
              v1_flag=bool(is_v1[i])) for i in order[:10]]
corr = float(np.corrcoef(dom_share, val_share)[0, 1])
stats = dict(
    V=int(len(V)), N48=int(N48), pct=round(len(V) / N48 * 100, 4),
    messages=int(len(V)) * 500, tokens=int(len(V)) * 13000,
    tickers_covered=f"{covered}/{len(tickers_u)}",
    ticker_share_corr=round(corr, 6),
    top10=top10,
    top10_share_val=round(float(val_share[order[:10]].sum()) * 100, 2),
    top10_share_dom=round(float(np.sort(dom_share)[::-1][:10].sum()) * 100, 2),
    flag_v1_total=int(flag.sum()), flag_v1_pct=round(float(flag.mean()) * 100, 2),
    goog=dict(val_n=int(val_by_tk[goog_i]), dec2025_val=0,
              dom_share=round(float(dom_share[goog_i]) * 100, 3),
              val_share=round(float(val_share[goog_i]) * 100, 3)),
    files_domain=int(nfiles), files_touched_by_val=int((val_per_file > 0).sum()),
    windows_per_file_p=dict(p10=int(np.percentile(seqs, 10)), p50=int(np.percentile(seqs, 50)),
                            p90=int(np.percentile(seqs, 90)), max=int(seqs.max())),
    val_per_file_p=dict(p50=float(np.percentile(val_per_file, 50)),
                        p90=float(np.percentile(val_per_file, 90)), max=int(val_per_file.max())),
    monthly=[dict(month=str(m), dom=int(d), val=int(v), ratio_pct=round(float(r) * 100, 3))
             for m, d, v, r in zip(months_u, dom_by_m, val_by_m, ratio_m)],
    yearly_ratio_pct={y: round(float(val_by_m[np.char.startswith(months_u, y)].sum() /
                                     dom_by_m[np.char.startswith(months_u, y)].sum()) * 100, 3)
                      for y in ["2022", "2023", "2024", "2025"]},
    seed_positions=pos_stats,
)
json.dump(stats, open(BASE / "stats_valset_v1.json", "w"), indent=1, ensure_ascii=False)
log(json.dumps({k: v for k, v in stats.items() if k not in ("monthly", "top10")}, indent=1)[:1500])
log("FIGS_OK")
