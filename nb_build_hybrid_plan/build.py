#!/usr/bin/env python3
"""Assemble the long-context hybrid training-plan notebook.

Kept on Lustre (not in the session scratchpad) so a node change does not eat it.
Writes   notebooks/hybrid_longctx_plan/hybrid_longctx_8stock_training_plan.ipynb
Reads    notebooks/hybrid_longctx_plan/data/{measurements,dataset_stats_8stock}.json
"""
import json, os
import nbformat as nbf

ROOT = "/lus/lfs1aip2/projects/public/u6gb/notebooks/hybrid_longctx_plan"
OUT = os.path.join(ROOT, "hybrid_longctx_8stock_training_plan.ipynb")

cells = []
def md(t):   cells.append(nbf.v4.new_markdown_cell(t.strip("\n")))
def code(t): cells.append(nbf.v4.new_code_cell(t.strip("\n")))

# ─────────────────────────────────────────────────────────────── preamble ────
md(r"""
# A long-context hybrid baseline on eight stocks

**This is a plan, not a result.** It fixes four things before any GPU time is spent:
how long the context can actually be, what the eight-stock corpus contains, what a
0.1% validation set should be, and what the compute efficiency really is.

Every number below is either measured on this cluster (the source file is named) or
derived in a visible cell from numbers that were. Projections are labelled as
projections.
""")

code(r"""
import json, math, os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from PIL import Image
import io, base64
from IPython.display import display, HTML, Markdown

mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "font.size": 8.5,
    "axes.titlesize": 9.5, "axes.labelsize": 8.5, "legend.fontsize": 7.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

C = {"base": "#3b6ea5", "hyb": "#c8532a", "warn": "#b3242b", "ok": "#2e7d5b",
     "grey": "#6b6b6b", "light": "#c9d4e0", "accent": "#8a6bbf", "sand": "#c9a227"}

def show(fig, name):
    # Render at 300 dpi, then quantise to a 128-colour palette so the committed
    # notebook still renders inline on GitHub (the size cap is ~1 MB for the file).
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    im = Image.open(buf).convert("RGB").quantize(colors=128, method=Image.MEDIANCUT)
    out = io.BytesIO(); im.save(out, format="PNG", optimize=True)
    b = out.getvalue()
    display(HTML(f'<img src="data:image/png;base64,{base64.b64encode(b).decode()}" '
                 f'style="max-width:100%"><div style="font-size:11px;color:#666">'
                 f'{name} &middot; {len(b)/1024:.0f} KB</div>'))

ROOT = "/lus/lfs1aip2/projects/public/u6gb/notebooks/hybrid_longctx_plan"
M = json.load(open(os.path.join(ROOT, "data", "measurements.json")))
S = json.load(open(os.path.join(ROOT, "data", "dataset_stats_8stock.json")))
TICKERS = M["recipe"]["data"]["tickers"]
print("measurements sections:", ", ".join(k for k in M if not k.startswith("_")))
print("corpus files:", f'{S["files"]:,}', " messages:", f'{S["messages"]:,}')
""")

# ────────────────────────────────────────────────── 1. why long context ────
md(r"""
## 1. Why the context length is the whole point

The hybrid is a Mamba-3 trunk with **one** global-attention layer at position 3. At a
500-message window it *loses* to the pure Mamba-3 baseline. At 2,000 messages it wins,
and the advantage grows monotonically with how far back the referenced order was.
That monotone shape is the reason to spend compute here: a single number can be a
fluctuation, a monotone curve across 21 points of spread cannot.
""")

code(r"""
v = M["ctx2k_verdict"]["reference_recall_by_age"]
lb = M["ctx2k_verdict"]["lobbench"]
buckets = v["bucket"][:-1]                      # drop 'before-window' from the age axis
b, h, n = np.array(v["baseline"][:-1]), np.array(v["hybrid"][:-1]), np.array(v["n"][:-1])
x = np.arange(len(buckets))

fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.85), layout="constrained",
                       gridspec_kw={"width_ratios": [2.1, 1.25, 1.05]})

a = ax[0]
a.plot(x, b, "o-", color=C["base"], lw=1.6, ms=4, label="pure Mamba-3")
a.plot(x, h, "s-", color=C["hyb"],  lw=1.6, ms=4, label="hybrid (+1 attention layer)")
for i in range(len(x)):
    a.plot([x[i], x[i]], [b[i], h[i]], color=C["grey"], lw=0.7, alpha=0.55, zorder=0)
a.annotate(f"+{h[-1]-b[-1]:.1f} pp", (x[-1], (b[-1]+h[-1])/2), xytext=(-4, 0),
           textcoords="offset points", ha="right", va="center", fontsize=8,
           color=C["hyb"], fontweight="bold")
a.annotate(f"+{h[0]-b[0]:.1f} pp", (x[0], h[0]+3), fontsize=7.5, color=C["hyb"])
a.set_xticks(x); a.set_xticklabels(buckets, rotation=30, ha="right")
a.set_xlabel("age of the referenced order (messages back)")
a.set_ylabel("exact reference recall (%)"); a.set_ylim(0, 104)
a.legend(loc="lower left", frameon=False)
a.set_title("Advantage grows monotonically with distance", loc="left")

a = ax[1]
d = h - b
a.bar(x, d, color=C["hyb"], alpha=0.85, width=0.68)
a.set_xticks(x); a.set_xticklabels(buckets, rotation=30, ha="right")
a.set_ylabel("hybrid $-$ baseline (pp)")
a.set_title("Monotone, +3.7 to +24.7 pp", loc="left")

a = ax[2]
mets = lb["metric"]; xb = np.arange(len(mets)); w = 0.36
a.bar(xb - w/2, lb["baseline"], w, color=C["base"], label="baseline")
a.bar(xb + w/2, lb["hybrid"],   w, color=C["hyb"],  label="hybrid")
a.set_xticks(xb); a.set_xticklabels([m.replace("-21", "") for m in mets])
a.set_ylabel("distance (lower is better)")
a.set_title("LOB-Bench: same sign, narrower", loc="left")
a.legend(frameon=False)

fig.suptitle("Figure 1 — measured at 2,000-message context, step 6265 vs 6258, "
             "3,136 frozen GOOG Jan-2026 sequences", y=1.06, fontsize=8.5, color="#444")
show(fig, "Figure 1")
""")

md(r"""
*Reading: the reason to build this baseline is the middle panel. Nothing else in the
project produces a monotone 21-point spread from a single architectural change.*
""")

# ───────────────────────────────────────────── 2. how long can it be ────
md(r"""
## 2. How long can the context actually be?

Two separate walls, and they sit in different places.

**Generation is not the binding one.** A 4,000-message forward pass is 21.15 GB, a
quarter of one GH200. **Training is.** Backward memory grows linearly at
35.24 GB per thousand messages, and the fit was built on three clean points
(500 / 1,000 / 1,500) with a measured OOM at 1,750 confirming it.
""")

code(r"""
ms = M["memory_scaling"]
fw, fb = ms["forward_only"], ms["forward_backward"]
HBM, fit = ms["hbm_gb"], ms["linear_fit_gb"]
peak = lambda n: fit["intercept"] + fit["slope_per_message"] * np.asarray(n, float)

fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.3), layout="constrained",
                       gridspec_kw={"width_ratios": [1.42, 1]})

a = ax[0]
nn = np.linspace(400, 4200, 200)
a.plot(nn, peak(nn), "--", color=C["hyb"], lw=1.2, alpha=0.8,
       label=f"fit  {fit['intercept']:.2f} + {fit['slope_per_message']:.5f}·n")
a.plot(fb["n_messages"], fb["peak_gb"], "s-", color=C["hyb"], lw=1.7, ms=6,
       label="training (fwd + bwd), measured")
a.plot(fw["n_messages"], fw["peak_gb"], "o-", color=C["base"], lw=1.7, ms=5,
       label="generation (fwd only), measured")
a.set_ylim(0, 168); a.set_xlim(300, 4400)

a.axhline(HBM, color=C["warn"], lw=1.2)
a.text(4380, HBM + 3.5, f"one GH200 = {HBM} GB", ha="right", color=C["warn"], fontsize=7.6)
a.axhline(0.85 * HBM, color=C["warn"], lw=0.9, ls=":")
a.text(4380, 0.85*HBM - 8, "MEM_FRACTION 0.85 = 72.7 GB", ha="right", color=C["warn"], fontsize=7.2)

a.plot([ms["oom_at_n_messages"]], [ms["oom_cap_gb"]], "X", color=C["warn"], ms=10, zorder=5)
a.annotate("OOM at 1,750", (1750, 63), xytext=(-10, -26), textcoords="offset points",
           ha="right", fontsize=7.4, color=C["warn"],
           arrowprops=dict(arrowstyle="-", color=C["warn"], lw=0.6))

n_wall = (0.85*HBM - fit["intercept"]) / fit["slope_per_message"]
a.axvline(n_wall, color=C["grey"], lw=0.9, ls="-.")
a.text(n_wall - 55, 150, f"wall  {n_wall:,.0f} msgs", rotation=90, ha="right", va="top",
       fontsize=7.2, color=C["grey"])

a.plot([2000], [peak(2000)], "*", color=C["ok"], ms=16, zorder=6)
a.annotate(f"2,000 msgs = 52,000 tok\n{peak(2000):.1f} GB = 82.9%",
           (2000, peak(2000)), xytext=(24, -34), textcoords="offset points",
           fontsize=7.6, color=C["ok"], fontweight="bold",
           arrowprops=dict(arrowstyle="-", color=C["ok"], lw=0.6))
a.text(420, 152, f"4,000 msgs: {peak(4000):.0f} GB = 1.66x the card",
       ha="left", fontsize=8.0, color=C["warn"], fontweight="bold")
a.set_xlabel("context (messages, 26 tokens each)"); a.set_ylabel("peak HBM (GB)")
a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3,
         frameon=False, fontsize=7.2)
a.set_title("Training memory is the wall; generation is not", loc="left")

a = ax[1]
r = M["same_work_different_cut"]["rows"]
lbl = [f'{x["micro_bsz"]} x {x["n_messages"]}' for x in r]
xb = np.arange(len(r))
a.bar(xb - 0.19, [x["peak_gb"] for x in r], 0.36, color=C["hyb"], label="peak HBM")
a2 = a.twinx(); a2.grid(False)
a2.bar(xb + 0.19, [x["s_per_step"] for x in r], 0.36, color=C["base"], label="s / step")
for i, x in enumerate(r):
    a.text(i-0.19, x["peak_gb"]+0.5, f'{x["peak_gb"]:.2f}', ha="center", fontsize=7.2, color=C["hyb"])
    a2.text(i+0.19, x["s_per_step"]+0.012, f'{x["s_per_step"]:.3f}', ha="center", fontsize=7.2, color=C["base"])
a.set_xticks(xb); a.set_xticklabels(lbl)
a.set_xlabel("micro-batch x messages per sequence")
a.set_ylabel("peak HBM (GB)", color=C["hyb"]); a2.set_ylabel("s / step", color=C["base"])
a.set_ylim(30, 38.5); a2.set_ylim(0, 0.78)
a.set_title("1,000 messages per GPU in every bar", loc="left")

fig.suptitle("Figure 2 — pure Mamba-3, d_model 640, one GH200, bsz 1, remat off "
             "(results/CTX_4K_FEASIBILITY.md)", y=1.03, fontsize=8.5, color="#444")
show(fig, "Figure 2")

print(f"arithmetic wall at MEM_FRACTION 0.85 : {n_wall:,.0f} messages "
      f"({n_wall*26:,.0f} tokens)")
print(f"projected peak at 2,000 messages     : {peak(2000):.1f} GB "
      f"({peak(2000)/HBM*100:.1f}% of the card)")
print(f"projected peak at 4,000 messages     : {peak(4000):.1f} GB linear, "
      f"{peak(4000)*ms['sublinear_correction_at_4000']:.0f} GB with the measured sublinear correction")
""")

md(r"""
The right panel is the load-bearing one for the design. Four windows of 500 and one
window of 2,000 cost **the same memory** (35.79 vs 35.71 GB) because a recurrent model's
activations depend on the number of messages, not on how they are grouped. So moving
from `bsz 4 × 500` to `bsz 1 × 2000` changes the context length **with no change in
memory, tokens per step, or step count** — the cleanest controlled comparison available.

**Answer: 2,000 messages / 52,000 tokens is the longest context this model can train at
today.** The fit puts the wall at 2,050 messages under the `MEM_FRACTION=0.85` in use,
and at 2,410 if the whole card were given to XLA. Going to 4,000 is not a setting, it is one of three
engineering changes:

| Route to 4,000 messages | What it costs | Risk |
|---|---|---|
| Wire `nn.remat` into the Mamba-3 layer | a few lines, plus a `remat` argument on `init_Mamba3SSM` | `REMAT=1` is **silently a no-op today** — it is only passed to attention-class factories, never to the recurrent trunk. Re-measure before believing any number |
| Sequence parallelism across the 4 GPUs | new collective in the scan | ~4× headroom, largest win, largest amount of new code |
| `d_state` 128 → 64, or bf16 activations | env-level | both change the model or its numerics, so the result stops being comparable to every existing checkpoint |

Recommendation: **train at 2,000 now**, and treat 4,000 as a separate follow-on that
starts with the remat wiring plus a fresh memory sweep.
""")

# ─────────────────────────────────────────────────── 3. the corpus ────
md(r"""
## 3. What the eight-stock corpus actually contains

Counted today by extracting `index.json` out of all 48 SquashFS shards and applying the
dataloader's own window rule
(`seqs_per_file = max((usable_rows − (L−1)) // L, 0)`, non-overlapping, random offset on).
Script: `data/count_windows_from_shard_indices.py`.

Two independent checks confirm the count: the 500-message total divided by the 4-node
global micro-batch of 64 gives 939,148 against the 939,147 printed by tqdm in the
baseline log, and the 2,000-message total divided by the 2-node micro-batch of 8 gives
1,877,169 against the 1,877,168 printed in the ctx2k logs.
""")

code(r"""
per_tk_msg = S["per_ticker_messages"]; per_tk_f = S["per_ticker_files"]
months = sorted(S["per_month_win500"])
wm = np.array([S["per_month_win500"][m] for m in months], float)

fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.0), layout="constrained",
                       gridspec_kw={"width_ratios": [1.1, 1.0, 1.55]})

a = ax[0]
vals = np.array([per_tk_msg[t] for t in TICKERS]) / 1e9
order = np.argsort(-vals)
a.barh(np.arange(len(TICKERS)), vals[order], color=C["base"], alpha=0.9)
a.set_yticks(np.arange(len(TICKERS)))
a.set_yticklabels([TICKERS[i] for i in order]); a.invert_yaxis()
for i, v in enumerate(vals[order]):
    a.text(v + 0.06, i, f"{v:.2f}", va="center", fontsize=7.4, color=C["base"])
a.set_xlabel("messages (billions)"); a.set_xlim(0, 6.1)
a.set_title("META is 3.7× lighter than TSLA", loc="left")

a = ax[1]
Ls = [500, 1000, 2000, 4000]
w = [S["windows"][str(L)] for L in Ls]
a.loglog(Ls, w, "o-", color=C["hyb"], lw=1.6, ms=6)
a.loglog(Ls, [x*0.001 for x in w], "s--", color=C["ok"], lw=1.4, ms=5)
for j, (L, y) in enumerate(zip(Ls, w)):
    ha = "right" if j == len(Ls)-1 else "left"
    dx = -4 if ha == "right" else 4
    a.annotate(f"{y/1e6:.1f}M", (L, y), xytext=(dx, 6), textcoords="offset points",
               fontsize=7.2, color=C["hyb"], ha=ha)
    a.annotate(f"{y*0.001:,.0f}", (L, y*0.001), xytext=(dx, -12), textcoords="offset points",
               fontsize=7.2, color=C["ok"], ha=ha)
a.set_xlabel("context L (messages)"); a.set_ylabel("windows in the corpus")
a.set_xlim(420, 5200); a.set_ylim(4e3, 2.2e8)
a.xaxis.set_minor_locator(mpl.ticker.NullLocator())
a.set_xticks(Ls); a.set_xticklabels([str(L) for L in Ls])
a.text(0.03, 0.06, "green = 0.1% of it", transform=a.transAxes, fontsize=7.5, color=C["ok"])
a.set_title("Windows tile exactly", loc="left")

a = ax[2]
xi = np.arange(len(months))
a.bar(xi, wm/1e3, color=C["light"], width=0.85)
a.set_ylim(0, wm.max()/1e3*1.18)
yr_edges = [i for i, m in enumerate(months) if m.endswith("-01")]
for e in yr_edges[1:]:
    a.axvline(e-0.5, color=C["grey"], lw=0.7, alpha=0.6)
for e in yr_edges:
    a.text(e+5.5, wm.max()/1e3*1.08, months[e][:4], ha="center", fontsize=8, color=C["grey"])
a.plot(xi, np.convolve(wm/1e3, np.ones(6)/6, mode="same"), color=C["hyb"], lw=1.4)
a.set_xticks(xi[::6]); a.set_xticklabels([months[i] for i in range(0, len(months), 6)],
                                          rotation=35, ha="right", fontsize=7)
a.set_ylabel("windows @ L=500 (thousands)")
a.set_title("Volume nearly triples over the period", loc="left")

fig.suptitle("Figure 3 — eight tickers, 48 months, counted from the shard indices "
             "(this notebook, today)", y=1.05, fontsize=8.5, color="#444")
show(fig, "Figure 3")
""")

code(r"""
rows = []
for L in [500, 1000, 2000, 4000]:
    w = S["windows"][str(L)]
    rows.append((L, L*26, f"{w:,}", f"{S['win_files'][str(L)]:,}", f"{round(w*0.001):,}"))
hdr = "| context L (messages) | tokens / window | windows | files contributing | 0.1% of windows |\n|---:|---:|---:|---:|---:|\n"
tbl = hdr + "\n".join(f"| {a:,} | {b:,} | {c} | {d} | {e} |" for a,b,c,d,e in rows)

summary = f'''
**Table 1 — the corpus**

| quantity | value |
|---|---:|
| tickers | {', '.join(TICKERS)} |
| months | 48 (2022-01 … 2025-12) |
| ticker-days (files) | {S['files']:,} |
| messages | {S['messages']:,} |
| tokens at 26 tok/message | {S['messages']*26:,} |
| mean messages per ticker-day | {S['messages']/S['files']:,.0f} |

**Table 2 — windows, by context length**

{tbl}
'''
display(Markdown(summary))

consumed = 32000 * 80          # cosine steps × effective batch, in windows @ L=2000
print(f"a 32,000-step run at effective batch 80 consumes {consumed:,} windows "
      f"= {consumed/S['windows']['2000']*100:.1f}% of one epoch at L=2,000")
print(f"                                           = {consumed*2000*26/1e9:,.1f}B tokens "
      f"of the {S['messages']*26/1e9:,.0f}B in the corpus")
""")

md(r"""
Two consequences worth carrying forward.

**The planned run sees 17% of the corpus, once.** At 32,000 optimiser steps and an
effective batch of 80 sequences it consumes 2.56M of the 15.0M available windows with no
repeats. Memorisation of a held-out window is therefore not the dominant leakage risk —
*neighbouring* windows from the same trading day being in the training stream is.

**The corpus is strongly non-stationary.** Monthly volume nearly triples from 2022 to
2025 and META carries 3.7× fewer messages than TSLA. A validation set drawn from the tail
of the period, or from one ticker, would confound the thing being measured with the
calendar. This has already bitten once on this project: a "memorisation" effect of
3.7 sigma collapsed to 1.0 sigma once the year-on-year trend in leak-free cross-entropy
was taken out.
""")

# ────────────────────────────────────────────── 4. validation set ────
md(r"""
## 4. The 0.1% validation set

**There is no validation set today.** `VAL_SPLIT=0.0` is hard-coded in
`run/base_model/train_full_autoreg.batch:333` and `train.py:443` sets
`train_only_validate_every_n_steps = 0`. Every training curve in this project so far is a
training loss.

Four design choices, and only the second is a matter of taste.
""")

code(r"""
files_total = S["files"]
msgs_total  = S["messages"]
target_msgs = 0.001 * msgs_total
mean_msgs_per_file = msgs_total / files_total

# candidate designs: hold out K whole ticker-days, subsample windows inside them
Ks = np.array([8, 24, 48, 96, 192, 400, 800])
holdout_cost = Ks / files_total * 100                      # % of corpus removed from training
frac_of_file = target_msgs / (Ks * mean_msgs_per_file) * 100   # % of each held-out day actually used

fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.9), layout="constrained",
                       gridspec_kw={"width_ratios": [1.2, 1.0, 1.15]})

a = ax[0]
a.plot(Ks, holdout_cost, "o-", color=C["warn"], lw=1.6, ms=5, label="withheld from training")
a.plot(Ks, np.full_like(Ks, 0.1, dtype=float), "--", color=C["ok"], lw=1.4,
       label="validation set (0.1%)")
a.set_xscale("log"); a.set_xticks(Ks); a.set_xticklabels([str(k) for k in Ks])
a.set_xlabel("ticker-days held out"); a.set_ylabel("% of corpus")
a.annotate("1.20%", (96, holdout_cost[list(Ks).index(96)]), xytext=(6, 6),
           textcoords="offset points", fontsize=7.3, color=C["warn"])
a.axvspan(72, 130, color=C["ok"], alpha=0.10)
a.annotate("recommended\n96 days", (96, 0.32), ha="center", fontsize=7.4, color=C["ok"], fontweight="bold")
a.legend(frameon=False, loc="upper left", fontsize=7.0); a.set_yscale("log")
a.set_title("Cost of a clean unit", loc="left")

a = ax[1]
a.plot(Ks, frac_of_file, "s-", color=C["accent"], lw=1.6, ms=5)
a.axhline(100, color=C["grey"], ls=":", lw=1)
a.text(9, 108, "whole day consumed", fontsize=7.2, color=C["grey"])
a.set_xscale("log"); a.set_yscale("log")
a.set_xticks(Ks); a.set_xticklabels([str(k) for k in Ks])
a.set_xlabel("ticker-days held out"); a.set_ylabel("% of each day used")
a.annotate(f"{frac_of_file[list(Ks).index(96)]:.1f}%", (96, frac_of_file[list(Ks).index(96)]),
           xytext=(5, 7), textcoords="offset points", fontsize=7.5, color=C["accent"])
a.set_title("Sampling depth per day", loc="left")

# evaluation cost
a = ax[2]
s_micro_base = M["ctx2k_training_arms"]["arms"][0]["s_per_micro_step"]
s_micro_hyb  = M["ctx2k_training_arms"]["arms"][1]["s_per_micro_step"]
gpus = 16
n_full = round(S["windows"]["2000"] * 0.001)
sizes = np.array([256, 512, 1024, 2048, n_full])
# forward-only ≈ 0.417/0.900 of the train step time, from Figure 2's two curves
fwd_ratio = M["memory_scaling"]["forward_only"]["s_per_call"][2] / M["memory_scaling"]["forward_backward"]["s_per_call"][2]
eval_s = sizes / gpus * s_micro_hyb * fwd_ratio
opt_step_s = s_micro_hyb * 5     # K=5 at 4 nodes
for every, col, ls in [(250, C["warn"], "-"), (500, C["hyb"], "-"), (2000, C["ok"], "-")]:
    a.plot(sizes, eval_s / (every * opt_step_s) * 100, ls, color=col, marker="o", ms=4,
           lw=1.5, label=f"every {every} optimiser steps")
a.axhline(2, color=C["grey"], ls=":", lw=1); a.text(300, 2.3, "2% budget", fontsize=7.2, color=C["grey"])
a.set_xscale("log"); a.set_yscale("log")
a.set_xticks(sizes); a.set_xticklabels([f"{s:,}" for s in sizes], rotation=30, ha="right")
a.set_xlabel("validation windows evaluated"); a.set_ylabel("training-time overhead (%)")
a.legend(frameon=False, fontsize=7); a.set_title("Two evaluation tiers", loc="left")

fig.suptitle("Figure 4 — sizing the validation set (0.1% is fixed; the holdout unit is the choice)",
             y=1.04, fontsize=8.5, color="#444")
show(fig, "Figure 4")

print(f"0.1% of the corpus              = {target_msgs/1e6:,.1f}M messages")
print(f"  as windows @ L=2,000          = {n_full:,}")
print(f"  as windows @ L=500            = {round(S['windows']['500']*0.001):,}")
print(f"full pass over {n_full:,} windows on 16 GPUs ≈ {n_full/gpus*s_micro_hyb*fwd_ratio/60:.1f} min")
print(f"512-window fast pass            ≈ {512/gpus*s_micro_hyb*fwd_ratio:.1f} s "
      f"= {512/gpus*s_micro_hyb*fwd_ratio/(500*opt_step_s)*100:.2f}% overhead at every-500-steps")
""")

md(r"""
### The design

| choice | decision | why |
|---|---|---|
| **size** | 0.1% of the corpus = 30.06M messages = **15,017 windows at L=2,000** (60,105 at L=500) | as requested; large enough that the paired standard error is far below any effect this project has cared about |
| **holdout unit** | **96 whole ticker-days**, 2 per month × 48 months, tickers rotated so each gets 12 days spread over four years | a whole day is the smallest unit with no window-level neighbour leakage. 96 costs 1.2% of training data to buy a 0.1% validation set — the 12× overhead is the price of a clean unit, and 1.2% of a corpus we consume 17% of is free |
| **stratification** | uniform across all 48 months and all 8 tickers; windows taken at a fixed stride inside each held-out day | the corpus triples in volume over the period and the tickers differ 3.7×. Anything drawn from the tail, or from one name, measures the calendar |
| **stored as** | a list of `(file_idx, seq_start)` **message offsets**, not a fixed tensor | `LOBSTER_Dataset` already accepts `explicit_windows` of shape `(N,2)` with `randomize_offset=False`. Storing offsets means **the same validation set can be re-cut at L = 500 / 1,000 / 2,000 / 4,000**, which is exactly the axis this baseline is exploring |
| **two tiers** | `valset_fast` = 512 windows (fixed deterministic subset) every 500 steps; `valset_full` = all 15,017 at every 4,000 steps and at the end | the fast tier costs 0.5% of training time, the full pass costs ~13 minutes and would cost 5% if run at every checkpoint |
| **budget** | register the number of evaluations up front; each full pass on the same frozen set spends some of its validity | the previous validation set on this project ran out of budget silently — 420,000 of 505,033 steps used before anyone checked |

### What has to be built

1. `build_valset_8stock.py` — pick the 96 ticker-days from the shard indices with a fixed seed, emit `windows_L{500,1000,2000,4000}.npy`, a manifest with per-file SHA-256, and the exclusion list.
2. A **training-side exclusion**: the sampler must skip those 96 files. Today `VAL_SPLIT=0.0` means the train dataset is the whole corpus, so the exclusion is new code, not a flag. This is the one place where getting it wrong is silent and fatal to the result.
3. A validation callback: `VAL_EVERY` in optimiser steps, `valset_fast` inline, `valset_full` at checkpoints.
4. A guard test that fails if any training window overlaps any validation file.
""")

# ────────────────────────────────────────────────────── 5. MFU ────
md(r"""
## 5. Compute efficiency — the honest number

**I never reached 30% MFU.** 30% was the *target* of a measurement campaign on
2026-09-01, and the campaign's conclusion was that 30% is unreachable at that model
shape. The highest number ever measured on this codebase is **11.5%**.

The 30% recollection is most likely the number **18.7%** being remembered as a result. It
is not a result — it is the *same* 9.4% run divided by a different denominator.
""")

code(r"""
ref = M["mfu_reference_75m"]
arms = M["ctx2k_training_arms"]
peak_bf16 = M["ctx2k_training_arms"]["peak_bf16_tflops_per_gpu"]
peak_tf32 = M["ctx2k_training_arms"]["peak_tf32_tflops_per_gpu"]

fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.1), layout="constrained",
                       gridspec_kw={"width_ratios": [0.95, 1.55, 1.05]})

# (a) the denominator story
a = ax[0]
bars = [("6ND\n÷ bf16", 9.4/2.63, C["light"]),
        ("GPM\n÷ bf16", ref["baseline_mfu_pct_bf16_denominator"], C["base"]),
        ("GPM\n÷ TF32", ref["baseline_mfu_pct_tf32_denominator"], C["accent"])]
a.bar(range(3), [b[1] for b in bars], color=[b[2] for b in bars], width=0.62)
for i, b in enumerate(bars):
    a.text(i, b[1]+0.6, f"{b[1]:.1f}%", ha="center", fontsize=8.5, fontweight="bold", color=b[2])
a.axhline(30, color=C["warn"], lw=1.3, ls="--")
a.text(-0.45, 31, "30% = the target", ha="left", color=C["warn"], fontsize=7.6)
a.set_xticks(range(3)); a.set_xticklabels([b[0] for b in bars], fontsize=7.6)
a.set_ylabel("MFU (%)"); a.set_ylim(0, 34)
a.set_title("One run, three conventions", loc="left")

# (b) levers
a = ax[1]
lv = [l for l in ref["levers"] if l["mfu_pct"] is not None]
base_mfu = ref["baseline_mfu_pct_bf16_denominator"]
SHORT = {"MAMBA3_CONTRACTION_PRECISION=default (TF32 einsum)": "TF32 einsum (free)",
         "headdim 64 -> 128": "headdim 128",
         "d_state 128 -> 64": "d_state 64",
         "chunk_size 64 / 128 / 256": "chunk_size (3 values)",
         "bf16 flag": "bf16 flag (a no-op)",
         "chunk128 + headdim128 together": "chunk128 + headdim128"}
names = ["baseline, as it ran"] + [SHORT[l["lever"]] for l in lv]
vals  = [base_mfu] + [l["mfu_pct"] for l in lv]
chg   = [False] + [l["changes_model"] for l in lv]
cols  = [C["grey"]] + [C["warn"] if c else C["ok"] for c in chg[1:]]
o = np.argsort(vals)
a.barh(range(len(vals)), [vals[i] for i in o], color=[cols[i] for i in o], alpha=0.9)
a.set_yticks(range(len(vals)))
a.set_yticklabels([names[i] for i in o], fontsize=7.6)
for j, i in enumerate(o):
    a.text(vals[i]+0.15, j, f"{vals[i]:.1f}", va="center", fontsize=7.2)
a.axvline(base_mfu, color=C["grey"], ls=":", lw=1)
a.set_xlabel("MFU (%) — bf16 denominator"); a.set_xlim(0, 15.5)
a.set_title("Green = free.  Red = changes the model.", loc="left")

# (c) path to 30
a = ax[2]
p = ref["path_to_30"]
xb = np.arange(len(p)); w = 0.27
a.bar(xb-w, [q["ceiling_pct"] for q in p], w, color=C["light"], label="arithmetic ceiling")
a.bar(xb,   [q["achieved_at_2x_overhead_pct"] for q in p], w, color=C["base"], label="at today's 2× overhead")
a.bar(xb+w, [q["achieved_with_fused_1p3x_pct"] for q in p], w, color=C["accent"], label="with a fused kernel (1.3×)")
a.axhline(30, color=C["warn"], lw=1.3, ls="--")
a.set_xticks(xb)
a.set_xticklabels(["today", "+d_state 64\n+bf16", "+d 2048\n+d_state 64\n+bf16"], fontsize=7.2)
a.set_ylabel("MFU (%)"); a.legend(frameon=False, fontsize=6.8, loc="upper left")
a.set_title("30% needs all three at once", loc="left")

fig.suptitle("Figure 5 — 75M Mamba-3, d_model 1024, 13,000-token sequences, 4× GH200 "
             "(measurement campaign, 2026-09-01)", y=1.05, fontsize=8.5, color="#444")
show(fig, "Figure 5")
""")

md(r"""
### What the three panels say

**Left.** The 9.4% and the 18.7% are the same run. The numerator is
`3ND × correction` where the correction is a GPM tensor-pipe measurement; the denominator
in the first two bars is the bf16 tensor-core peak of 989 TFLOPS. But **the model never
executes a bf16 kernel** — `args.dtype` has zero consumers anywhere in `src/`, and the
GPM FP16 pipe counter reads exactly 0.0%. Dividing by the TF32 peak instead, which is
what the hardware is actually doing, gives 18.7% for the identical run. That is a
relabelling, not an optimisation, and it is almost certainly where "about 30%" came from.

**Middle.** Only one lever is both effective and free: `MAMBA3_CONTRACTION_PRECISION=default`,
worth +24% wall-clock and reproduced twice. Everything else is either zero
(`chunk_size` — three settings inside noise), negative (`d_state` 64 is *slower* than
TF32 alone; `chunk128 + headdim128` together are worse than either alone), a no-op
(the bf16 flag), or changes the model (`headdim` 128 is the largest single win at +41%,
but it moves the parameter count and the head structure).

**Right.** The extrapolation, standing on the measured cost model: 30% requires
`d_state 64` **and** real bf16 **and** a working fused kernel, simultaneously. Two of
those three change what is being trained.

### Why this model is a worse case than the 75M reference

The hardware correction factor is width-dependent and was measured per width. At
`d_model=640` it is **3.09**; at 1024 it is **2.63**. A higher correction means more of
the work is non-matmul elementwise SSD arithmetic that runs on CUDA cores at 67 TFLOPS
rather than on tensor cores at 494. **Width is the real MFU lever, and 640 is on the
wrong side of it.**
""")

code(r"""
ct = {int(k): v for k, v in ref["correction_table"].items()}
ws = sorted(ct)
fig, ax = plt.subplots(1, 2, figsize=(9.6, 2.8), layout="constrained", gridspec_kw={"width_ratios": [1, 1.3]})

a = ax[0]
a.plot(ws, [ct[w] for w in ws], "o-", color=C["base"], lw=1.5, ms=5)
for w in (640, 1024):
    a.plot([w], [ct[w]], "o", color=C["hyb"] if w == 640 else C["ok"], ms=10, zorder=5)
    a.annotate(f"d={w}\ncorr {ct[w]:.2f}", (w, ct[w]), xytext=(8, 8), textcoords="offset points",
               fontsize=7.6, color=C["hyb"] if w == 640 else C["ok"], fontweight="bold")
wf = np.linspace(240, 1700, 100)
a.plot(wf, 2.0 + 750/wf, "--", color=C["grey"], lw=1, label="2.0 + 750/d fit")
a.set_xlabel("d_model"); a.set_ylabel("hardware FLOP correction  (× 6ND)")
a.legend(frameon=False); a.set_title("Narrow models do more non-matmul work", loc="left")

# measured MFU of the two ctx2k arms, both conventions
a = ax[1]
tps = 416000
rows = []
for arm in arms["arms"]:
    corr_true = ct[640]
    f6nd = 6 * arm["params"] * tps
    fl = corr_true * f6nd
    if "hybrid" in arm["name"]:
        L = 2000*26
        attn = 8 * (2 * L*L * 640) * 3      # 8 sequences, causal fwd 2L^2d, backward ~2x
        fl += attn
    ach = fl / arm["s_per_micro_step"] / 1e12
    rows.append((arm["name"], ach/(8*peak_bf16)*100, ach/(8*peak_tf32)*100,
                 arm["tflops_per_step_logged"]/arm["s_per_micro_step"]/(8*peak_bf16)*100))
xb = np.arange(2); w = 0.26
a.bar(xb-w, [r[3] for r in rows], w, color=C["light"], label="as logged today")
a.bar(xb,   [r[1] for r in rows], w, color=C["base"],  label="corrected ÷ bf16 peak")
a.bar(xb+w, [r[2] for r in rows], w, color=C["accent"],label="corrected ÷ TF32 peak")
for i, r in enumerate(rows):
    for dx, v in [(-w, r[3]), (0, r[1]), (w, r[2])]:
        a.text(i+dx, v+0.15, f"{v:.1f}", ha="center", fontsize=7.0)
a.set_xticks(xb); a.set_xticklabels(["baseline\nMamba-3", "hybrid\n+1 attention"], fontsize=7.6)
a.set_ylabel("MFU (%)"); a.legend(frameon=False, fontsize=7, loc="upper left")
a.set_ylim(0, 15)
a.set_title("The ctx2k arms — logged vs corrected", loc="left")

fig.suptitle("Figure 6 — width sets the ceiling; the hybrid arm's logged MFU is wrong by 3.09×",
             y=1.05, fontsize=8.5, color="#444")
show(fig, "Figure 6")

for r in rows:
    print(f"{r[0]:34s} logged {r[3]:5.2f}%   corrected/bf16 {r[1]:5.2f}%   corrected/TF32 {r[2]:5.2f}%")
""")

md(r"""
### A defect that has to be fixed before any MFU work

`src/lob/train.py:376` chooses the correction with
`correction = None if args.ssm_type == 'mamba3' else 1.0`. The hybrid launcher sets
`SSM_TYPE=` (empty) and selects the model through `ARCHITECTURE=hybrid_mamba3`, so
**the hybrid arm is charged a correction of 1.0 and the baseline 3.09**. The two arms'
logged `throughput/mfu_pct` differ by 3.09× for bookkeeping reasons alone — the hybrid
logs 1.4% where the comparable figure is about 5.5%. Any efficiency work that starts
from the dashboards would be optimising an artefact.

### The efficiency plan for this run

| # | action | expected | changes the model? |
|---|---|---|---|
| 1 | fix the correction gate to key on `architecture`, and add the attention layer's L² term for hybrids | correct numbers, no speed change | no |
| 2 | set `MAMBA3_CONTRACTION_PRECISION=default` and re-measure at d=640, L=52,000 | +24% is the d=1024/L=13,000 reading; **it must be re-measured here** — the arithmetic mix at 4× the length is different | no (numerics only) |
| 3 | report MFU against the TF32 peak, stating the convention on every figure | 5.6% → 11.2% by relabelling | no |
| 4 | one smoke run with `LOG_GRAD_NORMS=1` + throughput logging to fix the real step time at 4 nodes | sets the wall-clock estimate below | no |
| 5 | **not** in this run: `headdim` 128, `d_state` 64, bf16, fused kernels | these are the only route past ~12%, and each one breaks comparability with every existing checkpoint | yes |

The honest summary to carry into the run card: **expect 11–12% MFU against the TF32 peak
after step 2, and about 5.5–6% against the bf16 peak.** Anyone quoting 30% for this model
is quoting a target.
""")

# ─────────────────────────────────────────────────── 6. the plan ────
md(r"""
## 6. The run
""")

code(r"""
rec = M["recipe"]
s_base = arms["arms"][0]["s_per_micro_step"]; s_hyb = arms["arms"][1]["s_per_micro_step"]
EFF = rec["effective_batch_sequences"]; STEPS = rec["cosine_steps"]

fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.95), layout="constrained",
                       gridspec_kw={"width_ratios": [1.2, 0.85, 1.35]})

a = ax[0]
nodes = np.array([2, 4, 5, 8, 10])
K = EFF / (1 * 4 * nodes)
ok = np.isclose(K, np.round(K))
for s, col, lab in [(s_base, C["base"], "baseline"), (s_hyb, C["hyb"], "hybrid")]:
    hrs = STEPS * K * s / 3600
    a.plot(nodes[ok], hrs[ok], "o-", color=col, lw=1.6, ms=6, label=lab)
    for n_, h_ in zip(nodes[ok], hrs[ok]):
        a.annotate(f"{h_:.0f}h", (n_, h_), xytext=(4, 5), textcoords="offset points",
                   fontsize=7.2, color=col)
a.axhline(24, color=C["grey"], ls=":", lw=1); a.text(9.6, 25.5, "24 h", ha="right", fontsize=7.2, color=C["grey"])
a.set_xlabel("nodes (4 GH200 each)"); a.set_ylabel("wall-clock for 32,000 steps (h)")
a.set_xticks(nodes[ok]); a.legend(frameon=False)
a.set_title("Effective batch 80 fixes K = 20 / nodes", loc="left")

a = ax[1]
tok_run = STEPS * EFF * 2000 * 26 / 1e9
tok_corp = S["messages"] * 26 / 1e9
a.bar([0], [tok_corp], color=C["light"], width=0.5)
a.bar([0], [tok_run], color=C["hyb"], width=0.5)
a.text(0, tok_corp*1.02, f"{tok_corp:,.0f}B in the corpus", ha="center", fontsize=7.6, color=C["grey"])
a.text(0, tok_run*1.06, f"{tok_run:,.0f}B consumed\n({tok_run/tok_corp*100:.0f}%, one pass)",
       ha="center", fontsize=7.6, color=C["hyb"], fontweight="bold")
a.set_xticks([]); a.set_ylabel("tokens (billions)"); a.set_ylim(0, tok_corp*1.18)
a.set_title("Token budget", loc="left")

a = ax[2]
stages = ["smoke 150 steps", "valset build (CPU)", "train 32,000 steps", "full valset + LOB-Bench"]
dur    = [0.6, 1.5, STEPS*5*s_hyb/3600, 2.5]
colr   = [C["ok"], C["accent"], C["hyb"], C["base"]]
left = 0
for j, (st, d_, c_) in enumerate(zip(stages, dur, colr)):
    a.barh([0], [d_], left=left, color=c_, height=0.42)
    if d_ > 6:
        a.text(left + d_/2, 0, f"{d_:.1f} h", ha="center", va="center",
               fontsize=8, color="white", fontweight="bold")
        a.text(left + d_/2, 0.30, st, ha="center", fontsize=7.4, color=c_)
    else:
        yy = 0.30 if j % 2 == 0 else -0.34
        a.annotate(f"{st}  ({d_:.1f} h)", (left + d_/2, 0.21 if yy > 0 else -0.21),
                   xytext=(0, 16 if yy > 0 else -16), textcoords="offset points",
                   ha="center", va="center" , fontsize=7.0, color=c_,
                   arrowprops=dict(arrowstyle="-", color=c_, lw=0.6))
    left += d_
a.set_yticks([]); a.set_xlabel("hours (hybrid arm at 4 nodes)")
a.set_xlim(-1.5, left*1.06); a.set_ylim(-0.85, 0.95)
a.set_title(f"End to end ≈ {left:.0f} h", loc="left")

fig.suptitle("Figure 7 — the run, sized from the measured ctx2k step times",
             y=1.05, fontsize=8.5, color="#444")
show(fig, "Figure 7")

for nm, s in [("baseline", s_base), ("hybrid", s_hyb)]:
    for n_ in [4, 5]:
        k = EFF // (4*n_)
        print(f"{nm:9s} {n_} nodes  K={k:2d}  opt step {k*s:5.2f}s  "
              f"32,000 steps = {STEPS*k*s/3600:5.1f} h")
""")

code(r"""
r = M["recipe"]; a0 = arms["arms"]
card = f'''
**Table 3 — the run card**

| field | value |
|---|---|
| architecture | `{r['architecture']}` — Mamba-3 trunk, one global-attention layer at position 3 (Nemotron rule for L=6), NoPE |
| width / depth | d_model {r['d_model']}, {r['n_layers']} fused layers, blocks {r['blocks']} |
| Mamba-3 | d_state {r['mamba3']['d_state']}, expand {r['mamba3']['expand']}, headdim {r['mamba3']['headdim']}, chunk {r['mamba3']['chunk_size']}, RoPE fraction {r['mamba3']['rope_fraction']} |
| attention | {r['attention']['heads']} heads × head_dim {r['attention']['head_dim']}, flash, no positional encoding, d_ff {r['attention']['d_ff']} |
| parameters | {a0[1]['params']:,} (baseline {a0[0]['params']:,}, +5.4%) |
| **context** | **2,000 messages = 52,000 tokens** (fitted wall 2,050 at MEM_FRACTION 0.85) |
| tokenisation | {r['token_mode']}, vocab {r['vocab_size']}, {r['tokens_per_message']} tokens/message |
| data | {len(r['data']['tickers'])} tickers × {r['data']['months']} months, {r['data']['range']} |
| optimiser | {r['optimizer']['config']}, muon lr {r['optimizer']['muon_lr']}, ssm lr {r['optimizer']['ssm_lr_base']}, wd {r['optimizer']['weight_decay']}, {r['optimizer']['warmup']} |
| effective batch | {r['effective_batch_sequences']} sequences (declared; K is derived as 20 / nodes) |
| steps | {r['cosine_steps']:,} optimiser steps = {r['cosine_steps']*r['effective_batch_sequences']*2000*26/1e9:,.0f}B tokens = 17.0% of one epoch |
| micro-batch | 1 sequence per GPU (forced: 70.9 GB peak at 52,000 tokens) |
| MEM_FRACTION | 0.85 |
| checkpoint / logging | `CHECKPOINT_EVERY=auto` (15 min), wandb every 1 min, `LOG_GRAD_NORMS=1`, `LOG_EVERY=250` |
| efficiency | `MAMBA3_CONTRACTION_PRECISION=default`; MFU reported against the TF32 peak with the convention stated |
| validation | `valset_fast` (512 windows) every 500 steps; `valset_full` (15,017) every 4,000 steps and at the end |
| control arm | the pure Mamba-3 baseline, same seed, same effective batch, same steps — the only difference is the attention layer |
'''
display(Markdown(card))
""")

md(r"""
### Order of work, and what stops it

1. **Fix the FLOP-accounting gate** (`train.py:376`) — one line. Without it every
   efficiency number in the run is wrong by 3.09× for one arm only.
2. **Build the validation set** — CPU only, reads the 48 shard indices. Emits the window
   lists at all four context lengths plus a manifest with SHA-256, so the same set is
   reusable when the context axis is revisited.
3. **Wire the exclusion and the validation callback.** `VAL_SPLIT=0.0` means there is no
   existing code path to piggyback on. **This is the step where a silent mistake is
   fatal**: if the 96 held-out files are not actually excluded from the sampler, the
   validation curve measures nothing and nothing will report an error.
4. **Guard test** — assert zero file overlap between the training sampler and the
   validation window list, and assert the validation set's SHA-256 at load time.
5. **Smoke, 150 steps**, checking four things: 48 shards mount, the checkpoint round-trips
   with the new parameter tree, `len(dataset)` matches the count in Table 2 minus the
   96 held-out files, and the step time at 4 nodes.
6. **Both arms**, 32,000 steps, from the same seed. The baseline arm is not optional —
   every claim in Figure 1 is a paired claim.
7. **Evaluate**: full validation set, LOB-Bench on the frozen 3,136-sequence pool,
   reference-recall by age.

### Two things this plan deliberately does not do

**It does not chase 30% MFU.** Getting there needs `d_state` 64, real bf16 and a fused
kernel together, and two of those change the model. Doing them would produce a fast run
that cannot be compared to any existing checkpoint. The efficiency work here is the free
part: fix the accounting, turn on TF32, state the convention.

**It does not go to 4,000 messages.** The memory arithmetic says 134–142 GB against an
85.5 GB card, and the one lever that would fix it — activation checkpointing — is
currently a no-op that does not touch the recurrent trunk at all. That is a separate
piece of work with its own measurement, not a flag to flip in this run.
""")

# ───────────────────────────────────────────── 7. HF repo layout ────
md(r"""
## 7. Where this goes: `oxford-lob` on Hugging Face

The organisation `oxford-lob` (BOLD-Quant-Team, 3 members) currently holds **zero**
repositories. Proposed: one **dataset** repo whose README is the page this information
lives on.

```
huggingface.co/datasets/oxford-lob/lob-8stock-2022-2025
├── README.md                     ← the repo page: corpus tables, valset spec, provenance
├── corpus/
│   ├── dataset_stats_8stock.json     per-ticker, per-month, windows at each L
│   └── count_windows_from_shard_indices.py
├── valset_v1/
│   ├── holdout_files.json            the 96 ticker-days, with the selection seed
│   ├── windows_L500.npy              (N,2) int64 (file_idx, seq_start)
│   ├── windows_L1000.npy
│   ├── windows_L2000.npy
│   ├── windows_L4000.npy
│   ├── manifest.json                 sizes, SHA-256, evaluation budget
│   └── SHA256SUMS
└── notebooks/
    └── hybrid_longctx_8stock_training_plan.ipynb
```

Two constraints carried over from the existing Hugging Face work on this project.
The raw LOBSTER-derived message data **cannot** be published — the repo must stay
private and hold only indices, counts and window offsets, never message content. And a
free-tier account rate-limits at roughly 125 commits per 35 minutes, so the upload is a
single `upload-large-folder` call rather than a loop.

**Nothing is pushed to Hugging Face until this plan is confirmed.**
""")

md(r"""
---

### Open decisions

| # | decision | recommendation |
|---|---|---|
| 1 | context length: 2,000 now, or spend a day wiring remat first and aim at 4,000? | **2,000 now.** 4,000 is 1.66× the card and the enabling lever is currently a no-op |
| 2 | holdout unit: 96 ticker-days (1.2% withheld) vs 8 ticker-days (0.1% withheld, but only 8 days of diversity) | **96.** 1.2% of a corpus we consume 17% of is not a cost |
| 3 | run the baseline arm as well, doubling the compute? | **yes.** Figure 1 is a paired claim; an unpaired hybrid number cannot be read |
| 4 | efficiency: free levers only, or also `headdim` 128 (+41%)? | **free levers only** in this run; `headdim` 128 as a separate, explicitly-labelled architecture experiment |
| 5 | Hugging Face repo type | **dataset** repo, private, README as the page |
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.12"}}
nbf.write(nb, OUT)
print("wrote", OUT, os.path.getsize(OUT), "bytes,", len(cells), "cells")
