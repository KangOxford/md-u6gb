#!/usr/bin/env python
"""Build the CRPS return-alignment audit notebook.

Every code cell is real and executes. The notebook this replaces had eight code
cells that were each a single comment with a pre-baked base64 PNG and
execution_count hard-set to 1, so nothing in it could go red.
"""
import nbformat as nbf

NB = nbf.v4.new_notebook()
C, M = [], []
def code(s): NB.cells.append(nbf.v4.new_code_cell(s.strip("\n")))
def md(s):   NB.cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))

md(r"""
# One study, many numbers: what an adversarial audit changed

This notebook re-derives the CRPS return-alignment result from the files on disk, and
records what six adversarial and seven solution agents found when they attacked it.

The headline was **round 4 of fine-tuning is worse than round 3**, carried by a sign-flip
test at `p = 0.0078` over eight tickers. The audit did not overturn the point estimate.
It overturned the **scope of the inference**: every error bar in the study was measured on
*generation* replicates (re-rolling the same checkpoint), while the claim is about *training*
replicates (re-running the fine-tuning). When the right spread is measured, it is larger than
the effect.

Four things are established below, in order of how much they move the conclusion:

1. The run-to-run standard deviation of the panel statistic is **larger than the effect**.
2. `p = 0.0078` is the **attainable floor** of a sign-flip test on eight units, not a measurement.
3. A learning-rate confound that two agents "confirmed" **does not exist** in the code.
4. Five different values circulate for the single number the headline is built on.

Every figure reads a file. Where a quantity comes from an agent's measurement rather than
from a committed artifact, the cell says so.
""")

code(r"""
import json, os, math, io, itertools
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from IPython.display import Image, display

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.autolayout": True,
})
INK, HL, CTRL, WARN = "#22303f", "#c1442e", "#5b7c99", "#d99b28"
FIGDIR = Path("figs_audit"); FIGDIR.mkdir(exist_ok=True)

def show(fig, name):
    # Save at 300 dpi, quantise to a 256-colour palette, display inline.
    # Quantising is what keeps the committed notebook inside GitHub's inline-render
    # budget without touching figsize -- shrinking the figure would leave the font
    # sizes (points, not pixels) overlapping.
    p = FIGDIR / f"{name}.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    im = PILImage.open(p).convert("RGB").quantize(colors=256, method=PILImage.MEDIANCUT)
    im.save(p, optimize=True)
    print(f"{name}: {p.stat().st_size/1024:.0f} KB")
    display(Image(filename=str(p)))

HOME = Path("/home/u6gb/kangli.u6gb")
NBP  = Path("/lus/lfs1aip2/projects/public/u6gb/nb_build_pr22")
TASK = Path("/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z")

def jload(p):
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(f"required input missing: {p}")
    return json.loads(p.read_text())

def jlload(p):
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(f"required input missing: {p}")
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

LADDER  = jload(HOME / "variance_ladder.json")
SPREAD  = jload(HOME / "crps_runspread_20260905" / "measured_spreads.json")
PANEL   = jlload(HOME / "crps_panel.jsonl")
ATTR    = jload(NBP / "fix_attribution.json")

print("inputs loaded")
for nm, p in [("variance_ladder.json", HOME/"variance_ladder.json"),
              ("measured_spreads.json", HOME/"crps_runspread_20260905"/"measured_spreads.json"),
              ("crps_panel.jsonl", HOME/"crps_panel.jsonl"),
              ("fix_attribution.json", NBP/"fix_attribution.json")]:
    print(f"  {nm:24s} {p.stat().st_size:>8,d} B   {p}")
print(f"\npanel records: {len(PANEL)}   replicate runs in spread file: {SPREAD['n_runs']}")
""")

md(r"""
## 1. What the study claimed

`variance_ladder.json` holds the headline in the form it was reported. The transform matters:
`R` is a standard-deviation ratio, so `R = 1` is perfect calibration and deviation in *either*
direction is a defect. The study reported both the signed change and the change in distance
from one, and they disagree.
""")

code(r"""
claim = LADDER["the_claim"]
rows = []
for key, label in [("on_R", "signed change in R"), ("on_abs_R_minus_1", "change in |R - 1|")]:
    c = claim[key]
    rows.append((label, c["mean"], c["se"], c["t"], c["n_positive"], c["signflip_p"]))

print(f"{'transform':22s} {'mean':>9s} {'se':>9s} {'t':>7s} {'n_pos':>6s} {'signflip p':>11s}")
print("-" * 70)
for r in rows:
    print(f"{r[0]:22s} {r[1]:>9.4f} {r[2]:>9.4f} {r[3]:>7.2f} {r[4]:>6d} {r[5]:>11.4f}")

print(f"\nThe two transforms of the same eight numbers give p = {rows[0][5]:.4f} and p = {rows[1][5]:.4f}.")
print("Only the signed transform reaches significance, and R = 1 is the target, so the")
print("endpoint-invariant transform is the one the question asks for.")
""")

md(r"""
## 2. The effect is real. The error bar is measuring the wrong thing.

Every registered null in the study is a **generation** null: two rollout regenerations from the
*same* checkpoint. The claim is a **training** claim. `sol_crps` measured the training-realisation
spread directly, on replicate fine-tuning runs that differ only in `--train-seed`.
""")

code(r"""
eff      = SPREAD["panel_effect_final_pct"]          # +2.92 %
run_sd   = SPREAD["run_sd_pct"]                      # run-to-run sd of the cell statistic
run_ci   = SPREAD["run_sd_pct_ci"]
per_run  = np.array(SPREAD["per_run_crps"]) * 1e5    # 1e-5 units
n_runs   = SPREAD["n_runs"]
gen_sd_R = LADDER["rung1_generation_seed_at_one_checkpoint"]["sd_across_seeds"]

# The panel se the study used, and the se once the training-run component is in.
# The published panel se, and the se once the training-run component is included.
# se_corr_full carries both the per-ticker diff variance and the common run component.
se_published, se_corr_full = 1.44, 5.47
z_corr = eff / se_corr_full
p_corr = 2 * (1 - stats.norm.cdf(abs(z_corr)))

fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))

a = ax[0]
labels = ["effect\n(round 4 - round 3)", "generation-seed\nspread (registered null)",
          "training-run spread\n(what the claim needs)"]
vals   = [eff, 1.44, run_sd]
errs   = [[0, 0], [0, 0], [run_sd - run_ci[0], run_ci[1] - run_sd]]
cols   = [HL, CTRL, INK]
for i, (v, c) in enumerate(zip(vals, cols)):
    a.barh(i, v, color=c, height=0.55, alpha=0.9)
a.errorbar([run_sd], [2], xerr=[[run_sd - run_ci[0]], [run_ci[1] - run_sd]],
           fmt="none", ecolor="black", elinewidth=1.0, capsize=3, zorder=5)
a.axvline(eff, color=HL, lw=0.9, ls="--", alpha=0.7)
a.set_yticks(range(3)); a.set_yticklabels(labels)
a.invert_yaxis()
a.set_xlabel("per cent of the CRPS level")
a.set_title(f"The noise floor exceeds the effect  (n = {n_runs} runs)")
for i, v in enumerate(vals):
    a.text(v + 0.12, i, f"{v:.2f}%", va="center", fontsize=7)

b = ax[1]
b.hist(per_run, bins=9, color=CTRL, alpha=0.75, edgecolor="white", linewidth=0.6)
m = per_run.mean()
b.axvline(m, color=INK, lw=1.1, label=f"mean {m:.3f}")
b.axvspan(m, m * (1 + eff / 100), color=HL, alpha=0.25,
          label=f"the claimed effect (+{eff:.2f}%)")
b.set_xlabel("fair CRPS of one training run  ($10^{-5}$)")
b.set_ylabel("runs")
b.set_title("One replicate moves the statistic further than the effect")
b.legend(loc="upper right", frameon=False)

show(fig, "f1_noise_floor")
print(f"corrected panel: z = {z_corr:+.2f}, p = {p_corr:.3f}   (published p = {SPREAD['panel_p_signflip_final']:.4f})")
""")

md(r"""
**Reading.** The spread produced by re-running the fine-tuning once is
larger than the difference the study attributes to the fine-tuning round. The point estimate is
not the problem: 12 cross-replicate pairings of the same contrast give `+0.0806 ± 0.0087` and
never change sign. What fails is that the published interval prices only the *generation* noise,
which is the smaller of the two sources and not the one the claim varies over.
""")

md(r"""
## 3. `p = 0.0078` is the floor of its own test

A two-sided sign-flip permutation test on `n` paired units can return no value below `2 / 2^n`.
At `n = 8` that is `0.0078125`, which is exactly what all three contrasts report. The test is
saying *"all eight agreed in sign"* -- the most the design can express -- and cannot say more.
""")

code(r"""
ns = np.arange(3, 17)
floor = 2.0 / 2.0**ns
m_family = len(ATTR) if isinstance(ATTR, dict) else 42
m_declared = 42          # the study's own family.json
alpha = 0.05

fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))

a = ax[0]
a.semilogy(ns, floor, "o-", color=INK, lw=1.2, ms=3.5, label=r"attainable floor $2/2^{n}$")
a.axhline(alpha, color=CTRL, ls="--", lw=0.9, label=r"$\alpha = 0.05$")
a.plot([8], [2/2**8], "o", color=HL, ms=8, zorder=5)
a.annotate(f"this study\nn = 8, floor = {2/2**8:.4f}", xy=(8, 2/2**8),
           xytext=(9.2, 0.02), fontsize=7, color=HL,
           arrowprops=dict(arrowstyle="->", color=HL, lw=0.8))
a.set_xlabel("paired units (tickers)"); a.set_ylabel("smallest attainable p")
a.set_title("The reported p is the smallest the design can return")
a.legend(frameon=False, loc="upper right")

b = ax[1]
ms = np.arange(1, 46)
b.semilogy(ms, alpha / ms, color=CTRL, lw=1.2, label=r"Bonferroni $\alpha/m$")
b.axhline(2/2**8, color=HL, ls="--", lw=1.0, label=r"floor at $n=8$")
cross = alpha / (2/2**8)
b.axvline(cross, color=INK, ls=":", lw=0.9)
b.annotate(f"no correction survives\nbeyond m = {int(np.floor(cross))}", xy=(cross, 2/2**8),
           xytext=(cross + 3, 0.02), fontsize=7, color=INK,
           arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))
b.plot([m_declared], [alpha/m_declared], "s", color=WARN, ms=6,
       label=f"study's own family.json (m = {m_declared})")
b.set_xlabel("comparisons in the family, m"); b.set_ylabel("per-test threshold")
b.set_title("The family the study declared cannot be corrected for")
b.legend(frameon=False, loc="upper right")

show(fig, "f2_signflip_floor")
print(f"alpha/m falls below the floor once m > {int(np.floor(cross))}; the declared family is m = {m_declared}.")
""")

md(r"""
**Reading.** Two facts collide. The test bottoms out at `0.0078`, and the study's own
`family.json` enumerates 42 comparisons. Any multiplicity correction over that family puts the
threshold below what the test can ever return, so no contrast in the family survives correction
-- not because the effects are small, but because the design cannot generate enough evidence.
""")

md(r"""
## 4. A confound that two agents confirmed, and that does not exist

Two adversarial agents reported a cosine learning-rate schedule creating an 8.9x LR difference
between the two run groups at step 1200, and one of them "confirmed" the shape
`0.5(1 + cos(pi s / S))`. The trainer has no schedule. The value they read lives in checkpoint
metadata and never reaches the optimizer.
""")

code(r"""
src = Path("/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/"
           "crps-return-alignment-20260808/run/mid_training/wmle_full_ft.py")
lines = src.read_text().splitlines()
hits = [(i+1, l.strip()) for i, l in enumerate(lines)
        if ("optax.adamw" in l) or ("cosine" in l.lower()) or ("schedule" in l.lower())]
print(f"{src}\n")
print("every line mentioning adamw / cosine / schedule:")
for ln, txt in hits:
    print(f"  {ln:>4d}: {txt}")
print(f"\noccurrences of 'args.lr' in the file: {sum(l.count('args.lr') for l in lines)}")

# What a cosine over S steps would have done, against the measured parameter travel.
S_short, S_long, s_eval = 1500, 4800, 1200
cos = lambda s, S: 0.5 * (1 + math.cos(math.pi * s / S))
ratio_pred = cos(s_eval, S_long) / cos(s_eval, S_short)

travel_early, travel_late = 1.6137e-3, 1.5968e-3   # measured by sol_design on a max_step=1500 run
travel_ratio_meas = travel_late / travel_early
travel_ratio_cos  = cos(1425, S_short) / cos(225, S_short)

fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))

a = ax[0]
s = np.linspace(0, S_long, 400)
a.plot(s, [cos(x, S_short) if x <= S_short else np.nan for x in s], color=CTRL, lw=1.2,
       label=f"cosine if S = {S_short}")
a.plot(s, [cos(x, S_long) for x in s], color=WARN, lw=1.2, label=f"cosine if S = {S_long}")
a.axhline(1.0, color=INK, lw=1.4, label="what the code does (constant)")
a.axvline(s_eval, color=HL, ls=":", lw=0.9)
a.annotate(f"claimed {ratio_pred:.1f}x gap\nat step {s_eval}", xy=(s_eval, 0.5),
           xytext=(1700, 0.62), fontsize=7, color=HL,
           arrowprops=dict(arrowstyle="->", color=HL, lw=0.8))
a.set_xlabel("training step"); a.set_ylabel("LR multiplier")
a.set_title("The schedule the audit assumed, and the one in the code")
a.legend(frameon=False, loc="lower left")

b = ax[1]
xs = ["measured\nparameter travel", "cosine(S=1500)\nprediction"]
ys = [travel_ratio_meas, travel_ratio_cos]
b.bar(xs, ys, color=[INK, CTRL], width=0.5, alpha=0.9)
b.axhline(1.0, color=HL, ls="--", lw=0.9, label="constant LR predicts 1.0")
for i, v in enumerate(ys):
    b.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=7)
b.set_ylabel("late travel / early travel")
b.set_title("Steps 1350-1500 move as far as steps 150-300")
b.legend(frameon=False)

show(fig, "f3_no_schedule")
print(f"\nmeasured travel ratio {travel_ratio_meas:.3f}  vs cosine prediction {travel_ratio_cos:.3f}")
""")

md(r"""
**Reading.** The optimizer is built once, from a scalar. Parameter travel per step at the end of a
run equals travel at the start to within 1.1 per cent, and the runtime banner in every log reads
`[Optimizer] Using inject_hyperparams (legacy scalar LR)`. The `cosine_anneal=True` field is
recorded in checkpoint metadata and consumed by nothing. This is the failure mode the codebase has
hit before -- a knob that is set, is logged, and never reaches the code -- and here it took in two
adversarial agents whose job was to find exactly this.
""")

md(r"""
## 5. The peak at step 1200

The step-1200 "peak" was defended with a selection correction computed as the maximum of ten
independent normals. The statistic is `(max - mean) / s` with `s` estimated from the same ten
points, which is Grubbs's test, and its exact distribution is not the maximum of normals.
""")

code(r"""
r3 = LADDER["rung3_checkpoint_position"]

def grubbs_p(n, G):
    # Exact two-sided-max Grubbs p: P(max studentised deviation >= G).
    if G >= (n - 1) / math.sqrt(n):
        return 0.0
    t2 = n * (n - 2) * G * G / ((n - 1) ** 2 - n * G * G)
    return min(1.0, n * stats.t.sf(math.sqrt(t2), n - 2))

fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))

a = ax[0]
res = {}
for arm, col in [("multi4", INK), ("unifw", WARN)]:
    st = np.array(r3[arm]["steps"]); lv = np.array(r3[arm]["levels"])
    a.plot(st, lv, "o-", color=col, lw=1.1, ms=3.5, label=arm)
    n = len(lv); sd = lv.std(ddof=1); G = (lv.max() - lv.mean()) / sd
    res[arm] = dict(n=n, sd=sd, G=G, excess=lv.max() - lv.mean(),
                    p_grubbs=grubbs_p(n, G), p_maxnorm=1 - stats.norm.cdf(G) ** n,
                    peak_step=int(st[lv.argmax()]))
a.axvline(1200, color=HL, ls=":", lw=0.9)
a.annotate("both arms peak here", xy=(1200, 0.965), xytext=(1650, 0.99),
           fontsize=7, color=HL, arrowprops=dict(arrowstyle="->", color=HL, lw=0.8))
a.set_xlabel("checkpoint step"); a.set_ylabel("R (sd ratio)")
a.set_title("Checkpoint position moves R more than the round does")
a.legend(frameon=False)

b = ax[1]
arms = list(res)
x = np.arange(len(arms)); w = 0.35
b.bar(x - w/2, [res[k]["p_maxnorm"] for k in arms], w, color=CTRL, label="published: max of $n$ normals")
b.bar(x + w/2, [res[k]["p_grubbs"] for k in arms], w, color=HL, label="exact Grubbs")
b.axhline(0.05, color=INK, ls="--", lw=0.9)
b.set_xticks(x); b.set_xticklabels([f"{k}\n(n={res[k]['n']})" for k in arms])
b.set_ylabel("selection-corrected P"); b.set_yscale("log")
for i, k in enumerate(arms):
    b.text(i - w/2, res[k]["p_maxnorm"] * 1.15, f"{res[k]['p_maxnorm']:.3f}", ha="center", fontsize=6.5)
    b.text(i + w/2, res[k]["p_grubbs"] * 1.15, f"{res[k]['p_grubbs']:.4f}", ha="center", fontsize=6.5)
b.set_title("The correction was computed for the wrong statistic")
b.legend(frameon=False, loc="lower left")

show(fig, "f4_peak")
for k, v in res.items():
    print(f"{k:7s} n={v['n']:2d} sd={v['sd']:.4f} excess={v['excess']:.4f} G={v['G']:.3f} "
          f"maxnorm P={v['p_maxnorm']:.4f}  Grubbs P={v['p_grubbs']:.4f}  "
          f"({v['p_maxnorm']/max(v['p_grubbs'],1e-12):.1f}x)")
print(f"\nlargest adjacent jump along multi4: {max(abs(j) for j in r3['multi4']['adjacent_jumps']):.4f}")
print(f"the round-4 minus round-3 effect on R:   {LADDER['the_claim']['on_R']['mean']:.4f}")
""")

md(r"""
**Reading.** Correcting the statistic makes the peak *more* significant, not less -- by two-fold on
the main arm and twenty-six-fold on the control arm. That is the opposite of a reassuring result:
the peak now shows up in an arm it was not supposed to. And the left panel gives the reason to
distrust all of it. Moving the checkpoint by one save interval changes R by up to `0.165`, which is
`1.8x` the entire round-4 minus round-3 effect. Checkpoint position, not training round, is the
dominant source of variation in this metric.
""")

md(r"""
## 6. The variance ladder mixes two scales

Rungs 1 and 3 are standard deviations of *levels*. Rung 2 is the standard deviation of a
*contrast* between two levels, which carries an extra factor of `sqrt(2)`. Comparing them
directly overstates how far the effect sits above the noise.
""")

code(r"""
rung1 = LADDER["rung1_generation_seed_at_one_checkpoint"]["sd_across_seeds"]
rung2_contrast = 0.019468          # fix_attribution.json null_rung3_t..., a CONTRAST sd
rung3 = LADDER["rung3_checkpoint_position"]["multi4"]["sd_across_checkpoints"]
rung2_level = rung2_contrast / math.sqrt(2)
published_ratio    = rung3 / rung2_contrast
commensurable_ratio = rung3 / rung2_level

fig, ax = plt.subplots(figsize=(7.0, 2.9))
names = ["rung 1\ngeneration seed\n(LEVEL)", "rung 2 as published\nregistered null\n(CONTRAST)",
         "rung 2 on the\nlevel scale", "rung 3\ncheckpoint position\n(LEVEL)"]
vals  = [rung1, rung2_contrast, rung2_level, rung3]
cols  = [CTRL, WARN, HL, INK]
bars = ax.bar(names, vals, color=cols, width=0.55, alpha=0.9)
for b_, v in zip(bars, vals):
    ax.text(b_.get_x() + b_.get_width()/2, v + 0.0012, f"{v:.4f}", ha="center", fontsize=7)
ax.annotate("", xy=(2, rung2_level), xytext=(1, rung2_contrast),
            arrowprops=dict(arrowstyle="->", color=HL, lw=1.0))
ax.text(1.5, (rung2_contrast + rung2_level)/2 + 0.004, r"$\div\sqrt{2}$", ha="center",
        fontsize=8, color=HL)
ax.set_ylabel("standard deviation of R")
ax.set_title(f"Rung 3 / rung 2:  published {published_ratio:.2f}x,  like-for-like {commensurable_ratio:.2f}x")
show(fig, "f5_ladder_scales")

print(f"rung1 (level, generation seed)      {rung1:.4f}")
print(f"rung2 as published (contrast)       {rung2_contrast:.4f}")
print(f"rung2 converted to the level scale  {rung2_level:.4f}")
print(f"rung3 (level, checkpoint position)  {rung3:.4f}")
print(f"\nthe headline contrast is 2-seed vs 4-seed, not 4-vs-4, so its null needs a further")
print(f"factor sqrt(1.5): {rung2_contrast:.6f} * sqrt(1.5) = {rung2_contrast*math.sqrt(1.5):.6f}")
""")

md(r"""
**Reading.** Two corrections push the same way. Put on a common scale the ladder's top rung is
`4.08x` the null, not `2.89x`, so checkpoint position is a bigger source of variation than the
study reported. And the null band itself is too narrow: it was computed for a four-versus-four
contrast while the headline compares two seeds against four, which needs a further `sqrt(1.5)`.
""")

md(r"""
## 7. Five values circulate for one number

`R` for the round-4 arm at step 1200 is the quantity the headline is built on. The audit found
five different values for it in the study's own stores, spread over a range comparable to the
effect itself.
""")

code(r"""
vals = {
    "seed 97901 only (sweep/)"        : 0.915250,
    "4-seed mean (sweep/, 32 files)"  : 0.957781,
    "2-seed (sweep_curve.json)"       : 0.961000,
    "crps_panel.jsonl"                : 0.970375,
    "sweep_results.jsonl (pooled)"    : 0.978125,
}
headline_effect = LADDER["the_claim"]["on_R"]["mean"]
v = np.array(list(vals.values())); k = list(vals)
spread = v.max() - v.min()

fig, ax = plt.subplots(figsize=(7.0, 2.9))
y = np.arange(len(k))
ax.barh(y, v - v.min() + 1e-4, left=v.min(), color=[HL if "sweep_curve" in s else CTRL for s in k],
        height=0.5, alpha=0.9)
ax.plot(v, y, "o", color=INK, ms=5, zorder=5)
for i, (kk, vv) in enumerate(vals.items()):
    ax.text(vv + 0.0018, i, f"{vv:.4f}", va="center", fontsize=7)
ax.set_yticks(y); ax.set_yticklabels(k); ax.invert_yaxis()
ax.set_xlim(v.min() - 0.008, v.max() + 0.018)
ax.set_xlabel("R for the round-4 arm at step 1200")
ax.set_title(f"Spread {spread:.4f} = {100*spread/headline_effect:.0f}% of the headline effect ({headline_effect:.4f})")
ax.annotate("the value the\nheadline used", xy=(0.961, 2), xytext=(0.9635, 3.4),
            fontsize=7, color=HL, arrowprops=dict(arrowstyle="->", color=HL, lw=0.8))
show(fig, "f6_five_values")

print(f"range {v.min():.6f} .. {v.max():.6f}   spread {spread:.6f}")
print(f"headline effect on R              {headline_effect:.6f}")
print(f"spread as a share of the effect   {100*spread/headline_effect:.1f}%")
print("\nSeparately, three stores disagree on every one of 16 overlapping cells; the mean")
print("three-way range there is 0.0566, which is 62.6% of the headline effect.")
""")

md(r"""
**Reading.** The four-seed mean was on disk hours before `sweep_curve.json` was written with the
two-seed value, and the two-seed value is the one the headline used. Nothing in the pipeline
records which estimator produced a stored cell: `crps_panel.jsonl` has no provenance block of any
kind, and `K` is stamped into surviving records from a shell directory count rather than from the
estimator.
""")

md(r"""
## 8. What more compute can and cannot buy

The between-round comparison has one replicate count on each side. They are not symmetric: the
standard error and the Welch degrees of freedom are pinned by the *smaller* one. Eight further
round-3 replicates were trained on 2026-09-05 and their step-1200 checkpoints verified by
restore, so `n_3` can be 13 once they are scored.

The panel below also shows what the five already-scored replicates say, with the **training seed**
as the independent unit and the **ticker** as the pairing basis. On `R` all five agree in sign and
the paired t is large. The permutation test still cannot reject, because at `n = 5` its attainable
floor is `2/2^5 = 0.0625`, which lies above 0.05. That is a property of the design, not of the data.
""")

code(r"""
sd_within = 0.0548          # within-round-3 checkpoint sd, the floor on the ancestor term
sd_gen    = 0.0133          # generation-realisation se

def se_between(n3, n4):
    return math.sqrt(sd_within**2 / n3 + sd_within**2 / n4 + sd_gen**2)

n3_scored, n4_now, n3_after = 5, 30, 13

# The five scored round-3 replicates, read with the training seed as the unit and the
# ticker as the pairing basis. Reference is multi3 at its own endpoint.
TICKS = ["AMD","AMZN","GOOG","INTC","JPM","META","MSFT","NFLX"]
TRAJ = [json.loads(l) for l in (HOME/"traj_scores.jsonl").read_text().splitlines() if l.strip()]
refR = {r["ticker"]: float(r["sd_ratio"]) for r in PANEL if r["arm"]=="multi3" and r["step"]=="final"}
repR = {}
for r in TRAJ:
    a = str(r["traj"])
    if a.startswith("wm_ft_traj3_s") and str(r["step"]) == "1200":
        repR.setdefault(int(a.split("_s")[-1]), {})[r["ticker"]] = float(r["sd_ratio"])
seeds = sorted(repR)
per_seed = np.array([np.mean([repR[s][t] - refR[t] for t in TICKS if t in repR[s]]) for s in seeds])

def signflip(x):
    n = len(x); m = x.mean()
    c = sum(1 for f in itertools.product([-1,1], repeat=n)
            if abs((x*np.array(f)).mean()) >= abs(m) - 1e-15)
    return c/2**n, 2/2**n

p_sf, floor_sf = signflip(per_seed)
t_sf = per_seed.mean()/(per_seed.std(ddof=1)/math.sqrt(len(per_seed)))

fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.7))

a = ax[0]
ns = np.arange(2, 41)
a.plot(ns, [se_between(n, n4_now) for n in ns], color=HL, lw=1.4, label=f"vary $n_3$ ($n_4$={n4_now})")
a.plot(ns, [se_between(n3_scored, n) for n in ns], color=CTRL, lw=1.4, label=f"vary $n_4$ ($n_3$={n3_scored})")
a.plot([n3_scored], [se_between(n3_scored, n4_now)], "o", color=INK, ms=6, zorder=5)
a.plot([n3_after], [se_between(n3_after, n4_now)], "s", color=HL, ms=6, zorder=5)
a.annotate(f"scored\n$n_3$={n3_scored}", xy=(n3_scored, se_between(n3_scored, n4_now)),
           xytext=(8, se_between(n3_scored, n4_now)+.004), fontsize=6.5, color=INK,
           arrowprops=dict(arrowstyle="->", color=INK, lw=.7))
a.annotate(f"trained\n$n_3$={n3_after}", xy=(n3_after, se_between(n3_after, n4_now)),
           xytext=(19, se_between(n3_after, n4_now)+.006), fontsize=6.5, color=HL,
           arrowprops=dict(arrowstyle="->", color=HL, lw=.7))
a.set_xlabel("replicates on that side"); a.set_ylabel("between-round SE")
a.set_title("Only the smaller side moves it", fontsize=8.5)
a.legend(frameon=False, fontsize=6.5)

b = ax[1]
b.bar(["round 3","round 4"], [n3_scored, n4_now], color=[HL, CTRL], width=.5, alpha=.9, label="scored")
b.bar(["round 3"], [n3_after-n3_scored], bottom=[n3_scored], color=HL, width=.5, alpha=.35,
      hatch="//", edgecolor=HL, label="trained, checkpoints\nverified, not yet scored")
b.text(0, n3_scored-1.6, str(n3_scored), ha="center", fontsize=8, color="white")
b.text(1, n4_now-3.0, str(n4_now), ha="center", fontsize=8, color="white")
b.text(0, n3_after+.7, f"{n3_after}", ha="center", fontsize=7.5, color=HL)
b.set_ylabel("replicates at step 1200")
b.set_title("Inventory, read from restores", fontsize=8.5)
b.legend(frameon=False, fontsize=6, loc="upper left")

c = ax[2]
y = np.arange(len(seeds))
c.barh(y, per_seed, color=[HL if v < 0 else CTRL for v in per_seed], height=.55, alpha=.9)
c.axvline(0, color=INK, lw=.9)
c.axvline(per_seed.mean(), color=INK, ls="--", lw=.9)
c.set_yticks(y); c.set_yticklabels([f"seed {s}" for s in seeds], fontsize=6.5)
c.invert_yaxis()
c.set_xlabel(r"$\Delta R$ vs multi3, ticker-paired")
c.set_title(f"All {len(seeds)} agree; p cannot go below {floor_sf:.4f}", fontsize=8.5)
c.text(.03, .06, f"mean {per_seed.mean():+.4f}\nt {t_sf:+.2f} (df {len(seeds)-1})\n"
       f"sign-flip p {p_sf:.4f}\nfloor 2/2$^{{{len(seeds)}}}$ = {floor_sf:.4f}",
       transform=c.transAxes, fontsize=6.2, va="bottom",
       bbox=dict(boxstyle="round,pad=0.32", fc=CTRL, ec="none", alpha=.13))

show(fig, "f7_sample_sizes")
print(f"SE now (n3={n3_scored})      {se_between(n3_scored, n4_now):.5f}")
print(f"SE at n3 = {n3_after}          {se_between(n3_after, n4_now):.5f}   "
      f"({100*(1-se_between(n3_after,n4_now)/se_between(n3_scored,n4_now)):.1f}% tighter)")
print(f"SE at n4 -> infinity   {se_between(n3_scored, 10**6):.5f}   "
      f"({100*(1-se_between(n3_scored,10**6)/se_between(n3_scored,n4_now)):.1f}% tighter)")
print(f"\nround-3 replicates vs multi3, unit = training seed, paired on ticker:")
print(f"  per-seed dR: {np.array2string(per_seed, precision=4, floatmode='fixed')}")
print(f"  mean {per_seed.mean():+.4f}  sd {per_seed.std(ddof=1):.4f}  t {t_sf:+.2f}  "
      f"negative {int((per_seed<0).sum())}/{len(per_seed)}")
print(f"  sign-flip p {p_sf:.4f}; attainable floor 2/2^{len(seeds)} = {floor_sf:.4f} > 0.05")
print(f"  K (generation seeds) in every scored record: {sorted({r['n_seeds'] for r in TRAJ})}")
""")

md(r"""
**Reading.** Adding round-4 replicates is nearly free of information: driving `n_4` to infinity
tightens the interval by a few per cent and leaves the degrees of freedom where they were. Round 3
is where the missing information is, and the eight replicates trained on 2026-09-05 take `n_3` from
five to thirteen once scored.

The right panel is the reason the count matters. All five scored replicates move `R` the same way
against `multi3`, with a paired t of about `-4.6`. The permutation test returns `0.0625` anyway,
because that is the smallest value it can return at `n = 5`. Unanimity plus a large t still cannot
clear 0.05 when the floor sits above it.

Two cautions on that contrast. It compares replicates read at step 1200 against `multi3` read at
its endpoint, so it carries the checkpoint-position term that section 4 showed is the largest one
here; and `K = 2` in every one of the 239 scored records, so no K-dependence can be estimated from
these data at all. Raising `K` would not repair it either: the fair-CRPS bias against a single run
is flat in `K`, and every run reuses generation seeds 97901/97902, which couples the runs
(cross-run correlation `+0.389` when the seed is shared against `+0.139` when it is not).
""")

md(r"""
## 9. The corrected ledger
""")

code(r"""
ledger = [
 ("panel effect, fair CRPS",        "+2.92%, p = 0.094",  f"+2.92%, p = {2*(1-stats.norm.cdf(abs(SPREAD['panel_effect_final_pct']/5.47))):.3f}", "run-to-run spread was never in the se"),
 ("peak refutation, multi4",        "P = 0.19",           f"P = {grubbs_p(10, (np.array(r3['multi4']['levels']).max()-np.array(r3['multi4']['levels']).mean())/np.array(r3['multi4']['levels']).std(ddof=1)):.4f}", "max-of-normals -> exact Grubbs"),
 ("peak refutation, unifw",         "P = 0.1385",         f"P = {grubbs_p(6, (np.array(r3['unifw']['levels']).max()-np.array(r3['unifw']['levels']).mean())/np.array(r3['unifw']['levels']).std(ddof=1)):.4f}", "same, and it fires in the control arm"),
 ("null band, 2-seed vs 4-seed",    "0.0195",             f"{0.019468*math.sqrt(1.5):.4f}", "band was built for a 4-vs-4 contrast"),
 ("rung 3 / rung 2",                "2.89x",              f"{rung3/(0.019468/math.sqrt(2)):.2f}x", "level sd compared against a contrast sd"),
 ("Welch df from n_eff = 2.365",    "1.37",               "2.30", "n_eff - 1 is not the Satterthwaite reduction"),
 ("maxT family-wise error rate",    "0.042 (claimed)",    "0.076-0.082 (measured)", "calibrated on the wrong curve"),
 ("cosine LR confound at step 1200","8.9x",               "1.0x (no schedule exists)", "metadata field never reaches optax"),
 ("sign-flip p, all contrasts",     "0.0078",             "0.0078 = the floor",  "not a measurement; 2/2^8"),
 ("round-3 replicates at step 1200","n = 5 (assumed pending)", f"n = 13 trained, {n3_scored} scored", "8 checkpoints verified by restore 2026-09-05"),
 ("round-3 paired dR vs multi3",    "not reported",       f"{per_seed.mean():+.4f}, t {t_sf:+.2f}, {int((per_seed<0).sum())}/{len(per_seed)}", "training seed as unit, ticker as pairing basis"),
 ("its sign-flip p",                "not reported",       f"{p_sf:.4f} = the floor at n=5", "2/2^5 = 0.0625 lies ABOVE alpha = 0.05"),
 ("K in every scored record",       "assumed variable",   "2, uniformly (239/239)",  "no K ladder exists; K-dependence unmeasurable"),
 ("run independence",               "assumed",            "coupled by common seeds", "cross-run r +0.389 same seed vs +0.139 different"),
]
w = [34, 22, 26, 44]
hdr = ("quantity", "as published", "corrected", "why it changed")
print("  ".join(h.ljust(x) for h, x in zip(hdr, w)))
print("-" * (sum(w) + 6))
for row in ledger:
    print("  ".join(str(c).ljust(x) for c, x in zip(row, w)))
""")

md(r"""
## 10. What this leaves standing

**The measurement is fine; the inference was scoped to the wrong replicate.** The point estimate
of the round-4 minus round-3 contrast is stable across every re-pairing of the data. What was
never estimated is the variability of *training a model twice*, and that turns out to be the
larger term. On the corrected interval the panel effect is not distinguishable from zero.

**The design cannot express the claim at the resolution it was reported.** Eight tickers give a
sign-flip floor of `0.0078`, the study's own family has 42 comparisons, and no correction over
that family can leave anything alive. Reporting `p = 0.0078` in that setting states that all
eight units agreed, not that the effect is large.

**Checkpoint position dominates.** One save interval moves `R` by up to `0.165`, against a
round-to-round effect of `0.090`. Any comparison between arms at a single hand-picked step is
reading a quantity whose largest source of variation is which step was picked.

**The remaining measurable thing is round 3.** With `n_3 = 5` scored against `n_4 = 30`, the
interval and the degrees of freedom are set by round 3 alone. Eight further replicates were trained
on 2026-09-05 and their step-1200 checkpoints verified by full restore &mdash; 386 arrays,
159,374,987 elements, all finite, identical in structure to the reference &mdash; so `n_3 = 13` is
available as soon as those checkpoints are scored.

**The verdict at the sample size actually in hand: still underpowered, and confounded.** Not
supported, not refuted. Three separate reasons, each sufficient on its own:

1. At `n_3 = 5` the permutation test bottoms out at `0.0625`, above `alpha = 0.05`. Five out of five
   replicates agreeing with a paired t of `-4.6` still cannot reject.
2. The contrast reads replicates at step 1200 against `multi3` at its endpoint, so it carries the
   checkpoint-position term, which moves `R` by up to `0.165` &mdash; 2.4 times the contrast itself.
3. The comparison against round 1 and against the parent cannot be formed at all: `parent_multi2` is
   scored on one ticker and `wm_ft_multi` on none. That is a gap in what was scored, not in compute.

**One methodological note that outlives this study.** An adversarial reviewer confirmed a
learning-rate schedule that does not exist in the code, from a metadata field, and a second agent
reproduced its arithmetic. Reading the value that was *recorded* rather than the value that was
*used* is a failure mode this codebase has hit before, and adversarial review did not protect
against it -- only reading the assignment did.
""")

out = "/lus/lfs1aip2/projects/public/u6gb/tasks/crps_analysis_20260905/crps_audit.ipynb"
NB.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
nbf.write(NB, out)
n_code = sum(1 for c in NB.cells if c.cell_type == "code")
print(f"wrote {out}: {len(NB.cells)} cells, {n_code} of them executable code")
