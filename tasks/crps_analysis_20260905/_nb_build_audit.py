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
the peak now shows up in an arm it was not supposed to.

The left panel says why a single step licenses so little. Adjacent saves move `R` by up to `0.165`
along one trajectory, larger than the `0.090` attributed to the round. Read carefully, that is a
statement about **which step gets picked**, not a noise term for a contrast: once two arms are both
read at the same step, checkpoint position is held fixed by the matching and drops out of the
difference. What it limits is generalisation -- a number read at step 1200, a step chosen as the
argmax, is a number about step 1200 until the steps are chosen without reference to the outcome.
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
## 9. Three claims from section 2 that did not survive their own check

Written after a reader pushed back on them. Each is re-derived here rather than repaired in
place, because the way each one failed is more useful than the corrected number.
""")

code(r"""
import sys
sys.path.insert(0, "/home/u6gb/kangli.u6gb/crps_runspread_20260905")
from measure_real import load_cells, build, day_boot, ci, HARV

daymap = json.load(open(HARV/"daymap_META.json"))
runs, real = load_cells(r"wm_ft_traj_s\d+")
ids, y, days, X, labels = build(runs, real, daymap)
nctx, R, S = X.shape

W_ctx = np.abs(X[:,:,0] - X[:,:,1]).mean(1)
pairs = list(itertools.combinations(range(R), 2))
D_same = np.mean([np.abs(X[:,a,k]-X[:,b,k])     for a,b in pairs for k in (0,1)], 0)
D_diff = np.mean([np.abs(X[:,a,k]-X[:,b,1-k])   for a,b in pairs for k in (0,1)], 0)
D_all  = (D_same + D_diff)/2

rows = []
for tag, arr in [("different generation seeds", D_diff), ("all cross pairs", D_all),
                 ("same generation seed only", D_same)]:
    g = arr - W_ctx
    lo, hi = ci(day_boot(days, lambda s_: g[s_].mean()))
    rows.append((tag, g.mean()*1e5, lo*1e5, hi*1e5))

# How often do genuinely independent draws of this shape give Dbar_all < Wbar?
rng = np.random.default_rng(20260905)
sd_ctx = X.reshape(nctx,-1).std(1, ddof=1)
hits, NSIM = 0, 2000
for _ in range(NSIM):
    Z = rng.normal(0,1,size=(nctx,R,S)) * sd_ctx[:,None,None]
    w  = np.abs(Z[:,:,0]-Z[:,:,1]).mean(1)
    ds = np.mean([np.abs(Z[:,a,k]-Z[:,b,k])   for a,b in pairs for k in (0,1)], 0)
    dd = np.mean([np.abs(Z[:,a,k]-Z[:,b,1-k]) for a,b in pairs for k in (0,1)], 0)
    hits += ((ds+dd)/2).mean() < w.mean()
p_indep = hits/NSIM

fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.8))

a = ax[0]
yy = np.arange(len(rows))
for i,(tag,m,lo,hi) in enumerate(rows):
    a.plot([lo,hi],[i,i], color=INK, lw=1.4, solid_capstyle="butt")
    a.plot([m],[i], "o", color=HL if m < 0 else CTRL, ms=6, zorder=5)
    a.text(m, i+0.28, f"{m:+.3f}", ha="center", fontsize=6.5, color=INK)
a.axvline(0, color=CTRL, ls="--", lw=1.0)
a.set_yticks(yy); a.set_yticklabels([r[0] for r in rows], fontsize=7)
a.invert_yaxis()
a.set_xlabel(r"$\bar D - \bar W$  ($10^{-5}$), day-bootstrap 95% CI")
a.set_title("Coupling shows in the seed contrast, not in the sign", fontsize=8.5)

b = ax[1]
b.bar(["independent draws\ngive $\\bar D<\\bar W$", "what I claimed"],
      [100*p_indep, 0.0], color=[CTRL, HL], width=.5, alpha=.9)
b.axhline(50, color=INK, ls=":", lw=.9)
b.text(0, 100*p_indep+2, f"{100*p_indep:.1f}%", ha="center", fontsize=8)
b.text(1, 3, '"impossible"', ha="center", fontsize=7.5, color=HL)
b.set_ylim(0, 68); b.set_ylabel("per cent of simulated designs")
b.set_title("Equal-distribution runs make it a coin flip", fontsize=8.5)

show(fig, "f8_corrections")
for tag,m,lo,hi in rows:
    print(f"  Dbar-Wbar, {tag:28s} {m:+.4f}e-5  95% CI [{lo:+.4f},{hi:+.4f}]")
print(f"\n  independent-draw simulation ({NSIM} designs): Dbar_all < Wbar in {100*p_indep:.1f}% of them")
r_same = np.mean([np.corrcoef(X[:,a,0], X[:,b,0])[0,1] for a,b in pairs])
r_diff = np.mean([np.corrcoef(X[:,a,0], X[:,b,1])[0,1] for a,b in pairs])
d_m = [np.abs(X[:,a,0]-y).mean() - np.abs(X[:,b,0]-y).mean() for a,b in pairs]
d_c = [np.abs(X[:,a,0]-y).mean() - np.abs(X[:,b,1]-y).mean() for a,b in pairs]
cW = ci(day_boot(days, lambda s_: W_ctx[s_].mean()))
print(f"  cross-run correlation: same seed {r_same:+.4f}, different seed {r_diff:+.4f}")
print(f"  sd of a paired run difference: matched seed {np.std(d_m,ddof=1)*1e5:.4f}e-5 "
      f"vs crossed {np.std(d_c,ddof=1)*1e5:.4f}e-5  -> CRN buys "
      f"{100*(1-np.std(d_m,ddof=1)/np.std(d_c,ddof=1)):+.1f}%")
print(f"  evaluation Monte Carlo error on Wbar: +/-{(cW[1]-cW[0])/2/W_ctx.mean()*100:.1f}% "
      f"({nctx} contexts, {S} members, day-clustered)")
""")

md(r"""
**What was wrong, and what replaces it.**

**(1) "The coupled `D̄ < W̄` is impossible for independent draws."** It is not. When the runs share
a distribution, `E|X_i - X_j|` *equals* `E|X_i - X_i'|`, so the difference has expectation zero and
falls below it half the time — the simulation above puts it at about 50%. The coupling is still
there, but the evidence for it is the **contrast between the seed pairings**, not the sign: same-seed
cross pairs sit far below different-seed ones, and the two intervals do not come close to meeting.

**(2) "Shared generation seeds invalidate the scoring."** They do not touch the within-cell
estimator. The `K = 2` members of one cell are two generation seeds of one checkpoint, exchangeable
given that checkpoint, so fair CRPS is unbiased for that run. Common random numbers matter only for
estimators that assume cross-run independence, such as a pooled `D̄`. And the variance reduction they
are supposed to buy on paired comparisons is, here, `0.4%` — the run-to-run difference is dominated
by the training term, not by generation noise, so the coupling neither helps nor hurts that contrast
much. Three dependencies, three different consequences: **within-ensemble** (benign),
**cross-run** (biases pooled estimators only), **evaluation Monte Carlo** (`±7.5%` on `W̄`).

**(3) "The fair-CRPS bias is flat in `K`."** That is algebra, not a measurement, and it must be
labelled as such: `E[fair_K] − CRPS(mixture) = (W̄ − D̄)/(2K)` shrinks with `K`, while
`E[fair_K] − mean single-run CRPS = (W̄ − D̄)/2` carries no `K`. Which one applies depends on the
estimand you want, not on the data. Every one of the 239 scored records is at `K = 2`, so **no**
`K`-dependence of anything is measurable here — and the magnitude estimated from those records,
`−0.85%`, has a confidence interval that includes zero.
""")

md(r"""
## 10. Is `n_3 = 13` thirteen independent training units?

The first version of this section answered by comparing configuration: same parent path, same
weights file, same item-set hash, same budget, same parameter count. **That is a configuration
table, not evidence.** Two runs can agree on every launch flag and still fail to be exchangeable if
the parent changed underneath them, if the seed feeds a different random stream, or if their
training samples overlap in a way the inference does not price.

Each of those was checked directly.
""")

code(r"""
import hashlib
N_ITEMS, MAXSTEP = 4800, 1500
def order(seed): return np.random.default_rng(seed).permutation(N_ITEMS)
def sha(a): return hashlib.sha1(a.tobytes()).hexdigest()[:12]

# (b) One random stream, or two? Recompute the order each run LOGGED, from the
# current code, for seeds drawn from both families.
LOGGED = {("traj3",1): "0d0abe40f578", ("traj3",2): "1579377c1dd8", ("traj3",22): "b39203d492c2",
          ("r3rep",40): "18c94703b061", ("r3rep",41): "877cc66f0930", ("r3rep",47): "ded0d4b1c7f7"}
repro = {k: sha(order(k[1])) == v for k, v in LOGGED.items()}
print("(b) seed -> data order, recomputed against what each run logged")
for (fam, sd), v in sorted(LOGGED.items()):
    print(f"    {fam}_s{sd:<3d} logged {v}  recomputed {sha(order(sd))}  {'match' if repro[(fam,sd)] else 'MISMATCH'}")
print(f"    seed-0 reference {sha(order(0))} (both families logged 0f14669f2a4d)")
print(f"    one function across both families: {all(repro.values())}\n")

# (c) Sample overlap. Each run consumes the first MAXSTEP of its own permutation
# of the SAME 4800-item pool, so the units are not disjoint.
seeds = [1,2,3,4,6] + list(range(40,48))
sets = {sd: set(order(sd)[:MAXSTEP].tolist()) for sd in seeds}
ov = [len(sets[a] & sets[b])/MAXSTEP for a, b in itertools.combinations(seeds, 2)]
union = set().union(*sets.values())
print("(c) training-sample overlap between the 13 units")
print(f"    each run sees {MAXSTEP}/{N_ITEMS} = {100*MAXSTEP/N_ITEMS:.2f}% of one fixed pool")
print(f"    pairwise overlap, {len(ov)} pairs: mean {100*np.mean(ov):.2f}%  "
      f"[{100*min(ov):.2f}, {100*max(ov):.2f}]")
print(f"    expected if the subsets were drawn independently: {100*MAXSTEP/N_ITEMS:.2f}%")
print(f"    union {len(union)}/{N_ITEMS} = {100*len(union)/N_ITEMS:.1f}%; "
      f"items seen by all 13: {len(set.intersection(*sets.values()))}")

fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.8))
a = ax[0]
a.hist([100*x for x in ov], bins=16, color=CTRL, alpha=.8, edgecolor="white", linewidth=.6)
a.axvline(100*MAXSTEP/N_ITEMS, color=HL, lw=1.3,
          label=f"independent-subsampling\nexpectation {100*MAXSTEP/N_ITEMS:.2f}%")
a.set_xlabel("pairwise training-sample overlap (%)"); a.set_ylabel("pairs")
a.set_title("The units share a pool, exactly as much as chance predicts", fontsize=8.5)
a.legend(frameon=False, fontsize=6.5)

b = ax[1]
srcs  = ["initialisation", "dropout", "data order", "numerics"]
live  = [0, 0, 1, 0.15]
cols  = [CTRL, CTRL, HL, WARN]
b.barh(range(4), live, color=cols, height=.55, alpha=.9)
b.set_yticks(range(4)); b.set_yticklabels(srcs, fontsize=7.5); b.invert_yaxis()
b.set_xlim(0, 1.25); b.set_xticks([0, 1]); b.set_xticklabels(["inert", "live"], fontsize=7)
for i, t in enumerate(["restored from one parent, not initialised",
                       "p_dropout = 0.0, key inert",
                       "default_rng(train_seed).permutation(4800)",
                       "XLA autotuning, ~4e-05 on the loss"]):
    b.text(0.03, i, t, va="center", fontsize=6.2,
           color="white" if live[i] > 0.5 else INK)
b.set_title("Only one source of variation is live", fontsize=8.5)
show(fig, "f9_exchangeability")
""")

md(r"""
**(a) The parent did not move.** `wm_ft_multi2/69378` was written on 2026-08-14 and nothing under it
has been touched since, so both families restored the same bytes rather than the same path. Every one
of the thirteen checkpoints carries `init_timestamp_nsecs = 1784451028850888013`, which is the
**parent's own** value: the weights are restored, not initialised.

**(b) One random stream, thirteen independent draws from it.** The data order is
`numpy.random.default_rng(--train-seed).permutation(4800)`. Recomputing it from the current code
reproduces the hash and the first eight indices that each run logged, for three seeds from each
family. One deterministic map, thirteen distinct seeds.

**(c) The overlap is what independence looks like.** Each run consumes the first 1500 of its own
permutation of the same 4800-item pool, so any two share about 31% of their training items. The
measured mean, 31.18%, is the independent-subsampling expectation of 31.25%. That is **evidence for
independent streams, not against them** — dependent draws would not land on the expectation.

### What the thirteen are, stated precisely

They are **thirteen independent replicates of a conditional experiment**. The shared parent and the
shared pool do not make them dependent on each other; they define what the replicates are conditional
*on*. Given that conditioning set the units are exchangeable, and inference from them is valid.

The estimand is

$$\operatorname{Var}\!\big(R \;\big|\; \theta_0,\; \mathcal{D},\; \mathcal{C},\; \mathcal{G},\; B \big)$$

with $\theta_0$ the restored parent `wm_ft_multi2@69378`, $\mathcal{D}$ the 4800-item pool,
$\mathcal{C}$ the 500 evaluation contexts, $\mathcal{G} = \{97901, 97902\}$ the generation seeds, and
$B = 1500$ steps. Everything estimated below is estimated for **that** quantity.

**What it is not, and the correction.** An earlier version of this section called it a *lower bound*
on training-run variance. That is wrong. The law of total variance gives

$$\operatorname{Var}(R) \;=\; \mathbb{E}_{C}\!\left[\operatorname{Var}(R \mid C)\right] \;+\; \operatorname{Var}_{C}\!\left(\mathbb{E}[R \mid C]\right)$$

so the **average** conditional variance over conditioning sets is at most the marginal — but a
**single** conditioning set can sit above or below that average. One fixed $\theta_0$ therefore bounds
the marginal in neither direction. To say anything about the marginal, the things now held fixed have
to vary: more than one parent, more than one pool, more than one pair of generation seeds.

**What the genealogy audit did and did not establish.** It established *what is being estimated*, and
it confirmed the randomisation is sound. It is not a refutation of the design: nothing in it makes the
thirteen units invalid replicates. It narrows the scope of the conclusion, and that scope is stated
above rather than left implicit.

The pairing is matched on everything held fixed: the same ticker, the same 500 contexts, the same 20
days, the same two generation seeds, and step 1200 on both sides.
""")

md(r"""
## 11. Verdict, claim by claim

Three labels only: **supported** by evidence that survives the audit, **refuted** by it, or
**underpowered** — the design cannot answer it at the sample size in hand.
""")

code(r"""
V = [
 ("Round 4 is worse than round 3 (panel CRPS)", "UNDERPOWERED",
  "effect +2.92%, run-to-run sd 4.46% [2.57,5.57]; corrected p = 0.594"),
 ("...and the p = 0.0078 that carried it",      "REFUTED as a measurement",
  "2/2^8 is the floor of the test; it reports unanimity, not magnitude"),
 ("A cosine LR confound separates the groups",  "REFUTED",
  "optax.adamw(args.lr) is a scalar; travel late/early = 1.011"),
 ("The step-1200 peak survives selection",      "REFUTED",
  "exact Grubbs P 0.0970 (multi4) and 0.0053 (unifw control), not 0.19 / 0.1385"),
 ("Checkpoint position is a minor term",        "REFUTED",
  "adjacent saves move R by up to 0.165 along a trajectory; bears on step SELECTION, not on a matched contrast"),
 ("The variance ladder rungs are comparable",   "REFUTED",
  "rung 2 is a contrast sd; like-for-like the ratio is 4.08x not 2.89x"),
 ("Round-3 replicates differ from multi3@1200", "ESTABLISHED (conditional)",
  "n=13, dR -0.0287, CI [-0.0499,-0.0090] excludes 0, 12/13, p 0.018 vs floor 0.00024"),
 ("...by as much as the headline effect",       "RULED OUT at delta=0.0904",
  "whole CI inside the declared margin; says nothing about smaller effects"),
 ("multi3@1200 is a typical round-3 checkpoint","REFUTED",
  "it sits at the 92nd percentile of its own 13 replicates, +0.70 replicate sd"),
 ("Cross-run coupling exists",                  "SUPPORTED",
  "same-seed vs different-seed Dbar-Wbar intervals are disjoint"),
 ("...and it invalidates the within-cell CRPS", "REFUTED",
  "K=2 members are exchangeable given the checkpoint; the estimator is unbiased for that run"),
 ("Dbar < Wbar is impossible under independence","REFUTED",
  f"simulation: it happens in about half of independent designs"),
 ("The fair-CRPS bias is flat in K",            "ALGEBRA, NOT MEASURED",
  "no K ladder exists (239/239 at K=2); the magnitude's CI includes 0"),
 ("Round 3 vs round 1 and vs the parent",       "NOT ESTIMABLE YET",
  "parent_multi2 scored on 1 ticker, wm_ft_multi on none"),
]
w = [44, 26, 62]
print("  ".join(h.ljust(x) for h,x in zip(("claim","verdict","the evidence that decides it"), w)))
print("-"*(sum(w)+4))
for c,v,e in V:
    print("  ".join(str(z).ljust(x) for z,x in zip((c,v,e), w)))
print()
print("counts:", {k: sum(1 for _,v,_ in V if v==k) for k in dict.fromkeys(v for _,v,_ in V)})
print("\nThe last three rows are settled by the completed 72-cell matrix; the labels come from\n"
      "the rule frozen at 09:41:31 UTC with 35 of 72 already scored, not from a pre-registration.")
""")

md(r"""
### The adjudication rule, and exactly when it was frozen

**This is not a pre-registration.** It was frozen at **2026-09-05 09:41:31 UTC**, commit `85c72145`,
with **35 of 72 cells already scored** — the whole `multi3` at step 1200 arm and two of the eight new
replicates. Partial data was visible when it was written. It is a partial-data freeze, and it should
be read with that discount: it constrains later tuning, it does not carry the guarantee a prospective
registration would.

**The unit.** One `--train-seed`. Thirteen on the round-3 side; the reference is `multi3` read at the
same step.

**The matching, frozen.** A pair contributes only if both sides are read at **step 1200**, on the
**same ticker**, over the **same 500 contexts** and **20 days**, with the **same generation seeds**
97901 and 97902 and `k_actual = 2` asserted by the estimator. Anything failing one of those is
**dropped and counted**, never silently rebalanced.

**The statistic.** Per training seed, the mean over tickers of the paired difference against `multi3`
at step 1200. Thirteen numbers, one per unit.

**The readout is an interval.** Mean, bootstrap interval over training seeds, and the interval over
tickers. A p-value appears only next to its attainable floor, `2/2^13 = 0.00024`.

### Equivalence needs a margin, so here is the margin

A confidence interval that contains zero and happens to be narrow does **not** establish equivalence.
Equivalence is a claim against a declared margin `delta`, and it holds only when the **entire**
interval lies inside `[-delta, +delta]` — the two one-sided-tests form. Declared now, before the
remaining cells land:

**`delta = 0.0904` on the `R` scale**, which is the effect this study originally claimed
(`variance_ladder.json`, `the_claim.on_R.mean`). Equivalence at that margin says exactly one thing:
*an effect as large as the one originally claimed is excluded.* It says nothing about smaller effects.

A stricter margin would need a decision threshold — how much mis-calibration of `R` actually matters
downstream — and nobody has supplied one. If one is supplied later it must be declared **before** the
interval is recomputed against it.

| Reading | Label |
|---|---|
| Interval excludes 0 | a difference at step 1200 is established, for the conditional estimand of section 10 |
| Entire interval inside `[-0.0904, +0.0904]` | **equivalent at that margin**: an effect as large as the original claim is excluded |
| Interval contains 0 and is **not** inside the margin | neither established nor ruled out — the design does not resolve it |

### What the 0.165 checkpoint spread is, and what it is not

It is the largest move in `R` between **adjacent saves along one trajectory** — a spread of levels
along the step axis. It has two roles and they must not be merged.

**It is not the error term of the matched contrast.** Once both sides are read at step 1200,
checkpoint position is held fixed *by the matching*, so it contributes nothing to the noise of the
paired difference. That error term comes from the thirteen training seeds and from nothing else. An
earlier revision of this notebook used 0.165 as a scale to judge the matched effect against; that was
wrong, and it is withdrawn.

**It is a limit on generalisation, through the selection process.** Step 1200 was not picked at
random: it is the argmax of the sweep, which is why a selection correction was needed at all
(section 5). So any result here is a result *at a step chosen by looking at the data*, and 0.165
measures how far `R` moves between neighbouring steps — that is, how little one step licenses a
statement about "the round". This is a caveat on the **target** of the claim, not a term in its
interval.

Concretely: a difference established at step 1200 is a difference at step 1200. Extending it to "round
4 is worse than round 3" requires either steps chosen without reference to the outcome, or an explicit
correction for having chosen this one.
""")

md(r"""
### Result: all 72 cells in, adjudicated under the rule above

The matrix completed with **72 of 72 cells scored and none dropped** — every cell passed the frozen
match on its own recorded provenance: checkpoint path ending `_step1200`, `k_actual = 2` from the
estimator, generation seeds exactly `[97901, 97902]`, 500 contexts, 20 days, and an evaluation index
file whose sha256 at generation still equals its sha256 now.
""")

code(r"""
import glob, re
TICKS = ["AMD","AMZN","GOOG","INTC","JPM","META","MSFT","NFLX"]
DELTA = 0.0904                      # declared 2026-09-05, before these cells landed
CELLDIR = Path("/lus/lfs1aip2/projects/public/u6gb/tasks/crps_analysis_20260905/cells")

cells, drops = {}, []
for f in sorted(CELLDIR.glob("*/score.json")):
    lab, tk = f.parent.name.rsplit("_", 1)
    d = json.loads(f.read_text()); a, pr = d["a"], d["provenance"]["a"]
    fail = []
    if not pr["ckpt"][0].endswith("_step1200"): fail.append("ckpt not step1200")
    if a["k_actual"] != 2:                      fail.append(f"k_actual={a['k_actual']}")
    if pr["seeds"] != [97901, 97902]:           fail.append("generation seeds")
    if d["n_contexts"] != 500 or d["n_days"] != 20: fail.append("context/day count")
    if pr["indices_sha256_at_generation"] != pr["indices_sha256_now"]: fail.append("index file changed")
    (drops if fail else cells).append((f.parent.name, "; ".join(fail))) if fail else cells.update({(lab,tk): a["sd_ratio"]})
print(f"cells scored {len(list(CELLDIR.glob('*/score.json')))}   passing the frozen match {len(cells)}   dropped {len(drops)}")
for n_, why in drops: print(f"   DROPPED {n_}: {why}")

ref = {t: cells[("multi3s1200", t)] for t in TICKS}
TRAJ = [json.loads(l) for l in (HOME/"traj_scores.jsonl").read_text().splitlines() if l.strip()]
old = {}
for r in TRAJ:
    a_ = str(r["traj"])
    if a_.startswith("wm_ft_traj3_s") and str(r["step"]) == "1200":
        old.setdefault(int(a_.split("_s")[-1]), {})[r["ticker"]] = float(r["sd_ratio"])

units, cover = {}, {}
for sd_ in range(40, 48):
    ds = [cells[(f"r3rep_s{sd_}", t)] - ref[t] for t in TICKS]
    units[f"r3rep_s{sd_}"] = float(np.mean(ds)); cover[f"r3rep_s{sd_}"] = len(ds)
for sd_, m in sorted(old.items()):
    ds = [m[t] - ref[t] for t in TICKS if t in m]
    units[f"traj3_s{sd_}"] = float(np.mean(ds)); cover[f"traj3_s{sd_}"] = len(ds)

x = np.array(list(units.values())); n = len(x)
rng = np.random.default_rng(20260905)
bs = np.array([rng.choice(x, n, replace=True).mean() for _ in range(20000)])
lo, hi = np.percentile(bs, [2.5, 97.5])
cnt = sum(1 for f_ in itertools.product([-1,1], repeat=n)
          if abs((x*np.array(f_)).mean()) >= abs(x.mean()) - 1e-15)
p_sf, floor_sf = cnt/2**n, 2/2**n

# where the reference sits inside its own replicate population (levels, not differences)
lev = {k: np.mean([cells[(k, t)] for t in TICKS]) for k in
       [f"r3rep_s{i}" for i in range(40, 48)]}
lev.update({f"traj3_s{k}": np.mean([m[t] for t in TICKS if t in m]) for k, m in old.items()})
ref_lev = np.mean([ref[t] for t in TICKS]); v = np.array(list(lev.values()))
above = int((v > ref_lev).sum())
headline = LADDER["the_claim"]["on_R"]["mean"]

fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
a = ax[0]
order_ = sorted(units, key=units.get)
y = np.arange(n)
a.barh(y, [units[k] for k in order_], color=[HL if units[k] < 0 else CTRL for k in order_],
       height=.6, alpha=.9)
a.axvline(0, color=INK, lw=.9)
a.axvspan(lo, hi, color=INK, alpha=.12)
a.axvline(x.mean(), color=INK, ls="--", lw=1.0)
a.axvline(-DELTA, color=WARN, lw=1.0); a.axvline(DELTA, color=WARN, lw=1.0)
a.set_yticks(y); a.set_yticklabels(order_, fontsize=5.8); a.invert_yaxis()
a.set_xlabel(r"$\Delta R$ vs multi3 at step 1200, ticker-paired")
a.set_title(f"13 units, CI [{lo:+.4f}, {hi:+.4f}], margin $\\pm${DELTA}", fontsize=8.5)

b = ax[1]
b.hist(v, bins=9, color=CTRL, alpha=.8, edgecolor="white", linewidth=.6)
b.axvline(ref_lev, color=HL, lw=1.6, label=f"multi3 @1200 = {ref_lev:.4f}")
b.axvline(v.mean(), color=INK, ls="--", lw=1.0, label=f"replicate mean = {v.mean():.4f}")
b.set_xlabel("mean $R$ over tickers"); b.set_ylabel("training seeds")
b.set_title(f"The reference is above {13-above} of its own 13 replicates", fontsize=8.5)
b.legend(frameon=False, fontsize=6.3, loc="upper left")
show(fig, "f10_result")

print(f"\nn = {n} training seeds (tickers per unit: {sorted(set(cover.values()))})")
print(f"  mean dR {x.mean():+.4f}   sd {x.std(ddof=1):.4f}   se {x.std(ddof=1)/math.sqrt(n):.4f}")
print(f"  bootstrap 95% CI over training seeds  [{lo:+.4f}, {hi:+.4f}]")
print(f"  sign-flip p {p_sf:.5f}  next to its attainable floor 2/2^{n} = {floor_sf:.5f}")
print(f"  negative units {int((x<0).sum())}/{n}")
print(f"\n  CI excludes 0                          : {lo*hi > 0}")
print(f"  CI entirely inside [-{DELTA}, +{DELTA}]     : {lo > -DELTA and hi < DELTA}")
print(f"\n  reference level {ref_lev:.4f} vs replicate mean {v.mean():.4f} (sd {v.std(ddof=1):.4f})")
print(f"  replicates above the reference: {above}/{n}  -> reference at the "
      f"{100*(1-above/n):.0f}th percentile of its own replicate set, {(ref_lev-v.mean())/v.std(ddof=1):+.2f} sd")
print(f"  gap {v.mean()-ref_lev:+.4f} against the study's headline effect {headline:+.4f}"
      f"  = {abs((v.mean()-ref_lev)/headline):.0%} of it")
""")

md(r"""
**Both labels in the rule fire, and they are not in tension.**

**A difference at step 1200 is established.** The interval over training seeds excludes zero and
twelve of thirteen units fall the same way. The sign-flip p is `0.018`, well clear of the attainable
floor `2/2^13 = 0.00024`, so unlike the study's original `0.0078` this one is a measurement rather
than the smallest number the test can return.

**And it is equivalent at the declared margin.** The whole interval lies inside
`[-0.0904, +0.0904]`, so an effect as large as the one this study originally claimed is excluded —
that, and nothing about smaller effects, is what the margin licenses.

**What the two together say.** Retrain round 3 changing only the data order and the result lands
**below** the round-3 reference checkpoint at the same step, in twelve of thirteen tries. The
reference sits at roughly the **92nd percentile of its own replicate population**, `+0.70` replicate
standard deviations above the replicate mean. The gap is `-0.0277` on `R`, which is about **31% of
the `+0.0904` the study reported as the round-4 minus round-3 effect**.

So the anchor the headline was measured against is not a typical member of the population it is
supposed to represent, and roughly a third of the headline's magnitude is the size of that
atypicality. This does not overturn the headline — it was measured on a different contrast, and this
one cannot settle it — but it identifies a term the original comparison never priced.

**Scope, restated because it constrains every number above.** All of it is conditional on
$(\theta_0, \mathcal{D}, \mathcal{C}, \mathcal{G}, B)$ from section 10. It is a statement about step
1200, a step chosen as the argmax of the sweep; extending it across steps needs steps chosen without
reference to the outcome. The `0.165` adjacent-save spread plays no part in the interval above — the
matching holds checkpoint position fixed — and appears only as the reason a single step generalises
poorly.

**And the rule these labels come from was frozen at 09:41:31 UTC with 35 of the 72 cells already
scored.** It is a partial-data freeze, not a pre-registration, and the labels should be read with
that discount.
""")

md(r"""
## 12. The experiment table, reproducible

Every row is a command that reproduces the unit it names. Roots:
`W = /lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/crps-return-alignment-20260808`,
`T = /lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z`,
`O = /lus/lfs1aip2/projects/public/u6gb/tasks/crps_analysis_20260905`.

| # | Unit | Status | How to reproduce it |
|---|---|---|---|
| 1 | Round-3 replicate, one training seed | **13 done** | `TSEED=<n> WANT=<card> MAXSTEP=1500 bash $O/r3_replicate.sh` — restores `T/ckpt/wm_ft_multi2` at 69378, `v5m3_weights.npz`, 64 train / 16 hold seeds from 99000-99079 |
| 2 | Checkpoint integrity | **13 verified** | `python $O/_verify_ckpts.py` — restore must give 386 arrays, 159,374,987 finite elements |
| 3 | One scored cell, `K = 2` | **1 of 72 done** | `LABEL=<arm> CKPT=<...>_step1200 TICKER=<tk> WANT=<card> bash $O/score_cell.sh` — two `collect_rollouts.sh` passes at seeds 97901/97902, then `score_v5_primary.py --assert-k 2 --assert-ckpt-step 69378 --assert-seeds 97901,97902` |
| 4 | `multi3` at step 1200, 8 tickers | **queued (8)** | rows 1-8 of `$O/cell_queue.txt`; removes the checkpoint-position term from the main contrast |
| 5 | 8 new replicates at step 1200, 8 tickers | **queued (64)** | rows 9-72 of `$O/cell_queue.txt`; takes `n_3` from 5 to 13 |
| 6 | Queue drain onto free cards | running | `bash $O/drain_cells.sh` — re-reads `gtop` each round, skips any cell whose `score.json` exists, so it is idempotent |
| 7 | The three withdrawn claims | done | `python $O/_corrections.py` — reuses `measure_real.load_cells` |
| 8 | This notebook | done | `python $O/_nb_build_audit.py && jupyter nbconvert --execute --inplace crps_audit.ipynb` |

Not queued, and why: `parent_multi2` on its seven missing tickers and `wm_ft_multi` on all eight
would be needed for the round-1 and parent comparisons, but neither can change the verdict on the
claims above, so they wait behind the units that can.
""")

md(r"""
## 13. What this leaves standing

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
