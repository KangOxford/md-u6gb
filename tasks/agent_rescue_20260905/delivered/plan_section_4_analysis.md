# Section 4 — The analysis, prespecified

Written 2026-09-05, before any new data exists. Every number below was recomputed on
2026-09-05 from `/lus/lfs1aip2/projects/public/u6gb/nb_build_pr22/` using the verification
scripts banked at `/home/u6gb/kangli.u6gb/plan4_verify/` (`v1_ticker.py`, `v2_transform.py`,
`v3_null.py`, `v4_calib.py`), and reproduces the values recorded in
`/home/u6gb/kangli.u6gb/FACTS.md`. Where a number here differs from a previously published
one, the difference is stated.

---

## 1. The primary endpoint and its estimator

**Estimand, in one sentence a reader can check:** the change, averaged over the eight
tickers, in the absolute distance of the 250-message return-sd ratio from one, between a
checkpoint of the new fine-tuning run and the round-3 frontier checkpoint, both scored on
the same contexts, the same days, and the same generation seeds.

**Primary quantity: |R − 1|, not R.** R = sd(generated)/sd(real) has its target in the
interior of its range, so R is not monotone in quality: a ticker at R = 1.02 and a ticker at
R = 0.98 are equally well calibrated, and moving the first to 1.18 while moving the second
to 1.00 raises the mean R while making the ensemble worse. "Better calibrated" can only mean
"closer to 1", and the transform that expresses it is |R − 1|. Overshoot must be penalised
exactly as undershoot is.

This is not a hypothetical. On the published headline comparison (`multi4` step 1200 against
the round-3 frontier), the two transforms disagree in direction and in significance:

| transform | effect | tickers improved | exact sign-flip p |
|---|---|---|---|
| R | **+0.0904** | **8 / 8** | **0.0078** |
| \|R − 1\| | −0.0336 (an improvement of 0.0336) | **5 / 8** | **0.3047** |

Three tickers get worse on the meaningful transform, and they are precisely the three that
started closest to calibrated: AMZN 0.054 → 0.177, JPM 0.033 → 0.065, NFLX 0.004 → 0.018.
The correlation between a ticker's R-gain and its starting distance from 1 is −0.004, i.e.
the procedure moves every ticker up by roughly the same amount regardless of whether up is
towards 1 or past it. **An 8/8 result on R can be a 5/8 result on calibration, and reporting
R alone converts a coin-flip into a headline.**

The contrast in the other direction is not affected by the choice, and both transforms agree,
which is worth stating because it shows the transform is not being chosen to produce an
answer:

| contrast (round 4 minus round-3 frontier) | mean | sd | se | t (n = 8) | sign |
|---|---|---|---|---|---|
| ΔR | −0.0808 | 0.0436 | 0.0154 | −5.245 | 8/8 down |
| Δ\|R − 1\| | +0.0726 | 0.0293 | 0.0104 | +7.005 | 8/8 worse |

**Estimator.** Per ticker `i`, `D_i = |R_i^new − 1| − |R_i^ref − 1|`, and the point estimate
is the equal-weight mean `D̄ = (1/8) Σ D_i`. Pairing is at the ticker level and is exact:
both arms are scored on the same contexts, days and generation seeds, so `D_i` differences
out the ticker-specific difficulty that dominates the absolute level. Secondary, reported
alongside and never instead: the per-ticker vector itself, and the count of tickers improved.

**Reported alongside the mean, always:** the eight `D_i` values. A mean of eight numbers whose
sign is not unanimous is a different claim from one whose sign is unanimous, and the mean
alone cannot distinguish them.

---

## 2. The unit of replication

Everything in this study is nested inside **one fine-tuning run per arm**. The nesting, from
inside out:

```
contexts (many)  <  days  <  generation seeds (10 available: 97701..97710)
                                 <  checkpoint step on one trajectory
                                        <  the training trajectory itself  (n = 1 per arm)
```

**n = 8, and the 8 are tickers, not replicates of the treatment.** Tickers are the only axis
on which the two arms are paired and on which they vary independently enough to carry an
error bar for the *scoring* of a fixed pair of checkpoints. They say nothing about whether a
second fine-tuning run would land in the same place.

**Effective sample size must be reported next to every n = 8.** The eight tickers are not
equally informative. On the rung-3 null (disjoint seed quadruples, same checkpoint), the
per-ticker null sd's are:

```
AMD 0.0234   AMZN 0.0174   GOOG 0.1095   INTC 0.0301
JPM 0.0332   META 0.0191   MSFT 0.0490   NFLX 0.0386
```

GOOG carries **62.8%** of the null variance of the equal-weight mean. Two effective-n
definitions, both reported because they answer different questions:

| definition | value | what it answers |
|---|---|---|
| Kish, `1 / Σ pᵢ²` on variance shares | **2.365** | how many equally-noisy tickers the equal-weight mean is worth |
| `var(mean_indep) / var(mean_actual)` ratio applied to n | **6.295** | how much the residual cross-ticker correlation (mean off-diagonal ρ = 0.065, variance ratio 1.271) costs |

Both are far below 8, and the first is the one that governs a mean dominated by one loud
component. **Any interval computed as `sd/√8` is too narrow; the honest divisor is `√2.365`
for the equal-weight mean.** An alternative that avoids the problem, and which this plan
prefers for the primary endpoint, is to report the per-ticker vector and use a test that does
not depend on the equal-weight aggregation (§3).

**What n is NOT.** n is not the number of contexts, not the number of days, not the number of
generation seeds, and not the number of checkpoints. Each of those inflates the apparent
precision of a quantity that is a property of one training trajectory.

---

## 3. The test, its calibration, and its resolution floor

### The floor

The exact sign-flip (permutation) test on n = 8 paired differences has `2^8 = 256`
sign assignments. Its **smallest attainable two-sided p-value is `2/2^8 = 0.0078125`**, and
its attainable p-values near the bottom are only

```
0.007812, 0.015625, 0.023438, 0.031250, 0.039062, 0.046875, ...
```

**Consequence, stated in advance: no family of 7 or more sign-flip tests on n = 8 can be
corrected to FWER 0.05 by Bonferroni or Holm**, because `0.05/7 = 0.00714 < 0.0078125`. The
smallest achievable per-test p already exceeds the required threshold, so the adjusted p can
never fall below 0.05 no matter how large the effect. Any published Bonferroni-adjusted
result over such a family is arithmetically impossible and must be withdrawn rather than
recomputed.

### The calibration

The nominal sign-flip p-value is **anti-conservative here**, because a generation-seed main
effect is shared across tickers (mean off-diagonal correlation 0.065 on the null, higher on
some subsets), which the sign-flip null does not model. Empirical size on 1,575 true-null
contrasts:

| nominal | empirical | inflation |
|---|---|---|
| 0.0078 | 0.0146 | ×1.87 |
| 0.0156 | 0.0267 | ×1.71 |
| 0.0313 | 0.0514 | ×1.65 |
| 0.0500 | 0.0711 | ×1.42 |
| 0.1000 | 0.1346 | ×1.35 |

By contrast the **ticker-level t with 7 df is approximately correctly sized** on the same
null: nominal 0.05 → empirical 0.0444, nominal 0.01 → 0.0063, nominal 0.002 → 0.0013.

### What will therefore be used

1. **Primary test: joint sign-flip maxT over the prespecified family.** All tests in the
   family use the same eight tickers, so a single sign vector can be applied to every
   contrast at once and the maximum absolute standardised statistic taken. This controls FWER
   in the strong sense without needing per-test α below the floor, and it inherits the
   correlation structure rather than assuming it away. The floor applies to the *joint*
   statistic once, not once per test.
2. **Reported next to it: the ticker-level t₇ p-value**, because it is the correctly sized
   one, and any disagreement between the two is itself a finding.
3. **Never reported: a nominal sign-flip p without its measured size.** Any nominal p from
   this test is accompanied by its empirically calibrated counterpart from the table above.

### The family, fixed now

The FWER family for the primary endpoint is exactly these members, declared before data:

| # | contrast | endpoint |
|---|---|---|
| F1 | new arm vs round-3 frontier, at the prespecified step | Δ\|R − 1\| |
| F2 | new arm vs round-4, at the prespecified step | Δ\|R − 1\| |
| F3 | new arm vs its own uniform-weight control | Δ\|R − 1\| |

Three members, one endpoint each, one prespecified checkpoint each. Everything else computed
is exploratory and will be labelled as such in the deliverable, with no adjusted p attached.

---

## 4. The noise scale each comparison is read against

The ladder, all as sd of the 8-ticker pooled quantity:

| rung | what varies | sd | measured on |
|---|---|---|---|
| 1 measurement | contexts/days within a fixed seed set | — | absorbed into rung 2 |
| 2 registered null | checkpoint against itself, disjoint seed quadruples | **0.0195** | 1,575 contrasts, reproduced 0.019468 |
| 3 generation seed | seed set, same checkpoint | 0.0368 | |
| 4 checkpoint position | where you stop on **one** trajectory | **0.0562** | the step sweep; **2.89× rung 2** |
| 5 training trajectory | which run you are on | **NEVER MEASURED** | one run per arm |

**The assignment rule, fixed in advance:**

| comparison | rung that applies | why |
|---|---|---|
| same checkpoint, different seeds | 2 (0.0195) | nothing else varies |
| two checkpoints **from the same trajectory** | 4 (0.0562) | stopping point is the thing that varies |
| two checkpoints **from two different runs** | **5, unmeasured** | trajectory identity varies and has never been quantified |
| a fine-tuned arm vs its frontier reference | **5, unmeasured** | this is the headline comparison |

**Every verdict published so far used rung 2 (0.0195) for a comparison that needs rung 4 or
rung 5.** That is a factor of 2.89 understatement at best, and at worst an interval on a
quantity whose dominant variance component has no estimate at all. Concretely: the reported
effect is 0.0808, while the largest adjacent-checkpoint jump on a single trajectory is 0.1647
(multi4) and 0.1754 (unifw). **A single trajectory moves further between two neighbouring
checkpoints than the entire effect being claimed between arms.**

**Therefore the plan's first measurement is rung 5**, and no between-arm interval will be
published until it exists. Until then, between-arm differences are reported as point
estimates with the explicit annotation "no valid interval: the trajectory rung is
unmeasured".

---

## 5. Prespecified interval construction

For a comparison at rung `k`, the interval on `D̄` is

```
D̄  ±  z* · sqrt( s²_within / n_eff  +  s²_rung_k )
```

with the three parts fixed as follows.

**`z*` is empirical, not Gaussian.** On the rung-3 ticker-level null (3,150 signed draws) the
measured quantiles of `|contrast|/sd` are:

| multiplier | tail mass |
|---|---|
| 1.895 | **5.02%** ← this is the correct 5% multiplier |
| 1.960 | 4.19% |
| **2.000** | **3.68%** ← the ±2 sd band actually used is a 3.68% band, not 5% |
| 2.446 | 0.254% (= 0.05/20, the per-comparison level for a 20-member family) |
| pool max | 2.476 |

So `z* = 1.895` for a single prespecified test. **A 20-comparison family band needs 2.446 sd,
and the entire null pool's maximum is 2.476 sd, so that band is at the edge of what the pool
can estimate and will not be used**; the maxT construction of §3 replaces it.

**`n_eff = 2.365`** (Kish), not 8, for any equal-weight-mean interval.

**`s²_rung_k` is additive and is not optional.** For a between-arm comparison, `s_rung_5` has
no estimate, so the interval is not constructed. For a within-trajectory comparison,
`s_rung_4 = 0.0562` enters directly, which alone gives a half-width of `1.895 × 0.0562 =
0.1065` — larger than any effect reported in this study.

**What the interval covers, stated honestly.** With rung 4 included, the interval covers
"where this trajectory would have landed had we stopped at a different, equally arbitrary
step". It does **not** cover "where a second fine-tuning run with a different seed would have
landed". Only a rung-5 estimate can support that, and this study currently has n = 1
trajectory per arm, so no interval published to date covers the quantity its caption claims.

---

## 6. What falsifies the hypothesis, and what a null looks like

**Hypothesis under test:** the density-ratio weighted objective improves 250-message return-sd
calibration relative to the round-3 frontier, beyond what continued fine-tuning of any kind
would produce.

**Falsified if any of the following holds** (all fixed before data):

- **F-a.** `D̄ ≥ 0` on |R − 1| at the prespecified step, i.e. the arm is no closer to
  calibrated than the reference. (Already the case for the published headline once the
  transform is corrected: +0.0336 improvement, 5/8, p = 0.3047, which is a null, not a win.)
- **F-b.** The uniform-weight control moves at least as far. Already measured: the control
  moves −0.0969 against round 4's −0.0808 (all four verified verbatim in
  `/lus/lfs1aip2/projects/public/u6gb/nb_build_pr22/fix_attribution.json`), so continued
  fine-tuning accounts for **119.9%** of the exit — a derived ratio, 0.0969/0.0808, not a
  recorded field — and the weights term is −0.0161, inside the ±0.0389 null band. **The mechanism
  claim is already refuted; the plan must not re-litigate it, only avoid repeating it.**
- **F-c.** The maxT-adjusted p over family {F1, F2, F3} exceeds 0.05 using the *measured*
  size, not the nominal one.
- **F-d.** The effect is smaller than `1.895 × s_rung_5` once rung 5 exists.

**What a null result looks like, written now so it cannot be re-described later:** `D̄` within
±0.0389 of zero (the registered null band) with a non-unanimous sign pattern and an adjusted
p above 0.05. That is a specific, recognisable outcome, and it is a legitimate finding: it
says the objective does not move calibration beyond continued training, on this data, at this
scale, with this budget.

**Explicitly not a falsification:** a single checkpoint on the new trajectory looking good.
`P(max of 10 noisy points ≥ observed) = 0.19` for multi4 and 0.14 for unifw, and the
unpublished steps 1050 and 1350 show R going 0.806 → 0.915 → 0.819, so the entire +0.109
excursion reverses within 150 steps. Peaks are selected, not measured.

---

## 7. Decision table

Each row names the outcome and what is done next. Fixed before data.

| # | measurement | outcome | next action |
|---|---|---|---|
| D1 | rung-5 replication: k ≥ 3 independent fine-tuning seeds per arm | `s_rung_5 ≤ 0.03` | proceed to F1–F3 with intervals; the study becomes interpretable |
| D2 | same | `0.03 < s_rung_5 ≤ 0.08` | effect of 0.0726 is at ~1σ; report as inconclusive and enlarge k before any claim |
| D3 | same | `s_rung_5 > 0.08` | between-arm comparison at this budget is not resolvable; stop making it and report why |
| D4 | F1 (new vs round-3 frontier) on \|R − 1\| | maxT-adjusted p ≤ 0.05 **and** ≥ 6/8 tickers improve | claim supported at the trajectory rung; publish with the rung-5 interval |
| D5 | F1 | p ≤ 0.05 but sign 5/8 or fewer | do not claim; report the split and the per-ticker vector — this is the pattern that produced the wrong-transform headline |
| D6 | F3 (vs uniform-weight control) | control moves as far or further | mechanism attribution refused, permanently; the arm may still be reported as "continued training helps" |
| D7 | fair-CRPS panel | any cell reports K < 2 or duplicate member directories | discard the whole panel; the estimator degrades to mean absolute error silently (see §Pipeline in FACTS.md) |
| D8 | checkpoint selection | the chosen step was selected on the same seeds/contexts used to score it | the comparison is inadmissible; re-score on held-out seeds or fix the step a priori |
| D9 | any family of ≥ 7 sign-flip tests on n = 8 | Bonferroni/Holm attempted | reject the analysis outright: `0.05/7 < 2/2^8` makes it arithmetically impossible |

**Step 1200 cannot confirm itself.** It was selected as the maximum over the same ten
checkpoints, on the same seeds and contexts it is scored on. Under D8 it is inadmissible as a
confirmatory readout. It may be used only as the *hypothesis* to be tested at a fixed step on
a new trajectory.

---

## 8. What this analysis cannot settle

- Whether a different fine-tuning trajectory would reproduce any of this. That needs rung 5,
  which is D1, and until D1 lands every between-arm statement in this study is a point
  estimate without an interval.
- Whether the objective helps at a larger scale or a longer budget. Nothing here extrapolates
  past the measured step range.
- Whether conditioning is improved. Two of the three registered metrics are provably blind to
  it in `/lus/lfs1aip2/projects/public/u6gb/nb_build_pr22/fix_midtraining.json`, `qL1_actual`
  and `qL1_after_shuffle` are bit-identical (AMD r2: 0.15691976225003362 both), as are
  `sd_ratio_actual` and `sd_ratio_after_shuffle` (0.7177553570603361 both), for every ticker
  and arm; fair_crps is the only conditional metric and has never produced a number.
