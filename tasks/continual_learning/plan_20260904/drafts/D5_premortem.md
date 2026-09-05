# D5 — Pre-mortem for KangOxford/sigma-0 issue #73 (the failure-pool continual-learning system)

> Drafter D5, 2026-09-04. Scope: assume this line produced a confident wrong result; write how,
> then write the checks that would have caught it.
> All paths absolute. No GPU, sbatch or srun was used to produce this draft. Everything below
> marked "measured here" was recomputed on CPU from rollouts already on disk under
> `/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z/data`,
> total wall time about 6 minutes.
>
> Maintenance rule for this file (CONTEXT §5): mark completed items green, strike through
> anything found to be genuinely wrong. Do not delete; supersede in place.

---

## 0. The obituary — what the confident wrong result looks like

**2027-03-12, PR "continual learning: failure-pool CPT improves the 2024-08 stress slice".**

> We mined 500 contexts per ticker with one rollout each from the mamba3 main-line checkpoint,
> ranked them by squared error against the realised path, took the top decile as the failure
> pool, and continued pre-training with issue #73's 70 % replay / 30 % failure mix for 1 B tokens.
> Pool error fell 31 %. The random-pool control fell 9 %. Three seeds, p = 0.02.
> Conclusion: mining failures and retraining on them works.

Every number in that paragraph is real. The conclusion is wrong, for four reasons that
compound, and every one of them is measurable **before** the GPU is touched:

1. **Regression to the mean did most of the work.** A top decile picked from one rollout per
   context loses 77–82 % of its selection gap when re-scored on a *disjoint* set of rollouts with
   **no training at all** (measured here: top-decile persistence at k = 1 on the corrected score is
   0.183–0.234 across the eight tickers). The 31 % is measured against a baseline that was never
   re-scored, so most of it is the pool decaying back toward the mean.
2. **The random-pool control was the wrong control.** The failure pool's mean absolute realised
   move is 2.34–3.70× the population mean (measured here). Against a *random* pool it therefore
   also differs in volatility, message intensity and generated-path scale. A volatility-matched
   pool is the control; a random pool guarantees a positive result whenever the model is simply
   under-fit on high-activity windows, which replay would fix anyway.
3. **A third of the selected quantity was never learnable.** The dispersion term inside the top
   decile is 22.8–70.7 % of total squared error once the k/(k−1) bias is removed (measured here,
   H = 50, k = 10). Anything that "removes" it is the model becoming *more* wrong in distribution,
   not less, because the model is already under-dispersed.
4. **Three seeds is one degree of freedom short of a standard deviation you can trust.** This is
   the failure mode CONTEXT §4 puts first, and this repo has reversed an n = 2 claim three times in
   one session and once again with n = 8 → n = 13 (+1.66 % → +1.38 %).

The obituary's second act is worse: the result is published, the production retraining cadence is
built on it, and eighteen months later the pool turns out to be a **high-volatility-window
sampler**, which is a data-mixture knob that could have been set by one line in the sampler config
for zero rollout cost.

---

## 1. The five ways the premise is wrong, ranked by probability × cost

Probability is my prior after the measurements in §6 and §7. Cost is what is lost if it is not
caught until after the GPU spend. Rank = P × cost.

| # | Failure | P | Cost if uncaught | Rank | Status of evidence today |
|---|---|---|---|---|---|
| **R1** | **The pool is not reproducible enough to be a pool.** Selection on a stochastic score at the k the issue implies (k = 1) picks a set that a disjoint set of rollouts largely does not pick again. | 0.55 | The entire line, plus a published effect that is regression to the mean | **0.31** | measured, adverse |
| **R2** | **The reliable part of the score is the realised move plus the generated-path scale**, so "train on failures" = "up-weight high-volatility windows", a data-mixture effect obtainable for free. | 0.45 | The claim survives but the mechanism is wrong; the expensive machinery is unnecessary | **0.27** | measured, partly adverse |
| **R3** | **The dispersion floor caps the achievable gain below the noise floor of the training itself.** | 0.40 | Months chasing an effect that cannot exceed the seed spread | **0.22** | measured, adverse for one ticker |
| **R4** | **The intervention's effect is smaller than the seed-to-seed spread of continued training**, and n was chosen with an effect size estimated from the same small pilot. | 0.50 | A reversed claim at the next n; the documented repeat offence | **0.20** | not measured — no seed replicate exists |
| **R5** | **Training on its own failure selection moves the model off distribution** (self-training drift): old-window NLL rises faster than the new-window NLL falls, and the pool is small enough that the mix is effectively 58 repeats of 5.2 M tokens. | 0.30 | A model that scores better on the pool and worse everywhere | **0.14** | not measured; pool-size arithmetic in §4 is adverse |

Runners-up, kept because their cost is high even at low probability:

- **R6 — the evidence base is on the wrong slice and the wrong checkpoint.** Every one of the three
  prerequisite measurements was made on **2026-01 only** (20 trading days, 2026-01-02 … 2026-01-30;
  file names in
  `/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z/data/hp_v5me3_GOOG_s97701/member_0/data_real/`
  carry the date), from checkpoint `wm_ft_multi3` step 69378 (78,539,423 params, 26tok, per
  `.../member_0/inference.log`). The plan's slices are **2024-08 primary, 2025-04 secondary**
  (`/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/results/INVENTORY.md` §2), on the
  mamba3 main line. **Zero of the three prerequisites has been measured on the slice or the model the
  plan will use.** P = 0.35 that at least one of the three numbers moves enough to change a decision.
- **R7 — the score is confounded with an unregistered generation artefact.** See §6.4: `num_errors`,
  the count of generated messages that leave the L2 book unchanged, is 86–193 out of 250 generated
  messages (34.5 %–77.0 %), is *more* reliable across seeds (split-half ρ 0.60–0.78) than the failure
  score itself, and correlates with the corrected failure score at −0.19 … +0.21 with the **sign
  flipping between tickers**. Nobody has registered it.

### Why R1 outranks everything

The single number that decides the line is **pool purity**: given a pool selected from k rollouts,
what fraction of it is also selected from k *disjoint* rollouts. Chance is 0.10 for a top decile.
Measured here, mean of 120 draws per cell, 500 contexts per ticker, H = 50:

| pool rule | k = 1 | k = 2 | k = 3 | k = 5 |
|---|---|---|---|---|
| raw squared error (issue #73 as written) | 0.41 | 0.49 | 0.55 | 0.62 |
| ranked inside realised-move bins (the corrected pool) | **0.22** | 0.28 | 0.32 | **0.38** |

Per-ticker spread at k = 1 corrected: 0.20 (GOOG, AMZN, INTC) … 0.25 (MSFT).

Inverting the overlap for a pool of the same size as the true set gives the pool's precision
against the infinite-rollout pool: **≈ 0.43 at k = 1, ≈ 0.61 at k = 5** (corrected). So the honest
statement of what the idea buys is: *the corrected failure pool concentrates genuine failures
4.3× over chance at one rollout per context, 6.1× at five*. That is real leverage, and it is also
three to five times smaller than the "top decile" framing implies.

The raw pool looks far more reproducible (0.41 → 0.62), and that is exactly the trap: **50–74 % of
the raw top decile is literally the top decile of |realised move|** (measured here, per-ticker
overlap 0.50 … 0.74). The reproducible pool is the confounded one; the deconfounded pool is the
unreproducible one. That tension is the research problem, and no amount of GPU changes it.

---

## 2. The cheapest measurement that discriminates each risk

Every one of these runs on data already on disk under
`/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z/data`
or, where marked, needs generation only (no training). Times are single-core CPU on a login node,
which is inside the small-CPU-experiment allowance (no GPU, minutes, < 5 GB read, sequential).

| Risk | Discriminating measurement | Cost | Kill condition |
|---|---|---|---|
| **R1** | **Frozen-model re-scoring null.** Select the top decile from k rollouts; re-score it with k *disjoint* rollouts of the **unchanged** model; report the surviving fraction of the selection gap. | **3 min CPU** (done; §7 E3) | Already adverse: 0.18–0.23 survives at k = 1. Any training claim must be measured against a baseline re-scored the same way, or it is regression to the mean. |
| **R1b** | **Required-k curve refit with an intercept.** The published k ≈ 20 comes from a one-parameter fit forced through the origin. | **1 min CPU** (done; §6.2) | Already adverse: the implied noise/signal ratio *rises* with k in 7 of 8 tickers, and the two-parameter fit makes reliability 0.80 unreachable at any k in 7 of 8. |
| **R2** | **Held-out-seed partial.** Regress the corrected per-context score on (i) the rollouts' own dispersion and (ii) their mean generated move, both estimated from seeds **not** in either half, and re-measure split-half reliability. | **4 min CPU** (done; §6.3) | If < 40 % survives, the pool is a generated-scale pool. Result: **85–92 % survives** — R2 is *not* confirmed, and this is the strongest positive evidence the line has. |
| **R2b** | **Confounder loading of the pool.** Mean \|realised move\| in the pool ÷ population mean; overlap with the \|y\| top decile; correlation of the corrected score with \|y\| at *other* horizons. | **2 min CPU** (done; §6.4) | Adverse in part: naive pool 2.34–3.70×, corrected pool still 1.21–1.72×; the corrected score at H = 50 still correlates +0.05…+0.22 with \|y\| at H = 100…250, positive in 8/8 at H = 100. |
| **R2c** | **`num_errors` covariate.** Parse `num_errors` from each `.../member_0/inference.log`, map to context ids through `sample_indices_rank0.json`, correlate with the score. | **5 min CPU** (done; §6.4) | Adverse: split-half ρ 0.60–0.78, ρ with the score −0.19…+0.21, sign flips by ticker. Must be a pre-registered covariate. |
| **R3** | **Unbiased dispersion share.** Multiply the reported `spread_pop` share by k/(k−1); report per ticker, not as a cross-ticker mean; print the count of top-decile contexts whose bias-corrected systematic term is ≤ 0. | **1 min CPU** (done; §6.5) | Adverse: 22.8–70.7 % per ticker at H = 50 (published as "26–34 %", which is the cross-ticker mean of a downward-biased estimator). GOOG's ceiling on any training gain is 29 %, not 74 %. |
| **R4** | **Seed-only null run.** Two continued-training runs identical in every setting including the pool, differing only in the seed. | **1 extra GPU run** | If the intervention's effect is inside ±1 seed sd, stop. This is the single best-value item in the plan: it reprices every other comparison. |
| **R5** | **Two-coordinate readout on every arm** (old-window NLL and pool NLL), plus the pool-size arithmetic in §4 before any run. | **0 min** (arithmetic) + logging | If the 70/30 mix needs 58 repeats of a 5.2 M-token pool, redesign the pool before spending anything. |
| **R6** | **Re-run the three prerequisites on 2024-08 rollouts from the checkpoint the plan will use.** | generation only, **≈ 0.6 GPU-h** for 500 contexts × 5 rollouts × 8 tickers at the measured 0.354 s/rollout | If reliability, the confounder loading, or the dispersion share differ by more than the widths in §3, the 2026-01 numbers do not transfer and everything is re-derived. |

**Ordering.** R1, R1b, R2, R2b, R2c, R3 are all done and cost 16 minutes of CPU in total. R4 is one
run. R6 is under an hour of generation. **Nothing in this list justifies a training job yet.**

---

## 3. Pre-registered inequalities

Written before any of it runs. Each row: statistic → null → n → threshold. Where n is undecided,
the rule that decides it is given, not a guess.

**Global convention.** Contexts are clustered by trading day (measured here: 20 days, 1–58 contexts
per day). Every interval is a **cluster bootstrap over days**, not over contexts. The effective n
for any day-level statement is **20**, not 500. Every quantile statistic prints its event count
next to it (CONTEXT §4). No number enters a heading, gets bolded or is called "best" until n has
stopped growing; the trajectory against n is reported, not the value at the final n.

### P1 — "The failure pool is a property of the context"

| item | value |
|---|---|
| statistic | pool purity `Π(k)` = \|A ∩ B\| / \|A\|, A and B top-decile pools from disjoint k-rollout halves, corrected (rank inside realised-move deciles) score, H = 50 |
| null | Π = 0.10 (random selection) |
| n | 8 tickers × 20 days, 120 half-splits per ticker; cluster bootstrap over the 160 ticker-days |
| threshold | **Π(k) ≥ 0.50 with the 95 % cluster-bootstrap lower bound above 0.40** before any training run uses a pool built at that k |
| current reading | Π(1) = 0.22, Π(5) = 0.38 → **fails at every k available**; the smallest k satisfying it is not yet reached at k = 5 |

### P2 — "The score is not the realised move"

| item | value |
|---|---|
| statistic | (a) ρ(corrected score, \|y_h\|) at **every** horizon h ∈ {10, 25, 50, 100, 150, 200, 250}, not only the stratifying horizon; (b) mean \|y\| in the pool ÷ population mean |
| null | (a) ρ = 1/n_bins = 0.10 at the production `n_bins = 10` — this is the **analytic leak**, measured exactly (§6.4), not zero; (b) ratio = 1.00 |
| n | 8 tickers × 20 days |
| threshold | (a) max over h of \|ρ\| ≤ 0.10, i.e. no worse than the analytic leak, with the 95 % upper bound ≤ 0.15; (b) ratio ≤ 1.25 with 95 % upper bound ≤ 1.40 |
| current reading | (a) up to +0.22 at H = 200–250, positive in 8/8 at H = 100 → **fails**; (b) 1.21–1.72, mean 1.36 → **fails for 4 of 8 tickers** |

### P3 — "The pool's error is reducible"

| item | value |
|---|---|
| statistic | unbiased dispersion share inside the pool, `S = (k/(k−1))·Σ var_i(x) / Σ total_i`, per ticker per horizon |
| null | S = 1 (nothing reducible) |
| n | k = 10 rollouts, 8 tickers, 7 horizons, cluster bootstrap over days |
| threshold | report `1 − S` as the **pre-registered ceiling** on any claimed reduction; a claimed reduction exceeding `1 − S` for that ticker is evidence of a measurement error, not of learning |
| current reading | S = 0.228 (JPM) … 0.707 (GOOG) at H = 50 → ceilings 77 % … **29 %** |

### P4 — "Continual training on the pool reduces pool error"

| item | value |
|---|---|
| statistic | Δ = mean over pool contexts of (corrected score after training − corrected score before), **both re-scored with fresh rollout seeds disjoint from the selection seeds**, paired by context |
| null | the **frozen-model re-score**: the same quantity with no training at all. Measured here, that null is **not zero**: it is a gain of 77–82 % of the selection gap at k = 1, 46 % at k = 5 |
| n | decided by the power rule below; ≥ 5 seeds per arm before any claim (PLAN §3 Step 2) |
| threshold | Δ_trained − Δ_frozen > 0 with a 95 % paired cluster-bootstrap lower bound > 0, **and** the gap exceeding 1 seed sd measured on the seed-only null (P7) |

### P5 — "…more than a confounder-matched control pool"

| item | value |
|---|---|
| statistic | Δ_failure-pool − Δ_matched-pool, same token count, same schedule, same seeds |
| null | matched pool = contexts drawn from the middle of the corrected score but **matched decile-by-decile on \|y\| and on `num_errors`** |
| n | ≥ 5 paired seeds |
| threshold | difference > 0 with 95 % paired lower bound > 0. If the matched control reaches ≥ 70 % of the failure pool's gain, the mechanism is the confounder and the claim is downgraded to "a volatility-weighted mixture helps" |

### P6 — "…without unacceptable forgetting"

| item | value |
|---|---|
| statistic | old-window validation NLL on 2022-01 … 2024-07, reported **jointly** with the pool statistic on every arm (PLAN §2.4) |
| null | the pre-CPT checkpoint |
| n | same seeds |
| threshold | pre-registered tolerance: old-window NLL increase ≤ 0.010 nats/token. Above it the arm is reported as a forgetting failure regardless of its pool gain, and the teacher-KL stabiliser (PLAN §3 Step 3) is switched on |

### P7 — the noise floor (this one gates all of P4–P6)

| item | value |
|---|---|
| statistic | σ_d = sd across seeds of the paired per-arm difference, from two arms **identical in every respect including the pool**, differing only in the seed |
| null | σ_d = 0 |
| n | ≥ 4 seeds; report σ_d and its degrees of freedom next to it |
| threshold | no claim in P4–P6 is made whose effect is < 2 σ_d. σ_d from n = 2 is **not admissible** — it has one degree of freedom and is systematically small (CONTEXT §4; this repo has measured the same contrast's cross-seed sd as 0.0081 and 0.0593 on two cross-sections of the *same* runs, a factor of 7.3) |

### The power rule (and why the obvious version is wrong)

```
n_per_arm  =  (z_{1-alpha/2} + z_{1-beta})^2 * sigma_d^2 / Delta^2          (paired seeds)
           =  (1.96 + 0.84)^2 * sigma_d^2 / Delta^2
           =  7.85 * sigma_d^2 / Delta^2
```

`Delta` **must not be the pilot's point estimate.** A power calculation whose effect size is
estimated from the same small sample is systematically optimistic, because the pilot was more
likely to be run up when the estimate happened to be large, and because the estimator's own noise
inflates \|Δ̂\|. Pre-registered substitution:

```
Delta_planning  =  max( Delta_MID ,  |Delta_hat| - t_{0.975, n_pilot-1} * sigma_d / sqrt(n_pilot) )
```

with `Delta_MID` a minimum interesting difference fixed **before** the pilot. If
`Delta_planning <= 0`, the honest output is "the pilot does not bound the effect away from zero";
the answer is more seeds or a different design, not a smaller n.

Documented precedent in this repo, to be quoted in the plan so the mistake is not repeated:
an n = 8 pilot gave +1.66 %, the power calculation said 13 seeds reached t = 2, and at n = 13 the
effect had shrunk to +1.38 % and t was 1.85. **Report the trajectory of the effect against n.**

---

## 4. Null controls — one per arm, named concretely

A null control is an arm on which the intervention **provably cannot work**. Ranked by value per
run.

| # | Control | Why the intervention cannot work on it | Cost | What it reprices |
|---|---|---|---|---|
| **N1** | **Seed-only null.** Two runs identical in every setting, including the pool, differing only in `SEED`. | Nothing differs but the seed. | 1 extra run | The noise floor for every other comparison. **Best value item in the plan.** Note: neither `/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/code/attach_adaptation.sh` nor `/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/code/submit_adaptation_pair.sh` exports a seed at all, so this control is currently **unreachable with the existing scripts** (§6.7). |
| **N2** | **Frozen-model re-score.** Re-score the pool with the unchanged base model on fresh rollout seeds. | No weights changed. | **0 training GPU**, generation only | The regression-to-the-mean baseline. Measured here it recovers 77–82 % of the selection gap at k = 1. Any "improvement" smaller than this is nothing. |
| **N3** | **Permuted-pool arm.** Select the pool with the mis-paired score: apply one permutation of the context axis to the rollouts, then rank exactly as usual (this is the `shared` construction in `pairing_nulls`). | The selected contexts carry another context's rollouts, so the pool cannot contain information about which context is hard, but its size, score marginals and \|y\| loading are preserved. | 1 run | Separates "training on the *right* windows helps" from "training more on this month helps". |
| **N4** | **Anti-pool arm.** The **bottom** decile of the same corrected score, same token count. | Selected to be the contexts the model already handles best. | 1 run | If the anti-pool gains as much, the effect is extra tokens, not failure targeting. |
| **N5** | **Confounder-matched pool** (P5's control): middle-of-the-distribution contexts matched decile-by-decile on \|realised move\| and on `num_errors`. | Matched on the two known confounders, unmatched on the score. | 1 run | Separates the mechanism from the mixture. |
| **N6** | **Untouched-coordinate null.** A ticker (or a month) deliberately excluded from every pool and every replay bucket, evaluated on every arm. | The intervention never sees it. | 0 extra runs (evaluation only) | Catches the wiring. This repo has a documented case where a "control" moved +0.166 nats under the intervention, which proved the learning rate was destroying the model rather than the intervention working. **Print the drift on this coordinate on every line of the log, do not compute it only when asked.** |
| **N7** | **Provably-null score arm.** Build a pool from a score that is a deterministic function of \|y\| alone. | Contains, by construction, exactly the confounder and none of the signal. | 1 run | Puts a number on how much of any gain the confounder alone buys. |

N1 + N2 together cost one training run and zero training runs respectively, and between them they
reprice every claim the line intends to make. They are the first two things to schedule.

### Pool-size arithmetic — do this before any run

```
tokens_per_context   = 500 conditioning messages x 26 tok/message + 1 = 13,001
pool_now             = 8 tickers x 50 contexts (top decile of 500)    = 400 contexts
pool_tokens_now      = 400 x 13,001                                    = 5.20e6 tokens

issue #73's 30 % failure share of a 1 B-token stage
                     = 0.30 x 1.0e9                                    = 3.00e8 tokens
repeats_required     = 3.00e8 / 5.20e6                                 = 57.7 passes over the same 400 windows
```

Fifty-eight passes over 400 windows is not continual pre-training, it is fine-tuning on 400
examples, and R5 (drift) becomes near-certain. Filling the share without repeats:

```
contexts_needed      = 3.00e8 / 13,001 / 0.10 selection rate           = 230,752 contexts
rollout_cost         = 17 s per batch of 48 (steady state, from
                       .../hp_v5me3_GOOG_s97701/member_0/inference.log) = 0.354 s per rollout
gpu_hours(k)         = 230,752 x k x 0.354 / 3600                      = 22.7 * k GPU-hours
                       k = 5   -> 113 GPU-h  =  28 node-hours
                       k = 20  -> 454 GPU-h  = 114 node-hours
```

**This is the number the plan has to face.** The reliability requirement (§6.2: at least 16–39
rollouts per context, and possibly unreachable) multiplies the mining cost by 4× over the k = 5
figure before a single training step. The scoring pass is roughly the same order as the training
stage it feeds.

---

## 5. What would make me stop

These are stop conditions, not prompts for another mechanism hunt. Each is a specific observation
with a specific threshold, and each is checkable before or early in the line.

| # | Observation | Why it is terminal |
|---|---|---|
| **S1** | Pool purity Π(k) < 0.50 at the largest k the generation budget allows, with the reliability curve still rising in noise/signal ratio at that k. | The pool is not a set of contexts; it is a set of draws. No downstream training result can mean what it claims. **Currently at Π(5) = 0.38 with the ratio still rising in 7/8 tickers.** |
| **S2** | The confounder-matched control (N5) reaches ≥ 70 % of the failure pool's gain. | The mechanism is the mixture. Report "a volatility-weighted mixture helps by X", close the failure-pool framing, and spend the rest of the budget on the mixture knob, which is free. |
| **S3** | The trained arm minus the **frozen re-score** null (N2) has a 95 % cluster-bootstrap interval containing zero at n ≥ 5 seeds, and the point estimate has shrunk between n = 3 and n = 5. | The documented shrinking-effect pattern. Adding seeds to rescue it is the mistake this repo has made three times in one session. |
| **S4** | The claimed reduction exceeds `1 − S` (the pre-registered dispersion ceiling, P3) for the ticker it is claimed on. | Not a triumph — a measurement error. Debug the measurement, do not report the number. |
| **S5** | Old-window NLL rises by more than 0.010 nats/token on every arm that shows a pool gain, including with teacher-KL on. | The intervention trades stability for the pool and cannot be deployed; the answer is a different replay design, not a different pool. |
| **S6** | The three prerequisites re-measured on the 2024-08 slice and the main-line checkpoint (R6) give reliability, confounder loading or dispersion share outside the P1–P3 intervals from 2026-01. | The prerequisites are checkpoint- and regime-specific, so no general claim is available at this budget. Report the 2026-01 result as a case study and stop. |
| **S7** | The `num_errors` covariate explains more of the corrected score's reliable variance than the correct pairing does, on the slice actually used. | The score is measuring the sampler, not the model. |

**Explicit non-stop conditions**, so that "one more mechanism" cannot be smuggled in: a null result
in P4 does *not* justify (i) trying a new failure score, (ii) trying a new horizon, (iii) trying a
new selection quantile, or (iv) trying a new ticker subset, unless the new choice was in the
pre-registration. Each of those is a multiple-comparison lever, and this repo has a documented case
where 3/3 same-sign cells became a non-monotone 18-cell sweep once the sweep was completed.

---

## 6. Audit of the three measurements already made (commit e8425cb1)

Code read:
`/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/code/failure_pool_reliability.py`
and `/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/code/test_failure_pool_reliability.py`.
Results read:
`/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/results/failure_pool_reliability.json`.
The 12 CPU tests pass (`python3 -m pytest test_failure_pool_reliability.py -q` → `12 passed in 0.15s`).

Verdict in one line each:

| claim in the commit message | supported by the code? | correction |
|---|---|---|
| "0.36–0.48 raw, 0.15–0.25 corrected" | **partly** | The two numbers are computed on **different horizon sets** (see §6.1). |
| "one-parameter fit puts k at roughly 20" | **no** | The one-parameter model is rejected by its own residuals; the honest reading is ≥ 16–39 and rising, or unreachable (§6.2). |
| "correlates 0.65 with the realised move" | **yes** | Reproduced: 0.569–0.710, mean 0.647 at H = 50. |
| "keeps 0.46 … independent still agrees at 0.43" | **yes, but n = 1** | Single permutation draws. With 60 draws the two are indistinguishable (§6.3). |
| "drops both to 0.03 and 0.10" | **yes, but n = 1** | With 60 draws: 0.06–0.11 both, 95 % reaching 0.19. And 0.10 is exactly the **analytic leak** of the 10-bin stratification, not noise (§6.4). |
| "keeps genuine signal" | **yes — but not by the argument given** | The evidence that supports it is a partial that the code does not compute (§6.3). |
| "top-decile pools overlap by 40 percent" | **yes (as a mean)** | Range 0.30–0.56; report the range. |
| "dispersion is 26–34 percent inside the top decile" | **no** | That is the cross-ticker **mean per horizon** of a **downward-biased** estimator. Per ticker at H = 50, unbiased: 22.8–70.7 % (§6.5). |
| "the model is under-dispersed, so that is a floor" | **argument valid, input uncited** | The floor logic is right; the under-dispersion number comes from outside this code and is not reproduced with its own n. |
| "only the cross-pairing null separates them" | **construction correct, inference too strong** | `shared`/`cross` are built exactly as described. But a low `cross` does **not** discriminate the intended signal from a rollout-side property (§6.3). |

### 6.1 The raw and corrected reliabilities are computed on different horizon sets

In `main()`:

```python
sh_raw = [... split_half(real, gen, k, a.draws, rng, a.key)]                      # horizon_idx=None
sh_str = [... split_half(..., horizon_idx=a.horizon_idx, stratified=True)]        # horizon 2 only
```

`split_half` with `horizon_idx=None` averages the rank correlation over **all seven horizons**; the
stratified call uses **H = 50 only**. The JSON confirms it: `n_pairs` is 140 for the raw rows and 20
for the stratified rows. So the headline "0.36–0.48 raw vs 0.15–0.25 corrected" compares a
seven-horizon average against a single-horizon number, and `k_needed_raw` vs `k_needed_stratified`
inherit the mismatch.

Recomputed at H = 50 for both (40 draws per cell):

| ticker | raw k=1 | raw k=5 | corrected k=1 | corrected k=5 |
|---|---|---|---|---|
| AMD | 0.507 | 0.728 | 0.181 | 0.481 |
| AMZN | 0.383 | 0.622 | 0.179 | 0.444 |
| GOOG | 0.447 | 0.625 | 0.166 | 0.339 |
| INTC | 0.421 | 0.732 | 0.243 | 0.509 |
| JPM | 0.421 | 0.747 | 0.152 | 0.522 |
| META | 0.506 | 0.709 | 0.209 | 0.477 |
| MSFT | 0.447 | 0.661 | 0.192 | 0.382 |
| NFLX | 0.370 | 0.677 | 0.250 | 0.556 |
| **mean** | **0.438** | **0.688** | **0.197** | **0.464** |

The damage is small in this instance (the raw range moves from 0.36–0.48 to 0.37–0.51), which is
exactly why it survived: a defect whose numeric effect is small is a defect that nobody notices
until it is applied somewhere it matters. Fix: make `horizon_idx` mandatory on the reporting path.

### 6.2 The Spearman-Brown question, and the k ≈ 20 extrapolation

**There is no Spearman-Brown correction anywhere in the code, and the word "corrected" in the
commit message does not mean it.** "Corrected" there means *ranked inside realised-move bins*
(the `stratify` function). That is a naming collision with the standard term, and it is exactly the
documented "a metric's name is not its semantics" failure: any reader with a psychometrics
background will read "corrected split-half reliability" as disattenuated and will believe the
score is better than it is. **Rename it `stratified` or `move-conditional` everywhere.**

Is the *absence* of Spearman-Brown a defect? No, and this is worth stating so nobody "fixes" it.
Spearman-Brown converts a half-length reliability into the full-length one. Here the quantity of
interest is the reliability of a **k-rollout** score, because that is what the pool will actually
be built from, so `rho_k` measured between two disjoint k-sets is the right target and needs no
lengthening. Further, `rollouts_needed` fits

```
1/rho_k - 1  =  (n2/s2) * (1/k)
```

which **is** the Spearman-Brown prophecy formula rearranged, so the extrapolation machinery is
correct in form.

It is wrong in fit. The model has one parameter and is forced through the origin, and the residual
is reported (`max_abs_resid`) but never thresholded. Recomputing the implied noise/signal ratio
`(1/rho_k − 1)·k` at each k — a constant if the model holds:

| ticker | corrected k=1 | k=2 | k=3 | k=5 | trend | k for ρ=0.80 from the k=5 point | two-parameter intercept | implied ρ ceiling |
|---|---|---|---|---|---|---|---|---|
| AMD | 4.53 | 5.02 | 5.18 | 5.39 | rising | 21.6 | +0.29 | 0.78 |
| AMZN | 4.59 | 5.37 | 5.86 | 6.26 | rising | 25.0 | +0.53 | 0.65 |
| GOOG | 5.03 | 7.47 | 8.36 | 9.76 | rising | 39.0 | +1.50 | 0.40 |
| INTC | 3.11 | 4.07 | 4.78 | 4.82 | rising | 19.3 | +0.63 | 0.61 |
| JPM | 5.56 | 5.12 | 4.72 | 4.58 | falling | 18.3 | −0.33 | (>1, invalid) |
| META | 3.78 | 4.46 | 4.85 | 5.47 | rising | 21.9 | +0.49 | 0.67 |
| MSFT | 4.22 | 5.90 | 6.52 | 8.09 | rising | 32.3 | +1.12 | 0.47 |
| NFLX | 3.00 | 3.70 | 3.95 | 3.99 | rising | 15.9 | +0.38 | 0.72 |

Three consequences:

1. The one-parameter model is **rejected by the data it was fitted to** in 7 of 8 tickers. Its
   maximum absolute residual is 0.16–0.83 against y-values in [1.98, 5.56]; the two-parameter fit
   cuts that to 0.07–0.39.
2. Because least squares through the origin on `x = 1/k` is dominated by the k = 1 point
   (Σx² = 1.40, of which 71 % is k = 1), the fitted slope is close to the *smallest* observed
   ratio. The extrapolation is therefore **biased low**: the honest reading from the k = 5 point
   alone is **15.9 – 39.0 rollouts per context, median 21.7, and still rising**.
3. The two-parameter fit puts reliability 0.80 **out of reach at any k** in 7 of 8 tickers. That
   cannot literally be true — as k → ∞ each half converges to the same deterministic per-context
   expectation and ρ → 1 — so the apparent ceiling is a finite-k curvature artefact of taking a
   *rank* correlation of *means of heavy-tailed squared errors*. The correct conclusion is not
   "there is a ceiling of 0.40"; it is **"the linear law does not describe this regime, so no
   extrapolation from k ≤ 5 is supported, and the direction of the error is that k is larger than
   quoted."**

Required fix in the code: `rollouts_needed` must (a) return both fits, (b) refuse to emit
`k_for_rho_*` when the one-parameter residual exceeds a pre-registered fraction of the y-range, and
(c) carry a bootstrap interval, not a point estimate from four points.

### 6.3 The cross-pairing null: construction correct, inference too strong

Construction, verified line by line in `pairing_nulls`: `p1 = rng.permutation(N)` permutes the
**context axis** of `gen` while `real` stays in place, so `gen[:, p1]` scores context slot *i*'s
realised future against context `p1[i]`'s rollouts. `shared` applies the **same** `p1` to both
halves; `independent` applies `p1` to one and `p2` to the other; `cross` is the true half against
the `p1` half. That is exactly what the commit message says, and
`test_cross_null_separates_conditional_from_marginal` does go red on a shared-permutation-as-null
implementation. **This part is sound and the test earns its place.**

Two problems remain.

**(a) Every published null number is a single permutation draw.** `pairing_nulls` draws one `p`,
one `p1`, one `p2` and returns four scalars. Repeating it 60 times at H = 50, stratified, k = 5:

| ticker | true | shared | independent | cross |
|---|---|---|---|---|
| AMD | 0.476 ± 0.024 | **0.500** ± 0.029 | 0.106 ± 0.042 | 0.103 ± 0.046 |
| AMZN | 0.449 ± 0.020 | **0.461** ± 0.031 | 0.078 ± 0.050 | 0.058 ± 0.042 |
| GOOG | 0.344 ± 0.023 | **0.429** ± 0.032 | 0.098 ± 0.048 | 0.104 ± 0.044 |
| INTC | 0.503 ± 0.020 | **0.530** ± 0.029 | 0.074 ± 0.039 | 0.078 ± 0.044 |
| JPM | 0.533 ± 0.025 | **0.536** ± 0.029 | 0.078 ± 0.048 | 0.086 ± 0.041 |
| META | 0.483 ± 0.024 | **0.490** ± 0.033 | 0.092 ± 0.045 | 0.093 ± 0.044 |
| MSFT | 0.392 ± 0.018 | **0.451** ± 0.032 | 0.082 ± 0.036 | 0.081 ± 0.040 |
| NFLX | 0.544 ± 0.014 | **0.562** ± 0.024 | 0.082 ± 0.041 | 0.069 ± 0.042 |

The published "0.49 vs 0.46" understates it: **`shared` exceeds `true` in 8 of 8 tickers**, by
+0.003 to +0.085 against draw-to-draw sds of 0.02–0.03. The statement should be "a consistently
mis-paired score is *systematically more* self-consistent than the correct one", which makes the
point harder, not softer. And the published `independent` = 0.43 vs `cross` = 0.46 distinction
(raw) is inside one draw's noise; do not report a difference between them.

**(b) A low `cross` does not prove what it is being asked to prove.** `cross` is low whenever the
reliable component of the score is attached to the *correct context*. But "the model's rollouts at
this context are wide" and "the model generated a large move at this context" are **also** properties
of the correct context, and they are not what issue #73 means by failure. So `cross ≈ independent`
is consistent both with a genuine conditional-failure signal and with a pure generated-scale
ranking.

**The null that does discriminate**, and which the code does not compute: partial the corrected
score against the rollouts' own dispersion and their mean generated move, **estimated from seeds
held out of both halves** (with S = 10 and k = 3 per half, 4 seeds remain), then re-measure
split-half reliability. Measured here, 80 draws per ticker:

| ticker | ρ corrected | after removing own dispersion | after also removing \|mean generated move\| | fraction kept |
|---|---|---|---|---|
| AMD | 0.368 | 0.321 | 0.321 | 0.87 |
| AMZN | 0.342 | 0.296 | 0.296 | 0.87 |
| GOOG | 0.262 | 0.241 | 0.240 | 0.92 |
| INTC | 0.391 | 0.355 | 0.356 | 0.91 |
| JPM | 0.392 | 0.335 | 0.334 | 0.85 |
| META | 0.374 | 0.320 | 0.320 | 0.86 |
| MSFT | 0.311 | 0.269 | 0.265 | 0.85 |
| NFLX | 0.437 | 0.391 | 0.390 | 0.89 |
| **mean** | **0.360** | **0.316** | **0.315** | **0.88** |

**85–92 % of the corrected reliability survives.** So the commit message's conclusion ("keeps
genuine signal") is *correct*, but the argument offered for it is not sufficient, and this partial
is the evidence that actually establishes it. It should be added to
`failure_pool_reliability.py` and to the test file, with a known-answer test in which the score is
made a pure function of the rollouts' dispersion and the partialled reliability must fall to zero.

### 6.4 Two confounders, one of them unregistered

**The realised move.** ρ(raw score, \|y\|) = 0.569–0.710 (mean 0.647) — the published 0.65 is
reproduced. Beyond that:

| ticker | raw pool ∩ \|y\| top decile | mean \|y\| in raw pool ÷ population | mean \|y\| in corrected pool ÷ population | ρ(corrected@H50, \|y\|@H50) |
|---|---|---|---|---|
| AMD | 0.68 | 3.70 | 1.32 | 0.039 |
| AMZN | 0.50 | 2.37 | 1.21 | 0.025 |
| GOOG | 0.72 | 3.58 | 1.72 | 0.018 |
| INTC | 0.66 | 2.76 | 1.21 | 0.041 |
| JPM | 0.70 | 3.61 | 1.62 | 0.016 |
| META | 0.68 | 3.44 | 1.24 | 0.032 |
| MSFT | 0.74 | 3.16 | 1.32 | 0.020 |
| NFLX | 0.50 | 2.34 | 1.22 | 0.042 |

Two facts the commit message does not carry: **half to three-quarters of the raw pool is literally
the top decile of the realised move**, and **the corrected pool still over-weights the realised
move by 1.21–1.72×**. Two leaks explain the residual:

- **The binning leak is exactly 1/n_bins.** Feeding `stratify` a score that is a pure monotone
  function of \|y\| gives \|ρ\| = 0.2000 / 0.1000 / 0.0500 / 0.0250 at `n_bins` = 5 / 10 / 20 / 40.
  Production uses `n_bins = 10`, so the leak is **0.10** — and the observed `independent` and
  `cross` nulls sit at 0.06–0.11. **The stratified null floor is the leak, not noise.**
  The test `test_stratify_removes_a_score_that_is_only_the_realised_move` asserts \|ρ\| < 0.10 at
  `n_bins = 20`, where the leak is 0.05, so the test passes at twice the production tolerance.
- **Cross-horizon leak.** Stratifying on \|y\| at H = 50 does nothing about \|y\| at other horizons:
  ρ(corrected@H50, \|y\|@h) is +0.109 … +0.190 at H = 100 (positive in 8/8) and up to +0.216 at
  H = 250. Selection at one horizon still over-selects big realised moves at the others.

Also worth pre-registering: the pool is sensitive to the arbitrary `n_bins` choice. Overlap between
the `n_bins = 10` pool and the 5 / 20 / 40 pools is 0.66–0.86, 0.72–0.88 and 0.66–0.82 — roughly a
fifth to a third of pool membership is a consequence of a binning choice nobody registered.

**The unregistered confounder: `num_errors`.** Each `.../member_0/inference.log` prints a
per-rollout `num_errors`, and `sample_indices_rank0.json` in the same directory makes it joinable to
context ids (verified: the id sets match exactly in 8/8 tickers). Reading
`/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/crps-return-alignment-20260808/src/mid_training/return_alignment/gen_driver.py:1484`:

```python
num_errors = (l2_book_states[1:] == l2_book_states[:-1]).all(axis=1).sum()
```

It is **not** an error count. It is the number of generated messages that left the L2 book
completely unchanged — a textbook "the metric's name is not its semantics". With
`--n_gen_msgs` defaulting to 250
(`/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/crps-return-alignment-20260808/run/base_model/runtime/inference.py:112`):

| ticker | mean count | as a fraction of the 250 generated messages | cross-seed split-half ρ | ρ with corrected score | ρ with dispersion |
|---|---|---|---|---|---|
| AMD | 179.5 | 0.718 | 0.636 | −0.190 | −0.251 |
| AMZN | 165.5 | 0.662 | 0.664 | −0.174 | −0.149 |
| GOOG | 156.5 | 0.626 | 0.632 | −0.048 | −0.129 |
| INTC | 93.8 | 0.375 | 0.782 | **+0.210** | +0.254 |
| JPM | 172.1 | 0.688 | 0.598 | −0.100 | −0.152 |
| META | 192.5 | 0.770 | 0.647 | −0.075 | −0.080 |
| MSFT | 171.3 | 0.685 | 0.704 | −0.100 | −0.158 |
| NFLX | 86.2 | 0.345 | 0.691 | +0.043 | +0.163 |

Three things follow. (i) Between **34.5 % and 77.0 %** of every rollout does nothing to the visible
book, which mechanically compresses the generated return path and therefore feeds directly into
the score being used to define failure. (ii) This quantity is measured **more reliably** than the
failure score it confounds (0.60–0.78 versus 0.34–0.55 at k = 5). (iii) Its correlation with the
score **changes sign between tickers**, so a pool pooled across tickers mixes two different
selection regimes. It must be a pre-registered covariate, and the same statistic must be computed
on the *realised* paths in `.../member_0/data_real/` to establish what fraction of book-inert
messages is normal rather than pathological.

### 6.5 The decomposition: identity exact, share biased, range compressed

The identity `total = bias2_raw + spread_pop` holds exactly and is pinned by a test; the bias
correction `bias2 = bias2_raw − spread/k` is verified against a planted answer; the k = 1
degeneracy is pinned. **That machinery is right and the two-variance choice (ddof = 0 for the
partition, ddof = 1 for the correction) is correct and documented in the docstring.**

The reported *share* is not. `spread_share_top_decile` uses `spread_pop` (ddof = 0), which
underestimates σ² by (k−1)/k = 0.9 at k = 10, while the denominator `total` is unbiased for
bias² + σ². The irreducible share is therefore understated by about 11 % relative:

| ticker (H = 50, top decile) | as published (ddof = 0) | unbiased ×k/(k−1) | ceiling on any training gain |
|---|---|---|---|
| AMD | 0.295 | 0.328 | 67 % |
| AMZN | 0.339 | 0.376 | 62 % |
| **GOOG** | 0.636 | **0.707** | **29 %** |
| INTC | 0.214 | 0.238 | 76 % |
| JPM | 0.205 | 0.228 | 77 % |
| META | 0.282 | 0.313 | 69 % |
| MSFT | 0.285 | 0.317 | 68 % |
| NFLX | 0.396 | 0.440 | 56 % |

The published "26–34 percent" is the **cross-ticker mean per horizon** (0.263 at H = 250 rising to
0.343 at H = 10). The per-ticker range at a fixed horizon is 0.205–0.636, and the compression of
that range into a two-digit interval hides the ticker whose ceiling is 29 %. This is the documented
"measured one slice, wrote the whole line's conclusion" pattern in its range-compression form. Also
report `frac_top_decile_bias2_nonpositive` alongside it — it is 0.00–0.18, meaning up to 18 % of
the top decile has **no detectable systematic error at all** after the bias correction, and at
k = 1 it cannot even be computed.

The "under-dispersed, so it is a floor" argument is valid: if σ_true ≥ σ_model then a perfect model
scores σ_true² ≥ σ_model², so the removable part is at most bias². But the under-dispersion input
comes from outside this code and must be quoted with its own n and its own interval, on the same
slice, before the floor claim is made.

### 6.6 Where the tests do and do not bite

Good: the tie handling, the constant-score → NaN case, the exact partition, the bias correction
against a planted answer, and the cross-null red-on-shared-permutation test. That last one is the
best thing in the file.

Gaps, each of which corresponds to a defect above:

1. No test that the raw and stratified reporting paths use the same horizon set (§6.1).
2. `test_rollouts_needed_recovers_a_planted_noise_to_signal_ratio` plants an exactly linear curve,
   so it can never fail on a curve that is not linear. There is no test — and no threshold — on
   `max_abs_resid`, which is the quantity that actually rejects the fit on real data (§6.2).
3. The `stratify` leak test runs at `n_bins = 20` while production runs at `n_bins = 10`; the leak
   is exactly `1/n_bins`, so the test tolerance is twice as tight as production (§6.4). Add a test
   asserting the leak equals `1/n_bins` and a production-parameter test at `n_bins = 10`.
4. No test for the dispersion-partialled reliability, because the function does not exist (§6.3).
5. No test that `pairing_nulls` reports a spread; it returns four scalars from one draw (§6.3).

### 6.7 The launch scripts (blocking Step 2, and blocking the best null control)

Read:
`/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/code/attach_adaptation.sh` and
`/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/code/submit_adaptation_pair.sh`.

| # | Finding | Consequence |
|---|---|---|
| 1 | **No seed is exported anywhere**, although the header of `submit_adaptation_pair.sh` claims "Both members share every setting (data, seed, schedule, budget)". | The ≥ 5-seed requirement in PLAN §3 Step 2 is unreachable, and null control **N1 cannot be run**. A knob that is claimed in a comment and does not exist in the script. |
| 2 | `TEST_DATE_RANGE=` is exported **empty**. | If that is what selects the validation split, the validation-NLL AUC — the entire Step-2 readout — may not be produced. Verify by reading the last link of the assignment chain in the batch script before spending a GPU-hour, not after. |
| 3 | `TRAIN_DATE_RANGE=2024-08-01,2024-08-31` with `SQUASHFS_MONTHS=2024-08`: the adaptation slice and the evaluation slice are the same month. | The probe is only held out if the base checkpoint never consumed 2024-08. State the provenance explicitly with a citation, or the plasticity readout is contaminated. |
| 4 | `attach_adaptation.sh` backgrounds the run with `setsid nohup … &`. | Documented in this repo not to survive a real disconnect (the srun client lives on the login node and is the step's only control channel). Use tmux or a Slurm-managed job. |
| 5 | "Early" is **step 275**. | Step 275 is pre-warmup: this is an untrained-versus-trained contrast, not an early-versus-late plasticity contrast, and it will report PRESENT for reasons unrelated to plasticity. The retained middle checkpoint at step 33575 is the honest "early". |
| 6 | `MUON_LR=0.01` is identical for both members and is not swept. | A single learning rate can be near-optimal for one checkpoint age and wrong for the other. At minimum, log the drift on an untouched coordinate (N6) so a destroyed model is not read as a plasticity result. |

### 6.8 The A1 weight probes, in passing

`/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/results/a1_step{275,33575,69378}.json`,
one run (`j5705912`), no seeds:

| quantity | step 275 | step 33575 | step 69378 | change |
|---|---|---|---|---|
| `l2_non_embed` | 427.19 | 423.49 | 425.09 | −0.5 %, non-monotone |
| `mean_abs_non_embed` | 0.025299 | 0.024584 | 0.024135 | **−4.6 %, monotone down** |
| `spectral_norm_median` | 3.084 | 3.103 | 4.346 | **+41 %** |

Zyphra's rising-magnitude correlate goes **down** here while the spectral median goes up. A
plasticity narrative built on the spectral median has to explain why the two disagree, on one run
with no seed replicates, before either is quoted. Also, the top-5 spectral norms sit at 63.94–64.01
across all three checkpoints — pinned to within 0.1 % over 69 k steps. That is either a constraint
or a frozen subspace and is worth one line of investigation before it is reported as evidence
either way.

---

## 7. Reproduction and housekeeping

### What was run for this draft (CPU only, ~6 minutes total)

All scripts are in this session's scratchpad; the durable versions belong at
`/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/code/` as
`audit_failure_pool_premortem.py` when someone lands them. The data root is read with **one
`os.scandir`** and everything else is derived in memory; note that
`failure_pool_reliability.seeds_for()` currently re-globs that 2,308-entry directory once per
(config, ticker) — up to 24 readdirs per invocation. That is a refactor with a Lustre-metadata
justification, not a style preference.

| label | what it recomputes | result |
|---|---|---|
| E1 | reliability at H = 50 for raw and corrected, 40 draws | §6.1 |
| E2 | one- vs two-parameter fit of the reliability curve | §6.2 |
| E3 | pool purity vs k, 120 draws | §1 |
| E4 | confounder loading, binning leak, cross-horizon leak | §6.4 |
| E5 | unbiased dispersion share | §6.5 |
| E6 | held-out-seed partial of dispersion and generated-move magnitude | §6.3 |
| E7 | `num_errors` join and correlations | §6.4 |
| E8 | per-context day recovery and pool day-clustering | below |

**Day clustering (new, and needed for every interval in §3).** Recovering the date from
`.../member_0/data_real/<TICKER>_<YYYY-MM-DD>_message_real_id_<id>.csv`: 20 trading days,
2026-01-02 … 2026-01-30, 1–58 contexts per day. The top-decile pool spans 16–20 of those days
(HHI 0.060–0.099 against 0.050 for uniform; worst single day 10–22 % of the pool), so the pool is
**not** one or two bad days — a worry cheaply killed. But the clustering unit for inference is the
day: **effective n is 20, not 500**, and no interval published so far accounts for it.

### Refactor / rename list (standing instruction: always look for code to refactor and files to rename away)

1. `failure_pool_reliability.py`: cache one `os.scandir` of the data root instead of one `glob` per
   (config, ticker).
2. `failure_pool_reliability.py`: make `horizon_idx` mandatory on the reporting path so §6.1 cannot
   recur.
3. `failure_pool_reliability.py`: `rollouts_needed` returns both fits plus a bootstrap interval and
   refuses `k_for_rho_*` above a residual threshold.
4. `failure_pool_reliability.py`: `pairing_nulls` gains `n_perm` and returns mean/sd/percentiles.
5. `failure_pool_reliability.py`: add `partial_reliability(real, gen, regressor_seeds)` implementing
   §6.3, plus its known-answer test.
6. Rename "corrected" → "stratified" / "move-conditional" throughout the code, the JSON keys, the
   notebook and the commit-message vocabulary (§6.2).
7. In the sigma-0 line, rename `num_errors` → `n_book_inert_msgs` at
   `.../src/mid_training/return_alignment/gen_driver.py:1484` and the two sibling definitions,
   keeping the old key as an alias for one release (§6.4). Renaming, never deleting.
8. Nothing here needs deleting. `results/failure_pool_reliability_total_superseded_20260904T212643Z.json`
   is already correctly renamed rather than removed; leave it.

---

## 8. 中文速览

- **最可能的死法**：用一条随机分数选出来的「失败池」，在一批不相干的 rollout 上重新打分时
  大部分选不回来——k=1 时纯度只有 0.22（随机是 0.10），而**不做任何训练**、只换一批 rollout
  重新打分，就能收回 77–82% 的「选择落差」。任何训练效果都必须减掉这一段。
- **第二个死法**：原始分数与已实现涨跌幅相关 0.65，朴素池里 50–74% 就是涨跌幅最大的那一档；
  分箱校正之后仍然超配 1.21–1.72 倍。所以「训练失败样本」很可能等价于「多喂高波动窗口」，
  而后者改一行采样配置就能做到，不需要这整套机器。
- **已有三项测量的复核**：分解那部分的恒等式与偏差修正是对的；**k≈20 那个外推站不住**
  （单参数模型被自己的残差否掉，7/8 上升，诚实读数是 16–39 且仍在上升）；
  「raw 0.36–0.48 vs corrected 0.15–0.25」两个数**跑在不同的时间跨度集合上**；
  「离散度占 26–34%」是跨股票平均且用了偏低的估计量，逐股票实际是 22.8–70.7%（GOOG 70.7%，
  它的天花板只有 29%）。
- **好消息**：把「rollout 自身的离散度」和「生成移动幅度」用**留出种子**估计后偏出，
  校正后的可靠性仍保留 85–92%——所以确实存在超出「模型在这里生成得宽」之外的上下文信号。
  但支持这一点的证据不是 commit 里给的那条（cross null 低并不能区分这两者）。
- **新发现的混淆量**：`num_errors` 其实是「生成的消息里完全没有改变 L2 订单簿的条数」，
  250 条里占 34.5%–77.0%，跨种子可靠性 0.60–0.78（**比失败分数本身还可靠**），
  与校正分数的相关在不同股票上**正负相反**。必须登记为协变量。
- **现有证据全部来自 2026-01（20 个交易日）**，而计划要用的是 2024-08 / 2025-04 与 mamba3 主线，
  三项前置测量**没有一项**在计划真正要用的切片和检查点上做过。

