# D1 — End-to-end design for issue #73 ("keep training on the cases it failed to learn")

> Drafter D1, planning round 2026-09-04. No GPU job was submitted or attached while writing this.
> Every number below is tagged **[M]** measured (with the file and field it came from),
> **[D]** derived (arithmetic shown), or **[A]** assumption that still needs a measurement.
> Reproduction scripts for the new **[M]** rows are named in §11.

---

## 0. One-paragraph verdict up front

The selection side of issue #73 works, but **not in the form the issue states it**. The literal
rule — "rollouts that diverge hugely from the true data" — is `total|raw`, and two thirds of the
excess error in the pool it builds is the size of the realised move, not the model's handling of
the context **[M, §2]**. The defensible estimator is the **dispersion-corrected systematic error,
ranked inside realised-move strata** (`bias2` + `stratify`). It buys **0.591 ± 0.039 SD** of
out-of-sample excess systematic error against a null that measures **−0.002 ± 0.003 SD** **[M, §2]**.
The binding constraint is not reliability and not GPU training time: it is **rollout generation
cost, which exceeds the training cost of the data it selects by a factor of ~420 per epoch** **[D, §6]**.
Everything downstream follows from that one ratio.

---

## 1. What a "failure" is, as an estimator

### 1.1 The candidates and their cross-pairing nulls

`/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/code/failure_pool_reliability.py`
defines five per-context scores in `scores()`. Four are candidate failure definitions
(`spread_pop` is the model's own dispersion and is listed only because it exposes a trap).

Nulls at `k = 5`, `horizon_idx = 2` (H = 50 messages), mean over the 8 tickers, 500 contexts each.
`true` = correct pairing both halves; `shared` = one permutation applied to both halves;
`independent` = separate permutation per half (the zero line); `cross` = true half vs permuted half.

| score \| mode | true | shared | independent | **cross** | true − cross | verdict |
|---|---:|---:|---:|---:|---:|---|
| `total` \| raw | +0.694 | +0.705 | +0.404 | **+0.428** | +0.266 | **rejected** — 62% of the ranking survives destroying the pairing |
| `total` \| strat | +0.459 | +0.504 | +0.079 | **+0.105** | +0.354 | passes |
| `bias2_raw` \| raw | +0.681 | +0.705 | +0.525 | **+0.517** | +0.164 | **rejected** |
| `bias2_raw` \| strat | +0.359 | +0.412 | +0.078 | **+0.094** | +0.265 | passes |
| `bias2` \| raw | +0.565 | +0.633 | +0.488 | **+0.486** | +0.078 | **rejected** |
| **`bias2` \| strat** | **+0.272** | +0.316 | +0.090 | **+0.082** | +0.189 | **passes — selected** |
| `spread_pop` \| raw | +0.431 | +0.431 | +0.006 | **+0.007** | +0.424 | passes the null and is still not a failure score (see §1.3) |

**[M]** Source: `/run/.../scratchpad/d1_score_choice.json`, recomputed from the same `.npz` files
because `pairing_nulls()` at
`/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/code/failure_pool_reliability.py:198`
hard-codes `scores(...)["total"]` and therefore **ignores its own `--key` argument**; the nulls in
`/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/results/failure_pool_reliability.json`
have only ever been computed for `total`. That is a code defect, listed as **OI-8** in §10.

The CONTEXT brief's line "a score whose null is 0.43 is not a score" is `total|raw`, cross = **+0.428**.

### 1.2 The number that actually decides it: pool excess against an |y|-matched null

Rank nulls answer "does the ranking need the pairing". They do **not** answer "does the selected
pool carry excess systematic error". Those two questions have different answers, and the second is
what training consumes. Design: select the top decile from a half of 5 rollouts, evaluate on the
**disjoint** 5, in units of SD(`bias2`) across contexts, 200 draws × 8 tickers, H = 50.

| rule used to select | own pool excess | random pool with the **same \|y\|-bin counts** | **net = model signal** |
|---|---:|---:|---:|
| `total` \| raw  ← issue #73 as written | 1.345 | 0.900 (67% of it) | 0.445 ± 0.070 |
| `total` \| strat | 0.495 | 0.000 | 0.495 ± 0.067 |
| `bias2_raw` \| raw | 1.488 | 1.055 (71%) | 0.433 ± 0.072 |
| **`bias2` \| strat** | **0.592** | **0.000** | **0.591 ± 0.039** |

**[M]** `d1_null_and_power.py` / final null script, §11. SE is over the 8 tickers (n = 8, so it has
7 degrees of freedom — reported, not bolded, per the CONTEXT §4 rule on small n).

Reading: the naive rule's headline 1.345 SD is mostly a restatement of "the market moved a lot
that day". After the confound is priced, all four rules deliver a similar *net* (0.43–0.59), and
`bias2|strat` is the one whose **own** number needs no correction because its null is identically
zero. That is the reason to pick it, not its reliability — its reliability is the *worst* of the
four that pass (ρ = 0.272).

### 1.3 Two traps this section had to walk through, recorded so reviewers can check them

**Trap A — a score can pass the cross-pairing null and be useless.** `spread_pop` = `var_i(x)`
never touches the realised path, so permuting contexts permutes it along with them and `cross`
falls to +0.007. It is a perfectly reliable per-context quantity and it is not a failure score.
**The cross-pairing null is necessary, not sufficient.**

**Trap B — the obvious permutation null leaks.** Permuting the rollouts' context assignment while
keeping `real` gives a score `mean_i (x_{π(c),i} − y_c)^2` that still depends on `y_c`. Its **rank**
correlation with the true score is +0.082, i.e. near-independent, yet the pool it selects carries
**0.409 SD** — 69% of the true pool's excess:

| null construction | pool excess (SD units) |
|---|---:|
| A. true selection | +0.591 |
| B. permute rollouts' contexts, keep `real` — **not a null** | +0.409 |
| C. permute the whole score vector | +0.001 ± 0.002 |
| D. random pool matched to the true pool's \|y\|-bin counts | −0.002 ± 0.003 |

**[M]** same script. C and D are the valid nulls. B is the one that looks right and is not.

### 1.4 Why `bias2` and not `total`: the dispersion floor is real and measured

`total = bias2_raw + spread_pop` exactly (unit test
`test_total_partitions_exactly_into_bias_and_dispersion` in
`/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/code/test_failure_pool_reliability.py`).
Dispersion share of the total error **inside the selected pool**, k = 10 members, H = 50:

| pool built by | spread share | systematic share |
|---|---:|---:|
| `total` \| raw | 33.1% [20%, 64%] | 65.1% |
| `total` \| strat | 46.6% [25%, 79%] | 51.5% |
| **`bias2` \| strat** | **25.2% [16%, 34%]** | **65.8%** |

**[M]** `d1_score_choice.json`. The CONTEXT brief's "26–34% inside the top decile" is the
cross-horizon mean of `spread_share_top_decile` for the `total` pool: 34.3% at H = 10 falling
monotonically to 26.3% at H = 250 **[M]**
(`results/failure_pool_reliability.json`, `dispersion.per_horizon[*].spread_share_top_decile`).

The floor claim needs the under-dispersion fact, so it was measured rather than assumed. Rank of
the realised path among 10 ensemble members gives 11 bins; uniform puts 18.18% in the two outer
bins combined:

```
outer-2 rank-bin mass, mean over 8 tickers          uniform = 18.18%
H= 10  48.6%  ##################################################  2.67x
H= 25  35.3%  ####################################              1.94x
H= 50  30.4%  ###############################                   1.67x
H=100  28.2%  #############################                     1.55x
H=250  27.3%  ############################                      1.50x
```
**[M]** rank-histogram script, §11. The ensemble is under-dispersed at every horizon. A model moved
toward the truth would spread **wider**, so the dispersion term of the selected-on quantity is a
floor on what training removes — it does not shrink, it grows. Selecting on `total` therefore
deliberately over-weights the part of the error the intervention cannot touch.

### 1.5 DECISION 1

```
failure(c) = rank of  bias2(c) = (xbar_c - y_c)^2 - s_c^2 / k
             within a stratum of |realised move at H|,
             at H = 50 messages (primary), with H = 250 as the pre-registered
             secondary because the cross-pairing separation is widest there
             (true 0.301 / cross 0.083) and narrowest at H = 10 (0.252 / 0.156) [M].
cross-pairing null of the chosen score: cross = +0.082, independent = +0.090,
true = +0.272   (k=5, H=50, mean of 8 tickers) [M]
|y|-matched pool-excess null of the chosen score: -0.002 +- 0.003 SD [M]
```

---

## 2. A defect in the stratification that must be fixed before anything is built on it

`stratify()` at
`/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/code/failure_pool_reliability.py:213`
bins on `np.quantile(|y|, linspace(0,1,11))`. At H = 50, **101–207 of the 500 realised moves per
ticker are exactly zero**, so 2–4 of the 10 decile edges coincide and `np.digitize` produces:

| tickers | bin sizes (10 nominal deciles of 500) | empty bins |
|---|---|---:|
| AMD, AMZN, GOOG | `[0, 0, 0, 200, 50, 50, 50, 50, 50, 50]` | 3 |
| INTC, NFLX | `[0, 0, 0, 0, 250, 50, 50, 50, 50, 50]` | 4 |
| JPM, META, MSFT | `[0, 0, 150, 50, 50, 50, 50, 50, 50, 50]` | 2 |

**[M]** bin-size script, §11.

Two consequences. (i) It is not decile stratification: one stratum holds 30–50% of the sample.
(ii) `np.argsort` over per-bin ranks that live on **different grids** (a 250-member bin has
resolution 1/249, a 50-member bin 1/49) breaks the |y| balance of the global top decile — measured
maximum bin contribution to a 50-context pool is 19.4 rather than the 5 that a balanced selection
would give **[M]**. This is the mechanism behind Trap B in §1.3.

**Fix, pre-registered:** put the |y| = 0 mass in its own explicit stratum, bin the strictly positive
mass by its own quantiles, and **select the top q within each stratum** rather than the global top q
of the within-stratum rank. Then re-run every number in §1 and §3. Until that lands, the |y|-matched
random null (row D) is the only null that is valid by construction, which is why the readout in §5
is anchored to it and not to the permutation nulls.

---

## 3. k and N, derived — and why 0.80 reliability is the wrong target

### 3.1 The reliability extrapolation in the existing artifact does not support k ≈ 20

`rollouts_needed()` fits `1/rho − 1 = slope / k` through the origin. On the stratified score it
returns `k_for_rho_0.80` = 13.3 (NFLX) … 23.6 (JPM), mean 18.4 — the CONTEXT brief's "roughly 20"
**[M]** (`results/failure_pool_reliability.json`, `k_needed_stratified`). Its own residual field
says the fit is bad: `max_abs_resid` = 0.18 … **0.96**, and the residual sign pattern is systematic
(negative at k = 1, positive at k = 2, 3, 5) for 7 of 8 tickers **[M]**.

Adding the intercept the residuals demand gives `1/rho − 1 = a + b/k` with a > 0 for 7 of 8 tickers,
i.e. a **ceiling** `rho_inf = 1/(1+a)` of 0.39–0.76 (median 0.72) — under which 0.80 is unreachable
at any k **[M]**. So the two admissible fits of the same four points disagree qualitatively.

**Both are wrong.** A synthetic check with a planted per-context signal, run through the *same*
`split_half()` code with S = 10 and k ∈ {1,2,3,5}, where the true `rho_inf` is 1.0 by construction,
returns a fitted intercept a = +0.31 … +3.18 and a fabricated ceiling `rho_inf` = 0.24–0.77 **[M]**
(simulation script, §11). Cause: `rho_k` here is a Spearman correlation of a *mean of squared*
errors, not a Pearson correlation of a linear average, so the Spearman–Brown form
`rho_k = s^2/(s^2 + n^2/k)` is not the right functional form in either variant.

> **Conclusion: every published statement of the form "k ≈ 20 gives reliability 0.80" is an
> artifact of a misspecified extrapolation from k <= 5 and must not be used to size anything.**
> `rho_k` has been measured only at k in {1, 2, 3, 5} because S = 10 caps the split-half at k = 5
> (`hp_v5me3_<TICKER>_s*`: 80 directories = 8 tickers x 10 seeds **[M]**).

### 3.2 Size on the quantity the downstream comparison actually consumes

That quantity is the **out-of-sample excess of the pool**, not the ranking. Measured directly
(select on a half of size `k_sel`, evaluate on 5 disjoint rollouts, q = 0.10, H = 50):

| k_sel | 1 | 2 | 3 | 5 |
|---|---:|---:|---:|---:|
| pool excess (SD units) **[M]** | 0.379 | 0.488 | 0.534 | 0.591 |
| SE over 8 tickers | 0.050 | 0.050 | 0.042 | 0.041 |
| fit `0.738 * sqrt(k/(k+2.79))` **[D]** | 0.379 | 0.477 | 0.531 | 0.591 |

Residual <= 0.011 on all four points. Extrapolating the *fit* (flagged as extrapolation, one
functional form, four points): k = 10 -> 0.653, k = 20 -> 0.691, k = 40 -> 0.714, ceiling 0.738.

And against the selection fraction q, at k_sel = 5:

| q | 0.02 | 0.05 | **0.10** | 0.20 | 0.30 |
|---|---:|---:|---:|---:|---:|
| n_pool (of 500) | 10 | 25 | 50 | 100 | 150 |
| pool excess (SD) **[M]** | 1.453 | 0.932 | 0.591 | 0.360 | 0.259 |
| SE | 0.221 | 0.058 | 0.038 | 0.034 | 0.027 |

### 3.3 The trade: q beats k at equal cost

Generation cost is linear in `N_scored * k`, and `N_scored = N_pool / q`. At a fixed pool size:

```
    doubling k     5 -> 10 :  cost x2,  excess 0.591 -> 0.653   ( x1.10 )   [D, from the k fit]
    halving  q  0.10 -> 0.05:  cost x2,  excess 0.591 -> 0.932   ( x1.58 )   [M, measured]
```

**Score more contexts, not more rollouts per context.** k is bounded below by 2 (at k = 1 the
`s^2/k` correction is undefined and `scores()` silently returns `bias2 = bias2_raw`, i.e. the
dispersion is not removed at all — line 152 of `failure_pool_reliability.py`), and above by the
diminishing return above. **k_select = 5** sits where the fitted curve has reached 80% of its
ceiling; k = 10 would cost 2x for the last 10%.

### 3.4 What reliability the pool actually needs

At the chosen operating point (k_sel = 5, q = 0.05) the rank reliability of `bias2|strat` is
**rho ≈ 0.27** **[M]** — a third of the 0.80 convention — and the pool still carries **0.932 SD**
of genuine excess with a null of 0.000 ± 0.003 **[M]**. Reliability 0.80 is neither necessary nor
purchasable here; it is a convention imported from psychometrics, where the unit of analysis *is*
the individual score. Here the unit of analysis is the pool mean, and the pool mean of a weak
per-item score is strong because N averages the item noise away.

**The two constraints that do bind:**

| constraint | form | binds because |
|---|---|---|
| **C1 epochs over the pool** | `E = alpha * T_stage / (N_pool * 13,000 tok)`, require `E <= 4` | a 400-window pool at 30% of a 1B-token stage is 58 epochs — memorisation, not learning |
| **C2 readout precision** | `SE(Delta) = sqrt(2) * 0.4546 * k_eval^-0.669 * sqrt(50/N_eval)` SD | the effect must clear it at the pre-registered size |

`SE` fit **[M]**: median over 8 tickers of the half-to-half SD of the pool mean of `bias2`,
N = 50, at k = 1/2/3/5 → 0.4581 / 0.2837 / 0.2159 / 0.1565 SD; power fit `0.4546 * k^(-0.669)`
(between 1/sqrt(k) and 1/k, as expected for a bias-squared estimator).

### 3.5 DECISION 2

```
SELECTION   k_select = 5 rollouts/context   q = 0.05 (top 5% within each |y| stratum)
            N_scored  = N_pool / q
            N_pool    from C1: for T_stage = 1e9 tok, alpha = 0.30, E <= 4
                       N_pool >= 0.30 * 1e9 / (4 * 13,000) = 5,769 windows       [D]
                       -> N_scored = 115,380 contexts
READOUT     k_eval  = 20   N_eval = 500 contexts, selected by the SAME rule from a
            context set DISJOINT from the training pool, never trained on
            SE(Delta) = sqrt(2) * 0.4546 * 20^-0.669 * sqrt(50/500) = 0.0274 SD    [D]
            detectable fraction of the pool excess at t = 3, one seed pair:
                f = 3 * 0.0274 / 0.932 = 0.088                                    [D]
```

Detectable-f table (t = 3, single seed pair, rollout noise only — training-seed noise is **not** in
this table, see OI-3):

| N_eval \ k_eval | 5 | 10 | 20 | 40 |
|---|---:|---:|---:|---:|
| 50 | 1.110 | 0.698 | 0.439 | 0.276 |
| 200 | 0.555 | 0.349 | 0.220 | 0.138 |
| **500** | 0.351 | 0.221 | **0.139** | 0.087 |
| 2000 | 0.176 | 0.110 | 0.069 | 0.044 |

(The 0.139 cell uses gap = 0.592 at q = 0.10; at q = 0.05 the gap is 0.932 and the entry becomes
0.088 as in the box above.)

---

## 4. Pool construction

```
                 all candidate contexts in the CPT window
                                |
                 [1] sample N_scored = 115,380 contexts, stratified by ticker and by day
                                |
                 [2] k_select = 5 independent rollouts each   (0.4929 GPU-s each, M11)
                                |
                 [3] score  bias2(c) = (xbar - y)^2 - s^2/5   at H = 50 and H = 250
                                |
                 [4] stratify: |y| = 0 stratum + quantile bins of |y| > 0   (SEE SECTION 2 FIX)
                                |
                 [5] take the top q = 5% WITHIN EACH STRATUM  ->  N_pool = 5,769 windows
                                |
        +-----------------------+------------------------+
        |                       |                        |
   TRAIN POOL              EVAL POOL (disjoint       |y|-MATCHED
   5,769 windows           contexts, N = 500,        RANDOM POOL
                           k_eval = 20)              same bin counts, 5,769 windows
```

**Threshold or top-decile?** Top-q within stratum, not an absolute threshold. An absolute threshold
on `bias2` is a threshold in squared-return units, which differ by **two orders of magnitude across
tickers** (mean `bias2` 9.3e-10 for GOOG to 2.4e-8 for JPM **[M]**, `d1_sizing.json`), so a single
threshold would build a pool that is essentially "JPM and INTC". Top-q within stratum is scale-free
by construction. q = 0.05 rather than 0.10 per §3.3.

**Stratified by realised-move bin: yes, mandatory.** Without it the pool's excess is 67% confound
**[M, §1.2]**, and the CONTEXT brief's overlap number reproduces: the naive and stratified top-decile
pools share **40.0%** of their members [30%, 56%] **[M]**, and `total|raw` vs `bias2|strat` share
only **28.5%** [16%, 38%] **[M]**.

**Dispersion vs systematic inside the pool:**

| | share of the pool's total error |
|---|---:|
| systematic (`bias2_raw`), addressable | **65.8%** |
| dispersion (`spread_pop`), floor — and the model is under-dispersed so it should **rise** | **25.2%** [16%, 34%] |
| residual of the correction (`bias2` can go negative on noise; 8.3% of top-decile contexts have `bias2 <= 0`) | ~9% |

**[M]** `d1_score_choice.json` (`spread_share_in_pool`) and
`results/failure_pool_reliability.json` (`frac_top_decile_bias2_nonpositive`, mean 0.083 at H = 50).

**Ceiling on the intervention [D]:** even a perfect fix of every systematic error in the pool moves
the pool's *total* error by at most 65.8%. Any claim of a larger reduction is a measurement error or
a dispersion collapse — which is why §5 carries the under-dispersion guard.

---

## 5. The training stage

PLAN.md §3 Step 3 (`/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/PLAN.md:83-90`)
already commits to: `D_cpt = (1-rho) D_new + rho D_old`; grid peak LR in {0.3, 0.5} x pre-training
peak and rho in {0.05, 0.10, 0.25}; 1–2B-token pilots; **mandatory rewarm** (2–5% warmup, WSD or
cosine with non-zero tail); **Adam moments reset at the stage boundary**; teacher-KL off by default.
This draft keeps all of that and changes exactly one thing, with the reason recorded.

### 5.1 The one change to PLAN §3 Step 3, and why

PLAN's `rho` is the **old-data replay** share and its grid tops out at 0.25, i.e. 75–95% new data.
Issue #73 quotes the opposite convention ("70% historical replay, 30% failure"). Both cannot be
right here, and neither was derived. Constraint C1 settles it: `D_new` in PLAN Step 3 is a whole new
calendar window (large); here `D_new` is a curated pool of 5,769 windows = **75.0M tokens**
**[D]** (5,769 x 13,000). At PLAN's rho = 0.05 a 1B-token stage would be 12.7 epochs over the pool.

```
epochs over the pool  E = alpha * T_stage / (N_pool * 13,000)          [D]
   alpha = 0.95, T = 1e9, N_pool = 5,769  ->  E = 12.7   memorisation
   alpha = 0.30, T = 1e9, N_pool = 5,769  ->  E =  4.0   accepted
   alpha = 0.30, T = 1e9, N_pool =   400  ->  E = 57.7   (the pool sizes available today)
```

**Extend PLAN's grid to `alpha_failure in {0.10, 0.30}` (replay 0.90 / 0.70)** and pre-register the
reason as "C1, not a convention". This lands the issue's 70/30 inside the plan rather than beside it.

### 5.2 The stage, fully specified

| item | value | source |
|---|---|---|
| starting checkpoint | the checkpoint the rollouts came from, step 69378, 78,539,423 params, at `/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z/ckpt/wm_ft_multi3` | **[M]** `.../data/hp_v5me3_AMD_s97701/member_0/inference.log` |
| window | 500 messages x 26 tokens = 13,000 tokens/sample; loss on 500 x 21 = 10,500 | **[M]** `/lus/lfs1aip2/projects/public/u6gb/sigma-0/run/base_model/train_full_autoreg.batch:824-830` |
| stage tokens `T_stage` | 1e9 (PLAN Step 3 pilot range 1–2B) | PLAN.md:86 |
| mixture | `alpha_failure` in {0.10, 0.30}; replay drawn **stratified over the whole base window 2022-01..2024-07**, not the last month | PLAN.md:85, INVENTORY §2 |
| peak LR | 0.3 and 0.5 x pre-training peak. Pre-training `SSM_LR_BASE = 5e-4`, `LR_FACTOR = 1` -> CPT peak in {1.5e-4, 2.5e-4} | **[M]** `train_full_autoreg.batch:275-276` |
| rewarm | 3% of stage steps linear warmup (PLAN's 2–5%), then cosine to a non-zero tail of 0.1 x peak | PLAN.md:87 |
| optimiser state | Adam moments reset at the stage boundary | PLAN.md:87 |
| weight decay | 0.05, unchanged from pre-training (`OPT_CONFIG = standard`) | **[M]** `train_full_autoreg.batch:285` |
| logging | `LOG_GRAD_NORMS=1`, `CHECKPOINT_EVERY=auto`, plus all five `plasticity_probes.py` diagnostics at every eval | project CLAUDE.md; PLAN.md §3 Step 1 |
| seeds | 5 per arm; **report the effect's trajectory against n, not its value at the final n** | CONTEXT §4 |

**Why the LR must be at the low end and be checked, not assumed.** A previously recorded null arm
moved by +0.166 nats on a control the intervention could not reach, at a learning rate 320x too high
for the batch size. The no-op arm in §6 is what detects that here: if its old-window NLL moves at
all, the LR is wrong and every other arm's readout is uninterpretable.

---

## 6. The null controls

Two arms, answering two different questions. Both are trained with identical tokens, batch, schedule
and seeds to the failure arm.

| arm | pool it trains on | what it cannot do | what it prices |
|---|---|---|---|
| **N1 no-op** | `alpha = 0`: the stage is 100% base-distribution replay | cannot benefit from failure mining — no selected data is present | the **noise floor** of `Delta_pool`: everything that moves the pool metric from merely continuing to train, plus any LR-driven damage |
| **N2 \|y\|-matched** | same size, same per-stratum \|y\| counts as the failure pool, members chosen **at random within stratum** | cannot benefit from the model's error signal — the selection never consulted it | whether the mechanism is "failure mining" or "volatility reweighting" |

**N1 is the null control the design is priced against.** Its selection-side floor is already
measured: a pool built from a score carrying no information about the context has excess
**+0.001 ± 0.002 SD** (permute-both) and **−0.002 ± 0.003 SD** (|y|-matched random) **[M, §1.3]**.
What is *not* yet measured is the training-side floor — the across-seed SD of `Delta_pool` for N1.
That is **OI-3** and it is the single largest unknown in this plan.

**Do not use "permute the rollouts' contexts and keep `real`" as the null.** §1.3 row B: it retains
69% of the true pool's excess.

---

## 7. The readout, pre-registered

Definitions, all on the **held-out** eval pool of N_eval = 500 contexts (disjoint from the training
pool), k_eval = 20 rollouts, both before and after, paired by context:

```
E_pool   = out-of-sample excess of the eval pool over an |y|-matched random pool,
           measured BEFORE training. Predicted 0.932 SD at (k_sel=5, q=0.05) [M/D].
Delta_a  = mean_{c in eval pool} [ bias2_after,a(c) - bias2_before(c) ]   for arm a
f_a      = -Delta_a / E_pool        fraction of the pool's excess removed
dNLL_a   = old-window validation NLL(after, arm a) - NLL(before), nats/token,
           on the 2022-01..2024-07 base window
```

**SUCCESS requires all four, with 95% bootstrap CIs over (seed x context) and 5 seeds:**

| # | inequality | rationale |
|---|---|---|
| S1 | `f_fail - f_N1 > 0.20`, lower 95% bound `> 0` | 0.20 is 2.3x the single-seed detection limit of 0.088 **[D §3.5]**, so it is an effect the design can actually see; below it the intervention is not worth 79 GPU-h of scoring per stage |
| S2 | `f_fail - f_N2 > 0`, lower 95% bound `> 0` | otherwise the mechanism is volatility reweighting and issue #73's premise is wrong as stated |
| S3 | `dNLL_fail - dNLL_N1 < +0.005` nats/token, upper 95% bound `< +0.010` **[A]** | both coordinates always (PLAN §2.4). The absolute tolerance needs OI-3 |
| S4 | outer-2 rank-bin mass rises by `<= 2` percentage points from its baseline (30.4% at H = 50) **[M]** | the pool metric can be gamed by collapsing dispersion; the model is already under-dispersed by 1.67x, so a "win" bought that way is damage |

**DEAD — declare the idea dead, in writing, if any of:**

| # | condition | what it means |
|---|---|---|
| D1 | upper 95% bound of `f_fail - f_N1` `< 0.20` at 5 seeds | the effect is smaller than the cheapest useful size at this scale; do not extend to more seeds, extend the stage or abandon |
| D2 | CI of `f_fail - f_N2` contains 0 | "failure mining" == "train more on big-move contexts"; rewrite the issue, delete the rollout system |
| D3 | the §8 proxy test returns pool overlap `>= 0.80` **and** an indistinguishable excess | the rollout system is unnecessary; the same pool is free from a forward pass |
| D4 | `dNLL_N1` is itself non-zero beyond its CI | the LR/rewarm is damaging the model and no arm's readout means anything; fix before re-running |

**Reporting discipline (CONTEXT §4):** report `f` as a **trajectory against the number of seeds**
1, 2, 3, 4, 5. No `f` value enters a heading, gets bolded, or is called best until n stops growing.
Print the event count next to every quantile statistic. Every sentence carries its qualifier
("on `bias2|strat`, at H = 50, q = 0.05, k_eval = 20, ...").

---

## 8. The cheapest experiment that could falsify the whole idea

**F1 — Is the rollout machinery necessary at all? Cost: < 0.1 GPU-hours.**

One forward pass of the same checkpoint over the same 8 x 500 = 4,000 contexts already on disk,
recording per-context teacher-forced NLL. Then compare the top-5% pool by NLL against the top-5%
pool by `bias2|strat`, and compare their out-of-sample `bias2` excess.

```
cost  = 4,000 windows x 13,000 tok = 52.0e6 tokens, forward only
      ~ 3x the fwd+bwd rate 110.6e3 tok/s/GPU  =  332e3 tok/s/GPU        [D, M12]
      = 157 GPU-seconds  = 0.044 GPU-hours  + checkpoint load             [D]
      -> one 10-minute single-GPU job, dominated by startup
compare with  k_select = 5 rollout scoring of the same 4,000 contexts:
      4,000 x 5 x 0.4929 = 9,858 GPU-s = 2.74 GPU-hours                   [D, M11]
      ratio 63x per context; 1,264x per SELECTED context at q = 0.05
```

Falsifies if pool overlap `>= 0.80` and the NLL pool's out-of-sample `bias2` excess is inside the
`bias2|strat` pool's CI. Then issue #73 collapses to "keep training on the highest-loss windows",
which needs no rollouts, no ensemble, no dispersion correction, and no `stratify`.

This test is worth running **before** anything else in this plan, because it is the only one whose
negative result deletes the other 145 GPU-hours.

**F2 — Can training move the metric at all? Cost: ~31 GPU-hours.** The minimum viable version of
§4–§7: N_pool = 1,000 (scored 10,000 contexts at q = 0.10, k = 5), `T_stage` = 173M tokens
(`E = 4`), 3 arms x 3 seeds.

| line item | arithmetic | GPU-h |
|---|---|---:|
| score the training pool | 10,000 x 5 x 0.4929 s | 6.85 |
| score the eval pool (disjoint contexts) | 10,000 x 5 x 0.4929 s | 6.85 |
| train 3 arms x 3 seeds | 9 x 173e6 / 398e6 GPU-h | 3.91 |
| read out 10 models (9 + base) | 10 x 500 x 20 x 0.4929 s | 13.69 |
| **total** | | **31.3** |

**Full-scale version (§3.5 operating point), for comparison:**

| line item | arithmetic | GPU-h |
|---|---|---:|
| score the training pool | 115,380 x 5 x 0.4929 s | 79.0 |
| score the eval pool | 10,000 x 5 x 0.4929 s | 6.85 |
| train 3 arms x 5 seeds x 1e9 tok | 15 x 1e9 / 398e6 | 37.7 |
| read out 16 models | 16 x 500 x 20 x 0.4929 s | 21.9 |
| **total** | | **145.5** |

---

## 9. The cost ratio that governs the whole design

| quantity | value | source |
|---|---|---|
| generation | 246.44 GPU-s for 500 rollouts of ~250 messages on **1** GPU -> **0.4929 GPU-s / rollout** | **[M]** `/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z/data/hp_v5me3_AMD_s97701/member_0/inference.log`, last line; batch 48, 11.0 s pure generation per batch, 17.4–19.3 s/it wall |
| training | 13.312e6 tok/step at 0.94 s/step on 128 GPUs = 14.16e6 tok/s = **110.6e3 tok/s/GPU** = 398e6 tok/GPU-h | **[M]** `/lus/lfs1aip2/projects/public/u6gb/sigma-0/run/base_model/train_full_autoreg.batch:1141-1152`, job 2424846, 32N |
| one training window, one epoch | 13,000 / 110.6e3 = **0.1175 GPU-s** | **[D]** |
| scoring one *selected* window at k = 5, q = 0.05 | 5 x 0.4929 / 0.05 = **49.29 GPU-s** | **[D]** |
| **ratio** | **419x per epoch, 105x at E = 4** | **[D]** |

```
  per selected training window:
  scoring  |##################################################| 49.29 GPU-s
  training |#                                                 |  0.47 GPU-s (4 epochs)
```

**This is the fact the plan is built around.** Failure-pool CPT is not a training experiment with a
selection step in front of it; it is a **generation experiment with a cheap training step at the
end**. Every design choice that trades generation for anything else (q over k in §3.3, a small
high-precision eval pool in §3.5, the free NLL proxy in §8) follows from it.

Conflicting throughput evidence, unresolved (**OI-2**):
`/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/results/INVENTORY.md` §4 records
mamba3 at **0.565 s/step** (wandb `wqgghoyj`) with no node count, against 0.94 s/step at 32N here.
If the 0.565 figure is also 32N, every training GPU-hour above falls by 1.66x and the scoring:training
ratio rises to ~700x — the conclusion strengthens, so no decision in this draft depends on it.

---

## 10. Open items — each with the exact command or file that resolves it

| # | open item | exactly how to resolve |
|---|---|---|
| OI-1 | `N_GEN` (messages generated per rollout) is not recorded in the `.npz`, in `sample_indices_rank0.json`, or in `inference.log`. Inferred as ~250 from `HORIZONS[-1] = 250` and the `num_errors` cap of 249 **[M]** | `grep -rn --include='*.sh' --include='*.py' -e 'returns_multih' -e 'n_gen' -e 'num_gen' /lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z/code /lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/crps-return-alignment-20260808/run/base_model` |
| OI-2 | tokens/step and s/step conflict (0.94 s/step 32N vs 0.565 s/step, node count unrecorded). INVENTORY open item 3 | `grep -nE 's/step\|tokens_per|GLOBAL_BSZ|MSG_SEQ_LEN' <most recent mamba3 run log>`; root still unknown (OI-4) |
| OI-3 | **across-seed SD of `Delta_pool` and of `dNLL_old` for an unchanged stage.** Not measurable from rollouts. The largest unknown; S3's tolerance is **[A]** until it lands | run arm N1 (`alpha = 0`) at 5 seeds *first*, before any failure arm; its across-seed SD is the number |
| OI-4 | checkpoint roots for the R1-era 8M (`zkrtl2ef`), 78M (`pw8u0edj`), and the main-line mamba3 chain; and which long run kept a wide early/late pair | ask the user (INVENTORY §5 items 1–2 already record that this is fastest) |
| OI-5 | per-context teacher-forced NLL is stored nowhere; F1 in §8 needs it | the F1 job itself produces it |
| OI-6 | do the training windows for the pool equal the 500-message scoring context, or a wider window containing it? Affects C1 by the ratio of the two lengths | read the sampler in `/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/crps-return-alignment-20260808/run/base_model/` alongside `sample_indices_rank0.json` |
| OI-7 | all §1–§3 numbers are from checkpoint `wm_ft_multi3` step 69378 (78.5M params). Whether they transfer to the main-line mamba3 75m checkpoint is untested | re-run `failure_pool_reliability.py` on 500 contexts of rollouts from the main-line checkpoint: 500 x 5 x 0.4929 = 0.34 GPU-h per ticker |
| OI-8 | `pairing_nulls()` hard-codes `scores(...)["total"]` and ignores `--key`; `stratify()` bin degeneracy (§2) | both are single-function edits in `/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/code/failure_pool_reliability.py`; every table in §1–§3 must be re-run afterwards and this file updated in place |
| OI-9 | 2025 shard completeness for the secondary stress slice | INVENTORY §5 item 4 |

---

## 11. Reproduction

CPU-only, seconds each, no Lustre recursion. Written to the session scratchpad; **move them to**
`/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/code/` before they are needed again —
the scratchpad is node-local and does not survive a node change.

| script (scratchpad) | produces |
|---|---|
| `d1_score_choice.py` | §1.1 nulls for all four keys, §1.4 dispersion shares, §4 pool overlaps |
| `d1_sizing.py` | §3.4 SE table, metric scale, `d1_sizing.json` |
| `d1_honest.py` | §1.2 out-of-sample excess, horizon sensitivity |
| `d1_null_and_power.py` | §1.3 four nulls, §2 bin sizes, §3.5 detectable-f table |
| (inline) simulation | §3.1 fabricated-ceiling check |
| (inline) rank histogram | §1.4 under-dispersion |

Scratchpad root this session:
`/run/user/1483804540/claude-1483804540/-lus-lfs1aip2-projects-public-u6gb/275bc8e4-f92b-4b76-b93d-88d9ad5af8bd/scratchpad`

---

## 12. Order of work

```
 F1 proxy test (0.1 GPU-h) -----------------> if D3 fires, STOP and rewrite the issue
        |
 OI-8 code fixes + re-run every table in this file (0 GPU-h)
        |
 OI-3 arm N1 at 5 seeds (prices the noise floor and S3's tolerance)
        |
 F2 minimum viable, 3 arms x 3 seeds (31 GPU-h) --> if D1/D2 fires, STOP
        |
 full scale at the Section 3.5 operating point (145 GPU-h)
```

Nothing in this ladder starts before F1, because F1 is 0.07% of the total cost and can delete
the rest of it.

## 中文速览

- 「失败」的定义定为 **`bias2` 在实现波动分层内的排名**：交叉配对零线 +0.082（对照 `total|raw` 的 +0.428——那不是一个分数）。
- 决定性的一张表在 §1.2：按 issue 原文的规则选出的池子，超额误差里 **67% 只是当天行情走得远**；分层修正后净信号 0.591 ± 0.039 SD，而零效应组是 −0.002 ± 0.003。
- 「k≈20 才够 0.80」不成立——那个外推的函数形式本身是错的（合成数据上会凭空造出天花板）。真正该定尺寸的量是池子的样本外超额，按它算出 **k=5、q=0.05**；同样的预算下「多测几个上下文」比「每个多滚几次」划算 1.58 : 1.10。
- 全局成本比：**生成一个被选中窗口的滚动样本，比训练它一个 epoch 贵 419 倍**。所以这是一个生成实验，不是训练实验。
- 最便宜的证伪只要 **0.1 GPU 小时**：用一次前向的 teacher-forced NLL 排序，如果和滚动打分选出的池子重合 ≥80%，整套滚动机制就没有存在必要。
