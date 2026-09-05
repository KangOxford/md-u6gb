# R2 — adversarial review, lens: statistical inference and estimand validity

> Reviewer 2 of five, launched alone. Lens: the rank correlations between per-context scores,
> the null constructions, and the extrapolations that every conclusion on this line rests on.
> Everything below was recomputed on the current archive at
> `/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z/data/`
> using
> `/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/code/failure_pool_reliability.py`
> as a library. CPU only, no GPU, no job submitted.
>
> **Every measured number below is reproduced by**
> `/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/plan_drafts/R2_checks.py`:
>
> ```
> cd /lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning
> python3 plan_drafts/R2_checks.py           # all checks, about eight minutes
> python3 plan_drafts/R2_checks.py f1 f3     # the two BLOCKING ones, about three
> ```
>
> Each check prints the published figure beside the measured one, so a disagreement is visible
> without cross-referencing this file.

## 中文速览

- **三条 BLOCKING。**
  1. **`num_errors` 那条「REFUTED」是在两个不同的 k 上比的**：发表的 0.229–0.419 是**单个 rollout 对单个 rollout**（k=1），拿去比分数的 k=5 读数 0.330–0.545。在同一个 k 上重测，`num_errors` 在 **8/8 只票**上都比分数更可复现（k=1: 0.229–0.461 对 0.153–0.248；k=5: 0.605–0.793 对 0.333–0.545）。**原始审计的说法是对的，`PLAN` §0.11 与 RESULTS 附录 5 §B 记的结论要翻过来。** 这和本项目自己修过的 A1 缺陷（raw 与 stratified 在不同视界集上比）是同一个形状，只是换了一根轴。
  2. **`02` §1.1 的验收标准 E1a（Δρ ≥ 0.70）在任何 k 上都达不到。** 实测 k=3 时 0.155–0.384、k=5 时 0.232–0.490，**0/8 通过**；而 `02` 自己 §3.1 的两参数渐近线说 **7/8 只票的 ρ 上限低于 0.80**，所以 Δρ ≥ 0.70 不是「还没到」，是**结构上到不了**。这道关卡要么事后放宽，要么整条 Thread B 一开始就被登记为必然失败。
  3. **「什么都不做」的底噪与它要标定的效应结构不同。** repro/repB 是**同种子配对**的两次重生成，H10/H50/H250 上分别有 **97%/65%/19% 的成员逐位完全相同**，差值里那部分精确抵消；而两个不同模型之间不可能有任何逐位相同。用同一批数据把「同种子配对」和「交叉种子」并排测一遍，底噪分别被低估 **3.70× / 1.85× / 1.08×**（按只票取比值再平均是 4.47/2.16/1.19）。§3.4 的整张 MDE 表、§4 的 12.24% 与 33.24% 两个格子门槛全部继承这个低估。

- **七条 MAJOR。** `1/n_bins` 不是任何零分布的下界（合成实验：leak=0.100 时零分布读 1.000，leak=0.014 时读 0.041；真实数据 5/8 只票低于 0.10，八票均值 0.084 对 0.10 是 t=−2.21）；RESULTS §F「余量翻倍（3.5×→7.4×）」是**换了分母**换出来的，统一分母只有 1.33×，按 `02` §1.1 自己写的「超出量」读则是 **+0.0015，t=+0.22，8 只票里 4 只为正**；`dispersion_partial_floor` 的地板是在**退化到只有一个分层**的合成数据上、且只在信号=0 这一个点上标定的，而真实数据与它落在曲线完全不同的位置（干扰项与分数的秩相关 0.16–0.30 对合成的 0.81）；「只有 R=N·k 重要」是零分布的性质，效应一旦逐 context 异质，N=100 k=5 比 N=500 k=1 差 27%；「17–41 中位 22 不作预算」与同一文件第 723 行「预算退回 17–41 中位 22」直接矛盾，`05` 还据它逐票判 k=20/k=24 的通过与否；§1.1 的设计效应表用最大的一只票算的，`dataset_length` 在 25,018–226,002 之间差 9 倍，JPM 在 N=500 时设计效应已是 1.90 而非 1.01；§2.3 的数在今天的档案上复现不出来，且最会动的正是头条（池内 +7.99% t=1.74 → **+2.58% t=0.48**）。

- **确认干净的**：28 个重复生成的 context 的映射在 **8/8 只票**上成立（不止验证过的 3 只），`.npz` 里存的是第二次生成，**去掉这 28 个对已发表的数字没有实质影响**；φ 模型复现；largest-k 估计复现；`shared > true` 换成「只票为重抽单位」后**依然成立**（k=3 t=+3.06、k=5 t=+2.69，留一只票后 t 仍在 2.2–4.5）。

---

## Findings

| # | severity | file / section | claim attacked | defect |
|---|---|---|---|---|
| F1 | **BLOCKING** | `results/RESULTS_20260905.md` addendum 5 §B; `PLAN.md` §0.11, §0.12 | "`num_errors` … is measured **more** reliably than the failure score — **REFUTED** … 0.229–0.419, *below* the failure score's 0.330–0.545 at k=5" | the two sides are at different `k`. `num_errors` is at k=1, the score at k=5. At matched k the direction reverses, 8/8 tickers |
| F2 | **BLOCKING** | `plan_drafts/02_statistics.md` §1.1 (E1a) | "`Drho(t,h,k) = rho_true - rho_indep >= 0.70`" | 0/8 tickers reach it at k=3 or k=5, and `02` §3.1's own asymptote puts 7/8 tickers below ρ=0.80 at **any** k. The gate cannot be passed |
| F3 | **BLOCKING** | `plan_drafts/02_statistics.md` §2.3, §2.4, §3.4, §4.1, §4.2; `PLAN.md` §0.1 F7 | "two whole regenerations of the control arm … differenced" is the floor for the arm-level endpoints | the two regenerations are seed-matched and share φ = 0.97 / 0.65 / 0.19 of their members bitwise at H10 / H50 / H250; a control-vs-treatment contrast can share none. Measured understatement 3.70× / 1.85× / 1.08× |
| F4 | MAJOR | `code/failure_pool_reliability.py:240` `stratification_leak`; `RESULTS` §2, §0.5 A3; `results/nulls_and_partials.json` | "Any null measured on a stratified score has this as its floor" (= `1/n_bins` = 0.10) | `1/n_bins` is the correlation of a *pure-|y|* stratified score **with |y|**, not a bound on the correlation between two independently mis-paired scores. It is neither an upper nor a lower bound; 5/8 measured `independent` values sit below it |
| F5 | MAJOR | `RESULTS` addendum 5 §F | "the margin the conclusion actually rests on roughly doubles" (3.5× → 7.4×) | the two ratios use different denominators (leak for v1, `independent` null for v2). Held to one denominator: 1.33×. On the excess `true − independent` that `02` §1.1 writes E1a on: **+0.0015, t = +0.22, positive in 4/8 tickers** |
| F6 | MAJOR | `code/failure_pool_reliability.py:583` `dispersion_partial_floor`; `RESULTS` §3 | "a score that is *entirely* dispersion keeps **0.48** under the identical procedure" | the procedure is not identical: with `real = 0` for every context, `stratify` collapses to **one** stratum, so the floor is measured unstratified. And it is a one-point calibration: the real data sits at score-nuisance rank correlation 0.16–0.30, the synthetic at 0.81 |
| F7 | MAJOR | `plan_drafts/02_statistics.md` §3.3 | "at fixed rollout budget the split between contexts and rollouts is irrelevant … the *evaluation* budget can be split however is convenient" | measured on a null where the per-context difference has no systematic part. With a per-context heterogeneous effect, `N=100 k=5` costs 27% more sd than `N=500 k=1` at the same R |
| F8 | MAJOR | `RESULTS` addendum 2 §D ("Neither figure is used as a budget") vs addendum 5 §D; `PLAN.md` §0.3, §0.9, §0.10; `plan_drafts/05_execution.md` §2 | the `k ≤ 5` extrapolation is not used as a budget | it is, in five places, including the same file 330 lines later, and `05` reads per-ticker pass/fail at k=20 vs k=24 off it |
| F9 | MAJOR | `plan_drafts/02_statistics.md` §1.1 (design-effect table) | "context-level i.i.d. resampling is defensible at today's `N = 500`" | the table is computed with `dataset_length = 226,002`, the largest of the eight. `dataset_length` spans 25,018 (JPM) to 226,002 (GOOG). At N=500 the design effect is 1.01 for GOOG but **1.90 for JPM** and 1.61 for META |
| F10 | MAJOR | `plan_drafts/02_statistics.md` §2.3 | "Doing nothing produced an apparent 8% improvement at t = 1.74" ; "The geometric mean is disqualified: 36% RMS" | neither reproduces on today's archive: **+2.58%, t = 0.48**, and geometric-mean RMS **4.35%**. `M` has 21–84 exact zeros per ticker, so the geometric mean depends entirely on an unstated zero-handling rule |
| F11 | MINOR | `code/failure_pool_reliability.py:181` `stratify_v2`; `PLAN.md` §0.1 D1 | "`stratify_v2` is what any new pool should use" | nothing can call it. No call site in `pairing_nulls`, `split_half`, `regeneration_null`, `dispersion_partialled_reliability`, `pool_overlap` or `main`, and no CLI flag |
| F12 | MINOR | `plan_drafts/02_statistics.md` §6 trap T8 vs §1.1 E1b | T8: `assert abs(nulls["cross"] - nulls["independent"]) < 2*se` | E1b is written one-sided; T8 is two-sided. The two-sided form blocks GOOG and META at k=5 on the benign direction (`cross` **below** `independent`) |
| F13 | MINOR | `RESULTS` §2, §3 (the `±` columns) | per-ticker `±` presented as the uncertainty | those are **within-ticker draw** sds, conditional on these 500 contexts and 10 seeds. They are 3–4× the between-ticker se of the quantity actually claimed, so the published presentation is weaker than its own result |
| F14 | MINOR | `RESULTS` addendum 5 §A | "28 of 500 contexts (5.6%)" | those 28 are `rank_indices[0..27]`, the 28 **lowest dataset indices** — a contiguous early block spanning 4.6–7.9% of each ticker's index range, which is exactly the correlated unit `02` §1.1 defines |

---

## F1 — BLOCKING. The `num_errors` refutation compares k=1 against k=5

**The claim.** `results/RESULTS_20260905.md` addendum 5 §B:

> | It is measured **more** reliably than the failure score (0.60–0.78 vs 0.34–0.55) | **REFUTED** |
> With the join, cross-seed rank correlation is **0.229–0.419**, *below* the failure score's
> 0.330–0.545 at k=5.

Carried into `PLAN.md` §0.11 ("the audit's claim … is **REFUTED** (0.229–0.419 against the
score's 0.330–0.545)") and §0.12.

**Why it fails.** The score's `0.330–0.545` is a split-half correlation between two **5-rollout**
means. The `num_errors` figure is a correlation between **one member and one member**. Reliability
of a mean of `k` draws rises with `k`; comparing a k=1 estimate against a k=5 estimate is the same
defect this project already found and fixed as A1 (`raw` averaged over 7 horizons compared against
`stratified` at 1), moved from the horizon axis to the `k` axis.

**Measured, at matched k**, using the addendum-5 join
(`num_errors[i] ↔ rank_indices[i]` for `28 ≤ i < 500`, `num_errors[500+j] ↔ rank_indices[j]`),
60 draws, horizon 50:

| quantity | k = 1 | k = 5 |
|---|---|---|
| `num_errors` | **0.229 – 0.461** | **0.605 – 0.793** |
| stratified failure score | 0.153 – 0.248 | 0.333 – 0.545 |
| raw failure score | 0.379 – 0.512 | 0.631 – 0.747 |

The lower bound 0.229 matches the published figure exactly, which identifies the published number
as k=1. Against the **stratified** score — the score the pool would use, and the one the published
comparison quotes — `num_errors` is more reliable in **8/8 tickers at k=1 and 8/8 at k=5**.
Against the **raw** score it loses at k=1 and is comparable at k=5.

**The scenario in which the plan produces a wrong answer.** `num_errors` was demoted from "a
better-measured quantity than the score it confounds" to "a covariate whose only justification is
the sign flip". If a pool is built at k=3 and `num_errors` is treated as noisier than the score,
the confounder is under-controlled: it is in fact the *cleaner* of the two measurements at every
matched k, so a pool that is 60% book-inert rollouts can be selected by a quantity that is
measured better than the thing being selected on. Nothing in the plan currently blocks that.

**Cheapest check.** The command in the scratchpad, one minute, CPU only — or, minimally:
compute `split_half(..., k=1)` on the stratified score and correlate two single members'
`num_errors` arrays. The verdict flips on the first ticker.

**What the plan must say instead.** The comparison is `k`-dependent and score-dependent; state
both. `num_errors` at matched k beats the stratified score and ties the raw score.

---

## F2 — BLOCKING. E1a's acceptance target cannot be reached at any k

**The claim.** `plan_drafts/02_statistics.md` §1.1:

> ```
>   E1a  excess       Drho(t,h,k)       = rho_true(t,h,k) - rho_indep(t,h,k)      >=  0.70
> ```

**Why it fails.** Measured, stratified score, horizon 50, 60 draws:

| k | Δρ per ticker | max | passes ≥ 0.70 |
|---|---|---|---|
| 3 | 0.263 0.285 0.155 0.314 0.311 0.269 0.248 0.384 | 0.384 | **0/8** |
| 5 | 0.374 0.391 0.232 0.437 0.427 0.359 0.328 0.490 | 0.490 | **0/8** |

That alone is only "not yet reached". What makes it structural is `02` §3.1's own two-parameter
fit, which I reproduce: `rho_inf = 1/a` from `1/rho_k = a + b/k` gives
`0.795, 0.654, 0.410, 0.613, 1.223 (degenerate — a correlation above 1 means the fit carries no
information), 0.633, 0.550, 0.780` — **7 of 8 below 0.80**, matching the draft. Since
`rho_indep ≈ 0.08`, `Δρ ≥ 0.70` needs `rho_true ≥ 0.78`, which is above the asymptote for **5 of 8
tickers at every k**; AMD (0.795) and NFLX (0.780) are marginal and JPM's estimate is degenerate.
So the target is out of reach for the majority of the panel no matter how many rollouts are bought.

**The scenario.** `PLAN.md` §0.3 deploys `k = 3`. E1 gates "the failure pool is real". Either the
gate is silently relaxed after the numbers arrive — which is what a target set above the ceiling
always produces — or Thread B is pre-registered to fail on a criterion nobody can meet. Both are
worse than a target chosen from the asymptote.

**Cheapest check.** Already run above; `rollouts_needed` emits `two_param_intercept` for every
ticker, so `1/intercept_a` is one line of JSON away.

**Fix.** Write E1a against a target derived from the measured asymptote (e.g. `Δρ ≥ 0.30` at the
deployed k, with the ceiling reported beside it), or move the estimand off per-context reliability
entirely — which is what `02` §3.2's own "the deliverable is a training set, not a ranked list"
argument implies.

---

## F3 — BLOCKING. The do-nothing floor shares its draws with itself

**The claim.** `plan_drafts/02_statistics.md` §2.1/§2.3:

> | E2a broad improvement | two whole regenerations of the control arm, same seeds, same
> contexts, differenced with `Trim_{0.05}` | … | **Yes** |

and §2.4:

> estimate the floor as the spread of the do-nothing gap across the 8 tickers (7 df)

`PLAN.md` §0.1 F7 records the result as "Doing nothing moves an arm-level endpoint by up to 28%".

**Why it fails, mechanically.** `02` §2.2 measures — and I reproduce to ±0.005 — the fraction φ of
`(context, member, horizon)` forward returns that are **bitwise identical** between the two
regenerations:

```
horizon      H10    H25    H50   H100   H150   H200   H250
phi        0.971  0.865  0.650  0.389  0.279  0.227  0.190      (published 0.976 … 0.191)
```

A bitwise-identical member contributes the same value to both sides and **cancels exactly** in the
difference, so the do-nothing gap has variance proportional to `(1 − φ)`. The contrast this floor
is meant to calibrate is control-arm versus treated-arm: two different weight vectors cannot
produce bitwise-identical rollouts, so its variance carries the full `1`. **§2.2 establishes the
premise and §2.3 uses the pair anyway.**

**Measured, on the same two configs, same statistic, same k = 2, changing only the seed pairing.**
`matched` = `repro[{a,b}]` vs `repB[{a,b}]`; `crossed` = `repro[{a,b}]` vs `repB[{c,d}]`.
RMS of the 5%-trimmed-mean relative gap over all seed pairs, mean over 8 tickers:

| horizon | φ | matched | crossed | ratio | predicted `1/√(1−φ)` | crossed > matched, per ticker |
|---|---:|---:|---:|---:|---:|---:|
| H10 | 0.971 | 2.39% | 8.85% | **3.70×** | 5.85 | 8/8 |
| H50 | 0.650 | 3.26% | 6.05% | **1.85×** | 1.69 | 5/8 |
| H250 | 0.190 | 4.91% | 5.30% | 1.08× | 1.11 | 3/8 |

The ratio tracks the prediction across horizons and is ordered exactly as φ predicts — at H = 50
the agreement is 1.85 measured against 1.69 predicted. (Taking the ratio per ticker and then
averaging gives the larger 4.47 / 2.16 / 1.19; the table uses the ratio of the pooled figures,
which is the more conservative reading and is the form the floor is actually quoted in.) At the
plan's working horizon of 50 the floor is understated by about 1.9×; at H = 10 by about 3.7×.

**The scenario.** Every downstream number inherits it:

- `02` §3.4's MDE table is anchored on `sd = 10.76%` at `R = 500`. Corrected for structure at
  H = 50 that is about `20%`, so the `n_t = 8` MDE for the broad endpoint moves from 12.40% to
  about 23%, and the pool-restricted MDE from 15.6% to about 29%.
- `02` §4.1's cell-level threshold `p95 = 12.24%` and family-wise `max = 33.24%` are read off the
  same 56 do-nothing cells, and the cells span all seven horizons, where the correction factor
  runs from 1.08 to 3.70. Both thresholds are too lenient, most severely on the short-horizon
  cells, so a treated arm moving a H = 10 cell by 8% would clear a threshold that should be near
  30% there.
- `PLAN.md` §0.1 F7's "up to 28%" is the *matched* maximum; the crossed maximum is larger.

**Caveat, stated honestly.** `crossed` changes the seed labels as well as removing the bitwise
sharing, so it is an upper bound on the correction unless the seed label carries nothing but the
draw — which is what a seed is. The unavoidable point is narrower and survives that caveat: the
floor and the effect must have the same sharing fraction, the floor's is φ, the effect's is
unmeasured and is zero by construction if any weight changed. Whichever way that is resolved, it
has to be measured, not assumed.

**Cheapest check.** Two lines: `(B[:,:,h] == C[:,:,h]).mean()` for φ, then re-run the §2.3 table
with `repro[{a,b}]` against `repB[{c,d}]`. Under three minutes for all 8 tickers.

---

## F4 — MAJOR. `1/n_bins` is not the floor of any null

**The claim.** `code/failure_pool_reliability.py:240`:

> ```python
> def stratification_leak(n_bins: int = 10) -> float:
>     """The residual |y| correlation that stratification cannot remove, = 1 / n_bins.
>     … Any null measured on a stratified score has this as its floor; 0.095 measured at
>     n_bins = 10 is the floor reached, not noise."""
> ```

and `pairing_nulls_repeated`'s docstring ("for a stratified score any null is bounded below by
`stratification_leak`"), and the value is written into
`/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/results/nulls_and_partials.json`
as `stratification_leak: 0.1`. `RESULTS` §2: "the analytic floor of a 10-bin stratified null is
**0.10**. They are at the floor, not at zero." `PLAN.md` §0.5 A3: "The floor is 0.10 by
construction."

**Why it fails.** The identity is correct for what it actually computes: for a score that is a
monotone function of `|y|`, `corr(stratify(score), |y|) = 1/n_bins` to four decimals. That is a
statement about the correlation of the *stratified score with |y|*. The `independent` null is a
different quantity — the correlation between **two independently mis-paired stratified scores** —
and the two are not related by any inequality. Synthetic, `score = |y| + noise`, both halves,
10 bins, 40 draws:

| noise sd | leak = corr(strat score, \|y\|) | `independent` null |
|---:|---:|---:|
| 0.0 | **0.100** | **1.000** |
| 0.5 | 0.014 | 0.041 |
| 1.0 | 0.008 | 0.018 |
| 4.0 | 0.005 | −0.009 |

At noise = 0 the leak is exactly `1/n_bins` and the null is 1.000, so the leak is not an upper
bound. At noise = 0.5 the leak is 0.014 and the null is 0.041, so it is not a lower bound either.
The leak is diluted by the between-bin geometry; the null is not.

**On the real data**, k=5, 60 draws, horizon 50:

```
independent per ticker : 0.105 0.062 0.101 0.067 0.088 0.107 0.088 0.054
mean 0.0839, between-ticker sd 0.0206, se 0.0073
  vs the claimed 0.100 floor : t = -2.21 on 7 df,  5/8 tickers strictly below
  vs the MEASURED per-ticker leak (0.080-0.171, mean 0.128) : t = -6.10
```

`RESULTS` addendum 5 §F already corrected the "exactly 0.10" half (the leak on the real `|y|` is
0.080–0.171, not 0.10) but kept the class error: the leak is still being used as the floor of a
null.

**A self-test that cannot fail.** `code/test_failure_pool_reliability.py:233`:

```python
def test_repeated_nulls_carry_the_leak_floor_when_stratified():
    """A stratified null is bounded below by 1/n_bins; the reading must say so."""
    real, gen = np.zeros((300, 1)), rng.normal(size=(9, 300, 1))
    assert F.pairing_nulls_repeated(...)["leak_floor"] == pytest.approx(F.stratification_leak())
```

The assertion is `1/10 == 1/10`; the claim in the docstring is never tested. Run on its own
fixture the null reads **`independent = −0.0114`** against `leak_floor = 0.10` — the fixture
violates the docstring by 0.11 and the test passes. (The fixture also has `real = 0` everywhere,
so `stratify` produces **one** non-empty stratum, not ten.)

**The scenario.** The direction is *favourable* here — the true zero line is lower than claimed,
so the evidence for a real conditional signal is stronger, not weaker. The damage is in the other
direction: any future null measured near 0.05 will be dismissed as "at the floor" when it is real
structure, and the ratio arithmetic in F5 is built on it.

**Fix.** Delete the "floor" language. `independent` is directly measured; report `true − independent`,
which is what `02` §1.1 E1a already does correctly. Keep `stratification_leak` as what it is —
`corr(stratified pure-|y| score, |y|)` — and rename it so the name is its semantics.

---

## F5 — MAJOR. "The margin roughly doubles" is a denominator swap

**The claim.** `RESULTS` addendum 5 §F:

> | signal against its own floor | 0.460 / 0.13 ≈ **3.5×** | 0.438 / 0.059 ≈ **7.4×** |
>
> **The conclusion strengthens.** … the margin the conclusion actually rests on roughly doubles.

**Why it fails.** The `0.13` in the v1 column is the measured **leak** (`corr` with `|y|`). The
`0.059` in the v2 column is the measured **`independent` null**. Those are two different
quantities (F4), and the same table reports v1's independent null separately as "≈ 0.10".

**Measured**, k=5, 40 draws, horizon 50, both stratifiers on identical draws:

| | production `stratify` | `stratify_v2` |
|---|---:|---:|
| reliability (`true`) | 0.460 | 0.438 |
| `independent` null | 0.082 | 0.059 |
| leak on a pure-\|y\| score | 0.128 | 0.057 |

(These reproduce §F's own figures: 0.460 / 0.438, leak 0.13 / 0.057, v2 independent 0.059.)

| reading | v1 | v2 | change |
|---|---:|---:|---:|
| ratio over the **leak** | 3.59× | 7.63× | 2.13× |
| ratio over the **independent null** | 5.62× | 7.48× | 1.33× |
| **excess `true − independent`** (the E1a quantity) | 0.378 | 0.380 | **+0.0015** |

Paired over the 8 tickers the excess changes by `+0.0015 ± 0.0195`, **t = +0.22 on 7 df, positive
in 4 of 8 tickers** — a coin flip.

**Why the excess is the right reading and the ratio is not.** Two rank correlations do not compose
multiplicatively; `ρ_true / ρ_indep` is not a signal-to-noise ratio of anything. `02` §1.1 already
writes the acceptance criterion on `ρ_true − ρ_indep` and gives the reason ("the zero line is not
zero"). `RESULTS` §F then reports a ratio instead, with two different denominators.

**The scenario.** `stratify_v2` is a genuine improvement in the *construction* (F14 of D1: one
stratum holding 20–50% of the sample is not a decile stratification). Reporting it as a doubling
of the evidence sets an expectation that the next measurement will not meet, and invites the same
ratio arithmetic to be applied to the arm-level endpoints where it is meaningless.

**Cheapest check.** Recompute both stratifiers on the same draws and print `true − independent`.
Two minutes.

**Fix.** State it as: v2 removes a construction artefact and leaves the evidence unchanged
(`Δ excess = +0.002, t = 0.22`). That is a good outcome and it is defensible.

---

## F6 — MAJOR. The dispersion-partial floor is a one-point calibration on a degenerate fixture

**The claim.** `code/failure_pool_reliability.py:583` and `RESULTS` §3:

> Measured on a synthetic score that is *entirely* dispersion … the fraction kept is **0.48 with
> 10 seeds** … So the reading is **"0.87 against a 0.48 floor"** … the margin is roughly half what
> "0.87 of it survives" implies.

**Defect 1 — the floor is not measured under the same construction.** `dispersion_partial_floor`
sets `real = np.zeros((n_ctx, 1))`. Inside `dispersion_partialled_reliability` the score is then
`stratify(total, real[:, h])`, and with `|y| = 0` for every context `np.quantile` returns ten
coincident edges, `np.digitize` sends all 500 contexts to bin 9, and **the floor is measured with
one stratum**:

```
unique bin ids: [9]   counts: [0 0 0 0 0 0 0 0 0 500]
```

The production measurement has 6–9 non-empty strata (`RESULTS` §F's own table). Re-running the
same control with a non-degenerate `|y|` moves the pure-dispersion floor from **0.495 to 0.550**.

**Defect 2 — one point on the signal axis cannot calibrate a scale.** `fraction_kept` swept
against a planted per-context systematic error (the thing that *is* "being wrong"):

| planted signal sd | 0.00 | 0.50 | 1.00 | 2.00 | 4.00 |
|---|---:|---:|---:|---:|---:|
| `fraction_kept`, degenerate \|y\| (as the floor is measured) | 0.495 | 0.539 | 0.634 | 0.658 | 0.673 |
| `fraction_kept`, non-degenerate \|y\| | 0.530 | 0.646 | 0.766 | 0.839 | 0.836 |

The scale is not `[floor, 1]`. It runs from about 0.50 to about 0.84 over the whole range from
pure dispersion to signal-dominated, and it flattens at the top rather than approaching 1. The
measured 0.87 lies **above** the signal-dominated end of that scale. "Roughly half the margin" has
no calibration behind it in either direction.

**Defect 3 — `fraction_kept` is a restatement of one correlation.** In the rank-linear model
`fraction_kept = (ρ − a²)/(ρ(1 − a²))` where `a = corr(score, nuisance)`. Measured at k=3,
60 draws, horizon 50:

| ticker | ρ | ρ partialled | kept | `a_spread` | `a_move` | kept predicted from `a` alone |
|---|---:|---:|---:|---:|---:|---:|
| AMD | 0.360 | 0.315 | 0.876 | 0.254 | 0.176 | 0.813 |
| GOOG | 0.255 | 0.234 | 0.921 | 0.163 | 0.116 | 0.878 |
| JPM | 0.378 | 0.319 | 0.842 | 0.296 | 0.189 | 0.769 |
| … | | | mean 0.875 | 0.163–0.296 | 0.115–0.190 | mean 0.821 |

The prediction from `a` alone reproduces the ticker ordering and is within 0.05 of the measured
value. The **synthetic floor sits at `a_spread = 0.810`, `a_move = 0.636`** — a completely
different part of the curve, so far that the formula returns a value above 1.

**The scenario.** "0.87 against a 0.48 floor" reads as "roughly half of the reliability is real".
The defensible statement is much smaller and much more useful: **the held-out dispersion proxy
correlates with the stratified score at only ρ = 0.16–0.30 in rank, and the mean generated move
at 0.12–0.19, so about 4–12% of the score's rank variance is dispersion-explainable.**
Everything the partialling machinery reports follows from those two numbers.

**A more direct measurement the plan should make instead.** `scores()` already returns the exact
decomposition. Split-half reliability of each term, k=5, stratified, horizon 50, 60 draws:

| ticker | `total` | `spread_pop` (dispersion only) | `bias2_raw` | `bias2` (bias-corrected) |
|---|---:|---:|---:|---:|
| mean over 8 | **0.462** | **0.388** | 0.370 | **0.269** |

A score that never looks at the realised path at all reproduces 84% of `total`'s reliability, and
the bias-corrected systematic term — the only learnable half by the module's own docstring —
reaches 0.269. That is one `--key spread_pop` run away and it says more than the whole partialling
apparatus.

**Cheapest check.** `python3 -c "... F.dispersion_partial_floor(...)"` with a non-zero `real`, and
`F.split_half(..., key='spread_pop')`. Both under a minute.

---

## F7 — MAJOR. "Only R matters" is a property of the null, presented as a design licence

**The claim.** `plan_drafts/02_statistics.md` §3.3:

> The variance of a paired difference on shared contexts is dominated by the within-context term,
> because the between-context term cancels. So at fixed rollout budget the split between contexts
> and rollouts is irrelevant. … **This resolves the tension in §3.2 cleanly.** `k >= 20` is a
> *selection* requirement on a subsample; the *evaluation* budget can be split however is
> convenient, because only `R` enters.

**Why it fails.** The between-context term cancels **only when the two sides differ by nothing but
the rollout draw**, which is exactly the do-nothing pair used to measure it. Under a real treatment
the per-context effect `Δ_c` varies, and

```
Var(paired gap) = 2 sigma_within^2 / (N k)   +   sigma_Delta^2 / N
                  ^ depends on R only            ^ depends on N only
```

The second term is identically zero in the null and is the entire reason the experiment is being
run.

**Measured.** Same archive, horizon 50, 60 draws, 8 tickers, 5%-trimmed-mean relative gap, with a
per-context multiplicative effect of mean 1 and sd `het` applied to one side:

| design | R | het=0.00 | het=0.25 | het=0.50 | het=1.00 |
|---|---:|---:|---:|---:|---:|
| N=500 k=1 | 500 | 10.60% | 10.79% | 11.31% | **12.89%** |
| N=250 k=2 | 500 | 10.08% | 10.29% | 11.24% | 13.24% |
| N=100 k=5 | 500 | 10.76% | 11.11% | 12.78% | **16.33%** |

At het = 0 the three designs are equal, exactly as §3.3 reports. At het = 1.0 the `N=100 k=5`
design costs **27% more standard deviation at the same R**; at het = 0.5, 13% more.

**The scenario.** §3.5's budget tables offer `R = 2,000` split as `N=400, k=5` or `N=2000, k=1`.
Taking the licence and choosing the small-`N` split silently loses power in proportion to the
heterogeneity of the effect — and heterogeneity is guaranteed here, because the pool is *defined*
as the contexts where the model is worst.

**Cheapest check.** The table above; three minutes.

**Fix.** State the licence conditionally: "only `R` matters under a homogeneous effect; prefer the
largest `N` the budget allows, because the between-context term is reduced only by `N`."

---

## F8 — MAJOR. The k ≤ 5 extrapolation is used as a budget, in the file that says it is not

**The claim.** `results/RESULTS_20260905.md` addendum 2 §D, line 396:

> Published (mixed): 17–41, **median 21**. Uniform largest-k: 17–41, **median 22**. … **Neither
> figure is used as a budget.**

`PLAN.md` §0.10 repeats it: "the `k ≤ 5` extrapolation to `k ≈ 21` is **not** used as a budget
anywhere in this revision."

**Why it fails.** It is used as a budget in at least five places, one of them 327 lines further
down the same file:

| where | text |
|---|---|
| `results/RESULTS_20260905.md:723` | "the budget returns to the `k` implied by the largest-k estimator (**17–41, median 22**), which does not fit the inode budget and forces the pool to shrink" |
| `PLAN.md:51` (§0.3, the budget table) | "\| Ranked-list subsample at `k = 20` \| as needed \|" |
| `PLAN.md:131` (§0.9) | "If it fails, §0.3 reverts to `k ≈ 21` and the cycle-1 pool stops fitting in the inode budget" |
| `plan_drafts/05_execution.md:305-306` | "**k=20 clears the 0.80 line for five of eight tickers** and misses for MSFT, GOOG and JPM. **k=24 clears all eight.** The choice between k=20 (80 new members, 5.5 GPU-hours…)" |
| `plan_drafts/03_infrastructure.md:167,169` | inode tables computed at `k=20` (160,000 rollouts, 320 members) |

`05`'s use is the worst: it reads a **per-ticker pass/fail at k=20 versus k=24** off a curve that
`02` §3.1 and `rollouts_needed`'s own `rejected_reason` say the data rejects, and the estimator is
explicitly a **lower bound** ("still rising with k"). Using a lower bound to conclude "k=24 clears
all eight" is wrong in the direction that costs GPU hours for nothing.

I reproduce the estimator on today's data (60 draws, stratified, horizon 50): `k_for_rho_0.80`
largest-k-only = 16–40, median 23 (published 17–41, median 22), and `k_for_rho_0.90` = 37–91,
median 51.

**The scenario.** `k = 3` is defended by the dilution assumption, which is untested; the stated
fallback is `k ≈ 21`. If dilution fails, the plan reverts to a budget derived from an extrapolation
it has declared unsupported, sized 4× beyond the largest measured `k`, and known to be biased low.
The inode budget then fails against a number nobody believes.

**Cheapest check.** `grep -n "k = 20\|k≈21\|median 22" PLAN.md plan_drafts/*.md results/*.md`.

**Fix.** Either say "the `k` requirement is unmeasured above 5; the fallback is to measure `ρ` at
k=10 and k=20 (`02` open question 2) before any budget quotes it", or delete the fallback
sentence. As written, the standing instruction is contradicted by the document that states it.

---

## F9 — MAJOR. The design-effect table is computed on the largest ticker only

**The claim.** `plan_drafts/02_statistics.md` §1.1:

> With `dataset_length = 226,002` that is 452 blocks per ticker … So context-level i.i.d.
> resampling is defensible at today's `N = 500` and indefensible above `N ~= 2,000`.

**Why it fails.** `dataset_length` is per-ticker and spans a factor of 9:

| ticker | AMD | AMZN | GOOG | INTC | JPM | META | MSFT | NFLX |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `dataset_length` | 126,853 | 168,564 | **226,002** | 118,650 | **25,018** | 35,304 | 100,872 | 81,163 |
| contexts per 500-index block at N=500 | 1.97 | 1.48 | 1.11 | 2.11 | **9.99** | 7.08 | 2.48 | 3.08 |
| DEFF at ICC = 0.10 | 1.10 | 1.05 | **1.01** | 1.11 | **1.90** | 1.61 | 1.15 | 1.21 |

The published `1.01` is GOOG's. JPM's is 1.90 and META's 1.61 — so the standard error of any
context-resampled per-ticker statistic is already understated by 1.38× on JPM and 1.27× on META
at today's `N = 500`, not 1.01×. Read from `sample_indices_rank0.json`, one field.

Two smaller points in the same table: the text states the measured ICC as `0.089` but the DEFF
column is only reproducible with `0.10` (at N=10,000: `1 + 21.12×0.089 = 2.88`, published 3.11
= `1 + 21.12×0.10`); and `load_arm` sorts context ids as **strings** (`ids.dtype = <U6`), so row
position in the loaded arrays is lexicographic and carries no dataset-index ordering — any
index-based statistic must cast to `int` first.

**The scenario.** The conclusion "scale tickers, not contexts" is *strengthened* by the correction,
so the strategic advice survives. What does not survive is "i.i.d. context resampling is defensible
at N = 500": on two of eight tickers it already is not, and any per-ticker interval computed that
way is too narrow.

**Cheapest check.** The loop above over the eight `sample_indices_rank0.json` files; ten seconds.

---

## F10 — MAJOR. §2.3's floor table does not reproduce, and the headline moves most

**The claim.** `plan_drafts/02_statistics.md` §2.3:

> The arm-level do-nothing gap over the 8 tickers, pool-restricted, mean-of-`M`:
> `+19.91, -1.25, -7.91, +6.22, -7.64, +11.43, +27.82, +15.31` percent, i.e. **mean +7.99%,
> sd 13.02%, se 4.60%, t = 1.74, 5 of 8 positive.** Doing nothing produced an apparent 8%
> improvement at t = 1.74.

**Measured on today's archive**, same recipe (pool selected by the stratified score from `v5me3`
seeds ≥ 97706, graded on the `repro`/`repB` pair, horizon 50):

```
per ticker:  +7.96  +2.97  -17.55  +14.24  -16.86  -2.81  +28.35  +4.31
mean +2.58%   sd 15.33%   t = 0.48   5 of 8 positive
```

| statistic | published (all / pool) | measured (all / pool) |
|---|---|---|
| mean of `M` | 7.42% / 14.56% | 5.69% / **14.57%** |
| median | 2.93% / 11.96% | 4.46% / 13.93% |
| 5%-trimmed mean | 1.99% / 7.96% | 1.67% / 7.75% |
| geometric mean | **36.46%** / — | **4.35%** / 8.20% |

The RMS-level figures reproduce closely; the individual per-ticker gaps and the headline `t` do
not. This is consistent with `RESULTS` addendum 2 §B / addendum 3 §D ("reproduction fails; the
inputs or an unrecorded dependency changed") and is a further instance of it — but the number that
moves is the one being quoted, from `t = 1.74` to `t = 0.48`.

**The geometric-mean row is a consequence of an unstated rule.** `M` has **1–84 exact zeros per
500 contexts** (`[25, 27, 21, 84, 1, 31, 13, 72]` across the eight), so `log M` is `−inf` unless
zeros are excluded,
and AMZN's smallest positive `M` is `1.9e-19` — 26 nats of log range. Whether the geometric mean
reads 36% or 4% is decided entirely by how zeros and near-zeros are handled, which §2.3 does not
say. "The geometric mean is disqualified" is therefore a property of the handling rule, not of the
estimator.

**The scenario.** `PLAN.md` §0.1 F7 carries "doing nothing moves an arm-level endpoint by up to
28%" into the merged plan as a measured fact. The maximum (28.35%) survives; the mean and its `t`
do not. Anyone quoting `+8.0% at t = 1.74` as the do-nothing band is quoting a number that no
longer exists.

**Cheapest check.** Re-run §2.3's own snippet; three minutes. And print `(M == 0).sum()` beside the
geometric-mean row.

---

## F11 — MINOR. `stratify_v2` cannot be reached from any production path

`PLAN.md` §0.1, D1 §2 row: "`stratify` is kept unchanged so the published numbers stay
reproducible, and **`stratify_v2` is what any new pool should use**."

`grep -n "stratify" /lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/code/failure_pool_reliability.py`
shows `stratify_v2` defined at line 181 and called from **nowhere** (re-verified against the
2026-09-05 10:17 version of the module, 755 lines): `pairing_nulls` (line 274), `split_half`
(373–374), `regeneration_null` (416–417), `dispersion_partialled_reliability` (566) and
`pool_overlap` (652) all call `stratify`, and `main()`'s eleven `add_argument` calls include no
`--stratifier`. The only callers are two tests. This is a mild instance of the project's own documented shape — a fix that
is written, recorded and referenced but that no execution path can reach. Add a
`--stratifier {v1,v2}` argument that threads through, and record the choice in the output JSON.

---

## F12 — MINOR. T8's assertion is two-sided; E1b is one-sided

`plan_drafts/02_statistics.md` §1.1:

```
  E1b  pairing      rho_cross(t,h,k)  - rho_indep(t,h,k)   <=  2 * se
```

§6 trap T8: `assert abs(nulls["cross"] - nulls["independent"]) < 2*se`.

Measured, k=5, 60 draws, horizon 50: the one-sided form passes 8/8. The two-sided form fails on
GOOG (`cross − indep = −0.0219`, 2se = 0.0162) and META (`−0.0180`, 2se = 0.0144) — i.e. on
`cross` sitting **below** `independent`, which is the direction that indicates the pairing matters
more, not less. A script written from §6 would block a reliability claim that §1.1 accepts. Make
T8 one-sided.

---

## F13 — MINOR. The published `±` is not the uncertainty of the published claim

`RESULTS` §2 gives `true 0.364±0.030 … shared 0.368±0.039` per ticker and concludes
"`shared > true` in **8/8** tickers at k=5 (mean gap +0.032)". Those `±` are the **draw-to-draw**
sd, conditional on these 500 contexts and these 10 seeds; a reader comparing a gap of +0.032
against `±0.03–0.05` will read "not resolved".

The unit for a claim about eight tickers is the ticker. Measured:

| k | per-ticker gaps | mean | between-ticker sd | se | t (7 df) | leave-one-out t |
|---|---|---:|---:|---:|---:|---|
| 3 | +0.018 +0.007 +0.079 +0.030 +0.025 +0.009 +0.030 +0.007 | +0.0256 | 0.0237 | 0.0084 | **+3.06** | +2.58 … +4.48 |
| 5 | +0.011 +0.011 +0.101 +0.019 +0.015 +0.024 +0.043 +0.011 | +0.0294 | 0.0309 | 0.0109 | **+2.69** | +2.21 … +4.41 |

**The claim survives the correct unit, and the correct presentation is stronger than the published
one.** Report the ticker-level `t` and its leave-one-out range.

The **count** should not be reported at all. `RESULTS` says 7/8 at k=3; my draw gives 8/8 at k=3;
dropping the 28 wrapped contexts (F14) gives 7/8 at k=5. The 7-versus-8 distinction turns on
per-ticker gaps of +0.004 to +0.018, well inside their own draw sd, and is not a resolvable
quantity.

---

## F14 — MINOR. The 28 wrapped contexts are a contiguous early block, not a random 5.6%

`RESULTS` addendum 5 §A: "**28 of 500 contexts (5.6%) carry rollouts from a different position in
the RNG stream** … Any exchangeability argument over contexts within a member has to either
exclude those 28 or show the difference does not matter."

`rank_indices` is sorted ascending, so `rank_indices[0..27]` are the **28 lowest dataset indices**
in each ticker, spanning 4.6% (INTC) to 7.9% (GOOG) of that ticker's index range — a contiguous
early block, which under `02` §1.1's own block model (`block_id = dataset_index // 500`) is a
correlated cluster, not a random 5.6%.

**Quantified: dropping them changes nothing material.** k=5 for the nulls, k=3 for the partial,
60 draws, horizon 50, mean over 8 tickers:

| quantity | 500 contexts | 472 contexts | Δ |
|---|---:|---:|---:|
| `true` | 0.4635 | 0.4640 | +0.0005 |
| `independent` | 0.0839 | 0.0873 | +0.0034 |
| `shared − true` | +0.0294 | +0.0297 | +0.0003 |
| `fraction_kept` (dispersion partial) | 0.8745 | 0.8680 | −0.0065 |
| tickers with `shared > true` | 8/8 | **7/8** | AMZN flips to −0.0135 |

So the exchangeability worry is answerable and the answer is "it does not matter", except that the
8/8 count is not robust to it (see F13). Record the answer so the caveat can be retired.

**Constructive corollary the plan should use.** Because the pad wraps, **every one of the 80
members already contains 28 contexts generated twice by the same model in the same run**. That is
a free within-member regeneration replicate, present in every member, and it is *crossed* rather
than seed-matched, so it is exactly the structure F3 says the floor is missing. `02` §2.4 states
"a generation replicate costs a whole inference pass, so `G = 2` is the realistic ceiling"; 28
contexts × 80 members of that replicate are already on disk.

---

## What I checked and found clean

**The `num_errors` join, extended from 3 tickers to 8, from the logs alone.** Under the claimed
mapping, slot `j` (`j < 28`) and slot `500+j` are two generations of the same context. Correlating
the seed-mean of slots 0..27 against the seed-mean of slots 500..527 over the 28 contexts:

```
AMD +0.677  AMZN +0.827  GOOG +0.736  INTC +0.907  JPM +0.542  META +0.775  MSFT +0.660  NFLX +0.767
same-slot split-half ceiling (5 seeds vs 5): +0.300 to +0.853
28 random slots from 28..499 as a reference:  -0.275 to +0.270
```

Above the same-slot ceiling in 8/8 (it averages 10 seeds per side rather than 5) and far above the
random reference in 8/8. **No counterexample exists in the other five tickers.** The geometry is
also uniform: `Padded 28 indices to fill last batch (528 total)`, 11 batches of 48, in all 8
tickers × 3 seeds checked — and that line is printed by the log itself, so the *count* never
needed a recomputation; only the pad's *position* did.

One check addendum 5 did **not** make, which I did: the `.npz` (which every score is computed
from) carries the **second** generation for those 28, matching the `data_gen` CSVs. Measured by
comparing `|generated return|` against `num_errors` at slots 500..527 versus slots 0..27, using
the 472 unambiguous slots as the within-generation reference: the surplus slots are the better
match in 7/8 tickers. So the join and the scored arrays are on the same generation.

**The code that produced the archive is not in any tree I could reach.** `to fill last batch` does
not appear under `/lus/lfs1aip2/projects/public/u6gb/sigma-0/src`, nor under
`/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/crps-return-alignment-20260808/`, nor
`/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/continual-learning-20260827/`.
`gen_driver.py:1688/1698` chunk a supplied list and assert divisibility; they do not pad. So the
padding is done by a caller that is not recoverable — which is a further, concrete instance of the
missing `code_commit` that `RESULTS` addendum 3 §D already identifies as the gap P1 exists to
close. It does not change the verdict, because the empirical evidence above is 8/8.

**`shared > true` survives the correct resampling unit** (F13): ticker-level `t = +3.06` at k=3
and `+2.69` at k=5, all eight gaps positive, leave-one-ticker-out `t` never below 2.2. GOOG carries
the largest gap but removing it leaves `t = +4.5`.

**φ reproduces to ±0.005** at every horizon (`0.971 0.865 0.650 0.389 0.279 0.227 0.190` against
the published `0.976 0.873 0.661 0.393 0.281 0.225 0.191`), and **§2.2's one-parameter model
reproduces**: predicted-minus-measured over 8 tickers × 7 horizons has sd 0.040 and max |error|
0.100 (published 0.041 / 0.110). The mean error's **sign** flips (−0.024 here, +0.027 published),
which is the archive-change effect again; the substantive conclusion — generation nondeterminism
behaves like a partial redraw, not an irreducible ceiling — is well supported.

**`02` §3.1's held-out extrapolation test and its asymptote conclusion reproduce.** The
two-parameter `rho_inf` is below 0.80 in 7 of 8 tickers, exactly as published.

**§3.3's `1/sqrt(R)` law holds over a 5× range on the null**, not just the 2× range quoted: at
R=500 the trimmed-mean gap sd is 10.27%, at R=2,500 it is 4.48% against a predicted 4.59% (2%
error). The law is sound; the design licence drawn from it is not (F7).

**`rollouts_needed`'s largest-k estimator reproduces** (16–40, median 23 against the published
17–41, median 22) and the emit-for-every-ticker fix from R1-F6/F9 is correctly in place and pinned
by `test_the_largest_k_estimator_is_always_available`.

**The `num_errors` sign flip across tickers is resolvable.** With 500 contexts the Spearman
standard error is about 0.045 for a single seed and smaller after averaging ten, so −0.136 (AMD)
and +0.195 (INTC) are 3.0 and 4.4 standard errors from zero in opposite directions. Treating
`num_errors` as a pre-registered covariate is justified on that ground alone.

**`load_arm` joins by id, never by row order**, and `regeneration_null` refuses when the realised
arms differ — both correct, and both checked by reading the code path rather than trusting the
docstring.

**The exact decomposition and the bias correction are right.** `total = bias2_raw + spread_pop`
holds exactly; the `ddof=0` / `ddof=1` split between the share denominator and the correction is
correctly motivated and correctly implemented; `bias2 = bias2_raw − spread/k` is the right
unbiased form and is undefined at k=1, as the docstring says.

**The test suite runs green** (25 passed, 0.31 s) and the four new stratification tests pin the
zero-move collapse, the v2 fix, and the direction of the leak. The one test whose docstring
overstates its assertion is `test_repeated_nulls_carry_the_leak_floor_when_stratified` (F4).

**Not checked, and why.** `04_training_design.md` and `05_execution.md` beyond the `k = 20`
references (outside this lens; another reviewer's). The wandb config for `j5705912`. The X4
checkpoint comparison in `RESULTS` addendum 5 §E (weights, not statistics). `write_run_manifest.py`
and `generation_gate.py` (R1's and the infrastructure lens's ground). The dilution experiment
specified in addendum 5 §D was not run — but note that its stated fallback, `k ≈ 21`, is the
figure F8 shows the plan has been instructed not to use.
