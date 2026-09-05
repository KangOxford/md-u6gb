# Facet 02 — Estimands, noise floor, power, multiplicity, stopping rules

> Scope: the statistical contract for the merged Thread A (plasticity) / Thread B (failure-driven
> continual learning) plan. Everything below is either measured on
> `/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/results/failure_pool_reliability.json`,
> measured in this draft by re-running
> `/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/code/failure_pool_reliability.py`
> as a library on the arrays under
> `/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z/data/`,
> or flagged under `## Open questions`. No GPU was used. Every number labelled **[new]** was
> computed while writing this file and is reproducible with the snippet printed beside it.

## 中文速览

- **四条主张各写成公式**（§1）：池子是真的 / 继续训练有改进 / 旧分布没被破坏 / 可塑性下降。每条标出重抽单位与理由，并说明用错单位会怎样。
- **[新测] 0.81–0.87 那个「重生成一致度」不是天花板，方向恰好反了。** 同一个 k 下换种子的一致度只有 0.60–0.63，两次完整重生成反而更高（0.75–0.98），因为两次生成有 **19%（H=250）到 98%（H=10）的输出逐位相同**。用「共享抽样比例」这一个参数就能把 7 个视界的重生成一致度预测到 **±0.04**（8 票 × 7 视界，均值误差 +0.027）。所以生成的不确定性表现得像一次「部分重抽」，同样被 1/k 平均掉，不是不可消除的上限。
- **[新测] 什么都不做的重现带才是真正的底噪。** 同一模型、同样种子、两次完整重生成，在「池内均方误差」上给出 **+8.0% ± 4.6%（t=1.74）**，单票最大 **+27.8%**；全部 56 个「票 × 视界」格子里 4 个超过 10%，最大 **+33.2%**。任何小于这个带宽的「改进」都不构成证据。
- **[新测] 换一个统计量比加样本便宜得多。** 同一批数据上，5% 截尾均值的重现带 **2.0–3.4%**，未截尾均值 **6.0–7.4%**，几何均值 **17–36%**。
- **[新测] 配对差的精度只由 R = 上下文数 × 每上下文 rollout 数 决定，与两者怎么分配无关**（R=500 时三种分法给 18.8 / 17.1 / 17.7）。所以 k≈20 只是**选池**的要求，**评测**可以随便分。
- **[新测] k≈20 是下界不是估计。** 一参数外推在留出的 k=5 上系统性高估 ρ，16 例里 15 例偏高、平均 +0.08（符号检验 p=0.0005）；两参数拟合给出的渐近上限 8 票里 7 票低于 0.80。
- **[新测] 上下文不是独立单位。** 窗口下标相距 <500 的两个上下文相关 0.089（0.032–0.138）。N 超过约 2,000 就要按下标分块重抽；N=10,000 时有效样本只有 3,213。
- **[新测] 校正后的分数不能当作改进指标**：它是分箱内的名次，任意模型下均值恒等于 0.500000。**选池的尺子和验收的尺子必须是两把。**
- **多重比较不是「我们会小心」**：全网格是 **2,576 个格子级比较**。给一个预先登记的主终点 + BH q=0.10 的次级族 + 用实测零效应分布（p95 = 12.2%）当格子级门槛。
- **停手规则**：n 停止增长前任何数字不进标题、不加粗、不称「最」；任何分位统计量旁边必须印事件数；效应随 n 变化必须给整条轨迹。

---

## 0. What is measured, and what is carried in

| Source | Status |
|---|---|
| `results/failure_pool_reliability.json` (8 tickers × 500 contexts × 10 seeds × 7 horizons) | measured 2026-09-04, on disk |
| Everything marked **[new]** below | measured while writing this file, same data, CPU only, snippets inline |
| Training-seed variance of any endpoint | **never measured on this project** — see `## Open questions` |
| Effect size of continual training on the pool | **unknown** — no pilot exists, so §3 gives detectable effects, not required n |

Notation used throughout.

```
t   ticker,            t in T,           |T| = 8 today (AMD AMZN GOOG INTC JPM META MSFT NFLX)
c   frozen context,    c in C_t,         |C_t| = N = 500 today, drawn from 226,002 windows (GOOG)
h   forward horizon,   h in H = {10, 25, 50, 100, 150, 200, 250} messages
i   rollout index,     i = 1..k          k = 10 available today, k = 2..4 for the regeneration pair
a   arm (mix ratio x LR x replay), a = 0 is the control arm
s   training seed
g   generation replicate (one whole re-run of inference for a fixed arm and seed)
x[t,c,i,h]  generated forward return          y[t,c,h]  realised forward return
M[t,c,h](k) = (1/k) sum_i (x[t,c,i,h] - y[t,c,h])^2          "total", the selected-on quantity
              = (xbar - y)^2 + Var_i(x)                      exact, tested
S[t,c,h](k) = rank of M within its |y| decile bin, on [0,1]  "corrected", the ranking device
```

---

## 1. The estimand for every claim

Each claim is written as a functional of identified units before any estimator is chosen. The
**unit of resampling** is stated with the reason, because on this project the unit has been wrong
twice: a bootstrap over trading days when days carried 1–58 contexts each (so the mean of day means
is not the mean over contexts), and a bootstrap taken inside a single generation implementation, so
that the variance of "regenerate everything" was structurally absent from the interval.

### 1.1 "The failure pool is real"

The claim is that there exists a context-level latent quantity that the score estimates, and that
the ranking depends on the rollout-to-context pairing rather than on a shared artefact. It is a
conjunction of three parts, all at the deployed `k`, on the corrected score `S`:

```
                    rho_true(t,h,k)   = Corr_rank_{c in C_t} ( S_A[t,c,h](k) , S_B[t,c,h](k) )
                                        A, B disjoint sets of k rollouts

  E1a  excess       Drho(t,h,k)       = rho_true(t,h,k) - rho_indep(t,h,k)      >=  0.70
  E1b  pairing      rho_cross(t,h,k)  - rho_indep(t,h,k)                        <=  2 * se
  E1c  not shared   rho_true(t,h,k)   - rho_shared(t,h,k)                       is REPORTED, not
                                                                                   used as evidence
```

`rho_indep`, `rho_shared`, `rho_cross` are the four pairing nulls already implemented in
`failure_pool_reliability.pairing_nulls`. The target is written on the **excess** `Drho`, not on
`rho_true`, because the zero line is not zero: at `k=5, h=50` on the corrected score the measured
values are `true 0.464`, `shared 0.499`, `independent 0.086`, `cross 0.066` (means over 8 tickers).
`independent = 0.086` is residual `|y|` leakage inside the 10 coarse bins, so a raw
`rho_true = 0.80` target silently includes about 0.09 of nothing. E1c is reported and never used
as evidence because `shared (0.499) > true (0.464)`: a consistently mis-paired score is *more*
reliable than the correct one, so split-half reliability on its own certifies nothing.

**Unit of resampling: blocks of 500 consecutive dataset window indices, within ticker.**
Not the context. **[new]** Two contexts whose window indices differ by less than 500 have a score
correlation of `0.089` (per-ticker range `0.032` to `0.138`); beyond 2,000 indices the correlation
is `<= 0.05`, beyond 10,000 it is `~0`.

```python
d = np.abs(idx[:,None]-idx[None,:]); iu = np.triu_indices(len(idx),1)
np.corrcoef(v_i[iu][d[iu]<500], v_j[iu][d[iu]<500])[0,1]     # -> 0.032 .. 0.138 over 8 tickers
```

With `dataset_length = 226,002` that is 452 blocks per ticker, and the design effect
`DEFF = 1 + (m-1)*ICC` with `m = N*500/226002`:

| N per ticker | contexts per block | DEFF | effective N | SE understated by |
|---:|---:|---:|---:|---:|
| 500 | 1.11 | 1.01 | 495 | 1.01x |
| 2,000 | 4.42 | 1.34 | 1,490 | 1.16x |
| 10,000 | 22.1 | 3.11 | 3,213 | 1.76x |
| 50,000 | 110.6 | 11.96 | 4,180 | 3.46x |

So context-level i.i.d. resampling is defensible at today's `N = 500` and indefensible above
`N ~= 2,000`. It also caps what more contexts can buy: **going from `N = 10,000` to `N = 50,000`
is 5x the rollouts for 1.30x the effective sample.** That argues for more tickers, not more
contexts per ticker (§3.3).

**Why not trading days.** The day of a context is *not recorded anywhere in the artefact*. The
manifest at `.../hp_v5me3_GOOG_s97701/member_0/sample_indices_rank0.json` carries
`all_indices`, `all_indices_sha256 = 29a1e20f...`, `dataset_length = 226002`,
`requested_sequences = 500`, `selection_seed = null`, and the `.npz` carries only string context
ids and values. There is no `(ticker, date, message_offset)` column. A day bootstrap is therefore
not merely a bad choice here, it is **not computable from the artefact**. See §6 trap T11 for the
manifest field the generator must write before the first rollout of the next campaign.

### 1.2 "Continual training on the pool improved the model"

Two endpoints, because the corrected score cannot carry this claim at all. **[new]** the corrected
score is a within-bin rank, so its mean over contexts is exactly `0.500000` for every arm:

```
AMD   mean(regen A) = 0.500000   mean(regen B) = 0.500000   diff = +0.00e+00
GOOG  0.500000 / 0.500000 / +0.00e+00      JPM, NFLX identical
```

**The selection ruler and the acceptance ruler must be two different objects.** `S` selects; `M`
grades.

```
  E2a  broad        theta_imp(a,h) = Trim_{0.05}[ M^a[t,c,h] ]  -  Trim_{0.05}[ M^0[t,c,h] ]
                                     c ranging over a held-out evaluation set C_eval
  E2b  pool-like    theta_pool(a,h)= Trim_{0.05}[ M^a[t,c,h] : c in P ] - Trim_{0.05}[ M^0[t,c,h] : c in P ]
                                     P selected by S from a rollout set DISJOINT from the one graded
```

`Trim_{0.05}` is the mean after removing the lowest and highest 5% of contexts. It is part of the
estimand, declared in advance, not a robustness afterthought — §2.3 shows it is worth a factor of
3 on the noise floor. `P` must be selected from rollouts that are not the rollouts being graded,
otherwise the endpoint reads back the selection noise; this is why §2 measures the pool endpoint
with `P` chosen from `v5me3` seeds `97706..97710` and graded on the `repro/repB` pair.

**Unit of resampling: three nested levels, resampled outermost-first.**

| level | why it is the unit | replicates needed |
|---|---|---|
| training seed `s` | the arm is a *training procedure*, not a weight vector; the claim is about the procedure | >= 5 (§5.1) |
| generation replicate `g` | XLA autotuning makes generation non-reproducible; a bootstrap inside one `g` cannot see it | >= 2, and see §2.4 for a cheaper 7-df alternative |
| context block | §1.1 | all blocks |

Bootstrapping only the innermost level is exactly the error the brief names. Mechanically:
the bootstrap index tuple must be `(s*, g*, block*)` and a script must assert
`len(set(s_index)) > 1 and len(set(g_index)) > 1` before it will emit an interval (§6 trap T12).

### 1.3 "The old distribution was not damaged"

This is a **non-inferiority** claim and must be written as one. An equality test that fails to
reject is not evidence of no damage; it is evidence of no power.

```
  E3a  language-model  theta_stab = NLL_old(a) - NLL_old(0)   <=  delta_0     one-sided, delta_0 pre-declared
       model NLL       NLL_old(a) = -(1/|D_old|) sum_{token in D_old} log p_a(token | prefix)
                       D_old = held-out tokens from the pre-CPT window, 2022-01..2024-07

  E3b  generative      theta_div  = R_div(a) - R_div(0)       <=  delta_1     one-sided
                       R_div = LOBbench divergence ratio, last window
```

E3a involves **no generation at all**: it is a forward pass over fixed tokens, so its only noise
source is the training seed, and its floor is a pure seed floor. E3b does involve generation and
carries the full three-level structure of §1.2. Reporting one without the other repeats the
"single coordinate lies" lesson already written into `PLAN.md` §2.4.

**Unit of resampling for E3a: ticker-day blocks of held-out tokens.** Here the day *is* available,
because `D_old` is read from the SquashFS monthly shards where the file path carries the date. The
estimator must be a **ratio of sums** (total NLL over total tokens), not a mean of per-day means,
because days carry unequal token counts.

**`delta_0` and `delta_1` must be numbers in a file before the first arm runs.** A margin chosen
after seeing the arms is not a margin.

### 1.4 "Plasticity has declined"

Follows `PLAN.md` §2.3: early vs late checkpoint of the *same* run, identical fixed-budget
adaptation, never fresh-vs-continued.

```
  E4   theta_pl = AUC(theta_late) - AUC(theta_early)
       AUC(theta) = sum_{j=1..J} v_j(theta) * Delta      v_j = validation NLL at probe step j*Delta
       both checkpoints probed with the SAME set of adaptation seeds  ->  paired
       PRESENT iff  theta_pl > 0 with CI excluding 0  AND at least one co-moving diagnostic
                    (dormant fraction up, effective rank down, optimization readiness down)
```

**Unit of resampling: the adaptation seed, paired across the two checkpoints.** Not the probe
step: the `v_j` inside one run are a single trajectory, heavily autocorrelated, and treating them
as J independent draws inflates the sample by roughly J. The AUC is one number per (checkpoint,
seed); the sample size is the seed count, and nothing else.

The co-moving diagnostic is part of the estimand (a conjunction), not a separate claim to be
tested and corrected for. Written as a conjunction it costs no multiplicity; written as four
separate claims it costs four.

---

## 2. The noise floor, measured on the same structure as the effect

The rule: **the floor estimator and the effect estimator must be the same function, called with
the same pairing argument, on the same units.** Concretely `floor = f(arm0_rep1, arm0_rep2)` and
`effect = f(arm0, arm1)` with the same `f`. Anything else charges structure to noise.

### 2.1 Which repeat establishes which floor

| Estimand | The repeat that is its floor | Repeats needed | Covers generation nondeterminism? |
|---|---|---|---|
| E1 pool is real | `pairing_nulls(..., independent)` at the deployed k, plus the same nulls computed **across two generation replicates** | 20 permutation draws (done) + 2 generations (not done) | **No** as computed today: the nulls redraw rollouts within one generation |
| E2a broad improvement | two whole regenerations of the control arm, same seeds, same contexts, differenced with `Trim_{0.05}` | >= 2 generations; use the 8-ticker spread for df (§2.4) | **Yes** |
| E2b pool improvement | same, restricted to `P` chosen from a disjoint rollout set | same | **Yes** |
| E3a old-window NLL | the control arm retrained under >= 5 training seeds, differenced pairwise | >= 5 seeds | **N/A** (no generation) |
| E3b divergence ratio | two regenerations of the control arm scored through LOBbench | >= 2 generations x >= 5 seeds | **Yes** |
| E4 plasticity | the *same* checkpoint probed twice with different adaptation seeds | >= 5 seeds | **N/A** |

The distinction the brief asks for, stated plainly: **a floor built by resampling contexts inside
one generation covers redrawn rollouts only. A floor built from two whole regenerations covers
redrawn rollouts *and* the XLA autotuning fork.** The two answers differ and the gap is
measurable — §2.2.

### 2.2 [new] The "0.81–0.87 ceiling" is not a ceiling, and the sign is backwards

The brief records that two complete regenerations with identical seeds agree at rank correlation
0.81–0.87, "so ~15% of rank agreement is lost to nondeterminism alone". That reading does not
survive a matched-k comparison.

**[new]** Split-half agreement within one generation, at the same k as the regeneration pair
(k = 2 for AMD, 3 for AMZN, 4 for the rest), 60 draws, raw score:

```
horizon          H10    H25    H50   H100   H150   H200   H250
split-half     0.604  0.623  0.630  0.630  0.619  0.616  0.619    two disjoint seed sets
regeneration   0.982  0.942  0.889  0.820  0.781  0.760  0.751    two whole re-runs, same seeds
```

Two whole regenerations agree **more** than two disjoint seed sets, at every horizon, by +0.11 to
+0.37. So the regeneration figure is not an upper bound on ranking quality; it sits above the
sampling line. The reason is visible directly in the arrays. **[new]** fraction of
(context, member, horizon) forward returns that are **bitwise identical** between the two
regenerations:

```
horizon      H10    H25    H50   H100   H150   H200   H250
phi        0.976  0.873  0.661  0.393  0.281  0.225  0.191     mean over 8 tickers
```

A one-parameter model follows immediately. If a fraction `phi` of members did not fork, the two
sides share that part of the noise:

```
rho_regen(k) = ( s^2 + phi * n^2/k ) / ( s^2 + n^2/k ),      n^2/(k s^2) = 1/rho_split(k) - 1
```

Fitted with nothing but the measured `phi` and the matched-k split-half, over all 8 tickers x 7
horizons: **mean error +0.027, sd 0.041, max |error| 0.110.** Predicted vs measured means:

```
horizon      H10    H25    H50   H100   H150   H200   H250
predicted  0.990  0.953  0.878  0.780  0.730  0.707  0.696
measured   0.982  0.942  0.889  0.820  0.781  0.760  0.751
```

**Consequence for the plan.** Generation nondeterminism behaves like a *partial redraw of the
rollouts*, which the same `1/k` averaging removes. It is not an irreducible ceiling and it must
not be quoted as one. What it *does* do is make the number of independent draws smaller than `k`
suggests when the two things you are comparing were generated in separate runs, and it makes the
fork rate a function of rollout length: at `H = 250` only 19% of members survive intact, so any
long-horizon claim inherits four fifths of a fresh redraw.

Also: the regeneration null in `results/failure_pool_reliability.json` is computed on the **raw**
score only. **[new]** on the corrected score, which is the score the pool would actually use, the
same regeneration pair gives

```
horizon             H10    H25    H50   H100   H150   H200   H250
raw   (in file)   0.982  0.942  0.889  0.820  0.781  0.760  0.751   grand mean 0.846
corrected [new]   0.972  0.906  0.814  0.690  0.629  0.599  0.583   grand mean 0.742
```

so the headline "0.81–0.87" becomes **0.67–0.80** per ticker on the score that matters, and
**0.583** at the longest horizon.

### 2.3 [new] The floor for the arm-level endpoints: doing nothing moves the number by up to 28%

Same model, same seed labels, two whole regenerations (`hp_v5me3repro_*` vs `hp_v5me3repB_*`),
`h = 50`, relative gap in the arm-level statistic:

| estimand of the arm statistic | RMS over 8 tickers, all contexts | max | RMS, pool-restricted (n=50/ticker) | max |
|---|---:|---:|---:|---:|
| mean of `M` | **7.42%** | 15.11% | **14.56%** | 27.82% |
| geometric mean of `M` | 36.46% | 72.93% | — | — |
| median of `M` | 2.93% | 5.61% | 11.96% | 26.21% |
| **5%-trimmed mean of `M`** | **1.99%** | 3.35% | **7.96%** | 17.67% |

At `h = 250` the same table reads mean 6.01% / gmean 17.04% / median 6.69% / trimmed 3.40%.

Two things follow, and both are decisions, not observations.

1. **Pre-register the 5%-trimmed mean.** It is 3.7x tighter than the untrimmed mean at `h = 50`
   and 1.8x tighter at `h = 250`, for free. The geometric mean is disqualified: 36% RMS, because
   `M` reaches near-zero on easy contexts and log space is then dominated by them.
2. **The pool-restricted endpoint has roughly twice the floor of the broad endpoint**, because it
   is computed on 50 contexts instead of 500. That is the endpoint the whole of Thread B is aimed
   at, and its floor is 8–15%.

The arm-level do-nothing gap over the 8 tickers, pool-restricted, mean-of-`M`:
`+19.91, -1.25, -7.91, +6.22, -7.64, +11.43, +27.82, +15.31` percent, i.e.
**mean +7.99%, sd 13.02%, se 4.60%, t = 1.74, 5 of 8 positive.** Doing nothing produced an apparent
8% improvement at t = 1.74. Any pipeline that would have called that a result is broken before it
starts.

Sanity check that the two regenerations really are exchangeable (they are — this is a floor, not
a bug): over all contexts, the mean gap by horizon is `+0.60, +3.09, +3.43, +1.12, +1.27, +1.96,
+2.01` percent with `|t| <= 1.50` everywhere, and 5/8, 3/8, 5/8, 5/8, 4/8, 5/8, 5/8 positive.

### 2.4 How many repeats, and where to spend them

A generation replicate costs a whole inference pass, so `G = 2` is the realistic ceiling, and
`G = 2` gives the floor **1 degree of freedom**. The project has already been burned by a low sd
at 1 df. The fix is cheap and does not need more generations: **estimate the floor as the spread
of the do-nothing gap across the 8 tickers (7 df), not as the spread across replicates within a
ticker (1 df).** That is exactly the `sd 13.02%` above. Both readings use the same two
regenerations; only the df differs.

Minimum repeat counts, as a pre-commitment:

| level | minimum | reason |
|---|---|---|
| generation replicates `G` | 2 for the control arm **and** 2 for the primary treatment arm | a floor measured on one arm and applied to another is the "noise estimate from one group" error; the whole point of the floor is that it is on the same structure |
| training seeds `S` | 5 per arm | at 3 seeds the t multiplier is 4.303 vs 1.96, a 120% penalty; at 5 it is 2.776 (42%); at 8 it is 2.365 (21%) |
| context blocks | all of them | free |
| permutation draws for the E1 nulls | >= 20 (already) | `rho_sd` across draws is 0.02–0.05, small next to the effect |

---

## 3. Power

### 3.1 k per context: 20 is a lower bound, not an estimate

`results/failure_pool_reliability.json` fits `1/rho_k - 1 = slope/k` on `k in {1,2,3,5}` and
reports `k_for_rho_0.80` on the corrected score of **13.3 (NFLX) to 23.6 (JPM), mean 18.4**. That
is where `k ~ 20` comes from. It is an extrapolation 3.7x beyond the largest measured k, and it is
**biased optimistic**.

**[new] Held-out test of the extrapolation.** Fit on `k in {1,2,3}` only, predict `rho` at `k = 5`,
compare to the measured value:

```
score        AMD    AMZN    GOOG    INTC     JPM    META    MSFT    NFLX
raw       +0.098  +0.102  +0.105  +0.066  +0.039  +0.090  +0.101  +0.079
corrected +0.043  +0.069  +0.146  +0.073  -0.077  +0.067  +0.115  +0.048
```

**15 of 16 over-predictions, mean +0.076, sign test p = 0.00052.** Reliability grows *more slowly*
than `1/k`. A two-parameter fit `1/rho_k = a + b/k` (so `rho_inf = 1/a`) gives, on the corrected
score, `rho_inf` of `0.758, 0.611, 0.387, 0.716, 2.869(degenerate), 0.721, 0.516, 0.745` —
**7 of 8 tickers have an asymptote below the 0.80 target**, meaning `rho = 0.80` on the corrected
score would be unreachable at any k. The two-parameter fit is itself fragile (4 points, residuals
up to 0.25), so the honest reading is:

> `k = 20` is the **smallest defensible k**, the true requirement is larger, and there is real
> evidence for a ceiling below 0.80 that no k removes.

**This is exactly the failure mode the brief lists** — an effect that shrinks as n grows because
the power calculation used a quantity estimated from the same small sample. It is already visible
here, before any GPU has been used. The mechanical fix (§3.5) is to **measure** `rho` at `k = 10`
and `k = 20`, which needs 40 seeds on a subset of contexts, rather than extrapolate.

### 3.2 [new] Precision of the pool by k, and why the k requirement depends on what the pool is for

Precision of a top-decile pool selected with `k` rollouts, judged against a disjoint `k = 5`
reference top decile, corrected score, `h = 50`, 30 draws x 8 tickers:

| k | precision | chance | lift | true failures harvested per 1,000 rollouts |
|---:|---:|---:|---:|---:|
| 1 | 0.266 | 0.10 | 2.66x | **26.6** |
| 2 | 0.320 | 0.10 | 3.20x | 16.0 |
| 3 | 0.342 | 0.10 | 3.42x | 11.4 |
| 5 | 0.386 | 0.10 | 3.86x | 7.7 |

Recall of the reference top decile under a cheap screen (screen with `k_screen`, keep top `f`,
confirm the survivors at high k):

```
k_screen=1  top 20%: 0.396   top 30%: 0.505   top 50%: 0.664
k_screen=2  top 20%: 0.467   top 30%: 0.581   top 50%: 0.735
k_screen=3  top 20%: 0.511   top 30%: 0.612   top 50%: 0.768
```

Two-stage screening is worth having but is not a rescue: `k_screen = 3, keep 50%, confirm at 20`
costs `3 + 20*0.5 = 13N` rollouts for 0.768 recall, i.e. `16.9N` per unit of recovered pool,
against `20N` for the direct route — **a 15% saving**. `k_screen = 3, keep 30%` costs `9N` for
0.612 recall = `14.7N`, a **26% saving**.

**The requirement splits by purpose, and the plan must say which it is doing.**

- If the deliverable is a **ranked list of individual contexts** that someone will inspect, or a
  per-context claim, per-context reliability is the requirement and `k >= 20`.
- If the deliverable is a **training set whose only job is to over-weight a region of input
  space**, per-context reliability is not the requirement. A false positive is an ordinary market
  window, which is what replay supplies anyway, so impurity **dilutes the treatment; it does not
  contaminate it**. Under pure dilution the treatment strength scales as `p(k)` and required n as
  `1/p(k)^2`, while cost scales as `k`, so cost-per-power scales as `k/p(k)^2`:
  `14.1 (k=1), 19.5 (k=2), 25.6 (k=3), 33.6 (k=5)`. **Small k wins by 2.4x over k=5** for a
  training pool.

Whether impurity is genuinely dilutive rather than harmful is untested — see `## Open questions`.
Nothing in the plan should assume it.

### 3.3 [new] For a paired arm comparison, only `R = N * k` matters

The variance of a paired difference on shared contexts is dominated by the within-context term,
because the between-context term cancels. So at fixed rollout budget the split between contexts
and rollouts is irrelevant. Measured, do-nothing gap sd across draws, `h = 50`:

```
R = N*k = 500      N=500 k=1: 18.83%   N=250 k=2: 17.05%   N=100 k=5: 17.65%     (mean of M)
                   N=500 k=1: 10.76%   N=250 k=2: 10.30%   N=100 k=5: 10.81%     (trimmed mean)
R = N*k = 1000     N=500 k=2: 12.61%   N=200 k=5: 12.89%                          (mean of M)
                   N=500 k=2:  7.69%   N=200 k=5:  7.54%                          (trimmed mean)
```

`10.76 * sqrt(500/1000) = 7.61` against a measured `7.69`: the `1/sqrt(R)` law holds to 1%.

**This resolves the tension in §3.2 cleanly.** `k >= 20` is a *selection* requirement on a
subsample; the *evaluation* budget can be split however is convenient, because only `R` enters.

### 3.4 Minimum detectable effect, and where it becomes unaffordable

Anchor: **[new]** do-nothing trimmed-mean gap sd per ticker `= 10.76%` at `R = 500`, scaling as
`10.76 * sqrt(500/R)`. Resampling unit = ticker (8 today), paired on contexts. 80% power,
`alpha = 0.05` two-sided, `t_crit = t_{0.975,n-1} + t_{0.80,n-1}`.

| R per ticker per arm | sd per ticker | n_t = 8 | 16 | 32 | 64 | 128 |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 10.76% | **12.40%** | 8.06% | 5.50% | 3.83% | 2.69% |
| 1,000 | 7.61% | 8.77% | 5.70% | 3.89% | 2.71% | 1.90% |
| 2,000 | 5.38% | **6.20%** | 4.03% | 2.75% | 1.91% | 1.34% |
| 4,000 | 3.80% | 4.39% | 2.85% | 1.95% | 1.35% | 0.95% |
| 10,000 | 2.41% | **2.77%** | 1.80% | **1.23%** | 0.86% | 0.60% |
| 20,000 | 1.70% | 1.96% | 1.27% | 0.87% | 0.61% | 0.42% |

**These are lower bounds on the MDE.** They contain the generation and context components and
**omit the training-seed component entirely, which has never been measured on this project.** If
the seed component equals the generation component, every entry multiplies by `sqrt(2) = 1.41`.

Read against §2.3: at today's scale (`n_t = 8`, `R = 5,000` for the existing 500x10 grid) the
broad endpoint can detect about 4%, and the pool-restricted endpoint about **15.6%** (its sd is
13.53% per ticker at `k = 5`, vs 5.04% for the broad endpoint).

**Where the effect size must come from.** No pilot of continual training on the pool exists, so
there is no effect size and therefore no required-n table. The pre-registered rule is:

> Power calculations consume the **lower end of the 95% CI** of the pilot effect, never the point
> estimate. The registry field is `effect_lb`; `effect_point` is not readable by the power
> function (§6 trap T5).

Worked template, so the arithmetic is fixed before the number exists:

```
n_t  =  ( t_crit * sd_ticker(R) / effect_lb )^2
e.g. R = 2,000 -> sd = 5.38% ;  a pilot returning +6.0% [95% CI 2.0, 10.0]  ->  effect_lb = 2.0%
     n_t = (2.9 * 5.38 / 2.0)^2 = 61 tickers          <- not 8; the point estimate would have said 7
```

**Explicitly unaffordable comparisons at present scale**, stated so the plan does not quietly
attempt them:

| comparison | needs | verdict |
|---|---|---|
| pool endpoint, effect < 5%, at `n_t = 8` | `R > 40,000` per ticker per arm | 8 tickers x 2 arms x 5 seeds x 2 gens x 40,000 = **6.4M rollouts** for one contrast |
| any per-ticker per-horizon cell claim | must beat the empirical null p95 = 12.2% (§4) | not affordable at `k <= 5`; demote to descriptive |
| ranking individual contexts at `rho = 0.90` | `k = 30–53` (one-param fit), more under §3.1 | measure `rho` at k=10, 20 first |
| the full 24-cell grid at 5 seeds each | 120 CPT runs | see §4; collapse the grid instead |

### 3.5 Total rollout arithmetic, and the pool-volume constraint nobody has written down

**Evaluation.** `2 arms x S seeds x G generations x n_t tickers x R`:

```
n_t=8,  R= 2,000, S=5, G=2  ->    320,000 rollouts
n_t=8,  R=10,000, S=5, G=2  ->  1,600,000
n_t=32, R= 2,000, S=5, G=2  ->  1,280,000
n_t=32, R=10,000, S=5, G=2  ->  6,400,000
```

For scale: the entire measurement of 2026-09-04 used `8 x 500 x 10 = 40,000` rollouts.

**Selection, which is the binding constraint.** To fill a CPT stage of `E` tokens at pool
fraction `f_pool`, with `L` tokens contributed per pool context and a 10% selection rate:

```
pool contexts needed  =  f_pool * E / L
contexts to score     =  10 * f_pool * E / L
rollouts              =  10 * k * f_pool * E / L
```

| E (stage tokens) | f_pool | L (tokens/context) | k | contexts to score | rollouts |
|---:|---:|---:|---:|---:|---:|
| 1e9 | 0.70 | 6,500 (250 msgs x 26) | 20 | 1,076,923 | **21.5M** |
| 1e9 | 0.70 | 52,000 (2,000 msgs x 26) | 20 | 134,615 | **2.7M** |
| 1e9 | 0.70 | 52,000 | 3 | 134,615 | 404,000 |
| 1e8 | 0.70 | 52,000 | 3 | 13,462 | 40,000 |
| 1e8 | 0.30 | 52,000 | 3 | 5,769 | 17,300 |

**The row that matters: a 1B-token stage at 70% pool, `L = 6,500`, `k = 20` costs 21.5M rollouts
of 250 messages = about 140 billion generated tokens, more than the pre-training run itself.**
That combination is not a plan, it is an arithmetic error waiting to be discovered late. The plan
must pick one of: a long real window per pool context (`L >= 52,000`), a small first stage
(`E = 1e8`), a small selection k (§3.2 argues this is correct for a training pool anyway), or a
lower pool fraction.

Note also the §1.1 ceiling: 134,615 contexts is 596 per ticker at `n_t = 226`, or 16,827 per
ticker at `n_t = 8` — and at `N = 16,827` the design effect is about 4.5, so the 8-ticker version
buys far less than it looks. **Scale tickers, not contexts.**

---

## 4. Multiplicity

### 4.1 How many comparisons the plan is currently proposing

| axis | levels | source |
|---|---:|---|
| pool fraction | 4: {0.0, 0.3, 0.7, 1.0} | issue #73 asks 100% vs a mix, floats 70/30 |
| peak LR multiplier | 2: {0.3, 0.5} | `PLAN.md` §3 Step 3 |
| replay ratio rho | 3: {0.05, 0.10, 0.25} | `PLAN.md` §3 Step 3 |
| CPT stages | 2 | `PLAN.md` §3 Step 3 |
| tickers | 8 | today's grid |
| horizons | 7 | `HORIZONS` |

```
arms in the full grid                     4 x 2 x 3          =   24
contrasts against the control cell                            =   23
x stages                                                      =   46
x tickers x horizons                     46 x 8 x 7           = 2576   cell-level comparisons
plus LOBbench divergence (1) and return-bench IC / ranked IC / direction at 4 horizons (12)
```

Bonferroni over `m = 2576` requires `|z| >= 4.272`, i.e. `p < 1.94e-5`. Nothing in this project has
ever produced such a z on a generative endpoint. Bonferroni over the pooled family `m = 46`
requires `|z| >= 3.267`. **[new]** And the empirical do-nothing null over the same 56-cell
ticker x horizon shape already reads:

```
|relative gap| over 56 do-nothing cells:  p50 2.21%  p75 3.81%  p90 8.90%  p95 12.24%  p99 23.27%  max 33.24%
4 of 56 cells exceed 10%.  The largest, GOOG at H=25, is +33.24%.
```

### 4.2 The concrete correction

**One pre-registered primary endpoint, one test.** Written out so that a script can check it:

```yaml
primary:
  claim:        continual training on the failure pool improves pool-like contexts
  estimand:     theta_pool(a=0.7, h=50)            # section 1.2 E2b
  statistic:    Trim_0.05 of per-context M, differenced arm vs control, paired on contexts
  arms:         pool_fraction 0.7 vs 0.0, both at peak_lr_mult 0.5, replay_rho 0.10, stage 1
  unit:         ticker (n_t), paired
  seeds:        5 training seeds per arm
  generations:  2 per arm
  pool P:       selected by S from a rollout set disjoint from the graded set
  test:         two-sided paired t on the 8 ticker-level gaps, alpha = 0.05
  margin:       none (superiority)
  registered:   before the first arm runs, in results/preregistration.yaml, with a git SHA
```

**One pre-registered co-primary**, because `PLAN.md` §2.4 already forbids reporting a single
coordinate:

```yaml
co_primary:
  estimand:  theta_stab = NLL_old(0.7) - NLL_old(0.0)      # section 1.3 E3a
  test:      one-sided non-inferiority against delta_0, alpha = 0.05
  delta_0:   TO BE SET BEFORE THE FIRST ARM RUNS  (see Open questions)
```

Two co-primaries need no alpha split if the claim is the **conjunction** ("improved AND not
damaged"): an intersection-union test rejects only when both reject, and its level is `alpha`, not
`alpha/2`. That is the honest structure for this plan and it is free.

**Secondary family A — the mix-ratio axis.** Because `{0.0, 0.3, 0.7, 1.0}` is **ordered**, use one
trend contrast, not three pairwise tests: `sum_l w_l * theta_pool(f_l)` with
`w = (-0.55, -0.15, +0.25, +0.45)` (centred levels). That is 1 test, not 3, and it answers the
question issue #73 actually asks ("100% or a mix?") better than three pairwise ones do.

**Secondary family B — everything else in the 46-contrast pooled family.** Benjamini-Hochberg at
`q = 0.10`, with the family size `m = 46` **printed next to every q-value**. First rejection needs
`p <= 0.10/46 = 0.00217`.

**Tertiary — the 2,576 cell-level readings.** Reported as a heat map, no inference, and any cell
claim must clear the **measured** family-wise threshold rather than a Gaussian one:

```
cell-level threshold  =  p95 of the do-nothing null over the same 56-cell shape  =  12.24%
family-wise threshold =  max of the do-nothing null over the same shape         =  33.24%
```

These thresholds are re-estimated from the control arm's own two generation replicates each time
the design changes, so they are never imported from this file.

**What is dropped.** The `2 x 3` LR x replay grid is not part of any confirmatory claim. `PLAN.md`
Step 3 already frames it as a *fitting* exercise for the arXiv 2505.07796 shift term, not a
testing exercise: fit the law, report the fitted surface with its residuals, and make **no
pairwise cell claims at all**. That removes 23 of the 46 contrasts by construction.

---

## 5. Stopping and reporting rules

### 5.1 When a number may appear in a title, be bolded, or be called "best"

The project rule from prior sessions is that no number enters a title, bold, or a superlative
until `n` has stopped growing. Made mechanical:

> A number may appear in a title, be bolded, or be described as "best", "cleanest", "largest" only
> when **all** of:
> 1. `S >= 5` training seeds and `G >= 2` generation replicates are in the estimate;
> 2. the estimate has been recomputed at least once after `n` increased, and moved by less than
>    one of its own standard errors;
> 3. the 95% CI excludes the pre-registered null value;
> 4. its `slice` field is complete — score, k, horizon, ticker set, seed set, stage — and the
>    claim text does not generalise beyond that slice.
>
> Until then the number is reported in a table, unbolded, with its `n` beside it.

The df penalty makes 5 the floor rather than an aesthetic preference:

| seeds `S` | df | `t_{0.975}` | penalty vs z |
|---:|---:|---:|---:|
| 2 | 1 | 12.706 | 548% |
| 3 | 2 | 4.303 | 120% |
| 5 | 4 | 2.776 | 42% |
| 8 | 7 | 2.365 | 21% |
| 10 | 9 | 2.262 | 15% |

### 5.2 What must be printed next to any quantile statistic

> Every quantile, tail, decile, or top-`x`% statistic prints `n_events` in the same cell. The
> renderer refuses to emit the statistic if `n_events < 30`, and greys it if `n_events < 100`.

This is not decorative. Today's pool endpoint is a top-decile mean over `n_events = 50` contexts
per ticker, and its noise floor is **twice** that of the 500-context endpoint (§2.3). A reader
who sees `+8%` without `n = 50` cannot tell those apart. The project has already read a verdict at
a quantile carrying 8 events.

### 5.3 What "the effect shrank as n grew" obliges you to report

> If the point estimate at the final `n` differs from the estimate at any earlier `n` by more than
> one standard error, the report must contain the **whole trajectory** — a table or figure of the
> estimate and its CI against `n` — and the abstract must quote the final value, never the
> largest.
>
> Any power calculation performed at the earlier `n` is republished alongside the trajectory with
> the note that it used an effect size from the same sample, and it is recomputed from
> `effect_lb`.

This project has three recorded instances of the same shrinkage (`+1.66% -> +1.38%` from 8 to 13
seeds; `-0.01% -> -1.80%` from 1 to 6 seeds; `0.08pp -> 0.66pp`), and §3.1 above adds a fourth
before any GPU has run: the reliability curve over-predicts by `+0.076` at the largest held-out
`k`, 15 times out of 16. **The trajectory is the finding, not an appendix to it.**

### 5.4 Other standing rules

- Both coordinates always, per `PLAN.md` §2.4: an improvement number never appears without the
  old-window NLL number in the same table row.
- A selection threshold is never reported alone: every top-decile result is accompanied by the
  same result at top-25%, and if the sign flips, neither is reported as a result. (§6 trap T7.)
- Non-inferiority margins are quoted with their date of registration and the git SHA of the file
  that holds them.
- Any claim about a horizon quotes the horizon; any claim about a ticker set enumerates it.

---

## 6. The traps already committed on this project, as assertions a script can make

Each row is written so a `pytest` in
`/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/code/test_stats_contract.py` (to be
written) either passes or fails. "Be careful" appears nowhere.

| # | Trap | Assertion |
|---|---|---|
| T1 | A knob set, printed and recorded but never reaching the code (`TOKEN_MODE` pinned in five places, four silently) | The analysis re-derives `k`, `N`, `n_tickers`, `horizon`, `score_name` **from the loaded arrays**, writes them into the output JSON, and asserts they equal the requested values: `assert out["k_observed"] == args.k`. The report renders only from the output JSON; a lint rejects any literal number in the markdown that is absent from it. |
| T2 | A default that never applied because something downstream overwrote it | `assert not re.search(r"getenv\(|environ\.get\(", src_of(stats_module))` on the statistics path, and `out["argv"] = sys.argv` with `assert out["argv"]` non-empty. Any unresolved value is written as `null`, never `"unknown"`. |
| T3 | A metric whose name is not its semantics (`rollout/rewards` was normalised advantage) | Every metric name maps to a `formula` string in a registry, and a known-answer test evaluates it: `assert metric("trimmed_mean_M")(synthetic) == pytest.approx(known)`. A metric with no registry entry cannot be written to the output. |
| T4 | Dividing by a per-group constant shrinks noise but not bias, manufacturing significance | `assert estimator(arm, arm) == 0.0` exactly, and `assert t_stat(arm, arm) == 0.0`, for every arm-level estimator. Plus: `assert denominator(arm_a) is denominator(arm_b)` — the same object, not an equal value. |
| T5 | An effect that shrinks as n grows, because power used an effect size from the same small sample | `power_n(effect_lb=..., sd=...)` has no `effect_point` parameter; `assert "effect_point" not in inspect.signature(power_n).parameters`. The registry stores `effect_lb` and `effect_point` in separate fields and the power module imports only the former. |
| T6 | Claiming a whole line from one measured slice | Every claim record carries `slice = {tickers, horizons, score, k, seeds, stage}`; a lint rejects claim text matching `\b(best|cleanest|largest|always|universal)\b` unless `slice` covers the full declared axis. |
| T7 | A selection rule's consequence mistaken for a property of the data | Every selection runs at two thresholds (top 10% and top 25%) and both land in the output: `assert {"top10","top25"} <= set(out["selection"])`. If `sign(effect_top10) != sign(effect_top25)`, `out["blocked"] = True` and the renderer refuses the claim. |
| T8 | A null control that shares its error with the treatment | `assert diff_fields(arm_cfg, control_cfg) == {declared_factor}` — the control differs in exactly the intended field. And the four pairing nulls must be present with `assert abs(nulls["cross"] - nulls["independent"]) < 2*se` before any reliability claim (§1.1 E1b). |
| T9 | Noise floor estimated on a different structure from the effect | `assert floor_fn is effect_fn` (identity, not equality) and `assert floor_kwargs["paired"] == effect_kwargs["paired"]`. The floor call is `effect_fn(control_g1, control_g2)`; the effect call is `effect_fn(control, treatment)`. |
| T10 | A verdict read at a quantile with 8 events | Every quantile record carries `n_events`; `assert rec["n_events"] >= 30` or the renderer emits `n/a (n=8)` instead of the value. |
| T11 | Bootstrap over trading days when days carry 1–58 contexts each — and, here, when the day is not recorded at all | Generation writes `context_manifest.parquet` with `(context_id, ticker, date, message_offset, dataset_index, block_id)` **before the first rollout**; `assert set(manifest.columns) >= REQUIRED`. Any day-level estimator asserts `estimator_kind == "ratio_of_sums"` and records `day_context_counts` (min, median, max) in the output. Below the day level, resampling uses `block_id = dataset_index // 500` (§1.1). |
| T12 | Bootstrapping inside a single generation implementation, so "regenerate everything" variance is structurally absent | `assert out["n_generation_replicates"] >= 2` and `assert len(set(boot_index["g"])) > 1` before any interval is emitted. Also record `xla_gpu_autotune_level` per generation, so a deterministic and a non-deterministic replicate are never silently mixed. |
| T13 | Plug-in CRPS biased by `E|X-X'|/(2K)`, so a narrow ensemble is rewarded | `assert K_arm_a == K_arm_b` for every CRPS comparison, and use the fair (unbiased) form whenever `K` varies. Known-answer test: two arms differing **only** in `K` must return exactly `0.0` under the fair form. At `K = 5` the plug-in bias is 10% of the mean absolute pairwise spread; at `K = 20`, 2.5%. |
| T14 | Resampling ensemble members with replacement breaks the `K(K-1)` pairing term | `assert bootstrap_over_members.replace is False` (jackknife or without-replacement subsampling only), or recompute the pairing term for the resampled multiset. Known-answer test: a uniform proposal must reproduce the analytic SD. |
| T15 | Acceptance criteria that are unconditional statistics (PR#22: destroying context pairing left W1 bitwise unchanged while CRPS moved 12.8–20.0%) | Every acceptance metric runs a **context-shuffle probe** at registration time; `assert metric(shuffled) != metric(paired)` or the metric is tagged `unconditional=True` and cannot be a primary endpoint. |
| T16 | The selection score used as the acceptance score | `assert primary_endpoint.metric != selection.metric`, and specifically `assert not is_within_bin_rank(primary_endpoint.metric)` — a within-bin rank has mean exactly `0.500000` for every arm (§1.2, measured). |
| T17 | Reliability quoted from an extrapolation | `assert deployed_k <= max(measured_ks)` or the output carries `extrapolated=True` and the reported `k` is labelled `>= k_hat`, never `= k_hat`. The held-out check of §3.1 (fit on `k<max`, predict at `k=max`) is run and its signed error is written to the output. |

---

## Open questions

1. **Training-seed variance of every endpoint is unmeasured.** All the MDEs in §3.4 omit it, so
   they are lower bounds. The cheapest measurement is also a control the plan needs anyway: run
   the control arm (pool fraction 0.0) at 5 training seeds, change nothing else, and take the sd
   of each endpoint across those 5. Until that number exists, no required-n table should be
   published.
2. **Reliability at `k = 10` and `k = 20` has never been measured**, only extrapolated, and the
   extrapolation is biased optimistic by `+0.076` at the one held-out point (§3.1). Resolving it
   needs 40 independently seeded rollouts on a subset of contexts (2 tickers x 500 contexts x 40
   seeds = 40,000 rollouts, the size of the whole 2026-09-04 grid). Whether that is worth doing
   before the pool campaign is a decision, not a fact.
3. **The day of a context is not recorded** anywhere in the current artefact (§1.1). Whether the
   upstream dataset builder can emit `(date, message_offset)` for a given `dataset_index` without
   re-running the shard build is unknown. If it can, the block-resampling scheme of §1.1 should be
   replaced by real day clustering.
4. **`delta_0` and `delta_1`, the non-inferiority margins for E3a and E3b, do not exist.** They
   cannot be derived from anything measured so far. Someone has to decide how much old-window NLL
   the project is willing to pay for a given new-window gain, in nats per token, before the first
   arm runs.
5. **The trimmed-mean endpoint and the pool-restricted endpoint conflict.** The trim removes the
   top 5% of contexts by error, which is a subset of exactly the contexts the pool targets. On the
   broad endpoint the trim is correct (the tail is noise); on the pool endpoint it may be removing
   the signal. §2.3 shows the trim still helps on the pool endpoint (7.96% vs 14.56% floor) but
   whether it also removes the treatment effect is untested.
6. **Whether pool impurity is dilutive or harmful is untested**, and §3.2's cost argument for small
   `k` depends entirely on it. A cheap test exists: train one arm on a deliberately impure pool
   (top decile at `k = 1`, precision 0.266) and one on a pure-as-affordable pool (top decile at
   `k = 5`, precision 0.386), matched on pool size, and compare. That is 2 extra arms.
7. **Which horizon is primary is undecided.** §2.2 shows the generation fork rate varies from 2%
   at `h = 10` to 81% at `h = 250`, and §2.3 shows the floor differs by horizon. `h = 50` is used
   throughout this draft only because it is `--horizon-idx 2`, the existing default. That is not a
   reason.
8. **`L`, the tokens a pool context contributes to training, is a free parameter** in the budget
   arithmetic of §3.5 and swings the answer by 8x. It depends on a design choice nobody has made:
   does a pool context contribute its own 250-message continuation, a longer real window around
   it, or the generated rollout itself?
9. **Whether `episode_builder.py`'s frozen-background-flow machinery
   (`/lus/lfs1aip2/projects/public/u6gb/sigma-0/src/post_training/heuristic_learning/episode_builder.py`)
   can be reused by the return-rollout pipeline is unknown.** If it can, arms face a byte-identical
   market and the pairing in §1.2 becomes much stronger than context-sharing alone. **[new]**
   Today, sharing only the *seed labels* between two runs buys a factor of 0.90x to 3.71x on the
   standard error, median 1.70x, and is actually *worse* than reseeding for 3 of 8 tickers —
   because the arm statistic is a mean of squared errors dominated by a few contexts that fork
   regardless. So seed-sharing alone is not a substitute for frozen background flow.
10. **`--xla_gpu_autotune_level=0` costs 1.49x wall clock and changes the numerics.** Whether the
    deterministic kernels shift the model's *distribution* (as opposed to fixing one draw of it) is
    untested. If they do, a deterministic evaluation is measuring a slightly different model than
    the non-deterministic training produced, and that difference has to be bounded before
    determinism is adopted as the evaluation setting.
11. **`n_t = 8` is the current ticker count; the SP500 corpus holds roughly 483.** §3.4 shows
    tickers are the cheapest axis for power and §1.1 shows contexts saturate. Whether the
    contexts-per-ticker infrastructure generalises to 32 or 128 tickers without re-running the
    frozen-context selection is not known.

---

### Reproducing the [new] numbers

All of these run on a login node in under two minutes, CPU only, no `find`, no recursive listing.

```bash
cd /lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning
python3 -c "
import sys; sys.path.insert(0,'code')
import numpy as np, failure_pool_reliability as F
root=F.DEFAULT_ROOT
for tk in ['AMD','AMZN','GOOG','INTC','JPM','META','MSFT','NFLX']:
    rb,gb,sb=F.load_arm(root,'v5me3repro',tk); rc,gc,sc=F.load_arm(root,'v5me3repB',tk)
    cm=sorted(set(sb)&set(sc))
    B=gb[[sb.index(s) for s in cm]]; C=gc[[sc.index(s) for s in cm]]
    phi=[(B[:,:,h]==C[:,:,h]).mean() for h in range(7)]              # bitwise-identical fraction
    SB=F.scores(rb,B)['total']; SC=F.scores(rc,C)['total']
    rho=[F.spearman(SB[:,h],SC[:,h]) for h in range(7)]              # regeneration null
    print(tk, len(cm), np.round(phi,3), np.round(rho,3))
"
```

The split-half-at-matched-k, the trimmed-mean floor table, the fixed-`R` result, the two-stage
screen, the held-out `k` extrapolation test and the index-block ICC all follow the same shape:
`load_arm` -> `scores` / `stratify` -> `spearman` or a trimmed mean, with `numpy` only. The one
external dependency used above is `scipy.stats` for the `t` quantiles in §3.4 and §5.1.
