# R3 — causal and design validity: does each measurement answer the question it is quoted for?

> Third of five reviewers, launched alone. Lens: the mapping from measurement to claim, not
> the arithmetic (R2 covered that). Read: `RESULTS_20260905.md` (11 addenda), both
> pre-registrations, `PLAN.md` §0.1–§0.13 and the task board, and
> `plan_20260904/drafts/D3_profiling.md` / `D4_deletion_refactor.md`, which no reviewer has
> seen before. Every number below was recomputed on the login node from the archive; no GPU,
> no `sbatch`, no deletion, and nothing outside `plan_drafts/` was modified. Full paths:
>
> ```
> /lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/
> /lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z/data/
> /lus/lfs1aip2/projects/public/u6gb/sigma-0/logs_lobs5/
> ```

## 中文速览

- **三条 BLOCKING。**
  1. **稀释测试（附录 9 / S1 / C5）在算术上不可能给出阳性结果。** 杂质本身就是"从补集里随机抽"，
     而要检验的假说恰好是"杂质比从补集里随机抽更差"——**要检验的东西和它的零效应组是同一个对象**，
     正是本项目自己记过的那种失效形态。而且被测量是池内成员分数的**均值**，均值对混合比例 `p`
     **恒为一次函数**，所以那张"线性预测"列就是两个端点的插值（我复算到小数点后四位完全吻合）。
     那个 ±0.037 的"分辨率"量的是抽到哪些成员的抖动，不是污染的上界。
  2. **附录 6 里唯一被判为 ESTABLISHED 的那条（realised-move balance），把分数换成纯噪声后还剩 85%。**
     `stratify_v2` 返回的已经是层内秩，所以"全局取前 q"本来就近似等于"每层取前 q"：两条规则的池
     只差 0/0/1/1/1/5/5/5 个成员（JPM 与 NFLX **完全一样**）。真正剩下的差别是 `round(q·n_s)`——
     零位移那一层有 97–208 个成员被**向上**取整，十个正位移层各 29–41 个被**向下**取整，于是
     within 规则机械地少拿约 3 个大 |y| 成员、多拿 1 个零位移成员。零效应检验：把分数换成 i.i.d.
     噪声，bal 从 1.0263 掉到 0.9822，Δ = **−0.0441**，而发表的 Δ 是 **−0.0519**。
  3. **那 28 个被生成两次的 context 不是 5.6% 的随机样本，是一到两个交易日。** 我用
     `data_real` 文件名里的日期做了外部核对：id 与日期严格同增（500 个 id 零处逆序）。
     AMD 与 AMZN 的 28 个**全部落在 2026-01-02** 这一天，其余六只票分布在 2026-01-02 与 2026-01-05。
     它们的 |y| 只有其余 472 个的 **0.791 倍**（t = −2.75，df 7，95% CI [0.647, 0.968]，7/8 只票低于 1），
     JPM 上分数的秩和 z = **+3.17**。池里放进 11 个，bal 就动 **−0.112**——比附录 6 判为已确立的
     那个规则间差 −0.052 大一倍多。**"按总体比例出现而非集中"回答的是另一个问题**：
     它查的是选择有没有偏爱它们，查不出它们本身与其余部分不可交换。

- **六条 MAJOR。** X4 的"血缘"结论是拿**两个标量范数之差**（不是参数向量的距离）在 3 个梯级上单调
  推出来的，而参照那条链自己的范数在训练中**不单调**（427.19 → 423.49 → 425.09），整条梯级的
  跨度 0.525 只有那条链自身摆幅 3.69 的七分之一；`num_errors` 的对应关系确实强到足以排除任何错误
  置换（123 个不同取值、碰撞概率 0.0098），但它比的是**同一个写入方的两份输出**，而真正能锚定
  id↔市场窗口的外部证据（`data_real` 的日期）当时没用上，我补做了、结论是干净的；`bal` 把两个
  方向相反的通道乘在一起，构成通道的 |comp−1| 从 0.0155 变成 0.0156（**一点没改善**），
  "池的分层构成按构造等于总体"在实现里不成立；"不显著即等同"这个已被附录 8 撤回的动作
  在另外**七处**仍然活着，其中 M3 的时代关卡（`plan_drafts/04` §5.2 与 `plan_drafts/01` §2.4）
  写的是"分不出来 ⇒ 以欠学习为主 ⇒ 回放是白赚的 ⇒ §5.3 不再是承重的一步"；附录 6/7/9 的
  **分析脚本一行都没进仓库**，附录 9 连结果文件都没有，理由是 inode 打满——而此刻实测还有
  **2,169,668 个空闲 inode**；D3 §4.1 用"MFU 超过 100% 所以不是 FP32 路径"当证据是循环论证，
  因为分子里的 2.630 校正系数本身就是用**张量核计数器**标定的（去掉它读数是 64.3%，掉到 100% 以下）。

- **确认干净的**：D3 的吞吐表逐格复现（0.5687/0.0044/0.77% 等四行、`Tokens/step: 52,000`、
  correction 2.630、693 条 Timing 行）；3.038 GPU-小时/十亿 token、781.6e9 token/epoch 的两条独立
  交叉验证、2.07e8 的换算、22.9%/169%、3.2%/2.5% 的开销加总、"小 13–26 倍"全部对；D4 的
  36.47 GiB 垃圾包与 31,368 个被跟踪的 CSV 实测仍然一致；`world_size = 1` 所以
  `rank_indices == all_indices`，没有分片歧义；附录 8 自己的更正与 C4 的"NOT established"我没能推翻。

---

## Findings

| # | severity | file / section | claim attacked | defect |
|---|---|---|---|---|
| F1 | **BLOCKING** | `results/RESULTS_20260905.md` addendum 9; `results/RESULTS_20260905.md` addendum 5 §D; `PLAN.md` board S1, C5 | "contamination … impurities actively worse than a random draw from the complement — is not present at a size this test could see" | the impurities **are** a random draw from the complement, and the outcome is a mean over pool members, hence affine in `p` by construction. The alternative hypothesis is arithmetically unreachable |
| F2 | **BLOCKING** | `results/RESULTS_20260905.md` addendum 6 ("Realised-move balance: ESTABLISHED"); the adoption of `R_v2_within` | "it is a *design* property — the pool's stratum composition equals the population's by construction" | the two rules are the same rule up to `round(q·n_s)`: pools differ by 0–5 of 50 members and are **identical** for JPM and NFLX. A pure-noise score reproduces 85% of the published `bal` gain |
| F3 | **BLOCKING** | `results/RESULTS_20260905.md` addendum 7 ("at the population rate rather than concentrated"); `results/PREREG_context_holdout_20260905.md` Part 2 | the 28 twice-generated contexts are handled by counting them | they are **one or two calendar days of 20**, with |y| at 0.791× the rest (t = −2.75, CI [0.647, 0.968]). The rate check cannot see non-exchangeability, and 11 of them move `bal` by −0.112 |
| F4 | MAJOR | `results/RESULTS_20260905.md` addendum 5 §C/§E; `PLAN.md` §0.11, board §4 | "the distance grows monotonically with fine-tuning steps … So `wm_ft_multi3` is a fine-tune **descended from** the selftrain checkpoint" | the reported "distance" is |‖θ_A‖ − ‖θ_B‖|, not ‖θ_A − θ_B‖; the tool named in §C cannot produce the pre-registered "identical to machine precision" branch; and the same scalar is **non-monotone** in training on the reference run itself |
| F5 | MAJOR | `results/RESULTS_20260905.md` addendum 5 §A; `PLAN.md` §0.11 | "The mapping … Verified by recomputing the definition from the generated books" | it compares two outputs of the same writer. The external anchor (`data_real` filename dates) was available and unused. I ran it — it passes, and that is what should carry the claim |
| F6 | MAJOR | `results/PREREG_selection_rules_20260905.md` ("`bal` … A rule that has removed the realised-move confound has `bal ≈ 1`") | `bal` = pool mean \|y\| / population mean \|y\| as the summary for the confound | with a 19.4–41.6% zero atom, `bal` = comp × cond and the two move in opposite directions: comp 1.0155 → 0.9844 (\|comp−1\| **unchanged**), cond 1.2871 → 1.2759. `bal ≈ 1` is reachable by cancellation |
| F7 | MAJOR | `plan_drafts/04_training_design.md` §5.2, §5.3; `plan_drafts/01_measurements.md` §2.4; `results/RESULTS_20260905.md` addendum 6/7; `plan_20260904/drafts/D1_failure_pool.md` §8 | seven surviving instances of the move addendum 8 withdrew in one place | "**Indistinguishable** ⇒ underlearning dominates … replay is free" is the M3 era gate's pre-registered reading rule, and it is unfixed |
| F8 | MAJOR | commits `ca8f1047`, `e116e5eb`, `0a659850` | three pre-registered analyses reported as run | the analysis code was never committed; addendum 9 emitted **no artefact at all**. The stated reason (inode cap) is a stale reading: 2,169,668 inodes are free |
| F9 | MAJOR | `plan_20260904/drafts/D3_profiling.md` §4.1 | "the cheapest available falsification test … makes the claim … **evidence rather than assertion**" (MFU 169% > 100% against the FP32 peak) | the numerator's 2.630 correction is itself calibrated from **tensor-pipe** counters. Remove it and the reading is 64.3%, below 100%, and the falsification fails |
| F10 | MINOR | `plan_20260904/drafts/D3_profiling.md` §4.1 (P5 decision rule) | "if it moves it by under a percent … TF32 is the divisor to standardise on" | a null timing result on the contractions is mapped onto a dtype question about the projections; and it is F7's shape again |
| F11 | MINOR | `plan_20260904/drafts/D4_deletion_refactor.md` §2.1, §2.4, §6 step 6 | "18 unstaged deletions … Commit the deletion — risk: none" | contradicted by `RESULTS` addendum 2 §A ("**not re-deleted**") and now a no-op: `git ls-files --deleted` returns 0 |
| F12 | MINOR | `PLAN.md` board §2 | the claim table | nothing tracks the `R_v2_within` adoption that addendum 6 makes |
| F13 | MINOR | `plan_20260904/drafts/D3_profiling.md` §3.1 note 3 | "Weak-scaling loss is 8% out to 8 GPUs" | the 4-GPU and 8-GPU rows straddle the 1-node/2-node boundary, so the 8% mixes intra-node with inter-node scaling |

---

## F1 — BLOCKING. The dilution test cannot produce a positive result for any data

**The claim.** `results/RESULTS_20260905.md` addendum 9:

> "**No faster-than-linear fall is detected** … What can be said is that **contamination of the
> kind the design was built to catch — impurities actively worse than a random draw from the
> complement — is not present at a size this test could see, which is anything above about 5%
> of the effect.**"

`PLAN.md` board marks S1 $\color{green}{\textsf{DONE}}$ and grades C5 "no departure detected at
~5% resolution", and the `k = 3` budget of §0.3 "may be quoted only with that resolution
attached".

**Why it fails, mechanically. Two independent reasons, either of which is sufficient.**

**(a) The treatment and its own null are the same object.** The design
(`results/RESULTS_20260905.md` addendum 5 §D, steps 1–2) builds the impurity arm as "a
**matched-size pool of contexts drawn at random**", and addendum 9 confirms it: "the `p = 0`
arm sits at −0.09 rather than 0 because **the non-reference draws come from the complement of
the top decile**". The hypothesis under test is that the impurities are "actively worse than a
random draw from the complement". The impurities *are* a random draw from the complement. The
mean score of a random draw from the complement equals the complement's mean by construction,
so the contaminating alternative has been defined out of existence. This is the project's own
recorded failure shape — *a null control that shares its error with the treatment, so it is
not null* — with the roles reversed: here the treatment shares its construction with the null.

**(b) A mean over a mixture is affine in the mixing fraction.** The outcome is "mean score
inside the pool minus the population mean". For a pool of fixed size `N₀` holding `p·N₀`
members drawn from the reference set `R` and `(1−p)·N₀` from the complement `C`:

```
E[contrast(p)] = p·μ_R + (1−p)·μ_C − μ_pop      exactly affine in p, for any μ_R, μ_C
```

Nothing about a member being "harmful to training" enters a mean of scores. The published
"linear prediction" column is not a fitted model with residual degrees of freedom — it is the
two-endpoint interpolation, and I verified it to four decimals against addendum 9's own table:

```
p = 0.75   0.75(0.8064) + 0.25(−0.0915) = 0.581925   published 0.5819
p = 0.50   0.50(0.8064) + 0.50(−0.0915) = 0.357450   published 0.3575
p = 0.25   0.25(0.8064) + 0.75(−0.0915) = 0.132975   published 0.1330
sensitivity, p = 0.75: 0.75(0.8120) + 0.25(−0.0959) = 0.585025; published 0.5892 − gap 0.0042
```

So the three "deviations from linearity" (+0.0005, +0.0079, +0.0027) are the sampling
variation of *which* random members were drawn, and the ±0.037 SD "resolution" bounds that
draw noise. It is not a bound on contamination.

**The concrete wrong answer.** `k = 3` — the number that makes the whole cycle-1 pool fit in
the inode budget (§0.3: 96 members, 20% of free) — is now quoted with a resolution statement
derived from a test whose alternative was unreachable. A reader who takes "no departure
detected at ~5% resolution" at face value concludes that contamination above 5% of the effect
has been excluded. Nothing has been excluded. The `k = 3` budget is exactly as unsupported as
it was before addendum 9, and `PLAN.md` §0.9's "It is still untested" was the correct entry.

**The cheapest check that settles it.** Two commands, both CPU, both under a minute:

1. Recompute the "linear prediction" column as `p·contrast(1) + (1−p)·contrast(0)` from
   addendum 9's own endpoints. If it reproduces the column to four decimals — it does — the
   column is an identity, not a prediction.
2. Re-run the whole design with the score replaced by `rng.standard_normal(n)`. The same
   linearity, with the same residual size, appears. A test that gives the same answer on noise
   is not measuring the data.

**What would actually distinguish dilution from contamination.** The alternative has to be a
population that can differ from a random complement draw. Two candidates, in cost order:

- **CPU, on this archive.** Define the impurity as the rule's *observed* false positives:
  contexts in the top decile on `SELECT` seeds 97701–97705 but **not** in the top decile on
  `EVAL` seeds 97706–97710. Those are the windows the rule actually gets wrong, and they are
  not a random complement draw — they are complement members the rule ranked highly. Build two
  mixture ladders, one with those as the impurity and one with random-complement impurities,
  and compare the curves. Dilution predicts the two ladders coincide; contamination predicts
  the false-positive ladder falls faster. Both arms are means, so both are affine — but they
  are affine with **different slopes**, and the slope difference is the estimand. That is a
  reachable alternative and the archive already contains everything it needs.
- **GPU, and the only thing that answers the question asked.** Two training arms at `p = 1.0`
  and `p = 0.5`, same budget. `plan_drafts/02` §3.2's claim is about a *training* pool; no
  functional of the pool's own scores can test it.

Until one of those runs, C5 should read **UNTESTED**, as `PLAN.md` §0.9 had it, and S1 should
not be green.

**Severity: BLOCKING.** A pre-registered test is recorded as executed and bounding, its result
grades a claim, and that claim sizes the generation budget.

---

## F2 — BLOCKING. Addendum 6's only ESTABLISHED result is a rounding rule; noise reproduces 85% of it

**The claim.** `results/RESULTS_20260905.md` addendum 6:

> "**Realised-move balance: ESTABLISHED, and it is the only thing that is.** `|bal − 1|` is
> smaller for within-stratum selection in **6 of 8 tickers under both analyses** … That is the
> property D1 §2 predicted, and it is a *design* property — **the pool's stratum composition
> equals the population's by construction** — rather than an empirical win, which is why it
> survives an analysis in which the score contrast does not."
>
> "**Adopted, on the balance ground alone**: `R_v2_within` for any future pool."

**Why it fails, mechanically. Three layers.**

**(a) The two rules are the same rule.** `stratify_v2`
(`/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/code/failure_pool_reliability.py:181`)
returns, for every context, `_rank(score within its stratum) / (n_s − 1)` — a within-stratum
rank on `[0, 1]`. Taking the **global** top `q` of within-stratum ranks therefore already takes
approximately the top `q` of **every** stratum. Measured, `q = 0.10`, `H = 50`, selection seeds
97701–97705, all eight tickers:

```
ticker  |pool(global) Δ pool(within)|   identical?
AMD              5                       no
AMZN             5                       no
GOOG             5                       no
INTC             1                       no
JPM              0                      YES
META             1                       no
MSFT             1                       no
NFLX             0                      YES
```

Symmetric difference is 2.25 members of ~50 on average — the two pools agree on 97.7% of their
membership, and for **JPM and NFLX they are the identical index set**. So the "6 of 8" tally is
6 strict differences and 2 exact ties, and two of the eight replicates carry no contrast at all.

**(b) What remains is `round(q·n_s)`, and its direction is fixed.** Per-stratum membership,
`q = 0.10`:

```
AMD   stratum sizes  [169, 33, 33, 33, 33, 33, 34, 33, 32, 33, 34]
      proportional   [16.9, 3.3, 3.3, 3.3, 3.3, 3.3, 3.4, 3.3, 3.2, 3.3, 3.4]   sum 50.0
      global picks   [16,   3,   4,   3,   4,   3,   4,   3,   3,   3,   4  ]   sum 50
      within picks   [17,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3  ]   sum 47
```

`select_within_stratum` takes `int(round(frac * idx.size))` inside each stratum. With one
zero-move stratum of 97–208 members and ten positive strata of 29–41, `round` sends the single
large stratum **up** (16.9 → 17) and the ten small ones **down** (3.3 → 3). The within rule
therefore drops about three high-|y| members and adds one zero-|y| member, out of 50. `bal` is
a ratio of mean |y|, so it must fall. This is deterministic and has nothing to do with the
score.

It also breaks the pre-registration's own premise. `results/PREREG_selection_rules_20260905.md`
says "Three selection rules, **all building a pool of the same size** from the same contexts".
Measured sizes: `R_v1_global` and `R_v2_global` are 50 everywhere; `R_v2_within` is
[47, 47, 47, 51, 50, 51, 51, 50] in the sensitivity analysis and [45, 46, 46, 49, 49, 50, 50, 49]
in the primary. The arm being adopted is the one graded on a different pool size, and the size
varies with the zero-atom share — the very quantity `bal` exists to control.

**(c) A pure-noise score reproduces 85% of the effect.** The null control the plan did not run.
Replace the failure score with `rng.standard_normal(500)`, keep both selection rules, keep the
real |y|, 200 draws per ticker:

| | published (real score, sensitivity) | null control (noise score) |
|---|---:|---:|
| mean `bal`, `R_v2_global` | 1.307 | 1.0263 |
| mean `bal`, `R_v2_within` | 1.255 | 0.9822 |
| **Δ** | **−0.0519** | **−0.0441** |

The rule-attributable remainder is −0.008, about 15%. And in the null control the within rule
lands *below* 1 (0.9822) while the global rule lands *above* it (1.0263), so under the
`|bal − 1|` criterion the null control gives 4/8 — the sign of the "6 of 8" in the real data
comes from `bal` sitting at 1.3, far enough above 1 that the absolute value is inert and any
downward push scores as an improvement.

**The concrete wrong answer.** `R_v2_within` is adopted "for any future pool" on the strength
of a `bal` difference that a random score largely reproduces, while its actual mechanism —
rounding one stratum up and ten down — makes the pool smaller and less representative of the
positive-move mass, not more.

**The cheapest check.** The two blocks above, both pure CPU, both under a minute, using
functions already in `failure_pool_reliability.py`:

```python
g = set(F.select_global(F.stratify_v2(sc, y), 0.10))
w = set(F.select_within_stratum(F.stratify_v2(sc, y), y, 0.10))
len(g ^ w), len(g), len(w)                      # → 0..5, 50, 45..51
# then repeat the whole comparison with sc = rng.standard_normal(y.size)
```

**What the design should be.** If the claim is a *design* property, test it as one: report the
pool-vs-population **stratum share vector** (11 numbers) rather than a scalar ratio, and fix
the allocation so it is exact — largest-remainder allocation of exactly `round(q·N)` members
across strata instead of independent `round(q·n_s)` per stratum. Then `bal` becomes a check on
the residual *within-stratum* leak, which is the only part the rule can influence, and F6's
decomposition shows that part moves by 0.9% and not by the 4% the headline implies.

**Severity: BLOCKING** for the adoption decision, which is live and untracked (F12).

---

## F3 — BLOCKING. The 28 twice-generated contexts are one trading day; the rate is the wrong check

**The claim.** `results/RESULTS_20260905.md` addendum 7:

> "every emitted pool records how many of its members are among the 28 — in the sensitivity
> analysis, **10 / 14 / 11 across the three arms, which is at the population rate rather than
> concentrated**."

And `results/PREREG_context_holdout_20260905.md` Part 2 frames the worry as RNG-stream position,
carried from addendum 5 §A: "28 of 500 contexts (5.6%) carry rollouts from a different position
in the RNG stream".

**The rate is arithmetically right and answers a different question.** `HELD` holds 250 of the
500 contexts, of which 14 are among the 28; a pool of `0.10 × 250 = 25` per ticker over 8
tickers is 200 members, so the expected count is `200 × 14/250 = 11.2` against the observed
10 / 14 / 11. So selection is not *attracted* to them. That is all a count can establish. It
cannot establish that they are **exchangeable** with the other 472 — and they are not.

**What they actually are.** `rank_indices` is sorted ascending (verified: strictly monotone,
500 entries), the pad wraps to the start, so the duplicated set is the 28 **lowest dataset
indices**. R2-F14 called this "a contiguous early block spanning 4.6–7.9% of the index range".
It is sharper than that. Using the `data_real` filenames, which carry the session date and are
written by the real arm rather than by the generation loop:

```
ticker   sessions in the 500   sessions spanned by the 28        of that session's contexts
AMD              20            ['2026-01-02']                    28 of 35
AMZN             20            ['2026-01-02']                    28 of 29
GOOG             20            ['2026-01-02', '2026-01-05']      24 of 24,  4 of 31
INTC             20            ['2026-01-02', '2026-01-05']      22 of 22,  6 of  9
JPM              20            ['2026-01-02', '2026-01-05']      22 of 22,  6 of 20
META             20            ['2026-01-02', '2026-01-05']       1 of  1, 27 of 32
MSFT             20            ['2026-01-02', '2026-01-05']      21 of 21,  7 of 15
NFLX             20            ['2026-01-02', '2026-01-05']      20 of 20,  8 of 21
```

**The 28 are one or two calendar days out of twenty, and for AMD and AMZN they are a single
session taken essentially whole.** Eleven of them in a pool is one day repeated eleven times,
not eleven independent draws — so the "population rate" reading treats a cluster as a sample.

**They are not exchangeable on the covariate `bal` is built from.** Ratio of mean |y| for the
28 against the other 472, horizon 50:

```
AMD 0.916  AMZN 1.008  GOOG 0.638  INTC 0.969  JPM 0.827  META 0.508  MSFT 0.922  NFLX 0.694
geometric mean 0.791   log-ratio t = −2.75 (df 7)   95% CI on the ratio [0.647, 0.968]   7/8 below 1
```

And on the stratified score itself, rank-sum `z` against the other 472: AMD +0.74, AMZN −1.59,
GOOG −1.11, INTC 0.00, **JPM +3.17**, META −1.62, MSFT +0.61, NFLX +0.25. JPM survives a
Bonferroni threshold of |z| ≈ 2.73 over eight tickers.

**What the concentration would have to look like to matter, and it does.** The decision-relevant
quantity is leverage on the reported statistic, not membership count. One member of the 28 in a
50-member pool sitting at `bal ≈ 1.30` moves `bal` by `(0.791 − 1.30)/50 = −0.0102`. Eleven of
them move it by **−0.112** — more than twice the −0.0519 between-rule difference that addendum 6
declares ESTABLISHED. That is visible in the published table itself: every `bal` falls when the
28 are included (1.421 → 1.353, 1.324 → 1.307, 1.294 → 1.255). The primary-versus-sensitivity
contrast is therefore not an inclusion nuisance; it is a **day-in / day-out** contrast, and it
moves the headline quantity by more than the effect being measured.

**The mechanism is in the recipe, not in the archive.** `gen_driver.py:1698` chunks
`flat[i:i+batch_size]`; with `n_contexts % batch_size ≠ 0` the final partial batch is padded by
wrapping to the start of a **chronologically sorted** list. So *every* future generation run
with a non-divisible context count will double-generate the earliest session of its evaluation
window, and the surviving files will be its second pass. Recording the ids and applying "one
rule to every arm" does not remove this; it documents it.

**The cheapest check** (about 20 s, one `os.listdir` per ticker on a 1,500-entry directory):

```python
pat = re.compile(rf"{tk}_(\d{{4}}-\d{{2}}-\d{{2}})_message_real_id_(\d+)\.csv")
date_of = {int(m.group(2)): m.group(1) for n in os.listdir(f"{member}/data_real")
           if (m := pat.match(n))}
Counter(date_of[i] for i in twice_generated_ids[tk])      # → one or two dates
```

**The fix, and it is upstream of S6.** Pad by repeating the **last** index rather than wrapping
to the first, or truncate the context list to a multiple of `batch_size`, or shuffle before
chunking so the duplicate set is not a calendar block. Any of the three makes the duplicated
set exchangeable; none is more than a line. Whichever is chosen has to be recorded in the P1
manifest, because it changes which contexts an archive contains twice.

**Severity: BLOCKING** for the pre-registered handling of the 28 (the primary/sensitivity split
does not do what it says), **MAJOR** for the generation recipe S6 will use.

---

## F4 — MAJOR. X4's "descended from" rests on a difference of two scalar norms, and the reference run's own norm is not monotone

**The claim.** `results/RESULTS_20260905.md` addendum 5 §C specifies the measurement:

> "**Compare `wm_ft_multi3/69378` against `checkpoints_selftrain/.../69378` at the level of
> parameter values, not of file layout.** Restore both on CPU with
> `code/probe_weights_offline.py` … and **compare the flattened non-embedding parameter
> vector**."
>
> outcome table: "**identical to machine precision** ⇒ `wm_ft_multi3/69378` **is** the selftrain
> checkpoint".

The two roots the quotation abbreviates, in full, from `results/x4/*.json`:

```
/lus/lfs1aip2/projects/public/u6gb/sigma-0/checkpoints_selftrain/j5705912_b30675li_5705912
/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z/ckpt/wm_ft_multi3[_step150|_step1500|_step4800]
```

and §E reports the result:

> "**The weights differ, and the distance grows monotonically with fine-tuning steps.** … So
> `wm_ft_multi3` is a **fine-tune descended from** the selftrain checkpoint at step 69378".

`PLAN.md` §0.11 and board §4 record X4 as "closed by measurement".

**Why it fails.** Four points, in order of how much each moves the conclusion.

**(a) The pre-registered branch was never evaluable with the named tool.**
`/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/code/probe_weights_offline.py`
has no pairwise mode: `probe()` returns per-checkpoint scalars (`l2_total`, `l2_non_embed`,
`mean_abs_*`, `spectral_norm_*`) and `main()` prints early→late *ratios* of those scalars. It
never holds two parameter trees at once. So "identical to machine precision" of a flattened
parameter vector could not have been reached by the tool §C names, and the branch that would
have concluded "is the selftrain checkpoint" was unreachable in principle.

**(b) What was reported is not a distance.** The `|delta| vs reference` column is
`|‖θ_A‖ − ‖θ_B‖|`, not `‖θ_A − θ_B‖`. The former lower-bounds the latter and is zero for
arbitrarily distant vectors. It is sound in one direction only: a non-zero value proves the
weights differ, so "the weights differ" holds. It carries no information about how far.

**(c) The monotone ordering is three rungs of a quantity that is non-monotone in training on
the reference run itself.** From `results/a1_step275.json`, `a1_step33575.json`,
`a1_step69378.json` — already on disk, produced by the same script:

```
selftrain chain      l2_non_embed        Δ from previous
  step   275           427.186850
  step 33575           423.494134           −3.693        the norm goes DOWN
  step 69378           425.092646           +1.599        then UP

wm_ft_multi3 ladder  l2_non_embed        |Δ| vs 69378
  base (0 ft steps)    425.092646           —
  _step150             425.353723           0.261
  _step1500            425.438179           0.346
  _step4800            425.617485           0.525
```

So on the run this argument uses as its anchor, `l2_non_embed` is **not** monotone in training
steps. Monotonicity is therefore not a property of "more training" for this quantity, and its
appearance across three fine-tune rungs is not evidence of a training trajectory from that
base. Three points in the observed order have probability 1/6 under a permutation null — that
is not a finding.

**(d) The scale rules the measure out as a lineage test.** The ladder's entire spread is 0.525;
the reference run's own excursion across the three probed ages is 3.693, **seven times larger**.
Concretely, `wm_ft_multi3_step150` sits 0.261 from selftrain step 69378, while selftrain step
33575 sits 1.599 from it — the measure ranks a 150-step fine-tune as six times *closer* to
step 69378 than that run's own earlier checkpoint. It is not behaving like a distance.

**What else produces the same ordering.** Any fine-tune of any sigma-0 78.5M checkpoint whose
non-embedding norm lands near 425 — which, from (c), is every selftrain checkpoint from
somewhere before step 33575 onwards; two independent runs of the same recipe on the same data;
or a fine-tune of a sibling run entirely. The genuine evidence for a shared **ancestor** is the
byte-identical `_ROOT_METADATA` (md5 `028879b3aa96f727b3fb94a6f894d072`) that R1 found, and
addendum 5 §E itself says metadata identifies an ancestor, not a checkpoint. The L2 column adds
nothing to that, and the "So" in "So `wm_ft_multi3` is a fine-tune descended from …" is carrying
weight the measurement cannot support.

**The concrete wrong answer.** `PLAN.md` §0.11 withdraws §0.6's "those are the same
regeneration" argument and records "Thread B's rollouts were generated from the fine-tuned model
at ft-step 4800, which is on the selftrain chain's lineage. So Thread A does not need a
regeneration to be on the same lineage as Thread B." If the ladder descends from a different
base, that is false and the withdrawal is wrong. Addendum 11 has since moved M6 off the ladder
for an unrelated reason (nothing about the fine-tuning is recorded), so the live exposure is
the lineage claim, not the probe — but the lineage claim is what licenses reusing the existing
rollouts.

**The cheapest check.** Add a pairwise mode to `probe_weights_offline.py` — two CPU restores,
already demonstrated to work under `JAX_PLATFORMS=cpu`, no GPU:

```python
a = flatten(load_params(root_ft / "69378")); b = flatten(load_params(root_st / "69378"))
# per tensor, matched by name:  ||a - b|| / ||b||   and  cos(a, b)
# calibrate the scale with the pairs already on disk: (275, 33575), (33575, 69378)
```

If `wm_ft_multi3_step150` is a 150-step fine-tune of step 69378, its relative displacement will
be orders of magnitude below the selftrain 33575→69378 displacement. If it is comparable, the
base is not step 69378. That is a two-way test; the norm difference is not.

**Related, MINOR.** §E says "The unsuffixed `wm_ft_multi3` is **statistically
indistinguishable** from `_step4800`". There is no statistic and no sampling distribution here —
the two JSON rows agree to all fifteen printed digits on eight independent summaries
(`l2_total`, `l2_non_embed`, both `mean_abs`, three `spectral_norm_*`, and the entire
`spectral_top5` dict). That is a far stronger statement than "indistinguishable", and the
statistical vocabulary weakens it.

---

## F5 — MAJOR. The `num_errors` join compares two outputs of the same writer

**The claim.** `results/RESULTS_20260905.md` addendum 5 §A:

> "**Verified by recomputing the definition from the generated books rather than by reading the
> batching code** … `slots 28..499 == recomputed for rank_indices[28..499]` matched …
> **The mapping.** `num_errors[i] ↔ rank_indices[i]` for `28 ≤ i < 500`".

**The verification has the power its author claimed — against permutations.** I measured the
identifiability directly, AMD seed 97701: 528 slots, **123 distinct values** in the range
75–249, maximum multiplicity 12, and the probability that two random slots share a value is
**0.0098**. A wrong permutation surviving 472 positions has probability on the order of
`0.0098^472`. So `log slot ↔ CSV filename` is settled beyond any doubt, and I am not disputing it.

**What it does not establish.** Both sides of the comparison are outputs of the **same writer**:
the logged `num_errors` array and the `data_gen/..._id_<cid>_gen_id_0.csv` filenames are emitted
by one process from one index array. R2's clean section established that the code which produced
that padding **is not in any recoverable tree** — `to fill last batch` appears nowhere under
`/lus/lfs1aip2/projects/public/u6gb/sigma-0/src` or either worktree. So any error *upstream* of
that array is shared by both sides and invisible to this check.

**What a wrong-but-self-consistent mapping looks like.** The generator conditions the rollout on
window X, and names the file and logs the slot as Y. Every recomputation then matches, at every
slot, and the join is uniformly wrong. Nothing in a writer-versus-writer comparison can see it.
Two of the three conditions for that failure are already present here: the producing code is
unrecoverable, and `sample_indices_rank0.json` carries two index arrays (`all_indices` and
`rank_indices`) plus a `world_size`.

**One of those risks is closed and should be recorded as closed.** I read the member's
`sample_indices_rank0.json`: `world_size = 1`, `rank = 0`, and `all_indices == rank_indices`
element for element. So there is no rank-partition ambiguity in this archive. That is worth
stating in the plan, because it is the only reason the two arrays cannot diverge here.

**The external anchor exists, is free, and was not used.** `data_real/` is written by the real
arm, not by the generation loop, and its filenames carry the **session date**:
`AMD_2026-01-02_message_real_id_1045.csv`. The date is an independent function of the context
id, so if ids index a chronologically ordered dataset the dates must be non-decreasing in id.
I ran it for AMD seed 97701:

```
data_real files 1500, message_real parsed 500
rank_indices with no data_real file:            0
date inversions along ascending rank_indices:   0
first 2026-01-02   last 2026-01-30   distinct sessions 20
```

**Zero inversions in 500.** So the `id ↔ market window` mapping is anchored, by an artefact a
different writer produced. That is the evidence that should carry addendum 5 §A's claim; the
recomputation carries only the ordering. A stronger version, still CPU and about a minute:
recompute the realised move at horizon 50 from `data_real/..._id_<cid>.csv` and compare against
`real[:, 2]` in `.returns_multih_real.npz` for that id. `real` is byte-identical across seeds
(F13) and is the denominator of every stratification, `bal` and leak figure in this plan, so
anchoring against it closes the loop that matters.

**Severity: MAJOR.** The verdict is probably right; the argument as written does not establish
it, and the plan's own P1 requirement exists precisely because "the producing code is not
recoverable" is a live condition here.

---

## F6 — MAJOR. `bal` multiplies two channels that move in opposite directions

**The claim.** `results/PREREG_selection_rules_20260905.md`:

> "`bal` — the pool's mean |realised move| divided by the population's. **A rule that has
> removed the realised-move confound has `bal ≈ 1`**".

and addendum 6 reads a `bal` improvement as evidence that "the pool's stratum composition equals
the population's by construction".

**Why the estimand is wrong for the question.** With a zero-move atom of 19.4–41.6% of the sample,
`bal` factors exactly:

```
bal  =  comp × cond
comp =  (1 − z_pool) / (1 − z_pop)              the composition channel: does the pool take the population's share of zero-move contexts?
cond =  E[|y| | pool, y>0] / E[|y| | pop, y>0]  the conditional channel: given a non-zero move, does the pool prefer bigger ones?
```

The confound the plan cares about lives in `cond`; the *design* property addendum 6 claims lives
in `comp`. `bal` is their product, so a gain in one can be manufactured by a loss in the other.
That is what happened. Measured, `q = 0.10`, `H = 50`, selection seeds 97701–97705, means over
eight tickers (these reproduce the published sensitivity `bal` row exactly: 1.3529 / 1.3069 /
1.2550):

| rule | `bal` | `comp` | `\|comp − 1\|` | `cond` | mean `z_pool` (pop 0.3055) |
|---|---:|---:|---:|---:|---:|
| `R_v1_global` | 1.3529 | 0.9982 | 0.0018 | 1.3618 | 0.307 |
| `R_v2_global` | 1.3069 | 1.0155 | **0.0155** | 1.2871 | 0.295 |
| `R_v2_within` | 1.2550 | 0.9844 | **0.0156** | 1.2759 | 0.316 |

**`|comp − 1|` is unchanged: 0.0155 → 0.0156.** The within rule does not improve the pool's
zero-move composition at all — it crosses 1 and overshoots by the same margin. The conditional
channel, which is the part that measures the actual confound, moves from a 28.7% excess to a
27.6% excess: a 0.9% relative improvement, not the 4% that reading `bal` implies. The `bal`
gain is the two channels cancelling.

So "the pool's stratum composition equals the population's **by construction**" is not true of
the implementation, and `bal` is structurally unable to reveal that, because `bal = 1` is
reachable by cancellation: a pool with no zero-move contexts and systematically small non-zero
moves scores the same `bal` as a perfectly representative pool.

**Cheapest check.** The table above; it needs only `real[:, 2]` and the three pool index sets,
all of which `failure_pool_reliability.py` already produces. Under a minute.

**What to report instead.** The pool-vs-population **stratum share vector** (11 numbers under
`stratify_v2`, of which the zero share is the first) and `cond` separately. Both are already
computed inside `select_within_stratum`; neither needs new data. Report `bal` too if it is
wanted as a one-line summary, but never as the criterion, and never with `|bal − 1|` when `bal`
sits at 1.3 — at that distance the absolute value is inert and any downward push scores as a win
regardless of direction (see F2's null control, where the within rule lands at 0.98 and the
criterion flips to 4/8).

---

## F7 — MAJOR. Seven surviving instances of the move addendum 8 withdrew

Addendum 8 correctly withdrew one instance and stated the principle: "showing two things are
close enough to substitute requires a tolerance fixed in advance and an interval that fits
inside it, not a wide interval that happens to straddle zero." The same move is live in seven
other places. Listed with file and line, most consequential first.

| # | file : line | text | why it is the same move |
|---|---|---|---|
| 1 | `/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/plan_drafts/04_training_design.md` : 463 | "**Indistinguishable** ⇒ the pool is dominated by **underlearning**; 'regime shift' is not supported at this level of training; **replay is free**; §5.3 becomes a refinement rather than a load-bearing step." | M3's **pre-registered reading rule**. A failure to detect an era difference between 4,000 contexts from 2022-06 and 4,000 from 2024-08 is read as a positive finding that cancels §5.3 and licenses unlimited replay. No tolerance, no interval, and the comparison's power is never stated |
| 2 | `/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/plan_drafts/01_measurements.md` : 211 | "**Decision rule** as drafted: indistinguishable ⇒ underlearning dominates and replay is free" | `01` adopts `04`'s rule unchanged, so the defect is in the merged plan twice |
| 3 | `/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/plan_drafts/04_training_design.md` : 492 | "`Δ ≈ 0` → **underlearned**: the model fails on this *kind* of context regardless of era." | the `Δ ≫ 0` branch is properly defined ("bootstrap 95% CI entirely above zero"); the `Δ ≈ 0` branch is defined as its complement, so everything that fails to reject becomes a positive classification, per stratum |
| 4 | `/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/results/RESULTS_20260905.md` : 868 | "**The two rules concentrate the same amount of score.**" | \|mean\|/sd of 0.00–0.28 read as sameness. Not struck by addendum 8 |
| 5 | `/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/results/RESULTS_20260905.md` : 884 | "Adopted … **because** it matches stratum composition by construction **and the score contrast is indistinguishable between the two**" | the adoption decision rests on the equivalence, so this is the one instance with an operational consequence |
| 6 | `/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/results/RESULTS_20260905.md` : 893–894 | "against a spread of 0.07–0.08, i.e. **indistinguishable** … the collapsed binning **buys no measurable extra score concentration** … with the score half of it turning out to be **nothing** rather than a cost worth paying" | "turning out to be nothing" is the strong form. \|mean\|/sd is 0.12 (primary) and 0.20 (sensitivity) |
| 7 | `/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/plan_20260904/drafts/D1_failure_pool.md` : 439 | "the §8 proxy test returns pool overlap ≥ 0.80 **and an indistinguishable excess** ⇒ **the rollout system is unnecessary**" | a decision to abandon the whole rollout apparatus, taken on a non-detection |

**A separate defect of the same family: addendum 7's withdrawal was not applied at the point of
claim.** `results/RESULTS_20260905.md` lines 913–926 still carry the heading `## SUPPORTED`, the
bolded "**The stratum edges transfer to contexts the selection never saw.**", and "a pool built
with borrowed edges carries **no measurable cost**". Addendum 8 withdraws all three eighty lines
later. `PLAN.md` uses ~~strikethrough~~ in place for exactly this reason, and its own header says
"~~strikethrough~~ marks what measurement has overturned". A reader who stops at addendum 7 —
which is what "eight addenda, newest last" invites — reads a withdrawn conclusion as supported.

**Cheapest check.** `grep -n "indistinguishable\|no measurable\|the same amount\|is not supported"`
across the task tree; that is how I found all seven.

**The fix for the M3 rule, which is the only one that is not yet a sunk claim.** State the
tolerance first: what size of era difference would change the replay mix? Then report the
interval against it. If no tolerance can be named, the gate cannot be passed by a null result
and should be written so that "indistinguishable" leaves §5.3 load-bearing rather than
cancelling it — the conservative branch, which is the opposite of what is drafted.

---

## F8 — MAJOR. The analysis code behind addenda 6, 7 and 9 was never committed

**The claim.** Three analyses are reported as having been run against a pre-registration
committed first: `5d531af2` → `ca8f1047` (selection rules), `4611383a` → `e116e5eb` (context
holdout), and addendum 5 §D → `0a659850` (dilution).

**What the commits contain.**

```
ca8f1047  code/failure_pool_reliability.py  +41   results/RESULTS_20260905.md  +73
          results/selection_rule_comparison_20260905.{json,txt}          (results only)
e116e5eb  code/failure_pool_reliability.py        results/RESULTS_20260905.md
          results/context_holdout_20260905.{json,txt}, noise_floor_20260905.txt,
          twice_generated_contexts.json                                 (results only)
0a659850  PLAN.md  +95   results/RESULTS_20260905.md  +140
          code/attach_adaptation.sh, code/submit_adaptation_pair.sh     (NO result artefact at all)
```

No driver script exists in the repository for any of the three. `grep -rn
"selection_rule_comparison\|context_holdout_20260905\|twice_generated_contexts" --include=*.py`
over the task tree returns one hit, in the notebook builder, which *reads* the JSON. **Addendum
9 produced no machine-readable artefact whatsoever**: its five contrast values, three CIs and
the ±0.037 bound exist only as prose in a markdown file, and the addendum states the reason —
"Written to an existing file rather than a new one: the project is at its inode hard cap
(51,200,000 / 51,200,000)".

**Why it matters causally.** A pre-registration's entire value is that the analysis was fixed
before the data were seen. With no recorded analysis, "we ran what we pre-registered" cannot be
checked — and I found two places where the run departs from the pre-registered text:

- the pre-registration says all three rules build "a pool of the **same size**"; measured sizes
  are 50 / 50 / 45–51 (F2);
- addendum 5 §D's step 2 defines the mixture's impurities as "random ones", and addendum 9 then
  tests whether they are worse than a random draw — the departure is that the pre-registration
  never defined a population in which the alternative could hold (F1).

This is the `code_commit` gap that `PLAN.md` §0.12 lists as one of P1's seven unrecoverable
fields, now applied to the plan's own headline results rather than to a historical archive.

**The stated blocker is stale, and the plan's own rule says so.** `PLAN.md` §7 says "the project
is at its inode hard cap (**51,200,000 / 51,200,000**) … New files cannot be created on either."
Live reading, 2026-09-06T00:05Z:

```
/lus/lfs1aip2   125.1T used / 200T    files 49,030,332 / 51,200,000     → 2,169,668 free inodes
```

`PLAN.md` §5 already carries the rule this violates: "Inode ceiling: a plan may use at most half
the free inodes read at run time, **never a figure quoted from a document**." The blocker in §7
is a figure quoted from a document, and it is the reason addendum 9 shipped without an artefact.

**Cheapest fix, consistent with the inode discipline.** Land the three drivers as subcommands of
the existing module rather than as new files — `failure_pool_reliability.py --compare-rules`,
`--context-holdout`, `--dilution` — so the code cost is zero inodes and every future reader can
re-run the pre-registered analysis from the archive.

---

## F9 — MAJOR. D3's FP32 falsification assumes its own conclusion

**The claim.** `plan_20260904/drafts/D3_profiling.md` §4.1:

> | FP32 CUDA cores | 67 TFLOPS | 169% | **Above 100%, which falsifies this peak as the binding
> one** — the work is not on the FP32 non-tensor path |
>
> "The FP32 row is not a joke entry: it is the cheapest available falsification test, and it is
> what makes the claim 'the model executes on tensor cores despite `Precision.HIGHEST`'
> **evidence rather than assertion**."

**Why it fails.** The numerator is `achieved_tflops = correction × 6 × N × D / t` with
`correction = 2.630`. `/lus/lfs1aip2/projects/public/u6gb/sigma-0/src/s5/flops.py` states the
provenance at lines 12 and 33:

```
The correction factor is derived **empirically** via NVIDIA GPM tensor-pipe
Source: GPM profiling runs on a single GH200 GPU at BSZ=1, seq_len=13000
    1024: 2.63
```

So the numerator is a count of **tensor-pipe** work, calibrated on tensor-core counters.
Dividing tensor-pipe FLOPs by an FP32 **CUDA-core** peak and concluding that the work is not on
CUDA cores restates the calibration. The same draft says so two paragraphs later — "`correction
= 2.630` … is a measured hardware-FLOP factor" — without connecting it to the falsification.

**The arithmetic, from the log rather than from the draft.**
`/lus/lfs1aip2/projects/public/u6gb/sigma-0/logs_lobs5/training_5705912_node0.log`:

```
[FLOPs] Correction factor: 2.630        [FLOPs] TFLOPS/step: 64.45
recent 0.565 s/step, MFU 11.5% (114.1 TFLOPS)

with correction:     114.1 / 67  = 170%     "above 100%, so not the FP32 path"
without correction:  (64.45/2.630) / 0.5687 = 43.1 TFLOPS;  43.1 / 67 = 64.3%   below 100%
```

**The falsification survives only because the correction is in the numerator.** Remove the one
factor that was measured on tensor cores and the FP32 peak is no longer ruled out.

**Cheapest check.** The three lines above, from a log already on disk. **What would be
evidence**: read the executed kernel names and dtypes once (`XLA_FLAGS=--xla_dump_to=` on the
P5 cell, or one `nsys` window), which reports what actually ran rather than inferring it from a
ratio whose numerator presupposes the answer. D3 §10 already recommends the honest fallback —
"publish `achieved_tflops`, which has no chosen denominator, alongside every MFU percentage".

---

## F10 — MINOR. D3's P5 decision rule maps a timing null onto a dtype question

`plan_20260904/drafts/D3_profiling.md` §4.1: "if it moves it by under a percent, the
contractions are not the bottleneck and **TF32 is the divisor to standardise on**."

Two defects. First, a null timing result establishes only that the SSD contractions are not
rate-limiting; it says nothing about which arithmetic peak the executed work runs against — the
step could be bound by the scan, memory bandwidth or the loader while the contractions still
execute in fp32. Second, the divisor question is about the `nn.Dense` **projections**, which
D3 itself says take XLA's default precision; P5 varies `MAMBA3_CONTRACTION_PRECISION`, which
touches a different set of tensors. The measurement and the claim are about different objects.
It is also F7's shape a third time: no change detected read as a positive conclusion.

**Fix**: decide the divisor from the executed kernels, not from a step-time delta, and keep P5
for what it does measure — the contractions' share of the step, and whether TF32 is a free speed
knob (which, as D3 correctly notes, is a separate decision because the two cells differ in
numerics as well as speed).

---

## F11 — MINOR. D4's step 6 contradicts the accepted position and is now a no-op

`plan_20260904/drafts/D4_deletion_refactor.md` §2.1 reports "18 unstaged deletions" and §6 step 6
prescribes "Commit the 18 `bench2k_*` deletions … Risk: none — 18/18 verified in tar".

`results/RESULTS_20260905.md` addendum 2 §A and addendum 4 §E record the opposite decision:
"**They are not re-deleted** — they belong to another line, deletion is forbidden here, and
re-deleting would be a second uninvited change." Measured now, at the repository root:

```
git ls-files --deleted -- 'tasks/hybrid_system_20260811/05_hybrid_mamba3_lob/bench2k_*'   →  0
git status --short | grep -c '^ D'                                                        →  0
```

The rebase restored all 18 and they are byte-identical to HEAD, so D4's command would produce an
empty commit against a stale premise. D4 was written on 2026-09-04, before the incident, so this
is not an error of the draft — but the merged plan must drop step 6 rather than inherit it. The
same applies to D4 §6's ordering, which places step 6 before the refactor steps.

---

## F12 — MINOR. The `R_v2_within` adoption is live and untracked

`results/RESULTS_20260905.md` addendum 6 states "**Adopted, on the balance ground alone**:
`R_v2_within` for any future pool." `PLAN.md` board §2 has rows C1–C7 and none of them is the
selection rule; §3 has S1–S8 and none of them is this decision. So a decision that changes how
every future pool is built lives only in the results file, and F2 argues it should be reversed.
Add it as a claim row with its state, or withdraw it in the same file that made it.

---

## F13 — MINOR. D3's weak-scaling row straddles the node boundary

`plan_20260904/drafts/D3_profiling.md` §3.1 note 3: "**Weak-scaling loss is 8% out to 8 GPUs**
(91,437 → 84,437 tok/s/GPU)". Job 5705913 is 4 GPUs on one node; job 5705914 is 8 GPUs on two.
D3's own §2.3 records that crossing that boundary changes the code path (`args.num_devices =
jax.local_device_count()` overrides the flag, and the hierarchical mesh engages), so the 8%
mixes intra-node scaling with the first inter-node hop rather than tracing one curve. The draft
does have the right control for the *other* worry — two independent 8-GPU jobs agree to 1.0%
(0.6158 vs 0.6217 s/step), which bounds job-to-job variation well below the 8% — so the fix is
one sentence naming what the 8% is: the cost of going off-node, measured once.

---

## What I checked and found clean

**D3's throughput table reproduces exactly, from the logs.** I re-extracted `recent N s/step`
from four per-node logs under `/lus/lfs1aip2/projects/public/u6gb/sigma-0/logs_lobs5/`, dropped
the first 5 samples as the draft specifies, and got the draft's numbers to four decimals:

```
5705912  n=688  mean 0.5687  sd 0.0044  cv 0.77%      5705913  n=679  0.5762  0.0053  0.92%
5705914  n=364  mean 0.6158  sd 0.0104  cv 1.69%      5877859  n=315  0.4508  0.0091  2.02%
```

`[FLOPs] Params: 78,539,423`, `Tokens/step: 52,000`, `Correction factor: 2.630`,
`TFLOPS/step: 64.45`, `Peak BF16 (1 GPUs): 989 TFLOPS`, and 693 `[Timing]` lines — all as
stated. The derived constants check out: `1e9 / (3600 × 52,000/0.5687) = 3.038` GPU-hours per
billion tokens; the corpus epoch at `30.06e9 × 26 = 781.6e9` tokens agrees to three digits with
the independent tqdm route (`939,147 × 16 × 4 × 13,000 = 7.81e11`); the `tokens_per_sec` error
factor `2.630 × 78,539,423 = 2.066e8`; `989/494.5 = 2.0` giving 22.9%; `989/67 = 14.76` giving
169%; §7.2's overhead sums (0.03 + 0.8 + 2.4 = 3.23%, and 2.43% without readiness); §9.3's
"13 to 26 times smaller" (`1e9/78e6 = 12.8`, `2e9/78e6 = 25.6`). **D3's arithmetic is sound
throughout; my findings against it are about what the numbers are used to conclude.** Its
§10 self-audit table is also the best instance of the practice in this corpus.

**D4's two largest counts still hold on today's repository.** `git count-objects -vH` reports
`size-garbage: 36.47 GiB` with `garbage: 1`, so the `tmp_pack_OoiRqg` is still there and the
proposed `mv` has not been done. `git ls-files` under the CONTAMINATED directory returns exactly
**31,368** files, matching the draft to the unit; the tracked total has drifted from 32,529 to
32,615, so the share is 96.2% rather than 96.4% — drift, not error. D4's internal arithmetic is
also consistent (buckets sum to 75; 1,212 − 672 = 540 lines; 24 × 2,240 ≈ 54,000 dirents).

**The `num_errors` join has the power its author claimed against permutations.** 528 slots, 123
distinct values, maximum multiplicity 12, pairwise collision probability 0.0098 — a wrong
permutation cannot survive 472 positions. F5 attacks a different failure mode, not this one.

**The context id maps to a real market window, externally.** `data_real` filename dates are
non-decreasing along ascending `rank_indices` with **zero inversions in 500** for AMD, spanning
20 sessions from 2026-01-02 to 2026-01-30, and every one of the 500 has a `data_real` message
file. This was not part of addendum 5's argument; it should be.

**`world_size = 1` and `all_indices == rank_indices`** in `sample_indices_rank0.json`, so the
rank-partition ambiguity that would otherwise sit under F5 does not arise in this archive.

**The published `bal` figures reproduce from the module.** My independent recomputation of the
sensitivity row from `failure_pool_reliability.py` gives 1.3529 / 1.3069 / 1.2550 against the
published 1.353 / 1.307 / 1.255, and the pool sizes 50 / 50 / 47–51 — the agreement is what let
me localise F2 to the rounding rule rather than to a data difference.

**Addendum 8's own correction is right and I could not weaken it.** The withdrawal of the
transfer-cost equivalence, the statement of the resampling target (ticker, df = 7), and the
refusal to pick a tolerance after the fact are all correct, and the `|bal − 1|` comparison it
promotes — the literal issue-#73 rule against the stratified one, +1.661 CI [+1.171, +2.151],
8/8 — is a difference rather than a failure to find one. `PLAN.md` C4's "NOT established" label
is the right label.

**Addendum 9's construction note is correct and correctly flagged.** "The `p = 0` arm sits at
−0.09 rather than 0 because the non-reference draws come from the complement of the top decile.
That is the design, not a result." That sentence is exactly right — and it is also the sentence
that, read one step further, yields F1.

**Addendum 11's reasoning is sound and I have no objection to it.** That every `wm_ft_multi3`
rung carries the base's `_ROOT_METADATA` byte for byte, so its `train_date_range`, ticker list
and `jax_seed` describe the base pre-training and not the fine-tuning, is checkable and checked;
"lineage is not enough when the intervening training is unrecorded" is the right principle, and
resolving S8 against the ladder the previous addendum was leaning towards is the correct call.
The warmup derivation for 33575 (`warmup_end_step ≈ 694`, so 275 is 0.4× and 33575 is 48×) is
also correctly flagged as resting on an inferred `steps_per_epoch`, with the direction of the
inference's error stated.

**Not checked by me, and stated so the merge step knows.** I did not re-derive R2's statistical
findings, the noise-floor `φ` measurements of addendum 7, the `generation_gate.py` /
`write_run_manifest.py` behaviour, the worktree-incident reconciliation, or `plan_drafts/02`,
`04` and `05` beyond the sections named above. I did not open the published notebook. No GPU job
and no `sbatch` was run; no file outside `plan_drafts/` was modified; nothing was deleted.
