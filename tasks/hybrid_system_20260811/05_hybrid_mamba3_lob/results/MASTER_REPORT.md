# Hybrid Mamba3 × Nemotron Attention on Limit Order Books — Master Report

## TL;DR（太长不看版）

**问题**：纯递归主干（mamba3）生成订单簿消息流时，**远距离引用回指**会随距离崩溃——
要取消一张 500 条之前的订单，模型得从固定大小的隐状态里翻出它的 id。往主干里插**一层
全局注意力**（Nemotron 配方）能不能修好，代价是多少？**已判定**：在 **500 条上下文**下，
优势是**收敛速度**不是能力（32k 步时被抹平，A 级，2–3 seed 复现）；在 **2,000 条上下文**下
**结论翻转**——优势随引用距离放大，从 +3.02 pp（11–25 条前）一路到 **+24.70 pp**
（501–1000 条前），整体 +6.39 pp（A 级）。**注意措辞**：不是全程单调，最近的 1–10 档
（+3.67）**高于** 11–25 档（+3.02），凹点在三个 seed 上都复现，是结构不是噪声；准确说法是
「**从 11–25 档起单调**」（详 §4.1）。

**2026-08-15 主问题已判定（§6.5，A 级三 seed）**：步号匹配的三方比较
`base@4462 / hybpm@4479 / hyb@4555`（seed 2026/2027/2028）给出 `501–1000` 档
`hybpm − base = **+20.6157 pp**`（极差 0.3431，**24 格全部同号**，均值/极差 **30.8**），
越过预注册的 +15 pp 门槛 ⇒ **效应是机制性的**。等参数组用**比 baseline 还少 441 个**的参数
复刻了完整 hybrid 优势的 **87.7%**，所以那 +5.43% 参数至多解释 12.3%。
**尚未收敛的是份额**：步号更紧的三元组（跨度 0.81%，单 seed）给出 +13.87 pp / 26.7%，
两个点分歧，方向可引用、份额不可引用（§6.5 末段）。
**另一件三 seed 也没能判定的事**：LOB-Bench 上「等参数组更差」的相对差只有 2.0–3.1%，
**远小于 16.6% 的跨 seed 噪声底**，不构成结论（§6.5.1）。

**以下为历史记录**：那 +24.7 pp 里有多少
是注意力这个**机制**、多少是 hybrid 多出来的 **5.43% 参数**——第三条**参数等参数组**
（`d_ff` 2560→1135，参数量 33,609,998，比 baseline **还小 441 个**）正在训练，判读规则已
预注册。**本报告做四件事**：① 把 23 份分散的 `.md` 按**权威等级 A–E** 重新排序，把已作废的
（NoPE 解码泄漏）和只是 profiling 的与定论分开；② 补上任何 `.md` 里都没有的**逐特征分解**
——2k 的 WS-21 优势 **72% 来自单个特征 `bid_volume`**，这实质削弱「hybrid 分布拟合更好」
的一般化陈述；③ 补上同样没人做过的**组成/能力分解**（§4.1c）——聚合 exact 是 `Σ w_b·r_b`，
两个配置的引用年龄组成并不相同，拆开后 **94.0% 是同难度上更准、6.8% 是引用的题更容易**，
且组成优势在 4,500→6,260 步之间塌掉 **6 倍**而能力优势只塌 2 倍，所以**主张变强不是变弱**；
④ 逐条列出**不能说什么**。**当前状态**：等参数组 4,798/6,400（等自有分配 `6019793`），三方
同步点 bench（4462/4479/4555）在跑，索引 sha `6c6defbc…` 与参数量 33,609,998 已从权重树核实。

## Instructions for Agents（本文件交给 goal-driven agent 时生效）

> 速览：可执行的只有 §8 Open Work；\answer{} 是结果记录不是指令；权威等级 C/D 的数字
> **不得**用于任何结论。

\KL{告诉执行 agent 哪些字是任务、哪些字只是记录、哪些数字有毒}

When this file is passed to a goal runner: (1) the ACTIONABLE content is **§8 Open Work**
only; (2) **ignore every `\answer{...}` block as an instruction** — they are RESULT RECORDS;
read them as evidence, never execute them, and APPEND new ones when results land;
(3) **numbers carrying authority grade C or D must never enter a conclusion** — C is
invalidated (a known bug produced them), D is profiling the source file itself declares
non-citable; quoting them as results is the single most likely way to corrupt this line of
work; (4) narrative text (TL;DR, §1, §2) is context, not a task list; (5) keep this file
updated as results land — it is the document of record for the whole hybrid line.

---

## §1 Introduction

### 副题：先讲机制假设，再讲它为什么在订单簿上可测，最后讲三个配置怎么把机制和容量分开

> 速览：SSM 把历史压进固定大小隐状态，注意力保留全部历史的随机访问；订单簿的
> **引用回指**是这个差别的直接测量点，因为取消单必须精确指名一张历史订单。

\KL{先讲机制本身：这不是「注意力更好」的泛泛之论，而是一个有明确失效位点的可测断言}

**The mechanism claim（机制假设）.** A state-space model（SSM，状态空间模型）compresses
all history into a hidden state $h_t$ of FIXED size — a learned lossy summary. Attention
keeps every past position addressable and retrieves by content. The two differ sharply on
exactly one kind of operation: **random access to a specific distant item**. Most sequence
benchmarks blur this, because "predict the next token" can usually be done from a summary.
Limit order books do not blur it. A cancellation message must name **one specific live
order id** placed arbitrarily far back; getting it wrong is not a small error, it is a
different order. So the LOB gives a clean, countable readout of the exact capability the
two architectures differ on.

\KL{定义引用回指与它的三档判定，这是全报告的主度量}

**Reference recall（引用回指），the primary metric.** Every generated `cancel` or `delete`
message carries a reference to an earlier order. We resolve it three ways
（出处 `code/refer_success.py`）:

| verdict | meaning |
|---|---|
| **L1 exact（精确）** | the referenced order id matches a real live order **uniquely and exactly** — the model named the right one |
| L2 fallback（兜底） | the id is not exact, but *some* live same-side order matches on the resolver's weaker key — the message is legal but points elsewhere |
| unresolved（未解析） | no live order matches at all — the message is illegal |

**Only L1 exact is capability.** L2 is a legality mask doing the work, and a mask reaches
high numbers mechanically: every residual case has at least 5 live same-side candidates
(median 21, p90 42, max 294, 出处 `PLAN_26TOK.md`). *Example:* a stream can score 96.14 %
"any live id" while its exact rate is 74.89 % — the 21 pp gap is the mask, not the model.

\KL{定义引用年龄分档——距离是自变量，这条曲线的形状比任何单点数字有信息量}

**Reference age（引用年龄）and why the CURVE matters more than the LEVEL.** For each
resolved reference we record how many messages back the target was, and bucket it: 1–10,
11–25, 26–50, 51–100, 101–250, 251–500, 501–1000, and `before-window`（目标在条件窗口之前，
结构上不可达）. A single overall number can be moved by sample composition, training amount,
or luck. **A monotone gradient across eight buckets spanning 21 pp cannot** — it can only be
produced by a mechanism whose benefit scales with distance, which is precisely the
hypothesis. *This is the report's central epistemic move: the SHAPE of the win is stronger
evidence than the SIZE of the win.*

\KL{三个配置的关系：第三条不是补测，它是把「机制」和「容量」这两个候选解释拆开的唯一手段}

**Three arms（三个配置）.** Inserting an attention block *replaces* one mamba3 layer, but the
attention block is fatter (4,923,520 vs 3,098,536 params), so the faithful hybrid is
**+5.43 %** larger. Any win therefore has two live explanations — the mechanism, or the extra
capacity — and they are not separable from two arms. The third arm shrinks the attention
block's FFN (`d_ff` 2560 → 1135) until the total matches baseline, keeping the mechanism and
removing the capacity.

| arm | architecture | params | vs baseline | status |
|---|---|---:|---:|---|
| **baseline** | pure mamba3, L=6 | 33,610,439 | — | trained, 500 ctx @32001 and 2k @6265 |
| **hybrid faithful（完整配置）** | mamba3 + attention at trunk layer 3, `d_ff`=4·H=2560 | 35,435,423 | **+5.43 %** | trained, 500 ctx @32001 and 2k @6258 |
| **hybrid param-matched（等参数组）** | same, `d_ff`=1135 | **33,609,998** | **−0.0013 %** | 2k, training, 4,479/6,400 |

---

## §2 Background

### 副题：架构、编码、层放置规则、评测协议——每个数字标出处

> 速览：26tok 编码、d_model 640、L=6、注意力按 Nemotron 比例落在第 3 层；评测是冻结索引
> 上的 3,136 条 GOOG 2026-01 序列，四套度量（CE / IC / LOB-Bench / 回指）。

\KL{注意力层位置不是拍脑袋——Nemotron 的比例必须在序列混合层上算，不是在全部层上算}

**Layer placement（层放置）.** Nemotron's ratio must be computed over **sequence-mixing**
layers: 4/(27+4) = **12.9 %**, not 4/56 = 7.1 %. So `k = max(1, round((4/31)·L))`; at L=6
that is k=1, at trunk layer **3**. k=2 would violate the recipe's own constraints C2 and C4.
Self-check: substituting L=31 reproduces Nemotron's actual (8, 13, 17, 22) in three of four
positions（出处 `STAGE_1_BUILD.md`）.

\KL{注意力配置：NoPE 不是省事，是因为 mamba3 已经在状态里做了 RoPE，再注入正弦会是第二套位置观}

**Attention configuration.** `heads=10` ⇒ `head_dim = 640/10 = 64`, satisfying the Pallas
causal kernel's requirement (`head_dim ≤ 256` and `% 8 == 0`); anything else silently falls
back to a materialised $L\times L$ matrix, which at 52,000 tokens is not a fallback but an
OOM. `positional_encoding=False` (NoPE) follows Nemotron **and** avoids a second, unrelated
notion of position — the recurrent layers already rotate their state with RoPE.

\KL{参数量的构成，逐层给出——这张表是等参数组设计的全部依据}

**Parameter composition per trunk layer at production width**（出处 `STAGE_1_BUILD.md`）:

| trunk layer | baseline | hybrid faithful |
|---:|---:|---:|
| 0,1,2,4,5 (each) | 3,098,536 | 3,098,536 |
| **3** | 3,098,536 | **4,923,520** ← attention |
| **total** | **33,610,439** | **35,435,423** (+5.43 %) |

Attention block = q/k/v/o (each 640×640+bias = 1,640,960 for all four combined per the
audit) + 2×LayerNorm (2,560) + FFN ($1281\cdot d_{ff} + 640$; = 3,279,360 at $d_{ff}=2560$).
The `d_book=503` value was recovered by a stub scan reproducing the published baseline
bit-exactly: 501→33,600,719, 502→33,605,578, **503→33,610,439 (diff 0)**, 504→33,615,302.

\KL{评测协议：同一份冻结索引是配对比较的前提，索引 sha 必须核对——今天就因为它不同而作废了一次比较}

**Evaluation protocol.** 3,136 frozen GOOG 2026-01 sequences (3,136 = $2^6\times7^2$; note it
has **no factor 3**, so `world_size × batch_size` must divide it — a 3-GPU run cannot use the
full pool). Index sha256 must match across arms or the comparison is not paired:

| index sha256 | used by |
|---|---|
| `6c6defbcffaadae4ebec72cc4ccf7a404e7f88144088dfb5cad74fd098fab0ec` | **2k arms**: base@6265, hyb@6258 |
| `0c41de51…` | 500-ctx arms @12000 and @32001 |
| `434917a0268778c93d13fb61950e2c152dc335c20b9464cd8540118842c28d45` | **the 2026-08-14 pmatch early read (N=768)** — different set, **not paired** |

\KL{噪声底：任何显著性主张之前必须先知道尺子的抖动，否则会把噪声当效应}

**Noise floor（噪声底，出处 `BASELINE.md`）.** Critical for every significance claim below:

| source | spread |
|---|---|
| LOB-Bench harness alone | **0.6 %** |
| same checkpoint, two full-pool re-runs | **1.9 %** (0.20714 vs 0.21109) |
| across generation seeds (3 seeds) | **16.6 %** (WS 0.20949–0.24424) |
| production vs new harness protocol | ~1 % |

⇒ **a single-point WS-21 difference below about ±0.017 is meaningless**; any "X % better"
claim with X < 8 % relative needs a cross-seed variance estimate first.

\KL{为什么主标准不是 LOB-Bench——它测的是逐行边际，不是动态}

**Why LOB-Bench is not the primary criterion.** Shuffling the time order inside the
generation window makes WS-21 **13.7 % better** (0.1441 → 0.1244), with 13 of 21 features
changing by exactly ±0.0 %. LOB-Bench measures per-row marginals; it is a **safety
（非劣）gate**, not a capability metric.

---

## §3 Authority Grading（权威等级）

### 副题：本报告与「把 23 份 md 拼起来」的唯一区别就是这张表

> 速览：A 定论 / B 单点 / C 已作废 / D 只是 profiling / E 基建。**C 和 D 的数字不得进入任何结论。**

\claim{`results/` 下 23 份 md 至少 5 份已作废或自述「不是报告」；不分级的汇总会把废数据和定论混在一起，这比没有汇总更危险。}

| grade | criterion | files |
|---|---|---|
| **A 定论** | multi-seed **or** multi-step replicated, paired, same sign | `STAGE_5`（3 seed @12000）、`STAGE_8`（2 seed @32001）、`STAGE_2`（baseline 全池）、`STAGE_1_BUILD`（参数与 microbench） |
| **B 单点** | single seed / single step; direction credible, magnitude not quotable | `STAGE_3`（@12000 单 seed）、`CTX2K_FINAL`（2k 单 seed；但 §4.1 的单调性因 8 桶跨 21 pp 升为 A）、`PMATCH_ARM` 早读 |
| **C 已作废** | a known bug produced them, or superseded by a later config | `STAGE_6` **生成侧全部**（NoPE 解码泄漏，CE 部分仍有效）、`CTX2K_HYBRID_DIVERGED`、`CTX2K_EXPERIMENT` 的配置表 |
| **D 只是 profiling** | the file itself declares it is not a result | `CTX2K_LENGTH_GENERALIZATION`、`CTX2K_LENGTH_SWEEP_PROFILING`、`CTX2K_INTERIM` |
| **E 基建/复盘** | no scientific claim | `CKPT_DEADLOCK_ROOT_CAUSE`、`COMPILE_CACHE`、`CTX2K_RUN_LOG`、`NOTE_gpu_reclaim`、`CTX_4K_FEASIBILITY`、`SEQPAR_DESIGN`、`PLAN_26TOK`、`BASELINE`、`CTX2K_STATUS` |

---

## §4 Results at 2,000-Message Context — the current main line

### 副题：A/B 级；唯一改变的变量是序列长度

> 速览：从 11–25 档起六档单调放大到 +24.70 pp（A 级形状，最近一档有复现的小凹点）；
> 聚合差 **94.0% 经得起组成分解**（§4.1c）；LOB-Bench 三项全胜但收窄，且 **72% 的
> WS 优势来自单个特征**；IC 短程略好、长程无信号；代价是推理 +60%。

**Setup.** Both arms trained **from scratch to 6,400 optimizer steps**; compared at baseline
**6265** / hybrid **6258** (Δ = 7 steps = **0.1 %**). cond1000 / gen1000, seed 2026, effective
batch identical (`1/GPU × 4 GPU × 4 nodes × K5 = 80`), 4 nodes / 16 GPUs, index sha
`6c6defbc…` verified identical. **Only the sequence length changed: 500 → 2,000 messages.**

\KL{必须先破除「2k 配置训练不足」的错觉——两代 token 预算是逐位相同的}

**Token budget identity（token 预算恒等，A 级，避免误读）.** 500-ctx: effective batch 64 ×
13,000 tok = 0.8320 M tok/optimizer step × 32,000 steps = **26.624 B**. 2k: 80 × 52,000 =
4.16 M tok/step × 6,400 steps = **26.624 B**. **Bit-identical.** The `curtail=32000 (=6400×5)`
line in the launcher is the trace of this. **Do not present "6,265 steps vs 32,001 steps"
as if the 2k arms were undertrained** — they consumed the same tokens.

### §4.1 Reference recall by age — the headline

\claim{优势随引用距离放大、跨越 21 pp，且从 11–25 档起六档单调。整体差可被涨落解释，这个形状不能。}

Percentages recomputed from raw JSON to 4 dp; **both arms' n are given** because the arms
generate different message mixes and `CTX2K_FINAL.md` prints only one n column.

| reference age | n (base) | baseline | n (hyb) | hybrid | Δ (pp) |
|---|---:|---:|---:|---:|---:|
| 1–10 | 462,899 | 93.4750 % | 454,836 | **97.1451 %** | **+3.6701** |
| 11–25 | 274,073 | 93.0179 % | 286,027 | **96.0364 %** | **+3.0185** |
| 26–50 | 185,266 | 87.9633 % | 186,612 | **92.4908 %** | **+4.5276** |
| 51–100 | 149,434 | 79.2517 % | 151,099 | **86.9873 %** | **+7.7356** |
| 101–250 | 122,213 | 59.0273 % | 120,288 | **72.7388 %** | **+13.7115** |
| 251–500 | 47,139 | 28.0087 % | 45,801 | **49.5513 %** | **+21.5427** |
| **501–1000** | 17,471 | 9.2782 % | 16,211 | **33.9831 %** | **+24.7049** |
| before-window | 271,442 | 22.2530 % | 263,804 | **27.9363 %** | **+5.6833** |

Overall (cancel + delete):

| | n | exact | fallback | miss |
|---|---:|---:|---:|---:|
| baseline@6265 | 1,529,937 | 72.9699 % | 23.5465 % | 3.4835 % |
| **hybrid@6258** | 1,524,678 | **79.3593 %** | 17.5468 % | **3.0939 %** |
| Δ | | **+6.3894 pp** | −5.9997 | −0.3896 |

> **形状比水平差更有说服力。** 整体 +6.39 pp 可以被随机涨落、样本构成、训练量微差解释。
> **优势随距离放大、跨越 22 pp 只能由「有一个机制专门负责远距离回指」解释** —— 而那
> 正是往递归主干里插一层全局注意力的全部理由。**分高的方式与机制假设吻合。**

\KL{「单调」是可检验的断言，机械检验后发现要修正——凹点在三个 seed 上都复现，是结构不是噪声}

**Precision correction（2026-08-15）.** Earlier wording said the advantage "amplifies
**monotonically**". Checked mechanically it does not start monotone: the `1–10` bucket
(+3.6701) sits **above** `11–25` (+3.0185). The accurate claim is **monotone from 11–25
onward, +3.10 → +25.29 pp across six buckets**, with a small bump at the nearest bucket.
The dip reproduces in all three seeds (§4.1b), so it is structure, not noise.

#### §4.1b Three-seed replication — 24 cells, 24 same signs

\claim{分档 Δ 在三个 seed 上逐档同号，51–100 档的极差是效应的 1/480；这把 2k 主结果从 B 级升到 A 级。}

Paired Δ (hybrid − baseline) by age, three generation seeds on the **same** frozen index:

| reference age | s2026 | s2027 | s2028 | mean | spread | same sign |
|---|---:|---:|---:|---:|---:|:---:|
| 1–10 | +3.6701 | +3.8026 | +3.5844 | +3.686 | 0.218 | yes |
| 11–25 | +3.0185 | +3.1851 | +3.0952 | +3.100 | 0.167 | yes |
| 26–50 | +4.5276 | +4.7246 | +4.5854 | +4.613 | 0.197 | yes |
| **51–100** | +7.7356 | +7.7513 | +7.7375 | **+7.741** | **0.016** | yes |
| 101–250 | +13.7115 | +13.9224 | +13.4214 | +13.685 | 0.501 | yes |
| 251–500 | +21.5427 | +22.8895 | +22.1663 | +22.199 | 1.347 | yes |
| **501–1000** | +24.7049 | +25.1651 | +25.4210 | **+25.097** | 0.716 | yes |
| before-window | +5.6833 | +5.6285 | +5.7307 | +5.681 | 0.102 | yes |

Aggregate metrics across the same three seeds:

| metric | mean Δ | spread | mean/spread | grade |
|---|---:|---:|---:|---|
| **exact pp** | **+6.4567** | 0.2599 | **24.8** | **A** |
| KS-21 | −0.005163 | 0.000731 | 7.1 | A |
| L1-21 | −0.004222 | 0.002053 | 2.1 | B |
| **WS-21** | −0.022755 | 0.016053 | **1.4** | **B — spread swallows the mean** |

\KL{同一批数据里一个指标够格、另一个不够格——这正是必须逐指标报散布的理由}

**Not all four earn the same grade.** Reporting "three seeds, all same sign" without the
per-metric spread would launder WS-21 (mean/spread **1.4**) on the back of `exact` (**24.8**).
WS-21's weakness is exactly what `BASELINE.md`'s 16.6 % cross-seed calibration predicted.

### §4.1c Decomposition — 94.0 % capability, 6.8 % easier question mix（A 级，新增）

\claim{聚合 exact 是 `Σ w_b·r_b`，两个配置的年龄组成 `w_b` 并不相同；分解后 94.0% 是同难度上更准，6.8% 是引用的题更容易，且后者在训练中塌得比前者快 3 倍。}

\KL{这项检验此前从未做过——所有 md 报的都是聚合数，而聚合数对组成敏感}

An aggregate rate over buckets is `Σ_b w_b · r_b`. Nothing in the protocol forces the two arms
to emit the **same distribution of reference distances**, so part of any aggregate gap can be
the arm asking itself easier questions. Oaxaca decomposition,
`Δ = Σ w_b^base (r_b^hyb − r_b^base) + Σ (w_b^hyb − w_b^base) r_b^base + interaction`:

| term | ~4,500 steps (2 seeds) | ~6,260 steps (3 seeds) |
|---|---:|---:|
| **Δ total exact** | **+14.6342** | **+6.4566** |
| within-bucket — **same-difficulty capability** | +12.1893 (**83.3 %**) | +6.0669 (**94.0 %**) |
| composition — **easier mix** | +2.4788 (16.9 %) | +0.4423 (6.8 %) |
| interaction | −0.0340 | −0.0525 |

Per-seed composition terms at 6,260 are 0.4326 / 0.5110 / 0.3834 — **consistent, therefore
real**, merely small.

**The composition term has a single identifiable channel: the `before-window` share.** A
reference is `before-window` when the named order was never visible in the 2,000-message
context, and it is by far the hardest bucket (≈22 % exact vs ≈93 % for `1–10`). Emitting
fewer of them raises the aggregate without answering anything better:

| | base `before-window` share | hyb share | Δ |
|---|---:|---:|---:|
| ~4,500 steps | 22.0598 % | 18.7284 % | **−3.3314 pp** |
| ~6,260 steps | 17.7625 % | 17.3354 % | **−0.4271 pp** |

Baseline moves 22.06 → 17.76 between the two checkpoints; hybrid was already at 18.73 and ends
at 17.34. **Baseline catches up on composition and does not catch up on within-bucket
accuracy** — which is why the decomposition shifts from 83.3 % to 94.0 % capability as
training proceeds.

> **两个不同的能力，不该合成一个数报。** 「少引用看不见的订单」是生成流的自洽性；
> 「给定距离把它认对」是回指。前者到 6,260 步基本被追平，后者没有。拆开之后主张
> **变强**：最终那 +6.07 pp 是纯粹的同难度比较，不含挑题的便宜。
> 这也给 §4.2 的跨行不可减性提供了机制解释——组成差在两个上下文长度下量级完全不同。

出处：三 seed 的 `refer_success_{base2k_6265_s2026, base_6265_s2027, base_6265_s2028}` 与
`{hyb2k_6258_s2026, hyb_6258_s2027, hyb_6258_s2028}` 的 `by_age`；匹配步号档用
`base_4462_s{2026,2027}` 与 `hyb_4555_s{2026,2027}`（s2028 的 hyb 尚在跑）。

### §4.2 The sign flip against 500 context — B 级，但符号本身不受口径影响

| | baseline | hybrid | winner |
|---|---:|---:|:---:|
| L1 exact（500 ctx, step 32001） | **74.8939 %** | 74.4415 % | baseline（−0.60 %） |
| exact（2k ctx, step 6265/6258） | 72.9699 % | **79.3593 %** | **hybrid（+6.39 pp）** |

At 500 ctx hybrid **loses** this metric, and the then-current explanation was "the reference
advantage is convergence speed, not capability"（见 §5）. **2k shows that explanation's scope
is short windows only.** Cross-row absolute values are NOT subtractable (generation length
250 → 1000 changes the age composition entirely), **but the sign flip is invariant to that.**

\KL{这条顺带削弱了参数量混淆——但只是间接的，所以才要等参数组}

**And it partially answers the capacity confound（部分地）.** Parameter count does not depend
on sequence length, so the same +5.43 % existed at 500 ctx, where it produced a **loss**.
Capacity cannot explain a sign flip with length. This is a strong **indirect** rebuttal; the
param-matched arm (§6) makes it direct.

### §4.3 LOB-Bench distribution distances — and the concentration that no `.md` states

| | baseline@6265 | hybrid@6258 | Δ rel. |
|---|---:|---:|---:|
| **Wasserstein-21** | 0.2919225832891291 | **0.27610219721358664** | **−5.4 %** |
| KS-21 | 0.14786763803866776 | **0.14226326551499877** | −3.8 % |
| L1-21 | 0.2222398163556179 | **0.21842378859663122** | −1.7 % |

Relative advantage, 500 vs 2k（只比列内相对量）:

| | 500 ctx | 2k ctx | |
|---|---:|---:|---|
| WS-21 | −11.0 % | −5.4 % | **收窄** |
| KS-21 | −3.1 % | −3.8 % | 略增 |
| L1-21 | −9.2 % | −1.7 % | **收窄** |

\claim{2k 的 WS-21 优势 72% 来自单一特征 `bid_volume`，这使「hybrid 分布拟合更好」不能作为一般化陈述。}

\KL{这一节是本报告相对于所有既有 md 的净增量——它改变了一个已发表结论的强度}

**Per-feature decomposition（逐特征分解，从 `summary.json` 抽出，任何 `.md` 里都没有）：**

| feature | WS base | WS hyb | Δ |
|---|---:|---:|---:|
| **bid_volume** | 0.352191 | **0.113593** | **−0.238598** |
| **log_time_to_cancel** | 0.369460 | **0.300812** | −0.068648 |
| ask_volume | 0.355698 | 0.293274 | −0.062424 |
| bid_volume_touch | 0.137793 | 0.123682 | −0.014111 |
| ofi_stay | 0.392499 | 0.384198 | −0.008301 |
| ofi | 0.389471 | 0.381763 | −0.007708 |
| spread | 0.510838 | 0.503139 | −0.007699 |
| ask_volume_touch | 0.128072 | 0.120048 | −0.008024 |
| vol_per_min | 0.105444 | 0.098690 | −0.006754 |
| orderbook_imbalance | 0.135665 | 0.131865 | −0.003800 |
| bid_cancellation_ticks | 0.374143 | 0.374291 | +0.000148 |
| bid_cancellation_depth | 0.374105 | 0.374293 | +0.000188 |
| log_inter_arrival_time | 0.098039 | 0.102887 | +0.004848 |
| limit_bid_order_ticks | 0.347196 | 0.352480 | +0.005284 |
| limit_bid_order_depth | 0.346911 | 0.352231 | +0.005320 |
| limit_ask_order_depth | 0.238892 | 0.247656 | +0.008764 |
| limit_ask_order_ticks | 0.242587 | 0.251414 | +0.008827 |
| ask_cancellation_depth | 0.364126 | 0.374869 | +0.010743 |
| ask_cancellation_ticks | 0.364520 | 0.375287 | +0.010767 |
| ofi_down | 0.251808 | 0.267052 | +0.015244 |
| ofi_up | 0.250917 | 0.274622 | +0.023705 |

**`bid_volume` alone contributes −0.01136 of the −0.01582 total WS-21 gap — about 72 %.**
`log_time_to_cancel` adds another ~21 %. **Nine of 21 features are worse under hybrid.**
*Reading:* "hybrid fits the distribution better" is true as an aggregate and false as a
general statement; the honest claim is "hybrid fits **bid-side volume** and **cancel timing**
substantially better, is a wash or slightly worse elsewhere."

### §4.4 Return predictability (7.2) — B 级，长程无信号

| horizon | N_eff | IC base | IC hyb | ranked_IC base | ranked_IC hyb | DirAcc base | DirAcc hyb |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 853 | **0.56825** | 0.56362 | 0.57070 | **0.57798** | 0.84760 | **0.85698** |
| 50 | 2,113 | 0.47087 | **0.49635** | 0.52261 | **0.53519** | 0.63275 | **0.64127** |
| 100 | 2,588 | 0.29439 | **0.33152** | 0.33502 | **0.36500** | 0.53903 | **0.55410** |
| 250 | 2,889 | 0.14522 | **0.15554** | 0.15512 | **0.18067** | **0.49152** | 0.48217 |
| 500 | 2,970 | 0.05926 | **0.08472** | 0.11012 | **0.11346** | 0.49764 | 0.49057 |
| 1000 | 3,009 | **0.02707** | 0.01389 | **0.06284** | 0.04796 | **0.48255** | 0.47923 |

\KL{这里有一个必须写出来的读法约束，否则会把噪声当反转}

**DirAcc at $h\ge250$ is 0.48–0.50 for BOTH arms — there is no directional information left
in that range.** IC differences at $h\ge250$, including the $h=1000$ "reversal", must not be
read as signal.

### §4.5 Cost

| | baseline | hybrid |
|---|---:|---:|
| parameters | 33,610,439 | 35,435,423 (+5.43 %) |
| inference, 3,136 seq @ 2k | **39:48** | **1:03:47** (+60 %) |
| single-GH200 microbench @500 ctx (bsz4×13k) | 0.3703 s/step, 140,413 tok/s, peak 71.6 GB | 0.4092 s/step (+10.5 %), 127,089 tok/s, peak **67.1 GB (−6.3 %)** |

\KL{显存下降是反直觉的，但有机制解释——值得记下来}

Attention **reduces** peak memory: Mamba3's SSD chunked scan materialises per-chunk
intermediates while Pallas flash attention is $O(L)$ memory. The quadratic cost is in FLOPs,
not HBM: single-layer attention FLOPs $=4L^2 d_{model}$; at $L=52{,}000$ that is
$6.9\times10^{12}$ per sample per layer vs $4.3\times10^{11}$ at $L=13{,}000$ — length ×4,
this term **×16**（出处 `launch_2k_hybrid.sh` 头注释 + `CTX_4K_FEASIBILITY.md`）.

---

## §5 Results at 500-Message Context — A 级，且它的结论被 2k 限定了适用范围

### 副题：3 seed @12000 与 2 seed @32001；回指优势被抹平、分布优势不被抹平

> 速览：@12000 三 seed 全部同号、6 桶单调到 +13.10 pp；但 @32001 该优势归零甚至转负，
> 说明在短窗口下它是**收敛速度**；而 LOB-Bench 优势撑到了 32k。

### §5.1 Three-seed replication at step 12000（A 级）

**These are pre-NoPE-fix hybrid numbers, i.e. LOWER BOUNDS**（见 §7.1）.

Paired differences (hybrid − baseline), 3 seeds:

| metric | s2026 | s2027 | s2028 | mean | spread | same sign |
|---|---:|---:|---:|---:|---:|:---:|
| WS-21 ↓ | −0.03693 | −0.03194 | −0.03090 | **−0.033256** | 0.006032 | yes |
| KS-21 ↓ | +0.00152 | +0.00124 | +0.00209 | +0.001617 | 0.000847 | yes（consistently **worse**） |
| L1-21 ↓ | −0.01282 | −0.00782 | −0.00981 | **−0.010150** | 0.005001 | yes |
| L1 exact ↑ | +1.5696 | +2.4672 | +2.6230 | **+2.2199 pp** | 1.053385 | yes |
| IC h=250 ↑ | +0.0195 | **−0.0455** | **−0.0201** | −0.0154 | 0.0650 | **no** |
| DirAcc h=250 ↑ | +0.0098 | +0.0188 | +0.0059 | +0.0115 | 0.0129 | yes |

Mechanism criterion by age, all three seeds:

| reference age | s2026 | s2027 | s2028 | mean (pp) | same sign |
|---|---:|---:|---:|---:|:---:|
| 1–10 | +1.87 | +2.98 | +3.35 | +2.73 | yes |
| 11–25 | +1.95 | +2.53 | +2.21 | +2.23 | yes |
| 26–50 | +5.44 | +5.48 | +5.13 | **+5.35** | yes |
| 51–100 | +9.40 | +8.38 | +9.71 | **+9.16** | yes |
| **101–250** | +12.79 | +13.15 | +13.35 | **+13.10** | yes |
| before-window | +2.56 | +3.12 | +2.99 | +2.89 | yes |

\KL{`before-window` 是这条证据里最锋利的一刀——机制在它该失效的地方失效了}

`before-window` accounts for **35.7 %** of all references, and **attention has nothing to
look at there**. The effect appears exactly where the mechanism predicts and shrinks to
near-noise where the mechanism predicts. A confound that produced a general improvement
would not respect that boundary.

### §5.2 Fully trained, step 32001（A 级，2 seed）

| metric | baseline@32001 | hyb s2026 | hyb s2027 | rel. (s2026) | winner |
|---|---:|---:|---:|---:|:---:|
| (7.1) CE ↓ nats/token | 0.532610 ⚠ | **0.528384** ⚠ | same | **−0.79 %** | hybrid |

> ⚠ **这两个绝对值系统性偏低约 19.2 %，相对差不受影响**（2026-08-16 查明）。
> `_compute_ce_unified` 在 `ignore_times=True`（默认）下把 26 个位置里的
> `TIME_START_I..TIME_END_I`（11–15，即 `time_s`+`time_ns` 共 **5 位**）的 CE 置零，
> 而 `np.mean(ce)` 的分母仍是全部 **26** 位 ⇒ 每个报出的 nats/token 都被 5/26 的恒零项稀释。
> 换算 `nats/message = nats/token × 26` 同样偏低。**两配置同为 26tok、同一掩码，所以
> −0.79 % 这个比值以及本报告的一切结论都不受影响**；但绝对值不可跨编码引用，也不可
> 当作「每 token 的真实不确定度」。出处 `src/lob/train_helpers.py:745-770`。
| (7.3) WS-21 ↓ | 0.20714 | **0.18428** | **0.18878** | **−11.0 %** | hybrid |
| (7.3) KS-21 ↓ | 0.10458 | **0.10139** | **0.10435** | −3.1 % | hybrid |
| (7.3) L1-21 ↓ | 0.16451 | **0.14937** | **0.15158** | −9.2 % | hybrid |
| **(7.4) L1 exact ↑** | **74.8939 %** | 74.4415 % | 74.1217 % | **−0.60 %** | **baseline** |
| (7.4) unresolved ↓ | 4.0564 % | **3.9787 %** | **3.9543 %** | −1.9 % | hybrid |

\claim{在 500 上下文下，回指优势是收敛速度而不是能力——它在 32k 步被完全抹平甚至转负。}

Reference-recall Δ at two step counts (post-PE-fix):

| reference age | Δ @12000 (pp) | Δ @32001 (pp) |
|---|---:|---:|
| 1–10 | +5.05 | +0.17 |
| 11–25 | +4.04 | −0.43 |
| 26–50 | +7.80 | −0.69 |
| 51–100 | +11.15 | −0.54 |
| **101–250** | **+16.38** | **−2.03** |
| before-window | +4.35 | −1.26 |

"Just train the baseline longer" reproduces it:

| | baseline | hybrid |
|---|---:|---:|
| L1 exact @12000 | 63.46 % | **68.91 %** |
| L1 exact @32001 | 74.89 % | 74.44 % |

hybrid@12000's 68.91 % sits **between** baseline's 12k and 32k values — on this metric hybrid
≈ "baseline trained a few thousand steps more".

**What did NOT converge away:**

| metric | @12000 | @32001 |
|---|---:|---:|
| WS-21 | −11.7 % | **−11.0 %** |
| L1-21 | −6.0 % | **−9.2 %** |
| CE | −0.87 % | **−0.79 %** |

### §5.3 Baseline's own recall curve, full pool（A 级，出处 `STAGE_2`）

`j5877859_30nkkohd_5877859 @ 32001`, 3,136 sequences / 784,000 generated messages:

| age | n | L1 exact | L2 fallback | unresolved |
|---|---:|---:|---:|---:|
| 1–10 | 101,437 | **96.45 %** | 3.55 % | 0.00 % |
| 11–25 | 66,111 | 95.90 % | 4.10 % | 0.00 % |
| 26–50 | 41,070 | 91.88 % | 8.12 % | 0.00 % |
| 51–100 | 31,023 | 86.07 % | 13.93 % | 0.00 % |
| 101–250 | 14,948 | **74.57 %** | 25.43 % | 0.00 % |
| **before-window** | **129,150** | **39.16 %** | 48.78 % | 12.05 % |

By event class: `new` 389,090 (100 % not-a-cancel), **`cancel` (partial) 1,291 → 24.94 %
exact**, **`delete` (full) 382,448 → 75.06 %**, `execution` 11,171 (100 % not-a-cancel).

\KL{两个必须随这张表一起引用的天花板，否则会把结构性不可达算成模型能力不足}

1. **`before-window` has a hard visibility ceiling of 88.18 %** — an oracle audit shows only
   88.18 % of real continuations' reference targets have a history row to point at at all;
   the remaining 11.82 % are unreachable by **any** architecture. The 39.16 % is bounded by
   visibility, not only by capability.
2. **`execution` is not untested, it is undefined**: reference resolution applies only to
   event types 2/3 (`order_id = jax.lax.cond((event_type==2)|(event_type==3), ref_id,
   new_id)`); executions take a brand-new id.

`nan_branch_fired` = **0 / 784,000**; `price_differs_from_ref` = 146,267 (18.66 %) and is
explicitly **not** a defect counter.

---

## §6 The Parameter-Matched Arm（等参数组）— **A 级，已结案（2026-08-17）**

### 副题：把「注意力这个机制」和「多出来的 5.43% 容量」分开

> 速览：`d_ff` 2560→1135，参数 33,609,998（比 baseline 小 441 个）；判读规则**预注册**；
> 目前 4,479/6,400；**唯一在盘的 pmatch bench 是 step 2956、N=768、不同索引集，不构成比较。**

### §6.1 The seven-threat adversarial audit that motivated it

| threat | measurement | verdict |
|---|---|---|
| hybrid is a larger model | 33,610,439 → 35,435,423 (**+5.43 %**) | **real confound** |
| cherry-picked far-reference samples | 501–1000 share 1.14 % vs 1.06 %; max gap ±0.85 pp | excluded |
| different message composition | new 49.64/49.71, delete 48.67/48.44 | excluded |
| unequal steps | baseline has **7 more** | excluded, and **disfavours hybrid** |
| unequal effective batch | both 80 | excluded |
| unequal interruption counts | hybrid 4, baseline 2 | present, **disfavours hybrid** |
| single seed | seed 2026 only | real limitation, **this arm does not fix it** |

### §6.2 Parameter balancing — solved, not tuned

$d_{ff}$: 2560 → **1135**, verified by constructing one TransformerBlock on CPU
（出处 `scratchpad/count_pmatch_params.py`）:

| | params |
|---|---:|
| TransformerBlock $d_{ff}=2560$ | 4,923,520 |
| TransformerBlock $d_{ff}=1135$ | 3,098,095 |
| reduction | **1,825,425** |

\KL{手算和构造差 1,425——这个差本身是一条方法教训}

Predicted total = 35,435,423 − 1,825,425 = **33,609,998**. Hand arithmetic gives
$2\times640\times(2560-1135)=1{,}824{,}000$, off by **1,425 = 2560 − 1135** — the
up-projection **bias**, which the hand calculation omitted. *Lesson: initialise the module
and count, do not trust the formula.* Measured on hardware（6006717 node0）:
`[*] Trainable Parameters: 33609998` — matching the CPU prediction exactly.

| | params | vs baseline |
|---|---:|---:|
| baseline | 33,610,439 | — |
| hybrid faithful | 35,435,423 | +5.43 % |
| **hybrid param-matched** | **33,609,998** | **−0.0013 %（小 441 个）** |

A runtime contract `EXPECTED_PARAMS=33609998`（出处 `src/lob/train.py:349-364`）raises on
mismatch: a resume that failed to deliver `d_ff` would silently produce a **clone of the
faithful arm** and, hours later, a false "the matched arm also won".

### §6.3 Pre-registered reading rules（**写在结果之前**）

标准只看 **`hybpm − base` 在 `501–1000` 档的配对差**，因为那是机制假设唯一做出定量预测的位点：

- **≥ +15 pp**，且 `11–25 → 501–1000` 六档递增 → the effect is **architectural**;
  `CTX2K_FINAL.md`'s subject may change from "this hybrid configuration" to "**attention**".
- advantage **collapses to ~0 or goes negative** → the effect is mainly **capacity**; the
  +24.7 pp conclusion must be rewritten as "a larger model is better at long-range recall".
- **in between** (e.g. +8 pp) → both contribute; report the split, make **no** binary claim.

\KL{一处必须在结果落地之前做完的措辞修复——否则它就是事后挑}

> **2026-08-15 措辞修复（等参数组结果尚未落地时做的）。** 原文写的是「amplifies
> **monotonically** to ≥ +15 pp」。机械检验参照曲线本身发现它**不是全程单调**——`1–10`
> 档（+3.67）高于 `11–25` 档（+3.02），且这个凹点在三个 seed 上都复现。照原措辞，等参数组
> 无论出什么结果都会因为「不单调」被判成机制不成立，**这条规则字面上不可满足**。
> 因此把单调性的范围收窄到与参照曲线一致的 `11–25 → 501–1000` 六档，
> **数值阈值 +15 pp / ~0 / in-between 一个没动**。改的是一个描述错误，不是标准松紧。

### §6.4 The early read — B 级，且**不是**预注册的那个比较

\KL{必须先说它为什么不是，再给数字，否则数字会被当成结论用}

`bench2k_20260814T151453Z_j6007121_hybpm_s2026_11805`, internal run name
`hybrid-pmatch-early-step2956`. **Three reasons it must not be read against §6.3:**
(1) **step 2956**, less than half of baseline@6265's training; (2) **768 sequences**, not
3,136, `world_size=3`; (3) **different index set** — sha `434917a0…` vs the arms' `6c6defbc…`,
so it is **not paired**.

| age bucket | n | hybpm@2956 | (base@6265, different index) |
|---|---:|---:|---:|
| 1–10 | 76,205 | 87.0363 % | 93.4750 % |
| 11–25 | 55,604 | 84.1342 % | 93.0179 % |
| 26–50 | 43,714 | 76.9342 % | 87.9633 % |
| 51–100 | 41,767 | 68.9252 % | 79.2517 % |
| 101–250 | 40,133 | 52.7097 % | 59.0273 % |
| 251–500 | 17,421 | 28.1557 % | 28.0087 % |
| **501–1000** | 6,607 | **13.7884 %** | 9.2782 % |
| before-window | 78,484 | 17.0379 % | 22.2530 % |

Overall cancel+delete n = 359,935; exact **59.9744 %**, fallback 35.7540 %, miss 4.2716 %.
LOB-Bench: WS-21 **0.35060728787834794**, KS-21 0.18996612748796868, L1-21 0.24996376150661975.

*Suggestive shape only:* behind at the near end (less training) and ahead at the far end
**at equal-or-smaller parameter count**. Two confounds (training amount, index set) make this
a reason to finish the run, **not a result**.

---

### §6.5 THE ANSWER — 五个步号匹配点，全部指向**机制**（A 级，2026-08-16 收口）

\claim{`501–1000` 档的 hybpm−base 在五个测量点上是 +20.59 / +16.22 / +13.90 / +15.47 / +16.78 pp，均值 +16.59，跨度 [13.90, 20.59]，各点 seed 极差 0.17–0.95。四点越过预注册的 +15 门槛，无一落在容量区间。该档复刻比均值 66%。}

| 测量点（步号） | seed | **`501–1000`** | 极差 | 该档复刻比 | 整体（pp）| 整体复刻比 | 裁决 |
|---|---:|---:|---:|---:|---:|---:|---|
| ~4,500 `4462/4479/4555` | **5** | **+20.5912** | 0.3989 | 77.2 % | +13.0381 | 88.1 % | 机制性 |
| ~4,850 `4789/4854/4911` | 3 | **+16.2190** | 0.3501 | 63.9 % | +3.8614 | 58.7 % | 机制性 |
| ~5,063 `5062/5063/5103` | 4 | +13.8983 | 0.9544 | 57.7 % | +1.7787 | 26.5 % | 两者兼有 |
| ~5,300 `5330/5271/5297` | 4 | **+15.4707** | 0.1737 | 69.3 % | +0.1147 | 3.6 % | 机制性 |
| ~6,020 `6059/5994/6057` | 2 | **+16.7844** | 0.3045 | 63.7 % | +3.5114 | 45.6 % | 机制性 |

\KL{加到 6–8 个 seed 的复核，以及一条关于「极差」的警告}

**2026-08-16 11:1x 复核（seed 加到 6–8 个）**：`501–1000` 档五点变为
+20.5581 / +16.1267 / +13.8602 / +15.5397 / **+17.0210**，均值 **+16.62**（5-seed 版 +16.63），
**逐点移动仅 0.033–0.092 pp，五个裁决一个没变**。

**2026-08-17 终局（采样种子 7 / 9 / 9 / 9 / 9，共 43 组三方齐备）**：

| 测量点 | 采样种子 | **`501–1000`** | 极差 | **均值/极差** | 该档复刻比 | 整体（pp）| 整体复刻比 | 裁决 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| ~4,500 | **7** | **+20.5353** | 0.4330 | **47.4** | 75.9 % | +13.0186 | 88.0 % | 机制性 |
| ~4,850 | **9** | **+16.1339** | 1.3050 | **12.4** | 66.6 % | +3.9286 | 59.2 % | 机制性 |
| ~5,063 | **9** | +13.8784 | 1.2792 | **10.8** | 58.4 % | +1.8218 | 27.3 % | 两者兼有 |
| ~5,300 | **9** | **+15.5951** | 1.1201 | **13.9** | 69.6 % | +0.1411 | 4.5 % | 机制性 |
| ~6,020 | **9** | **+17.0823** | 0.9403 | **18.2** | 68.3 % | +3.6635 | 46.9 % | 机制性 |

均值 **+16.65 pp**，范围 [13.88, 20.54]；该档复刻比均值 **67.8%**，范围 [58.4, 75.9]。
**五点的均值/极差全部 ≥ 10 = 全部 A 级**——而同一批数据的**整体**掉到 B+（2.9）与
**B（0.2，极差完全吞掉均值）**。**噪声是同一批噪声，区别只在于整体把不需要注意力的近档
和需要注意力的远档加权平均了。** 这是「整体复刻比是坏统计量」的最强证据。

> **⚠ 但「均值/极差」这个等级标准会随 seed 数变差，尽管估计量在变好。**
> 极差 = 最大减最小，**随 n 单调增**（样本越多越容易撞到尾部）：本次
> ~5,300 的极差从 0.1737（4 seed）涨到 1.1201（7 seed），~4,850 从 0.3501 涨到 1.3050。
> 照极差判级会得出「加数据反而降级」的荒谬结论。**若要在更多 seed 下继续分级，
> 必须换成 2×SE**（`2σ/√n`，随 n 递减）。本报告的 A/B 分级是在 3–5 seed 下给出的，
> 结论有效；但**不要拿它和更大 n 的极差直接比**。同
> [[feedback_separate_eval_noise_from_seed_noise]]。

\KL{三条必须一起读的结论，少一条就会读错}

**(1) 幅度已收敛。** ~4,500 从 3→4→5 个 seed，值走 +20.6157 → +20.6324 → **+20.5912**
（总变动 **0.04 pp** 对 20.6 pp 的效应，信噪 **500 倍**）。**再加 seed 不会改变任何结论。**

**(2) 份额可引用，但只能用逐档的那个。** `501–1000` 档复刻比五点 77.2 / 63.9 / 57.7 /
69.3 / 63.7 %（均值 **66 %**，跨度 [58, 77]）；**整体**复刻比五点 88.1 / 58.7 / 26.5 /
**3.6** / **45.6** %，从 3.6 跳回 45.6，**不可引用**。差别的来源是 `before-window`
——~6,020 上窗口内七档复刻比全在 **63.4–81.1 %**（seed 2026），而 `before-window` 只有
**19.2 %**，把加权平均拖到 **45.6 %**（两 seed 均值）。**摆动的是统计量，不是效应。**

**(3) 「整体复刻比随训练单调塌到零」是错的**，那是四个点全落在 4,500–5,300（跨度 800 步）
内读出的外推；第五个点跨到 6,020，整体差从 +0.11 回到 **+3.51**。**跨度比密度值钱。**

**唯一的例外点是 ~5,063**（+13.90，落在「两者兼有」带），它同时是整体复刻比的低谷与
seed 极差最大的点（0.9544）。诚实的限制：**等参数组那条训练线被中断六次（另加十余次关卡
拒绝），而 base/hyb 是一次训完的——「步数」这个自变量对三个配置并不等价**，该混淆未被控制。

---

#### §6.5.0 首次判定的原始记录（三 seed，单测量点）

以下保留 2026-08-15 首次给出答案时的记录，数字已被上表取代，保留供追溯。

### ~~§6.5~~ THE ANSWER — matched-step three-way says **mechanism**（**A 级，三 seed**）

\claim{参数比 baseline 还少 441 个的等参数组，在 501–1000 档拿到 +20.6157 pp（预注册阈值 +15 pp，三 seed 极差 0.3431），复刻完整 hybrid 优势的 87.7%。24 格全部同号，均值/极差 30.8。参数量混淆已排除。}

**Three-way at ~4,500 steps, seeds 2026 / 2027 / 2028.** Arm identity from checkpoint metadata,
index sha `6c6defbc…` on all three, N=3,136 each: `base@4462` / `hybpm@4479` / `hyb@4555`.

| 引用距离 | **hybpm−base** | 极差 | 同号 | hyb−base | 极差 | **等参数组复刻** |
|---|---:|---:|:---:|---:|---:|---:|
| 1–10 | +7.5300 | 0.3116 | 3/3 | +8.2070 | 0.5173 | 91.8 % |
| 11–25 | +7.6977 | 0.1690 | 3/3 | +7.8967 | 0.2400 | **97.5 %** |
| 26–50 | +11.2951 | 0.2609 | 3/3 | +12.0064 | 0.3189 | 94.1 % |
| 51–100 | +14.9389 | **0.1947** | 3/3 | +16.5914 | 0.3316 | 90.0 % |
| 101–250 | +19.2244 | 0.8521 | 3/3 | +22.6927 | 0.4124 | 84.7 % |
| 251–500 | +22.7878 | 1.3497 | 3/3 | +29.0924 | 0.6542 | 78.3 % |
| **501–1000** | **+20.6157** | 0.3431 | 3/3 | +26.7176 | 0.5332 | 77.2 % |
| before-window | +4.9557 | 0.3112 | 3/3 | +9.6774 | 0.5210 | 51.2 % |
| **整体** | **+12.9017** | 0.4187 | 3/3 | +14.7175 | 0.7395 | **87.7 %** |

绝对水平（seed 2026）：`501–1000` 档 base **2.5560 %** → hybpm **23.3813 %**（**9.1×**）→ hyb 29.5319 %。

**Against §6.3's pre-registered rule** (`501–1000` bucket): **+20.6157 pp ≥ +15 pp →
architectural**，且是三 seed 支撑的（极差 0.3431，效应/极差 = **60×**）。整体的
**均值/极差 = 30.8 → A 级**。The subject of `CTX2K_FINAL.md`'s claim may widen from "this
hybrid configuration" to "**attention**". 至多 **12.3 %** 的优势可归给那 +5.43 % 参数，
而这个上界仍是**保守**的：`hyb@4555` 比 `base@4462` 多训 2.08 %，抬高 `hyb−base`、压低比值。

\KL{份额还没定——另一个步号更紧的测量点给出不同的比值，必须一起报}

**但份额尚未收敛。** 步号跨度更小的三元组 `5062/5063/5103`（**0.81 %**，单 seed 2027）给出
`501–1000` 档 **+13.8678 pp**（落在预注册的「中间」区间）、整体复刻比 **26.7 %**。
17 步的训练领先按实测斜率（0.013 pp/步）只值 **0.08 pp**，解释不了 87.7 % 与 26.7 % 的差距，
**所以这是真分歧**。当前可引用的是**方向**（等参数组用更少的参数在最远档拿到 +13.9 至 +20.6 pp
⇒ 纯容量解释被排除），**不可引用的是份额**。该比值是两个差的商，分母自身有涨落，除法放大方差
——结论需要同一步号上的多 seed，正在补到 4 个。

#### §6.5.1 LOB-Bench points the other way — but **三 seed 之后它仍然过不了噪声门槛**

\claim{等参数组的 WS-21 三 seed 均值比 baseline 差 +0.008280，相对差仅 3.07%，远小于 16.6% 的跨 seed 噪声底；它不是被推翻，而是始终没到可判的门槛。且单 seed 下 100.4% 的差额来自单个特征 `log_time_to_cancel`。}

**三 seed（2026/2027/2028）：**

| 指标 | **hybpm−base** 均值 | 极差 | hyb−base 均值 | 极差 | 等参数组占 | 相对差 | 判 |
|---|---:|---:|---:|---:|---:|---:|---|
| WS-21 | **+0.008280** | 0.003999 | −0.019022 | 0.005983 | −43.5 % | **3.07 %** | **不可分辨** |
| KS-21 | +0.002911 | 0.002637 | −0.004579 | 0.001848 | −63.6 % | 2.03 % | **不可分辨** |
| L1-21 | +0.005478 | 0.002753 | −0.011219 | 0.002448 | −48.8 % | 2.60 % | **不可分辨** |

> **加了两个 seed 也没把它抬过门槛。** 三个相对差 3.07 / 2.03 / 2.60 % 全部远小于
> `BASELINE.md` 标定的 **16.6 %** 跨 seed 噪声底。**「等参数组分布拟合更差」在三 seed 下
> 依然不构成结论**——正确的说法不是「被推翻」，而是「这个指标分辨不了这个量级的差别」。
> 比较之下回指的 +20.62 pp 建立在 2.56 % 的基数上、极差 0.34，效应/噪声差三个数量级。

单 seed（2026）的绝对值与分解仍保留如下，用于机制假设的构造，**不用于结论**：

| 指标 | base | hybpm | hyb | hybpm−base | hyb−base | 等参数组占 |
|---|---:|---:|---:|---:|---:|---:|
| WS-21 | 0.271528 | 0.280443 | 0.253993 | **+0.008916** | −0.017534 | **−50.8 %** |
| KS-21 | 0.143517 | 0.145794 | 0.138374 | +0.002277 | −0.005143 | −44.3 % |
| L1-21 | 0.211264 | 0.215941 | 0.199161 | +0.004677 | −0.012103 | −38.6 % |

**逐特征分解**：`log_time_to_cancel` 从 base 的 **0.10743** 变成 hybpm 的 **0.29548**
（**2.75×**，完整 hybrid 只有 0.12512），单项贡献 **+0.008955 = 总差的 100.4 %**，
**其余 20 项合计 −0.000039**。按语义分组：

| 组 | 项数 | **hybpm−base** | hyb−base |
|---|---:|---:|---:|
| 成交量 / 价差 | 7 | **−0.26655**（比完整 hybrid 还好）| −0.16929 |
| 订单流不平衡 | 4 | +0.02573 | −0.12434 |
| **时间** | 2 | **+0.18732** | −0.05433 |
| **撤单位置** | 4 | **+0.16044** | +0.02049 |
| 挂单位置 | 4 | +0.08029 | −0.04076 |

\KL{这个指标已经第二次把结论集中到一两个特征上——每次引用聚合值都必须同时给分解}

**机制假设（待三 seed 验证）**：注意力块的 FFN 从 2560 缩到 1135（2.26×）**换掉的是撤单
时机与位置的建模能力，换来的是成交量/价差的建模能力**。等参数组在 `spread`（占 319.5 %）、
`bid_volume`（132.7 %）、`bid_volume_touch`（163.6 %）上**超过**完整 hybrid。

**噪声门槛（必须先过）**：WS-21 跨 seed 噪声底 **16.6 %**；`hybpm−base` 仅 **+3.28 % 相对**，
`hyb−base` **−6.46 % 相对** —— **单 seed 下两者都不可分辨**。回指侧不受此限（+20.83 pp
建立在 2.56 % 的基数上）。三 seed 复现在跑（`6019793`，ETA 2026-08-15 13:55Z）。

出处：`refer_success_{base_4462,hybpm_4479,hyb_4555}_s2026.json` 与同目录 `summary.json`。

#### §6.5.1b 八个测量点之后：等参数组的方向**不可引用**，因为它随训练步数震荡

> **⚠ 本节已改写三次。** 08-17 判「落在 16.6% 噪声底以下、不可分辨」——**借错了噪声底**
> （标定在 8.1M 模型、3 个种子的极差、绝对水平的散布上）。08-18 判「真实的交叉，
> ≤4,850 更差、≥5,063 更好」——**只有 5 个点就读出了模式**，补到 8 个点后第 6 点符号翻回。
> **08-19 定版：方向不可引用。** 三次错法不同，根子相同：**沿着采样种子这条轴测精度，
> 而主要的变化在训练步数那条轴上。** §6.5.1 那句「加了两个 seed 也没把它抬过门槛」同样作废。

\claim{`hybpm−base` 的 WS-21 在八个步号匹配的测量点上符号是 `+ + − − − + − −`（3 正 5 负）。每个点自身精确可测（跨采样种子配对 t = 5.1–30.9），但**跨测量点的标准差 0.0196 大于均值 −0.0127，比值 0.65 < 1**，且翻正的 ~5,670 落在 hybpm3 训练段中间、不是重启造成的。⇒ 只能说「第 X 步上是 Y」，不能说方向。参照：`hyb−base` 的 WS-21 九点 9/9 全负、比值 2.19，方向可引用而幅度不可。}

| 测量点 | 采样种子 | `hybpm−base` WS-21 | 相对基线 | **实测 σ** | **t = d/SE** | 同号 | 等参数组占 |
|---|---:|---:|---:|---:|---:|:---:|---:|
| ~4,500 | 7 | **+0.010632**（更差）| 3.97% | 1.03% | **+8.95** | 9/9 | −50.3% |
| ~4,850 | 9 | **+0.006299**（更差）| 2.27% | 0.99% | **+6.88** | 9/9 | −15.5% |
| ~5,063 | 9 | **−0.029872**（更好）| 9.88% | 0.96% | **−30.87** | 9/9 | **+73.5%** |
| ~5,300 | 9 | **−0.041430**（更好）| 13.51% | 1.49% | **−27.28** | 9/9 | **+79.8%** |
| **~5,320** | 3 | −0.025636（更好）| — | — | **−26.0** | 3/3 | — |
| **~5,670** | 3 | **+0.009216（更差）** | — | — | **+5.1** | 3/3 | — |
| **~5,860** | 3 | −0.016541（更好）| — | — | **−11.2** | 3/3 | — |
| ~6,020 | 9 | **−0.013899**（更好）| 4.97% | 1.72% | **−8.69** | 9/9 | **+92.8%** |

KS-21 / L1-21 给出同样的交叉，t 最高到 **−84.64**（~5,063 的 L1-21）。

\KL{初版错在哪：借来的噪声底量的不是我要判的那个量，而它看起来是有出处的}

**根因。** 初版引用 `BASELINE.md:111` 的「跨 seed 噪声底 16.6%」，那一行原文是：

```
跨 seed（3 seed，8.1M）| 16.6%（WS 0.20949–0.24424）| research/B §3.3
        ↑        ↑              ↑
   3 个种子的极差  8.1M 模型   绝对水平的散布
```

**三处标定条件都不匹配**：(a) 模型 **8.1M** vs 本比较的 **33.6M**；(b) **3 个种子的极差**
而非标准差，极差随 n 单调增；(c) 量的是**绝对水平的散布**，而本比较是**同种子配对差**。
同一张表上一行的 `LOB-Bench harness 噪声底 0.6%` 才是同量级——**实测本比较的噪声
0.17–1.72%，与之吻合，与 16.6% 差 10–17 倍。**

> **规矩（新增，硬性）**：**引用任何噪声底 / 阈值 / 标定常数之前，先读它的标定条件**
> ——模型规模、样本数、是极差还是 σ、量的是绝对水平还是配对差。
> **常数错在标定条件上比错在数值上更难发现**，因为它带着出处，看起来已经被验证过。
>
> 附带：初版还把**均值效应**拿去比**单次运行噪声**（`d/σ` 与 `d/SE` 混用，差 `√n`）。
> 两个错误同向叠加成 17 倍。但**即便只用 `d/σ`，五点里也有四点超过 1**（2.29–10.29）
> ——初版的结论在任一口径下都不成立。同 [[feedback_per_run_rate_and_mean_effect_are_two_statistics]]。

**交叉本身是结果。** 可读的机制：注意力块的 FFN 从 2560 缩到 1135，**早期确实拖慢分布拟合**
（宽度减半，逐 token 特征统计学得慢），**训练推进后劣势消失并反超**——说明那部分容量买的是
**收敛速度**而非**终点质量**。同 [[feedback_judge_representations_at_convergence]]。

**因此 §6.5.1 那条机制假设（FFN 宽度买撤单时机、注意力买远程检索）需要重测而非作废**：
它建立在 ~4,500 的逐特征分解上，而 ~4,500 恰好落在交叉的**更差**一侧；要判它必须在
交叉之后的测量点（≥5,063）上重做逐特征分解。**在此之前不得引用。**

出处：`code/threeway.py` 的 LOB-Bench 段 + 配对 t 检验，八个测量点全部重算。

\KL{两条轴上的方差差一个量级，而我三次都只看了第一条}

| 方差来源 | 怎么测 | WS-21 上的量级 |
|---|---|---|
| 采样种子 | 同一步号换生成种子 | σ ≈ 0.001–0.005，t = 5–31 |
| **训练步数** | 换步号 | **SD = 0.0196** |

**t = 30.9 衡量的是「这一步上这个差测得准不准」——测得非常准。但要引用的命题
（「等参数组分布拟合更差」）里没有步数，所以它的误差棒必须跨步数取。**

| 比较 | 均值 | 跨步 SD | \|均值\|/SD | 符号 | 可引用 |
|---|---:|---:|---:|---|---|
| `hyb−base` `501–1000` | +24.12 pp | 1.50 | **16.10** | 9/9 正 | 方向 + 幅度 |
| `hybpm−base` `501–1000` | +16.89 pp | 2.00 | **8.44** | 8/8 正 | 方向 + 幅度 |
| `hyb−base` WS-21 | −0.0339 | 0.0155 | **2.19** | 9/9 负 | 仅方向 |
| **`hybpm−base` WS-21** | **−0.0127** | **0.0196** | **0.65** | **3 正 5 负** | **都不行** |

> **规矩（新增，硬性）：一个不含步数的命题，误差棒必须跨步数取。** 只报跨采样种子的
> t 值，会把「这一步测得准」冒充成「这个结论成立」。方向性主张至少三个测量点且全部同号。
> 同 [[feedback_separate_eval_noise_from_seed_noise]]（评测方差 vs 种子方差），
> **这里补上第三个成分：训练步数**，在这个指标上它是最大的一个。

---

#### §6.5.2 `before-window`: what the extra 5.43% of parameters actually buys

\claim{`miss`（点名了一张任何地方都不存在的订单）**100% 落在 `before-window`**，因而 100% 落在按构造无法检索的引用上。完整 hybrid 在五点中的四点显著降低 miss 率（−1.630 / −0.678 / −0.776 / +0.119 / −0.670，|t| = 4.7–48.6），而等参数组只复刻其中 **40%**——对比远程回指的 **68%**。注意力买检索，FFN 宽度买「知道自己不知道」，两者可分离。}

**为什么 `miss` 是这一档唯一无歧义的量。** `exact` 与 `fallback` 的高低取决于真实数据里
before-window 引用应有的比例，而产物里没有记录它，所以「发得少 = 更好」无法断言。
**`miss` 不需要基准**：它指模型点名的那张订单在整个序列里根本不存在，无论真实分布如何，
瞎编都是错的。

| 测量点 | 采样种子 | miss% base | hybpm | hyb | pm−base | t | hyb−base | t |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ~4,500 | 7 | 5.052 | 3.844 | 3.421 | **−1.207** | −34.8 | **−1.630** | −48.6 |
| ~4,850 | 9 | 4.432 | 3.936 | 3.754 | **−0.496** | −22.8 | **−0.678** | −32.0 |
| ~5,063 | 9 | 3.868 | 3.927 | 3.092 | +0.059 | +1.6 | **−0.776** | −26.6 |
| ~5,300 | 9 | 3.364 | 4.070 | 3.483 | +0.706 | +27.7 | +0.119 | +4.7 |
| ~6,020 | 9 | 3.879 | 3.609 | 3.209 | **−0.270** | −10.1 | **−0.670** | −31.7 |

分母是全部生成的撤单引用，所以份额效应（发出多少条 before-window 引用）与桶内效应
（发出的那些认对多少）都算了进去。

\KL{两个复刻比放在一起，就把「注意力买了什么」和「参数买了什么」分开了}

| 能力 | 完整 hybrid 的优势 | 等参数组复刻 | 机制 |
|---|---:|---:|---|
| 远程引用回指（`501–1000`）| +25.02 pp | **68%** | 内容寻址；注意力直接做这件事 |
| 不瞎编订单（miss 率）| −0.670 pp | **40%** | 先验质量；窗口外没有内容可寻址，注意力帮不上忙 |

**机制上这个差别是必然的**：`miss` 全部来自 `before-window`，即那些**按构造检索不到**的引用。
一个内容寻址的机制在没有内容可寻的地方不产生作用，只有更好的先验才行，而先验的容量在
FFN 的宽度里。**所以砍掉 FFN 宽度、保留注意力，两个数字就该分开——实测正是如此（68% vs 40%）。**

**这条同时把 §6.5.42 那个开放问题合上了**：那 +5.43% 的参数确实买到了东西，
买到的是**自洽性**（不去引用一张自己看不见的订单），不是远程检索。

---

## §7 Invalidated and Profiling-Only Material — C/D 级，**不得用于结论**

### 副题：保留它们是因为诊断链有价值；引用它们的数字是本条线最可能的污染方式

### §7.1 `STAGE_6` — generation side entirely VOID（C 级）

**Root cause（出处 `STAGE_7_ROOT_CAUSE_NOPE_LEAK.md`）**: `src/s5/transformer.py`'s
`TransformerBlock` has three forward paths and the `use_positional_encoding` switch was
applied to **one**:

| path | used for | checks the switch? |
|---|---|:---:|
| `__call__` | **training**, loss, held-out CE | yes (line 180) |
| `prefill` | conditioning window at generation | **no — unconditionally adds PE** |
| `__call_inference__` | per-token decode at generation | **no — unconditionally adds PE** |

hybrid is NoPE, so generation injected a sinusoid of amplitude ±1 the model had never seen.
`prefill` had a second inconsistency: returned activations came from the no-PE input while
the cached K/V came from the PE'd input. Fixed in `facf656`; regression test
`src/s5/tests/test_transformer_decode_parity.py`, discriminating power verified (reintroducing
the bug fails at error **1.992** while the `use_pe=True` control still passes).

\KL{方向很重要：这个 bug 使早期结果偏低而不是偏高，所以 12k 的结论只会变好不会变坏}

**Impact direction**: the @12000 generation results in §5.1 were all produced under this bug,
so they are **lower bounds** — WS-21 −0.03326, L1 exact +2.22 pp, and the six-bin monotone
pattern **can only improve or stay flat** after the fix. Conversely `STAGE_6`'s "hybrid is
worse" is **entirely void**.

**Why every gate missed it**: training, CE evaluation and all four smoke tests run through
`__call__`; only LOB-Bench and reference recall run through `prefill` + `__call_inference__`.
**Two inconsistent paths cannot depress any loss curve.** What caught it was the *shape* of
the reversal being wrong — hybrid's recall barely moved from 12k to 32k while baseline gained
11.43 pp. **"One curve stopped" is the shape of a fault, not of a disadvantage.**

**Still valid from `STAGE_6`**: the held-out CE trajectory (CE runs through the training
path): hybrid 12000→0.558281, 15000→0.547993, 18000→0.541424, 21000→0.535790,
24000→0.532319, 27000→0.530216, 30000→0.528783, **32001→0.528384**; monotone at all eight
points vs baseline@32001 = 0.532610.

### §7.2 `CTX2K_HYBRID_DIVERGED` — the abandoned first 2k run（C 级）

wandb `step_loss`: hybrid 2000→0.78901, 4000→0.87552, **6000→3.17533**, **8000→4.01307**,
while baseline stayed 0.57–0.72. Held-out CE independently confirms (2000→0.708324,
4000→0.868760, rising). Ruled out: gradient clipping was on; arms configurally identical;
baseline fine at the same length; 500-ctx hybrid trained 32,000 steps without diverging;
**bf16 ruled out** — `registry.py` computes `dtype` only in the transformer/nsa branch
(line 421), so the hybrid branch ran attention in `jnp.float32`.

Separate bug in the same file: `SLURM_TIMELIMIT` hardcoded `05:00:00` killed **both** arms at
exactly 5 h（`sacct` TIMEOUT 0:15, Elapsed 05:00:21）. `MAX_JOB_HOURS` is an inner-layer
graceful stop and cannot beat the outer hard limit.

**Its config table (2 nodes, K=2, effective batch 16, 32,000 steps) is NOT the arms in §4.**
Quoting "effective batch 16" next to §4's "80" contradicts itself.

### §7.3 Length-extrapolation profiling（D 级，文件自述不可引用）

A 500-ctx-trained model evaluated at 4× length, **N=64**:

| age | base@2k | hyb@2k |
|---|---:|---:|
| 1–10 | **93.72 %** | 7.32 % |
| 101–250 | **59.01 %** | 1.26 % |
| 501–1000 | **2.33 %** | **0.00 %** |

**This is not "hybrid is worse at long context".** It measures length **extrapolation**.
Convergence was ruled out (both CE flat: baseline −0.098 %, hybrid −0.076 % from 30000 to
32001). A real KV-cache bug was found and fixed (`32e09d3`, cache matched architecture names
`{'transformer','nsa'}` exactly so hybrid fell back to 26,000 while 2k needs 52,000) — but
after the fix L1 exact moved only 4.5503 % → **4.3309 %**, so it was **not the cause**.
Pipeline control passed: same code, window reset to cond250/gen250 gives 75.06 % (N=64) vs
the published 74.44 % (N=3,136).

**The window sweep localises it to the CONDITIONING segment**, N=64:

| setting | cond tokens | gen tokens | L1 total |
|---|---:|---:|---:|
| cond250/gen250 | 6,500 | 6,500 | **74.48 %** |
| cond500/gen500 | **13,000** | 13,000 | **66.80 %** |
| cond1000/gen250 | **26,000** | 6,500 | **12.63 %** |
| cond1000/gen1000 | 26,000 | 26,000 | **4.33 %** |

cond500/gen500 also reaches 26,000 total tokens during generation yet scores 66.80 % — the
collapse tracks conditioning length only. **Impact on §4: none** — the final 2k models train
at 52,000 and infer at 52,000, matched. This is the evidence base for
[[feedback_attention_extrapolation_collapse]]: **外推是崩塌不是退化，所以必须重训**。

### §7.4 `CTX2K_INTERIM`（D 级）— mismatched steps, banner says "not a report"

baseline@1895 vs hybrid@**2536** — hybrid had **34 % more training**, and 500-ctx experience
shows 12k→32k alone moves L1 exact 63.5 %→74.9 %. That confound explains the entire table
(L1 exact 30.81 % vs 62.45 %, +31.64 pp). Recorded only as timing calibration: N=512 / 4 GPUs
→ extrapolated N=3,136 ≈ 45 min (base) / 70 min (hybrid).

---

## §8 Open Work（**本文件唯一可执行的一节**）

### 副题：按依赖排序；每项给完成标准

> 速览：① 三方同步点比较（在跑）② 等参数组训到 6400 并做正式 bench ③ 第二个 seed
> ④ 逐特征分解写进结论 ⑤ 4k 上下文需要序列并行。

| # | task | completion criterion | status |
|---|---|---|---|
| **1** | **Matched-step three-way comparison** at 4462 / 4479 / 4555 | all three benched on index `6c6defbc…`, N=3,136; bucketed curve for hybpm−base reported against §6.3 rules | **running**（6014308 nid010723 / nid010896；pmatch 待训完后补） |
| **2** | Finish pmatch to 6,400 and bench at the step nearest 6,400 | `EXPECTED_PARAMS` contract green; comparison point within ~1 % of 6265/6258 | **running**（4,479/6,400，监督器 `code/handoff_pmatch.sh`） |
| **3** | **Second seed for the 2k round** | seed 2027 on all three arms; paired differences same sign | not started — **this is the single largest remaining weakness** |
| 4 | Fold §4.3's per-feature decomposition into `CTX2K_FINAL.md` §2 | that section no longer claims general distribution-fit superiority | not started |
| 5 | Event-type share vs real data | the +58.9 % execution inflation at 12k either reproduces at 6.4k or is retracted | not started |
| 6 | Block-bootstrap paired CI | replaces seed-pairing as the significance instrument | not started |
| 7 | 4,096-message context | needs sequence parallelism — 4k training is **134–142 GB = 1.57–1.66×** of 85.5 GB, impossible at bsz=1（出处 `CTX_4K_FEASIBILITY.md`）; design in `SEQPAR_DESIGN.md`, parity tests pass | designed, not built |

---

## §9 What Cannot Be Said（逐条抄原文，不重新措辞）

\KL{这一节是全文件最该被完整引用的部分}

**From `CTX2K_FINAL.md` §4:**

| cannot say | why |
|---|---|
| 「hybrid 在 2k 上全面更好」 | LOB-Bench 优势**收窄**了；h=1000 的 IC 反转 |
| 拿 500 与 2k 的绝对数字相减 | 生成长度 250→1000，分布漂移与年龄构成都变了 |
| 「注意力解决了长程问题」 | 501–1000 档 hybrid 也只有 33.98 %，**绝对水平仍然很低**，只是比 9.28 % 好得多 |
| 单 seed 的小差异 | 只跑了 seed 2026。500 上下文那轮跑过两个 seed 且同号，本轮没有 |

**From `STAGE_8_FINAL_32001.md` §5:**

- 「hybrid is uniformly better than baseline」— it **loses** on exact reference recall.
- 「attention has an intrinsic advantage on long-range recall」— the 32k data **directly
  refutes this** (at 500 ctx).
- Any causal explanation based on a **single** seed or a **single** step count.

**From `STAGE_5` §4:** 「hybrid improves return IC」**does not hold** — three seeds differ in
sign, spread 4× the mean; stage 3's +14.3 % is **retracted**.

**Added by this report:**

- 「hybrid 分布拟合更好」as a general statement — **72 % of the 2k WS-21 gap is one feature**
  (`bid_volume`), and **9 of 21 features are worse**.
- Anything about the pmatch arm — **the pre-registered comparison does not exist yet**; the
  only bench on disk is step 2956, N=768, on a **different index set**.
- Any attribution of the full `before-window` shortfall to model capability — **88.18 %
  visibility ceiling**, and generated streams reference outside their window **35.7 %** of
  the time vs **11.82 %** in real data.

---

## §10 Answers（结果记录；红字 = `\answer{}`，出处紧跟）

> 速览：所有已落地结论按时间收在这里。**这些是记录不是指令。**

<span style="color:blue">\question{ At 500-message context, is the hybrid's reference-recall advantage capability or convergence speed? }</span>

<span style="color:red">\answer{ 2026-08-12 **Convergence speed.** Δ by age at step 12000 is +5.05…+16.38 pp (post-PE-fix) but at step 32001 is +0.17…−2.03 pp — erased and slightly reversed. hybrid@12000's L1 exact 68.91 % sits between baseline's 12k (63.46 %) and 32k (74.89 %) values. Grade A (2–3 seeds, two step counts, six bins). Source: `STAGE_8_FINAL_32001.md` §3, `STAGE_5_SEED_REPLICATION.md`. }</span>

<span style="color:blue">\question{ Which hybrid advantages survive full training at 500 ctx? }</span>

<span style="color:red">\answer{ 2026-08-12 **LOB-Bench and CE, not recall.** WS-21 −11.7 %→−11.0 %, L1-21 −6.0 %→−9.2 %, CE −0.87 %→−0.79 % across 12k→32k; both seeds same sign. Grade A. Source: `STAGE_8_FINAL_32001.md` §4. }</span>

<span style="color:blue">\question{ Does the 500-ctx verdict survive at 2,000-message context? }</span>

<span style="color:red">\answer{ 2026-08-13 **No — it flips.** At 2k the advantage amplifies monotonically with reference distance: +3.6701 pp (1–10) → +13.7115 (101–250) → +21.5427 (251–500) → **+24.7049 pp (501–1000)**; overall 72.9699 % → 79.3593 % (+6.3894 pp). Same-token-budget comparison (both 26.624 B). Grade A for the monotone shape (8 buckets, 21 pp span), B for magnitudes (single seed). Source: `CTX2K_FINAL.md` §1, recomputed from `refer_success_*.json`. }</span>

<span style="color:blue">\question{ Is the 2k LOB-Bench advantage a general distribution-fit improvement? }</span>

<span style="color:red">\answer{ 2026-08-14 **No — it is concentrated in one feature.** `bid_volume` alone (0.352191→0.113593) contributes −0.01136 of the −0.01582 total WS-21 gap ≈ **72 %**; `log_time_to_cancel` adds ~21 %; **9 of 21 features are worse** under hybrid. This is derivable from `summary.json` and appears in **no** existing `.md`. It downgrades "hybrid fits the distribution better" from a general claim to a two-feature claim. Source: this report §4.3. }</span>

<span style="color:blue">\question{ Is the +24.7 pp effect the attention mechanism or the +5.43 % extra parameters? }</span>

<span style="color:red">\answer{ 2026-08-14 **UNDECIDED — arm in flight.** Indirect rebuttal is strong: parameter count does not vary with sequence length, so the same +5.43 % produced a **loss** of 0.60 % at 500 ctx; capacity cannot explain a sign flip with length. Direct test: param-matched arm at `d_ff`=1135, measured **33,609,998** params (**441 FEWER** than baseline, −0.0013 %), CPU-predicted and hardware-verified identically. Reading rules pre-registered in §6.3. Currently 4,479/6,400. The only pmatch bench on disk (step 2956, N=768, index `434917a0…`) is **not** the pre-registered comparison. Source: §6. }</span>

<span style="color:blue">\question{ Why did the checkpoint path deadlock every 900 seconds? }</span>

<span style="color:red">\answer{ 2026-08-13 **`broadcast_one_to_all` builds a pure inter-node NCCL clique.** Compile-time, GPU-free evidence (`code/check_replica_groups.py`, `--xla_force_host_platform_device_count=16`): training gradient AllReduce `replica_groups=[1,16]<=[16]`; checkpoint broadcast psum `[4,4]<=[4,4]T(1,0)` — group k = {k, k+4, k+8, k+12}, one card per node, **100 % Slingshot, 0 % NVLink**; shared kinds 0. `_reshard_for_ckpt`'s own comment says the host round-trip exists to avoid exactly this. Fixed by `CKPT_RESHARD_MODE`, default `none`. Also explains why the 2-node reproduction never triggered: the group degenerates to `[4,2]`. A regex that only recognised the old `replica_groups={{0,1},{2,3}}` form nearly falsified the correct hypothesis. Source: `CKPT_DEADLOCK_ROOT_CAUSE.md`. }</span>

<span style="color:blue">\question{ Can this line reach 4,000-message context by buying memory tricks? }</span>

<span style="color:red">\answer{ 2026-08-13 **No — sequence parallelism is required.** Fit peak(n) = 0.42 + 0.03524·n GB gives 2,000 msgs → 70.9 GB (82.9 % of 85.5); 4,000 → 134–142 GB = **1.57–1.66×**, impossible at bsz=1. `remat` is a **silent no-op on pure mamba3**: `remat=1` and `remat=0` give byte-identical peaks (18.02 / 35.71 GB) and OOM at the same point, because `init_Mamba3SSM` has no `remat` in its signature — `build_backbone` passes it only to attention-class factories. SP4 design + parity tests exist (`SEQPAR_DESIGN.md`, commit `a69bf05`); target is **4,096** not 4,000 because nc = n_msgs×26/64 must be divisible by D=4. Source: `CTX_4K_FEASIBILITY.md`, `SEQPAR_DESIGN.md`. }</span>

---

## Appendix A — Configuration Registry

\KL{任何重跑需要的数字都在这里，值查证自代码与 checkpoint 元数据}

**Verified from checkpoint metadata**（`<ckpt>/<step>/metadata/metadata` → `config`）:

| | baseline 2k | hybrid faithful 2k | hybrid pmatch 2k |
|---|---|---|---|
| checkpoint dir | `j6000409_je7cvor0_6000409`（4462）、`j6000409_gjnf0e03_6000409`（6265） | `j5998835_44awyydg_5998835`（4555…6258） | `j6011444_dbjkbfh5_6011444`（4479） |
| `architecture` | `mamba3` | `hybrid_mamba3` | `hybrid_mamba3` |
| **`hybrid_attn_d_ff`** | 0 | **0 → default 4·H = 2560** | **1135** |
| `hybrid_attn_heads` | 10 | 10 | 10 |
| `d_model` / `n_layers` | 640 / 6 | 640 / 6 | 640 / 6 |
| `msg_seq_len` | 2000 | 2000 | 2000 |
| `micro_bsz` / `process_count` / `grad_accum_steps` | 1 / 4 / **5** | 1 / 4 / **5** | 1 / 4 / **5** |

\KL{`hybrid_attn_d_ff = 0` 在元数据里意思是「没显式设，走 registry 默认」，所以 1135 是唯一被显式记下的值}

**Effective-batch identity**: `micro_bsz × GPU/node × nodes × K = 1 × 4 × 4 × 5 = 80`, i.e.
**nodes × K = 20**. Changing node count without changing K changes the experiment. Valid
configs: 2×K10, 4×K5, 5×K4, 10×K2, 20×K1.

**Optimizer**: Muon lr=0.01, ssm_lr=8e-4, wd=0.005, warmup 1 % → cosine over `COSINE_STEPS`.
**Data**: `lob_preproc_sp500_squashfs`, 48 months 2022-01…2025-12 × 8 tickers
（GOOG, AAPL, NVDA, AMZN, META, TSLA, MSFT, AMD）. **Encoding**: 26tok fixed-length,
vocab 2112. **Held-out**: GOOG 2026-01.

## Appendix B — Traps Recorded, So They Are Not Re-Paid

| trap | symptom | fix |
|---|---|---|
| 「16 BSZ」in logs | it is **per process**, not global — `--ntasks-per-node=1` means 1 process per node managing 4 GPUs, so per-process = 4×4 = 16 and global = 64 | read the `[bsz]` line, not the raw number |
| subset ≠ full pool | M1's subset/full-pool ratio = **1.1445**, so mixing 0.20714 with 0.23708 manufactures a **14 % fake gap** | never mix pools |
| two eval pools | `0c41de51…` (226,002 seqs) vs `4909799c…`; intersection **1.5 %** | check the sha |
| two arms, one output dir | under attach `SLURM_JOB_ID` is the *allocation* id, so both arms wrote the same dir and files named by sample index **overwrote each other** — LOB-Bench scored the mixture normally | path now includes `${ARM_ID}` and `$$`; contaminated run preserved as `CONTAMINATED_bench_20260812T092746Z_two_arms_same_dir` |
| `held` ≠ busy | four `--world_size=1` processes each held GPU0 (86.2 GB) while GPU1–3 held only 0.6 GB contexts — **12 of 16 cards were genuinely free** | card-level gates: `BENCH_GPU_OFFSET`, `BENCH_WORLD_SIZE`, 4096 MiB threshold |
| Elastic Resume rescales `state.step` | changing node count **or** K triggers `train.py:249`; 2956 became ~1478 and an entire 1h23m run was voided **with no error** | resume only at the same node count |
| `CURTAIL` derived from the restore step | `steps_per_epoch` shrinks as training progresses; at `_rs`=4479 it gives `start_epoch=1` and `range(1,1)` is **empty** → trains 0 steps and returns **exit 0** | `CURTAIL = COSINE_STEPS × K` fixed; `[curtail]` self-check aborts if `start_epoch ≥ 1` |
| `NO_AUTO_RESUME_DEPTH=99` | prints "Reached 20 consecutive auto-resumes" — a **deliberate sentinel** misreported as a fault counter | message now distinguishes the two cases |
