# Continual Learning for sigma-0: Plasticity Measurement and Continual Pre-Training Plan

> Task dir: `tasks/continual_learning/` · Source research: `deep-reseach.md` (two-pass deep research, 2026-08-26)
> Status: PLAN — experiments not yet started. This file is the single source of truth for this line of work.

## 中文速览

- 研究背景：σ0（LOB 基础模型）要服役多年，必须回答两个问题——模型会不会「变僵」（可塑性丧失），以及无标注的市场漂移下何时、如何继续训练。
- 深研结论（论文级，两轮）：可塑性丧失在 LLM 上成立（Zyphra：5M–314M 全部中招，onset ∝ P^0.83，**平稳**数据混合也发生，规模只推迟不免疫）；「没有机制理论」已过时一半（不变流形定理 + optimization readiness 预测器 + CPT 闭式损失律）；SSM/GDN 架构与 LOB 数据上**均无先例**——σ0 补这两格就是第一。
- 关键设计修正：可塑性比较必须用**同一模型的早 vs 晚检查点**（同探针预算比 AUC），不用 fresh-vs-continued——预训练模型赢 scratch 只证明表征迁移，判「可塑性完好」几乎必然假报。scratch 只作下界参考。
- 双坐标汇报纪律：旧窗 NLL（稳定性/遗忘）与探针 AUC（可塑性）必须同时报，只报一个会重演「单一坐标说谎」的教训。
- 工作序（五步，成本递增）：① 盘点现有检查点与数据窗口 → ② 诊断仪表落地（本 PR 附代码）→ ③ 早 vs 晚检查点探针实验 → ④ CPT 试点拟合 replay 比例 × rewarm 学习率 → ⑤ 多规模 onset 律（34M–617M）。
- 本 PR 先落地：本 PLAN + `code/plasticity_probes.py`（五个诊断量：dormant 比例、有效秩、权重/梯度范数、optimization readiness、top Hessian 特征值，纯 numpy、框架无关、带测试）。

---

## 1. Why this line of work

sigma-0 is a foundation model of NASDAQ limit-order-book message flow that must stay in service across years of non-stationary markets. Two failure modes threaten any such deployment:

1. **Loss of plasticity ("model aging")**: the network not only forgets, it stiffens — dormant units, representation collapse, gradients that stop carrying signal — until it can neither learn new structure nor discard old structure.
2. **Unlabeled regime drift**: markets never announce a task switch. Retraining cadence, learning-rate policy, and replay mix must be driven by measured distribution signals, not by task labels that do not exist.

Two passes of deep research (first over X/practitioner discourse, second over the primary papers) are archived in `deep-reseach.md`. This plan extracts the decisions.

## 2. What the research established

### 2.1 Plasticity loss is real for pre-trained sequence models, not just RL

- Hernandez-Garcia, Figliolia, Millidge (Zyphra, arXiv 2606.24752): GPT-style pre-norm Transformers, 5M–314M non-embedding params, multilingual next-token training. Plasticity measured as validation-loss AUC of a probing run (fixed 5B-token budget on held-out Vietnamese) from periodic checkpoints. Every size eventually loses plasticity; onset follows T ≈ 1.3e-5 · P^0.8269 (sublinear → scale delays, does not immunize). Critically, the effect appears **also under a stationary mixture** of all languages, so non-stationarity accelerates but is not required. "Scale alone cannot save us."
- Springer et al. (arXiv 2503.19206, ICML 2025): catastrophic overtraining — OLMo-1B pre-trained on 3T tokens fine-tunes *worse* than its 2.3T-token counterpart; progressive sensitivity formalized in a two-layer linear model.
- Lampinen (essay, 2026-05): scale plus pre-training substantially *reduce* interference and plasticity loss. Coheres with the above once "delayed" is separated from "defeated."

### 2.2 The "no mechanistic theory" claim is now half-outdated

- Joudaki et al. (arXiv 2510.00304): dynamical-systems definition — loss-of-plasticity manifolds (frozen-unit, cloned-unit / equitable partitions) proven invariant under GD, SGD, momentum, Adam; only symmetry-breaking perturbations (noisy SGD, dropout) escape. Simplicity bias and rank compression, which help static generalization, *steer networks onto* these manifolds.
- Wang et al. (arXiv 2605.09044): representation-rank and NTK-rank diagnostics can provably fail to predict trainability; proposes **optimization readiness** (gradient strength × gradient reliability), which lower-bounds one-step optimization gain and is cheap (a few forward/backward passes per checkpoint).
- Wang, Tissue et al. (arXiv 2505.07796, ICML 2025): closed-form continual-pre-training loss law — CPT loss decomposes into an LR-annealing term and a power-law distribution-shift term; replay ratio enters through the shift term; validated 106M–1.7B.
- Still missing (verified gaps): any plasticity measurement in SSM / linear-attention / GDN architectures, and any controlled LOB continual-learning study. **sigma-0 filling either gap is a first.**

### 2.3 The decisive design correction

The naive comparison — "continued pre-trained model vs fresh random init, matched compute; plasticity ABSENT if continued wins" — is broken. A pre-trained model beats scratch through representation transfer even while its plasticity decays, so that rule almost always reports ABSENT. The correct probe (following Zyphra):

- Compare **an early checkpoint θ_early and a late checkpoint θ_late of the same training run**, each given an identical fixed-budget adaptation on the same held-out slice, and compare validation-NLL AUC.
- PRESENT: AUC(θ_late) > AUC(θ_early) with CI excluding equality, plus at least one co-moving diagnostic (dormant fraction up, effective rank down, optimization readiness down).
- Scratch runs are kept only as a lower-bound reference; R(late/scratch) < 1 would be the far stronger "network nearly dead" statement, not the default expectation.

### 2.4 Reporting discipline

Every CPT or adaptation stage reports **both coordinates**: old-window validation NLL (stability / forgetting) and probe AUC or R_steps (plasticity). One coordinate alone can look healthy while the other collapses.

## 3. The experiment ladder

Ordered by cost; each step gates the next. No step assumes hardware or checkpoints that have not been inventoried in step 0.

### Step 0 — Inventory (no GPU)

List which sigma-0 checkpoints exist (run, size, step range, data window covered), which NASDAQ windows are tokenized and ready, and measured tokens/sec for the current model size. Output: a short table in `results/INVENTORY.md`. Everything below is parameterized by it.

### Step 1 — Diagnostics instrumentation (this PR, no GPU)

Land `code/plasticity_probes.py`: framework-agnostic implementations of

| Probe | Definition | Cost |
|---|---|---|
| dormant fraction | ReDo-style: unit-mean absolute activation, normalized by layer mean, below eps = 0.01 | free at eval |
| effective rank (Renyi-2) | er2(M) = (tr M)^2 / \|\|M\|\|_F^2 on the feature covariance | one small matmul |
| weight / gradient norms | global L2, non-embedding | free |
| optimization readiness | \|\|mean g\|\|^4 / mean \|\|g_i\|\|^2 over micro-batch gradients (strength × reliability; reconstruction of arXiv 2605.09044's descriptor) | K extra backward passes |
| top Hessian eigenvalue | power iteration over an injected HVP callable (Pearlmutter) | 10–20 HVPs, optional |

plus unit tests runnable on CPU. Wiring into the sigma-0 training loop is a follow-up commit; the target is that **every future long run logs these by default**, so plasticity evidence accumulates for free.

### Step 2 — Early-vs-late checkpoint probe (first GPU experiment)

- Take one existing long sigma-0 run; pick θ_early and θ_late checkpoints separated by as many tokens as the run allows.
- Fixed-budget adaptation of copies of both (identical tokens, batch, schedule, seeds) on a held-out later time slice; log validation NLL every fixed interval plus all Step-1 diagnostics.
- ≥ 5 seeds per group before any claim; 95% bootstrap CI on AUC difference and R_steps; declare a difference only if the CI excludes equality.
- Stress slices when data allows: the COVID window (2020-02..04) and the 2024-08 volatility spike (probe from a checkpoint trained through 2024-07 only — no leakage). **Superseded by Step 0**: tokenized data starts 2022-01 (`results/INVENTORY.md` §2), so the primary slice is 2024-08, the secondary 2025-04, and the base window 2022-01..2024-07.
- Decision: PRESENT / ABSENT / inconclusive per §2.3. If inconclusive, extend token budget before adding mechanisms.

### Step 3 — CPT pilot: fit replay × rewarm before committing a big run

- D_cpt = (1−ρ)·D_new + ρ·D_old, old data stratified over time (not just the last month).
- Grid: peak LR ∈ {0.3, 0.5} × pre-training peak, ρ ∈ {0.05, 0.10, 0.25}; short pilots of 1–2B tokens.
- Rewarm is mandatory: short warmup (2–5% of stage steps) up to the chosen peak, then WSD or cosine with a **non-zero tail**; reset Adam moments at each stage boundary. A checkpoint whose LR annealed to ~0 and is continued without rewarm *looks* dead without being dead — that failure mode must not contaminate plasticity claims.
- Fit the arXiv 2505.07796 law's shift term from the pilots; choose the production (peak LR, ρ) to hit target new-window NLL while bounding the old-window NLL increase.
- Stage layout by calendar, not task ids: θ0 (base years) → CPT1 → CPT2 → ... with shock windows (2020-03, 2024-08) reported separately, never averaged into a calm-year replay bucket.
- Optional stabilizer, off by default: teacher-KL to the previous stage checkpoint, λ starting at 0, raised only if old-window NLL climbs beyond tolerance.

### Step 4 — Mitigations, in cost order (only if Step 2 finds decay)

1. Weight decay up (cheapest knob, first).
2. Spectral regularization toward sigma_max ≈ 1 (Lewandowski et al., ~14% step-time cost).
3. Local ReDo-style resets only where dormant fraction climbs; never periodic whole-network resets on a pre-trained model.
4. AltNet-style dual-network swaps and continual backprop reserved for a genuinely online setting.

### Step 5 — Multi-size onset law (the publishable first)

Cyclic year/regime schedule over 34M / 100M / 300M-class sigma-0 models, probe slice held out (a distinctive shock window), fit T = c · P^k for the LOB/SSM setting. This would be the first plasticity onset law for state-space / linear-attention models — the cell the verification table marks NOT FOUND. Only started after Step 2 produces a verdict-grade readout at one size.

## 4. Non-goals and traps (pre-registered)

- No per-request weight updates in serving; adaptation happens between stages.
- No regime-classifier-plus-EWC pipeline; drift signals (validation-NLL band, volatility state, plasticity probe) drive *stage-level* decisions only.
- Sliding-window fine-tuning on the most recent month is not CPT and will not be labeled as such.
- Fresh-vs-continued alone never decides a plasticity claim (§2.3).
- SEC Rule 612 half-penny tick: adopted 2024-09-18, compliance now first business day of Nov 2026 (extended by the Oct 2025 exemptive order) — slice boundaries must not assume the 2025 date.
- Nested Learning / HOPE, TTT layers, Titans: narrative-relevant, but not the backbone for the first measurement; no streaming-plasticity evidence yet.

## 5. Compute envelope (order of magnitude, to be refined by Step 0)

- Step 2 at the ~100M scale: two groups × 5 seeds × (1–2B tokens each) ≈ tens of short runs, each fitting on 1 node; embarrassingly parallel; attach-first.
- Step 3 pilots: 6 grid cells × 1–2B tokens at one size.
- Step 5 is the only multi-thousand-GPU-hour item and is explicitly deferred.

## 6. Deliverables checklist

- [x] PLAN.md (this file)
- [x] `code/plasticity_probes.py` + `code/test_plasticity_probes.py` (Step 1, this PR; 13 CPU tests pass)
- [x] `results/INVENTORY.md` (Step 0; partial — 4 open items listed there, checkpoint roots need the user)
- [ ] Probe wiring into the sigma-0 training loop (follow-up)
- [ ] Step 2 readout: AUC(θ_late) vs AUC(θ_early), CI, diagnostics — reported on the PR as commits + comment updates
