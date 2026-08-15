## 0. TL;DR（决策摘要）

本 plan 操作化《agentic trading with world model》页上的问题形式化（⟨Curves, Rules⟩ 主版 + ⟨distribution, Rules⟩ 姊妹版），补上三阶段流水线缺失的闭环阶段：Pretrain (AR CE) → Post-train (DFM/GRPO, open-loop) → **Agentic Trading（闭环、有市场反馈）**。任务是 Market Making；用 frozen **78M Mamba3** 作 world model（市场模拟器），确定性 match engine 撮合，**LLM-as-optimizer (OPRO)** 搜索 rule-based 做市策略。明确：这不是 RL，也不是 ES/EGGROLL，没有策略梯度，没有种群搜索。

| Item | Decision | Note |
|---|---|---|
| Base / world model | 78M Mamba3 SISO (d_model=1024, L=6, 78,539,423 params) | only trained model in the 50-80M range; the only one with a full LOBbench eval |
| Checkpoint | step 46050 (Muon, run pw8u0edj, job 3417629) | lowest documented test CE; fully scored. s67840 exists but is un-benched |
| Policy | rule-based MM code (at-touch binary, OR AS reservation price + skew) | hard constraints pin the feasible set |
| World-model role | frozen ckpt: background order flow + future mid distribution (MC roll-out) | observation dist; it must NEVER generate the fills of your own quotes |
| Simulator | JAX match engine, price-time priority, deterministic | settles fills / inventory / book / PnL |
| Optimizer | LLM-as-optimizer (OPRO), gradient-free | NOT RL, NOT ES/EGGROLL |
| Objective | window PnL; inventory risk endogenous via end-window mark-to-market | win rate / PL ratio / drawdown are metrics, reported not optimized |
| Significance | real-data backtest is the final judge; Δ must pass 2σ (≈1.96·sqrt(2)·σ) | MC sim systematically overestimates profit |

---

## 1. Base Model 选型 + Model Card

用户要求在 R1 Mamba3 worktrees 里挑一个 50-80M、效果好的模型作 base model（提示"例如 d_model 512 之类的"）。实测候选里**只有 78M 落在 50-80M 区间**，且只有它做过完整 LOBbench 评测，故选 78M。

| Label | Params | d_model | n_layers | In 50-80M? | LOBbench scored? |
|---|---|---|---|---|---|
| 8M | 8,099,567 | 256 | 6 | no | RoPE-ablation only |
| 14M | 14,445,623 | 384 | 6 | no | ablation only |
| 23M | 22,949,247 | 512 | 6 | no | scaling-law fit |
| 34M | 33,610,439 | 640 | 6 | no | scaling-law fit |
| 46M | 46,429,199 | 768 | 6 | no (just below 50M) | scaling-law fit |
| 78M | 78,539,423 | 1024 | 6 | yes | yes (full h250 + h500) |

Note: d_model=512 maps to the 23M model, which is below the stated 50-80M range. The 78M (d_model=1024) is the only model inside 50-80M and the only one with a complete LOBbench evaluation, so it is the base model. If a model closer to the d_model=512 hint is preferred, the nearest alternative is the 46M (d_model=768), but it sits just below 50M and only has a scaling-law-fit eval.

**78M config + verified training facts**

| Field | Value |
|---|---|
| Architecture | Mamba3 SISO (expand=2, headdim=64, d_state=128, rope_fraction=0.5) |
| Optimizer | Muon (kernel LR=0.01, muP SSM-LR scaling) |
| Token mode | 26tok (26 tokens / message), msg_seq_len=500 → 13,000 tokens |
| Training jobs | 3417629 (primary, pw8u0edj) → 3420184 (resume, ra5i8nkd), steps 0→67,840 |
| Scaling-law fit | Loss = 0.515 + 1.70e11·N^(-1.36) + 3.05e11·D^(-1.12), R2 = 0.975 |
| Checkpoint (chosen) | exp_R1_Mamba3/checkpoints/j3417629_pw8u0edj_3417629/46050/ (verified EXISTS) |
| Test CE @ s46050 (GOOG) | 0.4482, token acc 0.9080 |

**78M real LOBbench (GOOG, n=3136, Pearson IC shown per house rule)**

| Setting | WS-21 | KS-21 | PIC | DirAcc | Sharpe |
|---|---|---|---|---|---|
| h250 (ctx250 → gen250) | 0.0438 | 0.090 | 0.142 | 0.5724 | 0.1308 |
| h500 (ctx500 → gen500) | 0.0539 | 0.0945 | 0.1042 | 0.5335 | 0.0705 |

Note: these are the verified 78M numbers from agent_outputs/adamw_vs_muon_lobbench.md. An earlier draft mistakenly reported the 8M RoPE-ablation scores (WS-21≈0.110) as the 78M result; that has been corrected. h250 has stronger directional signal (DirAcc 0.572, Sharpe 0.131) and is the more usable horizon for a 1-min MM window.

---

## 1.5 Why 78M is a sound world model for MM

The world model only needs to produce realistic background flow and a calibrated short-horizon mid distribution over a 1-min window. At h250 the 78M reaches DirAcc 0.572 and PIC 0.142, i.e. it carries genuine (if modest) short-horizon directional signal, and WS-21 0.044 means the per-level book distribution is close to real. That is exactly the regime MM operates in. The scaling-law page also shows a U-shaped realism curve where mid-scale models (~50-120M) generate the most market-realistic flow, so 78M is near the realism sweet spot rather than the over-smoothed large-model tail.

---

## 2. 问题形式化（operationalized）

**闭环 POMDP（与两个 formulation 页一致）**

State $s_t = \langle b_t, q_t, \tau, \mathcal{P}_{t:t+\tau} \rangle$: book state $b_t$, inventory $q_t$, time remaining $\tau$, and the world-model output over the remaining window. The fourth term is a single mid curve $\hat p_{t:t+\tau}$ in the ⟨Curves⟩ version, and a full predictive distribution $\mathcal{P}$ (direction prob / confidence / skew / tails) in the ⟨distribution⟩ version.

Action: at-touch binary {post bid, post ask, abstain} + smart cancel (⟨Curves⟩); OR AS reservation-price + spread + inventory-skew + signal-skew, clipped to tick (⟨distribution⟩).

Observation: closed-loop fills + new mid + new book. Inference conditions only on the orders that really happened, ignoring already-generated (simulated) ones. Latency constraint: wait 1s before quoting (inference on a 1-min window itself takes time).

**四部件闭环数据流**

LLM proposes a rule → policy posts quotes at $t$ → world model samples background flow over the remaining window → match engine matches (my quotes + background) by price-time priority → fills / new book / new mid → policy re-conditions on real observations → repeat to window end → window PnL (multi-seed / multi-roll-out average, past a 2σ noise floor; inventory marked-to-market at window end) → LLM reads the score + diagnostics → rewrites the rule → next round.

**核心因果点（必须守住）**: the world model learns $p(\text{market})$ (observational), but MM is an intervention $p(\text{market} \mid do(\text{my quotes}))$. Answering an interventional question with an observational model is systematically optimistic (adverse selection is under-counted). Therefore the generator must NOT produce the fills of your own orders; that job belongs to the deterministic match engine. Real-data backtest stays the final judge.

---

## 3. 系统架构 → 真实代码映射

| Component | Status | Path / entry point |
|---|---|---|
| ③ Match engine (simulator) | EXISTS (JAX, jittable) | JaxMARL-HFT/gymnax_exchange/jaxob/jorderbook.py (OrderBook, LobState, reset, process_order_array, get_best_bid/ask, get_L2_state); JaxOrderBookArrays.py (_match_against_bid/ask_orders, match_order, cancel_order) |
| ② World-model roll-out | EXISTS | LOBS5/lob/inference.py: generate(), generate_single_rollout(), generate_repeated_rollouts() (Monte Carlo, J roll-outs from same context = the distribution version's sampler); entry LOBS5/run_inference.py |
| Token decode + correction | EXISTS | LOBS5/lob/encoding_26tok.py: decode_msg(); inference.py: get_sim_msg() (validate/correct), construct_sim_msg() → (8,) message [type, side, qty, price, trade_id, order_id, time_s, time_ns] |
| Book → mid curve | EXISTS | inference.py: get_best_bid/ask → p_mid; l2_book_states; calculate_rollout_metrics() (mid returns) |
| Checkpoint load | EXISTS | LOBS5/lob/init_train.py: load_checkpoint(), load_metadata() |
| LOBbench scoring | EXISTS | lob_pipeline/lob_bench/run_bench.py; pipeline/run_lobbench_pipeline.sh; _integrated.batch |
| ① Rule-based MM policy | TO BUILD | new: lob/mm_policy.py (at-touch binary; AS reservation-price + inventory-skew + signal-skew); rule = code string the optimizer rewrites |
| Closed-loop orchestration | TO BUILD | new: lob/closed_loop_mm.py (background flow ↔ my quotes ↔ match engine ↔ re-condition; window loop; mark-to-market) |
| Inventory / PnL accounting | TO BUILD | new: lob/mm_accounting.py (fills → inventory path, realized PnL, end-window MTM) |
| ④ OPRO optimizer harness | TO BUILD | new: lob/opro_optimizer.py (meta-prompt = [rule code → fitness] sorted; Opus API proposes new rule; backfill history) |
| Fitness + diagnostics | TO BUILD | new: lob/mm_fitness.py (denoise mean+std+2σ; cross-scale Sharpe/bps; baseline-relative Δ; trajectory diagnostics: fill rate, adverse selection, inventory path, PnL attribution, per-regime, failure replays) |
| Real-data backtest | TO BUILD | new: lob/mm_backtest.py (replay real GOOG flow, same policy, no generator) |

Existing MM/RL/execution code: NONE integrated. JaxMARL-HFT has a gym-style multi-agent RL scaffold sharing the same OrderBook, useful only as a reference for the env loop. We do NOT reuse its RL training; per the formulation the optimizer is OPRO, not RL.

---

## 4. 分阶段可执行步骤（P0 → P5）

每个 phase 给出：要建的文件、怎么跑、通过判据（success gate）。GPU 步骤（world-model 生成）需 GPU；执行场所按当前 login-only 指令待确认（见风险 R5）。Match engine 是纯 JAX，可在 CPU/login 跑小规模。

**P0: Infra smoke（端到端打通一根线）**
- Build a 50-line script: load 78M ckpt (load_checkpoint) → take 500 real GOOG context msgs → generate_single_rollout for N=50 future msgs → feed each decoded (8,) msg into OrderBook.process_order_array → read p_mid trajectory.
- Run: tiny (N=50, 1 context) to verify decode → match-engine → mid all line up against inference.py's own l2_book_states.
- Gate: generated mid trajectory from our loop == inference.py's internal mid (bit-exact), and 0 decode/sim errors on the smoke sample.

**P1: Course AS baseline (rule-based), single window, single seed**
- Build lob/mm_policy.py with the course baseline: reservation price $r_t = m_t - q_t\,\gamma\,\sigma^2\,(T-t)$, optimal half-spread $\approx \gamma\sigma^2(T-t) + \frac{2}{\gamma}\ln(1+\frac{\gamma}{\kappa})$, clip to tick. Build lob/closed_loop_mm.py: at each 1s step, post AS quotes, let world model fill the remaining-window background flow, match, re-condition, mark-to-market at window end.
- Build lob/mm_accounting.py (inventory path, realized PnL).
- Gate: one full 1-min window runs end-to-end, produces a finite window PnL, inventory returns to/settles sanely; pure-AS PnL is the baseline number every later phase subtracts.

**P2: Distribution layer（在 AS 上叠加分布偏好 skew = 本任务的核心增量）**
- Use generate_repeated_rollouts (J roll-outs) to build the predictive distribution $\mathcal{P}$ (direction prob, confidence, skew, tails) over the remaining window.
- Extend mm_policy: $r_t = \underbrace{m_t - q_t\gamma\sigma^2(T-t)}_{\text{AS inventory skew}} + \underbrace{w\cdot g(\mathcal{P}_{t:t+\tau})}_{\text{signal skew (new)}}$, where $g$ maps the distribution to a directional offset (prob sets sign, confidence/skew sets magnitude) and $w$, depth are rule params.
- Also wire the ⟨Curves⟩ at-touch binary variant as a second policy class behind the same env.
- Gate: AS + distribution skew runs in closed loop; with a hand-set $g,w$ it changes fills/inventory vs pure AS in the expected direction.

**P3: OPRO LLM-optimizer loop**
- Build lob/opro_optimizer.py: meta-prompt = [history of (rule code → fitness)] sorted by score + the frozen hard constraints (AS structure, tick, inventory caps). Opus 4.8 proposes a new rule (code for $g,w$, depth, smart-cancel timing) → evaluate via P2 → backfill history → iterate.
- Optimizer input per round (formulation 4.5, option B): current rule code + aggregate fitness + readable trajectory diagnostics (fill rate, adverse selection, inventory path, PnL attribution, per-regime split, a few representative failure mini-replays). NOT raw tick trajectories.
- Gate: over K≈20 OPRO rounds, fitness (baseline-relative Δ vs pure AS) trends up past the 2σ floor on a fixed eval set; no constraint violations in proposed rules.

**P4: Multi-seed × multi-init × regime-stratified + normalization**
- Make fitness a proper random-variable estimate: average over multi roll-out × multi seed × multi orderbook-init-state, stratified by regime (open / close / high-vol / high-liquidity). Report mean + std + t-test/CI, past 2σ, never best-step.
- Normalize: denoise (mean+std+2σ) × cross-scale (Sharpe or bps-of-notional; order size aligned to a % of the ticker's own volume) × baseline-relative (Δ vs AS), with per-regime z-score when pooling.
- Gate: fitness is stable across re-seeds (CI overlaps), per-regime breakdown produced; no single regime dominates the pooled score.

**P5: Real-data backtest + significance + ablation**
- Build lob/mm_backtest.py: replay real held-out GOOG flow (Jan 2026 test set), run the OPRO-found rule with NO generator in the loop (generator only fed the optimizer; the judge is real data).
- Required ablation: pure AS (course baseline) vs AS + distribution skew (this task) vs ⟨Curves⟩ at-touch version.
- Gate: $\widehat\Delta = \widehat X_{\text{method}} - \widehat X_{\text{AS}}$ on real-data Sharpe/PnL passes the 2σ threshold (≈1.96·sqrt(2)·σ) with multi-seed CI; cross-ticker check on the 8-ticker set.

---

## 5. 实验设计

| Axis | Setting |
|---|---|
| Inventory level | sweep several levels; no explicit λ·Var penalty (risk is endogenous via end-window MTM) |
| Order size | aligned to a % of each ticker's own trading volume |
| Stochasticity | multi roll-out × multi seed × multi orderbook-init-state |
| Regime | stratified sampling + per-regime evaluation (open / close / high-vol / high-liquidity) |
| Tickers | GOOG primary; generalization check on the 8-ticker set (GOOG, AAPL, NVDA, AMZN, META, TSLA, MSFT, AMD) |
| Baselines / ablation | pure AS (course baseline) ; AS + distribution skew ; ⟨Curves⟩ at-touch ; (optional) closed-form AS skew as sanity init |

---

## 6. 评测 & 成功判据

Primary metric: Sharpe / PnL, with real-data backtest as ground truth (MC sim overestimates, per the agentic-trading slides). $\widehat\Delta = \widehat X_{\text{method}} - \widehat X_{\text{baseline}}$ is a random variable and must pass a 2σ threshold (≈1.96·sqrt(2)·σ) on multi-seed t-test / CI. Never report best-step. Report dense metric and the real holdout endpoint together. Win rate, PL ratio, max drawdown are reported as metrics (per regime) but are NOT optimization targets; the target stays single PnL.

---

## 7. 风险登记册（plan 必须正面回应）

| ID | Risk | Mitigation |
|---|---|---|
| R1 | DA saturation: MM alpha mostly set in pretraining (Problem 7) | this stage optimizes signal-usage (rules + how to use the full distribution), orthogonal to accuracy. "Framework beats accuracy": same DA=66%, AS Sharpe +0.52 (n.s.) vs HJB@Touch +10.7 (~20x), all from framework. Plus closed-loop reactivity alpha is orthogonal to accuracy |
| R2 | Observation vs intervention: generator is optimistic, adverse selection under-counted | never let the generator produce your own fills (deterministic match engine does that); real-data backtest is the final judge |
| R3 | OPRO stalls in high-noise black box | hard constraints (AS structure, tick, inventory caps) shrink the feasible set; multi-seed + 2σ floor denoise the fitness the LLM reads |
| R4 | World-model / simulator fidelity | monitor world-model realism separately (mid is not enough: spread / imbalance / queue dynamics); real-data backtest gates every claim |
| R5 | Compute venue: world-model generation needs a GPU, but the standing directive is login-only | match engine + accounting + OPRO bookkeeping are pure CPU/login; only on-demand generation needs GPU. Option: pre-cache rollouts (the pipeline already has 600+ generated runs) for offline-evaluable rule classes; confirm GPU venue before P2-scale runs |

---

## 8. 里程碑 / 交付物

| Phase | Deliverable | Gate |
|---|---|---|
| P0 | smoke script; loop == inference.py mid | bit-exact mid, 0 errors |
| P1 | mm_policy (AS), closed_loop_mm, mm_accounting | one window → finite PnL baseline |
| P2 | distribution-skew policy via generate_repeated_rollouts | skew changes fills vs AS as expected |
| P3 | opro_optimizer + mm_fitness | fitness Δ trends up past 2σ over ~20 rounds |
| P4 | multi-seed/init/regime eval + normalization | stable CI, per-regime breakdown |
| P5 | mm_backtest + ablation table | real-data Δ passes 2σ; 8-ticker check |

---

Source of truth: 问题形式化页《[MAIN PAGE] ⟨Curves, Rules⟩》和《⟨distribution, Rules⟩》；实验结果见《find all the inferences and scores》。Base-model facts verified against exp_R1_Mamba3/scaling_law_runs.md and agent_outputs/adamw_vs_muon_lobbench.md.
