# Step 0 Inventory: what exists for the plasticity / CPT experiments

> Status: partial — filesystem- and memory-verifiable facts recorded; three items need the user or a measurement run. Last updated 2026-08-27.

## 中文速览

- 代码：sigma-0 主 checkout 在 `/projects/public/u6gb/sigma-0`（remote `github.com/KangOxford/sigma-0`，HEAD 78908b0）；主线训练配置 `configs/train/mamba3_sp500.yaml`（mamba3，75m preset，26tok，32 节点）。
- 数据：`data/SP500-2022-2025 -> /projects/public/s5e/quant_team/lob_preproc_sp500_squashfs`，即可用窗口 **2022-01 至 2025-12**（SquashFS 月分片）。
- **对 plan 的修正**：深研草案里的 COVID 切片（2020-03）在数据窗口外，不可用。实际压力切片：**2024-08-05 波动冲击**（在窗内，首选）与 2025-04 关税窗口（次选）。base 窗口相应改为 2022–2023（或 2022 至 2024-07），探针切片取其后。
- 检查点：仓库约定 per-job 输出目录 + `latest_checkpoint.json` breadcrumb。R1 时代 8M（wandb zkrtl2ef）/ 78M（pw8u0edj）检查点根目录未知——**问用户最快**（上会话已定性）。Step 2 需要的「同一条长 run 的早/晚检查点」具体取哪条 run，待确认。
- 吞吐参照：mamba3 0.565 s/step（wandb wqgghoyj）、ttt_linear 1.46 s/step（c86ghhsn）；每步 token 数未记录，预算换算前要补一次测量。

## 1. Code

| Item | Value |
|---|---|
| Main checkout | `/projects/public/u6gb/sigma-0` |
| Remote | `https://github.com/KangOxford/sigma-0.git` (branch main, HEAD 78908b0 at inventory time) |
| Main training entry | `run/base_model/train_base_model.py` via `configs/train/mamba3_sp500.yaml` |
| Main line config | mamba3 backbone, `MODEL_PRESET=75m`, `TOKEN_MODE=26tok`, 32 nodes, 40 epochs |
| Self-train line | `configs/train/selftrain*.yaml` family, wandb project `oxford-lob/sigma0-selftrain` |
| Probe module (this PR) | `tasks/continual_learning/code/plasticity_probes.py` (md-u6gb repo), to be wired into the sigma-0 training loop as a follow-up |

## 2. Data

| Item | Value |
|---|---|
| SP500 shards | `sigma-0/data/SP500-2022-2025` → `/projects/public/s5e/quant_team/lob_preproc_sp500_squashfs` (monthly SquashFS shards) |
| Usable window | **2022-01 .. 2025-12** |
| Self-train slice | `sigma-0/data/selftrain_b43/` (GOOG-centered; used by PR#33/34 diagnostics) |

### Consequence for the experiment ladder (supersedes the draft slices in PLAN.md §3)

The deep-research draft assumed a 2015–2019 base window with a COVID-2020 stress slice. The tokenized data starts 2022-01, so:

- **Stress slice, primary**: the 2024-08-05 volatility spike plus the following weeks — inside the window, with a natural "trained through 2024-07" cut.
- **Stress slice, secondary**: the 2025-04 tariff window, if those months are shard-complete.
- **Base window**: 2022-01 .. 2023-12 (or .. 2024-07), giving roughly two years of pre-training data before the probe slice.
- The multi-year cyclic onset study (PLAN Step 5) has at most 4 calendar years to cycle over; regime-sliced (volatility-tiered) task instances are the more realistic axis than calendar years at this window length.

## 3. Checkpoints

| Item | Value |
|---|---|
| Convention | per-job output dirs, `latest_checkpoint.json` breadcrumb written atomically by rank 0 (repo rule; no directory listing needed) |
| R1-era 8M | wandb `zkrtl2ef` — **checkpoint root unknown, ask the user** |
| R1-era 78M | wandb `pw8u0edj` — **checkpoint root unknown, ask the user** |
| Early/late pair for Step 2 | needs one long run whose intermediate checkpoints were retained across a wide step range; candidate is the mamba3 SP500 main line — which job id retained its chain is the open question |

## 4. Throughput reference points (for budget arithmetic)

| Backbone | s/step | Source |
|---|---|---|
| mamba3 (75m-class) | 0.565 | wandb `wqgghoyj` |
| ttt_linear variant | 1.46 | wandb `c86ghhsn` (0.39x of mamba3) |

Tokens per step were not recorded alongside these; one measured number (batch x seq len x accumulation at the production setting) is needed before converting the Step 2/3 token budgets into wall-clock.

## 5. Open items (blocking which step)

| # | Item | Blocks | How to resolve |
|---|---|---|---|
| 1 | Checkpoint roots for R1-era 8M/78M and the main-line mamba3 chain | Step 2 | ask the user (fastest, per prior session) |
| 2 | Which long run retained an early/late checkpoint pair wide enough apart | Step 2 | user, or wandb run list + breadcrumb reads once roots are known |
| 3 | Tokens/step at production setting | Steps 2–3 budgets | read from a recent run log or one smoke step |
| 4 | 2025 shard completeness (for the secondary slice) | secondary slice only | check shard index for 2025-04.. months when needed |
