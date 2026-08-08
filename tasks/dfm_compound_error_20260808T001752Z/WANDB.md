# W&B 索引

UTC 2026-08-08T08:40Z

---

## ⚠️ 先说清楚：DFM 后训练**当时没有接 W&B**

`post_training/dfm/tools/dfm_train_worker.py` 里**零处** wandb 引用，
只逐步写 JSONL。这违反了工作区的长期约定
（`USE_WANDB=True` + `WANDB_MODE=online` + entity `oxford-lob`，2026-07-17 定）。

下面 Stage 2A/2B 的 run 全部是**事后回填**（`eval/backfill_wandb.py`），
每个 run 的 config 里带 `backfilled_from_jsonl`、notes 里写明「did NOT stream live」。

**为什么要区分**：live run 证明作业在那个 wall-clock 上确实在跑；
回填只证明**文件这么说**。因此**没有伪造时间戳**——step 按自身序号写入，
原始 node / GPU / grid cell 存进 config，保持可追溯。

---

## 1. 预训练基线（本研究一切度量的参照物）

| 项 | 值 |
|---|---|
| **W&B** | **https://wandb.ai/oxford-lob/sigma0-selftrain/runs/b30675li** |
| job | 5705912 |
| step | 69378 |
| checkpoint | `/lus/lfs1aip2/projects/public/u6gb/sigma-0/checkpoints_selftrain/j5705912_b30675li_5705912` |
| 参数量 | 78,539,423 |
| 实测因果 CE（本会话复算） | **0.6827**（参照 0.4475，闸门阈值 1.5 ✅） |

---

## 2. DFM 后训练 Stage 2B —— 四条臂（回填）

**项目**：https://wandb.ai/oxford-lob/sigma0-dfm-posttrain
**group**：`backfill-20260801`

| 臂 | steps | 说明 | W&B |
|---|---|---|---|
| **`2b_pre5e-5`** | 343 | **论文学习率，主线**；本研究用的 checkpoint 出自此臂（step 300） | https://wandb.ai/oxford-lob/sigma0-dfm-posttrain/runs/9wz87xrl |
| `2b_pre1e-4` | 332 | 主干 LR 加倍 | https://wandb.ai/oxford-lob/sigma0-dfm-posttrain/runs/rq8hp8rw |
| `2b_pre2.5e-5` | 340 | 主干 LR 减半 | https://wandb.ai/oxford-lob/sigma0-dfm-posttrain/runs/5kczravs |
| `2b_nowarmup` | 333 | 无 warmup（2A 需要 warmup，2B 不需要） | https://wandb.ai/oxford-lob/sigma0-dfm-posttrain/runs/r659lao9 |
| `2b_AAPL` | 300 | 早期单票臂，**无留出评估** | https://wandb.ai/oxford-lob/sigma0-dfm-posttrain/runs/wh6jsoq6 |

**每步记录**：`loss` / `grad_norm` / `p_cos` / `p_fro` / `retention` / `t_mean` / `dt`
＋ 8 个腐蚀分层各自的 `strata/i/{loss,t,retention}`。
**评估点（7 个）**：`eval/loss` / `eval/lo` / `eval/hi` ＋ `eval/strata/0..7`。

> `p_fro` 不是收敛信号（LayerNorm 尺度不变，只有方向进模型），看 `p_cos`。
> 这是 2A 阶段被数据推翻的三条判断之一。

---

## 3. DFM 后训练 Stage 2A —— 两个 16 格网格（回填，进行中）

**group**：`backfill-stage2a-20260801`，共 32 个 cell（`r1_cell00..15`、`r2_cell00..15`）。
`r1` 按 ticker 分票，`r2` 改用 488 票 in-distribution 分片并加留出评估。

---

## 4. 本会话自己的运行

**没有 W&B run**，因为本会话跑的全是**评测与分析**，不是训练：

| 运行 | 性质 | 产物 |
|---|---|---|
| 模拟器回放对照（3136 条） | 纯推理，无梯度 | `data/replay_control_full.npz` |
| AR 闸门 + 遗忘量化 | 8 batch 前向 | `ar_gate.json` |
| DFM 修正 rollout | 起草 + 修正 | `artifacts/rollouts/dfm_rollouts_L500_8seq.npz` |
| $t_{\rm start}$ 扫描 | 同上 × 3 | 进行中 |

**待办**：修正器扫描是一个真实实验，应当 **live** 上 W&B 而不是事后回填。
`dfm_correct_runner.py` 目前不接 W&B，需要补。
