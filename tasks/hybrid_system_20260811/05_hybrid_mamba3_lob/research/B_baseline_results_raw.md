# B · 纯 Mamba3 baseline 已有基准数值（原始汇总）

**调研时间**：UTC 2026-08-11T21:00Z

---

## 0. 最重要的一条：这些数字分属 5 个互不可比的模型/评测组

| 组 | 模型 | ckpt | 训练 job | 语料 | 评测池 | 四类齐全度 |
|---|---|---|---|---|---|---|
| **M1** | sigma-0 SP500 Mamba3 **33.6M** | step **32001** | `5877859` | 8 票 × 2022–2025 | model_zoo 池 sha `0c41de51…` | 7.3 ✅ · 7.4 ✅ · 7.1 ❌ · 7.2 ❌ |
| **M2** | model_zoo 26tok Mamba3 **8.1M** ×3 seed | step **30000** | `5798828` | GOOG × 2022–2025.11 | 同上 | 7.2 ✅ · 7.3 ✅ · 7.1 ❌ · 7.4 ❌ |
| **M3** | R1 Mamba3 **78.5M**（历史黄金参考） | step **46050** | `3417629` | GOOG 8 票 | **R1 旧池** sha `4909799c…` | 7.1 ✅ · 7.2 ✅ · 7.3 ✅ · 7.4 ❌ |
| **M4** | sigma-0 self-train **78.5M** | step **69378** | `5705912` | SP500 | model_zoo 池 | 7.3 ✅ |
| **M5** | R1 Mamba3 **293M** | step **150360** | `4559297` | — | GS 8 窗口（非 GOOG） | 7.4 ✅（旧口径） |

> **M1 与 M3 的评测池索引范围分别是 [1,21728] 与 [116,225976]，交集只有 46/3136 = 1.5%。0.0438 和 0.2088 绝对不能并排。**（`findings.md:1245` / F196）

---

## 1. (7.1) message-level perplexity / per-token CE

### 1.1 唯一直接可用的纯 Mamba3 held-out CE

| 数值 | 口径 | 来源 | job/step |
|---|---|---|---|
| **test_ce = 0.4482 nats/token**，acc = 0.908 | 26tok **per-token** CE，GOOG 2026-01 held-out，seq_len=13000 | `findings.md:115` (F011) | `3417629` @ **46050**，78.5M |
| ≈ **11.65 nats/message**（=0.4482×26） | **本报告算术换算，非实测**。26tok 全部 26 位都计损失（`m=ones_like(y)`），故 per-message = per-token×26 严格成立 | 换算依据 `tasks/bpe_varlen_torch_20260806T183132Z/RESULTS.md:1985-1995` | 同上 |

### 1.2 训练 loss（非 held-out，旧池 scaling sweep，仅参考）

| 模型 | 参数量 | 训练 loss | 步 | job |
|---|---|---|---|---|
| Mamba3-8m | 8,099,567 | 0.5802 | 29,790 | 3443014 |
| Mamba3-14m | 14,445,623 | 0.5804 | 26,840 | 3443015 |
| Mamba3-23m | 22,949,247 | 0.5735 | 39,490 | 3443016 |
| **Mamba3-34m** | **33,610,439** | **0.5623** | 31,770 | 3443017 |
| Mamba3-46m | 46,429,199 | 0.5290 | 25,380 | 3443018 |
| Mamba3-78m | 78,539,423 | 0.5634 | 46,050 | 3417629 |

来源：`.../AlphaTrade/experiments/exp_R1_Mamba3/scaling_law_summary.csv`

### 1.3 scaling law 拟合面（两套口径）

```
test CE (Jan-2026, macro over 487 ticker):
  L = 0.515 + 1.70e11·N^(-1.36) + 3.05e11·D^(-1.12)      R² = 0.975
  来源: findings.md:115 (F011, VERIFIED)

val CE (val_ce_macro, 256 点全轨迹, 33 chain):
  L = 0.5957 + 1.53e12·N^(-1.863) + 1.02e11·D^(-1.153)
  α = 1.863 [1.546, 1.954]   β = 1.153 [1.107, 1.176]
  来源: tasks/validation_set/hand_off_valset_scaling_law_fit_20260731.md:18-31
  实测 val_ce 范围 0.6003–2.5119 (256 行 × 12 尺寸)
```

### 1.4 明确「未测」

- **M1（33.6M step32001）没有任何本地 val CE / perplexity**。训练日志目录只有 chain/attach 日志无 loss；数据只在 W&B `oxford-lob/sp500-mamba3-35m`。
- **M2 也没有**：`artifacts/model_zoo_26tok/train_logs/mamba3/seed_*/` 三个目录**全空**；`validation_logs/mamba3/seed_42/` 只有 preflight 冒烟（2,344,884 参数、`validation_ce=7.694 @ step=21`，与正式模型无关）。
- **没有任何模型测过显式的「message-level perplexity」**，现有全是 per-token CE。M1/M2 都需补测。

> ⚠️ **必须写进 hybrid 评测设计的教训**（`findings.md:2853` / F296）：同一改动使 **CE 只 +0.815% 而 LOB-Bench KS +102%**。「perplexity 作为『有没有破坏模型』的守门员几乎失灵」，任何 perplexity 闸门必须配分布级闸门。

---

## 2. (7.2) DirAcc 与 return IC

### 2.1 M2 —— 唯一一组「同一份生成产物同时算了 LOB-Bench + IC + DirAcc」

模型：model_zoo 26tok Mamba3，**8,099,567 参数，step 30000**，GOOG，3 seed。
协议：cond 250 / gen 250 条消息，3136 序列，GOOG 2026-01，generation_seed 2026。
生成 job：seed5=`5841413`、seed42=`5841414`、seed137=`5841412`。

| seed | Pearson IC [CI95] | Spearman IC [CI95] | wmid Spearman | **DirAcc** | n_zero_ret | Sharpe | n_trades |
|---|---|---|---|---|---|---|---|
| 5 | **0.16046** [0.08209,0.24145] | 0.10650 [0.06868,0.14137] | 0.08839 | **0.53928** | 679 | 0.07500 | 2697 |
| 42 | **0.08108** [0.02939,0.13282] | 0.08844 [0.05212,0.12185] | 0.08377 | **0.53878** | 686 | 0.08517 | 2692 |
| 137 | **0.13109** [0.04323,0.21627] | 0.08382 [0.04963,0.11897] | 0.04345 | **0.54028** | 852 | 0.07872 | 2509 |
| **均值** | **0.1242** | **0.0929** | 0.0719 | **0.5394** | — | 0.0796 | — |

其它 IC：`ofi_ic_spearman` = −0.0481/−0.0297/−0.0619（**符号为负**）；`qi_ic_spearman` = +0.0500/+0.0328/+0.0674；`trade_flow_ic_spearman` = −0.0599/−0.0873/−0.0387。

来源：
```
sigma-0/artifacts/model_zoo_26tok/paper_runs_goog_20260727/downstream_30k/mamba3/seed_{5,42,137}/downstream_GOOG_production30k.json
计算脚本: AlphaTrade/lob_pipeline/pipeline/downstream_metrics.py
输入 inference: .../evaluation_30k_mamba3_fix_b1868a4/mamba3/seed_*/inference   ← 必须是 fix 版
```

**口径警告**：`downstream_metrics.py` 的 DirAcc 分母已剔除 zero-return 序列（n_zero_return 列）；n_valid=n_total=3136 表示无窗口被丢弃。

### 2.2 M3 —— 78M step46050 的两个 horizon（旧池）

| horizon | WS21 | KS | **PIC** | **DirAcc** | Sharpe |
|---|---|---|---|---|---|
| h250 | 0.0438 | 0.090 | **0.142** | **0.5724** | 0.1308 |
| h500 | 0.0539 | 0.0945 | **0.1042** | **0.5335** | 0.0705 |

来源 `findings.md:115` (F011)，job `3417629` @46050，n=3136，**R1 旧池**。

### 2.3 M7 —— IsoFLOP sweep（每 job 全 ckpt 取最优，旧池）

| 参数量 | Mean WS ↓ | Sharpe ↑ | **Return IC ↑** | 最优步 | job |
|---|---|---|---|---|---|
| 0.2M | 0.2196 | 0.0519 | 0.0611 | 8,150 | j4507398 |
| 1M | 0.2273 | 0.0972 | 0.1086 | 12,920 | j4507408 |
| 4M | 0.1684 | 0.0783 | 0.0851 | 37,670 | j4508758 |
| 6M | 0.1812 | 0.1030 | 0.1018 | 72,080 | j4508786 |
| 10M | 0.1792 | 0.1072 | 0.1410 | 61,760 | j4507579 |
| 14M | 0.1028 | 0.1204 | 0.1331 | 58,970 | j4507584 |
| 23M | 0.1864 | 0.0887 | 0.1397 | 55,530 | j4508675 |
| 50M | 0.1721 | 0.1207 | 0.1488 | 36,560 | j4508677 |
| 78M | 0.1460 | 0.1074 | 0.1095 | 22,180 | j4501171 |
| 120M | 0.1556 | 0.0936 | 0.1229 | 19,870 | j4499543 |
| 200M | 0.1837 | 0.1002 | 0.1132 | 16,220 | j4499578 |
| 350M | 0.1663 | 0.1085 | 0.1286 | 150,360 | j4559297 |

来源 `exp_R1_Mamba3/checkpoints_lobbench_scores.txt`（46 行）。**该表无 DirAcc 列**。唯一另一条 DirAcc：293M chain (j4553948) best dir-acc = **0.5523 @ step 120000**（`findings.md:290` / F039）。

### 2.4 明确「未测」

- **M1（33.6M step32001）从未跑过 return-bench**。已核实 `bench_20260806T160429Z_j5924045/inference/return_bench` 与 `bench_20260808T234338Z_j5950739/inference/return_bench` 均**不存在**；`baseline255/return_bench` 也不存在。
- 脚本现成可用：`AlphaTrade/lob_pipeline/return_bench/run_return_bench.py`（输出到 `<infer_dir>/return_bench/return_bench*.csv`），但目录下**只有脚本无结果 CSV**。
- ⚠️ return-bench 技能与 `downstream_metrics.py` **口径不同**：技能默认 n_cond=500、horizons=[10,50,100,250,500]；后者是 cond250/gen250 单口径。做对照前必须先固定用哪一个。

---

## 3. (7.3) LOB-Bench

### 3.1 权威口径

**WS-21 = 对 `scores_uncond_GOOG_*.pkl` 里全部 21 个特征各取 Wasserstein 点估计后求算术平均；任一特征 NaN 则整档作废。** KS-21/L1-21 同法。设置 `msg_seq_len=500` + c250g250，GOOG，2026-01，无条件生成。
来源：`findings.md:1097` + `exp_R1_Mamba3/extract_scaling_law_best.py` docstring。

21 特征：`ask/bid_cancellation_depth`、`ask/bid_cancellation_ticks`、`ask/bid_volume`、`ask/bid_volume_touch`、`limit_ask/bid_order_depth`、`limit_ask/bid_order_ticks`、`log_inter_arrival_time`、`log_time_to_cancel`、`ofi`、`ofi_up`、`ofi_down`、`ofi_stay`、`orderbook_imbalance`、`spread`、`vol_per_min`。

### 3.2 M1 —— 33.6M @ step32001（**当前主基线**）

| 评测 | 序列 | **WS-21** | **KS-21** | **L1-21** | bench job | 来源 |
|---|---|---|---|---|---|---|
| 生产链（原始） | 3136 | **0.20880** | **0.10645** | **0.16288** | `5924045` | `tasks/sp500_mamba3_35m_20260805T030348Z/bench_20260806T160429Z_j5924045/summary.json` |
| 新 harness 闸门1（**推荐引用**） | 3136 | **0.20714** | **0.10458** | **0.16451** | `5950739` | `tasks/j5877859_32001_bench_20260808/bench_20260808T234338Z_j5950739/summary.json` |
| 新 harness 复跑 | 3136 | 0.21109 | 0.10804 | 0.16620 | `5950739` | `.../bench_20260809T142459Z_j5950739/summary.json` |
| 子集口径 | 248/255 | **0.23708** | — | — | — | `findings.md:2449` (F270) |

- 生产 vs 新 harness 相对差 −0.79%/−1.71%/+0.99%，全在 ±2%。**harness 噪声底 0.6%，协议差异约 1%，两次复跑散布 1.9%。**
- **子集/全池放大系数 = 1.1445**（0.23708/0.20714）。子集取的是冻结池**开头** N 个而非随机 N 个（`findings.md:1240` / F193），**不是无偏估计**。这就是记忆里「0.20714/0.23708 不能混用」的出处。
- 同池校验：`sample_indices sha256=a0cd27b5…`、索引文件 `sha256=0c41de51…`、`benchmark_revision=1128d37c…`、`dataset_length=226002`、cond250/gen250/seed2026/world_size4。

**M1 逐特征 WS（生产 j5924045）**：

```
limit_bid_order_depth 0.34120   limit_bid_order_ticks 0.34021
ask_cancellation_depth 0.32708  ask_cancellation_ticks 0.32707
bid_cancellation_depth 0.32339  bid_cancellation_ticks 0.32329
ofi_stay 0.28456  ofi 0.28369  ofi_up 0.26012  ofi_down 0.24163
log_time_to_cancel 0.24834  limit_ask_order_ticks 0.23863  limit_ask_order_depth 0.23489
spread 0.22980  orderbook_imbalance 0.08735  log_inter_arrival_time 0.05974
ask_volume 0.05426  bid_volume 0.05049  ask_volume_touch 0.04891  vol_per_min 0.04406  bid_volume_touch 0.03612
```

最差一族是**深度/tick 类**（0.32–0.34，「挂在离最优价第几档」），最好是**量/时间类**（0.036–0.060）。

### 3.3 M2 —— 8.1M @ step30000，3 seed

**必须用 `evaluation_30k_mamba3_fix_b1868a4`（commit b1868a4）；旧 `evaluation_30k` 已被 bug 污染，两者不可混用。**

| seed | **WS-21** | **KS-21** | **L1-21** | gen job |
|---|---|---|---|---|
| 5 | **0.20949** | 0.12137 | 0.19543 | 5841413 |
| 42 | **0.21590** | 0.12206 | 0.19262 | 5841414 |
| 137 | **0.24424** | 0.12980 | 0.22858 | 5841412 |
| **均值** | **0.22321** | 0.12441 | 0.20554 | — |

**作废（bug 版，勿用）**：`evaluation_30k/mamba3/seed_{5,42,137}` = WS 0.33080/0.35267/0.33319。

M2 训练配置：8,099,567 参数；GOOG；train 2022-01-01→2025-11-30；val 2025-12；test 2026-01；4 节点×4 GPU；micro_batch 8；**global_batch 128**；`local_steps_k=10`；30,000 updates；26tok；**49.92B token**；job `5798828`。

### 3.4 M3 —— 78M step46050（**另一评测池，不可并排**）

| 数值 | WS-21 | KS-21 | L1-21 |
|---|---|---|---|
| 原始 scaling_law_summary.csv | **0.0442** | 0.0910 | 0.1555 |
| 冻结参考 j3417629 | **0.04376** | 0.08986 | 0.13645 |
| sigma-0 闸门0 复现（同 ckpt） | **0.04461** | 0.08939 | 0.13661 |

相对差 1.95/0.53/0.12% ✅ harness 已标定。复现 0.0438 只需 `MAMBA3_NORM_MODE=legacy`（换 epsilon），不需要 `historical_flax`。

### 3.5 M4 / M5

| 模型 | WS-21 | KS-21 | L1-21 |
|---|---|---|---|
| self-train 78.5M step69378（startmaskfix `2678fdb`，eval job `5823145`） | **0.22882** | 0.13444 | 0.20346 |
| 同 ckpt，无 start-mask fix | 0.24371 | 0.13273 | 0.22042 |
| 同 ckpt，2026-07-29 诊断（**strict summary FAILED，非正式**） | W20 0.22538 | KS20 0.13415 | 0.20551 |
| R1 78M SP500 488 票 j4501061 step46880（sigma-0 gate1） | **0.18075** | 0.10334 | 0.14784 |

### 3.6 三张成规模的旧榜

- **榜 A**（6 个 scaling-law 正式模型）：见 §1.2，WS 0.0442–0.1287。来源 `scaling_law_summary.csv`
- **榜 B**（45 job IsoFLOP，全 ckpt 取最优）：见 §2.3，Mean WS 0.1028–0.2474。来源 `checkpoints_lobbench_scores.txt`
- **榜 C**（R1×SP500，103 job × 14 档 2.63M→293.3M）：`ws21/ks21/l1_21` 齐全无 IC/Sharpe。来源 `sigma-0/artifacts/r1_sp500_sweep/sp500_sweep_results.csv`

榜 A 与榜 B 的 78M 分别是 Wass 0.0442 (s46050) 与 Mean WS 0.1460 (s22180)，**差 3.3 倍，代际+评法差异，不可并榜**（`findings.md:1095`）。

### 3.7 ⭐ 一条改变评测设计的限制

`findings.md:1379` (F234)：把生成窗口内**时间顺序完全打乱**后 WS-21 反而**好 13.7%**（0.1441 → 0.1244）。21 个指标里 **13 个恰好 ±0.0%**（纯逐行边际量），4 个真正惩罚乱序（ofi 族），4 个乱序反而更好。

> **「WS-21 更低」只能说成「更好地复现了这 21 个统计量的边际分布」，不能说成「生成的市场动态更真实」。**
> **推论：hybrid 主打「精确回指历史」，(7.3) 大概率测不出来，胜负必须压在 (7.4)。**

---

## 4. (7.4) refer order success rate

### 4.1 M1 —— 最干净、最新、最贴 PLAN §4 定义（2026-08-11 实测）

模型 M1（33.6M step32001）作为「26-token 基线臂」。序列集 paired-255，条件内容 `SHA256 = 9ae74b7d16466a3da2df5bde20aea941bc7274adff2c9fcaaca63f6c5b8ad648`。
**口径：CANCEL/DELETE 成功与否按「事件发生前的实时簿上是否存在该 order id」判定，不是从撮合引擎副作用倒推。**

| 指标 | 分子/分母 | 百分比 |
|---|---|---|
| **cancel_reference_live_order_id** | **29,941 / 31,142** | **96.1435%** |
| cancel_reference_live_order_id_**and_price** | 23,001 / 31,142 | **73.8585%** |
| final_message_syntax | 63,750 / 63,750 | 100.0000% |
| raw_candidate_canonical_syntax | 63,750 / 63,750 | 100.0000% |

来源：
```
tasks/varlen_bench_subset_20260809/generation_quality_supervision/r6p1_s9000_paired255_20260811T1810Z/report_v3/{summary.md,generation_quality_metrics.csv}
audit 输入清单: .../audit/generation_quality_manifest.json
baseline_gen_dir = tasks/varlen_bench_subset_20260809/baseline255/data_gen
```

**分层拆解（同一份输出，2026-08-11 resolver 审计）**：

| 解析层 | 命中 | 占 31,142 |
|---|---|---|
| **L1 精确纳秒时间戳** | 23,466 | **75.3516%** |
| L2 同当前价 + 最近毫秒回退 | +6,475 | **+20.7919 pp** |
| **合计 live-ID 成功** | **29,941** | **96.1435%** |
| 残余未解 | 1,201 | **3.8565%** |

残余 1,201 条结构：1,180 条 delete、21 条 partial cancel；100% 既错过精确时间戳又在生成的当前价上无活单；39.88% 在可见历史里有同价 NEW（已过期），60.12% 完全没有。最近可见价距离：1 tick 内 42.13%、10 tick 内 53.79%、100 tick 内 80.27%。

**真实数据 oracle（同 paired-255，判 hybrid 上限用）**：

| 量 | 值 |
|---|---|
| cancel/delete/execution 合计触及 | 31,630 |
| 目标 NEW 在条件段+已生成段可见 | 27,789 = **87.8565%** |
| 仅 cancel/delete | 27,269/30,925 = **88.1779%** |
| **仅 execution** | 520/705 = **73.7589%** |
| **可见目标上，精确纳秒时间唯一确定目标** | 27,789/27,789 = **100%** |

### 4.2 M5 —— 旧口径（293M，标的 **GS** 非 GOOG，8 窗口）

7,200 条生成消息，其中 3,275 条 cancel/delete：

| provenance | count | 占全部 | 占 cancel |
|---|---|---|---|
| 非撤单 | 3,925 | 54.5% | — |
| exact-timestamp hit | 2,854 | 39.6% | **87.1%** |
| price fallback | 66 | 0.9% | 2.0% |
| **total miss** | **355** | 4.9% | **10.8%** |
| NaN 分支 (D-R1) | 0 | 0.00% | — |
| ref price ≠ event price (D-R4) | 1,968 | 27.3% | 60.1% |

来源 `sigma-0-worktrees/mamba3-variable-bpe-20260807/docs/fidelity_defect_taxonomy.md:429-434`。生成产物 `tasks/agentic_mm_20260801T132010Z/`，ckpt `j4559297_u52a0g05_4559297/150360`（293M R1），stock=**GS**，cond500/gen900，job `5856560`。

⚠️ **D-R4 的 60.1% 已被同文件自身的对照证伪**：同一检查跑在**真实**消息上也触发 55.83%，说明 `price` 与 `price_ref` 根本不共享参考系。该 bit 已改名 `price_differs_from_ref`。**不能用作模型质量指标。**

### 4.3 四种口径对照（分母定义各不相同，**不可互换**）

| 口径 | 26tok Mamba3 | 真实数据 | 来源 |
|---|---|---|---|
| **live-ID 成功率**（paired-255，活簿判） | **96.1435%** | — | `report_v3/summary.md` |
| live-ID + price 双条件 | **73.8585%** | — | 同上 |
| 撤改命中率（CSV 未命中反推） | **81.2%** | **88.3%** | `findings.md:2693` (F281) |
| 撤单命中「生成窗口内 NEW 单」 | **69.6%** | **83.2%** | `findings.md:2408` (F266/F268) |
| exact-timestamp hit 占 cancel（GS/293M 旧口径） | **87.1%** | — | `fidelity_defect_taxonomy.md:432` |

**建议 hybrid 对照表只采用 §4.1 那一行（96.1435% 与 L1 75.3516%）**，因为它是唯一按活簿定义、有分子分母、有条件内容 SHA、有可复算 CSV 的。

### 4.4 明确「未测」

- **execution（成交）类的 success rate 从未在生成侧单独测过**。§4.1 的 31,142 分母是 cancel+delete；execution 只在真实数据 oracle 里出现过一次（705 条，可见率 73.7589%）。**(7.4) 要求三类分开报，baseline 目前只有两类混算的一个数。**
- M2（8.1M）**从未测过**。M3（78M step46050）**从未测过**。
- 26tok 基线**没有 order-id 字段**，靠 (price,size,time) 三元组回查 + 同价档时间最近回退，这是 96.14% 的机制来源。hybrid 若沿用 26tok 编码，**96.14% 就是要打败的数**。

---

## 5. 对照表方案：没有任何 ckpt 四类齐全

### 方案 A（推荐）：以 M1（33.6M step32001）为唯一 baseline，补测两项

| 行 | 指标 | 现状 | 数值 |
|---|---|---|---|
| 7.1 | message-level perplexity | ❌ **未测，需补** | — |
| 7.2 | DirAcc / return IC | ❌ **未测，需补**（脚本现成，登录节点 CPU 可跑） | — |
| 7.3 | LOB-Bench WS/KS/L1-21 | ✅ | **0.20714 / 0.10458 / 0.16451**（全池 3136，job 5950739） |
| 7.4 | refer success（cancel+delete，live-ID） | ✅ | **96.1435%**（29941/31142，paired-255）；L1 精确 **75.3516%** |

⚠️ 7.3 是全池 3136，7.4 是 paired-255，**样本集不同**，表里必须各自标注序列数。

### 方案 B：以 M2（8.1M ×3 seed）为 baseline，7.2/7.3 同源可直接并排

| 行 | 指标 | 现状 | 数值（3 seed 均值） |
|---|---|---|---|
| 7.1 | perplexity | ❌ 未测 | — |
| 7.2 | DirAcc / Pearson / Spearman | ✅ **同一份 inference** | **0.5394 / 0.1242 / 0.0929** |
| 7.3 | WS/KS/L1-21 | ✅ **同一份 inference** | **0.22321 / 0.12441 / 0.20554** |
| 7.4 | refer success | ❌ 未测 | — |

M2 的优点：**3 seed 给出天然方差估计**（WS-21 跨 seed 0.20949–0.24424，散布 16.6%），直接给了显著性门槛一个起点 —— **单点 WS 差异小于约 ±0.017 无意义**。

---

## 6. 引用这些数字时必须一起写的六条警告

1. **两个池不可混用**：M1/M2/M4 用 `0c41de51…`（226,002 个 500-msg 序列），M3/M6/M7 用 `4909799c…`（最大索引 21,728），交集 1.5%。
2. **子集 ≠ 全池**：`generate.py` 的 `todo` 是 `range(n_sequences)` 取开头 N 个，非随机抽样。M1 的子集/全池系数 = 1.1445。
3. **PyTorch 线自建 harness 的 0.2748 是坏基线**：同一个 26tok 模型在正确 harness 上是 0.20714，被打残 32%。凡见 0.2748/0.2243 请确认来自 `tasks/bpe_varlen_torch_20260806T183132Z`（PyTorch Transformer，**不是 Mamba3**）。
4. **model_zoo mamba3 有 fix 前/后两套评测**，差 1.5 倍，必须用 `evaluation_30k_mamba3_fix_b1868a4`。
5. **噪声底**：harness 0.6%，同 ckpt 两次全池复跑散布 1.9%，跨 seed 散布 16.6%。
6. **LOB-Bench 对时间顺序不敏感**（21 项里 13 项恰好零变化、4 项反向），测的是边际分布不是动态。命题 H 的「精确回指」必须靠 (7.4) 判，不能指望 (7.3)。

---

## 7. 主要来源清单（绝对路径）

```
# 汇总/叙事
u6gb/findings.md   行 115, 552, 1089-1097, 1163-1165, 1237-1254, 1379, 1652, 2272-2303, 2408, 2446-2452, 2693-2713, 3039-3052
u6gb/progress.md   行 988, 994
u6gb/tasks/r1_mamba3_lobbench_leaderboard_20260806T174426Z/R1_MAMBA3_LOBBENCH_LEADERBOARD.md
u6gb/tasks/varlen_bench_subset_20260809/HANDOVER_20260810T004000Z.md
u6gb/sigma-0/artifacts/r1_sp500_sweep/HANDOFF.md

# (7.3) LOB-Bench summary.json
tasks/sp500_mamba3_35m_20260805T030348Z/bench_20260806T160429Z_j5924045/summary.json
tasks/j5877859_32001_bench_20260808/bench_20260808T234338Z_j5950739/summary.json
tasks/j5877859_32001_bench_20260808/bench_20260809T142459Z_j5950739/summary.json
sigma-0/artifacts/model_zoo_26tok/paper_runs_goog_20260727/evaluation_30k_mamba3_fix_b1868a4/mamba3/seed_{5,42,137}/lobbench_summary.json
sigma-0/artifacts/selftrain_lobbench/j5705912_step69378_startmaskfix_2678fdb_j5823145_gpu0seq/evaluation/lobbench_summary.json
sigma-0/artifacts/r1_sp500_sweep/sp500_sweep_results.csv
AlphaTrade/experiments/exp_R1_Mamba3/{scaling_law_summary.csv,checkpoints_lobbench_scores.txt}

# (7.2) IC / DirAcc
sigma-0/artifacts/model_zoo_26tok/paper_runs_goog_20260727/downstream_30k/mamba3/seed_{5,42,137}/downstream_GOOG_production30k.json
AlphaTrade/lob_pipeline/pipeline/downstream_metrics.py
AlphaTrade/lob_pipeline/return_bench/run_return_bench.py   (脚本存在，无结果)

# (7.4) refer order success
tasks/varlen_bench_subset_20260809/generation_quality_supervision/r6p1_s9000_paired255_20260811T1810Z/report_v3/{summary.md,generation_quality_metrics.csv}
tasks/varlen_bench_subset_20260809/resolver_design_supervision/26tok_27pointer_l3_20260811T193458Z/summary.md
tasks/varlen_bench_subset_20260809/resolver_design_supervision/cascade_t_price_size_index_20260811T194716Z/summary.md
sigma-0-worktrees/mamba3-variable-bpe-20260807/docs/fidelity_defect_taxonomy.md   (行 157, 269-276, 429-440)

# (7.1) CE
tasks/validation_set/hand_off_valset_scaling_law_fit_20260731.md
tasks/bpe_varlen_torch_20260806T183132Z/RESULTS.md   (§22 nats/message 换算口径)
sigma-0/artifacts/model_zoo_26tok/paper_runs_goog_20260727/submission_mamba3_s5_5798828.json
```
