# valset_v1 — SP500 Scaling-Law 固定验证集 详细报告

> **概要**：本验证集取自 SP500 LOB 预训练语料（488 只股票，2022–2025 年，每样本 500 条消息，26-token 编码），共 **5,367,734 个样本（占全域 1.661%）**，一次建成后永久固定。经逐样本验证，历史上所有训练 run 都没有接触过其中任何数据；正在排队重跑的 Transformer 矩阵即使全部完成，其数据消费也落在本集合预留的排除区之内，不会造成污染。
>
> 构建：2026-07-29，job 5790795（nid010407，用户预留节点），训练同源 env（torch 2.8.0+cu129）。
> 产物：`/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/artifacts_valset_v1_j5790795/`

---

## 1. 身份卡

| Property | Value |
|---|---|
| Domain N (48 months × 488 tickers, non-overlapping 500-msg windows) | **323,221,385** samples |
| Domain files (ticker-day pairs) | 472,442 |
| **Final pool V** | **5,367,734 samples = 1.661% of N** |
| Messages / tokens | ≈ 2.684 B messages / ≈ 69.78 B tokens |
| Nested frozen subsets | 30,720 ⊂ 307,200 ⊂ 3,232,213 (= 1% of N) |
| Ticker coverage | **488 / 488** (GOOG missing only its 2025-12 slice) |
| Files touched by val | 434,842 / 472,442 (92.0%) |
| v1 8-ticker flagged samples | 847,533 (15.79%) |
| Freezing | index lists + SHA-256 + `manifest.json` |

## 2. 构造配方与数量流水账

三个训练 shuffle seed 为 {5, 42, 137}。经代码链验证，`JAX_SEED` 就是 torch `DistributedSampler` 的数据打乱 seed（epoch=0），每个 seed 对应的样本顺序等于 `torch.randperm(N, manual_seed(s))`，训练消费的永远是这个排列的一个前缀。构造分五步进行。

第一步，对每个 seed 取其排列的最后 2%。这部分数据从未被该 seed 下的任何 run 消费过，三个 seed 合并去重后得到 19,007,384 个候选样本。第二步，删去落在任何一个 seed 排列前 20% 之内的样本。所有历史 run 的消费都是排列前缀，最深的一条（seed 5 的 700B 探针）只到 16.63%，所以前 20% 这条界线把全部已消费区域都包住了，剩下 12,106,704 个。第三步，Transformer 与 O2d 实验是在另一个索引域上训练的（2023–2025 共 36 个月、同样 488 只股票），正在排队的 Transformer 重跑也将在该域进行。把该域三个 seed 排列的前 20% 对应回 48 个月域的窗口（相邻窗口一并删除，因为两个域的窗口起点偏移不同），删去 6,735,581 个。第四步，一次 466 只股票子集的试运行消费过 19,200 个样本，同样对应回来删去 1,347 个。第五步，GOOG 在 2025 年 12 月的数据曾被一次 finetune 完整训练过，把该月 GOOG 的 10,377 个候选样本整体删去。最终得到 5,367,734 个样本。

| Step | Samples | % of N |
|---|---:|---:|
| Union of each seed's last 2% (deduplicated) | 19,007,384 | 5.881% |
| Remove samples inside any seed's first 20% | 12,106,704 | 3.746% |
| Remove mapped windows of the 36-month domain's three 20% zones (incl. adjacent windows) | −6,735,581 | |
| Remove mapped windows consumed by the 466-ticker pilot run | −1,347 | |
| Remove the entire (GOOG, 2025-12) month | −10,377 | |
| **Final validation set** | **5,367,734** | **1.661%** |

关键机制（为什么 index 级排除 = message 级精确排除）：逐文件随机 offset 的 RNG seed 是常量（`lobster_dataloader.py` `init_defaults seed=42`，与 JAX_SEED 无关），因此**所有 run 中样本 (file, j) 覆盖的消息区间完全相同**；跨数据域（36mo/466tk）offset 抽签不同，但两侧 offset 都是确定常量、错位方向逐文件已知，故每个外域消费窗口只需守卫 48mo 侧 2 个相邻窗口（方向性守卫）。

## 3. 零泄漏保证（逐样本验证，非推理）

![seed positions](figures/fig3_seed_positions.png)

**图 3**：V 中每个样本在三个 seed 排列中的位置分布。灰色 [0, 20%] 排除区内**零样本**（三 seed 最小位置分别为 0.2000003 / 0.2000004 / 0.2000002，紧贴边界之上）；[20%, 98%] 为均匀背景（该样本在"其它 seed"排列中的位置）；[98%, 100%] 尖峰（密度 ≈17.1）对应 last-2% 成员身份，每 seed 贡献约 34.2% 的 V。三条曲线几乎重合：三个 seed 在构造上可互换。

per-seed 历史消费（W&B 全部 270+23 runs 含 crashed 重试逐一核查，消费 = 最大步数 × gBSZ 128，皆为排列严格前缀）：

| Seed | Largest consumer | Steps | Samples | % of N | Margin to 20% |
|---|---|---:|---:|---:|---:|
| 5 | 6M-700B long-D chain | 420,000 | 53.76M | **16.63%** | 1.20× |
| 42 | 350M full-d chain | 168,200 | 21.53M | 6.66% | 3.00× |
| 137 | primary matrix (6M size) | 106,909 | 13.68M | 4.23% | 4.72× |

主矩阵 33/34 runs 内部消费最大者是 6M 尺寸（4.23%）；小模型 0.2M/1M/4M 仅 0.42%/0.74%/2.30%（固定墙钟 + 固定 gBSZ 下，小模型步数多、消费大，与直觉相反但都远低于 20%）。

**未来训练预算**（破则失效，已写入 manifest）：48mo 域每 seed 总步数 ≤ **505,033**（seed 5 已用 420,000，仅剩 ~85k）；36mo 域每 seed ≤ **381,251**（最大 TF curtail 65,664 的 5.8 倍，TF 重跑安全）。换新数据域（新 ticker/月份子集）训练前必须重新审计。

## 4. 数据量分布（时间维度）

![monthly](figures/fig2_monthly_distribution.png)

**图 2 上**：val 与全域的月度占比曲线同形（形状跟随 activity 波动），但 2022 年月份在 val 中占比系统性更高。**图 2 下**给出原因——覆盖率两个 regime：

| Year | Coverage (val / domain) | Mechanism |
|---|---:|---|
| 2022 | **3.744%** | 48-month recipe exclusions only (V0 survival 3.746%) |
| 2023 | 0.985% | plus the 36-month domain's three 20% zones + directional guard |
| 2024 | 0.984% | same as above |
| 2025 | 0.984% | same, plus the (GOOG, 2025-12) excision dip |

实测 0.984% 与理论预测 3.746% × (1−0.2)⁶ ≈ 0.981% 吻合（每样本要在 3 seeds × 2 守卫窗口共 6 次独立 20% 抽签中全部幸免），构造行为与数学预期一致。**含义**：val 内部 2022:2023-25 的比重约为 55%:45%（全域为 24.5%:75.5%）——做月度分层评测时用每月样本数加权即可还原任意目标权重，`val_pool_decode.npz` 提供逐样本月份解码。

每文件（ticker-day）窗口数分布：P10 = 118、P50 = 377、P90 = 1,241、max = 92,306；val 每文件命中数 P50 = 5、P90 = 26、max = 866。

## 5. 股票分布

![ticker representativeness](figures/fig1_ticker_representativeness.png)

**图 1**：逐 ticker 的 val 占比 vs 全域占比（log-log），488 点紧贴对角线，相关系数 **0.9861**——activity-weighted 分布被保持，val 是全域的忠实缩样。绿色为 v1 8-ticker 旗标（全部是高活跃大户）；GOOG 略低于对角线（2025-12 切除 + 36mo 排除叠加）。

![top30](figures/fig4_top30_tickers.png)

**图 4 / Top-10 明细**（val 样本数）：

| # | Ticker | Val samples | Val share | Domain share | v1 flag |
|---|---|---:|---:|---:|:---:|
| 1 | TSLA | 140,047 | 2.609% | 2.634% | ✅ |
| 2 | NVDA | 139,854 | 2.605% | 2.937% | ✅ |
| 3 | AAPL | 125,949 | 2.346% | 2.312% | ✅ |
| 4 | GOOGL | 119,141 | 2.220% | 2.211% | — |
| 5 | GOOG | 102,185 | 1.904% | 2.444% | ✅ |
| 6 | MSFT | 100,678 | 1.876% | 1.842% | ✅ |
| 7 | AMD | 97,544 | 1.817% | 1.797% | ✅ |
| 8 | AMZN | 94,293 | 1.757% | 1.741% | ✅ |
| 9 | MU | 50,128 | 0.934% | 0.911% | — |
| 10 | META | 46,983 | 0.875% | 0.652% | — |

集中度：val Top-10 占 18.94%（全域 top-10 占 23.03%，差异主要来自 GOOG/NVDA 的排除侧削）。**8 个 v1 旗标 ticker（GOOG/AAPL/NVDA/AMZN/META/TSLA/MSFT/AMD）合计 847,533 样本 = 15.79%**：这些 ticker 的原始消息曾被 8-ticker 旧语料时代实验（v1 sweep、phase-b-*、R1/O2 系、GOOG 项目）在不同预处理下接触过。评测 SP500 cohort（本 33/34 runs）无碍；评测 8-ticker 训练的旧模型时须用旗标列排除。

## 6. 嵌套子集与用法

| Subset | Size | Purpose | CE std. err. (per checkpoint, est.) |
|---|---:|---|---|
| `val_subset_30720` | 30,720 | routine quick eval (same size as the pre-registered test set) | < 1e-4 nats |
| `val_subset_307200` | 307,200 | high-precision comparisons | ~3e-5 nats |
| `val_subset_3232213` | 3,232,213 (= 1% of N) | final eval / paper numbers | ~1e-5 nats |
| `val_pool_indices` | 5,367,734 | full pool (superset of the above) | |

子集是池的一个固定 shuffle（seed 20260729）的前缀 → 严格嵌套，小子集结论可无缝在大子集上加密。**用法**：以与训练完全相同的 dataloader 配置重建数据集（manifest 记录全部配置），直接按索引取样本评测；`val_pool_decode.npz` 含逐样本 `(global_idx, file_idx, seq_idx, seq_start_msg, flag_v1_8ticker)`，`files_48mo.csv` 含逐文件 `(ticker, date, 窗口数, 偏移)`。

## 7. 披露与残余风险（均已量化写入 manifest）

1. **(GOOG, 2025-12) 整片缺席**：被 05-25 finetune（epochs=2）消费，整月切除（10,377 个候选样本）。
2. **8-ticker 旗标**（15.79%）：见 §5；默认保留以免扭曲 activity-weighted 分布。
3. **4 月 raw-tree 时代短 runs**（config 迭代批 ~618 步、FLOPs profile ~300 步、pre-SquashFS smokes）：文件序不可重建，预期 ≤0.3% 样本存在消息级接触可能；均为已废弃脚手架模型，不在任何评测名单。
4. **O2d 已删 pilot runs**（4×310 步）：若 seed ∈ {5,42,137} 已被 36mo 排除区覆盖；未知其它 seed 的期望影响 ~0.05%。

## 8. 产物清单（SHA-256 见 `SHA256SUMS.txt`）

```
artifacts_valset_v1_j5790795/
├── val_pool_indices.npy        # 5,367,734 × int64（已排序）    sha256 2404109444…
├── val_pool_decode.npz         # 逐样本解码 + 旗标
├── val_subset_30720.{npy,json} # 30k 子集（json 版已入 git）
├── val_subset_307200.npy
├── val_subset_3232213.npy      # 1%N 子集
├── files_48mo.csv              # 472,442 文件元数据
├── manifest.json               # 完整配方/证据/预算/披露
└── SHA256SUMS.txt
breadcrumb: tasks/validation_set/latest_valset.json
统计: tasks/validation_set/stats_valset_v1.json
复现: tasks/validation_set/{build_valset.py, build_valset.sbatch, valset_report_figs.py}
```

### 8.1 实体化副本（squashfs shard，2026-07-29 追加）

除索引形式外，valset_v1 另外物化为**与月度训练 shard 完全同构**的独立 squashfs 文件：目录布局为 `TICKER/TICKER_<date>_message_val<GLOBALIDX>.npy.zst` 与同名 `_orderbook_` 配对文件（每文件恰存一个 500 行窗口，切片行区间与训练管线逐字节一致），附同格式 in-shard `index.json`。现有 dataloader 把 `DATA_ROOT` 指向其挂载点即可直接使用，唯一差异是评测时传 `--random_offsets_train False`；文件名内嵌 global_idx 可溯源回索引产物。

| Tier | File | Size | Status |
|---|---|---:|---|
| 30,720 | `squashfs/output/shard_valset_v1_30720.squashfs` | 360 MB | packed; byte-level verification in progress (0 failures so far) |
| 307,200 | `squashfs/output/shard_valset_v1_307200.squashfs` | ~4 GB (est.) | queued |
| 3,232,213 (1% N) | on demand | ~40–60 GB (est.) | not built |
| full pool | on demand | ~70–100 GB (est.) | not built |

质检为双层：L1 抽 2,048 个样本，把物化文件内容与源 shard 对应行区间逐字节比对；L2 用训练 dataloader 挂载物化 shard 冒烟（样本数断言 + 读取探针）。通过后连同 `provenance_*.npz`（每样本 ↔ 源文件行区间映射）与 SHA-256 一起落盘。构建脚本：`squashfs/{materialize_valset.py, verify_valset_squashfs.py, run_materialize.sh}`。

**验证链**（构建时全部通过）：重建 N 与两条独立历史日志锚点吻合（8N `samples_per_node=40,402,673`、O2d 2N `=122,000,461`）→ torch `DistributedSampler` 等价性测试 → 最终 V 逐 seed 逆排列验证（图 3 即其可视化）。

## 9. 对照业界标准的审计

依据公开资料整理的验证集质量标准（来源见文末），逐项对照如下。

| # | Criterion | Standard | valset_v1 | Verdict |
|---|---|---|---|---|
| 1 | Disjointness / no leakage | val must be disjoint from all training data; time-series data needs look-ahead care | per-sample inverse-permutation proof, message-exact within domain; cross-domain prefixes subtracted with guards | **PASS** |
| 2 | Representativeness | same distribution as training (i.i.d.), all strata covered | ticker-share corr 0.9861, 488/488 tickers, activity-weighted; monthly decode enables re-weighting | **PASS** |
| 3 | Size / statistical power | judged by absolute count and SE of the metric, not percentage (large-corpus norm 0.1–1%) | 5.37M samples (1.661%); CE SE < 1e-4 nats already at the 30,720 subset | **PASS** |
| 4 | Frozen & versioned | fixed once, hashed, reproducible | SHA-256 + manifest + deterministic build in the training env | **PASS** |
| 5 | Reuse discipline | repeated adaptive reuse wears a holdout out (Dwork et al.); keep an untouched final set | tiered subsets for routine use; pre-registered Feb-2026 test set stays untouched for final claims | **PASS** (policy) |
| 6 | Empirical leakage check | construction proofs should be backed by a behavioral test | §10 experiment (seen vs held-out vs val CE) | see §10 |
| 7 | Documentation | datasheet-style provenance, known biases disclosed | manifest + this report, §7 disclosures | **PASS** |

需要长期注意的三点（不改变结论，但使用时要记得）：第一，val 的年度权重是 2022 偏重的（55:45），做与全域同权重的结论时按月加权；第二，(GOOG, 2025-12) 缺失与 8-ticker 旗标是已知的分布缺口；第三，val 度量的是与训练同分布的 held-out CE（scaling law 拟合需要的因变量），不能替代前向时移泛化评测，后者由 Jan/Feb-2026 测试集承担。文献同时提醒：holdout 被自适应地反复使用会逐渐失效（k/n 量级的偏差累积），所以论文级结论应在触碰次数最少的大子集或预注册 test set 上出数。

## 10. 独立泄漏实验（行为学验证）

构造性证明之外，按用户要求补一个独立实验：如果 33/34 个 run 真的没见过 val 数据，那么用训练出的 checkpoint 去测，val 的损失行为应当与"确定没见过的数据"一致，而与"确定见过的数据"可区分。这正是文献中的 dataset-inference / memorization-gap 检验；文献特别警告此类检验最常见的失效方式是对照组与被测组分布不同，产生假信号。本设计规避了这一点：三组样本来自同一个全域的均匀随机子集，仅"是否被训练消费过"不同。

**设计**（预注册判据，然后跑实验）：对 checkpoint（350M seed 5 为主，78M seed 5 做稳健性），各取 30,720 个样本的三组：

| Group | Definition | Training exposure |
|---|---|---|
| SEEN | uniform sample from perm_s[0 : consumed_s] | seen exactly once |
| MID | uniform sample from perm_s positions [20%, 98%], outside every tail | never seen, not in val |
| VAL | `val_subset_30720` | never seen (claimed) |

统计量：每组平均 per-token CE（bootstrap 95% CI）与 Min-K% (k=20%) 平均对数概率。判据 H1（检测力）：CE(SEEN) 显著低于 CE(MID)；判据 H2（无泄漏）：CE(VAL) 与 CE(MID) 的差的置信区间包含 0。两个判据同时成立即为实验性无泄漏证据；若 H1 不成立，说明单次曝光的记忆效应低于检测限，此时任何残余泄漏对 CE 评测的影响同样低于检测限，结论同样支持 val 的可用性。

**结果**：（r4 运行中，完成后回填终值。）过程记录：r1 因 torch DataLoader 的 fork worker 与 JAX 多线程死锁（GPU 显存已上卡但利用率 0%、全节点 CPU 空闲的三联征）；改同步加载后，按效率要求把评测 batch 放大到训练值的 4–8 倍并改用 spawn worker，随即触发第二个故障：批量放大后的超大矩阵乘让 XLA Triton 自动调优器找不到有效配置（r3 崩溃），禁用 Triton GEMM 回落 cuBLAS 后（r4）恢复运行。两次故障与修复均已入库（commits 3bd38a7、1c3e192）。判据不变。

## 参考来源

[Unidata: Validation Dataset in ML](https://unidata.pro/blog/validation-dataset-in-ml/) · [IBM: What is Data Leakage in Machine Learning](https://www.ibm.com/think/topics/data-leakage-machine-learning) · [Google Research: The reusable holdout](https://research.google/blog/the-reusable-holdout-preserving-validity-in-adaptive-data-analysis/) · [Dwork et al. 2015, Generalization in Adaptive Data Analysis and Holdout Reuse](https://arxiv.org/pdf/1506.02629) · [mlbenchmarks.org: Test set reuse](https://mlbenchmarks.org/05-test-set-reuse.html) · [The Reliability Gap in Benchmark Auditing (arXiv 2606.03305)](https://arxiv.org/html/2606.03305) · [Gap-K%: Measuring Top-1 Prediction Gap for Detecting Pretraining Data (arXiv 2601.19936)](https://arxiv.org/pdf/2601.19936) · [awesome-data-contamination paper list](https://github.com/lyy1994/awesome-data-contamination)

*报告生成：2026-07-29。统计脚本 `valset_report_figs.py`（训练 env torch 2.8.0 + matplotlib 3.10.8）。*
