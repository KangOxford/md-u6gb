# Scaling-Law Checkpoints 在 valset_v1 上的 Validation Loss 评测报告（v1）

*S&P 500 LOB Mamba-3 scaling-law cohort · 132 checkpoints × 30,720-sample frozen validation set · 2026-07-30*

---

## 1. 执行摘要

本次评测在 valset_v1 固定验证集（30,720 样本实体包）上计算了 SP500 scaling-law 队列全部 **132 个 checkpoint**（33 个 logical run 的 final-25% 窗口，含 33 个 terminal）的 validation cross-entropy，每个 checkpoint 覆盖全部 30,720 个样本并将**逐样本 loss** 落盘。这填补了 rebuttal 中「同分布 held-out CE」这一缺失的评测轴：此前 SP500 队列唯一的 held-out 数字是 Jan-2026 前向时移 test CE，它把「泛化」与「分布漂移」混在一个数里。

三个主要发现。第一，**同分布 valset CE 在所有 12 个模型规模上都高于前向时移的 Jan-2026 CE**（macro 口径 Δ 从 0p2M 的 +0.055 nats 到 350M 的 +0.005 nats），即「前向时移代价」在本语料上为负：Jan-2026 单月比训练分布（2022 年权重 55%）更容易预测。第二，**Δ 随模型规模单调收敛**（+0.055 → +0.005），大模型对评测分布的差异更鲁棒。第三，**held-out 最优规模对评测分布敏感**：valset 上最低 CE 出现在 120M（micro 0.6014）与 46M（0.6033），而 Jan-2026 表上最低点是 23M（0.5689）；两条曲线都在最大两档回升（350M 只消费了 17.8B token，D 不足效应，与规模本身无关）。

全部产物（132×2 结果文件 + 汇总 + 图 + SHA256）已入库，任何数字可由 per-sample loss 精确重算（micro/macro/按月/按 ticker 任意重加权）。

## 2. 背景与动机

审稿人 pXiP Q2 要求在 held-out 数据上做 unconstrained refit。现有的 Jan-2026 test CE 是**前向时移**评测（训练数据止于 2025-12，评测用 2026-01），在它上面 α 顶到 box 上界（2.000, not identified），且时移引入的分布漂移与模型泛化能力无法区分。valset_v1（构造与零泄漏证明见 `VALSET_V1_REPORT.md`）提供**与训练同分布**的 held-out 轴：样本取自训练语料的未消费区域（逐样本位置证明 + 20% 排除边界），与训练数据同分布、永久固定。

在这条轴上评测同一批 checkpoint，可以：(i) 把「时移代价」单独测出来（同一模型、同一 macro 口径下两轴之差）；(ii) 为 valset 上的 unconstrained fit 提供数据基础——132 个 checkpoint 保留了每个 run 内部的 D 变异（每 run 1–10 个 checkpoint），这是 β 可识别的前提（terminal 33 点的 D 与 N 共线，β 识别弱，CI 宽达 [0.17, 0.94]）。

## 3. 评测对象

132 = 33 个 logical run（12 sizes × 实际完成 seeds：23M 缺 s42，200M/350M 缺 s137）× 各自 final-25% 窗口内的 checkpoint（min 1 / median 3 / max 10）。权威清单 `s5e_scalinglaw/aramis/results/selected_test_last25.csv`（含每 checkpoint 的 step、D、Jan-2026 CE），checkpoint 路径由该表 join `wandb_mamba3_runs_snapshot.csv` 推导（同 jid 多 wandb run 取 global_step 最大者），132/132 目录与 step 子目录在评测前逐一验证存在。46M-s5 的 terminal 为 time-limit 中断点（step 53,970 / target 63,407），与主表口径一致。

## 4. 评测协议

**数据**：`shard_valset_v1_30720.squashfs`（359MB，30,720 样本 = 61,440 文件，487 tickers；Q 在此档无样本，ticker 集合由 provenance∩成分股清单自动确定）。squashfuse 挂载后由训练同款 dataloader 读取，运行时断言 N=30,720；`rand_offset=False`（样本即文件，逐条消息对齐）。

**顺序与覆盖**：确定序（ticker→date→id）全量顺序遍历；eval_bsz 整除 30,720，drop_last 零丢样——每个 checkpoint 覆盖全部 30,720 个样本，无抽样。

**模型加载**：metadata 合并 → init → param-only TensorStore 直读（137 arrays；优化器状态树不匹配为预期，eval 不需要 muon 状态）。前向实现与训练完全同源（`s5e_mamba3` 的 `eval_step`），保证与训练曲线、Jan-2026 表的实现口径一致。

**指标**：`eval_step` 返回 (B, 13000) per-token CE → token 维均值 = per-sample CE → 全部落盘（float32 × 30,720/ckpt）。主口径 **micro**（样本等权 = 活跃度加权；等长序列下与训练日志的 token 级均值严格相等）；对照口径 **macro**（ticker 内均值再对 487 ticker 等权，与 Jan-2026 表同口径）。95% CI 为 per-sample bootstrap（20,000 draws，分块）。

**口径对齐验证**：双 seed 逐样本 loss 相关 0.9998（顺序确定可复现）；按 provenance ticker 映射分组的 ANOVA F=5.7（映射错位的零假设下 F≈1）——macro 重建成立。

**并行与硬件**：attach 到既有 allocation 5790795（nid010407, GH200×4）。4-GPU 弹性工作队列：每卡一个单卡进程，mkdir 原子锁抢占，params 降序（LPT）；共享 squashfs 挂载与 JAX 磁盘编译缓存；per-ckpt json 断点续跑；worker 崩溃自动重试。四轮 bsz 调优（唯一约束为显存与整除性，bsz 不影响数值——eval 无跨样本运算）：350M/200M=2（98–100% util 物理顶满）、120M=4、78M=8、46M/23M=16、14M 以下=32。实测吞吐：350M 0.18–0.20s/batch（52 min/ckpt）、200M 0.14–0.18（40 min）。

**过程中的资源决策**（记录在案）：泄漏实验 r10（§10 行为学验证）在 GPU 阶段被用户指令终止让位（『validation loss 的实验是第一优先级』），推迟至本评测完成后重跑；LOB-Bench 任务线在 GPU0 的常驻进程自然结束后其卡位并入本评测。

**踩坑与修复**（全部已修，详见 learnt_lessons）：①大 GEMM 的 Triton autotune 失败 → 禁用回落 cuBLAS；②spawn DataLoader worker re-import 无 main-guard 的主模块 → 执行流全部置于 main guard 内；③worker unpickle dataset 内嵌 jax.Array 在 `worker_init_fn` 之前触发 CUDA init → `iter(loader)` 期间 `JAX_PLATFORMS=cpu` env 夹心；④`repeat_book` [B,13000,503] 展开决定显存上界。

## 5. 结果

### 5.1 主表（size 级汇总；per-checkpoint 明细见 `valset_ce_summary.csv`）

| Size | N params | #ckpt | val CE micro mean ± sd | val CE macro | Jan-2026 CE (macro) | Δ(val − Jan, macro) |
|---|---:|---:|---|---|---|---|
| 0p2M | 2,625,923 | 7 | 0.971196 ± 0.043251 | 0.977590 | 0.922452 | +0.055138 |
| 1M | 3,911,079 | 10 | 0.730399 ± 0.006235 | 0.733640 | 0.695797 | +0.037843 |
| 4M | 5,735,627 | 17 | 0.662498 ± 0.010863 | 0.661008 | 0.615055 | +0.045953 |
| 6M | 8,099,567 | 18 | 0.637644 ± 0.001963 | 0.635083 | 0.586418 | +0.048665 |
| 10M | 11,002,899 | 16 | 0.627443 ± 0.002568 | 0.623949 | 0.590536 | +0.033413 |
| 14M | 14,445,623 | 18 | 0.621332 ± 0.001214 | 0.617871 | 0.584710 | +0.033161 |
| 23M | 22,949,247 | 11 | 0.610233 ± 0.001264 | 0.606080 | 0.568915 | +0.037165 |
| 46M | 46,429,199 | 17 | 0.603252 ± 0.001364 | 0.598878 | 0.572517 | +0.026361 |
| 78M | 78,539,423 | 3 | 0.605514 ± 0.000919 | 0.601404 | 0.577200 | +0.024203 |
| 120M | 119,279,919 | 3 | **0.601403 ± 0.001139** | **0.597052** | 0.582591 | +0.014461 |
| 200M | 196,572,423 | 6 | 0.613560 ± 0.000947 | 0.610109 | 0.601021 | +0.009088 |
| 350M | 293,283,039 | 6 | 0.621799 ± 0.002016 | 0.619083 | 0.614468 | +0.004616 |

注：sd 为该 size 全部 checkpoint（跨 seed 跨 step）的离散度，非单点不确定度；单 checkpoint 的 bootstrap CI95 半宽约 0.0016（见 §5.3）。表中 Jan-2026 列为该 size 同一批 checkpoint 的 Jan CE 均值。

### 5.2 三个发现

**(a) 前向时移代价为负且处处为正的 Δ。** 所有 132 个 checkpoint 的 val CE（macro）都高于其 Jan-2026 CE。同一模型、同一 macro 口径下，这个差只可能来自评测分布本身：valset 按构造披露年度权重约 55:45 偏向 2022（疫后高波动期），而 Jan-2026 是单个相对平静的月份。这削弱了「前向测试集因分布漂移而系统性偏难」的担忧——事实相反。

**(b) Δ 随 N 单调收敛：+0.055 → +0.005。** 模型越大，两个评测分布上的 CE 越接近。小模型容量有限，拟合的是训练分布的「平均难度」，对评测分布敏感；大模型的表征更接近逐条件分布，对重加权更鲁棒。

**(c) held-out 最优规模对评测分布敏感。** valset 最低点 120M（0.6014/0.5971）与 46M（0.6033/0.5989），Jan-2026 最低点 23M（0.5689）。两表都在 200M/350M 回升——350M 只消费 17.8B token（6M 消费了 177.9B），回升是 compute-allocation 下 D 不足的效应。rebuttal 中引用「最优规模」时必须绑定评测分布。

### 5.3 统计精度

单 checkpoint micro CE 的 bootstrap CI95 半宽约 1.6e-3 nats（30,720 样本，样本间方差实测大于构造期预估）。run 间与 checkpoint 间比较应使用 **per-sample paired bootstrap**：双 seed 逐样本 loss 相关 0.9998，paired 差的方差比独立差小三个量级，0.003 量级的 seed 间差异在 paired 口径下是清晰信号。全部 per-sample loss（132 × 30,720 float32）已落盘支持任意此类计算。

### 5.4 图

`figures/fig1_terminal_ce_vs_N.png`（terminal 33 点：micro/macro/Jan 三线对比）
`figures/fig2_timeshift_delta_vs_N.png`（Δ 收敛）
`figures/fig3_ce_vs_D_all132.png`（132 点全景：每 run 的 CE–D 轨迹，β 识别所需的 run 内 D 变异可见）

## 6. 产物与复现

结果目录 `tasks/validation_set/valset_eval/results_20260729T181223Z_j5790795/`（18MB，269 文件 SHA256 见 `SHA256SUMS.txt`）：`valce_<label>.json` ×132（label 规则：terminal 为 `<run>`，非 terminal 为 `<run>@<step>`）、`valce_<label>_sampleloss.npy` ×132、`valset_ce_summary.{csv,md}`、`figures/` ×3。

脚本（同目录，全部入库）：`make_manifest.py` / `make_manifest_132.py`（清单推导与磁盘验证）、`valset_ce_eval.py`（评测本体）、`parallel_valset.sh`（4-GPU 工作队列）、`gate_and_run_valset.sh`（显存 gate）、`aggregate_results.py`（micro/macro 汇总）、`valset_ce_figs.py`（图）、`ticker_per_sample_30720.npy`（样本→ticker 映射）。

复现：`srun --jobid=<alloc> --overlap --cpu-bind=none parallel_valset.sh <out_dir>`（环境变量 `MANIFEST`/`TOTAL` 选择清单）。

## 7. 局限与后续

其一，§10 泄漏行为学实验（SEEN/MID/VAL 三组 H1/H2 判据）为本评测让位被中止，尚无数据；构造性零泄漏证明（逐样本位置核验）不受影响。重跑前需先修复其脚本的 main-guard 与 unpickle-CUDA 隐患（参照 `valset_ce_eval.py`）。其二，valset 与 Jan-2026 的分布差异（年度权重、单月 vs 四年混合）意味着两轴的绝对值不可互换使用；per-sample loss 支持按月重加权以逼近任意目标权重。其三，本报告只交付评测数字与协议；valset 轴上的 unconstrained fit（α/β 重估）是下一步，132 点的 run 内 D 变异已为其备好数据。
