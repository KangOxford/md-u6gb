# Handoff：Jan-2026 shuffle 版 132-ckpt CE 评测 ✅ 已全部完成（2026-07-31T12:25Z）

> **本文档已完成使命，保留作执行记录。** 132/132 评测完成（job 5848062，COMPLETED 0:0，1h25m），
> Approach 3 三轴拟合完成（500 draws 零失败），报告写回、self-complete 子包（285 条 SHA256 自验通过）
> 与 Notion 回填（页面 `fit scaling law on validation loss`）全部落地。
> 结论见 `VALSET_CE_EVAL_REPORT.md`「第三把尺子：Jan-2026 shuffle」节与本文件 §10。
> 以下 §1–§8 为中断当时的续跑说明，已成为历史。

写于 UTC 2026-07-30T22:3xZ。拿到本文档即可无上下文续跑并完成全链条。所有路径均为绝对路径。

## 0. 一句话

在 **Jan-2026 全月窗口池随机 30,720 窗**（完全 shuffle、自然分布）上对 **132 个 checkpoint**（33 logical runs × final-25% 窗口）算 CE，作为第三把尺子（前两把：valset_v1、现行 Jan ticker-等权口径），跑完后汇总三口径对比并用 rebuttal_analysis 机器对该轴重拟合 Approach 3。

**缘起（用户指令原文）**：「这个test set上不是应该随机采样一个 1% training set 左右的大小的子集 (要完全shuffle ) 然后做测试吗?」+「这个估计你是不是还没有做/」→ 确认没做过，开工。

## 1. 当前状态（截至 2026-07-30T22:30Z）

| 项 | 值 |
|----|-----|
| 已落盘 | **71/132** json（+ 同名 `*_sampleloss.npy`） |
| 结果目录 | `/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/valset_eval/results_jan_shuffle_20260730T133812Z_j5823145/` |
| 目录指针 | `/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/valset_eval/latest_jan_results_dir.txt`（内容即上行目录，续跑必须复用它，**不要**新开目录） |
| 中断原因 | 宿主 job **5823145** 走满 24h walltime 整体 COMPLETED（Elapsed 23:55:04，End 2026-07-30T22:24:22），attach 的 step .29 随之 CANCELLED。非故障。 |
| 接力宿主 | **5827830**（u6gb-1-node-chain, 1 node, 24h），中断时 PENDING (Priority)。另有 5836919（u6gb-4-node-chain）也 PENDING 可作备选。 |
| 等待循环 | 后台 Bash b8x74g4r4：`until squeue -j 5827830 …RUNNING` 每 120s 轮询，转 RUNNING 即唤醒会话 |
| 已完成段 | 350M/200M/120M/78M/46M/23M/14M 全部 + 10M 大部（params 降序 LPT 队列） |
| 剩余 61 个 | 10M 尾部 + 6M/4M/1M/0p2M，全为 eval_bsz=32/GPU 快档，4 卡预计 **~1h** |

## 2. 接力动作（job 起来后逐字执行）

```bash
# ① 确认 5827830 RUNNING 且节点 GPU 空闲（1-node-chain 是 placeholder，应为 4×GH200 各 ~1MiB）
squeue -j 5827830 -o "%.10i %.8T %N %.12L"
srun --jobid=5827830 --overlap --ntasks=1 --cpu-bind=none \
  nvidia-smi --query-gpu=index,memory.used --format=csv

# ② 续跑（OUT_DIR 复用旧目录 → 71 个已有 json 自动跳过；孤儿锁由脚本启动时自动 rmdir）
cd /lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/valset_eval
OUT_DIR=$(cat latest_jan_results_dir.txt)
MODE=jan \
MANIFEST=/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/valset_eval/manifest_132ckpt.json \
TOTAL=132 \
srun --jobid=5827830 --overlap --ntasks=1 --cpu-bind=none \
  bash parallel_valset.sh "$OUT_DIR" \
  >> /lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/logs/jance_parallel_j5827830.out 2>&1 &

# ③ 新 monitor 盯（worker log 名含新 jobid：SLURM_JOB_ID 在 srun 远端=5827830）
#    盯文件: logs/jance_parallel_j5827830.out + logs/valce_par_gpu{0,1,2,3}_j5827830.out
#    过滤: "===== |val_ce_mean|workers done|PARALLEL_VALSET_OK|Traceback|RESOURCE_EXHAUSTED|CANCELLED|rc=[^0]"
```

要点：`MODE=jan` 时 Jan 参数（squashfs 路径、sampler、n_expect、date_range、provenance）**在 `parallel_valset.sh` 脚本内部组装**，环境只传 MODE/MANIFEST/TOTAL 三个无逗号标量。这是刻意设计，见坑 #1。

## 3. 完成信号与完整性验证

- launcher log 出现 `workers done: 132/132 jsons` + `PARALLEL_VALSET_OK`。
- **`PARALLEL_VALSET_OK` ≠ 全部完成**：它只表示 launcher 正常走完（历史上曾在 1/132 时打出，因四 worker 全死两次后退出）。必须看 `workers done: N/132` 的 N。
- 验证（对全部 132 个）：每 json 的 `n_samples==30720`；`valce_<label>_sampleloss.npy` 长度 30720 且 `mean(npy)` 与 json `val_ce_mean` 差 `<1e-5`；json 数 `ls $OUT_DIR/valce_*.json | wc -l` == 132。

## 4. 完成后的既定链条（用户已授意，勿再问）

1. **三口径对比表**：valset(macro/micro) × Jan ticker-等权（现行口径）× Jan-shuffle 自然分布，132 行主表补列。已知趋势：Jan-shuffle 全面更低，350M 差 ~0.09 nats（月初时段偏置 >> 加权效应）。
2. **Jan-shuffle 轴 Approach 3 重拟合**：复用 `/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/scaling_law_plots/valset_ce_eval_20260730/code/valset_fit_approach3.py` 的骨架（其内 `import rebuttal_analysis as ra` 全量复用 fit_model/select_protocol/bootstrap_fits）。新 frame：L=Jan-shuffle CE。注意 Jan-shuffle 的 micro 即自然口径；macro 可用 `jan_ticker_per_sample_30720.npy` 重建。在计算节点跑（attach step），不要 login 跑 bootstrap。
3. **写回**：VALSET_CE_EVAL_REPORT.md 增补 Jan-shuffle 节 → self-complete 交付目录（`…/scaling_law_plots/valset_ce_eval_20260730/`，复制不是 mv，更新 SHA256SUMS）→ Notion 回填 → 四件套。

## 5. 关键路径总表

| 物 | 路径 |
|----|------|
| 评测本体 | `/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/valset_eval/valset_ce_eval.py` |
| 4-GPU launcher | `/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/valset_eval/parallel_valset.sh`（`MODE=jan` 分支在第 19-23 行） |
| 132-ckpt manifest | `/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/valset_eval/manifest_132ckpt.json`（含每档 micro_bsz） |
| Jan 窗口清单生成器 | `/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/valset_eval/make_jan_shuffle_manifest.py`（seed=20260131） |
| sampler 索引 | `/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/valset_eval/jan_shuffle_30720_indices.npy`（30,720 个全局窗口索引，升序） |
| per-sample ticker | `/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/valset_eval/jan_ticker_per_sample_30720.npy` |
| 池 ticker 全表 | `/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/valset_eval/jan_pool_ticker_all.npy`（运行时对齐断言用，n=7,507,307） |
| 清单元数据 | `/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/valset_eval/jan_shuffle_manifest_meta.json` |
| Jan 数据本体 | `/lus/lfs1aip2/projects/public/s5e/quant_team/lob_preproc_sp500_squashfs/shard_2026-01.squashfs`（164.1GB, 487 tickers 实际, 缺 BAC, 20 交易日） |
| 结果目录 | `results_jan_shuffle_20260730T133812Z_j5823145/`（见 §1，经 latest_jan_results_dir.txt 解引用） |
| launcher log（旧） | `/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/logs/jance_parallel_j5823145.out` |
| worker log（旧） | `/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/logs/valce_par_gpu{0-3}_j5790795.out`（名字历史遗留 j5790795，实际内容含 5823145 轮） |
| valset 任务交付目录 | `/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/scaling_law_plots/valset_ce_eval_20260730/`（前序任务 self-complete 包，fit 代码在其 code/ 下） |
| fit 复用源 | `/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/scaling_law_plots/aramis/rebuttal_analysis.py`（只读） |
| 主报告 | `/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/VALSET_CE_EVAL_REPORT.md` |

## 6. 已有结果速览（71 个的形状）

Jan-shuffle CE（自然分布、全月随机时点）按 size 段：350M≈0.517-0.522，200M≈0.511-0.519，120M≈0.500-0.511，78M≈0.4997-0.5013，46M≈0.4997-0.5046，23M≈0.5026-0.5093，14M≈0.5062-0.5206，10M≈0.5160-0.5222（s5 terminal 0.5214，s42 terminal 0.5202）。

对比现行 Jan 口径（每股等窗、集中月初时段）：**全面低 ~0.09 nats（350M 档）**，说明现行口径的主要偏差来自月初时段选择而非 ticker 加权（此前量化过：自然加权 vs ticker 等权在大模型上差 ≈0，在 0p2M 上差 -0.080）。最优 size 在三把尺子上的移动是核心看点。

## 7. 坑清单（续跑者必读）

1. **srun `--export` 值内逗号截断**：含逗号/空格的复合串（如 EXTRA_ARGS、date_range）绝不能经 `--export`/环境传给 srun，会在逗号处被当成变量分隔符截断（2026-07-30 事故：provenance 丢失 → BAC assert 爆）。解决：`MODE=jan` 开关让脚本内部组装。续跑命令只传三个无逗号标量，安全。
2. **spawn DataLoader 两层坑**（已修，勿回退 valset_ce_eval.py）：worker re-import 主模块需 main guard；dataset 内嵌 jax.Array 在 worker unpickle 时触发 CUDA init，需在 `iter(loader)` 期间用 `JAX_PLATFORMS=cpu` 环境夹心。
3. **attach 宿主 walltime 是硬截止线**（本次中断根因）：attach 前查 `squeue -j <jid> -h -o %L`，要求「剩余 walltime ≥ 队列 ETA × 1.5」。5827830 是全新 24h job，剩余 61 个 ~1h，余量充足。
4. **孤儿锁**：step 被杀时正在跑的 ckpt 留下 `lock_*` 无 json。launcher 启动时（第 37 行）自动清理「无 json 的锁」，无需手动。但**并发双 launcher 会互踩锁**，确保同一时刻只有一个 launcher。
5. **显存 gate**：GATE_MB=2000，连续 2 次采样 <2GB 才放行。新链节点应为空，gate 秒开；若 chain job 本身跑了别的 GPU 任务，worker 会安静等待而非报错——log 停在 `gate: waiting` 属正常。
6. **Triton GEMM autotune**：`--xla_gpu_enable_triton_gemm=false` 已在 launcher 内常备（r3 教训），勿删。
7. **bsz 不影响数值**：纯 forward 无跨样本运算，manifest 内 micro_bsz 只是速度调优；但 `bsz×devices` 必须整除 30,720。
8. **Lustre 纪律**：绝不 `ls` 大目录/递归 find；done_count 用的 `ls $OUT_DIR/valce_*.json` 是 132 量级小目录，安全。

## 8. 会话恢复钩子

- 等待循环：后台任务 b8x74g4r4（本会话）。若会话已死，直接人肉 `squeue -j 5827830` 后执行 §2。
- 四件套最新条目：P137 / F131 / PG1785450367 / L1785450367（均 2026-07-30T22:26Z，root `/projects/public/u6gb/*.md`）。
- 任务清单：Task #10「Jan-2026 shuffle 版评测」in_progress。

## 9. 状态更新（2026-07-31T09:5xZ，另一会话补记，续跑者必读）

- 5827830 已于 2026-07-31T00:39 RUNNING（nid010937），但 §2 的 jan-shuffle 续跑**至今未执行**（launcher log 不存在、锁全为昨晚孤儿、json 仍 71/132）。
- **09:37 起该节点 4 GPU 被 backfill124 评测占满**（78.5GB×4，`valset_ce_eval.py --manifest manifest_backfill124.json`，结果目录 `results_backfill124_20260731T093723Z_attach5827830/`，由另一活跃会话驾驶）。backfill124 重档 58 个（350M×10/200M×12/120M×18/78M×18），参照昨天速率估 **8-12h**。
- **walltime 数学**：5827830 剩 ~15h（09:38 时点）。backfill(~10h) + jan-shuffle(~1h) 串行余量薄；若 backfill 超估，jan-shuffle 将重演 5823145 的 walltime 收割（§7 坑 3 / L1785450367）。备选：5836919（4-node chain，PENDING）起跑后可把 jan-shuffle 挂过去与 backfill 并行。
- ⚠️ **单 launcher 约束（§7 坑 4 的跨会话版）**：任何会话执行 §2 前，必须先确认 (a) 无其他 jan-shuffle launcher 在任何宿主上运行（查 `logs/jance_parallel_j*.out` mtime、`squeue -j <候选宿主> -s`、OUT_DIR 新锁），(b) 若挂到 5827830 需等 backfill124 完成（GPU gate 在 backfill 的 per-ckpt 进程间隙可能假开，禁止并行挂）。认领后立即在本节追加一行「launcher 已由 <会话/宿主> 于 <UTC> 启动」再开跑。
- **认领（2026-07-31T09:51:14Z）**：jan-shuffle 收尾改走**独立 job 5848062**（jance-finish61, 1N/2.5h, finish61_jan_shuffle.sbatch, OUT_DIR 复用断点目录）。**5827830 不挂 jan launcher**（本节 (b) 约束遵守）。若 backfill124 完成时 5848062 仍 PENDING，切换宿主前必须先 scancel 5848062（双 launcher 互清活锁，§7 坑 4）。

## 10. 完成记录（2026-07-31）

### 实际执行路径（与 §2 预案的差异）

§2 预案是 attach 到 5827830 续跑。实际未走这条路：5827830 自 09:37 起被另一会话的 **backfill124** 评测占满 4 GPU（8-12h），
GPU gate 不可能开。改为提交**独立短 job** `finish61_jan_shuffle.sbatch`（1 node / 4 GPU / 2:30），
即 **job 5848062**（10:44:20Z 起，nid011192），口径仍按 §2：`MODE=jan` + 全量 `manifest_132ckpt.json` + `TOTAL=132`，
OUT_DIR 复用断点目录靠已有 json 跳过。12:09:33Z 完成，`COMPLETED 0:0`，Elapsed 01:25:20，
`workers done: 132/132` + `PARALLEL_VALSET_OK` 双信号齐备，零 worker 错误。

### 结果（132/132）

三把尺子的最优规模：valset macro **120M** ｜ Jan-2026 现行 **23M** ｜ Jan-shuffle macro **120M** ｜ micro **120M**。

Approach 3（500 draws，2000 次拟合零失败）：

| 轴 × 协议 | α 点估计 | α boot 中位 [CI95] / 顶界占比 | β [CI95] | E |
|---|---|---|---|---|
| Jan-shuffle, last25 | 1.893 内点 | 1.943 [1.828, 2.000] / **35.6%** | 1.405 [1.266, 1.881] | 0.5234 |
| valset, last25 | 1.922 内点 | 1.936 [1.826, 2.000] / 34.0% | 1.337 [1.197, 2.000] | 0.5957 |
| Jan-2026 现行, last25 | 2.000 顶界 | — / **100%** | 0.801 [0.527, 1.509] | 0.5552 |

**核心结论**：Jan-shuffle 的顶界占比与 valset 几乎无法区分（35.6% vs 34.0%），而现行口径是 100%。
因 Jan-shuffle 与现行口径共享同月同批数据、valset 是完全不同数据集，拟合行为上 Jan-shuffle 却站到 valset 一侧，
说明划分「病理 vs 正常」的是**采样设计**而非数据分布。现行口径的 100% 顶界不能用「Jan-2026 这个月特殊」辩护。

**偏差结构**（这才是最优点左移的原因）：`valset − shuffle` 近乎常数（1M–350M 全落在 +0.0720~+0.0754），
常数偏移不移动最低点；`Jan现行 − shuffle` 在大模型端从 23M +0.0361 单调爬到 350M +0.0669（近乎翻倍），
把曲线尾部掰高使最低点左移。整体呈 U 形（谷底 6M +0.0253），小模型端非单调不影响结论。

**micro 不与 macro 同构**（勿套用 valset 节措辞）：Jan-shuffle micro 顶界占比 50.6%/51.4%，比自身 macro 高 15pp，
valset 只差约 3pp。机制用 Kish 有效样本量量化：valset 487 只折合 164.6 个等效单元（33.8%），Jan-shuffle 仅 74.6（15.3%）。

### 产物落点

| 物 | 路径 |
|----|------|
| 报告节 | `/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/VALSET_CE_EVAL_REPORT.md`「第三把尺子」节 |
| self-complete 子包 | `/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/scaling_law_plots/valset_ce_eval_20260730/jan_shuffle/`（285 条 SHA256 自验通过） |
| 主表 | 子包 `tables/jan_shuffle_132_table.csv`、`tables/three_ruler_comparison.{csv,md}` |
| 拟合 | 子包 `fits/`（main_fits / bootstrap_{samples,summary} / three_axis_fit_comparison） |
| 图 | 子包 `figures/fig_three_rulers.png` |
| Notion | 页面 `fit scaling law on validation loss`，id `3ad12c45-68fd-80ee-8f6a-e656a3761028`，2026-07-31T12:25Z 追加 7 block |
| 工作目录 | `/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/valset_eval/`（原样保留，复制非移动） |

### 新增脚本（均已冒烟验证并入子包 code/）

`build_jan_shuffle_table.py`（三道闸门 + 三尺并表 + macro 重建）、`jan_shuffle_fit_approach3.py`（Approach 3 三轴）、
`plot_three_rulers.py`（出图）、`run_jan_shuffle_downstream.sh`（下游一条龙）、`finish61_jan_shuffle.sbatch`（评测）。

三道闸门：`n_samples == 30720`；`|mean(npy) − json.val_ce_mean| < 1e-5`；
ticker↔sample 逐位对齐用单因素 ANOVA 加打乱阴性对照（对齐 F 24.9–31.9，打乱后塌到 0.87–1.12），任一不过即中止。

### 补充教训（§7 坑清单的续）

9. **placeholder chain 节点是多任务共享资源**：§2 写「等 5827830 起跑后 attach」时未料到别的工作线（backfill124）会先占住同一宿主。
   凡「等 job X 起跑后挂 Y」的计划都必须查**节点物理 GPU 占用**（`nvidia-smi --query-compute-apps`）而非只看 job 状态，
   且需要跨会话认领机制（本次靠 §9 文档声明行）。
10. **监控脚本末尾的诊断性 grep 会制造 exit-1 假警报**：本轮 monitor 正常跑到 `132/132 ALL DONE`，
    末尾 `grep -lE 错误模式` 因无匹配返回 1，整个后台任务被标记 failed。末尾 grep 必须 `|| true`。
11. **部分数据推出的趋势必须用全量复核**：91/132 时 `Jan−shuffle` 差值看起来全程单调递增，
    全量后暴露为 U 形（缺的正是 1M/4M 两个小档，属系统性缺失而非随机缺失）。
12. **不要跨轴照搬前一轴的结论句**：valset 节的「micro 与 macro 同构」在本轴是错的（差 15pp vs 3pp）。
    结构可复用，结论句必须逐条重验，尤其「X 与 Y 无差异」这类否定式断言。
