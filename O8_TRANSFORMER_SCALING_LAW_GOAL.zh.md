# /goal: O8 Transformer Scaling Law v2 当前进度核查、英文图表站点与 Notion 回写

## 2026-05-20 修正版范围

本 goal 的核心任务是核查 **Transformer / O8 scaling-law v2 当前进度**，不是泛化的 LOBbench 总结，也不是把旧 summary 里的“11-size ladder 已完成”当作事实。

执行时必须以这些 v2 Transformer 入口为主：

```text
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/scaling_law_plots/SCALING_LAW_PLAN_V2_LOCKED_20260426_v6.md
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/scaling_law_plots/v2-plans-and-results/plans/SCALING_LAW_PLAN_V2.md
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/scaling_law_plots/v2-transformers-plan-and-results/plan-table
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/scaling_law_plots/v2-transformers-plan-and-results/progress-table
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/scaling_runs_live_jobs.md
```

网页必须用英文。中文只用于本 goal 文件和给用户的简短说明。

当前已知、必须重新验证的 v2 结论候选：

- v2 O8 Transformer plan 预计 31 个 job、561.5 node-hours，覆盖 0p2M 到 200M。
- 当前 `v2-transformers-plan-and-results/progress-table` 只强证明 0p2M seed=5 的早期 bring-up attempts。
- 最新真实 O8 job 是 `4524395`，状态为 `COMPLETED 0:0 but INCOMPLETE POINT`，只到 `6120/7438` step，即 82.3%，不是有效最终 scaling-law point。
- `4540592` 和 `4540601` 如果仍然查不到 scheduler / ledger / manifest / log 证据，必须标为 `NOT FOUND`，不能当作完成 job。
- `/lus/lfs1aip2/projects/public/u6gb/transformer_scaling_summary.md` 可能包含旧的或生成式“11-size 已完成”表述，必须降级为背景材料；若它与 v2 `progress-table` 冲突，以 `progress-table` 为准。

## 目标

请执行一次完整的 O8 Transformer scaling-law v2 当前进度核查。本地 goal 结果和给用户说明可以用中文，但 **HTML / surge.sh dashboard 必须用英文**。需要回答：

1. O8 Transformer scaling law 当前进度是什么。
2. v2 plan table 和 progress table 分别说明了什么：计划了多少 job / node-hours，当前真正完成到哪一步。
3. 与本任务相关的 Codex/历史会话时间戳和 session id 是什么；如果本地文件里没有直接 session id，要从记忆系统或历史记录里追溯，并明确“已验证/未找到”。
4. 当前可验证结果是什么；不要把未完成的 31-job / 11-size production sweep 说成已经全部完成，除非重新验证到确凿证据。
5. 用 figure 和英文 HTML dashboard 展示结果，部署到 surge.sh。
6. 把英文当前进度报告和 dashboard 链接回写到 Notion，作为原 Notion 页的子页面或更新已有子页。
7. LOBbench / LOBS5 结果只作为背景或 cross-architecture context；不要让它替代 v2 Transformer progress-table 的结论。

用户给定 Notion 页：

```text
https://www.notion.so/O8-transformer-scaling-law-progress-checking-36612c4568fd80babc31dd007846248a?source=copy_link
```

已知该 Notion 页在 2026-05-20T15:03:34Z 抓取时内容为空，父页为 `tasks april 2026`：

```text
page: https://www.notion.so/36612c4568fd80babc31dd007846248a
parent: https://www.notion.so/34d12c4568fd8080a773e9d5de348efe
title: O8-transformer-scaling-law progress checking
```

## 最重要的执行原则

- 先验证数据，再写结论。
- 区分三类结果：
  - v1 / baseline 6-size loss curves：已有 CSV 可验证。
  - FLOPs / MFU probes：已有 CSV 可验证。
  - v6 / production 11-size ladder：manifest 和 job ledger 可验证，但不要默认全部完成。
- 公开 surge.sh 页面不要暴露过多内部绝对路径、用户名、私有 job 细节，除非用户明确要求公开这些内容。Notion 内部子页和本地 markdown 可以列完整核查入口。
- 报告里要明确写“数据源时间戳”和“核查时间戳”。
- dashboard 的可见文案必须是英文。

## 历史记忆入口

优先从记忆系统取这些 observation / session 线索。若有 `get_observations` 或 `mem-search`，请读取或搜索下列 ID/关键词：

### 直接相关 observation ID

```text
732  2026-05-18 11:51 UTC  Scaling Law v2 Notion Page — Transformer Experiment Paths and Task
733  2026-05-18 11:52 UTC  Transformer (O8) Scaling Law Experiment Structure Confirmed
734  2026-05-18 11:53 UTC  Scaling Law Plots Directory Contains Pre-Built CSV Data Files for All Architectures
735  2026-05-18 11:53 UTC  Transformer Scaling Law Data Sources Identified — No Pre-Built HTML Dashboard
736  2026-05-18 11:54 UTC  Scaling Law v2 Experiment Structure — exp_O8 Transformer and exp_R1 Mamba3 Sweep Paths Identified
737  2026-05-18 11:54 UTC  exp_O8 Transformer Scaling Law — 11-Size Production Grid and Training Data Located
739  2026-05-18 11:55 UTC  Scaling Law Plots Directory — Full CSV Inventory and Transformer FLOPs Data Located
740  2026-05-18 11:55 UTC  Multi-Architecture LOBbench Comparison Table — Transformer xfmr-o8-125m at CE=0.609 vs Mamba3-78m at CE=0.559
741  2026-05-18 11:56 UTC  transformer_scaling_summary.md Created — Full O8 Experiment Summary for Dashboard Generation
742  2026-05-18 11:57 UTC  train_ce_transformer_wandb.csv Only Contains 6 of 11 Model Sizes — Missing Smallest and Largest Models
743  2026-05-18 11:57 UTC  Exact Transformer Final CE Values Extracted — 6-Size Benchmark Sweep, ce_vs_flops Has Only 24tok 125M Reference
744  2026-05-18 11:58 UTC  exp_O8 Live Job Ledger Reveals 0.2M Model Chronic Failures on Specific Bad Nodes — 41 Total Submissions for Single Size
745  2026-05-18 11:59 UTC  Precise Transformer Min/Final CE Values Extracted — tf-46M Achieves Best Min CE=0.527; tf-8m Still Declining at Final Step
746  2026-05-18 11:59 UTC  Surge Skill SKILL.md Defines Mandatory 14-Section Dashboard Structure
747  2026-05-18 12:00 UTC  Surge Skill Caveat — Auto-Mode Classifier Blocks Publishing Internal HPC Paths/Job IDs Without Explicit Authorization
748  2026-05-18 12:02 UTC  extract_data.py Created and Executed — Generates data.json with 480 Downsampled Curve Points + FLOPs + v6 Grid
749  2026-05-18 12:02 UTC  data.json Shows Duplicate Step Entries — train_ce_transformer_wandb.csv Contains Repeated Data Points
750  2026-05-18 12:03 UTC  Duplicate CSV Rows Fixed in extract_data.py — sorted(set(pts)) Reduces Curve Points from 480 to 228
753  2026-05-18 12:22 UTC  Transformer (exp_O8) Scaling Law — Two-Phase Update Task Initiated
754  2026-05-18 12:23 UTC  exp_O8 Transformer v1 Sweep Checkpoints (j4130177-j4130182) Are Entirely Missing
755  2026-05-18 12:23 UTC  exp_O8 Checkpoint Inventory — 26 Real vs 37 Empty (Metadata-Only) Directories
```

### 记忆搜索关键词

```text
O8 transformer scaling law
exp_O8 self attention
train_ce_transformer_wandb
transformer_flops_measured
ce_vs_flops_all_arch
LOBbench
Mamba3 78M
xfmr-o8-125m
tf-46M best min CE
tf-78M undertrained
surge dashboard scaling law
```

### 关于 session id

必须尝试找出“Codex session id”。候选线索：

- 2026-05-18 11:51-12:03 UTC 这段是最相关的数据抽取与 dashboard 生成会话。
- 2026-05-18 12:22-12:23 UTC 这段是后续 checkpoint / two-phase update 会话。
- 如果记忆系统只返回 observation ID，没有 session id，请在报告里写：
  - `相关历史时间窗：2026-05-18 11:51-12:03 UTC；2026-05-18 12:22-12:23 UTC`
  - `可追溯 observation IDs：732-750, 753-755`
  - `session id：未在可访问本地文件/记忆摘要中直接暴露，需要上游记忆系统进一步解析`

## 本地核查入口：u6gb 工作区

当前工作目录：

```text
/lus/lfs1aip2/projects/public/u6gb
```

优先检查：

```text
/lus/lfs1aip2/projects/public/u6gb/transformer_scaling_summary.md
/lus/lfs1aip2/projects/public/u6gb/transformer_scaling_site/
/lus/lfs1aip2/projects/public/u6gb/transformer_scaling_site/index.html
/lus/lfs1aip2/projects/public/u6gb/transformer_scaling_site/dist/index.html
/lus/lfs1aip2/projects/public/u6gb/transformer_scaling_site/data.json
/lus/lfs1aip2/projects/public/u6gb/transformer_scaling_site/data.compact.json
/lus/lfs1aip2/projects/public/u6gb/transformer_scaling_site/extract_data.py
```

注意：

- `transformer_scaling_summary.md` 里可能有“11-size 全完成”的旧表述，要用原始 CSV/manifest 复核，不可直接照抄。
- `transformer_scaling_site/core` 是 core dump，不要读取或发布。
- `transformer_scaling_site/index.html` 是已有 dashboard 草稿，可复用样式和 Chart.js 结构，但要修正任何过时结论。

## 本地核查入口：AlphaTrade / exp_O8

O8 主目录：

```text
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/
```

必须检查的核心文件：

```text
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/scaling_runs_manifest.tsv
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/scaling_runs_live_jobs.md
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/scaling_law_sweep.sh
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/scaling_train.batch
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/scaling_node_wrapper.sh
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/scaling_job_ledger.sh
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/scaling_logs/
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/checkpoints/
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/wandb/
```

Agent 历史输出：

```text
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/agent_outputs/session_benchmark_results.md
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/agent_outputs/job_3360360_analysis.md
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/agent_outputs/job_3378771_status.md
```

测试 CE 文件：

```text
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/test_ce_tf-8m.csv
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/test_ce_tf-14m.csv
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/test_ce_tf-23m.csv
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/test_ce_tf-34m.csv
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/test_ce_tf-46m.csv
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/test_ce_tf78m.csv
```

旧的 24-token buggy 文件也要识别，但不要当作最终结果：

```text
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/test_ce_tf-8m_24tok_buggy.csv
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/test_ce_tf-14m_24tok_buggy.csv
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/test_ce_tf-23m_24tok_buggy.csv
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/test_ce_tf-34m_24tok_buggy.csv
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/test_ce_tf-46m_24tok_buggy.csv
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/test_ce_tf78m_24tok_buggy.csv
```

其他相关脚本/文档：

```text
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/aggregate_lobbench_curves.py
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/run_lobbench_sweep.sh
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/eval_test_ce.py
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/eval_test_ce.batch
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/run_train.py
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/train_full_autoreg.batch
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/learned_lessons.md
```

## 本地核查入口：scaling_law_plots

主数据目录：

```text
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/scaling_law_plots/
```

必须检查：

```text
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/scaling_law_plots/train_ce_transformer_wandb.csv
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/scaling_law_plots/transformer_flops_measured.csv
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/scaling_law_plots/ce_vs_flops_all_arch.csv
```

如果存在，也检查：

```text
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/scaling_law_plots/all_loss_curves.v3_with_293M_chain.csv
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/scaling_law_plots/SCALING_LAW_PLAN_V2.md
```

比较架构入口：

```text
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_R1_Mamba3/scaling_runs_manifest.tsv
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_R1_Mamba3/scaling_runs_live_jobs.md
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O2d_2d_rope_dim/scaling_runs_manifest.tsv
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O2d_2d_rope_dim/scaling_runs_live_jobs.md
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O2d_2d_rope_early_fusion/scaling_runs_manifest.tsv
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O2d_2d_rope_early_fusion/scaling_runs_live_jobs.md
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O2d_2d_rope_head/scaling_runs_manifest.tsv
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O2d_2d_rope_head/scaling_runs_live_jobs.md
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O2d_time_only_rope/scaling_runs_manifest.tsv
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O2d_time_only_rope/scaling_runs_live_jobs.md
```

## 已知但必须重新验证的事实

下面是 2026-05-20 初步核查到的事实，执行时必须从原始文件重新计算：

### 1. `train_ce_transformer_wandb.csv`

文件行数约为 2167 行，只包含 6 个 baseline size，不包含 0.2M、1M、4M、6M、10M、120M、200M 的完整 loss curves。

候选统计：

| model | points | final step | final CE | min CE | gap |
|---|---:|---:|---:|---:|---:|
| tf-8m | 418 | 51030 | 0.6089 | 0.5685 | 0.0405 |
| tf-14m | 304 | 38810 | 0.5790 | 0.5774 | 0.0017 |
| tf-23m | 456 | 58100 | 0.5501 | 0.5397 | 0.0105 |
| tf-34m | 380 | 48000 | 0.5671 | 0.5425 | 0.0246 |
| tf-46m | 342 | 40610 | 0.5267 | 0.5267 | 0.0000 |
| tf-78m | 266 | 30770 | 0.5563 | 0.5353 | 0.0210 |

关键解读候选：

- `tf-46m` 是当前 6-size baseline 里 final/min CE 最好的可验证点。
- `tf-78m` final CE 比 `tf-46m` 更差，且 final CE 明显高于 min CE，说明 78M 有 undertrained 或 schedule 末端反弹风险。
- `tf-8m` gap 最大，说明 final step 不是可靠指标；建议报告 min CE 或 last-window EMA。

### 2. `transformer_flops_measured.csv`

文件约 11 行，包含 8M/14M/23M/34M/46M/78M，以及 60m-a/90m-a/120m-a/200m-a probe。

候选事实：

- 8M MFU 约 22.6%。
- 78M MFU 约 42.0%。
- 200m-a probe MFU 约 50.474%，是当前测到的 peak MFU。
- 模型越大 MFU 越高，这是 GH200 上 dense Transformer 的有利信号。

### 3. `ce_vs_flops_all_arch.csv`

文件约 1232 行。按每个 `(arch, model)` 取最后记录时，候选结果包括：

| arch | model | CE | gBSZ | FLOPs | tokens |
|---|---|---:|---:|---:|---:|
| Transformer | xfmr_o8_125M_24tok | 0.614606 | 768 | 2.936360e20 | 390.113B |
| Mamba3 | mamba3_78M_26tok_muon | 0.539089 | 128 | 6.489899e19 | 112.886B |
| GDN | gdn_94M_26tok_muon | 0.557676 | 128 | 3.176351e19 | 50.253B |
| S5 | s5_120M_24tok_adamw_16n | 0.543455 | 256 | 3.495969e20 | 418.293B |
| S5 | s5_55M_24tok_adamw_16n | 0.640502 | 640 | 2.758709e20 | 715.638B |
| Mamba2 | mamba2_77M_24tok | 0.597849 | 128 | 1.092539e20 | 200.299B |
| GDN | gdn_94M_24tok_muon | 0.671813 | 256 | 9.834100e19 | 155.585B |
| KDA | kda_104M_24tok_k1 | 0.678134 | 128 | 4.240379e19 | 61.443B |
| MoE | moe_118M_24tok | 0.653625 | 512 | 1.984020e20 | 233.921B |

注意：历史记忆里写过 “Transformer CE=0.609 vs Mamba3 CE=0.559”，这可能来自不同聚合方式或 min/last-window 口径。执行时要同时报告“CSV last row 口径”和“历史口径”，并解释差异。

### 4. `scaling_runs_manifest.tsv`

文件约 276 行，数据记录约 275 条。候选统计：

```text
0p2M: 41 rows, last submit 2026-05-10T13:01:45+00:00
1M:   25 rows, last submit 2026-04-30T10:37:40+00:00
4M:   25 rows, last submit 2026-04-30T10:37:40+00:00
6M:   25 rows, last submit 2026-04-30T10:37:40+00:00
10M:  25 rows, last submit 2026-04-30T10:37:40+00:00
14M:  25 rows, last submit 2026-04-30T10:37:40+00:00
23M:  25 rows, last submit 2026-04-30T10:37:40+00:00
46M:  25 rows, last submit 2026-04-30T10:37:40+00:00
78M:  25 rows, last submit 2026-04-30T10:37:41+00:00
120M: 17 rows, last submit 2026-04-30T10:37:41+00:00
200M: 17 rows, last submit 2026-04-30T10:37:41+00:00
```

### 5. `scaling_runs_live_jobs.md`

候选事实：

- job ledger 最后文件修改时间约为 2026-05-11 08:44 UTC。
- `job 4524395` 是 0.2M 的一次成功重提，2026-05-10T14:20:57Z started，2026-05-10T15:14:36Z completed，elapsed 00:53:39。
- 0.2M 在 `job 4523901` 和 `job 4523971` 上出现过 exit 143 / failed。

必须检查是否有更新的 ledger 或 scheduler 状态，不要只靠这个文件。

### 6. 早期 O8 LOBbench benchmark

从 `agent_outputs/session_benchmark_results.md`、`job_3360360_analysis.md`、`job_3378771_status.md` 提取：

- 2026-03-26 benchmark session：
  - PyTorch O8 79.9M，BSZ/GPU 16/32/64，throughput 约 1205-1280 samples/s。
  - JAX O8 125.4M，BSZ/GPU 8，300 steps，steady state 约 0.520 s/step，985 samples/s。
  - JAX BSZ/GPU 12 只跑 50 steps，不可当 steady-state。
- job 3360360：
  - Started 2026-03-25 21:05:45 UTC。
  - Cancelled 2026-03-25 22:48:23 UTC。
  - 125,449,011 params，last step 约 15993/233061，latest checkpoint 15730。
  - train loss 0.6890 from checkpoint metadata。
- job 3378771：
  - status report timestamp 2026-03-27 ~03:45 UTC。
  - 125,449,011 params，step 33383/77687，43.0%，healthy。
  - W&B run id `4r3muy19`。

## 建议执行命令

从工作区根目录执行：

```bash
cd /lus/lfs1aip2/projects/public/u6gb
```

定位文件：

```bash
ls -la transformer_scaling_site
sed -n '1,240p' transformer_scaling_summary.md
find transformer_scaling_site -maxdepth 3 -type f | sort
```

检查 CSV 行数：

```bash
wc -l \
  /lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/scaling_law_plots/train_ce_transformer_wandb.csv \
  /lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/scaling_law_plots/transformer_flops_measured.csv \
  /lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/scaling_law_plots/ce_vs_flops_all_arch.csv \
  /lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/scaling_runs_manifest.tsv
```

复算 6-size loss summary：

```bash
awk -F, 'NR>1 {
  seen[$1]=1; count[$1]++; laststep[$1]=$2; lastce[$1]=$3;
  if(!($1 in min) || $3<min[$1]) min[$1]=$3
}
END {
  printf "model,count,last_step,last_ce,min_ce,gap\n";
  for (m in seen) printf "%s,%d,%d,%.4f,%.4f,%.4f\n", m,count[m],laststep[m],lastce[m],min[m],lastce[m]-min[m]
}' /lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/scaling_law_plots/train_ce_transformer_wandb.csv | sort
```

复算 architecture final row：

```bash
awk -F, 'NR==1 {next} {
  key=$1 "|" $2;
  last[key]=$5 "," $7 "," $9 "," $10
}
END {
  for (k in last) print k "," last[k]
}' /lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/scaling_law_plots/ce_vs_flops_all_arch.csv | sort
```

复算 manifest row count：

```bash
awk -F'\t' 'BEGIN{OFS="\t"} !/^#/ {
  n++; label[$2]++; last[$2]=$10
}
END {
  print "rows", n;
  for (l in label) print l,label[l],last[l]
}' /lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/scaling_runs_manifest.tsv | sort
```

检查 job ledger：

```bash
tail -120 /lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/scaling_runs_live_jobs.md
find /lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/scaling_logs -maxdepth 1 -type f -printf '%TY-%Tm-%Td %TH:%TM %p\n' | sort -r | head -60
```

检查 agent 输出：

```bash
sed -n '1,240p' /lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/agent_outputs/session_benchmark_results.md
sed -n '1,220p' /lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/agent_outputs/job_3360360_analysis.md
sed -n '1,220p' /lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/agent_outputs/job_3378771_status.md
```

## 需要生成的本地输出

建议输出目录：

```text
/lus/lfs1aip2/projects/public/u6gb/o8_transformer_scaling_law_goal_run/
```

需要创建：

```text
o8_transformer_scaling_law_goal_run/README.zh.md
o8_transformer_scaling_law_goal_run/o8_current_progress_report.zh.md
o8_transformer_scaling_law_goal_run/data/o8_verified_metrics.json
o8_transformer_scaling_law_goal_run/data/o8_verified_tables.csv
o8_transformer_scaling_law_goal_run/site/index.html
o8_transformer_scaling_law_goal_run/site/data.json
o8_transformer_scaling_law_goal_run/site/assets/   # 如需要静态图片
```

`o8_current_progress_report.zh.md` 必须包含：

- 一段 executive summary。
- 当前进度：baseline 6-size 已可验证，v6 11-size ladder 有 manifest/job ledger，但 completion 需复核。
- LOBbench 结果：Transformer、Mamba3、S5、GDN 等 CE/FLOPs/tokens 表。
- O8 训练结果：6-size CE 表、FLOPs/MFU 表。
- undertraining 结论：特别说明 78M 的 final CE 反单调问题。
- timestamp/session id：历史时间窗、observation IDs、任何查到的 session id。
- 数据源列表和核查命令。
- 风险和下一步。

## HTML / figure 要求

HTML 站点必须是中文，至少包含以下图：

1. KPI 顶部摘要：
   - 6-size baseline curves available
   - v6 manifest rows
   - best verified final/min CE
   - Transformer 125M CE/FLOPs
   - Mamba3 / GDN / S5 comparison
   - latest verified data timestamp
2. Figure A：O8 train CE vs global step，6 条曲线。
3. Figure B：final CE 和 min CE 对比柱状/点图，突出 46M 与 78M。
4. Figure C：MFU vs model/probe size。
5. Figure D：CE vs cumulative FLOPs across architectures，突出 Transformer 125M、Mamba3 78M、GDN 94M、S5 120M。
6. Figure E：历史时间轴：
   - 2026-03-25 job 3360360
   - 2026-03-26 benchmark session
   - 2026-03-27 job 3378771 status
   - 2026-04-21 test CE files
   - 2026-04-27/04-30 v6 manifest submissions
   - 2026-05-10 0.2M resubmission/job 4524395
   - 2026-05-18 dashboard/data extraction session
   - 2026-05-20 current Notion task

HTML 可以复用 Chart.js。若生成前端站点，请用 Playwright 或浏览器截图检查桌面和移动端渲染，确保图表非空、文字不重叠。

推荐 surge 域名：

```text
o8-transformer-scaling-law-progress.surge.sh
```

如果已占用，使用：

```text
o8-transformer-scaling-law-progress-20260520.surge.sh
```

## Notion 回写要求

在 Notion 原页下创建子页，而不是只更新原空白页。建议创建 3 个子页：

1. `O8 Transformer Scaling Law 当前进度报告`
2. `O8 LOBbench 结果与架构对比`
3. `O8 数据源、历史入口与复现核查清单`

每个子页必须包含：

- 中文摘要。
- 本地/Notion/记忆来源。
- dashboard surge 链接。
- 数据时间戳。
- 明确哪些结论已验证、哪些仍需验证。

使用 Notion 工具前先 fetch 原页。创建页面内容前，如果工具要求 Notion markdown spec，先读取 `notion://docs/enhanced-markdown-spec`。

## 最终回答格式

最终用中文简短汇报：

- goal 已执行完成。
- 本地报告路径。
- HTML 本地路径。
- surge URL。
- Notion 子页链接。
- 最关键结论 3-5 条。
- 如果 session id 未找到，明确说明“未在可访问来源中找到”，并给出已验证历史时间窗和 observation IDs。

## 验收标准

任务完成必须满足：

- 不再误称 11-size 全量训练已完成，除非执行中重新验证到了完整 completion 证据。
- 报告中能解释 78M undertrained / schedule rebound 信号。
- LOBbench 表含 Transformer 125M、Mamba3 78M、GDN、S5 至少四类。
- HTML 页面有至少 4 个非空图表。
- surge.sh 页面可访问。
- Notion 原页下出现子页。
- 所有关键数字都能追溯到本 markdown 列出的文件、记忆 observation 或 Notion 页面。
