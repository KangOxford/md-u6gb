# 进度

## 时间线

| 时间 (UTC) | 事件 |
|---|---|
| 11:0x | 确认 MiniMax H3 是**全模态视频+音频生成模型**，非 LLM；2026-07-31 发布 |
| 11:1x | 确认**技术报告未发布**、**训练数据未开源**——用户的两条前提均需修正 |
| 11:2x | 拉配置与 diffusers 官方移植源码，反推出完整架构 |
| 11:3x | 建 venv（diffusers 0.40.0.dev0 + transformers 4.57.6 + PyAV 18.0.0） |
| 11:42 | 提交 job **5998320**（取资产 + 官方 checkpoint 冒烟） |
| 11:5x | `h3nano.py` CPU 自检通过：**101.652 M** 参数，前向/损失/逆变换全对 |
| 12:0x | checkpoint 保存/剪枝/面包屑/resume 单独验证通过 |
| 12:1x | 采样回路（双调度器）CPU 验证通过 |
| 12:2x | 提交 job **5998498**（阶段 2–6 全流水线，依赖 5998320） |

## 作业

| Job ID | 名称 | 状态 | 说明 |
|---|---|---|---|
| 5998320 | h3-fetch-smoke | PENDING | 1 节点 6h。下载 143 GB checkpoint + 34 GB VGGSound，参数普查，T2VA 冒烟 |
| 5998498 | h3-nano-pipeline | PENDING (Dependency) | 1 节点 12h。语料 → 闸门 → 预训练 → FL2VA → CFG 蒸馏 → 评估 |

**排队原因诊断**

> 更正：本节最初的结论（「调度器停摆或额度耗尽」）是错的，根因是我误读了 `squeue` 的可见范围。下面保留完整的排查过程，因为那条误判本身是最值得记住的部分。

第一轮排查逐项排除了这些：

| 假设 | 检验 | 结论 |
|---|---|---|
| 资源不足 | `sinfo -p workq -t idle` 一度 155 节点空闲 | 排除 |
| `--mem=0` 要整节点内存 | 另投探针（`--mem=100G`、32 CPU、5 分钟）→ 同样不动 | 排除 |
| QOS / 账户限额 | `workq_qos`: MaxJobsPU=256（我 8 个）；assoc 无 GrpTRES | 排除 |
| fair-share 被压 | `PriorityWeightFairShare = 0` —— 实际上**所有权重都是 0**，全集群人人 Priority=1，调度退化为 FIFO by job ID | 非我方特有 |
| slurmctld 故障 | `scontrol ping` → primary + backup 均 UP | 正常 |

当时我据此得出「全集群 pending 作业只有 8 个且全是我的，却有 155 节点空闲」，因而判断是调度器停摆或机时耗尽。

**这个前提是假的。**

```
PrivateData = accounts,events,jobs,reservations,usage,users
```

`PrivateData=jobs` 让 Slurm 对我隐藏其他用户的作业。对账即可揭穿：

| | 我能看见 | 集群实际 |
|---|---|---|
| 作业数 | 11 | 不可见 |
| 运行中节点 | 12 | **1148**（`sinfo -t alloc,mix`） |

也就是说 1136 个节点上跑着我看不见的作业，前面还排着我看不见的队列。那 58 个
`plnd`（planned）节点正是 backfill 在为这些作业预留。**87% 满载的集群 + FIFO
+ 长队 = 我这些作业排 8 小时是正常的**，不是故障。

教训：`PrivateData` 会把 `squeue` 的全局视图裁成一个只含自己的子集，而这个子集
看起来和「集群空转」完全一样。下这类结论前必须先
`scontrol show config | grep PrivateData`。

作业留在队列中，一旦调度即自动开跑，监控已挂。

**还有一个可拉的杠杆：缩短申请时长。** 集群的 `SchedulerParameters` 含
`bf_interval=60, bf_max_time=30, bf_max_job_user=25`，backfill 会把短作业塞进长作业
开始前的空隙，所以**申请时长直接决定能塞进多少空隙**：6 小时的作业要等一个 6 小时的
空档，2 小时的只要 2 小时的。

当前两个 H3 作业申请 6 h / 12 h。实际下载 177 GB 在 200 MB/s 下约 15 分钟、
50 MB/s 下约 1 小时，所以「只下载」的作业申请 **2 h** 完全够用，且远比 6 h 好调度。

**没有立刻这么做的原因**：节点闸门。此刻 2 RUNNING（8 节点）+ 12 PENDING（12 节点）
= 20，正好触顶；而且已投的两个探针（5998518 / 5998812）按本仓库 P0 禁令无法取消。
再投只是增加队列压力。

如果等待继续拉长，值得做的是：投一个 **1 节点 / 0 或 1 GPU / 2 h 的纯下载作业**
（带「四卡路线在跑则让位」的守卫），让 143 GB 先落地；之后的训练作业启动时
`fetch_assets.py` 会因文件已存在而秒过。

## 已完成并验证的部分

| 组件 | 验证方式 | 结果 |
|---|---|---|
| `h3nano.py` 模型 | CPU 实例化 + 前向 | 101.652 M；AdaLN 35.4% / FFN 36.6% / attn 24.4%（33B 对应 39.4/35.0/23.3） |
| packed 布局 | 调用参考实现 `build_packed_sequence` | seq=1748（video 1408 + text 96 + audio 244） |
| 几何公式 `17n+5 → 5n+2` | 源码交叉验证 + 旋转时钟自洽 | F=73 → 22 latent 帧；4×17+(1+4)=73 ✓ |
| patchify / audio pack | 往返逆变换 | max\|err\| = 0.00e+00（两者） |
| 流匹配目标 | 从 scheduler 代数推导 | `v = x0 − ε` |
| 训练循环 | CPU 干跑 | 前向/反向/优化器/日志全通（登录节点 4 GiB cgroup 限制导致长跑被杀，非代码问题） |
| checkpoint | 独立测试 | 原子面包屑 ✓、精确保留 3 个 ✓、resume 状态一致 ✓ |
| 采样回路 | CPU，5 步 | guidance=0 → 4 次前向；guidance=5 → 8 次前向；输出有限 ✓ |
| 双 shift | scheduler σ 网格 | video 5 步中 4 步在 σ>0.75；audio 均匀分布 |
| **约定断言套件** | `code/test_h3nano.py`，13 项 | **13/13 通过**，见下 |

### 约定断言套件结果（`./venv/bin/python code/test_h3nano.py`）

```
PASS  velocity_target_is_data_minus_noise             v = x0 - eps at t in {0.05, 0.4, 0.95}
PASS  denoising_a_perfect_velocity_lands_on_x0        64 steps from pure noise land within 8.34e-07 of x0
PASS  the_two_shifts_produce_different_grids          fraction of grid above sigma=0.5: video 0.91 vs audio 0.73
PASS  patchify_round_trips_exactly                    (3, 24, 7, 8, 8) <-> (3, 112, 96) exact
PASS  audio_packing_is_channel_major_and_exact        channel-major order confirmed and exact on round trip
PASS  frame_geometry_matches_the_reference            17n+5 -> 5n+2 and 40 Hz audio confirmed over 8 frame counts
PASS  layout_row_counts_follow_the_geometry           seq=1748 = text 96 + audio 244 + video 1408
PASS  keyframe_anchors_add_leading_conditioning_rows  two anchors add 32 leading video rows
PASS  text_rows_inherit_the_video_timestep            text follows video (0.3); audio independent (0.7); table has 2 rows
PASS  anchors_sit_at_the_documented_noise_level       anchor rows held at max(t, 0.999)
PASS  a_batch_shares_one_timestep_pair                batch shares one (t_video, t_audio); per-item draws raise
PASS  loss_ignores_conditioning_rows                  perturbing 16 anchor rows by 1000 leaves the loss unchanged
PASS  model_forward_shapes_match_the_layout           micro forward OK; 32.73 M params, refiner 3.93 M

13/13 checks passed
```

第 2 项是最强的单项证据：它用**参考实现的调度器**（不是我写的欧拉步）把一个完美速度
从纯噪声走 64 步，落点距 x₀ 仅 8.34e-07。这同时检验了「速度朝向数据」「目标是
x₀−ε」「t=1−σ 且 t=1 为干净」三件事——任一符号搞反都会发散。

该套件已接入 `verify_environment()`，在作业第一分钟运行，因此**对已排队的作业同样生效**
（sbatch 快照的是 batch 脚本，外部 Python 是运行时读取）。

## 待办

- [ ] 5998320 落地：验证「13B 在 AdaLN」的实测值、packed 布局实测、官方 T2VA 出片
- [ ] 5998498 落地：E1/E3/E4/E6/E7 五项指标
- [ ] E5（双 shift 消融）需 3 组独立训练，视预算决定
- [ ] E2（蒸馏前后前向次数与墙钟）在流水线阶段 6 内测

## 遇到的问题与处理

| 问题 | 根因 | 处理 |
|---|---|---|
| `hf download --exclude` 把 json 全滤掉 | glob 语义与预期不符 | 改用 curl 点名拉 19 个配置文件 |
| diffusers 0.37.1 无 MiniMaxH3 | 只在 main 分支 | venv 装 git main（0.40.0.dev0） |
| `cannot import HybridCache` | **归因错了**。要 `HybridCache` 的是 **peft 0.17.1**（它在模块级 import 了这个被 transformers 5.x 删掉的符号），不是 diffusers——`HybridCache` 在 diffusers 全部 `.py` 里出现 0 次 | 最终解：**升 peft 到 0.20.0**（该 import 已挪进函数体的弃用分支）+ transformers 5.15.0 + hub 1.27.0。当初「降 transformers 到 4.57.6」方向相反，还顺手删掉了 H3 需要的 `create_mm_token_type_ids` |
| `libgomp: Thread creation failed` | 登录节点 pids.max | `taskset -c 0-3` + `OMP_NUM_THREADS=4` |
| 训练干跑 exit 137 | 登录节点 cgroup 上限 **4 GiB**，已用 3.31 GB | 确认是登录节点限制；计算节点 856 GB 无此问题 |
| 参数普查 `token_refiner` 只有 0.003 M | 分桶时 `.attn.` 判断在 `token_refiner` 之前，抢走了 refiner 的 attn/ffn | 调整判断顺序（两个文件都改） |
| `create_mm_token_type_ids` 不存在 | **我自己造的**：当初把 `HybridCache` 报错归因给 diffusers 并降 transformers 到 4.x，但真正要它的是 **peft 0.17.1**（`HybridCache` 在 diffusers 全部 `.py` 中出现 0 次）。降级正好删掉了 H3 文本编码需要的 API | 不打版本仗：纯文本 prompt 的 `mm_token_type_ids` 恒为全零，直接构造 + 断言防止将来喂进带图 prompt |

### 环境版本互锁（拧断两次才看清）

`diffusers.loaders.peft` 是整条导入链的单点：diffusers main 是唯一带 MiniMax-H3 的版本，
而它通过 `PeftAdapterMixin` **急切导入** peft。

| 尝试 | 结果 |
|---|---|
| 初始：transformers 5.5.0 + peft 0.17.1 | ❌ peft 模块级 `from transformers import HybridCache`，5.x 已删 |
| 第一次修：降 transformers 到 4.57.6 | ✅ 能导入，但**删掉了** H3 要的 `create_mm_token_type_ids` |
| 第二次修：升 transformers 5.15 + peft 0.20 | ❌ peft 0.20 改去探测 `transformer_engine`，其 import 要 libnvrtc 且**探测不捕获异常** |
| 最终：transformers 4.57.6 + peft 0.17.1 + 代码不依赖那个 API | ✅ |

真正的解法不是选对版本组合，而是**让代码不需要那个 API**：`mm_token_type_ids` 对纯文本
prompt 恒为全零，直接构造并加断言，于是 transformers 大版本从依赖里消失。互锁关系与理由
已写进 `requirements-pinned.txt`。

### 自己写出来的 bug 清单（全部属于「不报错、只出错结果」）

| # | bug | 若不修的后果 |
|---|---|---|
| 1 | per-item timestep 与 `(seq_len,)` 的 `timestep_indices` 冲突 | batch 中第 1..B−1 个样本被告知的噪声水平与实际不符 → 模型学会忽略 t |
| 2 | tarball 尾批静默丢弃 | 每片最多丢 48 个 clip，语料比日志声称的小 |
| 3 | `hash()` 逐进程随机化 | 注释写「确定性」，实际重建语料会抽到不同后验 |
| 4 | 参数普查分桶顺序 | token_refiner 显示 0.003 M（真值约 8.8 M） |
| 5 | `create_mm_token_type_ids` 版本冲突 | 文本库首条 prompt 崩溃 |

第 1 条最贵：它不会报错，损失也不会出现在损失曲线上，只会让模型在采样时对噪声水平失去响应。
