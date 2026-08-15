# 离线交接快照 — 2026-08-07T12:45Z

用户 12:42Z 离线约 4-5 小时。本文件是回来后的第一落点。

## 结论先行：你断网对已提交的作业零影响

**不需要 tmux/nohup 守着。** SLURM 作业跑在计算节点上，与客户端连接、与任何 Claude session
是否存活完全无关。Claude Code 的 session 会随客户端断开而停止，它本来就不是能守夜的 daemon；
**能守夜的是 SLURM 自己**。

（补充：登录节点禁 tmux/nohup 常驻是 CLAUDE.md 第 5 节的硬规矩；计算节点 job step 内跑后台
监控脚本不在此列。本轮没有起任何常驻进程，因为不需要。）

## 离线时段的作业清单

| JobID | 名称 | 状态 | 剩余 | 谁在管 |
|---|---|---|---|---|
| `5924043` | u6gb-4-node-chain | RUNNING | **13h49m** | BPE varlen **主臂**训练 |
| `5931446` | u6gb-4-node-chain | RUNNING | **13h46m** | BPE **26tok 对照臂**训练 |
| `5944378` | sp500-sweep-driver | PENDING | 12h 时限 | 排到自起 |
| `5943935` | ldm-bigbang-infer | PENDING | 30m 时限 | 排到自起 |

13.8 小时余量覆盖 4-5 小时离线**绰绰有余**，且 4-node-chain 带 `--chain` 自续链。

## BPE 两臂的身份（12:39Z 实测）

| | 主臂 `5924043` | 对照臂 `5931446` |
|---|---|---|
| 编码 | varlen 无损 BPE，词表 15847 | 26tok base-100，词表 2112 |
| OUT_DIR | `runs/prod20260807T094252Z` | `runs/ctrl26tok20260807T100231Z` |
| 显存/卡 | 33.8 GB | 18.5 GB（8 卡另有 71.6GB 的 lobgen）|
| 剩余/卡 | **~64 GB** | 8 卡 ~79GB / 8 卡 ~8GB |
| 功耗 | 246-326 W（TDP 700W 的 **35-47%**）| 401-445 W（**57-64%**）|

代码 `/lus/lfs1aip2/projects/public/u6gb/tasks/bpe_varlen_torch_20260806T183132Z/src/train.py`。
配置 d_model=512 / L=8 / SEQ_LEN=4096 / PER_GPU_BSZ=16 / TOTAL_STEPS=80900 / world=16。

**这两臂由并行 session 管理**（记录见 `findings.md` 的 F1786105568 / F1786106161），本 session 未触碰。

### 关于"用掉剩余显存"

`utilization.gpu=100%` **不是**算力饱和的证据——它只表示采样窗口内有 kernel 活跃，
一个占单个 SM 的 kernel 跑满时间也是 100%。诚实的读数是功耗：主臂只到 TDP 的 35-47%，
所以**确实有余量**。并行 session 的 F1786105568 已算出上限：26tok 臂可到 BSZ=64（4×）、
varlen 臂 BSZ=32（2×）。

**但没有执行**，因为改 BSZ 必须重启训练，而这两臂归另一个 session 管，
中途重启会毁掉它已跑的进度。这个决定需要你来做，不是可以顺手做掉的事。

## 本 session 这条线的状态：无待提交任务

本 session 从头到尾做的是**改模型架构前的地形侦察**，一个训练作业都没提交。
待办卡在一个未决问题上：**改哪一块架构还没定**（RoPE 改 2D / 换归一化 / 加新 backbone /
改双支路融合）。没有这个，提交任何 4 节点级别的作业都是猜测，撞车或跑错配置的代价是几十节点小时。

另有两个悬空项：
1. 上一条指令 "do this in a new gitworktree and PR to develop" 里的 `this` 未指明
2. **sigma-0 没有 `develop` 分支**，本地和远程都没有。历史 PR #7–#12 全是 feature 分支直接进 `main`

## 侦察成果（已可直接使用）

baseline = `/lus/lfs1aip2/projects/public/u6gb/sigma-0`（GitHub KangOxford/sigma-0, main）。

改架构的三层验证闸门，**全部实测通过**：

| 闸门 | 在哪跑 | 实测 |
|---|---|---|
| `tests/unit/test_model_registry.py` | 登录节点，无需限核 | **5 passed / 0.06s** |
| `tests/integration/test_backbone_contracts.py` | 登录节点，**必须 taskset 限核** | **18 passed / 9.39s** |
| `tests/integration/test_backbone_forward.py` | 计算节点（GPU） | **7 passed / 67.3s** |

登录节点限核命令（不限核会 SIGABRT，栈顶 `backend_compile_and_load`，288 核铺线程撞 cgroup `pids.max=500`）：

```bash
cd /lus/lfs1aip2/projects/public/u6gb/sigma-0
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 \
XLA_FLAGS="--xla_force_host_platform_device_count=1" taskset -c 0-3 \
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/miniforge3/bin/python \
  -m pytest tests/integration/test_backbone_contracts.py -q
```

GPU 闸门可 attach 到现有分配跑，**已验证可行**（用掉主臂 64GB 余量的一小部分，67 秒）：

```bash
srun --jobid=5924043 --overlap --exact -N1 -n1 --cpu-bind=none --gpus=1 \
  <conda>/bin/python -m pytest tests/integration/test_backbone_forward.py -q
```

详细落点表见 `memory/reference_mamba3_baseline_entrypoint.md`、
`memory/reference_sigma0_tests_on_login_node.md`、`memory/project_r1_mamba3_lineage.md`。

## 你回来后的第一步

```bash
squeue --me -o "%.12i %.24j %.9T %.12M %.12L"
bash /lus/lfs1aip2/projects/public/u6gb/gpu_status.sh
```

然后给我一句话定下改哪块架构，我建 worktree、改、跑三层闸门、开 PR。
