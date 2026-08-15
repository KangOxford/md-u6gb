#!/bin/bash
# 值度量训练臂。单因子对照是**阶梯臂 `dv25_lr1e4_s0u` 在 step 14000 的快照**：
# 同 lr 1e-4、同 seed 42、同 shard、同 128 窗口留出集、同 tritonOFF，唯一变的
# 是 `DFM_METRIC`。
#
# 不是 `dv10_lr1e4_s0u` —— 那个 tag 属于 6014307 上的**原始**批次，它是 tritonON
# 且留出集 512 窗口，而且此刻还在 step 2475 上以 3.4 s/step 爬（与 alpha 扫描
# 共卡）。留出集不影响训练、tritonON/OFF 已验过逐步 loss 差 <=1e-4，所以它
# **原则上**可比；但阶梯臂是**同一条轨迹**上的点，不需要靠这两条论证。
#
# 为什么值得单开一条臂（不是又一个想法，是一个可证伪的因果预测）：
# 全量 487 ticker 上，学到的残差在 `event_type` 上比**同范数随机方向**好 10 倍
# （0.151 vs 1.526），在 `price_rel`/`log10_dt`/`direction` 上也更好，
# **唯独 `size` 上差 1.65-1.73 倍**（0.958 vs 0.580）。四个字段帮忙、一个字段
# 主动添乱，这不能用「残差本来就有破坏性」解释 —— 那是目标函数奖励出来的一个
# 系统性变换。而目标函数就是 field 度量下的去噪损失，那条度量在 `size` 上
# 可证明坏掉：P(99)/P(101) = 1.5e-26（值度量下是 1.000）。
#
# 于是预测（跑之前锁死）：
#   值度量臂必须**特异地**消掉 `size` 的异常 —— 即 `size` 的水平代价显著下降，
#   而 `event_type`/`price_rel`/`log10_dt`/`direction` 四个字段**不显著变差**。
#   如果五个字段一起变好或一起变差，那就不是这个机制，是别的东西。
#
# 冒烟已过（600 步，0.39 s/step，与 field 度量同速；loss 4.15->3.81，|g| 39->7.2）。
# 时间表是 `apow p=2`，由预注册的信息集中度判据选出（0.457 通过，cosine 0.708 未过）。
set -u
T=/lus/lfs1aip2/projects/public/u6gb/tasks/dfm_largescale_20260813
S0=/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/dfm-post-training-20260801
JOB=${JOB:?need JOB}; NODE=${NODE:?need NODE}; GPU=${GPU:-0}
TAG=${TAG:-dv10_val_lr1e4_s0}
STEPS=${STEPS:-13880}
LR=${LR:-1e-4}
SEED=${SEED:-42}

env -u SLURM_NNODES -u SLURM_NTASKS -u SLURM_JOB_ID -u SLURMD_NODENAME \
setsid nohup srun --jobid=$JOB --overlap --exact --cpus-per-task=8 -w $NODE -N1 -n1 \
  --cpu-bind=none --job-name=dfm-val-${TAG} \
  --export=ALL,DFM_GPU_BASE=$GPU,DFM_TAG=$TAG,DFM_SHARD=2025-01..2025-12,\
DFM_STEPS=$STEPS,DFM_BATCH=1,DFM_N_MSG=500,DFM_EVAL_WINDOWS=128,\
DFM_EVAL_EVERY=2000,DFM_OUT=$T/artifacts_dv,DFM_LR=$LR,DFM_WARMUP=0,\
DFM_METRIC=value,DFM_A_TARGET=12,DFM_T_COND=0,DFM_CKPT_EVERY=2000,DFM_RESUME=1,\
DFM_SEED=$SEED,XLA_PYTHON_CLIENT_MEM_FRACTION=0.09,\
XLA_FLAGS=--xla_gpu_enable_triton_gemm=false \
  bash $S0/post_training/dfm/tools/run_train_node.sh \
  > $T/logs/dv_${TAG}.log 2>&1 < /dev/null &
echo "=== value 臂 $TAG -> $NODE g$GPU  steps=$STEPS lr=$LR seed=$SEED  $(date -u +%H:%M:%SZ) ==="
echo "    对照臂: artifacts_dv/ladder/dv25_lr1e4_s0u_s14000 (field 度量, 同 lr/seed/tritonOFF)"
echo "    日志 $T/logs/dv_${TAG}.log"
