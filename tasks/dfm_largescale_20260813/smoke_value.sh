#!/bin/bash
# 值度量的端到端冒烟测试（PR: smoke test before scale）。
#
# CPU 上 12/12 通过只证明**度量本身**对，不证明它在训练器里能跑通：配置到不到得了
# 度量、beta_max 有没有被默认值覆盖、`metric_digest` 写不写得进 sidecar、
# corrupt 的输出形状对不对 —— 这些只有走一遍作业实际走的路径才知道。
#
# 放在 6007121 上：`gtop` 显示它 16 张卡全空（`steps: 只有 .batch`），但**只剩
# 45 分钟**，所以只放 600 步（≈ 11 分钟启动 + 5 分钟训练），不放长跑。
#
# 三个必须在日志里出现的东西，缺一个就是没接上：
#   [metric] value: a_target=12.0 beta_max=248.6792 digest=...
#   [c0] held-out eval: 128 windows
#   loss 在下降且不是 nan
set -u
T=/lus/lfs1aip2/projects/public/u6gb/tasks/dfm_largescale_20260813
S0=/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/dfm-post-training-20260801
JOB=${JOB:-6007121}; NODE=${NODE:-nid010463}; GPU=${GPU:-0}
TAG=${TAG:-smokeval}
env -u SLURM_NNODES -u SLURM_NTASKS -u SLURM_JOB_ID -u SLURMD_NODENAME \
setsid nohup srun --jobid=$JOB --overlap --exact --cpus-per-task=8 -w $NODE -N1 -n1 \
  --cpu-bind=none --job-name=dfm-smoke-value \
  --export=ALL,DFM_GPU_BASE=$GPU,DFM_TAG=$TAG,DFM_SHARD=2025-01..2025-12,\
DFM_STEPS=600,DFM_BATCH=1,DFM_N_MSG=500,DFM_EVAL_WINDOWS=32,\
DFM_EVAL_EVERY=300,DFM_OUT=$T/artifacts_dv,DFM_LR=1e-4,DFM_WARMUP=0,\
DFM_METRIC=value,DFM_T_COND=0,DFM_CKPT_EVERY=300,DFM_RESUME=0,\
DFM_SEED=42,XLA_PYTHON_CLIENT_MEM_FRACTION=0.09,\
XLA_FLAGS=--xla_gpu_enable_triton_gemm=false \
  bash $S0/post_training/dfm/tools/run_train_node.sh \
  > $T/logs/smoke_value.log 2>&1 < /dev/null &
echo "=== value 度量冒烟 -> $NODE g$GPU  $(date -u +%H:%M:%SZ) ==="
echo "    日志 $T/logs/smoke_value.log"
