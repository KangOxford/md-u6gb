#!/bin/bash
# 第一次在 488 个 ticker 上训 DFM 残差。此前每个 stage2a run 只见过 1 个 ticker
# 的 1 个月（A02 那个 3/5 的残差就是这么来的）。
#   训练面：2025-01..2025-12 x 488 ticker = 5,856 个 ticker-月
#   对比：目标写的「8 股 x 4 年」= 384 个 ticker-月，这里是它的 15 倍
# 度量用 field 而非 embedding：worker 自己的注释记录了 embedding 的几何缺陷
#   （耦合比 6.69 vs 随机高斯 7.13；与数值距离的秩相关 price 0.237 / time 0.065）。
# t_cond=0 是被迫的：t_cond=1 时 init_dfm_params 不造 dfm_t_hidden，模型却要它。
set -u
T=/lus/lfs1aip2/projects/public/u6gb/tasks/dfm_largescale_20260813; S0=/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/dfm-post-training-20260801
i=0
for LR in 1e-5 1e-4 3e-4 1e-3; do
  env -u SLURM_NNODES -u SLURM_NTASKS -u SLURM_JOB_ID -u SLURMD_NODENAME \
  srun --jobid=5992007 --overlap --exact --cpus-per-task=16 -w nid010547 -N1 -n1 \
    --cpu-bind=none --job-name=dfm-lg-lr$i \
    --export=ALL,SLURM_LOCALID=$i,DFM_TAG=lg488_lr$i,DFM_SHARD=2025-01..2025-12,DFM_STEPS=3500,DFM_BATCH=1,DFM_N_MSG=500,DFM_EVAL_EVERY=100,DFM_EVAL_BATCHES=4,DFM_OUT=$T/artifacts,DFM_LR=$LR,DFM_WARMUP=0,DFM_METRIC=field,DFM_T_COND=0,DFM_CKPT_EVERY=100,XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
    bash $S0/post_training/dfm/tools/run_train_node.sh > $T/logs/lg488_lr$i.log 2>&1 &
  i=$((i+1))
done
wait
echo "=== LR GRID DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
