#!/bin/bash
# 修：worker 用 (SLURM_LOCALID + DFM_GPU_BASE) % 4 选卡。我上一版传 SLURM_LOCALID=$i，
# 但每个 srun 是独立单任务步，SLURM 自己把 LOCALID 置 0，四个进程于是全挤在 GPU 0。
# 正确的开关是 DFM_GPU_BASE。
set -u
T=/lus/lfs1aip2/projects/public/u6gb/tasks/dfm_largescale_20260813; S0=/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/dfm-post-training-20260801
for g in 1 2 3; do
  case $g in 1) LR=1e-5;; 2) LR=1e-4;; 3) LR=3e-4;; esac
  env -u SLURM_NNODES -u SLURM_NTASKS -u SLURM_JOB_ID -u SLURMD_NODENAME \
  srun --jobid=5992007 --overlap --exact --cpus-per-task=16 -w nid010547 -N1 -n1 \
    --cpu-bind=none --job-name=dfm-g$g \
    --export=ALL,DFM_GPU_BASE=$g,DFM_TAG=lg488b_g$g,DFM_SHARD=2025-01..2025-12,DFM_STEPS=3500,DFM_BATCH=1,DFM_N_MSG=500,DFM_EVAL_EVERY=100,DFM_EVAL_BATCHES=4,DFM_OUT=$T/artifacts,DFM_LR=$LR,DFM_WARMUP=0,DFM_METRIC=field,DFM_T_COND=0,DFM_CKPT_EVERY=50,XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
    bash $S0/post_training/dfm/tools/run_train_node.sh > $T/logs/lg488b_g$g.log 2>&1 &
done
wait
