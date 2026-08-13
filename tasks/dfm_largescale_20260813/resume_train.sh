#!/bin/bash
# 在新的 23h 分配上从 step 1650 续训到 3500。DFM_RESUME=1 是默认，靠 DFM_TAG +
# DFM_OUT 找到同名 checkpoint；worker 会同时推进采样器，否则会拿步 0..N 已用过的
# 序列重训 N..2N，而 loss 曲线看起来完全连续。
set -u
T=/lus/lfs1aip2/projects/public/u6gb/tasks/dfm_largescale_20260813; S0=/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/dfm-post-training-20260801
for g in 0 1 2; do
  case $g in 0) LR=1e-5; TAG=lg488b_g1;; 1) LR=1e-4; TAG=lg488b_g2;; 2) LR=3e-4; TAG=lg488b_g3;; esac
  env -u SLURM_NNODES -u SLURM_NTASKS -u SLURM_JOB_ID -u SLURMD_NODENAME \
  srun --jobid=6000412 --overlap --exact --cpus-per-task=16 -w nid010779 -N1 -n1 \
    --cpu-bind=none --job-name=dfm-rs-$TAG \
    --export=ALL,DFM_GPU_BASE=$g,DFM_TAG=$TAG,DFM_SHARD=2025-01..2025-12,DFM_STEPS=3500,DFM_BATCH=1,DFM_N_MSG=500,DFM_EVAL_EVERY=100,DFM_EVAL_BATCHES=4,DFM_OUT=$T/artifacts,DFM_LR=$LR,DFM_WARMUP=0,DFM_METRIC=field,DFM_T_COND=0,DFM_CKPT_EVERY=50,DFM_RESUME=1,XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
    bash $S0/post_training/dfm/tools/run_train_node.sh > $T/logs/rs_${TAG}.log 2>&1 &
done
wait
echo "=== RESUME TRAIN DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
for g in 1 2 3; do echo "g$g: $(cat $T/artifacts/lg488b_g${g}_state.msgpack.meta)"; done
