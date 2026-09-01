#!/bin/bash
# R2 第一格：corrupt(draft) → real 的残差训练（分布修复，架构不动）。
# 与 b7b_on_s42 同架构（offset_embed=1、t_cond=0、lr 1e-4、8000 步、field 度量），
# 唯一差别 = 训练分布（DFM_DRAFT_CORPUS）。对照即 b7b_on_s42（corrupt(real) 版）。
# 语料用启动时刻的快照目录，与仍在写入的 corpus_r2/ 隔离。
# MEMFRAC=0.10（b7b 同款）与语料生成（0.30）共卡：39G < 95.6G。
set -u
T=/lus/lfs1aip2/projects/public/u6gb/tasks/dfm_largescale_20260813
S0=/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/dfm-r2-20260830
mkdir -p $T/artifacts_r2 $T/logs
NODE=${NODE:-nid010392}; GPU=${GPU:-0}; JOB=${JOB:-6217606}
TAG=${TAG:-r2d1_s42}; SEED=${SEED:-42}; OFF=${OFF:-1}
CORPUS=${CORPUS:-$T/corpus_r2_snap1}

env -u SLURM_NNODES -u SLURM_NTASKS -u SLURM_JOB_ID -u SLURMD_NODENAME \
setsid nohup srun --jobid=$JOB --overlap --exact --cpus-per-task=16 -w $NODE -N1 -n1 \
  --cpu-bind=none --job-name=dfm-$TAG \
  --export=ALL,DFM_GPU_BASE=$GPU,DFM_TAG=$TAG,DFM_SHARD=2025-01..2025-12,\
DFM_STEPS=8000,DFM_BATCH=1,DFM_N_MSG=500,DFM_EVAL_WINDOWS=512,\
DFM_EVAL_EVERY=1000,DFM_OUT=$T/artifacts_r2,DFM_LR=1e-4,DFM_WARMUP=0,\
DFM_METRIC=field,DFM_T_COND=0,DFM_CKPT_EVERY=1000,DFM_RESUME=1,\
DFM_OFFSET_EMBED=$OFF,DFM_SEED=$SEED,DFM_DRAFT_CORPUS=$CORPUS,LOG_GRAD_NORMS=1,\
XLA_PYTHON_CLIENT_MEM_FRACTION=0.10,XLA_FLAGS=--xla_gpu_enable_triton_gemm=false \
  bash $S0/post_training/dfm/tools/run_train_node.sh \
  > $T/logs/r2train_$TAG.log 2>&1 < /dev/null &
echo "  $TAG -> $NODE g$GPU corpus=$(basename $CORPUS) steps=8000"
echo "=== R2 train launched $(date -u +%H:%M:%SZ) ==="
