#!/bin/bash
# 端到端验证：新代码训一小段 -> checkpoint 带 metric 身份 -> 推理自动读它并过守卫。
set -u
T=/lus/lfs1aip2/projects/public/u6gb/tasks/dfm_largescale_20260813; S0=/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/dfm-post-training-20260801; A=/lus/lfs1aip2/projects/public/u6gb/tasks/ce_orderflow_20260812T200352Z/A02_scale
echo "=== [1/2] train 60 steps (field metric) $(date -u +%H:%M:%SZ) ==="
env -u SLURM_NNODES -u SLURM_NTASKS -u SLURM_JOB_ID -u SLURMD_NODENAME \
srun --jobid=5992008 --overlap --exact --cpus-per-task=8 -w nid011165 -N1 -n1 --cpu-bind=none \
  --job-name=dfm-chain-tr \
  --export=ALL,DFM_GPU_BASE=2,DFM_TAG=chain_val,DFM_SHARD=2025-11..2025-12,DFM_STEPS=60,DFM_BATCH=1,DFM_N_MSG=500,DFM_EVAL_EVERY=60,DFM_EVAL_BATCHES=2,DFM_OUT=$T/artifacts,DFM_LR=1e-4,DFM_WARMUP=0,DFM_METRIC=field,DFM_T_COND=0,DFM_CKPT_EVERY=50,XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 \
  bash $S0/post_training/dfm/tools/run_train_node.sh > $T/logs/chain_train.log 2>&1
echo "train rc=$?  sidecar:"; cat $T/artifacts/chain_val_state.msgpack.meta 2>/dev/null; echo
echo "=== [2/2] inference with that checkpoint $(date -u +%H:%M:%SZ) ==="
env -u SLURM_NNODES -u SLURM_NTASKS -u SLURM_JOB_ID -u SLURMD_NODENAME \
srun --jobid=5992008 --overlap --exact --cpus-per-task=8 -w nid011165 -N1 -n1 --cpu-bind=none \
  --job-name=dfm-chain-inf \
  --export=ALL,DFM_GPU=3,DFM_SCRIPT=dfm_correct_runner.py,XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 \
  bash $S0/post_training/dfm/eval/run_eval_node.sh --month 2026-01 --n-cond 500 --n-gen 500 \
    --stocks GOOG --index-dir $A/idx --group-size 8 --validate-first 8 --gate-batches 2 \
    --state $T/artifacts/chain_val_state.msgpack --t-start 0.80 --n-steps 8 \
    --n-seq 4 --batch-size 4 --corr-batch 4 \
    --out-template "$T/rollouts/chain_{stock}_{month}_learned.npz" > $T/logs/chain_infer.log 2>&1
echo "infer rc=$?"
grep -E "\[metric\]|stage|trunk_l1|P_frob|shapes|FAILED|mismatch|done" $T/logs/chain_infer.log | tail -8
