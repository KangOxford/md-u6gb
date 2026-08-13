#!/bin/bash
# 冒烟：验证多 ticker 循环在同一进程里复用模型与挂载。
# 环境一律走 run_eval_node.sh —— 自己拼 conda 前缀正是上一次 chex ImportError 的原因。
set -u
S0=/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/dfm-post-training-20260801
A=/lus/lfs1aip2/projects/public/u6gb/tasks/ce_orderflow_20260812T200352Z/A02_scale
W=$S0/post_training/dfm/eval/run_eval_node.sh
export DFM_GPU=${DFM_GPU:-0} DFM_SCRIPT=dfm_correct_runner.py
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.40
COMMON="--month 2026-01 --n-cond 500 --n-gen 500 --stocks GOOG,MSFT
        --index-dir $A/idx --group-size 8 --validate-first 8 --gate-batches 2
        --state $S0/post_training/dfm/artifacts/stage2a/long_NVDA_state.msgpack
        --t-start 0.80 --n-steps 8
        --out-template $A/rollouts/dfm_smoke_{stock}_{month}_learned.npz"
echo "=== pass 1: build per-ticker indices ==="
bash $W $COMMON --build-index
echo "P1_EXIT=$?"
echo "=== pass 2: generate 2 tickers x 4 seq in ONE process ==="
bash $W $COMMON --n-seq 4 --batch-size 4
echo "P2_EXIT=$?"
