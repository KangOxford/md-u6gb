#!/bin/bash
# Four-arm do(Q) injection re-verification for PR #21 (hybrid branch edits the
# two files the injection evidence pins: run/base_model/runtime/inference.py
# and src/lob/inference_no_errcorr.py).
#
# Adapted from tasks/pr16_doq_injection_20260815T/code/run_four_arms.sh with
# exactly three changes: WT_P is now main content (ci worktree 89cf0ff, both
# pinned files hash-identical to origin/main b362a7e), WT_V is the PR#21
# branch (ece14dc + evidence commits), and artifacts land in a fresh HOME dir.
# All inference parameters are byte-for-byte the 2026-08-16 ones, so the new
# report answers the same six judgements (J0..J5) on the same rollout design.
#
# Runs inside an srun step on ONE node / ONE GPU: byte-identity is node-bound
# (D-X7), so P, P2, O, I must all execute here, sequentially.
set -u

TASK=/home/u6gb/kangli.u6gb/pr21_doq_artifacts_20260826
# 可用 env 覆盖：判别实验（同节点重跑 0816 原始 commit 对）用
#   WT_P=…/pr16-base-20260815  WT_V=…/pr16-verify-20260815(@41ec9284)  ARMS_SUBDIR=arms_v3
WT_P=${WT_P:-/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/ci-pipefail-20260826}
WT_V=${WT_V:-/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/hybrid-mamba3-nemotron-20260811}
QUANT_ROOT=/lus/lfs1aip2/projects/public/s5e/quant_team/quant
CONDA_PREFIX="$QUANT_ROOT/miniforge3"
PYTHON="$CONDA_PREFIX/bin/python"
NVIDIA_SITE="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia"
TORCH_LIB="$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib"

export TMPDIR="/tmp/${USER:-unknown}/sigma0_doq_pr21_$$"
mkdir -p "$TMPDIR" "$TASK/logs"

module load cuda/12.6
export PATH="$CONDA_PREFIX/bin:$PATH"
# The JAX CUDA plugin wheels dlopen these libraries. Loading the system CUDA
# module alone is insufficient on Isambard and silently falls back to CPU.
export LD_LIBRARY_PATH="$TORCH_LIB:$NVIDIA_SITE/cuda_nvrtc/lib:$NVIDIA_SITE/cuda_runtime/lib:$NVIDIA_SITE/cusparse/lib:$NVIDIA_SITE/cuda_cupti/lib:$NVIDIA_SITE/cufft/lib:$NVIDIA_SITE/nvjitlink/lib:$NVIDIA_SITE/cusolver/lib:$NVIDIA_SITE/nccl/lib:$NVIDIA_SITE/cublas/lib:$NVIDIA_SITE/cudnn/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED=0
unset JAX_COMPILATION_CACHE_DIR MAMBA3_LEGACY_NORM JAX_PLATFORMS

echo "[driver] host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES start=$(date -u +%FT%TZ)"
echo "[driver] WT_P=$(git -C "$WT_P" rev-parse --short HEAD)  WT_V=$(git -C "$WT_V" rev-parse --short HEAD)"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
"$PYTHON" -c 'import jax; b = jax.default_backend(); assert b == "gpu", b; print("[preflight] jax backend:", b, jax.devices())' || { echo "[driver] ABORT: no GPU backend"; exit 9; }

CKPT=/lus/lfs1aip2/projects/public/u6gb/model_zoo/m3-goog-78m-u6gb/checkpoint
# The 78M pin's own training corpus (43-col books: 3 metadata + 4*10 levels).
DATA=/lus/lfs1aip2/projects/u6gb/quant/Mamba3_GOOG_pretraining_runs/data_npy/GOOG
# arms_v1 = 2026-08-26 与 88GB 邻居共存的那轮（J0 确定性失败，留档不覆盖）。
ARMS=$TASK/${ARMS_SUBDIR:-arms_v1}
mkdir -p "$ARMS"

run_arm() {
    local name=$1 wt=$2; shift 2
    cd "$wt" || { echo "[driver] ABORT: no worktree $wt"; exit 8; }
    export PYTHONPATH="$wt/src:$wt"
    export GENERATION_DEBUG_LOG="$TASK/logs/${name}.debug.log"
    env | sort > "$TASK/logs/${name}.env"
    echo "[driver] arm $name start=$(date -u +%T) wt=$(basename "$wt") extra=[$*]"
    "$PYTHON" run/base_model/runtime/inference.py --stock GOOG \
        --ckpt_path "$CKPT" --checkpoint_step 28730 \
        --data_dir "$DATA" --test_split 0.001 \
        --n_sequences 4 --batch_size 1 --n_cond_msgs 16 --n_gen_msgs 32 --seed 42 \
        --save_dir "$ARMS/arm_${name}" "$@" > "$TASK/logs/${name}.log" 2>&1
    local rc=$?
    echo "[driver] arm $name rc=$rc end=$(date -u +%T)"
    if [ $rc -ne 0 ]; then
        echo "[driver] ABORT: arm $name failed; last log lines:"
        tail -5 "$TASK/logs/${name}.log"
        exit $rc
    fi
}

run_arm P  "$WT_P"
run_arm P2 "$WT_P"
run_arm O  "$WT_V"
run_arm I  "$WT_V" --inject_step 8 --inject_event_type 1 --inject_side 1 \
                   --inject_qty 777 --inject_price_offset_ticks -1

echo "[driver] DRIVER_DONE $(date -u +%FT%TZ)"
