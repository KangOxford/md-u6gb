#!/bin/bash
# Four-arm do(Q) injection verification (PR #16). Runs inside an srun step on
# ONE node / ONE GPU: byte-identity is node-bound (D-X7), so P, P2, O, I must
# all execute here, sequentially.
#
# Artifacts live in HOME (VAST, shared across nodes): the u6gb Lustre project
# hit its 51.2M-inode hard quota on 2026-08-16 and cannot take new files.
set -u

TASK=/home/u6gb/kangli.u6gb/pr16_doq_artifacts_20260816
WT_P=/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/pr16-base-20260815
WT_V=/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/pr16-verify-20260815
QUANT_ROOT=/lus/lfs1aip2/projects/public/s5e/quant_team/quant
CONDA_PREFIX="$QUANT_ROOT/miniforge3"
PYTHON="$CONDA_PREFIX/bin/python"
NVIDIA_SITE="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia"
TORCH_LIB="$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib"

export TMPDIR="/tmp/${USER:-unknown}/sigma0_doq_$$"
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
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
"$PYTHON" -c 'import jax; b = jax.default_backend(); assert b == "gpu", b; print("[preflight] jax backend:", b, jax.devices())' || { echo "[driver] ABORT: no GPU backend"; exit 9; }

CKPT=/lus/lfs1aip2/projects/public/u6gb/model_zoo/m3-goog-78m-u6gb/checkpoint
# The 78M pin's own training corpus (43-col books: 3 metadata + 4*10 levels).
# The 41-col GOOG2016TO2021 layout predates the wide-book contract and fails
# select_wide_book_levels ("width 38"): first attempt's arm_P shows it.
DATA=/lus/lfs1aip2/projects/u6gb/quant/Mamba3_GOOG_pretraining_runs/data_npy/GOOG
ARMS=$TASK/arms_v2
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
