#!/bin/bash
# 在计算节点上执行：环境 → 挂载 GOOG Jan-2026 → 4 个单 GPU 进程并行生成。
set -euo pipefail

WORKDIR="${WORKDIR:?}"
QUANT_ROOT="${QUANT_ROOT:?}"
PYTHON="${PYTHON:?}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:?}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:?}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:?}"
INFERENCE_DIR="${INFERENCE_DIR:?}"
OUTPUT_ROOT="${OUTPUT_ROOT:?}"
SAMPLE_INDICES_FILE="${SAMPLE_INDICES_FILE:?}"
N_SEQUENCES="${N_SEQUENCES:-3136}"
BATCH_SIZE="${BATCH_SIZE:-8}"
N_COND_MSGS="${N_COND_MSGS:-250}"
N_GEN_MSGS="${N_GEN_MSGS:-250}"
GENERATION_SEED="${GENERATION_SEED:-2026}"
ARCHITECTURE="${ARCHITECTURE:-mamba3}"
# 可配的 rank 数与 GPU 偏移。默认 4 张卡从 GPU0 起，与原版一致。
# 需要偏移是因为 attach 到共享节点时 GPU0 常被别人的单卡作业占着，而
# GPU1-3 是真空闲的：那种作业跑 --world_size=1，只在 GPU0 预分配，其余三张
# 卡上只留 context，gtop 因为看到 PID 就标 held，实际可用。
WORLD_SIZE=${BENCH_WORLD_SIZE:-4}
GPU_OFFSET=${BENCH_GPU_OFFSET:-0}

echo "[gen] host=$(hostname) ckpt=$CHECKPOINT_PATH step=$CHECKPOINT_STEP"

if [ $((N_SEQUENCES % (WORLD_SIZE * BATCH_SIZE))) -ne 0 ]; then
    echo "[gen] FATAL: $N_SEQUENCES 不能被 $WORLD_SIZE x $BATCH_SIZE 整除" >&2
    exit 2
fi
[ -f "$SAMPLE_INDICES_FILE" ] || { echo "[gen] FATAL: 缺样本索引 $SAMPLE_INDICES_FILE" >&2; exit 2; }
echo "[gen] sample_indices sha256=$(sha256sum "$SAMPLE_INDICES_FILE" | awk '{print $1}')"

JOB_TMP="/tmp/${USER:-kangli.u6gb}/m3bench/$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$INFERENCE_DIR" "$OUTPUT_ROOT" "$JOB_TMP"
export TMPDIR="$JOB_TMP"
export MPLCONFIGDIR="$JOB_TMP/matplotlib"
mkdir -p "$MPLCONFIGDIR"

module load cuda/12.6
CONDA_PREFIX="$QUANT_ROOT/miniforge3"
NVIDIA_SITE="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia"
TORCH_LIB="$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$TORCH_LIB:$NVIDIA_SITE/cuda_nvrtc/lib:$NVIDIA_SITE/cuda_runtime/lib:$NVIDIA_SITE/cusparse/lib:$NVIDIA_SITE/cuda_cupti/lib:$NVIDIA_SITE/cufft/lib:$NVIDIA_SITE/nvjitlink/lib:$NVIDIA_SITE/cusolver/lib:$NVIDIA_SITE/nccl/lib:$NVIDIA_SITE/cublas/lib:$NVIDIA_SITE/cudnn/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$WORKDIR/src:$WORKDIR:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"

"$PYTHON" -c 'import jax, torch; b=jax.default_backend(); assert b=="gpu", b; print(f"[cuda-preflight] jax={b} torch={torch.__version__}")'

# GOOG Jan-2026 评测数据 + 500 档 wide book，都从 SquashFS 挂到节点本地
source "$BENCHMARK_ROOT/pipeline/_squashfs_helpers.sh"
infer_squashfs_setup GOOG "${INFERENCE_MONTHS:-2026-01}" \
    "${INFERENCE_SQUASHFS_DIR:-/lus/lfs1aip2/projects/public/s5e/quant_team/lob_preproc_sp500_squashfs}"
infer_squashfs_setup_wide_book GOOG "${INFERENCE_MONTHS:-2026-01}" \
    "${WIDE_SQUASHFS_DIR:-/projects/public/s5e/quant_team/recon_2026-05/output/squashfs}" \
    "${WIDE_LEVELS:-500}"
trap infer_squashfs_cleanup EXIT

echo "[gen] sequences=$N_SEQUENCES cond=$N_COND_MSGS gen=$N_GEN_MSGS seed=$GENERATION_SEED"
echo "[gen] data=$INFER_DATA_DIR_NODE wide=$INFER_WIDE_BOOK_DIR_NODE"

PIDS=()
for LOCAL_RANK in $(seq 0 $((WORLD_SIZE - 1))); do
    RANK_TMP="$JOB_TMP/rank_$LOCAL_RANK"
    mkdir -p "$RANK_TMP"
    CUDA_VISIBLE_DEVICES=$((LOCAL_RANK + GPU_OFFSET)) TMPDIR="$RANK_TMP" "$PYTHON" -u -B \
        "$WORKDIR/run/base_model/runtime/inference.py" \
        --architecture="$ARCHITECTURE" \
        --token_mode=26tok \
        --ckpt_path="$CHECKPOINT_PATH" \
        --checkpoint_step="$CHECKPOINT_STEP" \
        --data_dir="$INFER_DATA_DIR_NODE" \
        --wide_book_dir="$INFER_WIDE_BOOK_DIR_NODE" \
        --wide_levels="${WIDE_LEVELS:-500}" \
        --save_dir="$INFERENCE_DIR" \
        --stock=GOOG \
        --n_sequences="$N_SEQUENCES" \
        --batch_size="$BATCH_SIZE" \
        --n_cond_msgs="$N_COND_MSGS" \
        --n_gen_msgs="$N_GEN_MSGS" \
        --test_split=1.0 \
        --sample_indices_file="$SAMPLE_INDICES_FILE" \
        --seed="$GENERATION_SEED" \
        --rank="$LOCAL_RANK" --world_size="$WORLD_SIZE" \
        > "$OUTPUT_ROOT/inference_rank${LOCAL_RANK}.log" 2>&1 &
    PIDS+=("$!")
done

GEN_EXIT=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "[gen] rank $i 失败，见 $OUTPUT_ROOT/inference_rank${i}.log" >&2
        GEN_EXIT=1
    fi
done
[ "$GEN_EXIT" -eq 0 ] || exit "$GEN_EXIT"

"$PYTHON" "$WORKDIR/run/benchmarking/validate_model_zoo_evaluation.py" inference \
    "$INFERENCE_DIR" --expected-sequences "$N_SEQUENCES" \
    --rows "$N_GEN_MSGS" --world-size "$WORLD_SIZE" \
    --expected-indices-file "$SAMPLE_INDICES_FILE" \
    --expected-dataset-length "${EXPECTED_DATASET_LENGTH:-226002}" \
    > "$OUTPUT_ROOT/inference_inventory.json"

echo "[gen] 完成，inventory 见 $OUTPUT_ROOT/inference_inventory.json"
