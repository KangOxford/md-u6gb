#!/bin/bash
# valset CE 评测 wrapper：挂 valset 实体包 → 33 ckpt 循环 eval → 卸载。
# 通过 srun --jobid=5790795 --overlap 执行于 nid010407；只用空闲 GPU（避开
# GPU0 上的 sigma-0 LOB-Bench inference）。
# 用法: run_valset_ce.sh <OUT_DIR> [ONLY_LABELS] [CUDA_DEVS] [NUM_DEVS]
set -euo pipefail
OUT_DIR="$1"
ONLY="${2:-}"
CUDA_DEVS="${3:-1,2,3}"
NUM_DEVS="${4:-3}"

QUANT_ROOT=/lus/lfs1aip2/projects/public/s5e/quant_team/quant
export PATH="$QUANT_ROOT/miniforge3/bin:$PATH"
EXP_DIR=$QUANT_ROOT/AlphaTrade/experiments/exp_R1_Mamba3
VE_DIR=/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/valset_eval
VALSET_SQFS=/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/squashfs/output/shard_valset_v1_30720.squashfs
export PYTHONPATH="$EXP_DIR"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVS"
# 0.60 而非独占式 0.85：GPU1-3 上另有 LOB-Bench 进程的小 context，共存留余量
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.60
# r3 教训：大 [B*13000, d] GEMM 令 Triton autotuner 失败 → 回落 cuBLAS
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false ${XLA_FLAGS:-}"

TMP_BASE="${TMPDIR:-/tmp}"
MOUNT="$TMP_BASE/valset30720_mount_$$"
cleanup() { mountpoint -q "$MOUNT" 2>/dev/null && fusermount -u "$MOUNT" 2>/dev/null || true; }
trap cleanup EXIT
mkdir -p "$MOUNT" "$OUT_DIR"
squashfuse "$VALSET_SQFS" "$MOUNT"
echo "[squashfs] mounted valset 30720 at $MOUNT"
nvidia-smi -L || true
echo "[gpu] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

cd "$EXP_DIR"
EXTRA=()
[ -n "$ONLY" ] && EXTRA+=(--only "$ONLY")
python -u "$VE_DIR/valset_ce_eval.py" \
    --manifest "$VE_DIR/manifest_33ckpt.json" \
    --data_root "$MOUNT" --out_dir "$OUT_DIR" \
    --num_devices "$NUM_DEVS" --n_data_workers 12 "${EXTRA[@]}"
echo "VALSET_WRAPPER_OK"
