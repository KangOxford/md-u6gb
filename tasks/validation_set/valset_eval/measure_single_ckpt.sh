#!/bin/bash
# 单 checkpoint 资源测量：PREALLOCATE=false（按需分配）跑一个真实队列条目，
# 后台 nvidia-smi 采样得出真实显存峰值。结果 json 落主 results 目录=队列进度。
# 用法: measure_single_ckpt.sh <GPU> <LABEL> <OUT_DIR> <MANIFEST>
set -uo pipefail
GPU=$1; LABEL=$2; OUT_DIR=$3; MANIFEST=$4
QUANT_ROOT=/lus/lfs1aip2/projects/public/s5e/quant_team/quant
export PATH="$QUANT_ROOT/miniforge3/bin:$PATH"
EXP_DIR=$QUANT_ROOT/AlphaTrade/experiments/exp_R1_Mamba3
VE_DIR=/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/valset_eval
VALSET_SQFS=/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/squashfs/output/shard_valset_v1_30720.squashfs
export PYTHONPATH="$EXP_DIR"
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false ${XLA_FLAGS:-}"

TMP_BASE="${TMPDIR:-/tmp}"
MOUNT="$TMP_BASE/valset_measure_$$"
TRACE="$OUT_DIR/mem_trace_gpu${GPU}_${LABEL//@/_}.csv"
cleanup() { kill "$SAMPLER" 2>/dev/null; mountpoint -q "$MOUNT" && fusermount -u "$MOUNT" 2>/dev/null || true; }
trap cleanup EXIT
mkdir -p "$MOUNT" "$OUT_DIR"
squashfuse "$VALSET_SQFS" "$MOUNT"

BASE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -dc 0-9)
( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -dc 0-9; echo; sleep 2; done >> "$TRACE" ) &
SAMPLER=$!
echo "[measure] gpu=$GPU label=$LABEL baseline=${BASE}MiB (on-demand alloc, no preallocation)"
T0=$(date +%s)
cd "$EXP_DIR"
CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -u "$VE_DIR/valset_ce_eval.py" \
    --manifest "$MANIFEST" --only "$LABEL" \
    --data_root "$MOUNT" --out_dir "$OUT_DIR" \
    --num_devices 1 --n_data_workers 12
RC=$?
T1=$(date +%s)
kill "$SAMPLER" 2>/dev/null
PEAK=$(sort -n "$TRACE" | tail -1)
echo "[measure] RESULT gpu=$GPU label=$LABEL rc=$RC wall=$((T1-T0))s baseline=${BASE}MiB peak=${PEAK}MiB real_need=$((PEAK-BASE))MiB"
