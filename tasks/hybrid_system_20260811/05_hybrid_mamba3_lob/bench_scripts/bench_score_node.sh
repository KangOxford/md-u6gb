#!/bin/bash
# 在计算节点上执行 LOBBench 打分（纯 CPU）。
set -euo pipefail

WORKDIR="${WORKDIR:?}"
QUANT_ROOT="${QUANT_ROOT:?}"
PYTHON="${PYTHON:?}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:?}"
INFERENCE_DIR="${INFERENCE_DIR:?}"
LOBBENCH_OUTPUT="${LOBBENCH_OUTPUT:?}"
SUMMARY="${SUMMARY:?}"
RUN_NAME="${RUN_NAME:-sigma0-mamba3-35m}"
LOBBENCH_WORKERS="${LOBBENCH_WORKERS:-48}"

echo "[score] host=$(hostname) inference=$INFERENCE_DIR workers=$LOBBENCH_WORKERS"

JOB_TMP="/tmp/${USER:-kangli.u6gb}/m3bench_score/$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$JOB_TMP" "$LOBBENCH_OUTPUT"
export TMPDIR="$JOB_TMP"
export MPLCONFIGDIR="$JOB_TMP/matplotlib"
mkdir -p "$MPLCONFIGDIR"

CONDA_PREFIX="$QUANT_ROOT/miniforge3"
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONPATH="$WORKDIR/src:$WORKDIR:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

"$PYTHON" -u -B "$WORKDIR/run/benchmarking/run_lobbench.py" \
    --run-name="$RUN_NAME" \
    --inference-dir="$INFERENCE_DIR" \
    --output-dir="$LOBBENCH_OUTPUT" \
    --stock=GOOG --period=2026-01 \
    --benchmark-root="$BENCHMARK_ROOT" \
    --python="$PYTHON" --n-workers="$LOBBENCH_WORKERS" \
    --run-id="${LOBBENCH_RUN_ID:-m3_35m}" \
    --summary="$SUMMARY"

echo "[score] 完成，summary=$SUMMARY"
