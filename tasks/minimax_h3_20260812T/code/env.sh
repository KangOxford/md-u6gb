#!/bin/bash
# Shared environment for every H3 stage. Sourced by the batch scripts so the settings
# live in one place instead of drifting between them.
#
# The one non-obvious line is LD_LIBRARY_PATH. peft >= 0.18 probes for
# `transformer_engine`, whose import dlopens libnvrtc, and the probe does not catch
# the RuntimeError when it is missing. torch ships libnvrtc inside its `nvidia/`
# wheel directories, which are not on the loader's default path, so without this the
# whole `diffusers.loaders.peft` import chain fails -- and that chain is on the way to
# `MiniMaxH3Transformer3DModel`. It has to be exported *before* python starts,
# because glibc caches the search path at process start.

TASKDIR=/lus/lfs1aip2/projects/public/u6gb/tasks/minimax_h3_20260812T
export PY="$TASKDIR/venv/bin/python"
export PATH="$TASKDIR/venv/bin:$PATH"

HOST_SP=/projects/public/s5e/quant_team/quant/miniforge3/lib/python3.12/site-packages
NVIDIA_LIBS=""
for d in "$HOST_SP"/nvidia/*/lib; do
    [ -d "$d" ] && NVIDIA_LIBS="${NVIDIA_LIBS}${d}:"
done
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:-}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${TMPDIR:-/tmp}/hf_home"
export HF_HUB_ENABLE_HF_TRANSFER=0
export WANDB_DIR="${TMPDIR:-/tmp}"
export WANDB_MODE=online
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
mkdir -p "$HF_HOME"

echo "[env] LD_LIBRARY_PATH has $(echo "$NVIDIA_LIBS" | tr ':' '\n' | grep -c .) nvidia lib dirs"
echo "[env] libnvrtc: $(ls "$HOST_SP"/nvidia/cuda_nvrtc/lib/libnvrtc.so* 2>/dev/null | head -1 || echo MISSING)"
