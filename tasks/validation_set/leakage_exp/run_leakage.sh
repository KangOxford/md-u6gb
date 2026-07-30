#!/bin/bash
# 泄漏行为学实验 wrapper：挂 48 shards → GPU eval 78M 与 350M（串行）→ 卸载。
# 通过 srun --jobid=<预留job> --overlap --gpus=4 执行于 nid010407。
set -euo pipefail
QUANT_ROOT=/lus/lfs1aip2/projects/public/s5e/quant_team/quant
export PATH="$QUANT_ROOT/miniforge3/bin:$PATH"
EXP_DIR=$QUANT_ROOT/AlphaTrade/experiments/exp_R1_Mamba3
LEAK_DIR=/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/leakage_exp
SQUASHFS_DIR=/lus/lfs1aip2/projects/public/s5e/quant_team/lob_preproc_sp500_squashfs
export PYTHONPATH="$EXP_DIR"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
# 大 BSZ eval 的 [B*13000, d] GEMM 令 Triton autotuner "No valid config found"（r3 实测）
# —— 禁 Triton GEMM 回落 cuBLAS，大矩阵乘无压力且省去 autotune。
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false ${XLA_FLAGS:-}"

TMP_BASE="${TMPDIR:-/tmp}"
MOUNT_ROOT="$TMP_BASE/sp500_squashfs_leak_$$"
cleanup() {
    [ -d "$MOUNT_ROOT" ] || return 0
    for d in "$MOUNT_ROOT"/*/; do
        mountpoint -q "$d" 2>/dev/null && fusermount -u "$d" 2>/dev/null || true
    done
}
trap cleanup EXIT

mkdir -p "$MOUNT_ROOT" "$LEAK_DIR/results"
DR=""
for y in 2022 2023 2024 2025; do for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
    ym="$y-$m"; MOUNT="$MOUNT_ROOT/$ym"; mkdir -p "$MOUNT"
    squashfuse "$SQUASHFS_DIR/shard_${ym}.squashfs" "$MOUNT"
    DR="${DR:+$DR,}$MOUNT"
done; done
echo "[squashfs] mounted 48 shards"
nvidia-smi -L || true

cd "$EXP_DIR"
# BSZ 用训练同款（78M 8/GPU、350M 2/GPU）：更大 BSZ 时 repeat_book 的 [B,13000,503] 展开触发 OOM（r4/r5 实测）；
# spawn workers 并行解压喂数据；两 checkpoint 共享一次数据集构建。
python -u "$LEAK_DIR/leakage_test.py" \
    --restore "${LEAK_RESTORES:-checkpoints/j4499538_5vu8avcx_4499538,checkpoints/j4499580_j8cfcraa_4499580}" \
    --label "${LEAK_LABELS:-78M-s5,350M-s5}" --micro_bsz "${LEAK_BSZS:-8,2}" --n_data_workers 0 \
    --num_devices "${LEAK_NUM_DEVICES:-4}" \
    --data_root "$DR" --out_json "$LEAK_DIR/results/leak.json"
echo "[done] both checkpoints"
echo "LEAKAGE_WRAPPER_OK"
