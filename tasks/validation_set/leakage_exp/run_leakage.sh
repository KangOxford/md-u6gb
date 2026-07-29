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
python -u "$LEAK_DIR/leakage_test.py" \
    --restore checkpoints/j4499538_5vu8avcx_4499538 --label 78M-s5 \
    --micro_bsz 8 --data_root "$DR" --out_json "$LEAK_DIR/results/leak_78M_s5.json"
echo "[done] 78M"
python -u "$LEAK_DIR/leakage_test.py" \
    --restore checkpoints/j4499580_j8cfcraa_4499580 --label 350M-s5 \
    --micro_bsz 2 --data_root "$DR" --out_json "$LEAK_DIR/results/leak_350M_s5.json"
echo "[done] 350M"
echo "LEAKAGE_WRAPPER_OK"
