#!/bin/bash
# valset_v1 物化：挂 48 源 shards → 切片提取（30720 与 307200 两档）→ mksquashfs
# → 挂载自检（逐字节 + dataloader 兼容）→ rsync 回 Lustre。
# 经 srun --jobid=<预留job> --overlap 执行（CPU-only，可与 GPU 泄漏实验并行）。
set -euo pipefail
QUANT_ROOT=/lus/lfs1aip2/projects/public/s5e/quant_team/quant
export PATH="$QUANT_ROOT/miniforge3/bin:/home/u6gb/kangli.u6gb/miniforge3/bin:$PATH"
SQ_DIR=/lus/lfs1aip2/projects/public/s5e/quant_team/lob_preproc_sp500_squashfs
TASK=/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set
ART=$TASK/artifacts_valset_v1_j5790795
export PYTHONPATH=$QUANT_ROOT/AlphaTrade/experiments/exp_R1_Mamba3

TMP_BASE="${TMPDIR:-/tmp}"
WORK="$TMP_BASE/valset_matz_$$"
SRC_ROOT="$WORK/src_mounts"
cleanup() {
    for d in "$SRC_ROOT"/*/ "$WORK"/check_*/; do
        mountpoint -q "$d" 2>/dev/null && fusermount -u "$d" 2>/dev/null || true
    done
}
trap cleanup EXIT
mkdir -p "$SRC_ROOT"

DR=""
for y in 2022 2023 2024 2025; do for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
    ym="$y-$m"; M="$SRC_ROOT/$ym"; mkdir -p "$M"
    squashfuse "$SQ_DIR/shard_${ym}.squashfs" "$M"
    DR="${DR:+$DR,}$M"
done; done
echo "[squashfs] 48 source shards mounted"

for SIZE in 30720 307200; do
    NAME="valset_v1_${SIZE}"
    TREE="$WORK/tree_$NAME"
    echo "===== materialize $NAME ====="
    python -u "$TASK/squashfs/materialize_valset.py" \
        --subset_npy "$ART/val_subset_${SIZE}.npy" --shard_name "$NAME" \
        --data_root "$DR" --out_tree "$TREE" --nproc 48

    SHARD="$WORK/shard_${NAME}.squashfs"
    mksquashfs "$TREE" "$SHARD" -comp zstd -Xcompression-level 3 \
        -no-progress -processors 48 -noappend
    echo "[mksquashfs] $(du -sh "$SHARD" | cut -f1) -> $SHARD"

    CHK="$WORK/check_$NAME"; mkdir -p "$CHK"
    squashfuse "$SHARD" "$CHK"
    python -u "$TASK/squashfs/verify_valset_squashfs.py" \
        --mount "$CHK" --provenance "$WORK/provenance_${NAME}.npz" \
        --n_samples "$SIZE" --n_check 2048
    fusermount -u "$CHK"

    DEST=$TASK/squashfs/output
    mkdir -p "$DEST"
    rsync -a "$SHARD" "$WORK/provenance_${NAME}.npz" "$DEST/"
    (cd "$DEST" && sha256sum "shard_${NAME}.squashfs" >> SHA256SUMS.txt)
    rm -rf "$TREE" "$SHARD"
    echo "[done] $NAME -> $DEST/shard_${NAME}.squashfs"
done
echo "MATERIALIZE_WRAPPER_OK"
