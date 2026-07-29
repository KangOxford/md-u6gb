#!/bin/bash
# 恢复流程：30,720 档提取+打包已完成（L1 已全过），只重跑修复版校验 + 落盘；
# 然后跑 307,200 档全流程。WORK 目录沿用首跑的 valset_matz_53990。
set -euo pipefail
QUANT_ROOT=/lus/lfs1aip2/projects/public/s5e/quant_team/quant
export PATH="$QUANT_ROOT/miniforge3/bin:/home/u6gb/kangli.u6gb/miniforge3/bin:$PATH"
SQ_DIR=/lus/lfs1aip2/projects/public/s5e/quant_team/lob_preproc_sp500_squashfs
TASK=/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set
ART=$TASK/artifacts_valset_v1_j5790795
export PYTHONPATH=$QUANT_ROOT/AlphaTrade/experiments/exp_R1_Mamba3

WORK="${TMPDIR:-/tmp}/valset_matz_53990"
[ -d "$WORK" ] || { echo "FATAL: $WORK missing (node/TMPDIR changed?)"; exit 1; }
SRC_ROOT="$WORK/src_mounts"
cleanup() {
    for d in "$SRC_ROOT"/*/ "$WORK"/check_*/; do
        mountpoint -q "$d" 2>/dev/null && fusermount -u "$d" 2>/dev/null || true
    done
}
trap cleanup EXIT

# 重挂 48 源 shards（上次失败退出时已被卸载）
DR=""
for y in 2022 2023 2024 2025; do for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
    ym="$y-$m"; M="$SRC_ROOT/$ym"; mkdir -p "$M"
    mountpoint -q "$M" 2>/dev/null || squashfuse "$SQ_DIR/shard_${ym}.squashfs" "$M"
    DR="${DR:+$DR,}$M"
done; done
echo "[squashfs] 48 source shards mounted"

DEST=$TASK/squashfs/output
mkdir -p "$DEST"

finish_tier() {
    local NAME="$1" SIZE="$2"
    local SHARD="$WORK/shard_${NAME}.squashfs"
    local CHK="$WORK/check_$NAME"
    mkdir -p "$CHK"
    mountpoint -q "$CHK" 2>/dev/null || squashfuse "$SHARD" "$CHK"
    python -u "$TASK/squashfs/verify_valset_squashfs.py" \
        --mount "$CHK" --provenance "$WORK/provenance_${NAME}.npz" \
        --n_samples "$SIZE" --n_check 2048
    fusermount -u "$CHK"
    rsync -a "$SHARD" "$WORK/provenance_${NAME}.npz" "$DEST/"
    (cd "$DEST" && sha256sum "shard_${NAME}.squashfs" >> SHA256SUMS.txt)
    rm -rf "$WORK/tree_$NAME" "$SHARD"
    echo "[done] $NAME -> $DEST/shard_${NAME}.squashfs"
}

# ── 30,720 档：shard 已打包、L1 已全过 → 只重跑修复版校验 + 落盘 ──
finish_tier valset_v1_30720 30720

# ── 307,200 档：全流程 ──
NAME=valset_v1_307200
TREE="$WORK/tree_$NAME"
python -u "$TASK/squashfs/materialize_valset.py" \
    --subset_npy "$ART/val_subset_307200.npy" --shard_name "$NAME" \
    --data_root "$DR" --out_tree "$TREE" --nproc 48
mksquashfs "$TREE" "$WORK/shard_${NAME}.squashfs" -comp zstd -Xcompression-level 3 \
    -no-progress -processors 48 -noappend
finish_tier "$NAME" 307200
echo "MATERIALIZE_WRAPPER_OK"
