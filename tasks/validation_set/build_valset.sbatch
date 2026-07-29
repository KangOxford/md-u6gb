#!/bin/bash
#SBATCH --job-name=valset-build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=02:30:00
#SBATCH --partition=workq
#SBATCH --output=/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/logs/valset_%j.out
#
# build_valset.sbatch — 单节点 CPU 作业：挂载 48 个月度 SquashFS shard，
# 用训练 env（torch 2.8.0）重建 48mo/36mo/466tk 三个数据域元数据，
# 逐位复现 seed {5,42,137} 的 shuffle 排列，装配 valset_v1 并 rsync 回 Lustre。
# CPU-only 元数据作业（index.json 驱动 + 单线程 randperm），无 GPU、无 wandb。
set -euo pipefail

QUANT_ROOT=/lus/lfs1aip2/projects/public/s5e/quant_team/quant
export PATH="$QUANT_ROOT/miniforge3/bin:$PATH"
EXP_DIR=$QUANT_ROOT/AlphaTrade/experiments/exp_R1_Mamba3
TASK_DIR=/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set
SQUASHFS_DIR=/lus/lfs1aip2/projects/public/s5e/quant_team/lob_preproc_sp500_squashfs

export PYTHONPATH="$EXP_DIR"
export CUDA_VISIBLE_DEVICES=-1
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=16

TMP_BASE="${TMPDIR:-/tmp}"
export SQUASHFS_MULTI_MOUNT_ROOT="$TMP_BASE/sp500_squashfs_${SLURM_JOB_ID}_0"
export ARTIFACTS_DIR="$TMP_BASE/valset_out_${SLURM_JOB_ID}"
FINAL_DIR="$TASK_DIR/artifacts_valset_v1_j${SLURM_JOB_ID}"

cleanup() {
    if [ -d "$SQUASHFS_MULTI_MOUNT_ROOT" ]; then
        for d in "$SQUASHFS_MULTI_MOUNT_ROOT"/*/; do
            [ -d "$d" ] || continue
            mountpoint -q "$d" 2>/dev/null && fusermount -u "$d" 2>/dev/null || true
        done
    fi
}
trap cleanup EXIT

# ── 挂载 48 个月度 shard（36mo 域复用其中 2023-2025 子集，无需另挂）──
mkdir -p "$SQUASHFS_MULTI_MOUNT_ROOT" "$ARTIFACTS_DIR"
n=0
for y in 2022 2023 2024 2025; do
  for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
    ym="$y-$m"
    SHARD="$SQUASHFS_DIR/shard_${ym}.squashfs"
    MOUNT="$SQUASHFS_MULTI_MOUNT_ROOT/$ym"
    [ -f "$SHARD" ] || { echo "FATAL: missing shard $SHARD" >&2; exit 1; }
    mkdir -p "$MOUNT"
    squashfuse "$SHARD" "$MOUNT" || { echo "FATAL: squashfuse $ym failed" >&2; exit 1; }
    n=$((n+1))
  done
done
echo "[squashfs] mounted $n/48 shards under $SQUASHFS_MULTI_MOUNT_ROOT"

python -u "$TASK_DIR/build_valset.py"
rc=$?
echo "[builder] exit=$rc"
[ $rc -ne 0 ] && exit $rc

# ── 回迁 Lustre（job-id 隔离目录）+ 原子 breadcrumb ──
mkdir -p "$FINAL_DIR"
rsync -a "$ARTIFACTS_DIR/" "$FINAL_DIR/"
BC_TMP="$TASK_DIR/.latest_valset.json.tmp"
cat > "$BC_TMP" <<EOF
{"version": "valset_v1", "path": "$FINAL_DIR", "job_id": "$SLURM_JOB_ID",
 "built_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
EOF
mv -f "$BC_TMP" "$TASK_DIR/latest_valset.json"
echo "[done] artifacts at $FINAL_DIR"
