#!/bin/bash
# build_valset_v2_worker.sh — 构建 SP500_2022_2025_Validation_Version_1。
#
# 与 build_valset.sbatch 的差异：
#   1. 跑 build_valset_v2.py（配方去掉 36 月子域排除项，规模固定 0.5%，加 (ticker,month) 分层）
#   2. 不再需要 36mo 域，只建 48mo 与 466tk 两个域，省约 12 分钟
#   3. 以 attach step 方式跑在已有 allocation 上，不另排队
#
# 用法（从 login 节点）：
#   srun --jobid=<ALLOC> --overlap --nodelist=<nid> --nodes=1 --ntasks=1 \
#        --cpus-per-task=72 --cpu-bind=none bash build_valset_v2_worker.sh
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
# 独立 worker：屏蔽宿主 allocation 的多节点变量，否则 JAX 会去 join 别人的协调服务
export SLURM_NNODES=1 SLURM_NTASKS=1 SLURM_PROCID=0 SLURM_NPROCS=1

HOST=$(hostname)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
# 不继承提交者 TMPDIR；/tmp 是本节点 334 GB tmpfs。挂载点带主机名+PID（死挂载不可复用）
WORK=/tmp/valset_v2_${HOST}_$$
export SQUASHFS_MULTI_MOUNT_ROOT="$WORK/mounts"
export ARTIFACTS_DIR="$WORK/out"
FINAL_DIR="$TASK_DIR/artifacts_valset_v2_${STAMP}"

cleanup() {
    if [ -d "$SQUASHFS_MULTI_MOUNT_ROOT" ]; then
        for d in "$SQUASHFS_MULTI_MOUNT_ROOT"/*/; do
            [ -d "$d" ] || continue
            mountpoint -q "$d" 2>/dev/null && fusermount -u "$d" 2>/dev/null || true
        done
    fi
}
trap cleanup EXIT
mkdir -p "$SQUASHFS_MULTI_MOUNT_ROOT" "$ARTIFACTS_DIR"

echo "[env] node=$HOST nproc=$(nproc) mem_free=$(free -g | awk '/^Mem:/{print $7}')G"
echo "[env] WORK=$WORK  FINAL_DIR=$FINAL_DIR"

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
echo "[squashfs] mounted $n/48 shards  $(date -u +%H:%M:%SZ)"

python -u "$TASK_DIR/build_valset_v2.py"
echo "[builder] exit=$?"

mkdir -p "$FINAL_DIR"
rsync -a "$ARTIFACTS_DIR/" "$FINAL_DIR/"
BC="$TASK_DIR/.latest_valset_v2.json.tmp"
cat > "$BC" <<EOF
{"version": "SP500_2022_2025_Validation_Version_1", "path": "$FINAL_DIR",
 "built_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)", "node": "$HOST"}
EOF
mv -f "$BC" "$TASK_DIR/latest_valset_v2.json"
echo "[done] artifacts at $FINAL_DIR"
ls -la "$FINAL_DIR"
echo "BUILD_V2_WORKER_OK $(date -u +%FT%TZ)"
