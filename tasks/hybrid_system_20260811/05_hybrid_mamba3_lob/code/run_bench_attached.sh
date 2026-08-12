#!/bin/bash
# 把 bench_hybrid.batch 挂到已有 allocation 的一个空闲节点上跑。
#
# 用途是「用掉本来空转的卡」，不是省排队：评测约 1-3 小时，仍属可 attach 的
# 量级，但必须 setsid 起，否则会随会话一起死（2026-08-12 丢过 12,735 步）。
#
# 必需 env: ATTACH_JOBID NODE ARCHITECTURE ARM_ID ARM_NAME CHECKPOINT_PATH CHECKPOINT_STEP
set -uo pipefail
: "${ATTACH_JOBID:?}" "${NODE:?}" "${ARCHITECTURE:?}" "${CHECKPOINT_PATH:?}" "${CHECKPOINT_STEP:?}"
TASKDIR=/lus/lfs1aip2/projects/public/u6gb/tasks/hybrid_system_20260811/05_hybrid_mamba3_lob

# 物理闸门：只用确认为空的那个节点，发现占用就停，绝不动别人的进程。
GATE=$(timeout 120 srun --jobid="$ATTACH_JOBID" --overlap --nodes=1 --ntasks=1 \
  -w "$NODE" --cpu-bind=none bash -c '
    echo "GATE $(hostname) worst=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1) pids=[$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr "\n" " ")]"' 2>&1 | grep "^GATE")
echo "$GATE"
echo "$GATE" | grep -q "pids=\[[0-9]" && { echo "FATAL: $NODE 上有别人的计算进程，中止。" >&2; exit 3; }

for v in $(env | grep -oE '^SLURM[A-Z_]*'); do unset "$v"; done
export ATTACH_JOBID NODE ARCHITECTURE CHECKPOINT_PATH CHECKPOINT_STEP
export ARM_ID="${ARM_ID:-arm}" ARM_NAME="${ARM_NAME:-arm}"
export SLURM_JOB_ID="$ATTACH_JOBID"

exec srun --jobid="$ATTACH_JOBID" --overlap --exact --nodes=1 --ntasks=1 \
     -w "$NODE" --gpus-per-task=4 --cpus-per-task=72 --cpu-bind=none \
     bash "$TASKDIR/bench_scripts/bench_hybrid.batch"
