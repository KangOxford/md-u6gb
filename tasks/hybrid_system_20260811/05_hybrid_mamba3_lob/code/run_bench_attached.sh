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

# 物理闸门：**只看我要用的那几张卡**，不看整个节点。
#
# 节点级判定在共享 allocation 上过严。这里的邻居跑的是 --world_size=1 的单卡
# 推理：它在 GPU0 预分配 86 GB，在 GPU1-3 只留 context（各 0.6 GB）。节点级
# 判定看到 PID 就整机拒绝，于是 12 张真正空闲的卡被白白锁住。按卡判定既不碰
# 别人的进程，又能把那部分算力用起来。
# 判据：目标卡上显存 < 阈值即视为可用（context 约 0.6 GB，阈值取 4 GB）。
BENCH_WORLD_SIZE=${BENCH_WORLD_SIZE:-4}
BENCH_GPU_OFFSET=${BENCH_GPU_OFFSET:-0}
MAX_RESIDUAL_MIB=${MAX_RESIDUAL_MIB:-4096}
TARGET_GPUS=$(seq $BENCH_GPU_OFFSET $((BENCH_GPU_OFFSET + BENCH_WORLD_SIZE - 1)) | tr '\n' ',' | sed 's/,$//')
echo "[gate] $NODE 目标卡 [$TARGET_GPUS]，阈值 ${MAX_RESIDUAL_MIB} MiB"
GATE=$(timeout 120 srun --jobid="$ATTACH_JOBID" --overlap --nodes=1 --ntasks=1 \
  -w "$NODE" --cpu-bind=none bash -c '
    used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
    worst=0
    for g in $(echo "'"$TARGET_GPUS"'" | tr "," " "); do
      m=$(echo "$used" | awk -F", *" -v g="$g" "\$1==g{print \$2}")
      [ -n "$m" ] && [ "$m" -gt "$worst" ] && worst=$m
    done
    echo "GATE $(hostname) target_worst_mib=$worst"' 2>&1 | grep "^GATE")
echo "$GATE"
WORST=$(echo "$GATE" | sed -E 's/.*target_worst_mib=([0-9]+).*/\1/')
if [ -z "$WORST" ] || [ "$WORST" -gt "$MAX_RESIDUAL_MIB" ]; then
    echo "FATAL: $NODE 目标卡 [$TARGET_GPUS] 上残留 ${WORST:-?} MiB > ${MAX_RESIDUAL_MIB}，中止，不动别人的进程。" >&2
    exit 3
fi

for v in $(env | grep -oE '^SLURM[A-Z_]*'); do unset "$v"; done
export ATTACH_JOBID NODE ARCHITECTURE CHECKPOINT_PATH CHECKPOINT_STEP
export BENCH_WORLD_SIZE BENCH_GPU_OFFSET
export ARM_ID="${ARM_ID:-arm}" ARM_NAME="${ARM_NAME:-arm}"
export SLURM_JOB_ID="$ATTACH_JOBID"

exec srun --jobid="$ATTACH_JOBID" --overlap --exact --nodes=1 --ntasks=1 \
     -w "$NODE" --cpus-per-task=72 --cpu-bind=none \
     bash "$TASKDIR/bench_scripts/bench_hybrid.batch"
