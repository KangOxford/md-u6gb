#!/bin/bash
# 等训练写出目标 checkpoint，然后立刻起正牌 bench。
#
# 为什么做成一个脚本而不是「我盯着、到点手动起」：本任务已经因为会话结束丢过
# 一次训练（12,735 步）。等待与触发都放在会话外，才不会因为下一次中断而空等。
# 用 setsid 起。
set -uo pipefail
TASKDIR=/lus/lfs1aip2/projects/public/u6gb/tasks/hybrid_system_20260811/05_hybrid_mamba3_lob
CKROOT="${CKROOT:?}"          # 训练的 checkpoint 运行根
WANT_STEPS="${WANT_STEPS:-32001 32000}"
BENCH_JOBID="${BENCH_JOBID:?}"
BENCH_NODE="${BENCH_NODE:?}"
TRAIN_STEP="${TRAIN_STEP:-5980745.433}"

echo "[watch] 等 $CKROOT 出现 {$WANT_STEPS} 之一，或训练 step $TRAIN_STEP 消失"
FOUND=""
while :; do
    for s in $WANT_STEPS; do
        if [ -d "$CKROOT/$s/state" ]; then FOUND=$s; break; fi
    done
    [ -n "$FOUND" ] && break
    if ! squeue --me -s -h -o "%i" 2>/dev/null | grep -q "^${TRAIN_STEP}$"; then
        echo "[watch] 训练 step 已消失，取现存最大 checkpoint"
        FOUND=$(ls -1 "$CKROOT" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
        break
    fi
    sleep 60
done
[ -n "$FOUND" ] || { echo "[watch] FATAL: 没有可用 checkpoint" >&2; exit 2; }
echo "[watch] 选定 step=$FOUND  $(date -u +%H:%M:%SZ)"

# checkpoint 目录可能正在写，等它稳定再用
for _ in $(seq 1 20); do
    A=$(du -s "$CKROOT/$FOUND" 2>/dev/null | cut -f1); sleep 15
    B=$(du -s "$CKROOT/$FOUND" 2>/dev/null | cut -f1)
    [ "$A" = "$B" ] && [ -n "$A" ] && break
done
echo "[watch] checkpoint 已稳定，起 bench $(date -u +%H:%M:%SZ)"

export ATTACH_JOBID="$BENCH_JOBID" NODE="$BENCH_NODE"
export BENCH_WORLD_SIZE=4 BENCH_GPU_OFFSET=0
export ARCHITECTURE=hybrid_mamba3 ARM_ID=hybrid_m3 ARM_NAME=hybrid-m3-nemotron
export CHECKPOINT_PATH="$CKROOT" CHECKPOINT_STEP="$FOUND"
exec bash "$TASKDIR/code/run_bench_attached.sh"
