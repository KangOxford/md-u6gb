#!/bin/bash
# 训练一到目标步就接 2k bench。不空等，也不提前抢卡。
#
# 判据用 checkpoint 目录而不是日志或 squeue：日志里 tqdm 的计数是 micro 步，
# squeue RUNNING 也不代表训练在推进（2026-08-12 有过 4h27m 空转）。产物存在
# 才是产物存在。
#
# 必需 env: CKPT_ROOT TARGET_STEP ARCHITECTURE ARM_ID ARM_NAME NODELIST ATTACH_JOBID
set -uo pipefail
: "${CKPT_ROOT:?}" "${TARGET_STEP:?}" "${ARCHITECTURE:?}" "${ARM_ID:?}" "${NODELIST:?}" "${ATTACH_JOBID:?}"
TASKDIR=/lus/lfs1aip2/projects/public/u6gb/tasks/hybrid_system_20260811/05_hybrid_mamba3_lob
POLL=${POLL:-600}
MAXWAIT=${MAXWAIT:-64800}          # 18h
GEN_SEED=${GENERATION_SEED:-2026}

echo "[watch] $ARM_ID 等 $CKPT_ROOT/$TARGET_STEP/state 出现（每 ${POLL}s 看一次）"
t0=$(date +%s)
while true; do
    if [ -d "$CKPT_ROOT/$TARGET_STEP/state" ]; then
        # 目录出现不等于写完。连续两次大小不变才算稳定。
        a=$(du -s "$CKPT_ROOT/$TARGET_STEP" 2>/dev/null | awk '{print $1}')
        sleep 60
        b=$(du -s "$CKPT_ROOT/$TARGET_STEP" 2>/dev/null | awk '{print $1}')
        [ -n "$a" ] && [ "$a" = "$b" ] && { echo "[watch] $ARM_ID checkpoint $TARGET_STEP 稳定"; break; }
    fi
    now=$(date +%s)
    [ $((now - t0)) -gt "$MAXWAIT" ] && { echo "[watch] FATAL 等超过 ${MAXWAIT}s，放弃" >&2; exit 2; }
    sleep "$POLL"
done

echo "[watch] $ARM_ID 起 2k bench $(date -u +%H:%M:%SZ)"
exec env ATTACH_JOBID="$ATTACH_JOBID" NODE="${NODELIST}" \
     BENCH_WORLD_SIZE="${BENCH_WORLD_SIZE:-4}" BENCH_GPU_OFFSET="${BENCH_GPU_OFFSET:-0}" \
     ARCHITECTURE="$ARCHITECTURE" ARM_ID="$ARM_ID" ARM_NAME="${ARM_NAME:-$ARM_ID}" \
     CHECKPOINT_PATH="$CKPT_ROOT" CHECKPOINT_STEP="$TARGET_STEP" GENERATION_SEED="$GEN_SEED" \
     BENCH_BATCH="$TASKDIR/bench_scripts/bench_2k.batch" \
     bash "$TASKDIR/code/run_bench_attached.sh"
