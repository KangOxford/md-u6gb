#!/bin/bash
# 训练一停就用两臂共有的最高 checkpoint 跑 2k bench。
#
# 判据不是「到某个固定步」。两条臂的 MAX_JOB_HOURS 不同、速率也不同，会停在
# 不同步数上；盯死 32001 的守望器永远不会触发。受控对照要的是**同步长**，不是
# 某个特定步长，所以取两臂都存在的最高 checkpoint。
#
# 训练是否结束用 GPU 上还有没有本臂的进程判断，不用 squeue：RUNNING 不代表在训
# （有过 4h27m 空转），而 GPU 进程消失是训练真的停了。
#
# 必需 env: CKPT_A CKPT_B ARM_ID ARCHITECTURE NODELIST ATTACH_JOBID TRAIN_NODE
set -uo pipefail
# 上一版死得无声无息：进程没了，日志最后一行还停在「等训练结束」，看不出是超时、
# 是 srun 出错、还是被谁杀了。守望器的价值全在它还活着，所以它必须在退出时说明
# 自己为什么退出。
trap 'echo "[watch] ${ARM_ID:-?} 退出 rc=$? 于 $(date -u +%H:%M:%SZ)"' EXIT
# 守望器必须比它守望的东西更耐活。它每轮用 srun 探一次卡，而清理节点、杀残留
# 进程这类动作会连带把探测 step 打断；上一版就是这样在 01:15 被信号带走的，
# 连 EXIT trap 都没来得及打印。忽略 HUP/PIPE，让一次失败的探测只是一次失败的
# 探测。TERM 与 INT 保留，否则就没法正常收掉它了。
trap '' HUP PIPE
# 每次轮询都写一行心跳，这样「最后一次醒着是什么时候」是可读的，而不是靠猜。
: "${CKPT_A:?}" "${CKPT_B:?}" "${ARM_ID:?}" "${ARCHITECTURE:?}" "${NODELIST:?}" "${ATTACH_JOBID:?}" "${TRAIN_NODE:?}"
TASKDIR=/lus/lfs1aip2/projects/public/u6gb/tasks/hybrid_system_20260811/05_hybrid_mamba3_lob
POLL=${POLL:-600}
MAXWAIT=${MAXWAIT:-72000}

steps_of() {   # 目录名里的纯数字步号；checkpoint 目录条目很少（2000 步一存），ls 安全
    ls "$1" 2>/dev/null | grep -E '^[0-9]+$' | sort -n
}

echo "[watch] $ARM_ID 等训练结束（看 $TRAIN_NODE 上还有没有 GPU 进程）"
t0=$(date +%s); idle=0
while true; do
    n=$(timeout 60 srun --jobid="$ATTACH_JOBID" --overlap --nodes=1 --ntasks=1 \
        -w "$TRAIN_NODE" --cpu-bind=none \
        nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u | wc -l)
    # 探测失败时 n 为空，按 1 处理 = 当作训练还在跑。误判成「停了」会让
    # bench 在半个 checkpoint 上起跑；误判成「还在跑」只是晚一轮。
    if [ "${n:-1}" -eq 0 ] 2>/dev/null; then
        idle=$((idle+1))
        echo "[watch] $(date -u +%H:%M:%SZ) ${ARM_ID} 卡上无进程（连续 $idle 次）"
        # 连续两次为空才算停，避开 checkpoint 保存期间的短暂空窗
        [ "$idle" -ge 2 ] && { echo "[watch] $ARM_ID 训练已停"; break; }
    else
        idle=0
        echo "[watch] $(date -u +%H:%M:%SZ) ${ARM_ID} 训练仍在跑（$n 个进程）"
    fi
    now=$(date +%s)
    [ $((now - t0)) -gt "$MAXWAIT" ] && { echo "[watch] FATAL 等超时" >&2; exit 2; }
    sleep "$POLL"
done

# 两臂停止时刻不同（MAX_JOB_HOURS 与速率都不同），先停的那个算出的共有最高步
# 会低于后停的。若各算各的，两臂就会在不同步号上评测，对照当场失效。
# 所以第一个算出来的把步号落盘，第二个照抄。用 O_EXCL 的 noclobber 做原子占位。
STEP_FILE="${STEP_FILE:-$TASKDIR/results/.ctx2k_bench_step}"
mkdir -p "$(dirname "$STEP_FILE")"
COMMON=$(comm -12 <(steps_of "$CKPT_A") <(steps_of "$CKPT_B") | sort -n | tail -1)
[ -n "$COMMON" ] || { echo "[watch] FATAL 两臂没有共同步号" >&2; exit 3; }
if (set -o noclobber; echo "$COMMON" > "$STEP_FILE") 2>/dev/null; then
    echo "[watch] $ARM_ID 首个到达，定步号 = $COMMON（已落 $STEP_FILE）"
else
    COMMON=$(cat "$STEP_FILE")
    echo "[watch] $ARM_ID 沿用已定步号 = $COMMON"
fi
[ -d "$([ "$ARM_ID" = "base2k" ] && echo "$CKPT_A" || echo "$CKPT_B")/$COMMON/state" ] || {
    echo "[watch] FATAL $ARM_ID 没有步号 $COMMON 的 checkpoint" >&2; exit 4; }

MY_CKPT=$([ "$ARM_ID" = "base2k" ] && echo "$CKPT_A" || echo "$CKPT_B")
echo "[watch] $ARM_ID 起 2k bench @ $COMMON  $(date -u +%H:%M:%SZ)"
exec env ATTACH_JOBID="$ATTACH_JOBID" NODE="$NODELIST" \
     BENCH_WORLD_SIZE="${BENCH_WORLD_SIZE:-4}" BENCH_GPU_OFFSET="${BENCH_GPU_OFFSET:-0}" \
     ARCHITECTURE="$ARCHITECTURE" ARM_ID="$ARM_ID" ARM_NAME="${ARM_NAME:-$ARM_ID}" \
     CHECKPOINT_PATH="$MY_CKPT" CHECKPOINT_STEP="$COMMON" \
     GENERATION_SEED="${GENERATION_SEED:-2026}" \
     BENCH_BATCH="$TASKDIR/bench_scripts/bench_2k.batch" \
     bash "$TASKDIR/code/run_bench_attached.sh"
