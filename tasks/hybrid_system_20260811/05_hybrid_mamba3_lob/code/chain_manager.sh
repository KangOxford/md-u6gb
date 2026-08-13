#!/bin/bash
# 一个控制器管到底：训练停了就判断该续训还是该评测，不需要人在场。
#
# 为什么必须合成一个而不是两个自动机
# ----------------------------------
# 之前是「训练一停就跑 bench」。那个判据在单个 allocation 内成立，跨 allocation
# 就错了：分配到期同样会让训练停止，而那时步数远没跑够，bench 会在半截
# checkpoint 上起跑，产出一份看起来正常的错误结果。
#
# 「训练停了」有两种截然不同的原因，必须分开：
#   跑够了      -> 评测
#   分配没了    -> 换一个分配续训，续不上才评测
#
# 两个独立的守望器无法表达这个分支，因为「该不该续」取决于另一件事（还有没有
# 空闲 allocation），而那不是任何一个守望器自己的状态。
#
# 判据
# ----
# 训练是否在跑：看该臂节点上有没有 compute 进程。不用 squeue —— RUNNING 不代表
# 在训练（2026-08-12 有过 4h27m 空转）。
# 步数：读 checkpoint 目录里的数字子目录，而目录本身从该臂的节点日志实时读，
# 因为每次续训 wandb id 变、目录跟着变。
#
# 必需 env: TARGET_STEP LOG_A LOG_B NODE_A NODE_B
set -uo pipefail
trap 'echo "[chain] 退出 rc=$? 于 $(date -u +%H:%M:%SZ)"' EXIT
trap '' HUP PIPE          # 清理节点时的 srun 中断不该带走控制器

: "${TARGET_STEP:?}" "${LOG_A:?}" "${LOG_B:?}" "${NODE_A:?}" "${NODE_B:?}"
TASKDIR=/lus/lfs1aip2/projects/public/u6gb/tasks/hybrid_system_20260811/05_hybrid_mamba3_lob
POLL=${POLL:-600}
MAXWAIT=${MAXWAIT:-172800}
STEP_FILE="$TASKDIR/results/.ctx2k_bench_step"

log() { echo "[chain] $(date -u +%H:%M:%SZ) $*"; }

ckpt_dir_of() { grep -a "Checkpoint dir:" "$1" 2>/dev/null | tail -1 | sed 's|.*Checkpoint dir: ||' | tr -d ' \r'; }
steps_of()    { ls "$1" 2>/dev/null | grep -E '^[0-9]+$' | sort -n; }
top_step()    { steps_of "$1" | tail -1; }

# 该臂节点上还有没有 compute 进程。读不到时返回 1（当作还在跑）：误判成「停了」
# 会让 bench 在半截 checkpoint 上起跑，误判成「在跑」只是晚一轮。
alive_on() {
    local alloc="$1" node="$2" n
    n=$(timeout 90 srun --jobid="$alloc" --overlap --nodes=1 --ntasks=1 \
        -w "$node" --cpu-bind=none \
        nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
        | sort -u | wc -l)
    [ "${n:-1}" -gt 0 ]
}

# 找一个能接手的 allocation：自己的、RUNNING、四节点、剩余够久、且卡是空的。
# 排除当前这个（它正是要被替换的那个）和别人锁着的节点。
find_free_alloc() {
    local cur="$1" jid nodes left
    while read -r jid nodes left; do
        [ "$jid" = "$cur" ] && continue
        [ -z "$jid" ] && continue
        # 剩余时间不足两小时的不接：启动加挂载就要十几分钟
        local m; m=$(python3 -c "
import sys
s=sys.argv[1]; d=0
if '-' in s: d,s=s.split('-',1); d=int(d)
p=[int(x) for x in s.split(':')]
while len(p)<3: p.insert(0,0)
print(d*1440+p[0]*60+p[1])" "$left" 2>/dev/null)
        [ "${m:-0}" -lt 120 ] && continue
        local hs; hs=$(scontrol show hostnames "$nodes" 2>/dev/null | tr '\n' ' ')
        local first; first=$(echo "$hs" | awk '{print $1}')
        [ -z "$first" ] && continue
        # 只探第一台：四节点的 chain 作业要么整体空要么整体忙
        local used
        used=$(timeout 90 srun --jobid="$jid" --overlap --nodes=1 --ntasks=1 \
               -w "$first" --cpu-bind=none \
               nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
               | sort -n | tail -1)
        [ -n "$used" ] && [ "$used" -lt 4096 ] && { echo "$jid $hs"; return 0; }
    done < <(squeue -u "$USER" -h -t RUNNING -o "%i %N %L" | awk '$0 ~ /nid/')
    return 1
}

CUR_ALLOC=${ATTACH_JOBID:-$(squeue -u "$USER" -h -t RUNNING -o "%i" | head -1)}
log "起。目标 $TARGET_STEP 优化器步，当前 allocation=$CUR_ALLOC"

t0=$(date +%s); idle=0
while true; do
    now=$(date +%s)
    [ $((now - t0)) -gt "$MAXWAIT" ] && { log "FATAL 超过 ${MAXWAIT}s，放弃"; exit 2; }

    a_live=0; b_live=0
    alive_on "$CUR_ALLOC" "$NODE_A" && a_live=1
    alive_on "$CUR_ALLOC" "$NODE_B" && b_live=1
    DA=$(ckpt_dir_of "$LOG_A"); DB=$(ckpt_dir_of "$LOG_B")
    SA=$(top_step "$DA"); SB=$(top_step "$DB")

    if [ "$a_live" = 1 ] || [ "$b_live" = 1 ]; then
        idle=0
        log "训练在跑 (A=$a_live B=$b_live)  最高步 A=${SA:-?} B=${SB:-?}"
        sleep "$POLL"; continue
    fi

    idle=$((idle+1))
    log "两臂都无进程（连续 $idle 次）  最高步 A=${SA:-?} B=${SB:-?}"
    [ "$idle" -lt 2 ] && { sleep 120; continue; }

    # 两臂都停了。跑够了吗？
    if [ -n "$SA" ] && [ -n "$SB" ] && [ "$SA" -ge "$TARGET_STEP" ] && [ "$SB" -ge "$TARGET_STEP" ]; then
        log "两臂均已达 $TARGET_STEP，转入评测"
        break
    fi

    # 没跑够：找地方续
    if read -r NJ NHOSTS < <(find_free_alloc "$CUR_ALLOC"); then
        set -- $NHOSTS
        log "找到可接手的 allocation $NJ ($1 $2 $3 $4)，续训"
        NA="$1"; NB="$3"
        setsid nohup env ATTACH_JOBID="$NJ" NNODES_ATTACH=2 NODELIST="$1,$2" SMOKE=0 DEDICATED_ALLOC=1 \
            COSINE_STEPS="${COSINE_STEPS:-6400}" CHECKPOINT_EVERY=auto LOG_GRAD_NORMS=1 \
            RESTORE_PATH="$DA" RESTORE_STEP="$SA" \
            bash "$TASKDIR/code/launch_2k_baseline.sh" \
            > "$TASKDIR/logs/chain_base_$(date -u +%H%M%S).log" 2>&1 &
        sleep 30
        setsid nohup env ATTACH_JOBID="$NJ" NNODES_ATTACH=2 NODELIST="$3,$4" SMOKE=0 DEDICATED_ALLOC=1 \
            COSINE_STEPS="${COSINE_STEPS:-6400}" CHECKPOINT_EVERY=auto LOG_GRAD_NORMS=1 \
            RESTORE_PATH="$DB" RESTORE_STEP="$SB" \
            bash "$TASKDIR/code/launch_2k_hybrid.sh" \
            > "$TASKDIR/logs/chain_hyb_$(date -u +%H%M%S).log" 2>&1 &
        CUR_ALLOC="$NJ"; NODE_A="$NA"; NODE_B="$NB"; idle=0
        log "已在 $NJ 上续训 A@$SA B@$SB，继续守"
        sleep 600; continue
    fi

    log "没有可接手的 allocation。就现有步数评测（A=$SA B=$SB）"
    break
done

# 两臂必须在同一步号上评测。先算的落盘，后到的照抄。
DA=$(ckpt_dir_of "$LOG_A"); DB=$(ckpt_dir_of "$LOG_B")
COMMON=$(comm -12 <(steps_of "$DA") <(steps_of "$DB") | sort -n | tail -1)
[ -n "$COMMON" ] || { log "FATAL 两臂没有共同步号"; exit 3; }
mkdir -p "$(dirname "$STEP_FILE")"
echo "$COMMON" > "$STEP_FILE"
log "评测步号 = $COMMON  (A=$DA  B=$DB)"

for arm in base hyb; do
    if [ "$arm" = base ]; then D="$DA"; ARCH=mamba3; ND="$NODE_A"; else D="$DB"; ARCH=hybrid_mamba3; ND="$NODE_B"; fi
    setsid nohup env ATTACH_JOBID="$CUR_ALLOC" NODE="$ND" BENCH_WORLD_SIZE=4 BENCH_GPU_OFFSET=0 \
        ARCHITECTURE="$ARCH" ARM_ID="${arm}2k" ARM_NAME="${arm}-m3-ctx2k" \
        CHECKPOINT_PATH="$D" CHECKPOINT_STEP="$COMMON" GENERATION_SEED=2026 \
        BENCH_BATCH="$TASKDIR/bench_scripts/bench_2k.batch" \
        bash "$TASKDIR/code/run_bench_attached.sh" \
        > "$TASKDIR/logs/chainbench_${arm}_$(date -u +%H%M%S).log" 2>&1 &
    log "已起 ${arm} 的 bench @ $COMMON on $ND"
    sleep 20
done
log "两臂 bench 均已起，控制器完成"
