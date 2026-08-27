#!/bin/bash
# 候位脚本（限时，最多 10 小时后自动退出）：等两个 4 节点分配里腾出整空节点，
# 依次放三件事，放完即退：
#   1. ask2 token 对齐点 bench：baseline-2k @6509，2,000 条窗口（cond1000/gen1000）
#   2. ask2 token 对齐点 bench：baseline-500 @32001，500 条窗口（cond250/gen250）
#   3. hybrid-2k-32k 训练（P6 六组之 #2，fresh 32k cosine，1N K=20）
#
# bench 同步跑（run_bench_attached 自带 walltime + 逐卡两道检查，撞车安全中止，
# 中止则重新排队）；训练 setsid 脱管，起后用 squeue -s 验 step 存在。
# 判空标准：目标节点 4 张卡显存全部 < 4096 MiB（gtop 头行不可作依据）。
set -u

TD=/lus/lfs1aip2/projects/public/u6gb/tasks/hybrid_system_20260811/05_hybrid_mamba3_lob
WT=/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/hybrid-mamba3-nemotron-20260811
CK500=/lus/lfs1aip2/projects/public/u6gb/sigma-0/checkpoints/j5877859_30nkkohd_5877859
CK2K=$WT/checkpoints/j6000409_gjnf0e03_6000409
ALLOCS="6141106 6136391"
POLL=600
MAX_CYCLES=60
QUEUE="bench2k bench500 hyb32k"

log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }

left_min() {  # $1=jobid -> 剩余分钟（读不到给 0）
    local L; L=$(squeue -j "$1" -h -o "%L" 2>/dev/null | tr -d ' ')
    [ -z "$L" ] && { echo 0; return; }
    python3 - "$L" <<'EOF'
import sys
s=sys.argv[1]; d=0
if '-' in s: d,s=s.split('-',1); d=int(d)
p=[int(x) for x in s.split(':')]
while len(p)<3: p.insert(0,0)
print(d*1440+p[0]*60+p[1])
EOF
}

node_free() {  # $1=alloc $2=node -> rc0 若 4 卡显存全 <4096MiB
    local out
    out=$(timeout 90 srun --overlap --jobid="$1" -w "$2" --ntasks=1 \
          --job-name=probe-ask2 --cpu-bind=none bash -c \
          'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits' 2>/dev/null)
    [ "$(echo "$out" | wc -l)" -ge 4 ] || return 1
    local m; while read -r m; do
        [ -n "$m" ] && [ "$m" -ge 4096 ] && return 1
    done <<< "$out"
    return 0
}

run_item() {  # $1=item $2=alloc $3=node -> rc
    local item=$1 alloc=$2 node=$3 lm mjh
    case "$item" in
      bench2k)
        log "起 bench2k：base2k@6509 on $node ($alloc)（同步，约 2-3h）"
        env ATTACH_JOBID="$alloc" NODE="$node" ARCHITECTURE=mamba3 \
            ARM_ID=base2ktok ARM_NAME=base2ktok \
            CHECKPOINT_PATH="$CK2K" CHECKPOINT_STEP=6509 \
            BENCH_BATCH="$TD/bench_scripts/bench_2k.batch" MIN_WALLTIME_MIN=150 \
            bash "$TD/code/run_bench_attached.sh" \
            > "$TD/logs/ask2_bench2k_20260826.log" 2>&1
        ;;
      bench500)
        log "起 bench500：base500@32001 on $node ($alloc)（同步，约 1-2h）"
        env ATTACH_JOBID="$alloc" NODE="$node" ARCHITECTURE=mamba3 \
            ARM_ID=base500tok ARM_NAME=base500tok \
            CHECKPOINT_PATH="$CK500" CHECKPOINT_STEP=32001 \
            BENCH_BATCH="$TD/bench_scripts/bench_hybrid.batch" MIN_WALLTIME_MIN=90 \
            bash "$TD/code/run_bench_attached.sh" \
            > "$TD/logs/ask2_bench500_20260826.log" 2>&1
        ;;
      hyb32k)
        lm=$(left_min "$alloc")
        if [ "$lm" -lt 240 ]; then
            log "hyb32k：$alloc 只剩 ${lm}min（<240），本轮不放训练"; return 7
        fi
        mjh=$(python3 -c "print(max(1.0, round(($lm-25)/60.0, 1)))")
        log "起 hyb32k on $node ($alloc)，MAX_JOB_HOURS=$mjh（setsid 脱管）"
        setsid nohup env ATTACH_JOBID="$alloc" NNODES_ATTACH=1 NODELIST="$node" \
            SMOKE=0 DEDICATED_ALLOC=0 GATE_OWNER=claude-ctx2k32k \
            STEP_NAME=hyb-m3-2k32k COSINE_STEPS=32000 \
            CHECKPOINT_EVERY=auto LOG_GRAD_NORMS=1 LOG_EVERY=250 \
            EXPECTED_PARAMS=35435423 MAX_JOB_HOURS="$mjh" \
            NODE_LOG_DIR="$WT/logs_lobs5/ctx2k32k_hyb" \
            WANDB_PROJECT=sp500-mamba3-35m-ctx2k32k \
            bash "$TD/code/launch_2k_hybrid.sh" \
            > "$TD/logs/launch_hyb32k_20260826.log" 2>&1 &
        sleep 300
        if squeue -s -j "$alloc" -h 2>/dev/null | grep -q "hyb-m3-2k32k"; then
            log "hyb32k step 已在 squeue 现身，放置成功"
            return 0
        fi
        log "hyb32k 300s 后未见 step，视为失败重排（看 launch_hyb32k 日志）"
        return 8
        ;;
    esac
}

log "候位开始：队列=[$QUEUE]  分配=[$ALLOCS]  POLL=${POLL}s"
cycle=0
while [ -n "$QUEUE" ] && [ "$cycle" -lt "$MAX_CYCLES" ]; do
    cycle=$((cycle+1))
    # 两侧同查：排队侧一并记录在案（放置对象仍是本队列，PENDING 已核对不可放）
    log "cycle $cycle  PENDING=$(squeue -u "$USER" -t PENDING -h 2>/dev/null | wc -l)  队列剩=[$QUEUE]"
    placed=0
    for alloc in $ALLOCS; do
        lm=$(left_min "$alloc")
        [ "$lm" -lt 100 ] && { log "  $alloc 剩 ${lm}min，跳过"; continue; }
        for node in $(scontrol show hostnames "$(squeue -h -j "$alloc" -o %N)" 2>/dev/null); do
            node_free "$alloc" "$node" || continue
            log "  $node ($alloc, 剩${lm}min) 四卡全空"
            item=${QUEUE%% *}
            run_item "$item" "$alloc" "$node"; rc=$?
            if [ "$rc" -eq 0 ]; then
                QUEUE=$(echo "$QUEUE" | sed "s/^$item//;s/^ //")
                log "  $item 完成/放置成功，队列剩=[$QUEUE]"
            else
                log "  $item rc=$rc（关卡拒绝或失败），保留在队列"
            fi
            placed=1
            break 2
        done
    done
    [ -z "$QUEUE" ] && break
    [ "$placed" -eq 0 ] && sleep "$POLL" || sleep 60
done
log "候位退出：队列剩=[$QUEUE]（空=全部放完；非空=超时未放完，需人接手）"
