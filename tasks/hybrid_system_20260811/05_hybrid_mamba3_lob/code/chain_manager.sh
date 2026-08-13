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
# 必需 env: TARGET_STEP LOGDIR_A LOGDIR_B
# 可选: ALLOC_A ALLOC_B STEP_NAME_A STEP_NAME_B COOLDOWN POLL DRY_RUN
set -uo pipefail
trap 'echo "[chain] 退出 rc=$? 于 $(date -u +%H:%M:%SZ)"' EXIT
trap '' HUP PIPE          # 清理节点时的 srun 中断不该带走控制器

: "${TARGET_STEP:?}" "${LOGDIR_A:?}" "${LOGDIR_B:?}"
# step 名要与 launcher 的 STEP_NAME 一致；活性判据就是「这个名字的 step 在不在」
STEP_NAME_A=${STEP_NAME_A:-base-m3-ctx2k}
STEP_NAME_B=${STEP_NAME_B:-hyb-m3-ctx2k}
TASKDIR=/lus/lfs1aip2/projects/public/u6gb/tasks/hybrid_system_20260811/05_hybrid_mamba3_lob
POLL=${POLL:-600}
MAXWAIT=${MAXWAIT:-172800}
STEP_FILE="$TASKDIR/results/.ctx2k_bench_step"

log() { echo "[chain] $(date -u +%H:%M:%SZ) $*"; }

steps_of()    { ls "$1" 2>/dev/null | grep -E '^[0-9]+$' | sort -n; }
top_step()    { steps_of "$1" | tail -1; }

# ── 日志：给目录不给文件名 ──────────────────────────────────────────────────
# 日志文件名里带 jobid，每次换分配续训都会变。写死一个文件名的话，控制器在续训
# 之后仍然盯着上一轮那个不再增长的文件。给目录，每次取最新的那个。
latest_log() { ls -t "$1"/training_*_node0.log 2>/dev/null | head -1; }

ckpt_dir_of() { local f; f=$(latest_log "$1"); [ -z "$f" ] && return 1
                grep -a "Checkpoint dir:" "$f" 2>/dev/null | tail -1 | sed 's|.*Checkpoint dir: ||' | tr -d ' \r'; }

# ── 进度：外化到状态文件，不依赖崩溃日志 ────────────────────────────────────
# ckpt_dir_of 从日志 grep「Checkpoint dir:」，而**崩在启动阶段的那次根本没打印过
# 这一行**。2026-08-13 的 dry-run 里 baseline 就读成 step=?，真要续训会传
# RESTORE_STEP=0 从头再来 —— 丢 4462 步。
#
# 根子上：用日志判断进程状态是循环论证 —— 进程死得越早，日志越不可能包含判据。
# 所以读得到就写进状态文件，读不到就用上一次的。
state_file() { echo "$TASKDIR/results/.ctx2k_state_$1"; }
remember()   { printf '%s %s\n' "$2" "$3" > "$(state_file "$1")".tmp && mv "$(state_file "$1")".tmp "$(state_file "$1")"; }
recall()     { cat "$(state_file "$1")" 2>/dev/null; }

# ── 活性：认自己的 step 名，不认「节点上有没有进程」 ────────────────────────
# 旧写法数节点上任意 compute PID。节点是共享的：2026-08-13 的 dry-run 里
# baseline 已经死了，却因为 nid010779 上有邻居的 dfm/sd3/vLLM 而被判「在跑」，
# 于是永远不会被续训。**「有进程」不等于「我的进程」。**
#
# 现在优先认 Slurm step 名（launcher 已加 --job-name），其次退回日志新鲜度。
# 两条都读不到时返回「在跑」：误判成停了会让 bench 在半截 checkpoint 上起跑，
# 误判成在跑只是晚一轮。
arm_alive() {
    local alloc="$1" stepname="$2" logdir="$3" f age
    if [ -n "$(squeue -s -j "$alloc" -h -o '%j' 2>/dev/null | grep -Fx "$stepname")" ]; then
        return 0
    fi
    f=$(latest_log "$logdir")
    if [ -n "$f" ]; then
        age=$(( $(date +%s) - $(stat -c %Y "$f" 2>/dev/null || echo 0) ))
        [ "$age" -lt "${LOG_FRESH:-1500}" ] && return 0
        return 1
    fi
    return 0
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
        # 别人声明过的分配不接（R4）。这一步在物理探测之前：读锁表是本地文件，
        # 探测要起一个 srun，先做便宜的那个。
        foreign_claimed "$nodes" >/dev/null && continue
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

# 两条臂可以在**不同的** allocation 上。之前假设只有一个 CUR_ALLOC，那在把某一臂
# 迁到空闲分配以加速时就不成立了：alive_on 会拿错误的 jobid 去探另一臂的节点，
# srun 直接失败，而失败被当作「还在跑」——控制器就永远不会推进。
ALLOC_A=${ALLOC_A:-${ATTACH_JOBID:-$(squeue -u "$USER" -h -t RUNNING -o "%i" | head -1)}}
ALLOC_B=${ALLOC_B:-$ALLOC_A}
CUR_ALLOC="$ALLOC_A"
log "起。目标 $TARGET_STEP 优化器步  A: $ALLOC_A step=$STEP_NAME_A   B: $ALLOC_B step=$STEP_NAME_B"

# ── 按臂独立处理，不再要求两臂同时停 ────────────────────────────────────────
#
# 旧逻辑是「两臂都无进程」才动作。2026-08-13 10:19 baseline 被邻居的 vLLM 挤死、
# hybrid 照常跑，于是控制器判「训练在跑」，什么都没做 —— 死掉的那条臂躺了 40 分钟
# 没人管。**一条臂的死活不该由另一条臂的状态决定。**
#
# 续训的落脚点优先选**该臂自己的 allocation**：训练崩溃通常不带走分配（分配有
# 12–18h，训练可能几分钟就死），换个新分配是舍近求远，还要重新排队。
#
# COOLDOWN 不是保守，是必需：启动窗口有 8 分钟（挂 48 个分片 + JAX init），
# 这期间**还没有任何 compute 进程**，alive_on 一律返回「停了」。没有冷却期的话
# 控制器会在启动过程中反复重投，越投越挤。
COOLDOWN=${COOLDOWN:-1200}
declare -A LAUNCHED_AT

# 该臂 allocation 的全部节点；分配没了返回空
nodes_of() { squeue -j "$1" -h -o "%N" 2>/dev/null | tr -d ' '; }

# 这批节点是不是被别人（非 LOCK_OWNER）声明了。只读锁表，不探测。
# 没有它的话，控制器会一直往别人的地盘上投，claim_gate 每次拒、每 COOLDOWN 白试
# 一轮，那条臂就永远停在那儿——无人值守时这等于实验静默死亡。
foreign_claimed() {
    local nl="$1" me="${LOCK_OWNER:-claude-ctx2k}" ns
    local NL=/lus/lfs1aip2/projects/public/u6gb/tasks/node_status/nodelock
    [ -x "$NL" ] || return 1
    ns=$(scontrol show hostnames "$nl" 2>/dev/null | tr '\n' ' ')
    timeout 60 "$NL" ls --json --no-probe 2>/dev/null | \
      NODES="$ns" OWNER="$me" python3 -c '
import json, os, sys
try: reg = json.load(sys.stdin)
except Exception: sys.exit(1)
me = os.environ["OWNER"]
for n in os.environ["NODES"].split():
    w = (reg.get(n) or {}).get("who")
    if w and w != me:
        print(w); sys.exit(0)
sys.exit(1)'
}

resume_arm() {   # $1=arm(A|B) $2=alloc $3=ckpt_dir $4=step
    local arm="$1" alloc="$2" dir="$3" step="$4" nl nn script tag who
    nl=$(nodes_of "$alloc")
    if [ -n "$nl" ] && who=$(foreign_claimed "$nl"); then
        log "$arm 的分配 $alloc 已被 $who 声明，换别处"
        nl=""
    fi
    if [ -z "$nl" ]; then
        log "$arm 的分配 $alloc 不可用，另找"
        local nj nhosts
        if read -r nj nhosts < <(find_free_alloc "$alloc"); then
            alloc="$nj"; nl=$(nodes_of "$nj")
            [ "$arm" = A ] && ALLOC_A="$nj" || ALLOC_B="$nj"
        else
            log "$arm 没有可接手的 allocation，跳过本轮"; return 1
        fi
    fi
    nn=$(scontrol show hostnames "$nl" 2>/dev/null | wc -l)
    [ "${nn:-0}" -lt 1 ] && { log "$arm 无法展开节点表 $nl"; return 1; }
    if [ "$arm" = A ]; then script=launch_2k_baseline.sh; tag=base
                       else script=launch_2k_hybrid.sh;   tag=hyb; fi
    log "$arm 续训：alloc=$alloc nodes=$nl (${nn}N) 从 step=$step"
    # DRY_RUN 存在的理由：这个控制器 2026-08-12 有过一次「bash -n 过了、跑起来
    # 还是错的」（python 引号转义把 awk 变成空操作，配对拿到 4300 而界是 2507）。
    # 唯一能证伪它的是跑一遍看日志，而真跑会起训练。
    if [ "${DRY_RUN:-0}" = "1" ]; then
        log "$arm [DRY_RUN] 本应执行：$script ATTACH_JOBID=$alloc NNODES_ATTACH=$nn NODELIST=$nl RESTORE_STEP=$step"
        LAUNCHED_AT[$arm]=$(date +%s); return 0
    fi
    # DEDICATED_ALLOC 不再设 1。它会无差别 kill 目标节点上的全部 compute 进程，
    # 与 PRIORITY.md 的 R4「抢占只向下且永不由 agent 执行」直接冲突。
    # 现在由 launcher 里的 claim_gate.sh 判：别人声明过或有别人的进程就不起。
    setsid nohup env ATTACH_JOBID="$alloc" NNODES_ATTACH="$nn" NODELIST="$nl" SMOKE=0 \
        DEDICATED_ALLOC=0 LOCK_OWNER="${LOCK_OWNER:-claude-ctx2k}" \
        COSINE_STEPS="${COSINE_STEPS:-6400}" GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-5}" \
        CHECKPOINT_EVERY=auto LOG_GRAD_NORMS=1 LOG_EVERY=250 \
        RESTORE_PATH="$dir" RESTORE_STEP="$step" \
        bash "$TASKDIR/code/$script" \
        > "$TASKDIR/logs/chain_${tag}_$(date -u +%H%M%S).log" 2>&1 &
    LAUNCHED_AT[$arm]=$(date +%s)
    return 0
}

t0=$(date +%s)
while true; do
    now=$(date +%s)
    [ $((now - t0)) -gt "$MAXWAIT" ] && { log "FATAL 超过 ${MAXWAIT}s，放弃"; exit 2; }

    # 进度：读得到就记住，读不到就用记住的。崩在启动阶段的那次日志里没有
    # 「Checkpoint dir:」，此时 recall 是唯一还知道该从 4462 而不是 0 续起的东西。
    for arm in A B; do
        [ "$arm" = A ] && ld="$LOGDIR_A" || ld="$LOGDIR_B"
        d=$(ckpt_dir_of "$ld"); s=$(top_step "$d")
        if [ -n "$d" ] && [ -n "$s" ]; then remember "$arm" "$d" "$s"
        else read -r d s < <(recall "$arm"); fi
        [ "$arm" = A ] && { DA="$d"; SA="$s"; } || { DB="$d"; SB="$s"; }
    done

    a_done=0; b_done=0
    [ -n "${SA:-}" ] && [ "$SA" -ge "$TARGET_STEP" ] && a_done=1
    [ -n "${SB:-}" ] && [ "$SB" -ge "$TARGET_STEP" ] && b_done=1
    [ "$a_done" = 1 ] && [ "$b_done" = 1 ] && { log "两臂均达 $TARGET_STEP（A=$SA B=$SB），转入评测"; break; }

    for arm in A B; do
        if [ "$arm" = A ]; then dn=$a_done; al=$ALLOC_A; ld=$LOGDIR_A; sn=$STEP_NAME_A; dir=${DA:-}; st=${SA:-}
                           else dn=$b_done; al=$ALLOC_B; ld=$LOGDIR_B; sn=$STEP_NAME_B; dir=${DB:-}; st=${SB:-}; fi
        [ "$dn" = 1 ] && { log "$arm 已达标 step=$st"; continue; }
        # 冷却期内不判死活：启动那 8 分钟里 step 还没注册、日志还没写
        since=$(( now - ${LAUNCHED_AT[$arm]:-0} ))
        [ "${LAUNCHED_AT[$arm]:-0}" -gt 0 ] && [ "$since" -lt "$COOLDOWN" ] && \
            { log "$arm 启动中（${since}s/${COOLDOWN}s）"; continue; }
        if arm_alive "$al" "$sn" "$ld"; then
            log "$arm 在跑  step=${st:-?}/$TARGET_STEP"
        else
            log "$arm 停了  step=${st:-?}/$TARGET_STEP  → 续训"
            resume_arm "$arm" "$al" "$dir" "${st:-0}"
        fi
    done
    sleep "$POLL"
done

# 两臂必须在同一步号上评测。先算的落盘，后到的照抄。
DA=$(ckpt_dir_of "$LOGDIR_A"); DB=$(ckpt_dir_of "$LOGDIR_B")
# 不能求交集。CHECKPOINT_EVERY=auto 是按**时间**存点的，而两条臂速度不同
# （实测 1.91 vs 1.35 it/s），所以它们的步号永远不会相同：
#   A: 1430 1895 2675 2987 3143 ... 4300 4339
#   B:  140  365 1265 1491 1604 ... 2507 2536
# comm -12 在这两串上返回空，控制器会在训练刚结束时 FATAL 退出——恰好是最不该
# 失效的时刻。旧的固定步数存点（2000/4000/6000...）碰巧对齐，把这个依赖藏住了；
# 换成 auto 修好了存点频率，同时破坏了依赖它的这一步。
#
# 正确的语义是「在最接近的训练进度上比」而不是「步号必须相同」：取两臂最高步的
# 较小者作上界，各自取自己不超过该上界的最高 checkpoint，然后把两个真实步号都
# 报出来。两者相差多少是可读的，不是被隐藏的。
BOUND_A=$(top_step "$DA"); BOUND_B=$(top_step "$DB")
[ -n "$BOUND_A" ] && [ -n "$BOUND_B" ] || { log "FATAL 读不到步号"; exit 3; }
BOUND=$(( BOUND_A < BOUND_B ? BOUND_A : BOUND_B ))
STEP_A=$(steps_of "$DA" | awk -v b="$BOUND" '$1 <= b' | tail -1)
STEP_B=$(steps_of "$DB" | awk -v b="$BOUND" '$1 <= b' | tail -1)
[ -n "$STEP_A" ] && [ -n "$STEP_B" ] || { log "FATAL 没有不超过 $BOUND 的 checkpoint"; exit 3; }
DIFF=$(( STEP_A > STEP_B ? STEP_A - STEP_B : STEP_B - STEP_A ))
mkdir -p "$(dirname "$STEP_FILE")"
printf 'base=%s hyb=%s bound=%s diff=%s\n' "$STEP_A" "$STEP_B" "$BOUND" "$DIFF" > "$STEP_FILE"
log "评测步号 base=$STEP_A  hyb=$STEP_B  (上界 $BOUND，相差 $DIFF 步)"
# 相差超过 5% 时明确告警：那已经不是「最接近」，而是两个不同训练量的模型。
PCT=$(( DIFF * 100 / (BOUND > 0 ? BOUND : 1) ))
[ "$PCT" -gt 5 ] && log "警告 两臂步数相差 ${PCT}%，对照的可比性下降，报告里必须写出这两个数"

for arm in base hyb; do
    if [ "$arm" = base ]; then D="$DA"; ARCH=mamba3; ND=$(scontrol show hostnames "$(nodes_of "$ALLOC_A")" 2>/dev/null | head -1); STEP="$STEP_A"; else D="$DB"; ARCH=hybrid_mamba3; ND=$(scontrol show hostnames "$(nodes_of "$ALLOC_B")" 2>/dev/null | head -1); STEP="$STEP_B"; fi
    if [ "$arm" = base ]; then AL="$ALLOC_A"; else AL="$ALLOC_B"; fi
    setsid nohup env ATTACH_JOBID="$AL" NODE="$ND" BENCH_WORLD_SIZE=4 BENCH_GPU_OFFSET=0 \
        ARCHITECTURE="$ARCH" ARM_ID="${arm}2k" ARM_NAME="${arm}-m3-ctx2k" \
        CHECKPOINT_PATH="$D" CHECKPOINT_STEP="$STEP" GENERATION_SEED=2026 \
        BENCH_BATCH="$TASKDIR/bench_scripts/bench_2k.batch" \
        bash "$TASKDIR/code/run_bench_attached.sh" \
        > "$TASKDIR/logs/chainbench_${arm}_$(date -u +%H%M%S).log" 2>&1 &
    log "已起 ${arm} 的 bench @ $STEP on $ND"
    sleep 20
done
log "两臂 bench 均已起，控制器完成"
