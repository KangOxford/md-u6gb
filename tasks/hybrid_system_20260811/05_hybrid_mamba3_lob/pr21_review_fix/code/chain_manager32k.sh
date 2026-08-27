#!/bin/bash
# 32k 代（全预算矩阵）的接续链控制器。派生自 code/chain_manager.sh，改动：
#
#   1. 多组：ARMS 列表（默认 "A B P"），A=baseline B=hybrid P=matched。
#      每组变量按 <NAME>_<arm> 命名（nameref），互不牵连。
#   2. 首启不归它管：某组的 NODE_LOG_DIR 里还没有任何日志时判「在跑」并跳过，
#      首启由候位脚本（ask2_then_hyb_watcher）完成 —— 这样两个自动机不抢首启，
#      而控制器接手一切续段。
#   3. 逐节点选空：分配可能被别的线占走部分节点，find_free_alloc 那种
#      「探第一台代表整个分配」不再成立。这里逐节点探（4 卡显存全 <4096MiB），
#      取能整除 K 的最大子集（4/2/1），EFFECTIVE_BSZ 恒等式保证换节点数不改实验。
#   4. 达标收刀按**组自己的节点**（.ctx2k32k_nodes_<arm>），不按整个分配 ——
#      三组可能落在同一分配的不同节点上，按分配杀会误伤别组。
#   5. 不自动转评测（32k 点的评测按 ask3 规格另行编排）。
#   6. 续段 env 全量显式：STEP_NAME / NODE_LOG_DIR / COSINE_STEPS=32000 /
#      WANDB_PROJECT / EXPECTED_PARAMS，不再依赖 launcher 默认值恰好相同。
#      GRAD_ACCUM_STEPS 一律不传，K 由 launcher 从节点数推导并硬校验。
#
# 必需 env: 无（全部有默认）。可选：ARMS TARGET_STEP POLL COOLDOWN STALL_SECS
#           MAXWAIT DRY_RUN MIN_LEFT_MIN
set -uo pipefail
trap 'echo "[chain32k] 退出 rc=$? 于 $(date -u +%H:%M:%SZ)"' EXIT
trap '' HUP PIPE

TASKDIR=/lus/lfs1aip2/projects/public/u6gb/tasks/hybrid_system_20260811/05_hybrid_mamba3_lob
WT=/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/hybrid-mamba3-nemotron-20260811

TARGET_STEP=${TARGET_STEP:-32000}
ARMS=${ARMS:-"A B P"}
POLL=${POLL:-600}
COOLDOWN=${COOLDOWN:-1200}
STALL_SECS=${STALL_SECS:-900}
MAXWAIT=${MAXWAIT:-345600}
MIN_LEFT_MIN=${MIN_LEFT_MIN:-120}
WANDB_PROJECT_32K=sp500-mamba3-35m-ctx2k32k

# 每组四件套：日志目录 / step 名 / launcher / 附加 env（KEY=VAL 空格分隔，值不带空格）
LOGDIR_A=${LOGDIR_A:-$WT/logs_lobs5/ctx2k32k_base}
STEP_NAME_A=${STEP_NAME_A:-base-m3-2k32k}
SCRIPT_A=${SCRIPT_A:-launch_2k_baseline.sh}
EXTRA_A=${EXTRA_A:-"EXPECTED_PARAMS=33610439"}

LOGDIR_B=${LOGDIR_B:-$WT/logs_lobs5/ctx2k32k_hyb}
STEP_NAME_B=${STEP_NAME_B:-hyb-m3-2k32k}
SCRIPT_B=${SCRIPT_B:-launch_2k_hybrid.sh}
EXTRA_B=${EXTRA_B:-"EXPECTED_PARAMS=35435423"}

LOGDIR_P=${LOGDIR_P:-$WT/logs_lobs5/ctx2k32k_pm}
STEP_NAME_P=${STEP_NAME_P:-pm-m3-2k32k}
SCRIPT_P=${SCRIPT_P:-launch_2k_hybrid.sh}
EXTRA_P=${EXTRA_P:-"EXPECTED_PARAMS=33609998 HYBRID_ATTN_D_FF=1135"}

log() { echo "[chain32k] $(date -u +%H:%M:%SZ) $*"; }
steps_of()   { ls "$1" 2>/dev/null | grep -E '^[0-9]+$' | sort -n; }
top_step()   { steps_of "$1" | tail -1; }
latest_log() { ls -t "$1"/training_*_node0.log 2>/dev/null | head -1; }
ckpt_dir_of(){ local f; f=$(latest_log "$1"); [ -z "$f" ] && return 1
               grep -a "Checkpoint dir:" "$f" 2>/dev/null | tail -1 | sed 's|.*Checkpoint dir: ||' | tr -d ' \r'; }

state_file() { echo "$TASKDIR/results/.ctx2k32k_state_$1"; }
nodes_file() { echo "$TASKDIR/results/.ctx2k32k_nodes_$1"; }
remember()   { printf '%s %s\n' "$2" "$3" > "$(state_file "$1")".tmp && mv "$(state_file "$1")".tmp "$(state_file "$1")"; }
recall()     { cat "$(state_file "$1")" 2>/dev/null; }

left_min() {
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

# 活性（承 chain_manager.sh，逐条理由见那边注释）：
# 收尾遗言 > 步号停滞 > step 名在 > 日志新鲜度；无日志 = 首启未发生 = 不动。
arm_alive() {
    local alloc="$1" stepname="$2" logdir="$3" f age last
    f=$(latest_log "$logdir")
    [ -z "$f" ] && return 0
    last=$(tr '\r' '\n' < "$f" 2>/dev/null | grep -av "^ *$" | tail -1)
    case "$last" in
        *"squashfs] unmounted"*|*"Step watchdog timeout"*|*"Training complete"*|*"[squashfs] umount"*)
            return 1 ;;
    esac
    local cnt_f="$TASKDIR/results/.ctx2k32k_progress_$(basename "$logdir")"
    local cur now_t; local prev=""; local prev_t=""
    cur=$(tr '\r' '\n' < "$f" 2>/dev/null | grep -aoE "[0-9]+/[0-9]+ \[" | tail -1 | cut -d/ -f1)
    now_t=$(date +%s)
    if [ -n "$cur" ]; then
        [ -f "$cnt_f" ] && read -r prev prev_t < "$cnt_f"
        if [ "$cur" != "${prev:-}" ]; then
            printf '%s %s\n' "$cur" "$now_t" > "$cnt_f"
        elif [ -n "${prev_t:-}" ] && [ $(( now_t - prev_t )) -gt "$STALL_SECS" ]; then
            log "  ${logdir##*/} 步号停在 $cur 已 $(( (now_t - prev_t)/60 )) 分钟 → 判死"
            return 1
        fi
    fi
    [ -n "$alloc" ] && [ -n "$(squeue -s -j "$alloc" -h -o '%j' 2>/dev/null | grep -Fx "$stepname")" ] && return 0
    age=$(( now_t - $(stat -c %Y "$f" 2>/dev/null || echo 0) ))
    [ "$age" -lt "${LOG_FRESH:-1500}" ] && return 0
    return 1
}

# 一个分配里的空节点清单（逐节点 4 卡显存全 <4096MiB）
free_nodes_in() {
    local jid="$1" nodes n out ok m
    nodes=$(squeue -h -j "$jid" -o "%N" 2>/dev/null | tr -d ' ')
    [ -z "$nodes" ] && return 0
    for n in $(scontrol show hostnames "$nodes" 2>/dev/null); do
        out=$(timeout 90 srun --overlap --jobid="$jid" -w "$n" --ntasks=1 \
              --job-name=probe-c32k --cpu-bind=none \
              nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
        [ "$(echo "$out" | grep -c .)" -ge 4 ] || continue
        ok=1
        while read -r m; do [ -n "$m" ] && [ "$m" -ge 4096 ] && ok=0; done <<< "$out"
        [ "$ok" = 1 ] && echo "$n"
    done
}

# 挑落脚点：遍历自己账户 RUNNING 的多节点分配，取空节点数∈{4,2,1} 的最大可用子集
pick_slot() {   # stdout: "jid n1[,n2...] count"；找不到 rc=1
    local jid free cnt take
    while read -r jid; do
        [ -z "$jid" ] && continue
        [ "$(left_min "$jid")" -lt "$MIN_LEFT_MIN" ] && continue
        free=$(free_nodes_in "$jid")
        cnt=$(echo "$free" | grep -c . || true)
        [ "${cnt:-0}" -lt 1 ] && continue
        # 不允许 1：单节点没有 shard_map，K 被静默忽略（train_helpers.py:1821），
        # 有效批量会变成 4 —— 2026-08-27 baseline 首段就是这样报废的。
        for take in 4 2; do
            if [ "$cnt" -ge "$take" ]; then
                echo "$jid $(echo "$free" | head -"$take" | paste -sd,) $take"
                return 0
            fi
        done
    done < <(squeue -u "$USER" -h -t RUNNING -S "-L" -o "%i %D" | awk '$2>=1{print $1}')
    return 1
}

resume_arm() {   # $1=arm $2=ckpt_dir $3=step
    local arm="$1" dir="$2" step="$3" jid nl nn lm mjh
    local -n _ld=LOGDIR_$arm; local -n _sn=STEP_NAME_$arm
    local -n _sc=SCRIPT_$arm; local -n _ex=EXTRA_$arm
    local slot; slot=$(pick_slot) || { log "$arm 没有可用落脚点，等下一轮"; return 1; }
    read -r jid nl nn <<< "$slot"
    lm=$(left_min "$jid")
    mjh=$(python3 -c "print(max(1.0, round(($lm-25)/60.0, 1)))")
    log "$arm 续段：alloc=$jid nodes=$nl (${nn}N) 从 step=$step，MAX_JOB_HOURS=$mjh"
    if [ "${DRY_RUN:-0}" = "1" ]; then
        log "$arm [DRY_RUN] 本应执行：$_sc ATTACH_JOBID=$jid NNODES_ATTACH=$nn NODELIST=$nl RESTORE_STEP=$step $_ex"
        LAUNCHED_AT[$arm]=$(date +%s); return 0
    fi
    local restore_env=""
    if [ -n "$dir" ] && [ "${step:-0}" -gt 0 ] 2>/dev/null; then
        restore_env="RESTORE_PATH=$dir RESTORE_STEP=$step"
    fi
    setsid nohup env ATTACH_JOBID="$jid" NNODES_ATTACH="$nn" NODELIST="$nl" SMOKE=0 \
        DEDICATED_ALLOC=0 GATE_OWNER=claude-ctx2k32k \
        STEP_NAME="$_sn" COSINE_STEPS=32000 \
        CHECKPOINT_EVERY=auto LOG_GRAD_NORMS=1 LOG_EVERY=250 \
        NODE_LOG_DIR="$_ld" WANDB_PROJECT="$WANDB_PROJECT_32K" \
        MAX_JOB_HOURS="$mjh" \
        $_ex $restore_env \
        bash "$TASKDIR/code/$_sc" \
        > "$TASKDIR/logs/chain32k_${arm}_$(date -u +%m%dT%H%M%S).log" 2>&1 &
    echo "$nl" > "$(nodes_file "$arm")"
    echo "$jid" > "$(nodes_file "$arm").alloc"
    LAUNCHED_AT[$arm]=$(date +%s)
    return 0
}

declare -A STOPPED LAUNCHED_AT
stop_at_target() {   # 只在该组自己的节点上、按 worktree 路径匹配收刀
    local arm="$1" nl jid nn
    [ -n "${STOPPED[$arm]:-}" ] && return 0
    nl=$(cat "$(nodes_file "$arm")" 2>/dev/null); jid=$(cat "$(nodes_file "$arm").alloc" 2>/dev/null)
    [ -z "$nl" ] || [ -z "$jid" ] && { log "$arm 达标但没有节点记录，跳过收刀"; STOPPED[$arm]=1; return 0; }
    squeue -h -j "$jid" -o "%T" 2>/dev/null | grep -q RUNNING || { STOPPED[$arm]=1; return 0; }
    nn=$(echo "$nl" | tr ',' '\n' | grep -c .)
    log "$arm 达标，在它自己的节点 [$nl] 上收刀"
    local _ce; _ce=$(env | grep -oE '^SLURM[A-Z_]*' | sed 's/^/-u /' | tr '\n' ' ')
    timeout 200 env $_ce srun --jobid="$jid" --overlap --nodes="$nn" --ntasks="$nn" \
        --ntasks-per-node=1 -w "$nl" --cpu-bind=none bash -c '
          K=""
          for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do
            tr "\0" " " < /proc/$p/cmdline 2>/dev/null \
              | grep -q "hybrid-mamba3-nemotron" && K="$K $p"
          done
          [ -z "$K" ] && { echo "[stop] $(hostname) left=0"; exit 0; }
          kill $K 2>/dev/null; sleep 10; kill -9 $K 2>/dev/null
          echo "[stop] $(hostname) killed=[$K]"' 2>&1 | grep "^\[stop\]" | sed 's/^/  /'
    STOPPED[$arm]=1
}

log "起。目标 $TARGET_STEP 优化器步，组=[$ARMS]，POLL=${POLL}s"
t0=$(date +%s)
while true; do
    now=$(date +%s)
    [ $((now - t0)) -gt "$MAXWAIT" ] && { log "超过 ${MAXWAIT}s，退出（人接手）"; exit 2; }

    all_done=1
    for arm in $ARMS; do
        # nameref 循环重绑定：不先 unset 的话，第二轮 declare -n 可能变成
        # 「透过旧引用赋值」而不是重绑定，B 的日志目录会读成 A 的。
        unset -n _ld _sn 2>/dev/null
        declare -n _ld=LOGDIR_$arm; declare -n _sn=STEP_NAME_$arm
        d=$(ckpt_dir_of "$_ld"); s=$(top_step "$d" 2>/dev/null)
        if [ -n "$d" ] && [ -n "$s" ]; then remember "$arm" "$d" "$s"
        else read -r d s < <(recall "$arm"); fi
        jid=$(cat "$(nodes_file "$arm").alloc" 2>/dev/null)

        if [ -n "${s:-}" ] && [ "$s" -ge "$TARGET_STEP" ]; then
            log "$arm 已达标 step=$s"
            stop_at_target "$arm"
            continue
        fi
        all_done=0

        if [ ! -e "$_ld" ] || [ -z "$(latest_log "$_ld")" ]; then
            log "$arm 首启未发生（候位脚本负责），本轮不动"
            continue
        fi
        since=$(( now - ${LAUNCHED_AT[$arm]:-0} ))
        [ "${LAUNCHED_AT[$arm]:-0}" -gt 0 ] && [ "$since" -lt "$COOLDOWN" ] && \
            { log "$arm 启动中（${since}s/${COOLDOWN}s）"; continue; }
        if arm_alive "${jid:-}" "$_sn" "$_ld"; then
            log "$arm 在跑  step=${s:-?}/$TARGET_STEP"
        else
            log "$arm 停了  step=${s:-?}/$TARGET_STEP → 续段"
            resume_arm "$arm" "${d:-}" "${s:-0}"
        fi
    done
    [ "$all_done" = 1 ] && { log "全部组达标 $TARGET_STEP，TRAINING_COMPLETE"; break; }
    sleep "$POLL"
done
log "控制器完成（评测按 ask3 规格另行编排）"
