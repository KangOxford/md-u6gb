#!/bin/bash
# ┌────────────────────────────────────────────────────────────────────────┐
# │ 4 节点舰队自链载荷。资源形状由 sbatch 选项传入，不写在本文件里。        │
# │                                                                        │
# │ 用法：续链必须在命令行显式开启，默认不续链。                            │
# │   sbatch ... fleet_self_chain.sbatch              一次性，跑完即止      │
# │   sbatch ... fleet_self_chain.sbatch --chain      成链，会提交后继      │
# │                                                                        │
# │ 流程是线性的三段：                                                      │
# │   0.5  解析命令行参数                                                   │
# │   A    四个判断  →  一条 sbatch 命令（提交后继）                        │
# │   主   占住节点睡到时限                                                 │
# │                                                                        │
# │ 与 four_node_chain_{24h,12h}.sbatch 的区别：本脚本用                    │
# │ --dependency=afterany 把后继钉在自己之后，存量天然是 1 跑 + 1 排；      │
# │ 那两个脚本不带 dependency（为了让排队等待与运行时间重叠），靠判断        │
# │ 「是否还有别的同名链活着」来限量。两种做法都可以，但都不能像             │
# │ 2026-08-02 那样只查 PENDING —— 那次两小时涨到 97 条、388 节点。         │
# └────────────────────────────────────────────────────────────────────────┘

set -euo pipefail


# ═══════════════════════════════════════════════════════════════════════════
# 0. 常量
# ═══════════════════════════════════════════════════════════════════════════

ROOT="/lus/lfs1aip2/projects/public/u6gb/tasks/u6gb_16_nodes_daily_log"
SELF="$ROOT/fleet_self_chain.sbatch"
BUDGET_LIB="$ROOT/node_budget.sh"
LOG_DIR="$ROOT/slurm_logs"
EVENTS_PATH="${U6GB_CHAIN_EVENTS_PATH:-$ROOT/events.jsonl}"
STOP_CHAIN_FLAG="$ROOT/stop_4node_chain.flag"
mkdir -p "$LOG_DIR"

: "${U6GB_CHAIN_PREFIX:=u6gb-4-node-chain}"
: "${U6GB_CHAIN_SEQ:=1}"
: "${U6GB_CHAIN_MAX:=0}"
: "${U6GB_CHAIN_WALLTIME:=23:59:00}"
: "${U6GB_CHAIN_SLEEP_SECONDS:=86100}"
: "${U6GB_CHAIN_NODES:=4}"
: "${U6GB_CHAIN_GPUS_PER_NODE:=4}"


# ═══════════════════════════════════════════════════════════════════════════
# 0.5 命令行参数
#
#     续链默认关闭。「这条命令会不会自我繁殖」必须写在命令行上，而不是藏在
#     脚本内部：2026-08-02 那次失控两小时无人察觉，一半原因就是提交命令和一个
#     普通的一次性占位作业长得一模一样。
# ═══════════════════════════════════════════════════════════════════════════

CHAIN_ENABLED=0
for arg in "$@"; do
    case "$arg" in
        --chain)    CHAIN_ENABLED=1 ;;
        --no-chain) CHAIN_ENABLED=0 ;;
        *)          echo "unknown argument: $arg" >&2 ;;
    esac
done


# ═══════════════════════════════════════════════════════════════════════════
# 1. 日志
# ═══════════════════════════════════════════════════════════════════════════

now_utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }

append_event() {
    local event_type="$1"; shift
    printf '{"ts":"%s","event_type":"%s","job_id":"%s","job_name":"%s","chain_seq":"%s"%s}\n' \
        "$(now_utc)" "$event_type" "${SLURM_JOB_ID:-unknown}" "${SLURM_JOB_NAME:-unknown}" \
        "$U6GB_CHAIN_SEQ" "$*" >> "$EVENTS_PATH"
}


# ═══════════════════════════════════════════════════════════════════════════
# 2. 前置条件
# ═══════════════════════════════════════════════════════════════════════════

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "SLURM_JOB_ID is not set; this script must run inside sbatch" >&2
    exit 2
fi

NEXT_SEQ=$((U6GB_CHAIN_SEQ + 1))

append_event "self_chain_started" \
    ',"mode":"'"$( ((CHAIN_ENABLED)) && echo chain || echo one_shot )"'"' \
    ',"walltime":"'"$U6GB_CHAIN_WALLTIME"'"' \
    ',"sleep_seconds":"'"$U6GB_CHAIN_SLEEP_SECONDS"'"'


# ═══════════════════════════════════════════════════════════════════════════
# 3. 判断函数
#
#    每个只回答一个是非题，不产生副作用，不互相调用。返回 0 表示「是」。
# ═══════════════════════════════════════════════════════════════════════════

# A0：本次提交是否要求续链（命令行 --chain）
is_chain_requested() {
    (( CHAIN_ENABLED == 1 ))
}

# A1：是否已到达代数上限（U6GB_CHAIN_MAX，0 表示不限）
is_max_generation_reached() {
    [[ "$U6GB_CHAIN_MAX" != "0" && "$NEXT_SEQ" -gt "$U6GB_CHAIN_MAX" ]]
}

# A2：用户是否要求停止续链
is_stop_chain_requested() {
    [[ -e "$STOP_CHAIN_FLAG" ]]
}

# A3：再要 U6GB_CHAIN_NODES 个空占节点，是否还在预算内。
#     只管「没在算」的节点，真实训练不受限；squeue 取不到时 fail closed。
is_node_budget_available() {
    # shellcheck source=/dev/null
    source "$BUDGET_LIB"
    node_budget_ok "$U6GB_CHAIN_NODES"
}


# ═══════════════════════════════════════════════════════════════════════════
# 4. 阶段 A：四个判断，然后一条 sbatch 命令
# ═══════════════════════════════════════════════════════════════════════════

phase_a_submit_successor() {
    # ── 判断 ──────────────────────────────────────────────────────────────
    if ! is_chain_requested; then
        append_event "a_skip_no_chain_flag"
        return 1
    fi
    if is_max_generation_reached; then
        append_event "a_skip_max_generation" ',"max":"'"$U6GB_CHAIN_MAX"'"'
        return 1
    fi
    if is_stop_chain_requested; then
        append_event "a_skip_stop_flag"
        return 1
    fi
    if ! is_node_budget_available; then
        append_event "a_skip_over_budget" \
            ',"budget":"'"$(node_budget_explain "$U6GB_CHAIN_NODES")"'"'
        return 1
    fi

    # ── 命令 ──────────────────────────────────────────────────────────────
    # 后继必须同样带上 --chain，否则链只走一跳就停。
    local next_name output_path submit_output next_job_id
    next_name="$(printf '%s-%03d' "$U6GB_CHAIN_PREFIX" "$NEXT_SEQ")"
    output_path="$LOG_DIR/%x-%j.out"

    submit_output="$(
        sbatch \
            --dependency="afterany:$SLURM_JOB_ID" \
            --nodes="$U6GB_CHAIN_NODES" \
            --ntasks-per-node=1 \
            --gpus-per-node="$U6GB_CHAIN_GPUS_PER_NODE" \
            --mem=0 \
            --time="$U6GB_CHAIN_WALLTIME" \
            --job-name="$next_name" \
            --output="$output_path" \
            --error="$output_path" \
            --export=ALL,U6GB_CHAIN_PREFIX="$U6GB_CHAIN_PREFIX",U6GB_CHAIN_SEQ="$NEXT_SEQ",U6GB_CHAIN_MAX="$U6GB_CHAIN_MAX",U6GB_CHAIN_WALLTIME="$U6GB_CHAIN_WALLTIME",U6GB_CHAIN_SLEEP_SECONDS="$U6GB_CHAIN_SLEEP_SECONDS",U6GB_CHAIN_NODES="$U6GB_CHAIN_NODES",U6GB_CHAIN_GPUS_PER_NODE="$U6GB_CHAIN_GPUS_PER_NODE",U6GB_CHAIN_EVENTS_PATH="$EVENTS_PATH" \
            "$SELF" --chain
    )"

    next_job_id="$(printf '%s\n' "$submit_output" | awk '/Submitted batch job/ {print $4}')"
    append_event "self_chain_successor_submitted" \
        ',"successor_job_id":"'"$next_job_id"'"' \
        ',"successor_job_name":"'"$next_name"'"' \
        ',"dependency":"afterany:'"$SLURM_JOB_ID"'"'
    printf '%s\n' "$submit_output"
    return 0
}


# ═══════════════════════════════════════════════════════════════════════════
# 5. 主流程
#
#    本脚本不做预算的自动取消。执行者只应有一个，而舰队里每一代的 job-name
#    都不同（-001、-002 ...），按名字选举会让每一代都当选，正是要避免的惊群。
#    自动取消统一放在 four_node_chain_{24h,12h}.sbatch 里：它们同名，选举得出
#    唯一执行者。
# ═══════════════════════════════════════════════════════════════════════════

# set -e 之下，判断不通过会返回非零，不能让它把整个作业带走。
phase_a_submit_successor || true

sleep "$U6GB_CHAIN_SLEEP_SECONDS"
append_event "self_chain_finished"
