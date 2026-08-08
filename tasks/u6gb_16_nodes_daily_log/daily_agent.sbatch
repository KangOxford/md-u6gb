#!/bin/bash
#SBATCH --job-name=u6gb-4-log
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --dependency=singleton
#SBATCH --output=/lus/lfs1aip2/projects/public/u6gb/tasks/u6gb_16_nodes_daily_log/slurm_logs/%j.out
#
# ┌────────────────────────────────────────────────────────────────────────┐
# │ 每日日志代理。跑昨天的分析，并（可选地）安排明天那一次。                │
# │                                                                        │
# │ 用法：续链必须在命令行显式开启，默认不续链。                            │
# │   sbatch daily_agent.sbatch              只跑这一次                     │
# │   sbatch daily_agent.sbatch --chain      跑这一次，并安排明天 00:15     │
# │                                                                        │
# │ 流程是线性的两段：                                                      │
# │   A  两个判断  →  一条 sbatch 命令（安排明天）                          │
# │   B  一条命令（跑昨天的分析）                                           │
# │                                                                        │
# │ 顺序是先安排明天再跑分析，这样今天的分析失败不会让整条日链断掉。        │
# └────────────────────────────────────────────────────────────────────────┘
#
# 为什么这里没有节点预算门：本作业只要 1 个 GPU，且带 --dependency=singleton，
# 同名作业最多同时存在一个，结构上不可能失控。而预算门一旦拒绝，这条日链就会
# 永久断掉（它只在启动时安排一次，没有轮询重试），代价大于收益。四节点链那边
# 情况相反，没有 singleton 约束，所以那里必须有预算门。

set -euo pipefail


# ═══════════════════════════════════════════════════════════════════════════
# 0. 常量
# ═══════════════════════════════════════════════════════════════════════════

ROOT="/lus/lfs1aip2/projects/public/u6gb/tasks/u6gb_16_nodes_daily_log"
SELF="$ROOT/daily_agent.sbatch"
STOP_CHAIN_FLAG="$ROOT/stop_daily_agent.flag"
NEXT_RUN="$(date -u -d 'tomorrow 00:15' +%Y-%m-%dT%H:%M:%S)"

mkdir -p "$ROOT/slurm_logs"


# ═══════════════════════════════════════════════════════════════════════════
# 0.5 命令行参数
#
#     续链默认关闭。「这条命令会不会自我繁殖」必须写在命令行上，而不是藏在
#     脚本内部。
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
# 1. 判断函数
# ═══════════════════════════════════════════════════════════════════════════

# A0：本次提交是否要求续链（命令行 --chain）
is_chain_requested() {
    (( CHAIN_ENABLED == 1 ))
}

# A1：用户是否要求停止日链
is_stop_chain_requested() {
    [[ -e "$STOP_CHAIN_FLAG" ]]
}


# ═══════════════════════════════════════════════════════════════════════════
# 2. 阶段 A：两个判断，然后一条 sbatch 命令
# ═══════════════════════════════════════════════════════════════════════════

if ! is_chain_requested; then
    echo "[daily_agent] one-shot mode, tomorrow is not scheduled (加 --chain 才续链)"
elif is_stop_chain_requested; then
    echo "[daily_agent] stop flag present at $STOP_CHAIN_FLAG, tomorrow is not scheduled"
else
    # 后继必须同样带上 --chain，否则日链只走一跳就停。
    sbatch --begin="$NEXT_RUN" "$SELF" --chain
fi


# ═══════════════════════════════════════════════════════════════════════════
# 3. 阶段 B：跑昨天的分析
# ═══════════════════════════════════════════════════════════════════════════

"$ROOT/run_daily_agent.sh" "$(date -u -d yesterday +%F)"
