#!/bin/bash
# ┌────────────────────────────────────────────────────────────────────────┐
# │ 提交 4 节点自链舰队的第一棒。                                           │
# │                                                                        │
# │ 用法：续链必须显式开启，默认不续链。                                    │
# │   ./submit_self_chain.sh                     只提交一棒，跑完即止       │
# │   ./submit_self_chain.sh --chain             成链，每一棒安排下一棒     │
# │                                                                        │
# │ 位置参数（--chain 可以放在任意位置，不占位）：                          │
# │   $1 prefix           默认 u6gb-4-node-chain                            │
# │   $2 seq              默认 1                                            │
# │   $3 walltime         默认 23:59:00                                     │
# │   $4 sleep_seconds    默认 86100                                        │
# │   $5 max_generations  默认 0（不限）                                    │
# │                                                                        │
# │ 环境变量：U6GB_CHAIN_NODES（默认 4）、U6GB_CHAIN_GPUS_PER_NODE（默认 4）│
# └────────────────────────────────────────────────────────────────────────┘

set -euo pipefail

ROOT="/lus/lfs1aip2/projects/public/u6gb/tasks/u6gb_16_nodes_daily_log"
BUDGET_LIB="$ROOT/node_budget.sh"


# ═══════════════════════════════════════════════════════════════════════════
# 0.5 命令行参数
#
#     --chain 从位置参数里摘出来，剩下的照原样按位置解析，所以老的调用方式
#     不受影响，加不加 --chain 也不影响 $1..$5 的含义。
# ═══════════════════════════════════════════════════════════════════════════

CHAIN_ENABLED=0
positional=()
for arg in "$@"; do
    case "$arg" in
        --chain)    CHAIN_ENABLED=1 ;;
        --no-chain) CHAIN_ENABLED=0 ;;
        *)          positional+=("$arg") ;;
    esac
done
set -- ${positional[@]+"${positional[@]}"}


# ═══════════════════════════════════════════════════════════════════════════
# 1. 参数
# ═══════════════════════════════════════════════════════════════════════════

prefix="${1:-u6gb-4-node-chain}"
seq="${2:-1}"
walltime="${3:-23:59:00}"
sleep_seconds="${4:-86100}"
max_generations="${5:-0}"
nodes="${U6GB_CHAIN_NODES:-4}"
gpus_per_node="${U6GB_CHAIN_GPUS_PER_NODE:-4}"

job_name="$(printf '%s-%03d' "$prefix" "$seq")"
output_path="$ROOT/slurm_logs/%x-%j.out"

mkdir -p "$ROOT/slurm_logs"


# ═══════════════════════════════════════════════════════════════════════════
# 2. 判断：这一棒的节点是否还在空占预算内
#
#    第一棒是人手动提交的，所以门禁在这里；后面每一棒的门禁在
#    fleet_self_chain.sbatch 内部，因为那些提交发生在计算节点上，
#    任何包装脚本都不在它们的路径上。
# ═══════════════════════════════════════════════════════════════════════════

# shellcheck source=/dev/null
source "$BUDGET_LIB"
if ! node_budget_ok "$nodes"; then
    echo "REFUSED: $(node_budget_explain "$nodes")" >&2
    echo "空占节点已达上限。要么等在跑的占位作业结束，要么调高 U6GB_NODE_LIMIT。" >&2
    exit 1
fi


# ═══════════════════════════════════════════════════════════════════════════
# 3. 命令
#
#    --chain 与否只改变载荷脚本的一个参数，其余完全一致，方便 diff 对照。
# ═══════════════════════════════════════════════════════════════════════════

chain_arg=()
if (( CHAIN_ENABLED == 1 )); then
    chain_arg=(--chain)
    echo "[submit_self_chain] chain mode: 每一棒都会安排下一棒"
else
    echo "[submit_self_chain] one-shot mode: 只提交这一棒（加 --chain 才成链）"
fi

exec "$ROOT/record_submission.py" -- \
  sbatch \
  --nodes="$nodes" \
  --ntasks-per-node=1 \
  --gpus-per-node="$gpus_per_node" \
  --mem=0 \
  --time="$walltime" \
  --job-name="$job_name" \
  --output="$output_path" \
  --error="$output_path" \
  --export=ALL,U6GB_CHAIN_PREFIX="$prefix",U6GB_CHAIN_SEQ="$seq",U6GB_CHAIN_MAX="$max_generations",U6GB_CHAIN_WALLTIME="$walltime",U6GB_CHAIN_SLEEP_SECONDS="$sleep_seconds",U6GB_CHAIN_NODES="$nodes",U6GB_CHAIN_GPUS_PER_NODE="$gpus_per_node" \
  "$ROOT/fleet_self_chain.sbatch" ${chain_arg[@]+"${chain_arg[@]}"}
