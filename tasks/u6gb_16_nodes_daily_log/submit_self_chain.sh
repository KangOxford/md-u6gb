#!/bin/bash
# Submit the first job in the 16-node self-chain fleet.

set -euo pipefail

ROOT="/lus/lfs1aip2/projects/public/u6gb/tasks/u6gb_16_nodes_daily_log"

prefix="${1:-u6gb-16-nodes-18-jluy-chain}"
seq="${2:-1}"
walltime="${3:-23:59:00}"
sleep_seconds="${4:-86100}"
max_generations="${5:-0}"
nodes="${U6GB_CHAIN_NODES:-16}"
gpus_per_node="${U6GB_CHAIN_GPUS_PER_NODE:-4}"

job_name="$(printf '%s-%03d' "$prefix" "$seq")"
output_path="$ROOT/slurm_logs/%x-%j.out"

mkdir -p "$ROOT/slurm_logs"

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
  "$ROOT/fleet_self_chain.sbatch"
