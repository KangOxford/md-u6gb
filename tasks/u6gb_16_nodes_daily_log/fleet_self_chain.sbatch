#!/bin/bash
# Thin 16-node fleet payload. Resource shape is supplied by sbatch options.
# The script submits its afterany successor as soon as it starts, then idles.

set -euo pipefail

ROOT="/lus/lfs1aip2/projects/public/u6gb/tasks/u6gb_16_nodes_daily_log"
LOG_DIR="$ROOT/slurm_logs"
EVENTS_PATH="${U6GB_CHAIN_EVENTS_PATH:-$ROOT/events.jsonl}"
mkdir -p "$LOG_DIR"

: "${U6GB_CHAIN_PREFIX:=u6gb-16-nodes-18-jluy-chain}"
: "${U6GB_CHAIN_SEQ:=1}"
: "${U6GB_CHAIN_MAX:=0}"
: "${U6GB_CHAIN_WALLTIME:=23:59:00}"
: "${U6GB_CHAIN_SLEEP_SECONDS:=86100}"
: "${U6GB_CHAIN_NODES:=16}"
: "${U6GB_CHAIN_GPUS_PER_NODE:=4}"

now_utc() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

append_event() {
  local event_type="$1"
  shift
  printf '{"ts":"%s","event_type":"%s","job_id":"%s","job_name":"%s","chain_seq":"%s"%s}\n' \
    "$(now_utc)" \
    "$event_type" \
    "${SLURM_JOB_ID:-unknown}" \
    "${SLURM_JOB_NAME:-unknown}" \
    "$U6GB_CHAIN_SEQ" \
    "$*" >> "$EVENTS_PATH"
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "SLURM_JOB_ID is not set; this script must run inside sbatch" >&2
  exit 2
fi

next_seq=$((U6GB_CHAIN_SEQ + 1))
submit_next=1
if [[ "$U6GB_CHAIN_MAX" != "0" && "$next_seq" -gt "$U6GB_CHAIN_MAX" ]]; then
  submit_next=0
fi

append_event "self_chain_started" \
  ',"walltime":"'"$U6GB_CHAIN_WALLTIME"'"' \
  ',"sleep_seconds":"'"$U6GB_CHAIN_SLEEP_SECONDS"'"'

if [[ "$submit_next" == "1" ]]; then
  next_name="$(printf '%s-%03d' "$U6GB_CHAIN_PREFIX" "$next_seq")"
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
      --export=ALL,U6GB_CHAIN_PREFIX="$U6GB_CHAIN_PREFIX",U6GB_CHAIN_SEQ="$next_seq",U6GB_CHAIN_MAX="$U6GB_CHAIN_MAX",U6GB_CHAIN_WALLTIME="$U6GB_CHAIN_WALLTIME",U6GB_CHAIN_SLEEP_SECONDS="$U6GB_CHAIN_SLEEP_SECONDS",U6GB_CHAIN_NODES="$U6GB_CHAIN_NODES",U6GB_CHAIN_GPUS_PER_NODE="$U6GB_CHAIN_GPUS_PER_NODE",U6GB_CHAIN_EVENTS_PATH="$EVENTS_PATH" \
      "$ROOT/fleet_self_chain.sbatch"
  )"
  next_job_id="$(printf '%s\n' "$submit_output" | awk '/Submitted batch job/ {print $4}')"
  append_event "self_chain_successor_submitted" \
    ',"successor_job_id":"'"$next_job_id"'"' \
    ',"successor_job_name":"'"$next_name"'"' \
    ',"dependency":"afterany:'"$SLURM_JOB_ID"'"'
  printf '%s\n' "$submit_output"
else
  append_event "self_chain_successor_skipped" ',"reason":"max_generation_reached"'
fi

sleep "$U6GB_CHAIN_SLEEP_SECONDS"
append_event "self_chain_finished"
