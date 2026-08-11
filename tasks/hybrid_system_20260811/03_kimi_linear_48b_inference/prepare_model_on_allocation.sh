#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then echo "Usage: $0 JOB_ID NODE" >&2; exit 2; fi
job_id="$1"; node="$2"
task_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
current_user="$(id -un)"
job_record="$(squeue -h -j "$job_id" -o '%u|%T|%N')"
if [[ -z "$job_record" ]]; then echo "Job absent" >&2; exit 3; fi
IFS='|' read -r owner state nodelist <<<"$job_record"
if [[ "$owner" != "$current_user" || "$state" != "RUNNING" ]]; then echo "Job gate failed" >&2; exit 3; fi
if ! scontrol show hostnames "$nodelist" | rg -F -x -q "$node"; then echo "Node gate failed" >&2; exit 3; fi
job_detail="$(scontrol show job "$job_id" -o)"; end_time="${job_detail#* EndTime=}"; end_time="${end_time%% *}"
remaining_seconds="$(( $(date -d "$end_time" +%s) - $(date +%s) ))"
if (( remaining_seconds < 900 )); then echo "Less than 15 minutes remain" >&2; exit 3; fi
if [[ ! -x "$task_dir/.venv/bin/python" ]]; then echo "Prepare runtime first" >&2; exit 4; fi
mkdir -p "$task_dir/hf_home" "$task_dir/model" "$task_dir/runs"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"; log="$task_dir/runs/prepare_model_${stamp}.log"
srun --jobid="$job_id" --overlap --exact --nodes=1 --nodelist="$node" --ntasks=1 --cpus-per-task=8 --mem=64G \
  /usr/bin/env HF_HOME="$task_dir/hf_home" "$task_dir/.venv/bin/python" "$task_dir/prepare_model.py" \
  --model-dir "$task_dir/model" --manifest "$task_dir/model/download_manifest.json" 2>&1 | tee "$log"
