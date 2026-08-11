#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 JOB_ID NODE" >&2
  exit 2
fi
job_id="$1"
node="$2"
task_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
current_user="$(id -un)"
job_record="$(squeue -h -j "$job_id" -o '%u|%T|%N')"

if [[ -z "$job_record" ]]; then
  echo "Job $job_id is absent from squeue" >&2
  exit 3
fi
IFS='|' read -r owner state nodelist <<<"$job_record"
if [[ "$owner" != "$current_user" || "$state" != "RUNNING" ]]; then
  echo "Job gate failed: owner=$owner state=$state expected_owner=$current_user" >&2
  exit 3
fi
if ! scontrol show hostnames "$nodelist" | rg -F -x -q "$node"; then
  echo "Node $node is not in job $job_id ($nodelist)" >&2
  exit 3
fi
job_detail="$(scontrol show job "$job_id" -o)"
end_time="${job_detail#* EndTime=}"
end_time="${end_time%% *}"
remaining_seconds="$(( $(date -d "$end_time" +%s) - $(date +%s) ))"
if (( remaining_seconds < 1200 )); then
  echo "Refusing a model download with less than 20 minutes remaining ($remaining_seconds seconds)" >&2
  exit 3
fi
if [[ ! -x "$task_dir/.venv/bin/python" ]]; then
  echo "Run prepare_runtime.sh first" >&2
  exit 4
fi

mkdir -p "$task_dir/hf_home" "$task_dir/model" "$task_dir/runs"
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_path="$task_dir/runs/prepare_model_${run_stamp}.log"
srun \
  --jobid="$job_id" \
  --overlap \
  --exact \
  --nodes=1 \
  --nodelist="$node" \
  --ntasks=1 \
  --cpus-per-task=4 \
  --mem=48G \
  /usr/bin/env \
    HF_HOME="$task_dir/hf_home" \
    "$task_dir/.venv/bin/python" "$task_dir/prepare_model.py" \
      --model-dir "$task_dir/model" \
      --manifest "$task_dir/model/download_manifest.json" \
  2>&1 | tee "$log_path"
