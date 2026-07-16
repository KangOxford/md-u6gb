#!/bin/bash
#SBATCH --job-name=u6gb-128nodes-status-1min
#SBATCH --nodes=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:01:00
#SBATCH --output=logs/128nodes_node_status/%x-%j.out
#SBATCH --error=logs/128nodes_node_status/%x-%j.err

set -euo pipefail

echo "job_id=${SLURM_JOB_ID}"
echo "job_name=${SLURM_JOB_NAME}"
echo "submit_host=${SLURM_SUBMIT_HOST:-unknown}"
echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "nodes=${SLURM_JOB_NUM_NODES}"
echo "nodelist=${SLURM_JOB_NODELIST}"
echo "account=${SLURM_JOB_ACCOUNT:-unknown}"
echo "partition=${SLURM_JOB_PARTITION:-unknown}"

echo "resolved_nodes_begin"
scontrol show hostnames "${SLURM_JOB_NODELIST}"
echo "resolved_nodes_end"

echo "per_node_status_begin"
srun --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 --cpu-bind=none bash -lc '
  set -euo pipefail
  printf "node=%s utc=%s " "$(hostname)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf "uptime=%s " "$(cut -d " " -f1 /proc/uptime)"
  printf "loadavg=%s,%s,%s " "$(cut -d " " -f1 /proc/loadavg)" "$(cut -d " " -f2 /proc/loadavg)" "$(cut -d " " -f3 /proc/loadavg)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    printf "gpus="
    nvidia-smi --query-gpu=index,name,uuid,temperature.gpu,memory.used,memory.total --format=csv,noheader,nounits | paste -sd "|" -
  else
    printf "gpus=nvidia-smi-not-found\n"
  fi
'
echo "per_node_status_end"
echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
