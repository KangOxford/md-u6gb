#!/bin/bash
#SBATCH --job-name=u6gb-16-log
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --dependency=singleton
#SBATCH --output=/lus/lfs1aip2/projects/public/u6gb/tasks/u6gb_16_nodes_daily_log/slurm_logs/%j.out

set -euo pipefail

ROOT="/lus/lfs1aip2/projects/public/u6gb/tasks/u6gb_16_nodes_daily_log"
NEXT_RUN="$(date -u -d 'tomorrow 00:15' +%Y-%m-%dT%H:%M:%S)"

mkdir -p "$ROOT/slurm_logs"

# Schedule the next logger before running today's analysis so one agent failure
# does not silently stop the daily chain.
sbatch --begin="$NEXT_RUN" "$ROOT/daily_agent.sbatch"

"$ROOT/run_daily_agent.sh" "$(date -u -d yesterday +%F)"
