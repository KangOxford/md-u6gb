#!/bin/bash
# Launch the four-arm driver as a detached srun step on nid010434 (job
# 6022465). env -u scrubs the calling step's SLURM context so the nested srun
# does not inherit its CPU binding or node bookkeeping; setsid detaches the
# srun client from the calling session so the step survives it.
TASK=/home/u6gb/kangli.u6gb/pr16_doq_artifacts_20260816
exec env -u SLURM_JOB_ID -u SLURM_JOBID -u SLURM_STEP_ID -u SLURM_STEPID \
    -u SLURM_NODELIST -u SLURM_JOB_NODELIST -u SLURM_NNODES -u SLURM_JOB_NUM_NODES \
    -u SLURM_NTASKS -u SLURM_PROCID -u SLURM_LOCALID -u SLURM_TASKS_PER_NODE \
    -u SLURM_CPU_BIND -u SLURM_CPU_BIND_TYPE -u SLURM_CPU_BIND_LIST \
    -u SLURM_CPU_BIND_VERBOSE -u SLURM_CPUS_PER_TASK -u TMUX -u TMUX_PANE \
    setsid srun --jobid=6022465 -w nid010434 --overlap -N1 -n1 \
    --cpus-per-task=16 --mem=100G --time=02:00:00 --immediate=30 \
    --job-name=sigma0-doq-verify --cpu-bind=none \
    bash -lc "$TASK/code/run_four_arms.sh"
