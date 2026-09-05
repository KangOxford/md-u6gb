#!/bin/bash
# B (attach form): early-vs-late fixed-budget adaptation on the 2024-08 slice,
# attached into an idle node of an existing allocation (no queueing).
# Usage: attach_adaptation.sh <RESTORE_STEP> <SHORT> [ALLOC] [NODE]
#   attach_adaptation.sh 275   e275   6141106 nid010252
#   attach_adaptation.sh 69378 l69378 6141106 nid010252
# Gate first (memory <100MiB, 0 compute PIDs) before calling this.
set -euo pipefail

STEP="${1:?RESTORE_STEP required}"
SHORT="${2:?short name required}"
ALLOC="${3:-6141106}"
NODE="${4:-nid010252}"

SIGMA0=/lus/lfs1aip2/projects/public/u6gb/sigma-0
YAML="$SIGMA0/configs/train/dfm_smoke_1gpu.yaml"
RUNTAG="cl-adapt-$SHORT"
LOGDIR="$SIGMA0/logs_cl_probe/$SHORT"
mkdir -p "$LOGDIR"

export GPUS_PER_NODE=1
export MODEL_PRESET=75m
export SSM_TYPE=mamba3
export TOKEN_MODE=26tok
export OPT_CONFIG=muon
export MUON_LR=0.01
export PER_GPU_BSZ=4
export EPOCHS=1
export CHECKPOINT_EVERY=auto
export MAX_JOB_HOURS=3.0
export NO_AUTO_RESUME=1
export NO_AUTO_RESUME_DEPTH=99
export QUANT_ROOT=/lus/lfs1aip2/projects/public/s5e/quant_team/quant
export PYTHONPATH=/lus/lfs1aip2/projects/public/u6gb/openreview-v2
export SQUASHFS_MULTI_MODE=1
export SQUASHFS_DIR=/lus/lfs1aip2/projects/public/s5e/quant_team/lob_preproc_sp500_squashfs
export SQUASHFS_MONTHS=2024-08
# Unique node-local mount root per run: avoids the stale dead mount left under
# the default ${SLURM_JOB_ID}-derived path, and keeps the pair from colliding.
export SQUASHFS_MULTI_MOUNT_ROOT=/tmp/kangli.u6gb/sigma0/cl_probe_${SHORT}/sp500_squashfs
export FORBID_RAW_NPYZST=1
export DATA_ROOT=/lus/lfs1aip2/projects/s5e/lob_preproc_sp500
TICKERS=$(grep '^env_TICKERS:' "$YAML" | sed 's/^env_TICKERS: //')
export TICKERS
export TRAIN_DATE_RANGE=2024-08-01,2024-08-31
export TEST_DATE_RANGE=
export N_DATA_WORKERS=12
export CHECKPOINT_BASE_DIR=$SIGMA0/checkpoints_cl_probe
export WANDB_PROJECT=sigma0-continual
export WANDB_ENTITY=oxford-lob
export WANDB_MODE=online
export USE_WANDB=True
export CURTAIL_EPOCHS=${CURTAIL_OVERRIDE:-1500}
export MAX_JOB_HOURS=${MAX_HOURS_OVERRIDE:-3.0}
export LOG_GRAD_NORMS=1
export RESTORE_PATH=$SIGMA0/checkpoints_selftrain/j5705912_b30675li_5705912
export RESTORE_STEP=$STEP
export RESTORE_RESET_SCHEDULE=True
export NODE_LOG_DIR="$LOGDIR"

# Fake the allocation environment for the batch script.
for v in $(env | grep -o '^SLURM_[A-Z_]*' ); do unset "$v"; done
export SLURM_JOB_ID=$ALLOC
export SLURM_NNODES=1
export SLURM_JOB_NODELIST=$NODE
export SLURM_SUBMIT_DIR=$SIGMA0

# Rewrite the internal srun into an attached, single-GPU, named step.
GEN="$LOGDIR/batch_${RUNTAG}.sh"
sed "s|^srun --nodes=\$NNODES \\\\|srun --jobid=$ALLOC --overlap -w $NODE --exact --cpus-per-task=64 --job-name=$RUNTAG --nodes=\$NNODES \\\\|" \
    "$SIGMA0/run/base_model/train_full_autoreg.batch" > "$GEN"
if ! grep -q -- "--jobid=$ALLOC" "$GEN"; then
    echo "FATAL: srun rewrite did not take" >&2
    exit 5
fi

cd "$SIGMA0"
setsid nohup bash "$GEN" > "$LOGDIR/${RUNTAG}.out" 2>&1 &
echo "launched $RUNTAG pid=$! log=$LOGDIR/${RUNTAG}.out"
