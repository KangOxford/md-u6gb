#!/bin/bash
# B: early-vs-late fixed-budget adaptation pair on the 2024-08 probe slice.
# Usage: submit_adaptation_pair.sh <RESTORE_STEP> <SHORTNAME> [JAX_SEED]
#   e.g. submit_adaptation_pair.sh 275 e275
#        submit_adaptation_pair.sh 69378 l69378
# Both members share every setting (data, seed, schedule, budget); the only
# difference is which checkpoint age they restore from. RESTORE_RESET_SCHEDULE
# makes both walk the same fresh LR trajectory.
set -euo pipefail

STEP="${1:?RESTORE_STEP required (275 or 69378)}"
# S2: see attach_adaptation.sh. The header's claim that both members share a seed was
# unbacked by any export; it is an argument now.
JAX_SEED="${3:-42}"
export JAX_SEED
SHORT="${2:?short name required (e.g. e275)}"

SIGMA0=/lus/lfs1aip2/projects/public/u6gb/sigma-0
YAML="$SIGMA0/configs/train/dfm_smoke_1gpu.yaml"

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
export QUANT_ROOT=/lus/lfs1aip2/projects/public/s5e/quant_team/quant
export PYTHONPATH=/lus/lfs1aip2/projects/public/u6gb/openreview-v2
export SQUASHFS_MULTI_MODE=1
export SQUASHFS_DIR=/lus/lfs1aip2/projects/public/s5e/quant_team/lob_preproc_sp500_squashfs
export SQUASHFS_MONTHS=2024-08
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
export CURTAIL_EPOCHS=1500
export LOG_GRAD_NORMS=1
export RESTORE_PATH=$SIGMA0/checkpoints_selftrain/j5705912_b30675li_5705912
export RESTORE_STEP=$STEP
export RESTORE_RESET_SCHEDULE=True

cd "$SIGMA0"
sbatch --nodes=1 --gres=gpu:1 --mem=115000M --cpus-per-task=72 \
    --time=03:30:00 --job-name="cl-adapt-$SHORT" \
    run/base_model/train_full_autoreg.batch
