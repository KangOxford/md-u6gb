#!/bin/bash
# B (attach form): early-vs-late fixed-budget adaptation on the 2024-08 slice,
# attached into an idle node of an existing allocation (no queueing).
# Usage: attach_adaptation.sh <RESTORE_STEP> <SHORT> [JAX_SEED] [ALLOC] [NODE]
#
# RESTORE_STEP -- which checkpoint of the selftrain chain to adapt from.
#
# The example in this header used to read 275, and 275 must not be used. Read from the run's
# own log, /lus/lfs1aip2/projects/public/u6gb/sigma-0/logs_lobs5/training_5705912_node0.log:
#
#     [Schedule] steps_per_epoch: 80805346
#     [Schedule] total_steps:     80805346
#     [Schedule] warmup_end_step:   808053
#
# so the run's LAST checkpoint, step 69378, is 0.0859% of one epoch and 8.59% of the way to
# the end of the linear warmup. EVERY checkpoint in the chain is inside the warmup:
#
#     step     275  ->  0.03% of the warmup ramp
#     step   33575  ->  4.16%
#     step   69378  ->  8.59%
#
# Two consequences the caller has to carry, not the script:
#   1. 275 is 0.03% up the ramp -- adapting from it is untrained-versus-trained, not
#      early-versus-late. Use 33575 as the early member; 69378 is the late one.
#   2. The interval 33575 -> 69378 spans a roughly 2x change in the learning rate the
#      checkpoints were produced under (4.16% -> 8.59% of peak). RESTORE_RESET_SCHEDULE=True
#      below gives both members the same fresh schedule at adaptation time, so the confound
#      is in what the checkpoints ARE, not in how they are adapted. It is unregistered in the
#      plan and must be reported with any early-versus-late reading.
#
# An earlier version of this header derived warmup_end_step ~= 693 by INFERRING
# steps_per_epoch from the last checkpoint number. That inference was wrong by three orders
# of magnitude. The figures above are read from the log, not inferred.
#   attach_adaptation.sh 33575 e33575 42   <alloc> <node>     # early member
#   attach_adaptation.sh 69378 l69378 42   <alloc> <node>     # late member
# Gate first (memory <100MiB, 0 compute PIDs) before calling this.
set -euo pipefail

STEP="${1:?RESTORE_STEP required -- 33575 (early) or 69378 (late); NOT 275, see the header}"
case "$STEP" in
  275) echo "REFUSED: step 275 is 0.03% up the linear warmup ramp (warmup_end_step 808053," \
            "read from logs_lobs5/training_5705912_node0.log). Adapting from it is an" \
            "untrained-versus-trained contrast, not early-versus-late. Use 33575." >&2
       exit 7 ;;
esac
SHORT="${2:?short name required}"
# S2: the seed is an ARGUMENT, not an assumption. Both this script and
# submit_adaptation_pair.sh carried a header claiming the two members "share every setting
# (data, seed, schedule, budget)" while exporting no seed at all -- so every run took the
# batch default and the >=5-seed replication PLAN section 3 Step 2 requires was unreachable,
# as was the same-age null pair (S3), which differs from a real pair ONLY in the seed.
JAX_SEED="${3:-42}"
export JAX_SEED
ALLOC="${4:-}"
NODE="${5:-}"

SIGMA0=/lus/lfs1aip2/projects/public/u6gb/sigma-0
YAML="$SIGMA0/configs/train/dfm_smoke_1gpu.yaml"
# the seed is in the run tag and the output path, so two seeds never share a directory
RUNTAG="cl-adapt-${SHORT}-s${JAX_SEED}"
# The project is at its inode hard cap (51,200,000/51,200,000), so a new directory on Lustre
# cannot be created at all -- mkdir fails, and a launcher that ignores that writes nothing
# while reporting success. Logs go node-local; retrieval is a separate, explicit step.
LOGDIR="${CL_LOGDIR:-/tmp/${USER}/sigma0/cl_probe_logs/${RUNTAG}}"
mkdir -p "$LOGDIR" || { echo "FATAL: cannot create $LOGDIR" >&2; exit 6; }

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
# node_wrapper.sh:342 blanks SQUASHFS_MULTI_MOUNT_ROOT unconditionally, so exporting it
# here never reached the code and both members of a pair collided on one mount root -- the
# defect that kept Step 2 from ever running (R1-F4). It cannot be fixed by editing
# node_wrapper.sh: 45 named steps belonging to other sessions are running against that file
# right now, and editing a script while it executes is how 14 workers once exited 127.
#
# The no-edit route: node_wrapper.sh:13 honours an inherited SIGMA0_JOB_TMPDIR and line 19
# exports it as TMPDIR, and line 370's mount-root default is
# "$TMPDIR/sp500_squashfs_${SLURM_JOB_ID}_${SLURM_PROCID}". A unique SIGMA0_JOB_TMPDIR
# therefore yields a unique mount root, through a path the blanking does not touch.
export SIGMA0_JOB_TMPDIR=/tmp/kangli.u6gb/sigma0/cl_probe_${SHORT}
export SQUASHFS_MULTI_MOUNT_ROOT=${SIGMA0_JOB_TMPDIR}/sp500_squashfs
export FORBID_RAW_NPYZST=1
export DATA_ROOT=/lus/lfs1aip2/projects/s5e/lob_preproc_sp500
# S5: the YAML asks for 488 tickers; the 2024-08 shard holds 482, all paired. Six of the
# requested names (BAC among them) are simply absent, and lobster_dataloader.py:375 asserts
# rather than skipping, so both members died at 51s inside dataset setup. The ticker set is
# now pinned to what the shard actually contains, read from one file both members share --
# a matched pair must train on identical tickers, so deriving the list per member would be
# a second bug even if each derivation succeeded.
TICKERS=$(cat /lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/results/tickers_2024-08.txt)
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
