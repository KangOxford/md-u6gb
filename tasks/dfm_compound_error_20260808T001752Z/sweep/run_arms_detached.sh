#!/bin/bash
# Launch every corrector arm on the attached allocation, one per free GPU.
#
# Why this file exists rather than a chain of srun calls from the agent's shell:
# an srun job step dies with the process that launched it. The previous attempt
# ran twelve arms from a backgrounded shell, the session restarted, and all
# twelve were killed mid-flight with 0 to 4 of 16 batches done and no output.
# Detaching the driver with setsid gives the steps a parent that outlives any
# one session, which is what makes "attach to a running allocation" workable at
# all -- an sbatch would get its lifetime from Slurm, an attached step gets it
# from whoever called srun.
#
# This is a compute driver: it launches work, waits, and exits. It is not a
# resident agent and it does not restart anything.
set -u
S0=/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/dfm-post-training-20260801
D=/lus/lfs1aip2/projects/public/u6gb/tasks/dfm_compound_error_20260808T001752Z
OUT=$S0/post_training/dfm/artifacts/rollouts
JOB=${JOB:-5944477}
mkdir -p "$D/sweep"

go () {   # node gpu tag  [flags...]
  local node=$1 gpu=$2 tag=$3; shift 3
  # Skip anything already finished, so a re-run of this driver resumes rather
  # than redoing hours of compute.
  if [ -f "$OUT/dfm_$tag.npz" ]; then echo "skip $tag (exists)"; return; fi
  srun --jobid=$JOB --overlap -w "$node" -N1 -n1 --cpu-bind=none \
    --export=ALL,DFM_GPU=$gpu,DFM_SCRIPT=dfm_correct_runner.py,XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 \
    bash $S0/post_training/dfm/eval/run_eval_node.sh \
    --n-seq 64 --batch-size 4 --n-cond 250 --n-gen 250 --gate-batches 2 \
    --wide-book --wide-levels 500 "$@" \
    --out "$OUT/dfm_$tag.npz" > "$D/sweep/$tag.log" 2>&1 &
  echo "launched $tag on $node GPU$gpu (pid $!)"
}

# The decisive pair first, on the least contended node, so that if anything
# goes wrong the refresh-versus-frozen comparison still exists.
go nid011240 0 refresh_t070_n64      --n-steps 8  --t-start 0.70 --book-refresh
go nid011240 1 frozen_t070_n64       --n-steps 8  --t-start 0.70
go nid011240 2 a2_refresh_t070_n64   --n-steps 8  --t-start 0.70 --book-refresh --random-p --random-p-seed 7
go nid011240 3 refresh_t085_n64      --n-steps 8  --t-start 0.85 --book-refresh
go nid011264 0 frozen_t085_n64       --n-steps 8  --t-start 0.85
go nid011264 1 refresh_t095_n64      --n-steps 8  --t-start 0.95 --book-refresh
go nid011264 2 refresh_t050_n64      --n-steps 8  --t-start 0.50 --book-refresh
go nid011264 3 refresh_t060_n64      --n-steps 8  --t-start 0.60 --book-refresh
go nid011312 0 refresh_t080_n64      --n-steps 8  --t-start 0.80 --book-refresh
go nid011312 1 refresh_t090_n64      --n-steps 8  --t-start 0.90 --book-refresh
go nid011312 2 refresh_t070_N16_n64  --n-steps 16 --t-start 0.70 --book-refresh
go nid011312 3 refresh_t070_N32_n64  --n-steps 32 --t-start 0.70 --book-refresh

echo "all launched at $(date -u +%Y-%m-%dT%H:%M:%SZ); waiting"
wait
echo "ALL ARMS COMPLETE at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
ls -la "$OUT"/dfm_*n64*.npz 2>/dev/null
