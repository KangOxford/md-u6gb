#!/bin/bash
# The decisive arms, on a DAY-STRATIFIED index.
#
# The n=64 arms all landed on 2026-01-02, because the frozen index file is
# ordered so a contiguous slice sits inside one session. Sixty-four windows from
# one morning are nowhere near sixty-four independent samples: they share a
# regime, a spread level and an activity level, and a day-block bootstrap over
# them has exactly one block and returns a zero-width interval. Any claim that
# post-training is better has to survive ACROSS days, not within one.
#
# n=128 over 20 interleaved days gives six or seven windows per session, which
# is enough for the block bootstrap to have something to resample.
set -u
S0=/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/dfm-post-training-20260801
D=/lus/lfs1aip2/projects/public/u6gb/tasks/dfm_compound_error_20260808T001752Z
OUT=$S0/post_training/dfm/artifacts/rollouts
IDX=$D/sweep/GOOG_jan2026_daystratified.txt
JOB=${JOB:-5944477}

go () {   # node gpu tag flags...
  local node=$1 gpu=$2 tag=$3; shift 3
  [ -f "$OUT/dfm_$tag.npz" ] && { echo "skip $tag"; return; }
  srun --jobid=$JOB --overlap -w "$node" -N1 -n1 --cpu-bind=none \
    --export=ALL,DFM_GPU=$gpu,DFM_SCRIPT=dfm_correct_runner.py,XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 \
    bash $S0/post_training/dfm/eval/run_eval_node.sh \
    --n-seq 128 --batch-size 4 --n-cond 250 --n-gen 250 --gate-batches 2 \
    --wide-book --wide-levels 500 --indices-file "$IDX" "$@" \
    --out "$OUT/dfm_$tag.npz" > "$D/sweep/$tag.log" 2>&1 &
  echo "launched $tag on $node GPU$gpu"
}

# t_start 0.80 was the only setting whose depth beat its own draft; 0.70 was the
# only one whose mid did. Both get a paired frozen control and 0.80 gets the
# random-P arm, since that is the configuration a positive claim would rest on.
go nid011240 0 refresh_t080_strat_n128    --n-steps 8 --t-start 0.80 --book-refresh
go nid011240 1 frozen_t080_strat_n128     --n-steps 8 --t-start 0.80
go nid011240 2 a2_refresh_t080_strat_n128 --n-steps 8 --t-start 0.80 --book-refresh --random-p --random-p-seed 7
go nid011240 3 refresh_t070_strat_n128    --n-steps 8 --t-start 0.70 --book-refresh
go nid011264 0 frozen_t070_strat_n128     --n-steps 8 --t-start 0.70
go nid011264 1 refresh_t090_strat_n128    --n-steps 8 --t-start 0.90 --book-refresh
echo "launched at $(date -u +%H:%M:%SZ); waiting"
wait
echo "STRATIFIED ARMS COMPLETE at $(date -u +%H:%M:%SZ)"
