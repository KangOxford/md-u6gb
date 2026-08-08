#!/bin/bash
# Stage 2A as the corrector: the trunk is FROZEN, only the residual was trained.
#
# The whole negative result so far may be a consequence of which checkpoint was
# used. Stage 2B unfroze the trunk and paid 3.9x of the autoregressive ability
# for it -- causal cross entropy 0.6827 pretrained against 2.6723 for the 2B
# trunk. Correction leans on exactly that ability, so a corrector built on a
# damaged trunk is being asked to do the thing it lost.
#
# Stage 2A never touches the trunk. Its state is 12 MB of residual against 2B's
# 911 MB of trunk-plus-optimiser, and the pretrained weights are restored
# underneath unchanged. It is also far more converged: long_NVDA ran to step
# 3500 where the 2B arms stopped at 300.
#
# If DFM post-training can reduce compound error at all in this setup, this is
# the configuration where it should show.
set -u
S0=/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/dfm-post-training-20260801
D=/lus/lfs1aip2/projects/public/u6gb/tasks/dfm_compound_error_20260808T001752Z
OUT=$S0/post_training/dfm/artifacts/rollouts
ST=$S0/post_training/dfm/artifacts/stage2a
IDX=$D/sweep/GOOG_jan2026_daystratified.txt

go () {   # node gpu tag state flags...
  local node=$1 gpu=$2 tag=$3 state=$4; shift 4
  [ -f "$OUT/dfm_$tag.npz" ] && { echo "skip $tag"; return; }
  srun --jobid=5944477 --overlap -w "$node" -N1 -n1 --cpu-bind=none \
    --export=ALL,DFM_GPU=$gpu,DFM_SCRIPT=dfm_correct_runner.py,XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
    bash $S0/post_training/dfm/eval/run_eval_node.sh \
    --n-seq 128 --batch-size 4 --n-cond 250 --n-gen 250 --gate-batches 2 \
    --wide-book --wide-levels 500 --indices-file "$IDX" --state "$state" "$@" \
    --out "$OUT/dfm_$tag.npz" > "$D/sweep/$tag.log" 2>&1 &
  echo "launched $tag on $node GPU$gpu"
}

# long_NVDA is the most converged 2A residual available at step 3500.
go nid011313 0 s2a_refresh_t080_n128 $ST/long_NVDA_state.msgpack --n-steps 8 --t-start 0.80 --book-refresh
go nid011313 1 s2a_refresh_t070_n128 $ST/long_NVDA_state.msgpack --n-steps 8 --t-start 0.70 --book-refresh
go nid011313 2 s2a_frozen_t080_n128  $ST/long_NVDA_state.msgpack --n-steps 8 --t-start 0.80
# Attribution for whatever 2A gives.
go nid011264 0 s2a_a2_t080_n128      $ST/long_NVDA_state.msgpack --n-steps 8 --t-start 0.80 --book-refresh --random-p --random-p-seed 7
# r2_cell06 is the 600-step grid cell that stage 2B was seeded from, so it is
# the like-for-like 2A against the 2B arms already measured.
go nid011264 2 s2a_cell06_t080_n128  $ST/r2_cell06_state.msgpack --n-steps 8 --t-start 0.80 --book-refresh
echo "launched at $(date -u +%H:%M:%SZ); waiting"; wait
echo "STAGE-2A ARMS COMPLETE at $(date -u +%H:%M:%SZ)"
