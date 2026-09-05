#!/usr/bin/env bash
# One cell of the interleaved batch, generated and scored on node-local disk.
#
# The confound this removes: the two arms' rollouts were generated 18 days apart
# on disjoint node sets, while every "null" used to bound that pairs cells minutes
# apart in one warm allocation. The null therefore bounds a much narrower
# perturbation than the comparison applies. Running both arms back-to-back on the
# same card, in an order fixed by a recorded seed, turns the batch effect from a
# confound into randomised noise.
#
# Only a small JSON returns to Lustre; the project is at 99.4% of its inode quota
# and the raw rollouts have no value once the ratio is read off them.
#
# The cell records where and when it ran alongside the numbers.  Randomising the
# arm order per card only converts the batch effect into noise if that noise can
# be *shown* to be noise, and the registered placebo -- regressing the ratio on
# node and on start time -- needs those fields to exist in the result.
set -uo pipefail
S=/lus/lfs1aip2/projects/public/u6gb/nb_build_pr22
T=/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z
W=/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/crps-return-alignment-20260808
STOCK="${STOCK:?}"; SEED="${SEED:?}"; ARM="${ARM:?}"   # ARM = r3 | r4
case "$ARM" in
  r3) CK="$T/ckpt/wm_ft_multi3" ;;
  r4) CK="$T/ckpt/wm_ft_multi4" ;;
  *)  echo "[arm] unknown $ARM" >&2; exit 6 ;;
esac
KEYS=arm,ticker,seed,qL1,sd_ratio
OUTJSON="$S/inter/${ARM}_${STOCK}_s${SEED}.json"
CELL_LOG="${CELL_LOG:-$S/logs/inter_${STOCK}_${SEED}.log}"
mkdir -p "$S/inter" "$S/logs"

# "Already done" is a property of the content, not of the inode.  The old guard
# tested [ -f ], and the script itself created the file empty with `>` before
# awk wrote anything, so a cell killed during scoring left a zero-byte file that
# the guard then read as done -- the cell could never retry itself.
if python3 "$S/cell_json.py" valid "$OUTJSON" --require "$KEYS"; then
  echo "[skip] $ARM $STOCK $SEED (valid result already present)"; exit 0
fi
if [ -e "$OUTJSON" ]; then
  echo "[retry] $ARM $STOCK $SEED: existing result is not valid JSON, regenerating" >&2
fi

T_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
NODE=$(hostname -s)
GPU="${CUDA_VISIBLE_DEVICES:-unset}"
GPU_UUID=$(command -v nvidia-smi >/dev/null 2>&1 \
           && nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | head -1 || true)
echo "[cell] $ARM $STOCK s$SEED node=$NODE gpu=$GPU job=${SLURM_JOB_ID:-none} start=$T_START"

LOCAL="/local/user/$(id -u)/inter_$$_${ARM}_${STOCK}_${SEED}"
mkdir -p "$LOCAL/data"
export SIGMA0_ROOT="$W" LAUNCHER="$W"
export CKPT="$CK" CKPT_STEP=69378
export OUT_ROOT="$LOCAL/data/cell" STOCK
export INDICES="$T/data/v5m_eval_idx_${STOCK}.txt"
export N_SEQ=500 BATCH=48 N_COND=250 N_GEN=250
export K=1 SEED0="$SEED" SEED_STRIDE=1
export INFERENCE_MONTHS=2026-01 WIDE_LEVELS=500
# Success is the product, not the exit code -- see r3null_cell.sh for the same repair.
_rc=0; bash "$W/run/mid_training/collect_rollouts.sh" || _rc=$?
if [ ! -s "$OUT_ROOT/member_0/.done" ] || \
   ! python3 -c 'import json,sys; json.load(open(sys.argv[1]))' \
             "$OUT_ROOT/member_0/.done" 2>/dev/null; then
    echo "[gen] produced no complete member (collect exit=$_rc)" >&2; exit 7
fi
[ "$_rc" -ne 0 ] && echo "[gen] collect exited $_rc but member_0 is complete; keeping it" >&2

cd "$W/run/mid_training"
# The scorer's stderr used to go to /dev/null, which threw away every [fatal]
# line -- the one place it says WHY it gave up.  It goes to the cell log now.
SCORE_OUT="$LOCAL/score_stdout.txt"
# --no-crps --assert-k 1: this cell generates ONE member, so fair CRPS has no
# pair of distinct members and is not defined.  Saying so is now required --
# the scorer stops on a K<2 CRPS request rather than printing "undefined(K=1)",
# which is how 32 cells came to be recorded as unparseable files asserting K=4.
MIDTRAIN_TASK_ROOT="$LOCAL" python3 score_v5_primary.py --task "$LOCAL" \
    --arms a=cell b=cell --baseline a --no-crps --assert-k 1 \
    --n-boot 100 --min-overlap 0 \
    --out "$LOCAL/score.json" > "$SCORE_OUT" 2>> "$CELL_LOG"
SCORE_RC=$?
if [ "$SCORE_RC" -ne 0 ]; then
  echo "[score] scorer exited $SCORE_RC for $ARM $STOCK $SEED; stdout follows" >&2
  sed -n '1,60p' "$SCORE_OUT" >&2
  exit 9
fi

T_END=$(date -u +%Y-%m-%dT%H:%M:%SZ)
# `[a]` alone matched two lines: the scorer prints an `[a] K=... requested`
# census line long before it prints `[a] CRPS ... qL1 ... sd_ratio ...`, so the
# old selector emitted a first record whose qL1 was the literal word `per-seed`
# and whose sd_ratio was the context count.  Anchor on the metric line itself.
awk -v a="$ARM" -v tk="$STOCK" -v sd="$SEED" '/^\[a\] CRPS/{
      printf "{\"arm\":\"%s\",\"ticker\":\"%s\",\"seed\":%s,\"qL1\":%s,\"sd_ratio\":%s}\n", a, tk, sd, $(NF-2), $NF}' \
    "$SCORE_OUT" \
  | python3 "$S/cell_json.py" finalize --out "$OUTJSON" --require "$KEYS" \
      --add node="$NODE" --add gpu="$GPU" --add gpu_uuid="${GPU_UUID:-unknown}" \
      --add jobid="${SLURM_JOB_ID:-unknown}" --add stepid="${SLURM_STEP_ID:-unknown}" \
      --add t_start="$T_START" --add t_end="$T_END"
FIN_RC=$?
if [ "$FIN_RC" -ne 0 ]; then
  echo "[score] no valid result for $ARM $STOCK $SEED (finalize rc=$FIN_RC); scorer stdout:" >&2
  sed -n '1,60p' "$SCORE_OUT" >&2
  exit 8
fi
mv "$LOCAL" "${LOCAL}_done" 2>/dev/null || true
