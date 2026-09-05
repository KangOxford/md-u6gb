#!/usr/bin/env bash
# A replication null for the round-2 -> round-3 gain.
#
# That gain is the claim the whole line rests on, and it was measured today on
# fair_crps for the first time. It has no noise floor: nothing has ever
# regenerated round 3 at the 97701-97704 seeds the gain was computed from. This
# supplies one, on node-local disk so it costs no Lustre inodes.
#
# Node, card, allocation and wall-clock go into the result next to the numbers:
# a null that cannot be regressed on where and when it ran cannot rule out the
# batch effect it exists to bound.
set -uo pipefail
S=/lus/lfs1aip2/projects/public/u6gb/nb_build_pr22
T=/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z
W=/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/crps-return-alignment-20260808
STOCK="${STOCK:?}"; SEED="${SEED:?}"
KEYS=ticker,seed,qL1,sd_ratio
OUTJSON="$S/r3null/r3null_${STOCK}_s${SEED}.json"
CELL_LOG="${CELL_LOG:-$S/logs/r3null_${STOCK}_${SEED}.log}"
mkdir -p "$S/r3null" "$S/logs"

# Skip on a valid result, not on a filename.  The old guard tested [ -f ] while
# the script created the file empty with `>` before awk ran, so a cell killed
# during scoring blocked its own retry for good.
if python3 "$S/cell_json.py" valid "$OUTJSON" --require "$KEYS"; then
  echo "[skip] $STOCK $SEED (valid result already present)"; exit 0
fi
[ -e "$OUTJSON" ] && echo "[retry] $STOCK $SEED: existing result invalid, regenerating" >&2

T_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
NODE=$(hostname -s)
GPU="${CUDA_VISIBLE_DEVICES:-unset}"
GPU_UUID=$(command -v nvidia-smi >/dev/null 2>&1 \
           && nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | head -1 || true)
echo "[cell] r3null $STOCK s$SEED node=$NODE gpu=$GPU job=${SLURM_JOB_ID:-none} start=$T_START"

LOCAL="/local/user/$(id -u)/r3null_$$_${STOCK}_${SEED}"; mkdir -p "$LOCAL/data"
export SIGMA0_ROOT="$W" LAUNCHER="$W"
export CKPT="$T/ckpt/wm_ft_multi3" CKPT_STEP=69378
export OUT_ROOT="$LOCAL/data/cell" STOCK
export INDICES="$T/data/v5m_eval_idx_${STOCK}.txt"
export N_SEQ=500 BATCH=48 N_COND=250 N_GEN=250
export K=1 SEED0="$SEED" SEED_STRIDE=1
export INFERENCE_MONTHS=2026-01 WIDE_LEVELS=500
# Success is the product, not the exit code.  collect_rollouts.sh has been observed
# returning nonzero with the member fully written, intermittently -- so `|| exit 7` threw
# away good cells and left a biased sample nobody notices.  Check what was produced.
_rc=0; bash "$W/run/mid_training/collect_rollouts.sh" || _rc=$?
if [ ! -s "$OUT_ROOT/member_0/.done" ] || \
   ! python3 -c 'import json,sys; json.load(open(sys.argv[1]))' \
             "$OUT_ROOT/member_0/.done" 2>/dev/null; then
    echo "[gen] produced no complete member (collect exit=$_rc)" >&2; exit 7
fi
[ "$_rc" -ne 0 ] && echo "[gen] collect exited $_rc but member_0 is complete; keeping it" >&2

cd "$W/run/mid_training"
# stderr used to be discarded, taking every [fatal] explanation with it.
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
  echo "[score] scorer exited $SCORE_RC for $STOCK $SEED; stdout follows" >&2
  sed -n '1,60p' "$SCORE_OUT" >&2
  exit 9
fi

T_END=$(date -u +%Y-%m-%dT%H:%M:%SZ)
# `/^\[a\]/` matched the `[a] K=... requested` census line as well as the metric
# line, so every file got a bogus first record (qL1 = the word `per-seed`).
awk -v tk="$STOCK" -v sd="$SEED" '/^\[a\] CRPS/{
      printf "{\"ticker\":\"%s\",\"seed\":%s,\"qL1\":%s,\"sd_ratio\":%s}\n", tk, sd, $(NF-2), $NF}' \
    "$SCORE_OUT" \
  | python3 "$S/cell_json.py" finalize --out "$OUTJSON" --require "$KEYS" \
      --add node="$NODE" --add gpu="$GPU" --add gpu_uuid="${GPU_UUID:-unknown}" \
      --add jobid="${SLURM_JOB_ID:-unknown}" --add stepid="${SLURM_STEP_ID:-unknown}" \
      --add t_start="$T_START" --add t_end="$T_END"
FIN_RC=$?
if [ "$FIN_RC" -ne 0 ]; then
  echo "[score] no valid result for $STOCK $SEED (finalize rc=$FIN_RC); scorer stdout:" >&2
  sed -n '1,60p' "$SCORE_OUT" >&2
  exit 8
fi
mv "$LOCAL" "${LOCAL}_done" 2>/dev/null || true
