#!/usr/bin/env bash
# One point on the step-budget curve, scored on the objective this line is named
# for rather than on two statistics that cannot see it.
#
# The previous version scored each generation alone -- one directory, so K=1 in
# score_v5_primary, so fair_crps returned NaN and the cell reported only sd_ratio
# and qL1. Both of those are provably invariant to context shuffling (0.0 and
# 2.2e-16): a generator that reproduced the unconditional return law and ignored
# its input entirely scores perfectly on both. K in the scorer is the number of
# DIRECTORIES in an arm, and load_arm reads member_0 of each, so two seeds must
# be two OUT_ROOTs joined at scoring time. Same two generations, CRPS defined.
set -uo pipefail
S=/lus/lfs1aip2/projects/public/u6gb/nb_build_pr22
T=/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z
Wo=/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/crps-return-alignment-20260808
STOCK="${STOCK:?}"; STEP="${STEP:?}"; ARM="${ARM:-multi4}"; RES="${RES:?}"
SEEDS="${SEEDS:-97901 97902 97903 97904}"
CK="$T/ckpt/wm_ft_${ARM}_step${STEP}"; [ "$STEP" = final ] && CK="$T/ckpt/wm_ft_${ARM}"
[ -d "$CK" ] || { echo "[ckpt] missing $CK" >&2; exit 6; }
grep -q "\"arm\":\"$ARM\",\"step\":\"$STEP\",\"ticker\":\"$STOCK\"" "$RES" 2>/dev/null && { echo "[skip]"; exit 0; }
ROOT="/local/user/$(id -u)/sc_$$_${ARM}_${STEP}_${STOCK}"; mkdir -p "$ROOT/data"
export SIGMA0_ROOT="$Wo" LAUNCHER="$Wo" CKPT="$CK" CKPT_STEP=69378 STOCK
export INDICES="$T/data/v5m_eval_idx_${STOCK}.txt"
export N_SEQ=500 BATCH=48 N_COND=250 N_GEN=250 K=1 SEED_STRIDE=1
export INFERENCE_MONTHS=2026-01 WIDE_LEVELS=500
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
ARMLIST=""
for sd in $SEEDS; do
    export OUT_ROOT="$ROOT/data/s$sd" SEED0="$sd"
    bash "$Wo/run/mid_training/collect_rollouts.sh" >> "$ROOT/gen.log" 2>&1
    _rc=$?
    # Success is the product, not the exit code. collect_rollouts.sh has been
    # observed returning nonzero with the member fully written -- 4 files and
    # its .done marker on disk -- and the old `|| exit 7` threw that away. It
    # did so intermittently, which is worse than always: the surviving cells
    # are then a biased sample nobody notices. So check what was produced, and
    # report a nonzero code as the anomaly it is rather than as the verdict.
    # `-f` accepted a 0-byte .done, and a 0-byte .done is what a redirection that
    # succeeded followed by a writer that died leaves behind -- an allocation expiring
    # mid-write produces exactly that.  Such a member would then be scored as complete.
    # Existence is not integrity: require content AND a parsable manifest.
    if [ ! -s "$OUT_ROOT/member_0/.done" ] || \
       ! python3 -c 'import json,sys; json.load(open(sys.argv[1]))' \
                 "$OUT_ROOT/member_0/.done" 2>/dev/null; then
        echo "[gen] seed $sd produced no complete member (collect exit=$_rc)" >&2
        tail -6 "$ROOT/gen.log" >&2
        exit 7
    fi
    [ "$_rc" -ne 0 ] && echo "[gen] seed $sd: collect exited $_rc but member_0 is complete; keeping it" >&2
    ARMLIST="${ARMLIST:+$ARMLIST,}s$sd"
done
cd "$Wo/run/mid_training"
LINE=$(MIDTRAIN_TASK_ROOT="$ROOT" python3 score_v5_primary.py --task "$ROOT" \
    --arms a="$ARMLIST" b="$ARMLIST" --baseline a --n-boot 60 --min-overlap 0 \
    --out "$ROOT/score.json" 2>>"$ROOT/gen.log" \
  | awk -v a="$ARM" -v st="$STEP" -v tk="$STOCK" -v nd="$(hostname)" '/^\[a\] CRPS/{
      printf "{\"arm\":\"%s\",\"step\":\"%s\",\"ticker\":\"%s\",\"node\":\"%s\",\"crps\":\"%s\",\"qL1\":%s,\"sd_ratio\":%s}", a, st, tk, nd, $3, $(NF-2), $NF}')
[ -n "$LINE" ] || { echo "[score] empty"; tail -12 "$ROOT/gen.log"; exit 8; }
python3 -c "import json,sys;json.loads(sys.argv[1])" "$LINE" 2>/dev/null || { echo "[score] invalid: $LINE" >&2; exit 8; }
flock "$RES" -c "printf '%s\n' '$LINE' >> '$RES'"
echo "$LINE"
mv "$ROOT" "${ROOT}_done" 2>/dev/null || true
