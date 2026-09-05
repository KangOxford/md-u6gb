#!/usr/bin/env bash
# Score the PARENT checkpoint on the same eval contexts and the same generation
# seeds as the continued-training runs, keeping the per-context returns.
#
# The panel already holds a pooled number for this arm. A pooled number cannot
# answer the question this session owns, because parent and continued share the
# same 500 contexts and the comparison that uses that pairing is far more
# precise than one that throws it away -- and fair CRPS has now been shown to
# read conditioning (context shuffle degrades it 41.5%, z=10.66), so a per-context
# contrast is the estimand, not a convenience.
set -uo pipefail
S=/lus/lfs1aip2/projects/public/u6gb/nb_build_pr22
T=/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z
Wo=/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/crps-return-alignment-20260808
STOCK="${STOCK:-META}"; ARM="${ARM:?}"; RES="${RES:?}"
SEEDS="${SEEDS:-97901 97902}"
CK="$T/ckpt/wm_ft_${ARM}"
[ -d "$CK" ] || { echo "[ckpt] missing $CK" >&2; exit 6; }
# The skip key must include K. It did not, so a K=2 row from an earlier run
# matched and silently skipped the K=4 rescoring -- the job reported OK in five
# seconds having done nothing, and the record still said n_seeds 2. Same family
# as a record asserting a parameter the estimator never used.
_NS=$(echo $SEEDS | wc -w)
grep -q "\"traj\":\"parent_${ARM}\",\"step\":\"final\",\"ticker\":\"$STOCK\",\"node\":\"[^\"]*\",\"n_seeds\":$_NS" "$RES" 2>/dev/null && { echo "[skip] already have K=$_NS"; exit 0; }
ROOT="/local/user/$(id -u)/pc_$$_${ARM}_${STOCK}"; mkdir -p "$ROOT/data"
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
    # `-f` accepts a 0-byte .done, which is what a redirection that succeeded followed by
    # a writer that died leaves behind -- an allocation expiring mid-write produces exactly
    # that, and the member is then scored as complete.  Existence is not integrity.
    if [ ! -s "$OUT_ROOT/member_0/.done" ] || \
       ! python3 -c 'import json,sys; json.load(open(sys.argv[1]))' \
                 "$OUT_ROOT/member_0/.done" 2>/dev/null; then
        echo "[gen] seed $sd produced nothing usable (exit=$_rc)" >&2
        tail -6 "$ROOT/gen.log" >&2; exit 7
    fi
    ARMLIST="${ARMLIST:+$ARMLIST,}s$sd"
done
cd "$Wo/run/mid_training"
LINE=$(MIDTRAIN_TASK_ROOT="$ROOT" python3 score_v5_primary.py --task "$ROOT" \
    --arms a="$ARMLIST" b="$ARMLIST" --baseline a --n-boot 60 --min-overlap 0 \
    --out "$ROOT/score.json" 2>>"$ROOT/gen.log" \
  | awk -v a="parent_$ARM" -v tk="$STOCK" -v nd="$(hostname)" -v ns="$(ls -d "$ROOT"/data/s* 2>/dev/null | wc -l)" '/^\[a\] CRPS/{
      printf "{\"traj\":\"%s\",\"step\":\"final\",\"ticker\":\"%s\",\"node\":\"%s\",\"n_seeds\":%s,\"crps\":\"%s\",\"qL1\":%s,\"sd_ratio\":%s}", a, tk, nd, ns, $3, $(NF-2), $NF}')
[ -n "$LINE" ] || { echo "[score] empty"; tail -12 "$ROOT/gen.log"; exit 8; }
flock "$RES" -c "printf '%s\n' '$LINE' >> '$RES'"
echo "$LINE"
# keep the per-context returns where the paired analysis can find them
mv "$ROOT" "/home/u6gb/kangli.u6gb/crps_runspread_20260905/parent_${ARM}_${STOCK}_K$(ls -d "$ROOT"/data/s* 2>/dev/null | wc -l)_done" 2>/dev/null || true
