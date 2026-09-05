# Score an existing node-local cell without regenerating. Read-only on the rollouts.
set -uo pipefail
ROOT=${ROOT:?}; LABEL=${LABEL:?}; TICKER=${TICKER:?}; CKPT=${CKPT:?}
W=/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/crps-return-alignment-20260808
OUTHOME=/lus/lfs1aip2/projects/public/u6gb/tasks/crps_analysis_20260905/cells
PY=/lus/lfs1aip2/projects/public/s5e/quant_team/quant/miniforge3/bin/python
[ -d "$ROOT/data/s97901" ] && [ -d "$ROOT/data/s97902" ] || { echo "[rescore] ABORT: rollouts missing in $ROOT"; exit 4; }
"$PY" "$W/run/mid_training/score_v5_primary.py" --task "$ROOT" \
  --arms "a=s97901,s97902" --baseline a --assert-k 2 \
  --assert-ckpt "$CKPT" --assert-ckpt-step 69378 --assert-seeds 97901,97902 \
  --out "$ROOT/score.json" 2>&1 | tail -8
[ -f "$ROOT/score.json" ] || { echo "[rescore] no score.json"; exit 5; }
mkdir -p "$OUTHOME/${LABEL}_${TICKER}"
cp -p "$ROOT/score.json" "$OUTHOME/${LABEL}_${TICKER}/score.json"
for SD in 97901 97902; do
  d="$OUTHOME/${LABEL}_${TICKER}/s$SD"; mkdir -p "$d"
  for f in .returns_gen.npz .returns_real.npz sample_indices_rank0.json .done; do
    [ -f "$ROOT/data/s$SD/member_0/$f" ] && cp -p "$ROOT/data/s$SD/member_0/$f" "$d/${f#.}"
  done
done
echo "[rescore] SCORED $LABEL/$TICKER"
