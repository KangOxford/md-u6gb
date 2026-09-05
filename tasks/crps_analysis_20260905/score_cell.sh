# One scored cell: two generation seeds of one checkpoint on one ticker, K=2.
# The empty-card check runs INSIDE this step. --assert-* make the provenance
# come from the estimator, not from a shell directory count.
set -uo pipefail
CKPT=${CKPT:?}; TICKER=${TICKER:?}; WANT=${WANT:?}; LABEL=${LABEL:?}
W=/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/crps-return-alignment-20260808
T=/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z
OUTHOME=/lus/lfs1aip2/projects/public/u6gb/tasks/crps_analysis_20260905/cells
PY=/lus/lfs1aip2/projects/public/s5e/quant_team/quant/miniforge3/bin/python

USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n "$((WANT+1))p")
echo "[cell] $(hostname) card=$WANT used=${USED}MiB label=$LABEL ticker=$TICKER"
[ "${USED:-999999}" -gt 2000 ] && { echo "[cell] ABORT: card holds ${USED}MiB"; exit 3; }
export CUDA_VISIBLE_DEVICES=$WANT

ROOT=/local/user/$(id -u)/cell_${LABEL}_${TICKER}_${SLURM_JOB_ID:-na}_$$
mkdir -p "$ROOT/data"
IDX=$T/data/v5m_eval_idx_${TICKER}.txt
[ -f "$IDX" ] || { echo "[cell] ABORT: no index file $IDX"; exit 4; }

for SD in 97901 97902; do
  CKPT="$CKPT" CKPT_STEP=69378 STOCK="$TICKER" INDICES="$IDX" \
  N_SEQ=500 BATCH=48 N_COND=250 N_GEN=250 K=1 SEED0=$SD SEED_STRIDE=0 \
  OUT_ROOT="$ROOT/data/s$SD" \
  bash "$W/run/mid_training/collect_rollouts.sh" >> "$ROOT/collect_s$SD.log" 2>&1
  rc=$?
  [ -f "$ROOT/data/s$SD/member_0/.done" ] || { echo "[cell] ABORT: seed $SD produced no .done (collect rc=$rc)"; exit 5; }
  echo "[cell] seed $SD collected"
done

"$PY" "$W/run/mid_training/score_v5_primary.py" --task "$ROOT" \
  --arms "a=s97901,s97902" --assert-k 2 \
  --assert-ckpt "$CKPT" --assert-ckpt-step 69378 --assert-seeds 97901,97902 \
  --out "$ROOT/score.json" >> "$ROOT/score.log" 2>&1
rc=$?
if [ -f "$ROOT/score.json" ]; then
  mkdir -p "$OUTHOME/${LABEL}_${TICKER}"
  cp -p "$ROOT/score.json" "$OUTHOME/${LABEL}_${TICKER}/score.json"
  for SD in 97901 97902; do
    d="$OUTHOME/${LABEL}_${TICKER}/s$SD"; mkdir -p "$d"
    for f in .returns_gen.npz .returns_real.npz sample_indices_rank0.json .done; do
      [ -f "$ROOT/data/s$SD/member_0/$f" ] && cp -p "$ROOT/data/s$SD/member_0/$f" "$d/${f#.}"
    done
  done
  echo "[cell] SCORED $LABEL/$TICKER -> $OUTHOME/${LABEL}_${TICKER}"
else
  echo "[cell] SCORING FAILED rc=$rc; tail:"; tail -12 "$ROOT/score.log" 2>/dev/null
fi
exit $rc
