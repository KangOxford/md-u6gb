#!/usr/bin/env bash
# One round-3 replicate trajectory, to step 1500 so that step 1200 is recorded.
# Three fixes over the launcher that produced 0/14 usable runs:
#   (1) the empty-card check runs INSIDE this step, not in a separate srun -- a
#       gtop reading taken on the login node is stale by the time the step lands;
#   (2) DEST carries a RUNID, so two runs can never sync into one directory
#       (that is what made wm_ft_traj_s30 a three-run chimera);
#   (3) checkpoints go to node-local storage and are copied back, so Lustre sees
#       no per-step metadata traffic.
set -uo pipefail
TSEED=${TSEED:?}; WANT=${WANT:?}; MAXSTEP=${MAXSTEP:-1500}
W=/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/crps-return-alignment-20260808
T=/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z
QUANT_ROOT=/lus/lfs1aip2/projects/public/s5e/quant_team/quant
PY=$QUANT_ROOT/miniforge3/bin/python

USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n "$((WANT+1))p")
echo "[r3] $(hostname) card=$WANT used=${USED}MiB tseed=$TSEED maxstep=$MAXSTEP"
if [ "${USED:-999999}" -gt 2000 ]; then echo "[r3] ABORT: card $WANT holds ${USED}MiB"; exit 3; fi
export CUDA_VISIBLE_DEVICES=$WANT

RUNID="${SLURM_JOB_ID:-na}_$$"
LOCAL=/local/user/$(id -u)/r3rep_s${TSEED}_${RUNID}
DEST=$T/ckpt/wm_ft_r3rep_s${TSEED}
mkdir -p "$LOCAL" "$DEST" || { echo "[r3] ABORT: cannot create dirs"; exit 4; }

# Copy back every 180 s. Failures are reported, not swallowed: the old syncer
# printed "final sync done" unconditionally and read the step from the SOURCE,
# so a total copy-back failure was indistinguishable from success.
# wmle_full_ft.py saves to "${--out}_step<N>", i.e. a SIBLING of $LOCAL, not a child.
# Copying "$LOCAL/." therefore succeeds while persisting only ft_progress.json -- a
# check that passes because it measures the wrong thing. Copy the siblings, and count
# what actually landed rather than trusting cp's exit status.
sync_ckpts() {
  local n=0
  cp -a --update "$LOCAL"/. "$DEST"/ 2>/dev/null || true
  for d in "${LOCAL}"_step*; do
    [ -d "$d" ] || continue
    cp -a --update "$d" "$DEST"/ 2>/dev/null && n=$((n+1))
  done
  local on_dest; on_dest=$(ls -d "$DEST"/*_step* 2>/dev/null | wc -l)
  echo "[r3][copyback] $(date -u +%H:%M:%SZ) local=$n on_dest=$on_dest"
  [ "$on_dest" -gt 0 ] || echo "[r3][copyback] WARNING: nothing persisted to $DEST"
}
( while sleep 180; do sync_ckpts; done ) & SYNC=$!
trap 'kill $SYNC 2>/dev/null; sync_ckpts; echo "[r3][copyback] final -> $DEST"' EXIT

export PYTHONPATH="$W/src:$W:${PYTHONPATH:-}" PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
unset MAMBA3_LEGACY_NORM || true
CONDA_PREFIX=$QUANT_ROOT/miniforge3
NV=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia
TL=$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$TL:$NV/cuda_nvrtc/lib:$NV/cuda_runtime/lib:$NV/cusparse/lib:$NV/cuda_cupti/lib:$NV/cufft/lib:$NV/nvjitlink/lib:$NV/cusolver/lib:$NV/nccl/lib:$NV/cublas/lib:$NV/cudnn/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# Round 3's registered split: 64 train seeds (99000-99077, last digit not 8/9),
# 16 hold seeds (last digit 8 or 9). 64 x 75 files = the 4800-item epoch.
TRAIN=$(seq 99000 99079 | awk '$1%10<8'  | tr '\n' ' ')
HOLD=$( seq 99000 99079 | awk '$1%10>=8' | tr '\n' ' ')

"$PY" -u "$W/run/mid_training/wmle_full_ft.py" \
  --task "$T" --out "$LOCAL" --ckpt "$T/ckpt/wm_ft_multi2" --step 69378 --start-step 0 \
  --weights v5m3_weights.npz --prefix v5m3 \
  --train-seeds $TRAIN --hold-seeds $HOLD \
  --lr 1e-5 --epochs 1 --anchor-lambda 1.0 --clip 1.0 \
  --micro 2 --group-items 1 --eval-every 50 --save-every 150 \
  --train-seed "$TSEED" --max-step "$MAXSTEP"
rc=$?
echo "[r3] tseed=$TSEED trainer exit=$rc"
exit $rc
