#!/bin/bash
# S7 + S8 + the evaluations that need trained weights, as an attached step.
#
# Takes over where 04_attached_run.sh stops. Both post-training stages start from the
# same pretrained checkpoint rather than chaining, because they answer different
# questions and chaining would confound them: FL2VA adds keyframe conditioning, CFG
# distillation folds guidance into the weights, and running the second on top of the
# first would make it impossible to say which stage moved a number.
#
#   [1] sft_fl2va    anchors at t=0.999          -> E4
#   [2] distill_cfg  guidance into the weights   -> E2, E3
#   [3] sample       guided vs distilled          -> E1, E2
#   [4] evaluate     E1 / E3 / E4
#
# Usage:  PRE_RUN=<pretrain run name> bash code/05_posttrain_run.sh

set -uo pipefail
TASKDIR=/lus/lfs1aip2/projects/public/u6gb/tasks/minimax_h3_20260812T
source "$TASKDIR/code/env.sh"

RUNS="$TASKDIR/runs"
RUN_TAG="post_${SLURM_JOB_ID:-na}_$(hostname)"
EVAL="$RUNS/eval_${RUN_TAG}"
mkdir -p "$EVAL"

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
MODEL=${MODEL:-nano}
BATCH=${BATCH:-8}
ACCUM=${ACCUM:-2}
SFT_STEPS=${SFT_STEPS:-5000}
DISTILL_STEPS=${DISTILL_STEPS:-4000}
GUIDANCE=${GUIDANCE:-5.0}

if [ -z "${PRE_RUN:-}" ]; then
    echo "FATAL: set PRE_RUN to the pretrained run name (a directory under runs/)"; exit 2
fi
TEACHER="$RUNS/$PRE_RUN/checkpoints"
if [ ! -f "$TEACHER/latest_checkpoint.json" ]; then
    echo "FATAL: no checkpoint breadcrumb at $TEACHER"; exit 2
fi

SFT_RUN="sft_fl2va-${MODEL}-${RUN_TAG}"
DIS_RUN="distill_cfg-${MODEL}-${RUN_TAG}"
FAILS=0
note () { FAILS=$((FAILS + 1)); echo "WARN: $1"; }
banner () { echo; echo "================================================================"; echo "=== $1  ($(date -u +%FT%TZ))"; echo "================================================================"; }

banner "post-training from $PRE_RUN on $(hostname), $NGPU GPU(s)"
# `$PY -m torch.distributed.run`, never bare `torchrun`: the latter resolves off PATH
# to the system python 3.13 torch, which cannot see these cards.
LAUNCH="$PY -m torch.distributed.run --nproc_per_node=$NGPU"

# ---------------------------------------------------------------- [1] FL2VA
banner "[1/4] FL2VA post-training (anchors held at t=0.999)"
$LAUNCH --master_port=29541 "$TASKDIR/code/train.py" \
    --root "$TASKDIR" --stage sft_fl2va --model "$MODEL" --run-name "$SFT_RUN" \
    --teacher "$TEACHER" --anchors first,last \
    --steps "$SFT_STEPS" --batch-size "$BATCH" --grad-accum "$ACCUM" \
    --lr 1e-4 --warmup 200 --cfg-dropout 0.1 --ckpt-every 1000 --log-every 50
[ $? -ne 0 ] && note "fl2va training failed"

banner "[E4] anchor sweep -- does t=0.999 beat t=1.0?"
"$PY" "$TASKDIR/code/evaluate.py" --root "$TASKDIR" --device cuda:0 \
    --out "$EVAL/E4_anchors.json" anchors --checkpoint "$RUNS/$SFT_RUN/checkpoints" --n 96 \
    || note "E4 anchor sweep failed"

# ---------------------------------------------------------------- [2] distill
banner "[2/4] distilling guidance into the weights (w=$GUIDANCE)"
$LAUNCH --master_port=29542 "$TASKDIR/code/train.py" \
    --root "$TASKDIR" --stage distill_cfg --model "$MODEL" --run-name "$DIS_RUN" \
    --teacher "$TEACHER" --guidance-scale "$GUIDANCE" \
    --steps "$DISTILL_STEPS" --batch-size "$BATCH" --grad-accum "$ACCUM" \
    --lr 5e-5 --warmup 200 --ckpt-every 1000 --log-every 50
[ $? -ne 0 ] && note "distillation failed"

# ---------------------------------------------------------------- [3] sample
banner "[3/4] sampling: guided teacher vs distilled student"
# E2 lives here: the teacher pays two forward passes per step and the student one,
# and `sample.py` records both the count and the wall-clock into samples.pt.
"$PY" "$TASKDIR/code/sample.py" --root "$TASKDIR" --checkpoint "$TEACHER" \
    --out "$EVAL/samples_guided" --steps 32 --guidance-scale "$GUIDANCE" \
    --save-decoded --device cuda:0 || note "guided sampling failed"
"$PY" "$TASKDIR/code/sample.py" --root "$TASKDIR" --checkpoint "$RUNS/$DIS_RUN/checkpoints" \
    --out "$EVAL/samples_distilled" --steps 32 --guidance-scale 0 \
    --save-decoded --device cuda:0 || note "distilled sampling failed"

# ---------------------------------------------------------------- [4] evaluate
banner "[4/4] E1 audio-visual sync, E3 held-out loss"
for V in guided distilled; do
    [ -f "$EVAL/samples_${V}/samples.pt" ] && "$PY" "$TASKDIR/code/evaluate.py" \
        --root "$TASKDIR" --device cuda:0 --out "$EVAL/E1_avsync_${V}.json" \
        avsync --samples "$EVAL/samples_${V}/samples.pt"
done
for R in "$PRE_RUN:teacher" "$DIS_RUN:distilled"; do
    NAME="${R%%:*}"; TAG="${R##*:}"
    [ -f "$RUNS/$NAME/checkpoints/latest_checkpoint.json" ] && "$PY" "$TASKDIR/code/evaluate.py" \
        --root "$TASKDIR" --device cuda:0 --out "$EVAL/E3_heldout_${TAG}.json" \
        heldout --checkpoint "$RUNS/$NAME/checkpoints" --n 192
done

banner "post-training finished, $FAILS soft failure(s)"
echo "eval: $EVAL"; ls "$EVAL"
[ "$FAILS" -gt 0 ] && exit 1
exit 0
