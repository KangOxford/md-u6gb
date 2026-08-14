#!/bin/bash
# H3 replication, run as a step attached to an allocation that was sitting idle.
#
# Not an sbatch: `gtop` showed four GH200s doing nothing inside an allocation this
# account already holds, so the work goes there instead of into the queue. See
# CLAUDE.md section 4.0 -- checking gtop and attaching is now a hard precondition of
# submitting anything.
#
# An attached step dies with its parent allocation, and this one has hours, not days.
# So the stages are ordered by how durable their output is: everything that survives
# the step runs first, and training -- which checkpoints and can resume anywhere --
# runs last with whatever time is left.
#
#   [1] assets      -> ckpt/h3, data/vggsound        durable, ~177 GB, no GPU needed
#   [2] gates       -> env + 13 convention checks    seconds; fails loudly and early
#   [3] corpus      -> data/latents                  durable, the expensive GPU pass
#   [4] roundtrip   -> E6                            gate: is the corpus even right
#   [5] pretrain    -> runs/<name>/checkpoints       resumable, takes the remainder

set -uo pipefail

TASKDIR=/lus/lfs1aip2/projects/public/u6gb/tasks/minimax_h3_20260812T
source "$TASKDIR/code/env.sh"

RUN_TAG="attached_${SLURM_JOB_ID:-na}_$(hostname)"
RUNS="$TASKDIR/runs"
EVAL="$RUNS/eval_${RUN_TAG}"
mkdir -p "$RUNS" "$EVAL"

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
SHARDS=${SHARDS:-01}          # shard 01 is already staged from the failed job 5998320
CLIP_LIMIT=${CLIP_LIMIT:-12000}
MODEL=${MODEL:-nano}
PRETRAIN_STEPS=${PRETRAIN_STEPS:-20000}
BATCH=${BATCH:-8}
ACCUM=${ACCUM:-2}
PRE_RUN="pretrain-${MODEL}-${RUN_TAG}"

banner () { echo; echo "================================================================"; echo "=== $1  ($(date -u +%FT%TZ))"; echo "================================================================"; }
banner "attached H3 run on $(hostname), $NGPU GPU(s)"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
echo "shards=$SHARDS clips<=$CLIP_LIMIT model=$MODEL steps=$PRETRAIN_STEPS batch=${BATCH}x${ACCUM}"

# ---------------------------------------------------------------- [1] assets
if [ -f "$TASKDIR/ckpt/h3/transformer/config.json" ] && [ -f "$TASKDIR/data/vggsound/vggsound.csv" ]; then
    banner "[1/5] assets already staged"
else
    banner "[1/5] staging assets"
    "$PY" "$TASKDIR/code/fetch_assets.py" --root "$TASKDIR" --shards "$SHARDS" --workers 8
    RC=$?; [ $RC -ne 0 ] && { echo "FATAL: asset staging failed rc=$RC"; exit $RC; }
fi

# ---------------------------------------------------------------- [3] corpus
# (stage [2], the env and convention gates, runs inside preprocess_vggsound.py's
# `verify_environment()` -- it lives there so it also reaches queued jobs.)
if [ -f "$TASKDIR/data/latents/manifest.json" ]; then
    banner "[2-3/5] latent corpus already present"
else
    banner "[2-3/5] env gate, convention checks, then the latent corpus"
    # `media` alone when the text bank is already built. `both` would rebuild it, and
    # a rebuild at the wrong --max-text-tokens silently truncates every Context-IR
    # prompt: the template runs 101-111 tokens, so a 96 cap loses the
    # `non_diegetic_music` section entirely. 128 is the correct cap.
    PHASE=both
    [ -f "$TASKDIR/data/latents/text_bank.pt" ] && PHASE=media
    echo "[stage] phase=$PHASE (text bank $( [ "$PHASE" = media ] && echo present || echo absent ))"
    "$PY" "$TASKDIR/code/preprocess_vggsound.py" --root "$TASKDIR" --phase "$PHASE" \
        --shards "$SHARDS" --num-frames 73 --size 256 --max-text-tokens 128 \
        --per-shard 2000 --decode-workers 16 --limit "$CLIP_LIMIT" --device cuda:0
    RC=$?; [ $RC -ne 0 ] && { echo "FATAL: preprocessing failed rc=$RC"; exit $RC; }
fi

# ---------------------------------------------------------------- [4] gate
banner "[4/5] VAE round-trip gate (E6)"
"$PY" "$TASKDIR/code/evaluate.py" --root "$TASKDIR" --device cuda:0 \
    --out "$EVAL/E6_roundtrip.json" roundtrip --shard "${SHARDS%%,*}" --n 12 \
    --num-frames 73 --size 256
RC=$?; [ $RC -ne 0 ] && { echo "FATAL: round-trip gate failed rc=$RC"; exit $RC; }

# ---------------------------------------------------------------- [5] pretrain
banner "[5/5] pretraining H3-nano from scratch on $NGPU GPU(s)"
# Checkpoints go to $TMPDIR and sync to Lustre at the end, but an attached step can be
# killed with its parent, so they also sync every save via the run directory the
# trainer resumes from. `--ckpt-every 1000` is tighter than the 15-minute rule of
# thumb for exactly that reason.
torchrun --nproc_per_node="$NGPU" --master_port=29531 "$TASKDIR/code/train.py" \
    --root "$TASKDIR" --stage pretrain --model "$MODEL" --run-name "$PRE_RUN" \
    --steps "$PRETRAIN_STEPS" --batch-size "$BATCH" --grad-accum "$ACCUM" \
    --lr 3e-4 --warmup 800 --cfg-dropout 0.1 --ckpt-every 1000 --log-every 50
RC=$?

banner "attached run finished rc=$RC"
echo "eval:  $EVAL"
echo "run:   $RUNS/$PRE_RUN"
exit $RC
