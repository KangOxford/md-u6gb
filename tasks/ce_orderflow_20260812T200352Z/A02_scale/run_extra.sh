#!/bin/bash
# 借用邻居节点上物理空闲的卡。每张卡的分配都先过了 nvidia-smi 实测闸门：
#   nid010367 gpu1,2,3 -> 97.8 GB 空、util 0%
#   nid010272 gpu0,2   -> 46 GB 空、util 0%
#   nid011165          -> 只剩 11 GB，装不下修正器，故意不用
# --overlap 不会让被占的显存变空，所以判据是实测空闲字节数，不是 squeue 说什么。
# 共享卡上用 0.30 而不是 0.40：邻居随时可能再要显存，我不该把余量吃满。
set -u
S0=/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/dfm-post-training-20260801
A=/lus/lfs1aip2/projects/public/u6gb/tasks/ce_orderflow_20260812T200352Z/A02_scale
W=$S0/post_training/dfm/eval/run_eval_node.sh
S2A=$S0/post_training/dfm/artifacts/stage2a/long_NVDA_state.msgpack

go () {   # jobid node gpu chunk arm memfrac extra...
  local job=$1 node=$2 gpu=$3 chunk=$4 arm=$5 mf=$6; shift 6
  env -u SLURM_NNODES -u SLURM_NTASKS -u SLURM_JOB_ID -u SLURMD_NODENAME \
  srun --jobid=$job --overlap --exact --cpus-per-task=8 -w $node -N1 -n1 \
    --cpu-bind=none --job-name=dfm-xtra-$arm-${node#nid}-g$gpu \
    --export=ALL,DFM_GPU=$gpu,DFM_SCRIPT=dfm_correct_runner.py,XLA_PYTHON_CLIENT_MEM_FRACTION=$mf \
    bash $W --month 2026-01 --n-cond 500 --n-gen 500 \
      --stocks $chunk --index-dir $A/idx --group-size 8 --validate-first 8 \
      --gate-batches 2 --state "$S2A" --t-start 0.80 --n-steps 8 \
      --n-seq 8 --batch-size 4 --corr-batch 4 --skip-existing "$@" \
      --out-template "$A/rollouts/dfm_a02_{stock}_{month}_$arm.npz" \
      > $A/logs/xtra_${arm}_${node#nid}g${gpu}.log 2>&1 &
}

# (jobid node gpu memfrac) x 5，按实测空闲显存给的配额
SLOTS="5992007:nid010367:1:0.40 5992007:nid010367:2:0.40 5992007:nid010367:3:0.40 \
       5992007:nid010272:0:0.30 5992007:nid010272:2:0.30"

for arm in learned random; do
  echo "=== extra: $arm $(date -u +%H:%M:%SZ) ==="
  i=0
  for s in $SLOTS; do
    IFS=: read -r job node gpu mf <<< "$s"
    ex=""; [ "$arm" = random ] && ex="--random-p --random-p-seed 7"
    go $job $node $gpu $A/logs/xtra_0$i $arm $mf $ex
    i=$((i+1))
  done
  wait
  echo "$arm done: $(ls $A/rollouts/*_$arm.npz 2>/dev/null | wc -l) npz total"
done
echo "=== EXTRA DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
