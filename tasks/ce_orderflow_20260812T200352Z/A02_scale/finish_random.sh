#!/bin/bash
# 补齐 random 对照臂。两台邻居节点此刻实测几乎全空（8 张卡 57-98 GB 空闲、util 0%），
# 所以配额可以给 0.40；判据仍是实测空闲字节数，不是 squeue。
set -u
S0=/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/dfm-post-training-20260801; A=/lus/lfs1aip2/projects/public/u6gb/tasks/ce_orderflow_20260812T200352Z/A02_scale
W=$S0/post_training/dfm/eval/run_eval_node.sh
S2A=$S0/post_training/dfm/artifacts/stage2a/long_NVDA_state.msgpack
i=0
for slot in nid010367:0 nid010367:1 nid010367:2 nid010367:3 \
            nid010272:0 nid010272:1 nid010272:2 nid010272:3; do
  IFS=: read -r node gpu <<< "$slot"
  env -u SLURM_NNODES -u SLURM_NTASKS -u SLURM_JOB_ID -u SLURMD_NODENAME \
  srun --jobid=5992007 --overlap --exact --cpus-per-task=8 -w $node -N1 -n1 \
    --cpu-bind=none --job-name=dfm-rnd-${node#nid}-g$gpu \
    --export=ALL,DFM_GPU=$gpu,DFM_SCRIPT=dfm_correct_runner.py,XLA_PYTHON_CLIENT_MEM_FRACTION=0.40 \
    bash $W --month 2026-01 --n-cond 500 --n-gen 500 \
      --stocks $A/logs/rnd_0$i --index-dir $A/idx --group-size 8 --validate-first 8 \
      --gate-batches 2 --state "$S2A" --t-start 0.80 --n-steps 8 \
      --n-seq 8 --batch-size 4 --corr-batch 4 --skip-existing \
      --random-p --random-p-seed 7 \
      --out-template "$A/rollouts/dfm_a02_{stock}_{month}_random.npz" \
      > $A/logs/rndfix_${node#nid}g${gpu}.log 2>&1 &
  i=$((i+1))
done
wait
echo "random total: $(ls $A/rollouts/*_random.npz 2>/dev/null|wc -l)  $(date -u +%H:%M:%SZ)"
