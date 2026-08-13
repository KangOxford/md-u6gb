#!/bin/bash
# A02 全量扫描：488 ticker × 留出月，learned + 同范数 random 两臂。
#
# 为什么循环在进程内：四个跑完的点拟合出每次调用 640s 固定开销
# (装 78.5M 模型 + JIT + 挂 squashfs)，按 ticker 逐个调用要 87 小时纯启动。
# 挂载是按月的，488 个 ticker 都是它下面的子目录，所以一次挂载全覆盖。
#
# 为什么每卡只起两次进程而不是按 ticker 交替两臂：交替能让任何中断点都配对完整，
# 但每个 ticker 要付两次 640s。两次进程总开销 8×640s，中断时两臂 ticker 集合不等，
# 分析时取交集即可 —— 这个损失远小于 6 小时启动开销。
#
# --skip-existing：当前分配只剩 ~8.8h 而两臂全量需 ~8.1h，续跑能力是必需的而非可选。
set -u
S0=/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/dfm-post-training-20260801
A=/lus/lfs1aip2/projects/public/u6gb/tasks/ce_orderflow_20260812T200352Z/A02_scale
JOB=${JOB:-5992007}
NODE=${NODE:-nid010547}
NSEQ=${NSEQ:-8}
MONTH=${MONTH:-2026-01}
W=$S0/post_training/dfm/eval/run_eval_node.sh
S2A=$S0/post_training/dfm/artifacts/stage2a/long_NVDA_state.msgpack

mkdir -p $A/idx $A/rollouts $A/logs
split -n l/4 -d $A/tickers_$MONTH.txt $A/logs/chunk_

launch () {   # gpu chunk arm extra...
  local gpu=$1 chunk=$2 arm=$3; shift 3
  env -u SLURM_NNODES -u SLURM_NTASKS -u SLURM_JOB_ID -u SLURMD_NODENAME \
  srun --jobid=$JOB --overlap -w $NODE -N1 -n1 --cpu-bind=none \
    --job-name=dfm-scale-$arm-g$gpu \
    --export=ALL,DFM_GPU=$gpu,DFM_SCRIPT=dfm_correct_runner.py,XLA_PYTHON_CLIENT_MEM_FRACTION=0.40 \
    bash $W --month $MONTH --n-cond 500 --n-gen 500 \
      --stocks $chunk --index-dir $A/idx --group-size 8 --validate-first 8 \
      --gate-batches 2 --state "$S2A" --t-start 0.80 --n-steps 8 \
      --n-seq $NSEQ --batch-size 4 --corr-batch 4 --skip-existing "$@" \
      --out-template "$A/rollouts/dfm_a02_{stock}_{month}_$arm.npz" \
      > $A/logs/${arm}_g${gpu}.log 2>&1 &
}

split -n l/8 -d $A/tickers_$MONTH.txt $A/logs/sub_

echo "=== phase 0: build indices (no model loaded, build-index guard skips the gate) $(date -u +%H:%M:%SZ) ==="
for g in 0 1 2 3; do launch $g $A/logs/chunk_0$g idx --build-index; done
wait
echo "indices: $(ls $A/idx/*.txt 2>/dev/null | wc -l) files  $(date -u +%H:%M:%SZ)"

# 两臂在 chunk 的一半上交替，而不是先跑完全部 learned 再跑 random。
# 分配只剩 ~9h 而两臂全量需 ~10.6h（实测 0.039 s/msg），必然跑不完；
# 交替让「时间到了」得到的是「一半 ticker、两臂齐全」而不是「全部 ticker、没有对照」。
# A01 已经证明没有同范数随机对照的结论站不住。代价是 4 次额外启动 ≈ 16 分钟。
for half in 0 1; do
  echo "=== half $half : learned $(date -u +%H:%M:%SZ) ==="
  for g in 0 1 2 3; do launch $g $A/logs/sub_0$((g*2+half)) learned; done
  wait
  echo "=== half $half : random  $(date -u +%H:%M:%SZ) ==="
  for g in 0 1 2 3; do launch $g $A/logs/sub_0$((g*2+half)) random --random-p --random-p-seed 7; done
  wait
  echo "half $half done: learned=$(ls $A/rollouts/*_learned.npz 2>/dev/null|wc -l) random=$(ls $A/rollouts/*_random.npz 2>/dev/null|wc -l)"
done
echo "=== SWEEP DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
