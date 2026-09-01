#!/bin/bash
# R2 格的下游推理：与 B7b/p0bidir 完全同管线（TSTART=0.80、NSEQ=32、seed 2026），
# 只换 STATE 与 TAG。走 R2 worktree 的 runner（含 dump 改动但此处不用 dump）。
# 用法：TAG=r2d1_s42 STATE=$T/artifacts_r2/r2d1_s42_state.msgpack \
#       NODE=nidXXXX GPU0=0 NCH=4 TKL=$T/logs/tkb7b_feb_s42.txt bash launch_r2infer.sh
set -u
T=/lus/lfs1aip2/projects/public/u6gb/tasks/dfm_largescale_20260813
S0=/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/dfm-r2-20260830
A=/lus/lfs1aip2/projects/public/u6gb/tasks/ce_orderflow_20260812T200352Z/A02_scale
W=$S0/post_training/dfm/eval/run_eval_node.sh
STATE=${STATE:?}; TAG=${TAG:?}; NODE=${NODE:?}
JOB=${JOB:-6217606}; MO=${MO:-2026-01}
TKL=${TKL:-$T/logs/tkb7b_feb_s42.txt}
NCH=${NCH:-4}; GPU0=${GPU0:-0}; LOBTREE=${LOBTREE:-}

[ -f "$STATE" ] || { echo "FATAL: 缺 $STATE" >&2; exit 5; }
_st=$(python3 -c "import json;print(json.load(open('$STATE.meta')).get('step','?'))" 2>/dev/null)
[ "$_st" = "8000" ] || { echo "FATAL: $TAG state step=$_st != 8000" >&2; exit 5; }
_live=$(squeue -u "$(whoami)" -h -s -o "%j" 2>/dev/null | grep -c "^dfm-${TAG}-${MO}-")
[ "${_live:-0}" -gt 0 ] && { echo "FATAL: $TAG 已有 step 在跑" >&2; exit 7; }

mkdir -p $T/rollouts_anc $T/logs
CHPFX=$T/logs/${TAG}_${NODE}_${MO}_g${GPU0}_chunk_
split -n l/$NCH -d -a 1 $TKL $CHPFX
i=0
for CH in $(seq 0 $((NCH-1))); do
  env -u SLURM_NNODES -u SLURM_NTASKS -u SLURM_JOB_ID -u SLURMD_NODENAME \
  setsid nohup srun --jobid=$JOB --overlap --exact --cpus-per-task=8 -w $NODE -N1 -n1 \
    --cpu-bind=none --job-name=dfm-${TAG}-${MO}-$CH \
    --export=ALL,DFM_GPU=$((GPU0+i)),DFM_SCRIPT=dfm_correct_runner.py,\
DFM_LOB_TREE=$LOBTREE,DFM_SRC=$LOBTREE,DFM_S0=$S0,XLA_PYTHON_CLIENT_MEM_FRACTION=0.30,XLA_FLAGS=--xla_gpu_enable_triton_gemm=false \
    bash $W --month $MO --n-cond 500 --n-gen 500 \
      --stocks ${CHPFX}$CH --index-dir $A/idx --group-size 8 \
      --validate-first 8 --gate-batches 2 --state "$STATE" \
      --t-start 0.80 --n-steps 8 --n-seq 32 --batch-size 2 --corr-batch 2 \
      --anc-mode digit --anc-order-fields "" --anc-msg-rounds 0 \
      --residual-scale 1.0 --seed 2026 --skip-existing \
      --out-template "$T/rollouts_anc/${TAG}_{stock}_{month}_learned.npz" \
    > $T/logs/${TAG}_${MO}_${NODE}_$CH.log 2>&1 < /dev/null &
  echo "  $TAG chunk$CH -> $NODE gpu$((GPU0+i))"; i=$((i+1)); sleep 20
done
echo "=== $TAG inference launched $(date -u +%H:%M:%SZ) ==="
