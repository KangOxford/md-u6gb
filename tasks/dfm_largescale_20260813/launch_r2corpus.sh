#!/bin/bash
# R2 语料生成：AR draft token + real token + draft book 的配对语料。
# 用法：MONTH=2025-06 NODE=nid011132 GPU0=0 NCH=4 bash launch_r2corpus.sh
# 每 chunk 两遍：--build-index（建窗口清单）→ --dump-draft-only（生成并存盘）。
# runner 来自 R2 worktree（dfm-r2-20260830），与 PR#22 评判线代码隔离。
set -u
T=/lus/lfs1aip2/projects/public/u6gb/tasks/dfm_largescale_20260813
S0=/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/dfm-r2-20260830
A=/lus/lfs1aip2/projects/public/u6gb/tasks/ce_orderflow_20260812T200352Z/A02_scale
W=$S0/post_training/dfm/eval/run_eval_node.sh
STATE=$T/artifacts_b7b/b7b_off_s42_state.msgpack   # 只为过载入；dump 模式不跑修正
MONTH=${MONTH:?}; NODE=${NODE:?}; JOB=${JOB:-6197253}
TKL=${TKL:-$T/logs/g2x_tk60.txt}
NCH=${NCH:-4}; GPU0=${GPU0:-0}; NSEQ=${NSEQ:-32}
TAG=r2c_${MONTH}

_live=$(squeue -u "$(whoami)" -h -s -o "%j" 2>/dev/null | grep -c "^dfm-${TAG}-")
[ "${_live:-0}" -gt 0 ] && { echo "FATAL: $TAG 已有 step 在跑" >&2; exit 7; }

mkdir -p $T/corpus_r2 $T/logs
CHPFX=$T/logs/${TAG}_${NODE}_${MONTH}_g${GPU0}_chunk_
split -n l/$NCH -d -a 1 $TKL $CHPFX
i=0
for CH in $(seq 0 $((NCH-1))); do
  CARGS="--month $MONTH --n-cond 500 --n-gen 500 \
      --stocks ${CHPFX}$CH --index-dir $A/idx --group-size 8 \
      --validate-first 8 --gate-batches 2 --state $STATE \
      --t-start 0.80 --n-steps 8 --n-seq $NSEQ --batch-size 2 --corr-batch 2 \
      --anc-mode digit --anc-order-fields '' --anc-msg-rounds 0 \
      --residual-scale 1.0 --seed 2026 --skip-existing \
      --out-template $T/corpus_r2/${TAG}_{stock}_{month}_learned.npz"
  env -u SLURM_NNODES -u SLURM_NTASKS -u SLURM_JOB_ID -u SLURMD_NODENAME \
  setsid nohup srun --jobid=$JOB --overlap --exact --cpus-per-task=8 -w $NODE -N1 -n1 \
    --cpu-bind=none --job-name=dfm-${TAG}-$CH \
    --export=ALL,DFM_GPU=$((GPU0+i)),DFM_SCRIPT=dfm_correct_runner.py,\
XLA_PYTHON_CLIENT_MEM_FRACTION=0.30,XLA_FLAGS=--xla_gpu_enable_triton_gemm=false \
    bash -c "bash $W $CARGS --build-index && bash $W $CARGS --dump-draft-only" \
    > $T/logs/${TAG}_${MONTH}_${NODE}_$CH.log 2>&1 < /dev/null &
  echo "  $TAG chunk$CH -> $NODE gpu$((GPU0+i))"; i=$((i+1)); sleep 20
done
echo "=== $TAG launched $(date -u +%H:%M:%SZ) ==="
