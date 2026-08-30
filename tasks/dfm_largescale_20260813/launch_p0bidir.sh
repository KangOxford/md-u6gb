#!/bin/bash
# P=0 纯 bidirectional 消融（AUDIT_BIDIRECTIONAL §4 的 b0）：
# 把 teacher-forced −0.53 与生成侧差异拆成「双向结构效应」vs「学到的 P 修正」。
# runner:869-877 的注释注册过这格的语义（random_p_scale=0.0 = 无 DFM 残差的
# 纯双向前向），但它从未出过数。
# 启动行照抄 run_anc.sh（ANC_MODE=off 的形状），只加 --random-p 三件。
set -u
T=/lus/lfs1aip2/projects/public/u6gb/tasks/dfm_largescale_20260813
S0=/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/dfm-post-training-20260801
A=/lus/lfs1aip2/projects/public/u6gb/tasks/ce_orderflow_20260812T200352Z/A02_scale
W=$S0/post_training/dfm/eval/run_eval_node.sh
STATE=$T/artifacts_b7b/b7b_off_s42_state.msgpack
TAG=p0bidir_s42; NODE=nid011131; JOB=6197253; MO=2026-01
TKL=$T/logs/tkb7b_feb_s42.txt          # 与 B7b 四格同一个 27-ticker 池
NCH=4; GPU0=0

[ -f "$STATE" ] || { echo "FATAL: 缺 $STATE" >&2; exit 5; }
_live=$(squeue -u "$(whoami)" -h -s -o "%j" 2>/dev/null | grep -c "^dfm-${TAG}-")
[ "${_live:-0}" -gt 0 ] && { echo "FATAL: $TAG 已有 step 在跑" >&2; exit 7; }

mkdir -p $T/rollouts_anc $T/logs
CHPFX=$T/logs/${TAG}_${NODE}_${MO}_g${GPU0}_chunk_
split -n l/$NCH -d -a 1 $TKL $CHPFX
i=0
for CH in $(seq 0 $((NCH-1))); do
  env -u SLURM_NNODES -u SLURM_NTASKS -u SLURM_JOB_ID -u SLURMD_NODENAME \
  setsid nohup srun --jobid=$JOB --overlap --exact --cpus-per-task=8 -w $NODE -N1 -n1 \
    --cpu-bind=none --job-name=dfm-${TAG}-$CH \
    --export=ALL,DFM_GPU=$((GPU0+i)),DFM_SCRIPT=dfm_correct_runner.py,\
XLA_PYTHON_CLIENT_MEM_FRACTION=0.30,XLA_FLAGS=--xla_gpu_enable_triton_gemm=false \
    bash $W --month $MO --n-cond 500 --n-gen 500 \
      --stocks ${CHPFX}$CH --index-dir $A/idx --group-size 8 \
      --validate-first 8 --gate-batches 2 --state "$STATE" \
      --t-start 0.80 --n-steps 8 --n-seq 32 --batch-size 2 --corr-batch 2 \
      --anc-mode digit --anc-order-fields "" --anc-msg-rounds 0 \
      --residual-scale 1.0 --seed 2026 --skip-existing \
      --random-p --random-p-seed 7 --random-p-scale 0.0 \
      --out-template "$T/rollouts_anc/${TAG}_{stock}_{month}_learned.npz" \
    > $T/logs/${TAG}_${MO}_${NODE}_$CH.log 2>&1 < /dev/null &
  echo "  $TAG chunk$CH -> $NODE gpu$((GPU0+i))"; i=$((i+1)); sleep 20
done
echo "=== $TAG launched $(date -u +%H:%M:%SZ) ==="
