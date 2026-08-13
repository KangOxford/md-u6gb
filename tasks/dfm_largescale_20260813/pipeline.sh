#!/bin/bash
# 训练 -> 推理 -> benchmark/scoring 的端到端链条，参数化到一个残差 checkpoint。
#
# 为什么写成一个脚本：这四步此前分散在三个任务目录里，每次换 checkpoint 都要手动
# 串一遍，而「产物名不带 checkpoint 身份」已经害过一次（两个配置写同名 npz，慢的
# 那个后写盘覆盖了快的，导致一个正确结果被错误撤回）。这里每个产物名都带 $TAG。
#
# 用法：
#   TAG=lg488b_g2 STATE=/path/to/xxx_state.msgpack JOB=<jobid> NODE=<nid> \
#   MONTHS="2026-01 2026-02" NSEQ=8 bash pipeline.sh
#
# 前置：STATE 必须是 Stage 2A（主干逐位未动）。runner 会探 msgpack 的键自行判断，
# 并把 stage / trunk_l1_shift 写进每个 rollout 的 sidecar —— 判定读那里，不读日志。
set -u
TAG=${TAG:?need TAG}
STATE=${STATE:?need STATE}
JOB=${JOB:?need JOB}
NODES=${NODES:?need NODES}          # 空格分隔，可跨节点
MONTHS=${MONTHS:-"2026-01 2026-02"}
NSEQ=${NSEQ:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
GPU0=${GPU0:-0}

S0=/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/dfm-post-training-20260801
A=/lus/lfs1aip2/projects/public/u6gb/tasks/ce_orderflow_20260812T200352Z/A02_scale
OUT=/lus/lfs1aip2/projects/public/u6gb/tasks/dfm_largescale_20260813
W=$S0/post_training/dfm/eval/run_eval_node.sh
mkdir -p $OUT/{rollouts,logs,out}

TK=${TKLIST:-$A/logs/tk_feb.txt}   # 可用 TKLIST 覆盖，冒烟时给小表
[ -f "$TK" ] || { echo "missing ticker list $TK"; exit 2; }

run () {  # node gpu chunk month arm extra...
  local NODE=$1 gpu=$2 chunk=$3 mo=$4 arm=$5; shift 5
  env -u SLURM_NNODES -u SLURM_NTASKS -u SLURM_JOB_ID -u SLURMD_NODENAME \
  srun --jobid=$JOB --overlap --exact --cpus-per-task=8 -w $NODE -N1 -n1 \
    --cpu-bind=none --job-name=pipe-$TAG-$arm-g$gpu \
    --export=ALL,DFM_GPU=$gpu,DFM_SCRIPT=dfm_correct_runner.py,XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.15} \
    bash $W --month $mo --n-cond 500 --n-gen 500 \
      --stocks $chunk --index-dir $A/idx --group-size 8 --validate-first 8 \
      --gate-batches 2 --state "$STATE" --t-start 0.80 --n-steps 8 \
      --n-seq $NSEQ --batch-size 4 --corr-batch 4 --skip-existing "$@" \
      --out-template "$OUT/rollouts/${TAG}_{stock}_{month}_$arm.npz" \
      > $OUT/logs/${TAG}_${arm}_${mo}_${NODE#nid}g${gpu}.log 2>&1 &
}

NSLOT=0
for _n in $NODES; do for _g in $(seq 0 $((GPUS_PER_NODE-1))); do
  SLOT_NODE[$NSLOT]=$_n; SLOT_GPU[$NSLOT]=$_g; NSLOT=$((NSLOT+1)); done; done
echo "slots: $NSLOT ($NODES x $GPUS_PER_NODE)"
split -n l/$NSLOT -d -a 2 $TK $OUT/logs/${TAG}_chunk_

for MO in $MONTHS; do
  for ARM in learned random; do
    echo "=== $TAG $MO $ARM $(date -u +%H:%M:%SZ) ==="
    for k in $(seq 0 $((NSLOT-1))); do
      ex=""; [ "$ARM" = random ] && ex="--random-p --random-p-seed 7"
      run ${SLOT_NODE[$k]} ${SLOT_GPU[$k]} \
          $OUT/logs/${TAG}_chunk_$(printf %02d $k) $MO $ARM $ex
    done
    wait
    echo "  npz: $(ls $OUT/rollouts/${TAG}_*_${MO}_${ARM}.npz 2>/dev/null | wc -l)"
  done
done

# ---- scoring：与 A02 同一个估计量，import 而非重写 ----
for MO in $MONTHS; do
  python3 $A/code/a02_across_tickers.py --rollouts $OUT/rollouts --month $MO \
    --prefix "${TAG}_" --out $OUT/out/${TAG}_${MO}.json 2>&1 | tail -9
done
echo "=== PIPELINE $TAG DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
