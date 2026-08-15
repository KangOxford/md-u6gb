#!/bin/bash
# 度量 A/B：值空间度量 vs 逐位 field 度量，**单因子**。
#
# 启动行由 `run_alpha_sweep.sh` 逐字复制，只换 `--state` 与度量相关的标志。
#
# 两条臂：
#   value  artifacts_dv/dv10_val_lr1e4_s0_state.msgpack        13880 步
#   field  artifacts_dv/ladder/dv25_lr1e4_s0u_s14000_state...  14000 步
# 同 lr 1e-4、同 seed 42、同 shard、同 128 窗口留出集、同 tritonOFF。
# 唯一变的是 `DFM_METRIC`（步数差 120 步 = 0.9%，而留出损失在这一段每 2000 步
# 只动 0.003，所以这个差可忽略）。
#
# 预测（跑之前锁死，来自三条独立测量）：
#   值度量必须**特异地**消掉 `size` 的水平异常，而 `event_type`/`price_rel`/
#   `log10_dt`/`direction` 四个字段**不显著变差**。
#   若五个字段一起动，就不是这个机制，是别的东西。
#
# 三条指向 `size` 的独立证据：
#   1. 学到的残差在 size 上比同范数随机方向**差 1.7 倍**，其余四个字段都更好
#   2. 剔除非法箱只让 size 动 0.001（不是非法质量）
#   3. 缩残差（alpha）只在 size 上改善水平，在 event_type 上恶化 4.2 倍
#
# 每张卡 30 个 ticker，约 2.5-3 小时。
set -u
T=/lus/lfs1aip2/projects/public/u6gb/tasks/dfm_largescale_20260813
S0=/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/dfm-post-training-20260801
A=/lus/lfs1aip2/projects/public/u6gb/tasks/ce_orderflow_20260812T200352Z/A02_scale
W=$S0/post_training/dfm/eval/run_eval_node.sh
VAL_STATE=$T/artifacts_dv/dv10_val_lr1e4_s0_state.msgpack
FLD_STATE=$T/artifacts_dv/ladder/dv25_lr1e4_s0u_s14000_state.msgpack
# 值度量的 sidecar 早于「把这三个字段写进产物」的修复，所以显式传；
# runner 拿不到会 fail-closed 而不是猜（错的时间表训练与推理都不报错）。
VAL_FLAGS="--a-target 12 --schedule apow --sched-p 2"
JOB=${JOB:-6014308}; NODE=${NODE:-nid010723}; MO=${MO:-2026-01}; NTK=${NTK:-60}
mkdir -p $T/rollouts_metric $T/logs
head -n $NTK $A/logs/tk_feb.txt > $T/logs/metric_tk.txt
split -n l/2 -d -a 1 $T/logs/metric_tk.txt $T/logs/metric_chunk_

i=0
for SPEC in "value 0" "value 1" "field 0" "field 1"; do
  set -- $SPEC; MET=$1; CH=$2
  if [ "$MET" = value ]; then STATE=$VAL_STATE; XF="$VAL_FLAGS"; else STATE=$FLD_STATE; XF=""; fi
  TAG=${MET}
  env -u SLURM_NNODES -u SLURM_NTASKS -u SLURM_JOB_ID -u SLURMD_NODENAME \
  setsid nohup srun --jobid=$JOB --overlap --exact --cpus-per-task=8 -w $NODE -N1 -n1 \
    --cpu-bind=none --job-name=dfm-met-${TAG}${CH} \
    --export=ALL,DFM_GPU=$i,DFM_SCRIPT=dfm_correct_runner.py,\
XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.30},XLA_FLAGS=--xla_gpu_enable_triton_gemm=false \
    bash $W --month $MO --n-cond 500 --n-gen 500 \
      --stocks $T/logs/metric_chunk_$CH --index-dir $A/idx --group-size 8 \
      --validate-first 8 --gate-batches 2 --state "$STATE" $XF \
      --t-start 0.80 --n-steps 8 --n-seq 8 --batch-size 2 --corr-batch 2 \
      --skip-existing \
      --out-template "$T/rollouts_metric/met_${TAG}_{stock}_{month}_learned.npz" \
    > $T/logs/met_${TAG}$CH.log 2>&1 < /dev/null &
  echo "  $MET chunk$CH -> $NODE gpu$i"; i=$((i+1)); sleep 3
done
echo "=== 度量 A/B launched $(date -u +%H:%M:%SZ) ==="
