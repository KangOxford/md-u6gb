#!/bin/bash
# Smoke B: mini 4 层模型,单节点 4 卡 LoRA 微跑(EP4,8 步)
# 在登录节点执行本脚本;它 attach 到指定分配的指定节点。
set -u
TASK=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/dsv4_flash_lora
JOBID=${JOBID:-6136391}
NODE=${NODE:-nid010413}

echo "=== 逐卡显存检查 $NODE(规则:>100MiB 即视为被占) ==="
srun --jobid=$JOBID --overlap -N1 -n1 -w $NODE --job-name=dsv4-gpucheck --cpu-bind=none \
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader || exit 2

echo "=== step 1: 构建 mini 模型(若无)==="
srun --jobid=$JOBID --overlap -N1 -n1 -w $NODE --job-name=dsv4-mini-build --cpu-bind=none \
  bash -c "source $TASK/code/env_dsv4.sh && python $TASK/code/make_mini4.py" || exit 3

echo "=== step 2: 4 卡 LoRA 微跑 ==="
export MASTER_ADDR=$NODE MASTER_PORT=29512
export MODEL_DIR_OVERRIDE=/lus/lfs1aip2/projects/public/u6gb/models/DeepSeek-V4-Flash-mini4
export EP_SIZE=4 GBS=8 MBS=1 TRAIN_ITERS=8 DATASET_CAP=200 TAG=smoke_mini MTP_LAYERS=1
srun --jobid=$JOBID --overlap -N1 -n1 -w $NODE --job-name=dsv4-smoke-mini --cpu-bind=none \
  bash $TASK/code/train_lora.sh
rc=$?
echo "SMOKE_MINI_EXIT=$rc"
exit $rc
