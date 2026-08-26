#!/bin/bash
# Smoke C: 全量模型,4 节点 16 卡 LoRA 微跑(EP16 —— 与 64 卡完全相同的单卡显存形态)
# 在登录节点执行;attach 到 4 节点分配。EP16 时每卡:专家 ~33G + 非专家 ~11G(BF16)。
set -u
TASK=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/dsv4_flash_lora
JOBID=${JOBID:-6136391}
NODES=${NODES:-nid010329,nid010367,nid010380,nid010413}
FIRST=${NODES%%,*}

echo "=== 逐卡显存检查(4 节点全查)==="
srun --jobid=$JOBID --overlap --nodes=4 --ntasks=4 --ntasks-per-node=1 -w $NODES \
  --job-name=dsv4-gpucheck --cpu-bind=none \
  bash -c 'echo "$(hostname): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr "\n" " ")"' || exit 2

echo "=== 16 卡 LoRA 微跑(全量模型,EP16)==="
export MASTER_ADDR=$FIRST MASTER_PORT=29513
export EP_SIZE=16 GBS=16 MBS=1 TRAIN_ITERS=12 DATASET_CAP=600 TAG=smoke_full16 MTP_LAYERS=1
srun --jobid=$JOBID --overlap --nodes=4 --ntasks=4 --ntasks-per-node=1 -w $NODES \
  --job-name=dsv4-smoke-full16 --cpu-bind=none \
  bash $TASK/code/train_lora.sh
rc=$?
echo "SMOKE_FULL16_EXIT=$rc"
exit $rc
