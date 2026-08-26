#!/bin/bash
#SBATCH --job-name=dsv4-smoke-mini
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --partition=workq
#SBATCH --time=00:59:00
#SBATCH --output=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/dsv4_flash_lora/logs/smoke_mini_%j.out

# Smoke B(sbatch 版):mini 4 层模型构建 + 单节点 4 卡 LoRA 微跑(EP4,8 步)
# 用途:分钟级验证 mcore-bridge 加载 / DSA 前向(FlashMLA)/ DSA 反向(cudnn)/
#       hadamard / MoE grouped gemm / LoRA / 优化器 的全链路
set -u
TASK=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/dsv4_flash_lora

echo "[smoke-mini] node=$(hostname) job=$SLURM_JOB_ID"

echo "=== step 1: 构建 mini 模型(若无)==="
srun -N1 -n1 --cpu-bind=none bash -c "source $TASK/code/env_dsv4.sh && python $TASK/code/make_mini4.py" || exit 3

echo "=== step 2: 4 卡 LoRA 微跑 ==="
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29512
export MODEL_DIR_OVERRIDE=/lus/lfs1aip2/projects/public/u6gb/models/DeepSeek-V4-Flash-mini4
export EP_SIZE=4 GBS=8 MBS=1 TRAIN_ITERS=8 DATASET_CAP=200 TAG=smoke_mini MTP_LAYERS=1
srun -N1 -n1 --cpu-bind=none bash $TASK/code/train_lora.sh
rc=$?
echo "SMOKE_MINI_EXIT=$rc"
exit $rc
