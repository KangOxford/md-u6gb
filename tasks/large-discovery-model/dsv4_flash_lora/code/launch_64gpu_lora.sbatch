#!/bin/bash
#SBATCH --job-name=dsv4-lora-64g
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --partition=workq
#SBATCH --time=06:00:00
#SBATCH --output=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/dsv4_flash_lora/logs/dsv4_64g_%j.out

# DeepSeek-V4-Flash LoRA 64 卡(16 节点 × 4 GH200)正式跑通
# 并行结构: TP1 / PP1 / EP16 / DP64(专家侧 expert-DP=4),GBS=64,MBS=1
set -u
TASK=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/dsv4_flash_lora

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29511
export EP_SIZE=16 GBS=64 MBS=1 DATASET_CAP=1000 TAG=lora64_${SLURM_JOB_ID} MTP_LAYERS=1
export TRAIN_ITERS=   # 空 = 完整 1 epoch(3000 样本 ÷ GBS64 ≈ 46 步)

echo "[64g] master=$MASTER_ADDR nodes=$SLURM_NNODES nodelist=$SLURM_JOB_NODELIST"
srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --cpus-per-task=72 --cpu-bind=none \
  bash $TASK/code/train_lora.sh
rc=$?
echo "[64g] EXIT=$rc"
exit $rc
