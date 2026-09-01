#!/bin/bash
#SBATCH --job-name=ldmrl-p0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --partition=workq
#SBATCH --output=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/ldm_rl/logs/p0_%j.out
#
# P0 关卡:1.5B GRPO 冒烟(真 reward,真 docking)。
#
# 为什么是 sbatch 而不是 attach:提交时两个自有分配的 16+16 张卡里只剩 1 张空的,
# 其余都在跑别的战线(利用率 88-100%)。attach-first 的前提「有空卡」此刻不成立。
#
# 为什么 --exclusive 且要 4 卡:run_train_real_slime.sh 内部按
# CUDA_VISIBLE_DEVICES=1,2,3 分卡(actor 1 张 + sglang rollout 2 张),
# 那是 slime 自己的分配方式。申请 4 张时 step 内看到 0,1,2,3(已实测),
# 1,2,3 有效;只申请 3 张的话设备 3 不存在,而那类错误不报错,
# 只会让那一路指向不存在的卡。
set -u
T=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/ldm_rl
source "$T/code/site_env.sh"

export EPISODES=$REPO_ROOT/rl_episodes_sm_acqmax.jsonl
export SAVE=$REPO_ROOT/rl/qwen2.5-1.5B_slime_p0_smoke

echo "[p0] job=$SLURM_JOB_ID 节点=$(hostname)"
echo "[p0] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<未设>}"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "[p0] EPISODES=$EPISODES"
echo "[p0] SAVE=$SAVE"

srun --ntasks=1 --gres=gpu:4 --cpus-per-task=128 --cpu-bind=none \
     --job-name=ldmrl-p0-train \
     bash -lc "module load gcc-native/12.3 2>/dev/null || true
               source $T/code/site_env.sh >/dev/null 2>&1
               export EPISODES=$REPO_ROOT/rl_episodes_sm_acqmax.jsonl
               export SAVE=$REPO_ROOT/rl/qwen2.5-1.5B_slime_p0_smoke
               bash \$LAUNCH/run_train_real_slime.sh"
echo "[p0] 结束 rc=$?"
