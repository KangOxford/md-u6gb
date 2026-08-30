#!/bin/bash
#SBATCH --job-name=ldm-repro-ab
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --partition=workq
#SBATCH --output=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/glm53_flash/logs/repro_clean_%j.out
# LDM v0.1 C1 复现的正式跑:3 种子 × 2 组 = 6 条 campaign,每条独占一张 GPU。
# --exclusive 是必需的:评测按 300 秒定时,与人共卡会让同样时间跑更少 token,
# 直接把「这组更差」写进结果。试跑那一对就是在共卡下跑的,只用来验机制不出数。
set -u
T=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/glm53_flash
GLM_HOST=${GLM_HOST:-nid011109}

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
echo "[repro] nodes=${NODES[*]} glm=$GLM_HOST job=$SLURM_JOB_ID"
curl -s --max-time 10 -o /dev/null -w "[repro] glm_health=%{http_code}\n" "http://${GLM_HOST}:8383/health"

# 布局:节点 0 放 A1/B1/A2/B2,节点 1 放 A3/B3 —— 同种子的两组落同一节点,
# 让「节点差异」这个混杂因素在成对比较里被抵消掉。
launch() {  # $1=node $2=gpu $3=arm $4=seed
    srun --exclusive --nodes=1 --ntasks=1 -w "$1" --gres=gpu:1 \
        --cpus-per-task=64 --job-name="repro-$3$4" --cpu-bind=none \
        bash -c "CUDA_VISIBLE_DEVICES=$2 ARM=$3 SEED=$4 GLM_HOST=$GLM_HOST bash $T/code/run_repro_arm.sh" \
        > "$T/logs/clean_arm$3_s$4_${SLURM_JOB_ID}.log" 2>&1 &
    echo "[repro] launched arm$3 seed$4 on $1 gpu$2"
}

launch "${NODES[0]}" 0 A 1
launch "${NODES[0]}" 1 B 1
launch "${NODES[0]}" 2 A 2
launch "${NODES[0]}" 3 B 2
launch "${NODES[1]}" 0 A 3
launch "${NODES[1]}" 1 B 3

wait
echo "[repro] ALL_ARMS_DONE rc=$?"
