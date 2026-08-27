#!/bin/bash
#SBATCH --job-name=glm53-vllm-serve
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --output=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/glm53_flash/logs/serve_%j.out
# GLM-5.3-Flash vLLM 服务的排队后备。与 attach 路径二选一:
# 若 attach 版已先落地且健康,这里直接退出,不双跑。
set -u
T=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/glm53_flash

HOSTF=$T/results/server_host
if [ -f "$HOSTF" ]; then
    H=$(cat "$HOSTF")
    if curl -s --max-time 5 "http://$H:8383/health" >/dev/null 2>&1; then
        echo "[sbatch-serve] $H:8383 已有健康服务,本作业退出"; exit 0
    fi
fi

echo "[sbatch-serve] 在 $(hostname) 启动"
exec bash $T/code/serve_vllm.sh
