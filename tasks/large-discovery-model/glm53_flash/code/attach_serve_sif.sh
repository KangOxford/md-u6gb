#!/bin/bash
# SIF 版起服 attach:逐卡验空后把 apptainer-vLLM 服务放上指定节点
# 用法: bash attach_serve_sif.sh <jobid> <node>
set -u
JOBID=${1:?需要 jobid}
NODE=${2:?需要节点名}
T=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/glm53_flash

BUSY=$(srun --overlap --jobid=$JOBID -w $NODE --ntasks=1 --gres=gpu:4 --job-name=glm53-probe \
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '$1>2000{n++} END{print n+0}')
if [ "$BUSY" != "0" ]; then
    echo "FATAL: $NODE 有 $BUSY 张卡显存 >2GB,不是空节点,拒绝起服" >&2; exit 7
fi
echo "[attach-sif] $NODE 4 卡验空通过,启动 apptainer vLLM (job $JOBID)"

exec srun --overlap --jobid=$JOBID -w $NODE \
    --ntasks=1 --ntasks-per-node=1 --gres=gpu:4 --cpus-per-task=64 \
    --job-name=glm53-vllm --cpu-bind=none \
    bash $T/code/serve_vllm_sif.sh
