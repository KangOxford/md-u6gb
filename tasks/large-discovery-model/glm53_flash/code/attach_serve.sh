#!/bin/bash
# 把 vLLM 服务 attach 到已有分配的一个空节点(4 卡整占)。
# 用法: bash attach_serve.sh [jobid] [node]   缺省 6141106 / nid010815
# 起跑前逐卡验空(1-4 MiB 才算真空;gtop 头行不可作依据)。
set -u
JOBID=${1:-6141106}
NODE=${2:-nid010815}
T=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/glm53_flash

# 逐卡显存验空
BUSY=$(srun --overlap --jobid=$JOBID -w $NODE --ntasks=1 --gres=gpu:4 --job-name=glm53-probe \
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '$1>2000{n++} END{print n+0}')
if [ "$BUSY" != "0" ]; then
    echo "FATAL: $NODE 有 $BUSY 张卡显存 >2GB,不是空节点,拒绝起服" >&2; exit 7
fi
echo "[attach] $NODE 4 卡验空通过,启动 vLLM (job $JOBID)"

exec srun --overlap --jobid=$JOBID -w $NODE \
    --ntasks=1 --ntasks-per-node=1 --gres=gpu:4 --cpus-per-task=64 \
    --job-name=glm53-vllm --cpu-bind=none \
    bash $T/code/serve_vllm.sh
