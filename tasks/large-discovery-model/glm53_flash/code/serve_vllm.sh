#!/bin/bash
# GLM-5.3-Flash vLLM 服务 —— 在计算节点上由 srun 执行
# 显存账(GH200 95.6G/卡):FP8 权重 306GiB / TP4 = 76.5G/卡,util 0.92 ⇒ ~88G 预算,
# 剩 ~11G 给 KV + CUDA graph。KDA 线性层状态为常数、MLA 层 kv_lora_rank=512 压缩,
# 64k 上下文的 KV 占用很小,先按 65536 起。
# 第一轮点亮不带 MTP 投机解码(--speculative-config),最少活动部件;稳了再加。
set -u
source /home/u6gb/kangli.u6gb/envs/glm53-vllm/bin/activate

MODEL=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/models/GLM-5.3-Flash
PORT=${PORT:-8383}
MAXLEN=${MAXLEN:-65536}

# 编译缓存放节点本地盘,不打 Lustre(/local/scratch 计算节点不可写,沿 dsv4 用 TMPDIR)
export TRITON_CACHE_DIR=${TMPDIR:-/tmp}/glm53_triton_${USER}
export VLLM_CACHE_ROOT=${TMPDIR:-/tmp}/glm53_vllm_cache_${USER}
mkdir -p "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"

echo "[serve] host=$(hostname) port=$PORT maxlen=$MAXLEN tp=4"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
# 记下服务落点,test_chat.sh 与 sbatch 防双跑检查都读这个文件
hostname > /lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/glm53_flash/results/server_host

exec vllm serve "$MODEL" \
    --served-model-name GLM-5.3-Flash \
    --tensor-parallel-size 4 \
    --kv-cache-dtype fp8 \
    --max-model-len "$MAXLEN" \
    --gpu-memory-utilization 0.92 \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --host 0.0.0.0 \
    --port "$PORT"
