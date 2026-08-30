#!/bin/bash
# GLM-5.3-Flash vLLM 服务 —— 官方镜像 vllm/vllm-openai:glm53-flash-arm64-cu129
# 经 apptainer 运行(PR#53906 未合并期,官方指定 docker 路线;cu129 与 565 驱动
# 同 major 前向兼容,venv 侧已实测)。在计算节点上由 srun 执行。
set -u
T=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/glm53_flash
MODEL=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/models/GLM-5.3-Flash
SIF=$T/images/vllm-glm53-arm64-cu129.sif
PORT=${PORT:-8383}
MAXLEN=${MAXLEN:-65536}
[ -f "$SIF" ] || { echo "FATAL: SIF 不存在: $SIF" >&2; exit 6; }

echo "[serve-sif] host=$(hostname) port=$PORT maxlen=$MAXLEN tp=4"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
hostname > $T/results/server_host

# 编译/下载类缓存全部指向节点本地 /tmp(apptainer 默认绑定 /tmp)
exec apptainer exec --nv \
    --bind /lus/lfs1aip2:/lus/lfs1aip2 \
    --env VLLM_CACHE_ROOT=/tmp/glm53_vllm_cache_$USER \
    --env TRITON_CACHE_DIR=/tmp/glm53_triton_$USER \
    --env HF_HUB_OFFLINE=1 \
    "$SIF" \
    vllm serve "$MODEL" \
      --served-model-name GLM-5.3-Flash \
      --tensor-parallel-size 4 \
      --kv-cache-dtype fp8 \
      --max-model-len "$MAXLEN" \
      --gpu-memory-utilization 0.92 \
      --host 0.0.0.0 \
      --port "$PORT"
