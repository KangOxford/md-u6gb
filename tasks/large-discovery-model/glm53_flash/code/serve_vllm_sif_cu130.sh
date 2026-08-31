#!/bin/bash
# GLM-5.3-Flash vLLM 服务 —— cu130 镜像 + CUDA 13 forward compatibility
#
# 为什么要 compat:内核驱动是 565(CUDA 12.7),cu130 的应用要求 ≥580。
# NVIDIA 对数据中心卡(GH200=H100)提供 forward compat:装一份用户态 libcuda.so,
# 通过 LD_LIBRARY_PATH 让应用加载它而不是系统那份。已实测 torch 2.13.0+cu130
# 在 compat 下 cuda_available=True 且矩阵乘正确。
#
# compat 来源的优先级:
#   1) 容器自带 /usr/local/cuda/compat  ← NGC 系镜像通常有,优先用它(版本与镜像配套)
#   2) 宿主 ~/envs/cuda13-compat        ← 容器没有时绑进去兜底
set -u
T=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/glm53_flash
MODEL=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/models/GLM-5.3-Flash
SIF=${SIF:-$T/images/vllm-glm53-arm64-cu130.sif}
HOST_COMPAT=/home/u6gb/kangli.u6gb/envs/cuda13-compat/usr/local/cuda-13.0/compat
PORT=${PORT:-8384}          # 与 cu129 版(8383)错开,可并存对比
MAXLEN=${MAXLEN:-65536}
[ -f "$SIF" ] || { echo "FATAL: SIF 不存在: $SIF" >&2; exit 6; }

# 容器里有没有自带 compat
if apptainer exec "$SIF" test -d /usr/local/cuda/compat 2>/dev/null; then
    COMPAT_PATH=/usr/local/cuda/compat
    BIND_EXTRA=""
    echo "[serve130] 用容器自带 compat"
else
    COMPAT_PATH=/host_compat
    BIND_EXTRA="--bind $HOST_COMPAT:/host_compat:ro"
    echo "[serve130] 容器无 compat,绑宿主 $HOST_COMPAT"
fi

echo "[serve130] host=$(hostname) port=$PORT compat=$COMPAT_PATH"
nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader | head -4
hostname > $T/results/server_host_cu130

exec apptainer exec --nv \
    --bind /lus/lfs1aip2:/lus/lfs1aip2 \
    $BIND_EXTRA \
    --env LD_LIBRARY_PATH=$COMPAT_PATH:/usr/local/cuda/lib64 \
    --env VLLM_CACHE_ROOT=/tmp/glm53_vllm_cache130_$USER \
    --env TRITON_CACHE_DIR=/tmp/glm53_triton130_$USER \
    --env TMPDIR=/tmp \
    --env DG_JIT_CACHE_DIR=/tmp/glm53_dgjit130_$USER \
    --env HF_HUB_OFFLINE=1 \
    "$SIF" \
    vllm serve "$MODEL" \
      --served-model-name GLM-5.3-Flash \
      --tensor-parallel-size 4 \
      --max-model-len "$MAXLEN" \
      --gpu-memory-utilization 0.88 \
      --max-num-seqs 256 \
      --tool-call-parser glm47 \
      --reasoning-parser glm45 \
      --enable-auto-tool-choice \
      --host 0.0.0.0 \
      --port "$PORT"
