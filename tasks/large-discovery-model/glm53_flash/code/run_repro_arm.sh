#!/bin/bash
# 跑一组复现 campaign(A 或 B)。必须在计算节点、独占 GPU 上跑:
# 评测是 300 秒定时训练,GPU 被别人抢 ⇒ 同样 300 秒跑更少 token ⇒ bpb 变差,
# 指标直接被污染。所以每组一张卡,不共享。
#
# 用法: ARM=A SEED=1 GLM_HOST=nid011109 bash run_repro_arm.sh
set -u -o pipefail
L=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/Large-Discovery-Models
T=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/glm53_flash
ARM=${ARM:?需要 ARM=A 或 B}
SEED=${SEED:-1}
GLM_HOST=${GLM_HOST:-nid011109}

case "$ARM" in
  A) CFG=$T/code/ng_A_ldm.yaml ;;
  B) CFG=$T/code/ng_B_llmonly.yaml ;;
  *) echo "FATAL: ARM 只能是 A 或 B" >&2; exit 2 ;;
esac

export AUTORESEARCH_CACHE_DIR=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/data/autoresearch_cache
export PYTHONPATH=$L/tasks/nanogpt/scripts
export LLM_BASE_URL=http://${GLM_HOST}:8383/v1
export LLM_MODEL_NAME=GLM-5.3-Flash
export LLM_API_KEY=dummy
export TTS_LLM_URL=$LLM_BASE_URL TTS_LLM_MODEL=$LLM_MODEL_NAME TTS_LLM_API_KEY=$LLM_API_KEY
# 采集训练数据(顺带产出 SFT 语料,与复现同一批轨迹)
export LDM_DATA_COLLECTION_ENABLED=1
export LDM_DATA_COLLECTION_DIR=$L/data/generated/repro_arm${ARM}_s${SEED}

echo "[arm$ARM] host=$(hostname) seed=$SEED glm=$GLM_HOST gpu=${CUDA_VISIBLE_DEVICES:-all}"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -4
curl -s --max-time 10 -o /dev/null -w "[arm$ARM] glm_health=%{http_code}\n" http://${GLM_HOST}:8383/health

cd $L
exec /home/u6gb/kangli.u6gb/envs/ldm-nanogpt/bin/python scripts/run_ldm_tts.py "$CFG" \
    --set "args.run-name=repro_arm${ARM}_s${SEED}"
