#!/bin/bash
#SBATCH --job-name=cu130-ldm-test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --partition=workq
#SBATCH --output=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/glm53_flash/logs/cu130_ldm_%j.out
# cu130 镜像起服 + LDM 端到端测试。
# CUDA 13 在 565 驱动上靠 forward compat 跑:容器自带 /usr/local/cuda/compat,
# 但 `--nv` 注入宿主 565 的库,必须显式把 compat 放进 LD_LIBRARY_PATH 才生效
# (不设时容器内 torch.cuda.is_available() 就是 False,已实测)。
set -u
T=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/glm53_flash
L=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/Large-Discovery-Models
NODE=$(hostname)
echo "[cu130] job=$SLURM_JOB_ID node=$NODE"

bash $T/code/serve_vllm_sif_cu130.sh > "$T/logs/cu130_serve_${SLURM_JOB_ID}.log" 2>&1 &
SVC=$!

READY=0
for i in $(seq 1 180); do
    curl -s --max-time 5 -o /dev/null "http://${NODE}:8384/health" && { READY=1; break; }
    kill -0 $SVC 2>/dev/null || { echo "[cu130] FATAL: 服务进程已退出"; exit 3; }
    sleep 10
done
[ "$READY" = "1" ] || { echo "[cu130] FATAL: 30 分钟未就绪"; exit 4; }
echo "[cu130] 服务就绪,用时 $((i*10))s"

# --- 测 1: 普通对话 ---
echo "[cu130] === 测 1 对话 ==="
curl -s --max-time 300 "http://${NODE}:8384/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.3-Flash","messages":[{"role":"user","content":"用一句话说明什么是贝叶斯优化,并算 47*53。"}],"max_tokens":400}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print('答:', d['choices'][0]['message']['content'][-200:].replace(chr(10),' ')); print('tokens:', d['usage']['total_tokens'])"

# --- 测 2: tool calling(LDM 的 operation_tool 走这条路) ---
echo "[cu130] === 测 2 tool calling ==="
curl -s --max-time 300 "http://${NODE}:8384/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.3-Flash","messages":[{"role":"user","content":"Set learning_rate to 0.001 using the tool."}],
       "tools":[{"type":"function","function":{"name":"set_numeric","description":"Set a numeric hyperparameter",
       "parameters":{"type":"object","properties":{"name":{"type":"string"},"value":{"type":"number"}},"required":["name","value"]}}}],
       "max_tokens":400}' \
  | python3 -c "
import json,sys
d=json.load(sys.stdin)
m=d['choices'][0]['message']
tc=m.get('tool_calls')
print('tool_calls:', json.dumps(tc)[:300] if tc else 'None')
print('TOOLCALL_OK' if tc else 'TOOLCALL_FAIL')"

# --- 测 3: 真跑一条 LDM campaign(nanoGPT,短预算) ---
echo "[cu130] === 测 3 LDM campaign ==="
export AUTORESEARCH_CACHE_DIR=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/data/autoresearch_cache
export PYTHONPATH=$L/tasks/nanogpt/scripts
export LLM_BASE_URL=http://${NODE}:8384/v1 LLM_MODEL_NAME=GLM-5.3-Flash LLM_API_KEY=dummy
export TTS_LLM_URL=$LLM_BASE_URL TTS_LLM_MODEL=$LLM_MODEL_NAME TTS_LLM_API_KEY=$LLM_API_KEY
cd $L
set -o pipefail   # 上一版 rc=0 是假的:契约拒绝后退出码被 tail 吃掉了
timeout 4500 /home/u6gb/kangli.u6gb/envs/ldm-nanogpt/bin/python scripts/run_ldm_tts.py \
    $T/code/ng_smoke_cu130.yaml \
    --set "args.run-name=cu130_smoke_${SLURM_JOB_ID}" 2>&1 | tail -25
echo "[cu130] campaign rc=${PIPESTATUS[0]:-$?}"
echo "[cu130] ALL_TESTS_DONE"
