#!/bin/bash
# vLLM 推理 venv(独立纯净 venv,不动 slime-train)。venv 放 HOME
# (u6gb 项目 inode 只剩 ~44 万,venv 几万个文件不能落项目配额)。
#
# v2(cu129):节点驱动 565.57 = CUDA 12.7。PyPI 默认的 vllm 0.28 轮子是 cu130 构建,
# major 版本跨代,torch.cuda.is_available()=False(已实测)。改用 GitHub release 的
# +cu129 aarch64 变体轮子 + pytorch cu129 索引 —— CUDA 12.x 内小版本前向兼容,
# dsv4 的 cu128 栈在同驱动上已生产验证。
set -u
BASE_PY=/lus/lfs1aip2/projects/public/u6gb/tasks/openmle_rsi_20260812/envs/slime-train/bin/python
VENV=/home/u6gb/kangli.u6gb/envs/glm53-vllm

# 旧 cu130 版本整目录挪走(删除一律改改名)
if [ -d "$VENV" ] && [ "$(cat "$VENV/.variant" 2>/dev/null)" != "cu129" ]; then
    mv "$VENV" "${VENV}_cu130_deprecated_$(date -u +%Y%m%dT%H%MZ)"
fi
[ -d "$VENV" ] || "$BASE_PY" -m venv "$VENV"
source "$VENV/bin/activate"
pip install -q -U pip

pip install \
    "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.28.0/vllm-0.28.0+cu129-cp38-abi3-manylinux_2_28_aarch64.whl" \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    2>&1 | tail -8
echo cu129 > "$VENV/.variant"

python - <<'EOF'
import importlib.metadata as im
for p in ['vllm','torch','transformers','flashinfer-python']:
    try: print(f"{p:22s}", im.version(p))
    except Exception: print(f"{p:22s}", 'ABSENT')
EOF
echo "[venv] DONE"
