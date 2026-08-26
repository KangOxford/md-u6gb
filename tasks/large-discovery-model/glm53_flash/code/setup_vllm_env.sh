#!/bin/bash
# vLLM 推理 venv(独立纯净 venv,不动 slime-train)。
# vLLM 0.28.0 有官方 cp38-abi3 manylinux aarch64 轮子,自带 pinned torch 与
# nvidia CUDA 运行库——GH200 上零编译。venv 放 HOME(u6gb 项目 inode 只剩 ~44 万,
# 一个 venv 几万个文件,不能落在项目配额里)。
set -u
BASE_PY=/lus/lfs1aip2/projects/public/u6gb/tasks/openmle_rsi_20260812/envs/slime-train/bin/python
VENV=/home/u6gb/kangli.u6gb/envs/glm53-vllm

[ -d "$VENV" ] || "$BASE_PY" -m venv "$VENV"
source "$VENV/bin/activate"
pip install -q -U pip
# flashinfer ≥0.6.17 是 GLM-5.3-Flash 部署页的硬性要求,与 vllm 一次性联合解析
pip install vllm==0.28.0 flashinfer-python 2>&1 | tail -8

python - <<'EOF'
import importlib.metadata as im
for p in ['vllm','torch','transformers','flashinfer-python','xformers']:
    try: print(f"{p:22s}", im.version(p))
    except Exception: print(f"{p:22s}", 'ABSENT')
EOF
echo "[venv] DONE"
