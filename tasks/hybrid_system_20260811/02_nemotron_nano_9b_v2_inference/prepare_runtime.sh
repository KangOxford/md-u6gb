#!/usr/bin/env bash
set -euo pipefail

task_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base_python="${HYBRID_BASE_PYTHON:-/home/u6gb/kangli.u6gb/miniforge3/bin/python3}"
uv_bin="${HYBRID_UV_BIN:-/projects/public/u6gb/.local/bin/uv}"
torch_index="${HYBRID_TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
export UV_HTTP_TIMEOUT="${HYBRID_UV_HTTP_TIMEOUT:-300}"

if [[ ! -x "$base_python" || ! -x "$uv_bin" ]]; then
  echo "Missing executable base Python or uv" >&2
  exit 2
fi
if [[ ! -x "$task_dir/.venv/bin/python" ]]; then
  "$uv_bin" venv --python "$base_python" --system-site-packages "$task_dir/.venv"
fi

"$uv_bin" pip install \
  --python "$task_dir/.venv/bin/python" \
  --index "$torch_index" \
  --index-strategy unsafe-best-match \
  --requirement "$task_dir/requirements.txt"

"$task_dir/.venv/bin/python" "$task_dir/verify_runtime.py" \
  --uv "$uv_bin" \
  --python "$task_dir/.venv/bin/python"

"$task_dir/.venv/bin/python" - <<'PY'
import importlib.metadata as metadata
import torch

packages = ["torch", "transformers", "accelerate", "huggingface-hub", "safetensors", "tokenizers"]
print({name: metadata.version(name) for name in packages})
print({"torch_cuda_build": torch.version.cuda, "cuda_built": torch.backends.cuda.is_built()})
PY
