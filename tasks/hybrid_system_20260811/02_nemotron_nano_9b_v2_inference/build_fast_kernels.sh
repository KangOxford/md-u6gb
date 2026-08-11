#!/usr/bin/env bash
set -euo pipefail

task_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
uv_bin="${HYBRID_UV_BIN:-/projects/public/u6gb/.local/bin/uv}"
export UV_HTTP_TIMEOUT="${HYBRID_UV_HTTP_TIMEOUT:-300}"
export MAX_JOBS="${HYBRID_MAX_JOBS:-8}"

if ! type module >/dev/null 2>&1; then
  source /etc/profile.d/modules.sh
fi
module load cuda/12.6
if [[ ! -x "$task_dir/.venv/bin/python" || ! -x "$uv_bin" ]]; then
  echo "Missing task runtime or uv" >&2
  exit 2
fi
nvcc --version

"$uv_bin" pip install \
  --python "$task_dir/.venv/bin/python" \
  ninja==1.13.0 \
  einops==0.8.1 \
  wheel==0.46.1

"$uv_bin" pip install \
  --python "$task_dir/.venv/bin/python" \
  --no-build-isolation \
  causal-conv1d==1.6.2.post1

"$uv_bin" pip install \
  --python "$task_dir/.venv/bin/python" \
  --no-build-isolation \
  mamba-ssm==2.3.2.post1

"$task_dir/.venv/bin/python" - <<'PY'
import causal_conv1d
import mamba_ssm

print({"causal_conv1d": causal_conv1d.__version__, "mamba_ssm": mamba_ssm.__version__})
PY
