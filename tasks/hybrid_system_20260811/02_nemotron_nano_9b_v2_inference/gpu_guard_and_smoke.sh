#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 PHYSICAL_GPU MODEL_DIR OUTPUT_DIR [FAST_KERNELS]" >&2
  exit 2
fi
physical_gpu="$1"
model_dir="$2"
output_dir="$3"
fast_kernels="${4:-0}"
task_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! "$physical_gpu" =~ ^[0-3]$ ]]; then
  echo "Invalid physical GPU index: $physical_gpu" >&2
  exit 2
fi
memory_used="$(nvidia-smi -i "$physical_gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
pids="$(nvidia-smi -i "$physical_gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
if [[ -n "$pids" ]]; then
  echo "GPU $physical_gpu has compute processes: $pids" >&2
  exit 70
fi
if (( memory_used > 64 )); then
  echo "GPU $physical_gpu uses ${memory_used} MiB, above the 64 MiB empty-device gate" >&2
  exit 70
fi
mkdir -p "$output_dir"
nvidia-smi -i "$physical_gpu" \
  --query-gpu=timestamp,index,uuid,name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader > "$output_dir/gpu_preflight.csv"
export CUDA_VISIBLE_DEVICES="$physical_gpu"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
extra_args=()
if [[ "$fast_kernels" == "1" ]]; then
  extra_args+=(--fast-kernels)
fi
exec "$task_dir/.venv/bin/python" "$task_dir/smoke.py" \
  --model-dir "$model_dir" \
  --output-dir "$output_dir" \
  "${extra_args[@]}"
