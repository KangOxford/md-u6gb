#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then echo "Usage: $0 GPU_A GPU_B MODEL_DIR OUTPUT_DIR" >&2; exit 2; fi
gpu_a="$1"; gpu_b="$2"; model_dir="$3"; output_dir="$4"
task_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! "$gpu_a" =~ ^[0-3]$ || ! "$gpu_b" =~ ^[0-3]$ || "$gpu_a" == "$gpu_b" ]]; then echo "Invalid GPUs" >&2; exit 2; fi
mkdir -p "$output_dir"
for gpu in "$gpu_a" "$gpu_b"; do
  memory="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  pids="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -n "$pids" || "$memory" -gt 64 ]]; then echo "GPU $gpu is occupied: memory=${memory}MiB pids=$pids" >&2; exit 70; fi
  nvidia-smi -i "$gpu" --query-gpu=timestamp,index,uuid,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader >> "$output_dir/gpu_preflight.csv"
done
export CUDA_VISIBLE_DEVICES="$gpu_a,$gpu_b"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True
exec "$task_dir/.venv/bin/python" "$task_dir/smoke.py" --model-dir "$model_dir" --output-dir "$output_dir"
