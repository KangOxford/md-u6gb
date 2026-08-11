#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import platform
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
MODEL_REVISION = "e1df551a447157d4658b573f9a695d57658590e9"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError(f"Expected exactly two visible GPUs, found {torch.cuda.device_count()}")
    manifest = json.loads((args.model_dir / "download_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("resolved_revision") != MODEL_REVISION:
        raise RuntimeError("Pinned revision mismatch")
    for device in range(2):
        probe = torch.empty(1, dtype=torch.uint8, device=f"cuda:{device}")
        torch.cuda.synchronize(device)
        del probe
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=torch.bfloat16,
        device_map="balanced",
        max_memory={0: "88GiB", 1: "88GiB", "cpu": "120GiB"},
        local_files_only=True,
        trust_remote_code=True,
    )
    device_map = getattr(model, "hf_device_map", {})
    assigned = {str(value) for value in device_map.values()}
    if any(value in {"cpu", "disk"} for value in assigned):
        raise RuntimeError(f"Unexpected CPU/disk offload: {device_map}")
    messages = [{"role": "user", "content": "In one sentence, explain why hybrid linear and full attention can be useful."}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
    ).to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    for device in range(2):
        torch.cuda.synchronize(device)
    generated_ids = output_ids[0, inputs.shape[-1] :]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if not generated_text:
        raise RuntimeError("Empty generation")
    result = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "generated_text": generated_text,
        "input_tokens": int(inputs.shape[-1]),
        "generated_tokens": int(generated_ids.shape[-1]),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": [int(torch.cuda.max_memory_allocated(i)) for i in range(2)],
        "gpu_names": [torch.cuda.get_device_properties(i).name for i in range(2)],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_map_counts": dict(Counter(str(value) for value in device_map.values())),
        "runtime": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": metadata.version("transformers"),
            "fla_core": metadata.version("fla-core"),
        },
    }
    (args.output_dir / "generated.txt").write_text(generated_text + "\n", encoding="utf-8")
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
