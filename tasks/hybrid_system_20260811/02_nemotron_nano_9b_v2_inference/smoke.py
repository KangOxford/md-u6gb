#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
MODEL_REVISION = "6533e8de2c68e4536bf7c411d7a3ce5734111476"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real one-GPU Nemotron Nano 9B v2 generation smoke test.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected exactly one available GPU, found {torch.cuda.device_count()}")
    source_manifest = json.loads((args.model_dir / "download_manifest.json").read_text(encoding="utf-8"))
    if source_manifest.get("resolved_revision") != MODEL_REVISION:
        raise RuntimeError("The linked source snapshot is not the pinned revision")
    compatibility_manifest = json.loads((args.model_dir / "compatibility_manifest.json").read_text(encoding="utf-8"))

    started = time.perf_counter()
    torch.cuda.set_device(0)
    allocator_probe = torch.empty(1, dtype=torch.uint8, device="cuda:0")
    torch.cuda.synchronize()
    del allocator_probe
    torch.cuda.reset_peak_memory_stats(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True)
    config.use_mamba_kernels = False
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        config=config,
        dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
        trust_remote_code=True,
    )
    model.eval()
    messages = [
        {"role": "system", "content": "/no_think"},
        {"role": "user", "content": "In one sentence, state one benefit of combining Mamba-2 and attention layers."},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to("cuda:0")
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    generated_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if not generated_text:
        raise RuntimeError("Generation completed but produced empty text")

    properties = torch.cuda.get_device_properties(0)
    result = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "model_type": config.model_type,
        "use_mamba_kernels": config.use_mamba_kernels,
        "use_cache": config.use_cache,
        "compatibility_modeling_sha256": compatibility_manifest["derived_modeling_sha256"],
        "generated_text": generated_text,
        "input_tokens": int(inputs["input_ids"].shape[-1]),
        "generated_tokens": int(generated_ids.shape[-1]),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(0)),
        "gpu": {
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "visible_device_count": torch.cuda.device_count(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "runtime": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": metadata.version("transformers"),
            "accelerate": metadata.version("accelerate"),
        },
    }
    (args.output_dir / "generated.txt").write_text(generated_text + "\n", encoding="utf-8")
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
