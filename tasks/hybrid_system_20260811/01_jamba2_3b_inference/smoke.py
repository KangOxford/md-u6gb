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


MODEL_ID = "ai21labs/AI21-Jamba2-3B"
MODEL_REVISION = "525c6c8e1d9f5bddedfbdc1dbb0ade2df84230c9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real one-GPU Jamba2 generation smoke test.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected exactly one visible GPU, found {torch.cuda.device_count()}")

    download_manifest_path = args.model_dir / "download_manifest.json"
    download_manifest = json.loads(download_manifest_path.read_text(encoding="utf-8"))
    if download_manifest.get("resolved_revision") != MODEL_REVISION:
        raise RuntimeError("The local model snapshot is not the pinned revision")

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    config = AutoConfig.from_pretrained(args.model_dir, local_files_only=True)
    config.use_mamba_kernels = False
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        config=config,
        dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
    )
    model.eval()

    messages = [
        {"role": "system", "content": "Answer accurately and concisely."},
        {
            "role": "user",
            "content": "In one sentence, why can a hybrid SSM-attention language model benefit from both layer types?",
        },
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda:0")

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
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
