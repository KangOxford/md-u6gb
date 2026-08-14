#!/usr/bin/env python
"""Check the staged H3 checkpoint is complete, without loading a single weight.

A directory-size assertion catches "the download fetched one file"; it does not
catch "every shard is present but one of them is truncated", which shows up much
later as a confusing load error. Each safetensors index names its shards and states
`metadata.total_size`, so both questions are answerable from the filesystem:

* is every shard the index references actually on disk, and
* do the shard sizes add up to what the index says they should.

Reads only the JSON indexes and `os.stat`, so it runs anywhere in seconds. Exits
non-zero on the first component that fails, which makes it usable as a gate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# subfolder -> (index filename or None, expected minimum bytes)
# The minimums are the published shard arithmetic: the transformer is 33 B params at
# 2 bytes each, the conditioner is Qwen3-VL-32B, the video VAE is the asymmetric
# CNN-encoder/ViT-decoder one.
COMPONENTS = {
    "transformer": ("diffusion_pytorch_model.safetensors.index.json", 60e9),
    "text_encoder": ("model.safetensors.index.json", 60e9),
    "vae": ("diffusion_pytorch_model.safetensors.index.json", 9e9),
    "audio_vae": (None, 0.5e9),
    "tokenizer": (None, 0),
    "processor": (None, 0),
    "scheduler": (None, 0),
    "audio_scheduler": (None, 0),
}


def check_component(root: Path, name: str, index_name: str | None, min_bytes: float) -> list[str]:
    problems: list[str] = []
    folder = root / name
    if not folder.is_dir():
        return [f"{name}: directory missing"]

    present = {p.name: p.stat().st_size for p in folder.iterdir() if p.is_file()}
    total = sum(present.values())

    if index_name:
        index_path = folder / index_name
        if not index_path.exists():
            return [f"{name}: {index_name} missing"]
        index = json.loads(index_path.read_text())
        shards = sorted(set(index["weight_map"].values()))
        missing = [s for s in shards if s not in present]
        if missing:
            problems.append(f"{name}: {len(missing)}/{len(shards)} shards missing, e.g. {missing[:2]}")
        shard_bytes = sum(present.get(s, 0) for s in shards)
        declared = int(index.get("metadata", {}).get("total_size", 0))
        if declared and abs(shard_bytes - declared) > 1e6:
            problems.append(
                f"{name}: shards total {shard_bytes/1e9:.2f} GB but the index declares "
                f"{declared/1e9:.2f} GB -- a shard is truncated"
            )
        print(f"  {name:16s} {len(shards):3d} shards  {shard_bytes/1e9:7.2f} GB "
              f"(index says {declared/1e9:.2f})", flush=True)
    else:
        print(f"  {name:16s} {len(present):3d} files   {total/1e9:7.2f} GB", flush=True)

    if total < min_bytes:
        problems.append(f"{name}: {total/1e9:.2f} GB is below the expected {min_bytes/1e9:.1f} GB")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(args.ckpt)
    print(f"[verify] {root}", flush=True)
    problems: list[str] = []
    for name, (index_name, min_bytes) in COMPONENTS.items():
        problems += check_component(root, name, index_name, min_bytes)

    if not (root / "model_index.json").exists():
        problems.append("model_index.json missing (the pipeline entry point)")

    total = sum(
        os.stat(os.path.join(dirpath, f)).st_size
        for name in COMPONENTS
        for dirpath, _dirs, files in os.walk(root / name)
        for f in files
    ) if root.is_dir() else 0
    print(f"[verify] total {total/1e9:.1f} GB", flush=True)

    result = {"root": str(root), "total_gb": total / 1e9, "problems": problems,
              "ok": not problems}
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))

    if problems:
        print("[verify] FAILED:", flush=True)
        for p in problems:
            print(f"  - {p}", flush=True)
        return 1
    print("[verify] checkpoint complete and self-consistent", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
