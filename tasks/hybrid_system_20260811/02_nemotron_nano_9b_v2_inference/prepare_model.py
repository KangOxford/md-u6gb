#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


MODEL_ID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
MODEL_REVISION = "6533e8de2c68e4536bf7c411d7a3ce5734111476"
EXPECTED_SAFETENSOR_BYTES = 17_776_492_512


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and verify the pinned Nemotron Nano 9B v2 snapshot.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION, files_metadata=True)
    if info.sha != MODEL_REVISION:
        raise RuntimeError(f"Revision mismatch: requested {MODEL_REVISION}, resolved {info.sha}")

    resolved_path = Path(
        snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, local_dir=args.model_dir, max_workers=4)
    ).resolve()
    records = []
    for path in sorted(resolved_path.rglob("*")):
        if not path.is_file() or ".cache" in path.parts:
            continue
        records.append(
            {"path": str(path.relative_to(resolved_path)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    safetensor_bytes = sum(record["size_bytes"] for record in records if record["path"].endswith(".safetensors"))
    if safetensor_bytes != EXPECTED_SAFETENSOR_BYTES:
        raise RuntimeError(f"Unexpected safetensor bytes: {safetensor_bytes}")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "requested_revision": MODEL_REVISION,
        "resolved_revision": info.sha,
        "model_dir": str(resolved_path),
        "safetensor_bytes": safetensor_bytes,
        "files": records,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model_dir": str(resolved_path), "files": len(records), "safetensor_bytes": safetensor_bytes}))


if __name__ == "__main__":
    main()
