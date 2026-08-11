#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


MODEL_ID = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
MODEL_REVISION = "e1df551a447157d4658b573f9a695d57658590e9"
EXPECTED_SAFETENSOR_BYTES = 98_248_224_120


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(32 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION, files_metadata=True)
    if info.sha != MODEL_REVISION:
        raise RuntimeError(f"Revision mismatch: {info.sha}")
    root = Path(snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, local_dir=args.model_dir, max_workers=8)).resolve()
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and ".cache" not in path.parts:
            records.append({"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    safetensor_bytes = sum(record["size_bytes"] for record in records if record["path"].endswith(".safetensors"))
    if safetensor_bytes != EXPECTED_SAFETENSOR_BYTES:
        raise RuntimeError(f"Unexpected safetensor bytes: {safetensor_bytes}")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "requested_revision": MODEL_REVISION,
        "resolved_revision": info.sha,
        "model_dir": str(root),
        "safetensor_bytes": safetensor_bytes,
        "files": records,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model_dir": str(root), "files": len(records), "safetensor_bytes": safetensor_bytes}))


if __name__ == "__main__":
    main()
