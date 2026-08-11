#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


RECIPES = {
    "vllm_kimi_k3.html": "https://recipes.vllm.ai/moonshotai/Kimi-K3",
    "sglang_kimi_k3.html": "https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3",
}


def main() -> None:
    task_dir = Path(__file__).resolve().parent
    destination = task_dir / "source_recipes"
    manifest_path = task_dir / "recipe_manifest.json"
    if destination.exists() or manifest_path.exists():
        raise RuntimeError("Refusing to overwrite an existing recipe capture")
    destination.mkdir()
    records = []
    for filename, url in RECIPES.items():
        request = urllib.request.Request(url, headers={"User-Agent": "hybrid-source-audit/1.0"})
        data = urllib.request.urlopen(request, timeout=120).read()
        path = destination / filename
        path.write_bytes(data)
        records.append(
            {"path": filename, "url": url, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        )
    manifest_path.write_text(
        json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(), "files": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(records))


if __name__ == "__main__":
    main()
