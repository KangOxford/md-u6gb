#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a non-overwriting Python 3.13 compatibility view.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    if destination.exists():
        raise RuntimeError(f"Refusing to overwrite {destination}")
    if not (source / "download_manifest.json").is_file():
        raise RuntimeError("Pinned source manifest is missing")
    destination.mkdir(parents=True)

    name = "modeling_kimi.py"
    upstream = (source / name).read_text(encoding="utf-8")
    marker = "    @auto_docstring\n"
    if upstream.count(marker) != 2:
        raise RuntimeError("Expected exactly two auto_docstring decorators")
    derived = upstream.replace(marker, "")
    forced_flash = '''        if getattr(config, "_attn_implementation", None) is not None:
            if config._attn_implementation != "flash_attention_2":
                logger.warning_once(
                    f"Ignoring the provided attention implementation {config._attn_implementation}")
                logger.warning_once("Using flash_attention_2 backend instead.")
                config._attn_implementation = "flash_attention_2"
        else:
            config._attn_implementation = "flash_attention_2"
'''
    preserve_backend = '''        if getattr(config, "_attn_implementation", None) is None:
            config._attn_implementation = "eager"
'''
    if derived.count(forced_flash) != 1:
        raise RuntimeError("Expected exactly one forced FlashAttention backend block")
    derived = derived.replace(forced_flash, preserve_backend)
    linked = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or ".cache" in path.parts or path.name == name:
            continue
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative == Path(".gitattributes"):
            shutil.copyfile(path, target)
        else:
            target.symlink_to(os.path.relpath(path, target.parent))
        linked.append(str(relative))
    (destination / name).write_text(derived, encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "destination": str(destination),
        "patch": "remove two runtime-only auto_docstring decorators and preserve the available eager MLA backend instead of forcing absent FlashAttention",
        "upstream_modeling_sha256": sha256(upstream.encode()),
        "derived_modeling_sha256": sha256(derived.encode()),
        "linked_files": linked,
    }
    (destination / "compatibility_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
