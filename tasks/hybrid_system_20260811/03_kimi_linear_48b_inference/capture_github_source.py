#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import io
import json
import shutil
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


REVISION = "8c1d85eb6b5f8fcefb15758691b0ce50b0827ce3"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    task_dir = Path(__file__).resolve().parent
    destination = task_dir / "source_github"
    manifest_path = task_dir / "github_source_manifest.json"
    if destination.exists() or manifest_path.exists():
        raise RuntimeError("Refusing to overwrite an existing Kimi Linear source capture")
    destination.mkdir()
    url = f"https://github.com/MoonshotAI/Kimi-Linear/archive/{REVISION}.tar.gz"
    archive = urllib.request.urlopen(url, timeout=120).read()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
        files = [member for member in bundle.getmembers() if member.isfile()]
        prefix = files[0].name.split("/", 1)[0] + "/"
        for member in files:
            if not member.name.startswith(prefix):
                raise RuntimeError(f"Unexpected archive member: {member.name}")
            relative = Path(member.name[len(prefix) :])
            if not relative.parts or relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                raise RuntimeError(f"Could not extract {member.name}")
            with target.open("wb") as output:
                shutil.copyfileobj(source, output)
    records = [
        {"path": str(path.relative_to(destination)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    ]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "url": "https://github.com/MoonshotAI/Kimi-Linear",
        "revision": REVISION,
        "archive_url": url,
        "files": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(records), "revision": REVISION}))


if __name__ == "__main__":
    main()
