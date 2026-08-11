#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import io
import json
import shutil
import tarfile
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


GITHUB_REVISION = "3cb39dfd32e51c3328e2e4b4af21341247d06c43"
MODEL_ID = "moonshotai/Kimi-K3"
MODEL_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
EXPECTED_WEIGHT_BYTES = 1_560_936_091_448
EXPECTED_WEIGHT_SHARDS = 96


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def records(root: Path) -> list[dict[str, object]]:
    return [
        {"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def main() -> None:
    task_dir = Path(__file__).resolve().parent
    github_dir = task_dir / "source_github"
    model_dir = task_dir / "source_model_metadata"
    manifest_path = task_dir / "source_manifest.json"
    for target in (github_dir, model_dir, manifest_path):
        if target.exists():
            raise RuntimeError(f"Refusing to overwrite existing target: {target}")
    github_dir.mkdir()
    model_dir.mkdir()

    archive_url = f"https://github.com/MoonshotAI/Kimi-K3/archive/{GITHUB_REVISION}.tar.gz"
    archive = urllib.request.urlopen(archive_url, timeout=120).read()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
        members = [member for member in bundle.getmembers() if member.isfile()]
        prefix = members[0].name.split("/", 1)[0] + "/"
        for member in members:
            if not member.name.startswith(prefix):
                raise RuntimeError(f"Unexpected archive member: {member.name}")
            relative = Path(member.name[len(prefix) :])
            if not relative.parts or relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
            target = github_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                raise RuntimeError(f"Could not extract {member.name}")
            with target.open("wb") as output:
                shutil.copyfileobj(source, output)

    api_url = f"https://huggingface.co/api/models/{MODEL_ID}/revision/{MODEL_REVISION}?blobs=true"
    model_info = json.load(urllib.request.urlopen(api_url, timeout=120))
    if model_info.get("sha") != MODEL_REVISION:
        raise RuntimeError(f"Model revision mismatch: {model_info.get('sha')}")
    weight_files = []
    downloaded = []
    for sibling in model_info.get("siblings", []):
        name = sibling["rfilename"]
        if name.endswith(".safetensors"):
            weight_files.append(sibling)
            continue
        target = model_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        quoted_name = urllib.parse.quote(name, safe="/")
        url = f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{quoted_name}"
        with urllib.request.urlopen(url, timeout=120) as response, target.open("wb") as output:
            shutil.copyfileobj(response, output)
        downloaded.append(name)
    if len(weight_files) != EXPECTED_WEIGHT_SHARDS:
        raise RuntimeError(f"Expected {EXPECTED_WEIGHT_SHARDS} weight shards, found {len(weight_files)}")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "github": {
            "url": "https://github.com/MoonshotAI/Kimi-K3",
            "revision": GITHUB_REVISION,
            "archive_url": archive_url,
            "files": records(github_dir),
        },
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "metadata_files_downloaded": downloaded,
            "files": records(model_dir),
            "weight_download": "skipped_by_capacity_gate",
            "weight_shards": EXPECTED_WEIGHT_SHARDS,
            "weight_bytes": EXPECTED_WEIGHT_BYTES,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"github_files": len(manifest["github"]["files"]), "model_files": len(manifest["model"]["files"])}))


if __name__ == "__main__":
    main()
