#!/usr/bin/env python
"""Stage every asset the MiniMax-H3 replication needs onto Lustre, once.

Two families of asset are fetched here:

1. The released MiniMax-H3 checkpoint, in its *diffusers* layout (root-level
   ``transformer/``, ``text_encoder/`` ... directories). The repository also ships
   the same weights a second time in MiniMax's native ``FL2VA/`` / ``Ref2VA/``
   layout; fetching both would cost 498 GB for no extra information, so only the
   diffusers layout is taken.

2. A slice of VGGSound (real 10 s mp4 clips carrying audio, plus the label CSV
   that supplies the text condition). MiniMax has not released H3's training
   data, so this is the open stand-in, not a reproduction of their corpus.

Downloads are resumable: ``hf download`` skips blobs already present with a
matching etag, so a re-run after a timeout costs one HEAD per file.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# The diffusers-layout subfolders that `MiniMaxH3ModularPipeline` loads. The
# `transformer_ref/` variant (Ref2VA, a further 66 GB) is deliberately excluded:
# the FL2VA transformer is enough to validate the architecture end to end.
H3_PATTERNS = [
    "model_index.json",
    "transformer/*",
    "text_encoder/*",
    "tokenizer/*",
    "processor/*",
    "vae/*",
    "audio_vae/*",
    "scheduler/*",
    "audio_scheduler/*",
]

VGGSOUND_REPO = "Loie/VGGSound"


def run(cmd: list[str], *, what: str) -> None:
    print(f"\n[fetch] {what}\n[fetch] $ {' '.join(cmd)}", flush=True)
    started = time.time()
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(f"[fetch] FAILED ({what}) rc={proc.returncode}")
    print(f"[fetch] done in {time.time() - started:.0f}s: {what}", flush=True)


def shallow_bytes(path: Path, max_depth: int = 2) -> int:
    """Sum file sizes over a *bounded* walk.

    Deliberately not `du -sh`: on Lustre a recursive stat over a large tree is a
    metadata storm, and this cluster has had group jobs suspended for exactly
    that. The checkpoint tree is ~50 files two levels deep, so a depth-bounded
    walk visits everything that matters and can never wander into a big tree.
    """
    total = 0
    root_depth = len(path.parts)
    for root, dirs, files in os.walk(path):
        if len(Path(root).parts) - root_depth >= max_depth:
            dirs[:] = []
        for name in files:
            try:
                total += os.stat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def fetch_h3(dest: Path, workers: int) -> None:
    cmd = ["hf", "download", "MiniMaxAI/MiniMax-H3", "--local-dir", str(dest), "--max-workers", str(workers)]
    for pattern in H3_PATTERNS:
        cmd += ["--include", pattern]
    run(cmd, what="MiniMax-H3 checkpoint (diffusers layout, ~143 GB)")
    print(f"[fetch] checkpoint size: {shallow_bytes(dest) / 1e9:.1f} GB", flush=True)


def fetch_vggsound(dest: Path, shards: list[int], workers: int) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [
        "hf", "download", VGGSOUND_REPO, "--repo-type", "dataset",
        "--local-dir", str(dest), "--max-workers", str(workers),
        "--include", "vggsound.csv",
    ]
    for shard in shards:
        cmd += ["--include", f"vggsound_{shard:02d}.tar.gz"]
    run(cmd, what=f"VGGSound shards {shards} + label CSV")
    # The tarballs are deliberately *not* unpacked. Each holds ~10k mp4s; exploding
    # them onto Lustre would turn every preprocessing epoch into ~10k MDT opens per
    # shard. The encoder streams the tarball sequentially instead, so the data stays
    # "few and large": one big sequential read in, one latent shard out.
    for shard in shards:
        tarball = dest / f"vggsound_{shard:02d}.tar.gz"
        if tarball.exists():
            print(f"[fetch]   shard {shard:02d}: {tarball.stat().st_size / 1e9:.1f} GB (left packed)", flush=True)
        else:
            print(f"[fetch]   shard {shard:02d}: MISSING", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Task root directory")
    parser.add_argument("--shards", default="00,01", help="Comma-separated VGGSound shard indices")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent hf download workers")
    parser.add_argument("--skip-h3", action="store_true")
    parser.add_argument("--skip-data", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    shards = [int(s) for s in args.shards.split(",") if s.strip()]

    if not args.skip_h3:
        fetch_h3(root / "ckpt" / "h3", args.workers)
    if not args.skip_data:
        fetch_vggsound(root / "data" / "vggsound", shards, args.workers)

    print("\n[fetch] ALL ASSETS STAGED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
