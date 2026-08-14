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
    # `--include` takes *many* patterns after one flag (nargs='+'). Repeating the flag
    # does not accumulate -- the last occurrence wins and every earlier one is
    # silently discarded. Job 5998320 lost 143 GB to this: nine `--include` flags
    # fetched exactly one file, the last pattern's, and `hf` exited 0 while doing it.
    cmd = ["hf", "download", "MiniMaxAI/MiniMax-H3", "--local-dir", str(dest),
           "--max-workers", str(workers), "--include", *H3_PATTERNS]
    run(cmd, what="MiniMax-H3 checkpoint (diffusers layout, ~143 GB)")

    # The root JSONs are fetched one at a time rather than trusted to the pattern
    # list. `model_index.json` was the *first* pattern above and still did not
    # arrive -- 144.1 GB of weights and no pipeline entry point, which a size check
    # cannot see and which only surfaces when `from_pretrained` opens it. Small
    # files, exact names, checked individually.
    from huggingface_hub import hf_hub_download

    for filename in ("model_index.json", "modular_model_index.json"):
        target = dest / filename
        if target.exists():
            continue
        cached = hf_hub_download("MiniMaxAI/MiniMax-H3", filename)
        target.write_bytes(Path(cached).read_bytes())
        print(f"[fetch]   {filename}: {target.stat().st_size} bytes", flush=True)

    size = shallow_bytes(dest) / 1e9
    print(f"[fetch] checkpoint size: {size:.1f} GB", flush=True)
    # `hf download` exits 0 having downloaded nothing at all, so the size is the only
    # signal that the patterns actually matched.
    if size < 100:
        raise SystemExit(
            f"[fetch] FATAL: checkpoint is {size:.1f} GB, expected ~143 GB. The include "
            f"patterns did not match what they were meant to."
        )


def fetch_vggsound(dest: Path, shards: list[int], workers: int) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    # One file per invocation, asserted after each.
    #
    # `hf download` has now mis-parsed its own flags three separate times in this
    # task: `--exclude` swallowed every config file, repeated `--include` kept only
    # the last pattern, and `--include a b` let the positional FILENAMES argument
    # take one of them so the CSV was never requested -- each time exiting 0. The
    # form below has no ambiguity left to exploit: name one file, then check that
    # file is on disk. A few extra HTTP round trips is a fair price for that.
    wanted = ["vggsound.csv"] + [f"vggsound_{shard:02d}.tar.gz" for shard in shards]
    for filename in wanted:
        target = dest / filename
        if target.exists() and target.stat().st_size > 0:
            print(f"[fetch]   {filename}: already present "
                  f"({target.stat().st_size / 1e9:.1f} GB), skipping", flush=True)
            continue
        run(["hf", "download", VGGSOUND_REPO, filename, "--repo-type", "dataset",
             "--local-dir", str(dest), "--max-workers", str(workers)],
            what=f"VGGSound {filename}")
        if not target.exists() or target.stat().st_size == 0:
            raise SystemExit(f"[fetch] FATAL: {filename} still absent after a download that "
                             f"reported success")
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
