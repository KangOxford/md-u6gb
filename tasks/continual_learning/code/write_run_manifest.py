"""Write a rollout run's manifest BEFORE its first member exists.

This is the half of P1 that a future run can satisfy. It is deliberately a separate program
from the gate: the gate refuses, this one makes refusal unnecessary, and neither can be used
to make the other's job disappear.

Two things it will not do.

**It will not write into a directory that already has members.** A manifest whose purpose is
to precede the data cannot be added afterwards, and a file that claims to have preceded data
it did not is worse than no file: the next reader would trust it.

**It will not invent a value.** Every field it cannot read becomes JSON `null` together with
an entry in `unrecoverable` saying why. `null` is falsy and fails a check; "unknown" is a
truthy string that passes every `if manifest.get(field):` and silently certifies a run that
recorded nothing.

Usage
    python3 write_run_manifest.py --run-root <dir that does not yet exist or is empty> \
        --run-tag v5me4 --checkpoint-root <path> --checkpoint-step 69378 \
        --context-file <path> --k 3 --seed0 97701 --seed-stride 1 \
        --n-cond-msgs 500 --n-gen-msgs 250 --token-mode 26tok --batch-size 48
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

HORIZONS = [10, 25, 50, 100, 150, 200, 250]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state(repo: Path) -> Dict[str, object]:
    def run(*a: str) -> Optional[str]:
        r = subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"code_commit": commit, "code_dirty": None if status is None else bool(status)}


def jax_version() -> Optional[str]:
    try:
        import jax  # noqa: F401 -- imported for its version only
        return jax.__version__
    except Exception:
        return None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--checkpoint-root", type=Path, required=True)
    ap.add_argument("--checkpoint-step", type=int, required=True)
    ap.add_argument("--context-file", type=Path, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--seed0", type=int, required=True)
    ap.add_argument("--seed-stride", type=int, required=True)
    ap.add_argument("--n-cond-msgs", type=int, required=True)
    ap.add_argument("--n-gen-msgs", type=int, required=True)
    ap.add_argument("--token-mode", required=True)
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--xla-flags", default="", help="verbatim, e.g. '--xla_gpu_autotune_level=0'")
    ap.add_argument("--params-sha256", default=None,
                    help="hash of the RESTORED parameter tree; omit and it is recorded null")
    ap.add_argument("--optimizer-state-present", choices=["true", "false"], default=None)
    ap.add_argument("--real-arm-path", type=Path, default=None,
                    help="when the real arm is referenced rather than written per member")
    ap.add_argument("--streams", nargs="+",
                    default=["data_cond", "data_gen", "data_real"],
                    help="data_cond is required: without it no replay can be initialised")
    ap.add_argument("--repo", type=Path,
                    default=Path("/lus/lfs1aip2/projects/public/u6gb/sigma-0"))
    a = ap.parse_args(argv)

    root = a.run_root
    if root.exists():
        members = [d for d in root.iterdir() if d.is_dir() and d.name != "logs"]
        if members:
            print(f"REFUSED: {root} already holds {len(members)} member director"
                  f"{'y' if len(members) == 1 else 'ies'} "
                  f"(e.g. {sorted(d.name for d in members)[:3]}).\n"
                  f"A manifest that is supposed to precede the data cannot be written after it. "
                  f"Use `generation_gate.py --mode historical` to attest what is recoverable "
                  f"from an existing archive; that produces an attestation, not a manifest, and "
                  f"it says so in its own schema.")
            return 2
    else:
        root.mkdir(parents=True)

    if not a.context_file.is_file():
        print(f"REFUSED: context file does not exist: {a.context_file}")
        return 2

    unrecoverable: Dict[str, str] = {}
    g = git_state(a.repo)
    if g["code_commit"] is None:
        unrecoverable["code_commit"] = f"{a.repo} is not a git checkout this process can read"
    if a.params_sha256 is None:
        unrecoverable["params_sha256"] = (
            "not supplied; hash the restored parameter tree inside the generation process "
            "and pass it here, otherwise two runs cannot be shown to have used one model")
    if a.optimizer_state_present is None:
        unrecoverable["optimizer_state_present"] = (
            "not supplied; wm_ft_multi3 restores with Muon absent, and a manifest that does "
            "not say so lets an inference-only artefact be mistaken for a trainable one")
    jv = jax_version()
    if jv is None:
        unrecoverable["jax_version"] = "jax not importable in the process writing the manifest"

    m = {
        "manifest_version": 1,
        "kind": "prospective",
        "written_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "run_tag": a.run_tag,
        "code_commit": g["code_commit"],
        "code_dirty": g["code_dirty"],
        "checkpoint_root": str(a.checkpoint_root),
        "checkpoint_step": a.checkpoint_step,
        "params_sha256": a.params_sha256,
        "optimizer_state_present": (None if a.optimizer_state_present is None
                                    else a.optimizer_state_present == "true"),
        "partial_restore": None,
        "token_mode": a.token_mode,
        "n_cond_msgs": a.n_cond_msgs,
        "n_gen_msgs": a.n_gen_msgs,
        "horizons": HORIZONS,
        "context_file": str(a.context_file),
        "context_file_sha256": sha256_file(a.context_file),
        "n_contexts": None,
        "seed0": a.seed0,
        "seed_stride": a.seed_stride,
        "k": a.k,
        "batch_size": a.batch_size,
        "xla_flags": a.xla_flags.split() if a.xla_flags else [],
        "jax_version": jv,
        "platform": platform.platform(),
        "real_arm_written": a.real_arm_path is None,
        "real_arm_path": None if a.real_arm_path is None else str(a.real_arm_path),
        "streams_written": list(a.streams),
        "unrecoverable": unrecoverable,
    }
    try:
        idx = json.loads(a.context_file.read_text())
        m["n_contexts"] = len(idx.get("rank_indices", idx.get("all_indices", [])))
    except (json.JSONDecodeError, AttributeError):
        unrecoverable["n_contexts"] = "context file is not JSON with rank_indices/all_indices"

    out = root / "manifest.json"
    out.write_text(json.dumps(m, indent=1))
    print(f"wrote {out}")
    if unrecoverable:
        print("recorded as null, with a reason, rather than invented:")
        for k, v in unrecoverable.items():
            print(f"  {k}: {v}")
    if "data_cond" not in m["streams_written"]:
        print("WARNING: data_cond is not among the streams; fidelity.py cannot replay this run, "
              "which is precisely why the existing 80 members are unusable for M1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
