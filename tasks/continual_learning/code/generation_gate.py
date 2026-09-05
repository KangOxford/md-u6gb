"""P1/P2/P6 — the three preconditions, as one executable gate.

Nothing generates rollouts until this exits 0. It is a gate rather than a checklist because
a checklist is satisfied by intention: this project has shipped a knob that was set, printed
and recorded and never reached the code, and a report that printed a header and no rows.

Usage
    python3 generation_gate.py --run-root <dir about to be written>          # before a run
    python3 generation_gate.py --run-root <dir> --archive <existing archive> # audit an archive

Exit codes: 0 all three pass · 1 a precondition fails · 2 the gate could not evaluate.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PER_MEMBER_INODES_UNPACKED = 3007      # measured 2026-09-05, 500 contexts x ~6 files
PER_MEMBER_INODES_DEDUPED = 1507       # real arm written once per ticker
FREE_INODE_SAFETY = 0.5                # plan at most half the free inodes; the project is shared

REQUIRED_MANIFEST_FIELDS = [
    "manifest_version", "written_at_utc", "run_tag", "code_commit", "code_dirty",
    "checkpoint_root", "checkpoint_step", "params_sha256", "optimizer_state_present",
    "partial_restore", "token_mode", "n_cond_msgs", "n_gen_msgs", "horizons",
    "context_file", "context_file_sha256", "n_contexts", "seed0", "seed_stride", "k",
    "batch_size", "xla_flags", "jax_version", "platform",
    "real_arm_written", "real_arm_path", "streams_written",
]
FORBIDDEN_PLACEHOLDERS = {"unknown", "n/a", "na", "none", "tbd", ""}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fail(msgs: List[str], text: str) -> None:
    msgs.append(text)


# --------------------------------------------------------------------------------------
# P1 — the rollout manifest
# --------------------------------------------------------------------------------------

def check_p1(run_root: Path) -> Tuple[bool, List[str]]:
    """A manifest written BEFORE the first member, complete, and honest about absences.

    The failure this prevents already happened: an archive recorded which contexts were
    scored and nothing about how the rollouts were produced, so when the analysis concluded
    that more members would settle the question, the extension could not be done -- not for
    want of GPUs, for want of a record.
    """
    bad: List[str] = []
    mf = run_root / "manifest.json"
    if not mf.is_file():
        return False, [f"no manifest at {mf}"]
    try:
        m = json.loads(mf.read_text())
    except json.JSONDecodeError as e:
        return False, [f"manifest is not valid JSON: {e}"]

    for field in REQUIRED_MANIFEST_FIELDS:
        if field not in m:
            _fail(bad, f"manifest field missing: {field}")
        elif isinstance(m[field], str) and m[field].strip().lower() in FORBIDDEN_PLACEHOLDERS:
            # null is falsy and fails a check; "unknown" is a truthy string that passes
            # every `if manifest.get(field):` and silently certifies a run that recorded
            # nothing.
            _fail(bad, f"manifest field {field}={m[field]!r} -- write null, not a placeholder")

    if "data_cond" not in (m.get("streams_written") or []):
        _fail(bad, "streams_written does not include data_cond: without the conditioning "
                   "window no replay can be initialised and fidelity.py cannot run "
                   "(the existing 80 members are unusable for exactly this reason)")

    written = m.get("written_at_utc")
    if written:
        try:
            t0 = datetime.datetime.fromisoformat(written.replace("Z", "+00:00")).timestamp()
            members = [d for d in run_root.iterdir() if d.is_dir() and d.name != "logs"]
            early = [d.name for d in members if d.stat().st_mtime < t0 - 1]
            if early:
                _fail(bad, f"manifest is younger than {len(early)} member dir(s) "
                           f"(e.g. {early[:3]}); it must be written before the first member "
                           f"so a run killed halfway still says what it was doing")
        except ValueError:
            _fail(bad, f"written_at_utc is not ISO-8601: {written!r}")

    cf = m.get("context_file")
    if cf:
        p = Path(cf)
        if not p.is_file():
            _fail(bad, f"context_file does not exist: {cf}")
        elif sha256_file(p) != m.get("context_file_sha256"):
            _fail(bad, "context_file_sha256 does not match the file on disk")

    if m.get("real_arm_written") is False and not Path(m.get("real_arm_path") or "").is_dir():
        _fail(bad, "real_arm_written is false and real_arm_path does not resolve -- the "
                   "inode dedupe would be data loss, not a saving")

    return not bad, bad


# --------------------------------------------------------------------------------------
# P2 — one shared, hashed context set
# --------------------------------------------------------------------------------------

def check_p2(archive: Path, config: str, index_name: str = "sample_indices_rank0.json",
             shared_index: Optional[Path] = None) -> Tuple[bool, List[str]]:
    """Every member of a ticker must score the same contexts, provably.

    Verified on the existing archive: within each of the 8 tickers all 10 seeds carry a
    byte-identical index, and the 8 tickers differ, as they should. What is still missing is
    a single index file living outside the member directories, so that the shared object is
    a file rather than a coincidence.
    """
    bad: List[str] = []
    by_ticker: Dict[str, Dict[str, List[str]]] = {}
    # Scoped to one config. A bare hp_*_s* glob pools every config in the archive, and
    # different configs legitimately score different contexts -- pooling them reports 7-9
    # "distinct context sets" per ticker and hides whether any single config is coherent.
    # Found by running this gate, which is what a gate is for.
    members = sorted(d for d in archive.glob(f"hp_{config}_*_s*") if d.is_dir()
                     and d.name[len(f"hp_{config}_"):].rsplit("_s", 1)[-1].isdigit()
                     and "_" not in d.name[len(f"hp_{config}_"):].rsplit("_s", 1)[0])
    if not members:
        return False, [f"no member directories for config {config!r} under {archive}"]

    for d in members:
        idx = d / "member_0" / index_name
        if not idx.is_file():
            _fail(bad, f"{d.name}: no {index_name}")
            continue
        name = d.name
        ticker = name[len(f"hp_{config}_"):].rsplit("_s", 1)[0]
        by_ticker.setdefault(ticker, {}).setdefault(sha256_file(idx), []).append(name)

    for ticker, groups in sorted(by_ticker.items()):
        if len(groups) != 1:
            _fail(bad, f"{ticker}: {len(groups)} distinct context sets across its members "
                       f"-- members scoring different contexts cannot be pooled")

    if shared_index is None:
        _fail(bad, "no shared context-set manifest given: the context set is identical by "
                   "coincidence of construction, not by reference to one hashed object")
    elif not shared_index.is_file():
        _fail(bad, f"shared context-set manifest does not exist: {shared_index}")
    else:
        man = json.loads(shared_index.read_text())
        for ticker, groups in by_ticker.items():
            want = (man.get("tickers", {}).get(ticker) or {}).get("sha256")
            if want is None:
                _fail(bad, f"{ticker}: absent from the shared context-set manifest")
            elif want not in groups:
                _fail(bad, f"{ticker}: members do not match the manifest's hash {want[:16]}")
            else:
                f = Path(man["tickers"][ticker]["file"])
                if not f.is_file():
                    _fail(bad, f"{ticker}: manifest points at a missing file {f}")
                elif sha256_file(f) != want:
                    _fail(bad, f"{ticker}: the promoted index file no longer matches its hash")

    return not bad, bad


# --------------------------------------------------------------------------------------
# P6 — the inode write plan
# --------------------------------------------------------------------------------------

def free_inodes(mount: str = "/lus/lfs1aip2") -> Optional[Tuple[int, int]]:
    """(used, limit) from `lfs quota`, or None when it cannot be read.

    Returning None rather than a default matters: a probe that fails must not look like a
    probe that found plenty of room.
    """
    try:
        pid = subprocess.run(["lfs", "project", "-d", "/lus/lfs1aip2/projects/public/u6gb"],
                             capture_output=True, text=True, timeout=60).stdout.split()
        if not pid:
            return None
        q = subprocess.run(["lfs", "quota", "-p", pid[0], mount],
                           capture_output=True, text=True, timeout=60).stdout.split("\n")
        for line in q:
            f = line.split()
            if len(f) >= 9 and f[0].startswith(mount):
                return int(f[5]), int(f[7])
    except (subprocess.SubprocessError, ValueError, IndexError):
        return None
    return None


def check_p6(members_planned: int, deduped: bool,
             safety: float = FREE_INODE_SAFETY) -> Tuple[bool, List[str], Dict[str, object]]:
    """Plan against inodes read now, not against a figure quoted from a document.

    The headroom is borrowed: the project sat 118 inodes from its hard cap on 2026-09-04 at
    17:54Z and the room that exists was released by a cleanup effort. A budget quoted without
    its timestamp is not a budget, so this reads the quota and records the reading.
    """
    bad: List[str] = []
    q = free_inodes()
    if q is None:
        return False, ["could not read `lfs quota` -- a probe that fails is not a probe that "
                       "found room"], {}
    used, limit = q
    free = limit - used
    per = PER_MEMBER_INODES_DEDUPED if deduped else PER_MEMBER_INODES_UNPACKED
    need = members_planned * per
    rec = {"read_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "inodes_used": used, "inodes_limit": limit, "inodes_free": free,
           "per_member": per, "deduped": deduped, "members_planned": members_planned,
           "inodes_planned": need, "fraction_of_free": need / free if free else float("inf")}
    if need > safety * free:
        _fail(bad, f"plan needs {need:,} inodes, which is {need/free:.0%} of the {free:,} free; "
                   f"the cap is {safety:.0%} because the project is shared and the release rate "
                   f"is not under this plan's control")
    return not bad, bad, rec


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", type=Path, help="the run directory about to be written (P1)")
    ap.add_argument("--archive", type=Path,
                    default=Path("/lus/lfs1aip2/projects/public/u6gb/tasks/"
                                 "crps_return_alignment_20260808T025024Z/data"))
    ap.add_argument("--config", default="v5me3", help="which config's members P2 checks")
    ap.add_argument("--shared-index", type=Path, help="the one context index file (P2)")
    ap.add_argument("--members", type=int, required=True, help="members this run will write")
    ap.add_argument("--deduped", action="store_true", help="real arm written once per ticker")
    ap.add_argument("--record", type=Path, help="write the P6 quota reading here")
    a = ap.parse_args(argv)

    results = []
    if a.run_root:
        results.append(("P1", *check_p1(a.run_root)))
    else:
        results.append(("P1", False, ["--run-root not given: P1 cannot be evaluated, and an "
                                      "unevaluated precondition is a failed one"]))
    results.append(("P2", *check_p2(a.archive, a.config, shared_index=a.shared_index)))
    ok6, bad6, rec = check_p6(a.members, a.deduped)
    results.append(("P6", ok6, bad6))

    for name, ok, msgs in results:
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
        for m in msgs:
            print(f"    - {m}")
    if rec:
        print(f"P6 reading: {rec['inodes_free']:,} free at {rec['read_at_utc']}, "
              f"plan {rec['inodes_planned']:,} = {rec['fraction_of_free']:.0%}")
        if a.record:
            a.record.write_text(json.dumps(rec, indent=1))

    passed = all(ok for _, ok, _ in results)
    print("\nGATE:", "OPEN -- generation may start" if passed else
          "CLOSED -- no rollout generation")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
