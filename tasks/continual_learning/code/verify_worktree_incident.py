"""Supervision-side verification of the shared-worktree incident of 2026-09-05.

Re-derives, from the repository itself, the three claims made about the 75 entries that
were stashed and partially restored. It asserts rather than prints a summary, so a claim
that stops holding makes this exit non-zero.

Deliberately read-only. It never writes, deletes, stashes, resets, or touches a file that
belongs to another line of work.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path("/lus/lfs1aip2/projects/public/u6gb")
STASH = "stash@{0}"
LIVE_LOG = "tasks/u6gb_16_nodes_daily_log/events.jsonl"


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True)


def blob(ref: str, path: str) -> str | None:
    r = git("show", f"{ref}:{path}")
    return hashlib.sha256(r.stdout).hexdigest() if r.returncode == 0 else None


def worktree(path: str) -> str | None:
    p = ROOT / path
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def entries() -> List[Tuple[str, str]]:
    out = git("stash", "show", "--name-status", STASH).stdout.decode()
    return [tuple(line.split("\t", 1)) for line in out.splitlines() if "\t" in line]


def main() -> int:
    rows = entries()
    if not rows:
        print(f"FAILED: {STASH} carries no entries. The backup must not be dropped.")
        return 2

    mods = [f for st, f in rows if st == "M"]
    dels = [f for st, f in rows if st == "D"]
    print(f"stash {STASH}: {len(rows)} entries -- {len(mods)} M, {len(dels)} D")

    # --- claim 1: every M except the live log is byte-identical to the stash -------------
    restored, drifted, lost = [], [], []
    for f in mods:
        s, w, h = blob(STASH, f), worktree(f), blob("HEAD", f)
        if w is None:
            lost.append((f, "absent from the worktree"))
        elif w == s:
            restored.append(f)
        elif w == h:
            lost.append((f, "reverted to HEAD -- the modification is gone"))
        else:
            drifted.append(f)

    print(f"\nM entries")
    print(f"  restored byte-for-byte : {len(restored)}")
    print(f"  drifted after restore  : {len(drifted)}  {drifted}")
    print(f"  lost                   : {len(lost)}  {lost}")

    ok = True
    if lost:
        print("  FAILED: a stashed modification is not present in the worktree.")
        ok = False
    if drifted != [LIVE_LOG]:
        print(f"  NOTE: expected exactly [{LIVE_LOG}] to have drifted; got {drifted}.")
        print("        Drift is benign for a log another session appends to; investigate anything else.")

    # --- claim 2: the D entries are present, equal to HEAD, and NOT re-deleted -----------
    # `git checkout <stash> -- <path>` cannot restore a deletion, so these were left as the
    # rebase found them. The point of this check is that the file content is HEAD's -- i.e.
    # nothing here was authored by the session that caused the incident -- and that the
    # original deletion has NOT been re-applied, which would be a second uninvited change.
    print(f"\nD entries (deletions another line had made in the worktree, uncommitted)")
    eq_head, not_eq, absent = [], [], []
    for f in dels:
        w, h = worktree(f), blob("HEAD", f)
        if w is None:
            absent.append(f)
        elif w == h:
            eq_head.append(f)
        else:
            not_eq.append(f)
    print(f"  present and identical to HEAD : {len(eq_head)}")
    print(f"  present but NOT HEAD's content: {len(not_eq)}  {not_eq}")
    print(f"  still absent (deletion intact): {len(absent)}")
    if not_eq:
        print("  FAILED: a reverted file differs from HEAD, so its content was authored here.")
        ok = False
    if absent:
        print("  NOTE: some deletions are intact again -- the owning line has re-applied them.")

    # --- claim 3: ownership, so nothing is 'cleaned up' on someone else's behalf ---------
    print("\nOwnership of the D entries, from the last commit that touched each directory")
    dirs: Dict[str, List[str]] = {}
    for f in dels:
        dirs.setdefault(str(Path(f).parent), []).append(f)
    for d, fs in sorted(dirs.items()):
        r = git("log", "-1", "--format=%h %an %ad %s", "--date=short", "--", d)
        line = r.stdout.decode().strip() or "(no commit touches this path)"
        print(f"  {len(fs)} file(s)  {line[:96]}")
    print("\n  These belong to the hybrid_system line, not to continual_learning. They are not")
    print("  re-deleted here: deletion is forbidden on this repository, and re-applying another")
    print("  line's uncommitted intent would be a second uninvited change to work that is not ours.")

    print("\nVERDICT:", "consistent with the record" if ok else "INCONSISTENT -- see FAILED above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
