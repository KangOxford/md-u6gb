#!/usr/bin/env python3
"""Run one sbatch command and append its exact result to submissions.jsonl."""

from __future__ import annotations

import argparse
import fcntl
import json
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path


LOG_PATH = Path(__file__).resolve().parent / "submissions.jsonl"


def append(row: dict) -> None:
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or Path(command[0]).name != "sbatch":
        parser.error("command must begin with sbatch")

    rendered = shlex.join(command)
    if args.dry_run:
        print(rendered)
        return 0

    append({"ts": now_utc(), "event_type": "attempt", "command": rendered, "argv": command})
    completed = subprocess.run(command, capture_output=True, text=True)
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    match = re.search(r"Submitted batch job (\d+)", stdout)
    append(
        {
            "ts": now_utc(),
            "event_type": "result",
            "command": rendered,
            "argv": command,
            "returncode": completed.returncode,
            "job_id": match.group(1) if match else None,
            "stdout": stdout,
            "stderr": stderr,
        }
    )
    if stdout:
        print(stdout)
    if stderr:
        print(stderr)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
