#!/usr/bin/env python3
"""Keep the first RUNNING fleet candidate and cancel explicit siblings."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


TASK_DIR = Path(__file__).resolve().parent
DEFAULT_EVENTS_PATH = TASK_DIR / "monitor_events.jsonl"
DEFAULT_NAME_PREFIX = "u6gb-16-nodes-18-jluy-"
ACTIVE_STATES = {"PENDING", "CONFIGURING", "RUNNING", "COMPLETING", "SUSPENDED"}


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_event(path: Path, event_type: str, **fields: object) -> None:
    row = {"ts": now_utc(), "event_type": event_type, **fields}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def parse_queue(output: str) -> list[dict[str, str]]:
    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("|", 4)
        if len(parts) != 5:
            raise ValueError(f"unexpected squeue row: {line!r}")
        job_id, name, state, start, nodes = parts
        rows.append(
            {"job_id": job_id, "name": name, "state": state, "start": start, "nodes": nodes}
        )
    return rows


def query_candidates(job_ids: Iterable[str]) -> list[dict[str, str]]:
    ids = list(job_ids)
    completed = subprocess.run(
        ["squeue", "-h", "-j", ",".join(ids), "-o", "%A|%j|%T|%S|%D"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "squeue failed")
    return parse_queue(completed.stdout)


def choose_winner(rows: Iterable[dict[str, str]], name_prefix: str) -> str | None:
    running = [
        row
        for row in rows
        if row["state"] == "RUNNING" and row["name"].startswith(name_prefix)
    ]
    if not running:
        return None
    running.sort(
        key=lambda row: (
            row["start"] if row["start"] not in {"", "N/A", "Unknown"} else "9999",
            int(row["job_id"]),
        )
    )
    return running[0]["job_id"]


def cancellation_targets(
    rows: Iterable[dict[str, str]],
    candidate_ids: set[str],
    winner_id: str,
    name_prefix: str,
) -> list[str]:
    return sorted(
        (
            row["job_id"]
            for row in rows
            if row["job_id"] in candidate_ids
            and row["job_id"] != winner_id
            and row["name"].startswith(name_prefix)
            and row["state"] in ACTIVE_STATES
        ),
        key=int,
    )


def cancel_candidates(job_ids: Iterable[str], dry_run: bool) -> list[dict[str, object]]:
    results = []
    for job_id in job_ids:
        if dry_run:
            results.append({"job_id": job_id, "returncode": 0, "dry_run": True})
            continue
        completed = subprocess.run(["scancel", job_id], capture_output=True, text=True, check=False)
        results.append(
            {
                "job_id": job_id,
                "returncode": completed.returncode,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            }
        )
    return results


def validate_visible_names(rows: Iterable[dict[str, str]], name_prefix: str) -> None:
    mismatches = [row for row in rows if not row["name"].startswith(name_prefix)]
    if mismatches:
        rendered = ", ".join(f"{row['job_id']}={row['name']}" for row in mismatches)
        raise ValueError(f"candidate name mismatch: {rendered}")


def reconcile(
    job_ids: list[str], name_prefix: str, events_path: Path, dry_run: bool
) -> str | None:
    rows = query_candidates(job_ids)
    validate_visible_names(rows, name_prefix)
    winner_id = choose_winner(rows, name_prefix)
    if winner_id is None:
        append_event(events_path, "reconcile_no_winner", candidates=job_ids, queue=rows)
        return None

    targets = cancellation_targets(rows, set(job_ids), winner_id, name_prefix)
    cancel_results = cancel_candidates(targets, dry_run)
    append_event(
        events_path,
        "winner_selected",
        winner_job_id=winner_id,
        candidates=job_ids,
        cancelled_job_ids=targets,
        cancel_results=cancel_results,
        dry_run=dry_run,
        queue=rows,
    )
    print(f"winner={winner_id} cancelled={','.join(targets) if targets else 'none'}")
    return winner_id


def stop_waiters(waiters: dict[int, tuple[str, subprocess.Popen[bytes]]]) -> None:
    for _, process in waiters.values():
        if process.poll() is None:
            process.terminate()
    for _, process in waiters.values():
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def monitor(job_ids: list[str], name_prefix: str, events_path: Path, dry_run: bool) -> int:
    initial_rows = query_candidates(job_ids)
    validate_visible_names(initial_rows, name_prefix)
    append_event(events_path, "monitor_started", candidates=job_ids, queue=initial_rows)

    waiters: dict[int, tuple[str, subprocess.Popen[bytes]]] = {}
    for job_id in job_ids:
        process = subprocess.Popen(
            ["scontrol", "wait_job", job_id],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        waiters[process.pid] = (job_id, process)

    try:
        while waiters:
            pid, raw_status = os.waitpid(-1, 0)
            job_id, process = waiters.pop(pid)
            process.returncode = os.waitstatus_to_exitcode(raw_status)
            append_event(
                events_path,
                "waiter_returned",
                job_id=job_id,
                returncode=process.returncode,
            )
            winner_id = reconcile(job_ids, name_prefix, events_path, dry_run)
            if winner_id is not None:
                return 0
        append_event(events_path, "monitor_finished_without_winner", candidates=job_ids)
        return 1
    finally:
        stop_waiters(waiters)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_ids", nargs="+", help="explicit candidate Slurm job IDs")
    parser.add_argument("--name-prefix", default=DEFAULT_NAME_PREFIX)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS_PATH)
    parser.add_argument("--dry-run", action="store_true", help="never call scancel")
    parser.add_argument("--once", action="store_true", help="reconcile once without waiting")
    args = parser.parse_args()
    if any(not job_id.isdigit() for job_id in args.job_ids):
        parser.error("every job ID must be numeric")
    args.job_ids = list(dict.fromkeys(args.job_ids))
    return args


def main() -> int:
    args = parse_args()
    if args.once:
        return 0 if reconcile(args.job_ids, args.name_prefix, args.events, args.dry_run) else 3
    return monitor(args.job_ids, args.name_prefix, args.events, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
