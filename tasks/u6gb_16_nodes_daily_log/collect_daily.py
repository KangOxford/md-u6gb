#!/usr/bin/env python3
"""Build one deterministic UTC-day coverage report from Slurm accounting."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "manifest.json"
SUBMISSIONS_PATH = ROOT / "submissions.jsonl"
EVENTS_PATH = ROOT / "events.jsonl"
STATE_PATH = ROOT / "state.json"
SUMMARY_PATH = ROOT / "summary.md"
DAILY_DIR = ROOT / "daily"
UTC = timezone.utc
SACCT_FIELDS = [
    "JobIDRaw",
    "JobName",
    "State",
    "Submit",
    "Start",
    "End",
    "Elapsed",
    "AllocNodes",
    "NodeList",
    "ExitCode",
]


@dataclass(frozen=True)
class Job:
    job_id: str
    name: str
    state: str
    submit: str
    start: datetime | None
    end: datetime | None
    elapsed: str
    alloc_nodes: int
    node_list: str
    exit_code: str


@dataclass(frozen=True)
class Segment:
    start: datetime
    end: datetime
    running_nodes: int

    @property
    def seconds(self) -> int:
        return max(0, int((self.end - self.start).total_seconds()))


@dataclass(frozen=True)
class BatchSource:
    path: str
    sha256: str
    directives: tuple[str, ...]


def iso_z(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_slurm_time(value: str) -> datetime | None:
    value = value.strip()
    if not value or value in {"Unknown", "None", "N/A"}:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_batch_source(manifest: dict) -> BatchSource:
    path = Path(manifest["batch_script"])
    payload = path.read_bytes()
    directives = []
    for index, line in enumerate(payload.decode("utf-8").splitlines()):
        stripped = line.strip()
        if index == 0 and stripped.startswith("#!"):
            continue
        if stripped.startswith("#SBATCH "):
            directives.append(stripped.removeprefix("#SBATCH "))
        elif stripped and not stripped.startswith("#"):
            break
    return BatchSource(
        path=str(path),
        sha256=hashlib.sha256(payload).hexdigest(),
        directives=tuple(directives),
    )


def run_sacct(
    job_name: str,
    start: datetime,
    end: datetime,
    lookback_hours: int,
) -> tuple[list[Job], str]:
    query_start = start - timedelta(hours=lookback_hours)
    command = [
        "sacct",
        "-X",
        "-S",
        query_start.strftime("%Y-%m-%dT%H:%M:%S"),
        "-E",
        end.strftime("%Y-%m-%dT%H:%M:%S"),
        "-nP",
        "-o",
        ",".join(SACCT_FIELDS),
    ]
    user = os.environ.get("USER")
    if user:
        command.insert(1, f"--user={user}")
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    job_name_pattern = re.compile(rf"^{re.escape(job_name)}(?:-resume[0-9]+)?$")
    jobs = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        values = line.split("|")
        if len(values) != len(SACCT_FIELDS):
            raise RuntimeError(f"Unexpected sacct row with {len(values)} fields: {line}")
        if not job_name_pattern.fullmatch(values[1]):
            continue
        alloc_nodes = int(values[7]) if values[7].isdigit() else 0
        jobs.append(
            Job(
                job_id=values[0],
                name=values[1],
                state=values[2],
                submit=values[3],
                start=parse_slurm_time(values[4]),
                end=parse_slurm_time(values[5]),
                elapsed=values[6],
                alloc_nodes=alloc_nodes,
                node_list=values[8],
                exit_code=values[9],
            )
        )
    evidence = (
        shlex.join(command)
        + f"\n# client-side JobName filter: ^{re.escape(job_name)}(?:-resume[0-9]+)?$"
    )
    return jobs, evidence


def build_segments(jobs: Iterable[Job], start: datetime, end: datetime) -> list[Segment]:
    events: dict[datetime, int] = defaultdict(int)
    events[start] += 0
    events[end] += 0
    now = datetime.now(UTC)
    for job in jobs:
        if job.start is None or job.alloc_nodes <= 0:
            continue
        interval_start = max(start, job.start)
        interval_end = min(end, job.end or now)
        if interval_end <= interval_start:
            continue
        events[interval_start] += job.alloc_nodes
        events[interval_end] -= job.alloc_nodes

    running = 0
    previous = start
    segments: list[Segment] = []
    for timestamp in sorted(events):
        if timestamp > previous:
            segment = Segment(previous, timestamp, max(0, running))
            if segments and segments[-1].running_nodes == segment.running_nodes:
                prior = segments.pop()
                segment = Segment(prior.start, segment.end, segment.running_nodes)
            segments.append(segment)
        running += events[timestamp]
        previous = timestamp
    return segments


def submissions_for_window(start: datetime, end: datetime) -> list[dict]:
    selected = []
    for row in load_jsonl(SUBMISSIONS_PATH):
        try:
            timestamp = parse_slurm_time(str(row.get("ts", "")))
        except ValueError:
            continue
        if timestamp is not None and start <= timestamp < end and row.get("event_type") == "result":
            selected.append(row)
    return selected


def duration_text(seconds: int) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes}m {seconds}s"


def markdown_table(rows: list[list[str]], headers: list[str]) -> str:
    def safe(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    output = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    output.extend("| " + " | ".join(safe(value) for value in row) + " |" for row in rows)
    return "\n".join(output)


def render_report(
    report_date: date,
    start: datetime,
    end: datetime,
    job_name: str,
    target_nodes: int,
    jobs: list[Job],
    sacct_command: str,
    submissions: list[dict],
    segments: list[Segment],
    batch_source: BatchSource,
) -> tuple[str, dict]:
    window_seconds = int((end - start).total_seconds())
    full_seconds = sum(segment.seconds for segment in segments if segment.running_nodes >= target_nodes)
    partial_seconds = sum(segment.seconds for segment in segments if 0 < segment.running_nodes < target_nodes)
    zero_seconds = sum(segment.seconds for segment in segments if segment.running_nodes == 0)
    covered_node_seconds = sum(
        min(segment.running_nodes, target_nodes) * segment.seconds for segment in segments
    )
    denominator = target_nodes * window_seconds
    coverage_ratio = covered_node_seconds / denominator if denominator else 0.0

    command_rows = [
        [
            row.get("ts", ""),
            row.get("command", ""),
            row.get("returncode", ""),
            row.get("job_id", ""),
            row.get("stdout", "") or row.get("stderr", ""),
        ]
        for row in submissions
    ]
    job_rows = [
        [
            job.job_id,
            job.name,
            job.state,
            job.submit,
            iso_z(job.start) if job.start else "-",
            iso_z(job.end) if job.end else "-",
            job.node_list or "-",
            job.exit_code or "-",
        ]
        for job in jobs
    ]
    segment_rows = [
        [
            iso_z(segment.start),
            iso_z(segment.end),
            f"{segment.running_nodes}/{target_nodes}",
            max(0, target_nodes - segment.running_nodes),
            "FULL" if segment.running_nodes >= target_nodes else "GAP",
        ]
        for segment in segments
    ]

    report = [
        f"# {report_date.isoformat()} UTC - {job_name} Daily Coverage",
        "",
        "## 1. First-Principles Coverage",
        "",
        f"- Window: `{iso_z(start)}` to `{iso_z(end)}`",
        f"- Full 16/16 coverage: `{duration_text(full_seconds)}`",
        f"- Partial coverage: `{duration_text(partial_seconds)}`",
        f"- Zero coverage: `{duration_text(zero_seconds)}`",
        f"- Node-minute coverage: `{coverage_ratio:.2%}`",
        "- Operational reading is minute-level; a few boundary seconds or minutes are not material.",
        "",
        markdown_table(segment_rows, ["Start UTC", "End UTC", "Running", "Missing", "Class"]),
        "",
        "## 2. Slurm State, Commands, And Results",
        "",
    ]
    if command_rows:
        report.append(markdown_table(command_rows, ["Time UTC", "Command", "RC", "Job ID", "Result"]))
    else:
        report.append("No experiment submission was recorded in this window.")
    report.extend(["", "### Slurm Results", ""])
    if job_rows:
        report.append(
            markdown_table(
                job_rows,
                ["Job ID", "Name", "State", "Submit", "Start", "End", "Nodes", "Exit"],
            )
        )
    else:
        report.append(f"No `{job_name}` rows were returned by Slurm accounting.")
    report.extend(
        [
            "",
            "Evidence query:",
            "",
            "```bash",
            sacct_command,
            "```",
            "",
            "## 3. Command Source",
            "",
            f"- Batch: `{batch_source.path}`",
            f"- SHA256 observed today: `{batch_source.sha256}`",
            "- Header resource directives: `" + " ".join(batch_source.directives) + "`",
            f"- Fleet override: `--job-name={job_name} --nodes=16 --time=23:59:00`",
            "- Effective fleet size: `16 nodes x 4 GPUs/node = 64 H100 GPUs`",
            f"- Coverage includes `{job_name}` and `{job_name}-resumeN`.",
            "- Accounting uses a 24-hour lookback, then clips runtime to this UTC day.",
            "",
            "This report is evidence-only. It did not submit, retry, cancel, or modify any experiment.",
            "",
        ]
    )
    metrics = {
        "window_seconds": window_seconds,
        "submitted_jobs": len(submissions),
        "slurm_rows": len(jobs),
        "full_coverage_seconds": full_seconds,
        "partial_coverage_seconds": partial_seconds,
        "zero_coverage_seconds": zero_seconds,
        "coverage_ratio": coverage_ratio,
        "batch_sha256": batch_source.sha256,
    }
    return "\n".join(report), metrics


def write_outputs(
    report_date: date,
    report: str,
    metrics: dict,
    start: datetime,
    end: datetime,
    notion_status: str,
    manifest: dict,
) -> None:
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    report_path = DAILY_DIR / f"{report_date.isoformat()}.md"
    report_path.write_text(report, encoding="utf-8")

    updated_at = datetime.now(UTC)
    event_id = f"daily_summary:{report_date.isoformat()}"
    existing_events = load_jsonl(EVENTS_PATH)
    event = {
        "event_id": event_id,
        "ts": iso_z(updated_at),
        "type": "daily_summary",
        "window_start": iso_z(start),
        "window_end": iso_z(end),
        **metrics,
        "notion_status": notion_status,
        "notion_page_id": manifest["notion_log_page_id"],
    }
    if not any(row.get("event_id") == event_id for row in existing_events):
        with EVENTS_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True, separators=(",", ":")) + "\n")

    state = {
        "updated_at": iso_z(updated_at),
        "last_report_date": report_date.isoformat(),
        "last_window_start": iso_z(start),
        "last_window_end": iso_z(end),
        "fleet_job_name": manifest["fleet_job_name"],
        "target_nodes": manifest["target_nodes"],
        **metrics,
        "notion_status": notion_status,
        "scheduled_log_job_id": manifest.get("scheduled_log_job_id"),
        "batch_script": manifest["batch_script"],
        "alert": None if metrics["coverage_ratio"] >= 1.0 else "Coverage below 100%; see daily gap intervals.",
    }
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    SUMMARY_PATH.write_text(
        "\n".join(
            [
                f"# {manifest['fleet_job_name']} Daily Coverage Summary",
                "",
                f"- Updated: `{state['updated_at']}`",
                f"- Window: `{state['last_window_start']}` to `{state['last_window_end']}`",
                f"- Submitted jobs: `{metrics['submitted_jobs']}`",
                f"- Slurm rows: `{metrics['slurm_rows']}`",
                f"- Full coverage: `{duration_text(metrics['full_coverage_seconds'])}`",
                f"- Partial coverage: `{duration_text(metrics['partial_coverage_seconds'])}`",
                f"- Zero coverage: `{duration_text(metrics['zero_coverage_seconds'])}`",
                f"- Node-minute coverage: `{metrics['coverage_ratio']:.4%}`",
                f"- Notion status: `{notion_status}`",
                f"- Daily report: `{report_path.relative_to(ROOT)}`",
                "",
                "No experiment action was taken. This system only records evidence.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="UTC date to report, default: yesterday")
    parser.add_argument("--write", action="store_true", help="Persist daily/state/event outputs")
    parser.add_argument(
        "--notion-status",
        choices=["pending", "updated", "failed"],
        default="pending",
    )
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    batch_source = load_batch_source(manifest)
    report_date = date.fromisoformat(args.date) if args.date else datetime.now(UTC).date() - timedelta(days=1)
    start = datetime.combine(report_date, time.min, tzinfo=UTC)
    end = start + timedelta(days=1)
    jobs, sacct_command = run_sacct(
        manifest["fleet_job_name"],
        start,
        end,
        manifest["accounting_lookback_hours"],
    )
    submissions = submissions_for_window(start, end)
    segments = build_segments(jobs, start, end)
    report, metrics = render_report(
        report_date,
        start,
        end,
        manifest["fleet_job_name"],
        manifest["target_nodes"],
        jobs,
        sacct_command,
        submissions,
        segments,
        batch_source,
    )
    if args.write:
        write_outputs(report_date, report, metrics, start, end, args.notion_status, manifest)
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
