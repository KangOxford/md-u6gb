#!/usr/bin/env python3
"""Monitor all GPUs in two existing multi-node Slurm allocations.

The script starts one read-only, overlapping Slurm step per allocation and keeps
that step alive for the whole monitoring session.  It therefore avoids creating
new Slurm steps on every refresh.  No allocation is submitted, cancelled, or
otherwise modified.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime as dt
import getpass
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import IO, Sequence


DEFAULT_JOB_IDS = ("5859913", "5856560")
DEFAULT_TARGET_GPUS = 32


# Executed once per node by srun.  Each task samples its node repeatedly and
# emits machine-readable rows.  DONE is emitted after all rows for one sample,
# so the login-node controller can render only complete eight-node snapshots.
REMOTE_PROBE = r'''
set -u
set -o pipefail

job_id=$1
interval=$2
sample_limit=$3
sequence=0
host=$(hostname -s)

while [ "$sample_limit" -eq 0 ] || [ "$sequence" -lt "$sample_limit" ]; do
    epoch=$(date -u +%s)

    nvidia-smi \
        --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit \
        --format=csv,noheader,nounits 2>/dev/null | \
    awk -F ', *' -v seq="$sequence" -v epoch="$epoch" -v jid="$job_id" -v host="$host" '
        BEGIN { OFS="|" }
        function clean(value) {
            sub(/^[[:space:]]+/, "", value)
            sub(/[[:space:]]+$/, "", value)
            gsub(/\|/, "/", value)
            return value
        }
        {
            for (i = 1; i <= NF; i++) $i = clean($i)
            print "GPU", seq, epoch, jid, host, $1, $2, $3, $4, $5, $6, $7, $8, $9
        }
    '
    gpu_rc=${PIPESTATUS[0]}

    if [ "$gpu_rc" -ne 0 ]; then
        printf 'ERROR|%s|%s|%s|%s|nvidia-smi GPU query failed with rc=%s\n' \
            "$sequence" "$epoch" "$job_id" "$host" "$gpu_rc"
    fi

    nvidia-smi \
        --query-compute-apps=gpu_uuid,pid,used_gpu_memory,process_name \
        --format=csv,noheader,nounits 2>/dev/null | \
    awk -F ', *' -v seq="$sequence" -v epoch="$epoch" -v jid="$job_id" -v host="$host" '
        BEGIN { OFS="|" }
        function clean(value) {
            sub(/^[[:space:]]+/, "", value)
            sub(/[[:space:]]+$/, "", value)
            gsub(/\|/, "/", value)
            return value
        }
        {
            for (i = 1; i <= NF; i++) $i = clean($i)
            name = $4
            for (i = 5; i <= NF; i++) name = name ", " $i
            print "PROC", seq, epoch, jid, host, $1, $2, $3, name
        }
    ' || true

    printf 'DONE|%s|%s|%s|%s|%s\n' \
        "$sequence" "$epoch" "$job_id" "$host" "$gpu_rc"

    sequence=$((sequence + 1))
    if [ "$sample_limit" -eq 0 ] || [ "$sequence" -lt "$sample_limit" ]; then
        sleep "$interval"
    fi
done
'''


@dataclasses.dataclass(frozen=True)
class Allocation:
    job_id: str
    owner: str
    state: str
    nodes: int
    gpus: int
    nodelist: str
    end_time: str

    @property
    def gpus_per_node(self) -> int:
        return self.gpus // self.nodes


@dataclasses.dataclass
class Snapshot:
    gpu_rows: list[list[str]] = dataclasses.field(default_factory=list)
    process_rows: list[list[str]] = dataclasses.field(default_factory=list)
    done_nodes: set[tuple[str, str]] = dataclasses.field(default_factory=set)
    errors: list[str] = dataclasses.field(default_factory=list)


class MonitorError(RuntimeError):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "通过两个已有 Slurm allocation 监控 8 节点 / 32 GPU；"
            "默认使用作业 5859913 和 5856560。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "job_ids",
        nargs="*",
        metavar="JOB_ID",
        help="要连接的 Slurm job ID；省略时使用脚本内置的两个 job",
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=int,
        default=5,
        help="两次 GPU 采样之间的秒数",
    )
    parser.add_argument(
        "-n",
        "--count",
        type=int,
        default=0,
        help="采样次数；0 表示持续运行到 Ctrl-C 或 allocation 结束",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="只采样一次，等价于 --count 1",
    )
    parser.add_argument(
        "--target-gpus",
        type=int,
        default=DEFAULT_TARGET_GPUS,
        help="启动前必须验证到的 GPU 总数",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="不清屏，保留每次历史快照（重定向到文件时会自动采用此模式）",
    )
    return parser


def require_commands(names: Sequence[str]) -> None:
    missing = [name for name in names if shutil.which(name) is None]
    if missing:
        raise MonitorError("缺少命令: " + ", ".join(missing))


def extract_field(line: str, name: str, default: str = "") -> str:
    match = re.search(rf"(?:^|\s){re.escape(name)}=([^\s]+)", line)
    return match.group(1) if match else default


def inspect_allocation(job_id: str) -> Allocation:
    try:
        result = subprocess.run(
            ["scontrol", "show", "job", "-o", job_id],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired as exc:
        raise MonitorError(f"查询 job {job_id} 超时") from exc

    if result.returncode != 0 or not result.stdout.strip():
        detail = result.stderr.strip() or "scontrol 没有返回作业信息"
        raise MonitorError(f"无法查询 job {job_id}: {detail}")

    line = result.stdout.strip()
    owner_field = extract_field(line, "UserId")
    owner = owner_field.split("(", 1)[0]
    state = extract_field(line, "JobState", "UNKNOWN")
    nodelist = extract_field(line, "NodeList", "-")
    end_time = extract_field(line, "EndTime", "-")

    try:
        nodes = int(extract_field(line, "NumNodes", "0"))
    except ValueError as exc:
        raise MonitorError(f"job {job_id} 的 NumNodes 无法解析") from exc

    alloc_tres = extract_field(line, "AllocTRES")
    gpu_match = re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)", alloc_tres)
    if gpu_match:
        gpus = int(gpu_match.group(1))
    else:
        tres_per_node = extract_field(line, "TresPerNode")
        per_node_match = re.search(r"(?:^|,)gres/gpu:(\d+)(?:,|$)", tres_per_node)
        gpus = nodes * int(per_node_match.group(1)) if per_node_match else 0

    if nodes <= 0 or gpus <= 0 or gpus % nodes != 0:
        raise MonitorError(
            f"job {job_id} 的资源无法解析为均匀 GPU 节点: "
            f"NumNodes={nodes}, AllocTRES={alloc_tres or '-'}"
        )

    return Allocation(
        job_id=job_id,
        owner=owner,
        state=state,
        nodes=nodes,
        gpus=gpus,
        nodelist=nodelist,
        end_time=end_time,
    )


def preflight(job_ids: Sequence[str], target_gpus: int) -> list[Allocation]:
    if not job_ids:
        raise MonitorError("至少需要一个 Slurm job ID")
    if len(set(job_ids)) != len(job_ids):
        raise MonitorError("job ID 不能重复")
    if any(not job_id.isdigit() for job_id in job_ids):
        raise MonitorError("job ID 必须是纯数字")

    current_user = getpass.getuser()
    allocations = [inspect_allocation(job_id) for job_id in job_ids]

    for allocation in allocations:
        if allocation.owner != current_user:
            raise MonitorError(
                f"job {allocation.job_id} 属于 {allocation.owner}，"
                f"当前用户是 {current_user}"
            )
        if allocation.state != "RUNNING":
            raise MonitorError(
                f"job {allocation.job_id} 当前是 {allocation.state}，不是 RUNNING"
            )

    total_gpus = sum(allocation.gpus for allocation in allocations)
    if total_gpus != target_gpus:
        detail = ", ".join(
            f"{allocation.job_id}={allocation.gpus} GPU"
            for allocation in allocations
        )
        raise MonitorError(
            f"GPU 总数校验失败: 检测到 {total_gpus}，要求 {target_gpus} ({detail})。"
            "如有意监控较少资源，请同时传入对应 job ID 和 --target-gpus。"
        )

    return allocations


def remaining_time(end_time: str) -> str:
    try:
        end = dt.datetime.fromisoformat(end_time)
    except ValueError:
        return "-"
    if end.tzinfo is None:
        end = end.replace(tzinfo=dt.timezone.utc)
    seconds = int((end - dt.datetime.now(dt.timezone.utc)).total_seconds())
    if seconds <= 0:
        return "已到期"
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def as_number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def compact_process_name(name: str, width: int = 52) -> str:
    if len(name) <= width:
        return name
    return "..." + name[-(width - 3) :]


def render_snapshot(
    sequence: int,
    snapshot: Snapshot,
    allocations: Sequence[Allocation],
    target_gpus: int,
    clear_screen: bool,
    runtime_messages: collections.deque[str],
) -> None:
    if clear_screen and sys.stdout.isatty():
        print("\033[2J\033[H", end="")
    else:
        print("\n" + "=" * 104)

    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"Slurm 32-GPU 监控 | 快照 {sequence + 1} | {now}")
    print("只读 nvidia-smi 采样；Ctrl-C 只停止监控 step，不会取消 allocation。")
    print()
    print(f"{'JOB':<10} {'STATE':<9} {'NODES':>5} {'GPUS':>5} {'REMAIN':>10}  NODELIST")
    for allocation in allocations:
        print(
            f"{allocation.job_id:<10} {allocation.state:<9} "
            f"{allocation.nodes:>5} {allocation.gpus:>5} "
            f"{remaining_time(allocation.end_time):>10}  {allocation.nodelist}"
        )

    process_count_by_uuid: collections.Counter[str] = collections.Counter()
    for row in snapshot.process_rows:
        if len(row) >= 9:
            process_count_by_uuid[row[5]] += 1

    def gpu_sort_key(row: list[str]) -> tuple[int, str, int]:
        try:
            job_number = int(row[3])
        except ValueError:
            job_number = 0
        try:
            gpu_number = int(row[5])
        except ValueError:
            gpu_number = 9999
        return job_number, row[4], gpu_number

    gpu_rows = sorted(snapshot.gpu_rows, key=gpu_sort_key)
    print()
    print(
        f"{'JOB':<10} {'NODE':<10} {'GPU':>3} {'UTIL':>6} "
        f"{'MEMORY MiB':>18} {'TEMP':>7} {'POWER W':>17} {'PROC':>5}"
    )
    for row in gpu_rows:
        if len(row) < 14:
            continue
        _, _, _, job_id, host, index, uuid, _, util, used, total, temp, power, limit = row[:14]
        memory = f"{used}/{total}"
        power_text = f"{power}/{limit}"
        print(
            f"{job_id:<10} {host:<10} {index:>3} {util + '%':>6} "
            f"{memory:>18} {temp + 'C':>7} {power_text:>17} "
            f"{process_count_by_uuid[uuid]:>5}"
        )

    unique_uuids = {row[6] for row in gpu_rows if len(row) >= 14}
    unique_nodes = {(row[3], row[4]) for row in gpu_rows if len(row) >= 14}
    models = sorted({row[7] for row in gpu_rows if len(row) >= 14})
    util_values = [
        value
        for row in gpu_rows
        if len(row) >= 14 and (value := as_number(row[8])) is not None
    ]
    memory_used = sum(
        value
        for row in gpu_rows
        if len(row) >= 14 and (value := as_number(row[9])) is not None
    )
    memory_total = sum(
        value
        for row in gpu_rows
        if len(row) >= 14 and (value := as_number(row[10])) is not None
    )
    active_by_util = sum(value > 0 for value in util_values)
    active_by_process = sum(
        uuid in process_count_by_uuid for uuid in unique_uuids
    )
    average_util = sum(util_values) / len(util_values) if util_values else 0.0
    max_util = max(util_values, default=0.0)

    print()
    print(
        "汇总: "
        f"{len(unique_nodes)}/{sum(a.nodes for a in allocations)} 节点 | "
        f"{len(gpu_rows)} 行 / {len(unique_uuids)} 唯一 GPU / 目标 {target_gpus} | "
        f"利用率 avg={average_util:.1f}% max={max_util:.0f}% | "
        f"util>0: {active_by_util} 卡 | 有计算进程: {active_by_process} 卡 | "
        f"显存 {memory_used / 1024:.1f}/{memory_total / 1024:.1f} GiB"
    )
    if models:
        print("型号: " + ", ".join(models))

    uuid_to_location = {
        row[6]: (row[3], row[4], row[5])
        for row in gpu_rows
        if len(row) >= 14
    }
    if snapshot.process_rows:
        print()
        print(f"{'JOB':<10} {'NODE':<10} {'GPU':>3} {'PID':>9} {'MEM MiB':>9}  PROCESS")
        process_rows = sorted(
            snapshot.process_rows,
            key=lambda row: (
                int(row[3]) if len(row) > 3 and row[3].isdigit() else 0,
                row[4] if len(row) > 4 else "",
                int(uuid_to_location.get(row[5], ("", "", "9999"))[2]),
                int(row[6]) if len(row) > 6 and row[6].isdigit() else 0,
            ),
        )
        for row in process_rows:
            if len(row) < 9:
                continue
            _, _, _, job_id, host, uuid, pid, used_memory, name = row[:9]
            gpu_index = uuid_to_location.get(uuid, (job_id, host, "?"))[2]
            print(
                f"{job_id:<10} {host:<10} {gpu_index:>3} {pid:>9} "
                f"{used_memory:>9}  {compact_process_name(name)}"
            )

    warnings: list[str] = []
    if len(gpu_rows) != target_gpus:
        warnings.append(f"本次仅收到 {len(gpu_rows)}/{target_gpus} 行 GPU 数据")
    if len(unique_uuids) != target_gpus:
        warnings.append(f"唯一 GPU UUID 为 {len(unique_uuids)}/{target_gpus}")
    expected_nodes = sum(allocation.nodes for allocation in allocations)
    if len(snapshot.done_nodes) != expected_nodes:
        warnings.append(f"完成采样的节点为 {len(snapshot.done_nodes)}/{expected_nodes}")
    warnings.extend(snapshot.errors)
    warnings.extend(runtime_messages)
    if warnings:
        print()
        print("警告:")
        for warning in warnings[-12:]:
            print(f"  - {warning}")

    sys.stdout.flush()


def stream_reader(
    stream: IO[str],
    job_id: str,
    channel: str,
    event_queue: queue.Queue[tuple[str, str, str]],
) -> None:
    try:
        for line in stream:
            event_queue.put((channel, job_id, line.rstrip("\n")))
    finally:
        event_queue.put((f"{channel}_eof", job_id, ""))


def build_srun_command(
    allocation: Allocation, interval: int, count: int
) -> list[str]:
    return [
        "srun",
        f"--jobid={allocation.job_id}",
        "--overlap",
        "--exact",
        f"--nodes={allocation.nodes}",
        f"--ntasks={allocation.nodes}",
        "--ntasks-per-node=1",
        f"--gpus-per-node={allocation.gpus_per_node}",
        "--cpus-per-task=1",
        "--unbuffered",
        "bash",
        "-c",
        REMOTE_PROBE,
        "monitor-probe",
        allocation.job_id,
        str(interval),
        str(count),
    ]


def start_streams(
    allocations: Sequence[Allocation], interval: int, count: int
) -> tuple[
    dict[str, subprocess.Popen[str]],
    queue.Queue[tuple[str, str, str]],
    list[threading.Thread],
]:
    processes: dict[str, subprocess.Popen[str]] = {}
    event_queue: queue.Queue[tuple[str, str, str]] = queue.Queue()
    threads: list[threading.Thread] = []

    for allocation in allocations:
        command = build_srun_command(allocation, interval, count)
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        if process.stdout is None or process.stderr is None:
            raise MonitorError(f"无法读取 job {allocation.job_id} 的监控输出")
        processes[allocation.job_id] = process

        for channel, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
            thread = threading.Thread(
                target=stream_reader,
                args=(stream, allocation.job_id, channel, event_queue),
                daemon=True,
            )
            thread.start()
            threads.append(thread)

    return processes, event_queue, threads


def stop_streams(processes: dict[str, subprocess.Popen[str]]) -> None:
    alive = [process for process in processes.values() if process.poll() is None]
    if not alive:
        return

    for process in alive:
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if all(process.poll() is not None for process in alive):
            return
        time.sleep(0.1)

    for process in alive:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    for process in alive:
        if process.poll() is None:
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                print(
                    f"警告: 监控进程 PID {process.pid} 未及时退出；"
                    "allocation 本身没有被取消。",
                    file=sys.stderr,
                )


def parse_structured_line(
    line: str,
    snapshots: dict[int, Snapshot],
) -> tuple[int | None, Snapshot | None]:
    parts = line.split("|")
    if len(parts) < 2 or parts[0] not in {"GPU", "PROC", "DONE", "ERROR"}:
        return None, None
    try:
        sequence = int(parts[1])
    except ValueError:
        return None, None

    snapshot = snapshots.setdefault(sequence, Snapshot())
    kind = parts[0]
    if kind == "GPU" and len(parts) >= 14:
        snapshot.gpu_rows.append(parts)
    elif kind == "PROC" and len(parts) >= 9:
        snapshot.process_rows.append(parts)
    elif kind == "DONE" and len(parts) >= 6:
        snapshot.done_nodes.add((parts[3], parts[4]))
        if parts[5] != "0":
            snapshot.errors.append(
                f"job {parts[3]} node {parts[4]} nvidia-smi rc={parts[5]}"
            )
    elif kind == "ERROR" and len(parts) >= 6:
        snapshot.errors.append(
            f"job {parts[3]} node {parts[4]}: {'|'.join(parts[5:])}"
        )
    return sequence, snapshot


def run_monitor(args: argparse.Namespace) -> int:
    require_commands(("scontrol", "srun"))
    job_ids = args.job_ids or list(DEFAULT_JOB_IDS)
    allocations = preflight(job_ids, args.target_gpus)
    expected_nodes = sum(allocation.nodes for allocation in allocations)

    print("已验证现有 allocation:")
    for allocation in allocations:
        print(
            f"  job {allocation.job_id}: RUNNING, {allocation.nodes} nodes, "
            f"{allocation.gpus} GPUs, {allocation.nodelist}, "
            f"EndTime={allocation.end_time}"
        )
    print(
        f"合计: {expected_nodes} nodes / {sum(a.gpus for a in allocations)} GPUs。"
        "正在启动每个 allocation 一个只读 overlap 监控 step..."
    )
    sys.stdout.flush()

    processes: dict[str, subprocess.Popen[str]] = {}
    interrupted = False
    ended_early: str | None = None
    rendered_sequences: set[int] = set()
    snapshots: dict[int, Snapshot] = {}
    runtime_messages: collections.deque[str] = collections.deque(maxlen=12)

    try:
        processes, event_queue, _ = start_streams(
            allocations, args.interval, args.count
        )
        stdout_eof: set[str] = set()
        reported_exit: set[str] = set()

        while True:
            try:
                channel, job_id, line = event_queue.get(timeout=0.5)
            except queue.Empty:
                channel = job_id = line = ""

            if channel == "stdout":
                sequence, snapshot = parse_structured_line(line, snapshots)
                if sequence is None or snapshot is None:
                    if line:
                        runtime_messages.append(f"job {job_id} stdout: {line}")
                elif (
                    sequence not in rendered_sequences
                    and len(snapshot.done_nodes) == expected_nodes
                ):
                    render_snapshot(
                        sequence=sequence,
                        snapshot=snapshot,
                        allocations=allocations,
                        target_gpus=args.target_gpus,
                        clear_screen=not args.no_clear,
                        runtime_messages=runtime_messages,
                    )
                    rendered_sequences.add(sequence)
                    snapshots.pop(sequence, None)
            elif channel == "stderr" and line:
                runtime_messages.append(f"job {job_id}: {line}")
            elif channel == "stdout_eof":
                stdout_eof.add(job_id)

            for current_job_id, process in processes.items():
                return_code = process.poll()
                if return_code is None or current_job_id in reported_exit:
                    continue
                reported_exit.add(current_job_id)
                runtime_messages.append(
                    f"job {current_job_id} 的监控 step 已退出，rc={return_code}"
                )
                if args.count == 0:
                    ended_early = current_job_id

            if ended_early is not None:
                break
            if args.count > 0 and len(stdout_eof) == len(processes):
                break

    except KeyboardInterrupt:
        interrupted = True
    finally:
        stop_streams(processes)

    if interrupted:
        print("\n监控已停止；仅终止监控 step，两个 allocation 未被 scancel。")
        return 130

    if ended_early is not None:
        try:
            final_state = inspect_allocation(ended_early).state
        except MonitorError:
            final_state = "UNKNOWN/已离开队列"
        print(
            f"\njob {ended_early} 的监控 step 已结束；当前作业状态: {final_state}。"
        )
        if final_state != "RUNNING":
            print("32-GPU 条件已不再成立，脚本已停止其余监控 step；未取消 allocation。")
            return 0
        print("作业仍为 RUNNING，说明监控 step 异常退出；请查看上方 srun 警告。")
        for message in runtime_messages:
            print(f"  - {message}")
        return 1

    expected_sequences = set(range(args.count)) if args.count > 0 else set()
    missing_sequences = sorted(expected_sequences - rendered_sequences)
    nonzero_steps = {
        job_id: process.returncode
        for job_id, process in processes.items()
        if process.returncode not in (None, 0)
    }
    if missing_sequences or nonzero_steps:
        if missing_sequences:
            print(f"缺少完整快照: {missing_sequences}", file=sys.stderr)
        if nonzero_steps:
            print(f"监控 step 非零退出: {nonzero_steps}", file=sys.stderr)
        for message in runtime_messages:
            print(f"  - {message}", file=sys.stderr)
        return 1

    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval 必须大于 0")
    if args.count < 0:
        parser.error("--count 不能小于 0")
    if args.target_gpus <= 0:
        parser.error("--target-gpus 必须大于 0")
    if args.once:
        if args.count not in (0, 1):
            parser.error("--once 不能和大于 1 的 --count 同时使用")
        args.count = 1

    def handle_term(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, handle_term)

    try:
        return run_monitor(args)
    except MonitorError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"系统错误: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
