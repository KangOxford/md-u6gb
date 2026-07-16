# 128 Nodes Node Status Findings

## 2026-07-16

- Task: submit a 128-node, one-minute Slurm experiment to check allocated node status.
- Scope is intentionally lightweight: allocate 128 full Isambard-AI nodes and sample per-node hostname, time, uptime, load average, and GPU visibility once.
- Existing top-level record files had unrelated uncommitted changes before this task, so this task uses a dedicated directory to keep the commit narrow.
