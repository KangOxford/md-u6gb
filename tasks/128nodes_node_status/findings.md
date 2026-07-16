# 128 Nodes Node Status Findings

## 2026-07-16

- Task: submit a 128-node, one-minute Slurm experiment to check allocated node status.
- Scope is intentionally lightweight: allocate 128 full Isambard-AI nodes and sample per-node hostname, time, uptime, load average, and GPU visibility once.
- Existing top-level record files had unrelated uncommitted changes before this task, so this task uses a dedicated directory to keep the commit narrow.
- `sbatch --test-only` accepted the script and predicted a possible start at `2026-07-19T10:43:50` on 128 nodes.
- Real submission returned Job `5679501`.
- Live Slurm state immediately after submission: `PENDING (Resources)`, `NumNodes=128-128`, `ReqTRES=cpu=128,mem=57500G,node=128,billing=128,gres/gpu=512`, `NodeList=` empty, and `StartTime=Unknown`.
