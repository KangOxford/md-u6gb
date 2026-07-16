# 128 Nodes Node Status Lessons

## 2026-07-16

- For a node-status probe, keep the payload observability-only: one task per node is enough to verify allocation, hostnames, and GPU visibility.
- A one-minute walltime limits cost, but queued or accepted submission is not runtime evidence; require Slurm state and logs to distinguish pending from actual execution.
