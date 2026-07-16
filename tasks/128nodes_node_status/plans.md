# 128 Nodes Node Status Plan

## 2026-07-16

- Create a dedicated `sbatch` file with `--nodes=128`, `--gpus-per-node=4`, `--exclusive`, and `--time=00:01:00`.
- Validate the batch syntax with `bash -n` and Slurm acceptance with `sbatch --test-only`. Done.
- Submit the job and record the returned job ID plus live queue state. Done for Job `5679501`.
- Do not cancel or modify the existing 16-node allocation request unless explicitly asked.
- Next step is runtime follow-up when Job `5679501` leaves `PENDING`: inspect `logs/128nodes_node_status/u6gb-128nodes-status-1min-5679501.out` and `.err`, then confirm whether all 128 node reports landed.
