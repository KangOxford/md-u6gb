# 128 Nodes Node Status Plan

## 2026-07-16

- Create a dedicated `sbatch` file with `--nodes=128`, `--gpus-per-node=4`, `--exclusive`, and `--time=00:01:00`.
- Validate the batch syntax with `bash -n` and Slurm acceptance with `sbatch --test-only`.
- Submit the job and record the returned job ID plus live queue state.
- Do not cancel or modify the existing 16-node allocation request unless explicitly asked.
