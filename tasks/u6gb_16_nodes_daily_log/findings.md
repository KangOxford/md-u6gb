# Findings

- 2026-07-16: Baseline `sacct` and `squeue` evidence showed no `u6gb-16-nodes` jobs in the observed 24-hour window, so coverage was 0%.
- The daily Notion log page is `39f12c45-68fd-8106-a60b-e64bed63bd91`.
- `train_full_autoreg.batch` uses one task per node and four GPUs per task; `--nodes=16` therefore means 16 nodes and 64 H100 GPUs.
- Auto-resume renames jobs to `u6gb-16-nodes-resumeN`, so coverage must include that chain.
