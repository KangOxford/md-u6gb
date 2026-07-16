# Findings

- 2026-07-16: Baseline `sacct` and `squeue` evidence showed no `u6gb-16-nodes` jobs in the observed 24-hour window, so coverage was 0%.
- The daily Notion log page is `39f12c45-68fd-8106-a60b-e64bed63bd91`.
- `train_full_autoreg.batch` uses one task per node and four GPUs per task; `--nodes=16` therefore means 16 nodes and 64 H100 GPUs.
- Auto-resume renames jobs to `u6gb-16-nodes-resumeN`, so coverage must include that chain.
- The primary daily fact is actual 16-node RUNNING coverage; command and hash provenance are secondary audit detail.
- Job `5678750` was accepted for 16 nodes / 64 GPUs / 23:59 and renamed in place to `u6gb-16-nodes-18-jluy-001`; it is still PENDING for Priority.
- Jobs `5678908` and `5678913` were intentionally cancelled at `2026-07-16T14:33:13Z` with zero runtime; Job `5678750` is the only active candidate and has not failed.
- Queue convergence is an outer-monitor responsibility; the submitted allocation payload must remain minimal.
- Job `5678750` has baseline priority `1`; all cluster priority weights are zero, hidden jobs/reservations prevent attribution beyond `Reason=Priority`, and no start estimate is available.
- `scontrol wait_job` is available and blocks until a candidate's nodes are usable or the candidate terminates.
