# Baseline 200K Resume Status

Created: 2026-06-22T10:57:47Z

Submitted baseline resume job `5333005` from `/lus/lfs1aip2/projects/public/u6gb/kangli/X-VLA`.

- Command: `MODEL=baseline SUBMIT=1 scripts/resume_200k_from_latest.sh`
- Resolved checkpoint: `runnings/baseline_200k_joint_7datasets/ckpt-40000`
- Resolved `START_STEP`: `40000`
- Output directory: `runnings/baseline_200k_joint_7datasets`
- Submission check: `PENDING (Priority)`

This is a resumed 24h baseline job, not evidence of completed 200K training yet. The next concrete gate is creation of `ckpt-50000`; the final target remains `ckpt-200000`.

Next resume check:

1. `squeue -j 5333005 -o "%.18i %.35j %.10T %.12M %.20S %.30R"`
2. `sacct -j 5333005 --format=JobIDRaw,JobName%35,State,ExitCode,Elapsed,Start,End,WorkDir%90 -P`
3. If running, inspect `logs/baseline_200k_5333005.out` for `start_step=40000` and `models='.../ckpt-40000'`.
