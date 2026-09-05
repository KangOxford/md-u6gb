# Round 9 — pipeline preconditions repaired (CPU), and the exact GPU ask

Delivered from a node-local clone: the shared Lustre project is at **51,200,000 / 51,200,000
inodes**, so `git hash-object -w` in the worktree failed with `Disk quota exceeded`. Backups
for these edits were also written to node-local scratch for the same reason; the source files
were overwritten in place, same inode, with the original held in memory and read back after.

## What was repaired

Three call sites that would have corrupted the first GPU hour, all on the scoring path.

| file | defect | repair |
|---|---|---|
| `sweep_cell.sh:38` | `[ ! -f .done ]` — a 0-byte `.done` passed as a complete member | `-s` **plus** a JSON parse |
| `parent_cell.sh:31` | same `-f` test | same guard |
| `r3null_cell.sh:45` | `\|\| { echo "[gen] failed"; exit 7; }` — decided on the exit code, and the failure is intermittent, so good cells were discarded and the survivors were a biased sample | `_rc=0; … \|\| _rc=$?` then check the artefact; keep a complete member and report the nonzero code as an anomaly |
| `inter_cell.sh:60` | same | same |

`sweep_cell.sh.diff` is empty because that file was repaired before its backup was taken; its
current guard is visible in `sweep_cell.sh` at the `if [ ! -s "$OUT_ROOT/member_0/.done" ]`
line.

## Behavioural test, run before and after

Fixture on node-local scratch: one member with a valid `.done`, one with a 0-byte `.done`.

| case | old behaviour | new behaviour |
|---|---|---|
| complete member, `collect_rollouts.sh` exits **7** | `[gen] failed` → **cell discarded** | **KEPT**, anomaly logged, branch rc=0 |
| complete member, collect exits 0 | kept | kept, rc=0 |
| **0-byte `.done`**, collect exits 0 | **ACCEPTED as complete** | **REJECTED**, rc=7 |

## Not touched, deliberately

`run_v5w_dump.sh`, `supervise_round4_recheck.sh`, `collect_highpower.sh`, `run_ce_control.sh`,
`launch_e13b.sh`, `e13b_slot.sh` also test `.done` with `-f`. They drive **other experiments**,
are not on this scoring path, and are untracked with no shared-storage backup available, so
they were left alone rather than edited blind.

## The exact GPU ask

Nothing was started; this session holds **0 GPU workers**. What is ready to run:

```
# one cell = one trajectory x one ticker, K=4 seeds, ~17.2 min on 1 node x 4 GPU (measured)
STOCK=<ticker> SEEDS="97901 97902 97903 97904" \
ROOT=/local/user/$(id -u)/straj_<seed>_<jobid> \
CKPT=/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z/ckpt/wm_ft_traj_s<NN>_step4800 \
PACK_MEMBER=1 \
bash /lus/lfs1aip2/projects/public/u6gb/nb_build_pr22/sweep_cell.sh
```

- **output path**: node-local `ROOT`, four files per member copied back — required, the shared
  project has zero inodes free.
- **minimum useful request**: **1 node, 4 GPUs, ~2.5 h** → an 8-cell strip over
  `wm_ft_traj_s30…s37` on one ticker. Enough to separate a trajectory spread ≤ 0.03 from > 0.08.
- **full estimate**: 12 trajectories × 8 tickers = 96 cells ≈ 27.5 node-h.
- **still missing**: (a) a coordinated allocation/node/device, and (b) authorisation to score
  `wm_ft_traj_s30…s41`, which came from job 6317365 — another line.

## Scientific state

**C6 — does round 4's exit survive trajectory-to-trajectory noise — remains INSUFFICIENT.**
Twelve trajectories are trained; none is scored. No output in this repository decides it.
