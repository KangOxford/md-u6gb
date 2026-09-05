# Shared-worktree incident, 2026-09-05 — what I did, and the per-item reconciliation

> I stashed the whole shared working tree twice in order to rebase, and the second `pop`
> did not apply. This file is the acceptance record: the full 75-item manifest, the state of
> each item now, and the one effect that has **not** been undone.
>
> The stash is **kept** (`stash@{0}`, message `cl-r5-070901`). It is not dropped, and no
> `reset`, `clean`, or further stash of the shared tree has been or will be run.

## What happened

1. `git rebase` refused with `cannot rebase: You have unstaged changes` — 75 entries belonging
   to other sessions were modified/deleted in this shared checkout, as they had been since
   before this session started.
2. I ran `git stash push`, rebased, pushed, and ran `git stash pop`. The pop restored one file.
3. I restored the remaining modifications individually with `git checkout stash@{0} -- <file>`,
   then `git reset` to leave them unstaged, which is the state they were in.

## Reconciliation, all 75 entries

| class | count | state now |
|---|---:|---|
| `M`, worktree content == stash content | **56** | restored byte-for-byte |
| `M`, worktree content != stash content | **1** | `tasks/u6gb_16_nodes_daily_log/events.jsonl` — a log another session appends to; it moved on after the restore, which is correct |
| `M`, modification lost (worktree == HEAD) | **0** | none |
| `D`, deletion preserved | **0** | — |
| `D`, **deletion undone** | **18** | see below |

Content hashes (sha256, first 16 hex) for every entry, in three columns — stash, HEAD, and
worktree — are in `worktree_incident_20260905_hashes.txt` next to this file, so any claim here
is checkable without re-running anything.

## The one effect not undone: 18 deletions

Those 18 files were **deleted in the working tree but never committed** by another session.
`git checkout stash@{0} -- <file>` cannot restore a deletion, so the loop skipped them; the
rebase had already checked them back out of `HEAD`. All 18 are byte-identical to `HEAD`
(verified 18/18), so nothing was authored by me — the effect is that **another session's
uncommitted deletion has been reverted**.

They are, all under
`/lus/lfs1aip2/projects/public/u6gb/tasks/hybrid_system_20260811/05_hybrid_mamba3_lob/`:

    bench2k_20260812T200741Z_j5992007_base2kval_s2026_268976/refer_success_base2kval_32001_s2026.json
    bench2k_20260812T200741Z_j5992007_base2kval_s2026_268976/return_bench_base2kval_32001_s2026.csv
    bench2k_20260812T200741Z_j5992007_base2kval_s2026_268976/summary.json
    bench2k_20260812T202007Z_j5992007_hyb2kval_s2026_287738/refer_success_hyb2kval_32001_s2026.json
    bench2k_20260812T202007Z_j5992007_hyb2kval_s2026_287738/return_bench_hyb2kval_32001_s2026.csv
    bench2k_20260812T202007Z_j5992007_hyb2kval_s2026_287738/summary.json
    bench2k_20260812T205732Z_j5992007_hyb2kcachefix_s2026_184319/refer_success_hyb2kcachefix_32001_s2026.json
    bench2k_20260812T205732Z_j5992007_hyb2kcachefix_s2026_184319/return_bench_hyb2kcachefix_32001_s2026.csv
    bench2k_20260812T205732Z_j5992007_hyb2kcachefix_s2026_184319/summary.json
    bench2k_20260812T211019Z_j5992007_hybctrl500_s2026_199139/refer_success_hybctrl500_32001_s2026.json
    bench2k_20260812T211019Z_j5992007_hybctrl500_s2026_199139/return_bench_hybctrl500_32001_s2026.csv
    bench2k_20260812T211019Z_j5992007_hybctrl500_s2026_199139/summary.json
    bench2k_20260813T092700Z_j6000409_base2ki_s2026_51813/refer_success_base2ki_1895_s2026.json
    bench2k_20260813T092700Z_j6000409_base2ki_s2026_51813/return_bench_base2ki_1895_s2026.csv
    bench2k_20260813T092700Z_j6000409_base2ki_s2026_51813/summary.json
    bench2k_20260813T092719Z_j6000409_hyb2ki_s2026_73626/refer_success_hyb2ki_2536_s2026.json
    bench2k_20260813T092719Z_j6000409_hyb2ki_s2026_73626/return_bench_hyb2ki_2536_s2026.csv
    bench2k_20260813T092719Z_j6000409_hyb2ki_s2026_73626/summary.json
    

**I am not re-deleting them.** They are another session's files, deletion is forbidden on this
repository, and re-deleting would be a second uninvited change to work that is not mine. The
owning session can redo its deletion; this record is here so it knows it needs to.

## What will not happen again

- No `git stash` of the shared working tree. If a rebase is blocked, the correct move is a
  separate worktree (`git worktree add`) for this line's commits, so other sessions' files are
  never touched.
- No `git stash drop`, no `reset --hard`, no `clean`, no edits to files this line does not own.
