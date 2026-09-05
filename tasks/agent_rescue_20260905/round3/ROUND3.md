# Round 3 (2026-09-05) — shell-state fixes, decisive-experiment design, and a hard quota stop

## 1. Reconciliation (nothing had moved)

All four target tasks were unchanged since round 2, and the forbidden one was untouched:

| task | declared output | mtime | verdict |
|---|---|---|---|
| `sol_decisive_experiment` | `plan_section_1_decisive.md` | absent | unfinished |
| `sol_history` | none declared | no output of any name | unfinished |
| `sol_pipeline_fixes` | tests under `pipefix_20260904T174156Z/` | dir mtime 09-04 18:03, 32 entries | unfinished |
| `sol_notebook_fixes` | four builder scripts | one at 09-04 18:04, three older | unfinished |
| `sol_corrected_inference` | `corrected_inference.py` | 09-05 02:40, unchanged since round 2 | **off limits** |

A scare worth recording: **four `collect_rollouts.sh` processes were running** while a fix to
that filename was pending. They belong to a different worktree — `tailfix-20260902`, inode
144120970343426879 — than the file edited here — `crps-return-alignment-20260808`, inode
144120874595787108. Same basename, different file. Checking the inode rather than the name is
what made the edit safe; every edit was still done write-to-temp-then-`os.replace`, so any
reader holding the old inode was unaffected.

## 2. The three prioritised defects: fixed, red before, green after

Test: `/home/u6gb/kangli.u6gb/pipefix_20260904T174156Z/t_shellstate.sh`. Each case **binds to
the shipping file** — the predicate and the failure branch are read out of the real script and
executed — so it goes red on the code that ships and green only when that file changes.

| id | defect | fix |
|---|---|---|
| D1 | an empty `.done` read as "member already collected" (`collect_rollouts.sh:95`) | `_done_ok()`: `-s` **plus** a JSON parse |
| D2 | `set -e` kills the script before the guard that names the log (`collect_rollouts.sh:181-182`) | `_rc=0` then `… \|\| _rc=$?` |
| D3 | the failure branch logs `rc=0` (`eval_shard.sh:12`) | read `rc=$?` on its own line, before any substitution |

`BEFORE: 3 passed, 3 failed` → `AFTER: 6 passed, 0 failed`. Full output in
`/home/u6gb/kangli.u6gb/pipefix_20260904T174156Z/REPORT_shellstate_20260905.md`.

### A correction to round 2

Round 2 asserted that `set -u` on an unbound variable leaves a 0-byte file because the
redirection is performed first. **That is wrong and is withdrawn.** Measured here:

```
rm -f a; bash -c 'set -euo pipefail; printf "%s\n" "$NOPE" > a'   # rc=1, a is NOT created
printf 'previous\n' > b
bash -c 'set -euo pipefail; printf "%s\n" "$NOPE" > b'            # rc=1, b still 9 bytes
printf 'y\n' > c; bash -c 'set -euo pipefail; cat /nonexistent > c'  # rc=1, c IS 0 bytes
bash -c 'printf "%s\n" "" > d'                                    # rc=0, d is 1 byte
```

The round-2 observation came from a test whose target had been created empty beforehand, so
the 0 bytes were the setup, not the mechanism. The real mechanism is the opposite order: the
redirect succeeds and the writer then fails or is killed — which is exactly what an allocation
expiring mid-write looks like, so the conclusion (`-f` on `.done` is unsafe) is unchanged and
stronger. The delivered section has been corrected in place.

## 3. `sol_decisive_experiment` — delivered

`/home/u6gb/kangli.u6gb/plan_section_1_decisive.md`, 10,576 B. Power computed exactly; cost
measured from checkpoint mtimes and `sacct`, not estimated.

**Power** (paired t on k trajectories, two-sided 0.05, 80%), read against the trajectory-rung
floor sd = 0.0562:

| effect | k needed | against the null wrongly used (0.0195) |
|---|---|---|
| R +0.09 (the published headline) | **6** | 3 |
| \|R−1\| +0.034 (same result, correct transform) | **24** | 5 |
| \|R−1\| +0.0726 (round 4 vs round 3) | **7** | 3 |

Smallest detectable effect: k=3 → 0.1834, k=4 → 0.1196, k=6 → 0.0806, k=8 → 0.0650.

**Cost**, measured:

| quantity | source | value |
|---|---|---|
| s/step | `wm_ft_multi3`, 32 ckpts, steps 150→4800, span 4.92 h | **3.81 s** |
| cross-check | `wm_ft_multi4` clean segment 450→4800 on 09-02 | 3.58 s |
| one trajectory (4800 steps, 1 node × 4 GPU) | derived | **5.08 node-h = 20.3 GPU-h** |
| independent cross-check | `sacct` `unifw-train` COMPLETED | **04:18:04** |
| one eval cell (ckpt × ticker, K=4) | `sacct` `crps-*`: 17:12, 17:06, 17:28 | **17.2 min** |

`wm_ft_multi4`'s 157.25 h checkpoint span must **not** be used: steps 150→300 land on 08-27 and
step 450 on 09-02, so it was interrupted and resumed six days later.

Totals: k=6 → 88.5 node-h (354 GPU-h), 22 h wall on four nodes. k=24 → 353.9 node-h
(1,415 GPU-h), 88 h wall. **Recommended staging:** run k=6, read `s_trajectory` only, and
continue to k=24 only if it is ≤ 0.08.

## 4. `sol_notebook_fixes` — partial, and honest about it

The **four builder scripts were not edited.** What was done is the verification tooling:

- Five of six banked verify scripts run and produce results (`verify_d1_d3`, `verify_d2`,
  `verify_d5`, `verify_prov`, `verify_absdev_null`).
- **`verify_sweep.py` crashes** with `KeyError: 'seed'`, and the crash is correct.
  `sweep_results.jsonl` holds **two record shapes that are different estimands**: 265 per-seed
  rows `{arm,step,ticker,seed,sd_ratio,qL1}` and **48 pooled rows** `{arm,step,ticker,node,crps,…}`
  with **no `seed`**. Keying both by `(ticker,seed)` cannot work. Skipping the seedless rows —
  the obvious fix — would silently discard exactly the 48 rows that carry `crps`.
  The patch partitions them and prints both counts:
  `round3/verify_sweep_partition.patch`. **It could not be applied — see §5.**
- `verify_prov.py` surfaces a live provenance discrepancy worth its own follow-up:
  `unifw step 1200` has `stored abs_dev 0.10756`, `mean|R_t−1| 0.09238` and `|mean R − 1| 0.02438`
  — three different numbers for one nominal quantity — and `multi4 step 4200` stores
  `abs_dev 0.21463` while the recomputation is `nan`.

## 5. Hard stop: `/home` hit its byte quota mid-round

```
quota -u kangli.u6gb  ->  105494396* KB used / 105468748 KB hard   (25,648 KB OVER, flagged)
                          inodes 1,585,990 / 15,000,000
a 1-byte write returns Errno 122
```

**This was not caused by this round**: everything written under `$HOME` since 07:00Z totals
26 files and **2.55 MiB**, and the quota was already at 99.63% when the round began. Inodes rose
by 20,748 in the same hour, so other writers are active.

Three consequences, all recorded rather than worked around:

1. **The `verify_sweep.py` patch could not be applied.** It is delivered as a patch file.
   The target file was verified **intact** — `open(path,"w")` failed before truncating, and a
   re-run reproduced the original `KeyError` at the original line number.
2. **The ledger itself stopped.** `registry.jsonl` lives on `/home`, so `agent_reg.sh stage`
   raised Errno 122. The bookkeeping that exists to survive failures is on the filesystem that
   failed — the same shape as the 2026-09-04T18:30Z incident, where `prompt.txt` writes failed
   for the same reason. Rows that could not be appended are spilled to
   `round3/LEDGER_SPILL_20260905.jsonl` and must be replayed once headroom exists.
3. **Nothing was deleted to make room.** Per the standing rule, no `rm`. The only truncation
   was of this session's own test leftovers, which freed 0 bytes. In particular
   `.claude/agent_registry/transcripts/` (**103 MB** of rescued `.output` copies, redundant
   because the durable `/home` copy was ≥ the rescued one in every measured case) was
   **left alone** — freeing it is the user's call, not mine.

**A finding that belongs in the design, not just the incident log:** the durable transcript
store that makes agent-rescue work is itself an unbounded consumer of the quota that recovery
writes into. `.claude/projects` is **1,940 MiB** — 1,179.5 MiB across 259 session transcripts
and 404.2 MiB across 1,219 subagent transcripts — and it grows with every message. The
checkpoint is free in tokens and not free in bytes.

## 6. `sol_history` — scoped, not executed

The chronological reconstruction needs cross-month `sacct` plus reads across four roots and has
no declared output path. It was not started this round. Scoping it is recorded in the ledger
spill; no partial artefact was produced, and none is claimed.

## 7. Ledger semantics extended

A fifth stage was added: **`content`** — the output *contains* what the task required, checked
by a stated regex that can fail. `artifact` stats a path; `content` reads the bytes. Both are
measurements and are kept apart from the three testimony stages. Proved able to fail by
recording a deliberately absent pattern (`matches: 0, passed: False`) before re-recording the
true check.
