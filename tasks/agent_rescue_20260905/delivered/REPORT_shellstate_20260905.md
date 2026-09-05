# Pipeline fixes, shell-state group — D1/D2/D3

Delivered 2026-09-05. Continues `sol_pipeline_fixes`, whose banked work already covered
P1 (K-collapse), P3 (.bad counted as done) and P4 (provenance) in
`t_p1_p4.py` and `t_p3.sh`. This adds the three defects those tests did not reach.

Test: `/home/u6gb/kangli.u6gb/pipefix_20260904T174156Z/t_shellstate.sh`
Every case **binds to the shipping file** — the predicate and the failure branch are read
out of the real script, so the test goes red on the code that ships and green only when
that file itself changes. A test written against a restatement of the logic proves nothing
about what runs.

| id | defect | file | fix |
|---|---|---|---|
| D1 | an empty `.done` is read as "member already collected" | `collect_rollouts.sh:95` | `_done_ok()`: `-s` **plus** a JSON parse |
| D2 | `set -e` kills the script before the guard that names the log | `collect_rollouts.sh:181-182` | `_rc=0` then `… || _rc=$?` |
| D3 | the failure branch logs `rc=0` | `eval_shard.sh:12` | read `rc=$?` on its own line, before any substitution |

## The failure cases are real, not mocks

D1's 0-byte `.done` is produced by the mechanism that actually produces it, and that
mechanism was **corrected during this work**:

```
set -u on an unbound variable aborts BEFORE the redirection
  -> no file is created, and an existing file is not truncated
a 0-byte file comes from the opposite order: redirection succeeds, writer then dies
  -> cat /nonexistent > f   leaves f at 0 bytes
```

An earlier draft claimed `set -u` leaves the 0-byte file. That was an artefact of a test
whose target had been created empty beforehand, and it is withdrawn. The conclusion is
unchanged and stronger: a successful redirect followed by a dying writer is exactly what an
allocation expiring mid-write looks like, so the pipeline will keep meeting this state.

D2's case runs a payload that exits 9 and asserts the operator is told **which log to read**.
The exit code always survived; the diagnostic did not. That is the error state being lost.

D3's case executes the failure branch **extracted from `eval_shard.sh`** against a failing
payload and reads the code it logged.

## BEFORE (run 2026-09-05T07:29:02Z, against the unfixed files)

```
D1  PASS  the failure case reproduces: .done is 0 bytes
    shipping predicate: 95:    if [ -f "$MEMBER_DIR/.done" ]; then
    FAIL  empty .done accepted as complete
          this is how a lost member is skipped on resume
    PASS  genuine .done still accepted

D2  probe rc=9
    PASS  exit code 9 propagates
    FAIL  error state lost: the guard never ran
          stderr was: <empty>

D3  shipping line: 12:  || echo "[$(date -u +%H:%M:%S)] shard $SHARD cell $i FAILED rc=$?: $STOCK $SEED"
    produced: [07:28:11] shard 0 cell 0 FAILED rc=0: X Y
    FAIL  a failure logged rc=0
          every failure in this log reads as a success

3 passed, 3 failed
```

## AFTER

```
workdir: /home/u6gb/kangli.u6gb/pipefix_20260904T174156Z/t_shell_final_073159

==============================================================
D1  an empty .done must NOT count as a collected member
==============================================================
  PASS  the failure case reproduces: .done is 0 bytes
  shipping predicate: 217:    printf '%s\n' "$MANIFEST" > "$MEMBER_DIR/.done"
  PASS  empty .done rejected
  PASS  genuine .done still accepted

==============================================================
D2  set -e must not destroy the error state before it is reported
==============================================================
  probe rc=9
  PASS  exit code 9 propagates
  PASS  the guard reported which log to read

==============================================================
D3  a failure branch must log the real exit code, not 0
==============================================================
  shipping line: 16:                 echo "[$(date -u +%H:%M:%S)] shard $SHARD cell $i FAILED rc=$rc: $STOCK $SEED"; }
  produced: [07:31:59] shard 0 cell 0 FAILED rc=1: X Y
  PASS  the failure logged a non-zero code

==============================================================
6 passed, 0 failed
```

## Regression signature retained

Rewriting D3's branch back to the old form still logs `rc=0`, so the test has not been made
green by weakening it:

```
$ bash -c 'false || echo "[$(date -u +%H:%M:%S)] shard 0 cell 0 FAILED rc=$?: X Y"'
[07:30:27] shard 0 cell 0 FAILED rc=0: X Y
```

## Not done in this pass

- The other **9 sites** testing `.done` with `-f` (`parent_cell.sh:31`, `run_v5w_dump.sh:25,37`,
  `supervise_round4_recheck.sh:22-24`, `collect_highpower.sh:45`, `run_ce_control.sh:15`,
  `launch_e13b.sh:64`, `e13b_slot.sh:37`). `_done_ok` is now available to them.
- The **8 launchers ending in `exec`**, which defeat their EXIT trap.
- `r3null_cell.sh:45` and `inter_cell.sh:60`, which still decide on the exit code rather than
  the artefact. `sweep_cell.sh` already decides correctly and is the pattern to copy.

## Safety notes

Files were replaced atomically (write to a temp name, then `os.replace`), so readers holding
the old inode were unaffected. Four `collect_rollouts.sh` processes were running at the time;
they belong to a different worktree (`tailfix-20260902`, inode 144120970343426879) than the
file edited here (`crps-return-alignment-20260808`, inode 144120874595787108), and were not
touched. Every replaced file has a `.bak_20260905T072947Z` sibling; nothing was deleted.
