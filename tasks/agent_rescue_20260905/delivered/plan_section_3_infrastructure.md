# Section 3 — Preconditions: what must be true before a measurement counts as evidence

Written 2026-09-05. Every mechanism below was reproduced on this machine on 2026-09-05; the
reproduction command is given so each claim can be re-checked in seconds. Where a claim comes
from someone else's observation and could **not** be reproduced here, it is marked as such
rather than repeated.

Code roots:
```
/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/crps-return-alignment-20260808/run/mid_training/
/lus/lfs1aip2/projects/public/u6gb/nb_build_pr22/
```

---

## 1. Precondition checklist

Each item is a test that **fails on the current code and passes on fixed code**. A
precondition that cannot go red is not a precondition.

### P1 — K is the number of ensemble members, not the number of directories

`compare_arms.py:165` `load_arm(root, member=0)` reads only `root/member_0`, and
`score_v5_primary.py:212` calls it once per arm **directory**. A run that writes
`member_0..3` into one root and is scored with `--arms a=cell b=cell` presents as K = 1;
`fair_crps` divides by `2k(k−1) = 0` and returns NaN. 32 of 32 cells were lost this way.

**Test (red now):** build a fixture with `member_0..3` under one root, score it, and assert
the record's effective K equals 4. Currently records K = 1.

**Not fixable by `--arms a=cell,cell,cell,cell`.** Four identical copies give a spread term of
exactly 0, and fair CRPS then degrades **silently** to mean absolute error. That is worse than
NaN: NaN stops the pipeline, a silently-wrong number does not.

**Second test (must also be red):** score four identical member directories and assert the
scorer **refuses**. A guard that only catches K < 2 does not catch K = 4-identical.

**Correct pattern, already present** in `sweep_cell.sh`: one `OUT_ROOT` per seed, joined at
scoring time as `--arms a=s97901,s97902,s97903,s97904`.

### P2 — `.done` must be tested for content, not existence

`.done` carries the provenance manifest, and every consumer tests it with `-f`. **Ten sites
use `-f`; none uses `-s`:**

```
collect_rollouts.sh:95        skips the member as "already collected"
parent_cell.sh:31             declares the cell a success
run_v5w_dump.sh:25,37   supervise_round4_recheck.sh:22,23,24
collect_highpower.sh:45  run_ce_control.sh:15  launch_e13b.sh:64  e13b_slot.sh:37
```

A zero-byte `.done` passes all ten. That state has a specific and **reproducible** cause:
under `set -u`, a redirection is performed before the command's word expansion fails, so the
target file is created and truncated and the command never runs.

```bash
# reproduced 2026-09-05
$ bash -c 'set -euo pipefail; printf "%s\n" "$NO_SUCH_VAR" > f'   # rc=1, f is 0 bytes
$ bash -c 'set -euo pipefail; E=""; printf "%s\n" "$E" > f'       # rc=0, f is 1 byte
```

The two cases are distinguishable: **0 bytes means the variable was unset, 1 byte means it was
set and empty.** That turns a puzzling artefact into a discriminating test.

`collect_rollouts.sh` writes `.done` at lines 194 and 203, both `printf '%s\n' "$MANIFEST" >
…`, and `MANIFEST` is built at line 142.

**Honest limit:** a previous agent reported observing a zero-byte `.done` next to a 769-byte
one. Scanning
`/lus/lfs1aip2/projects/public/u6gb/nb_build_pr22/crps_res_kcollapse_20260904T163807Z/*/member_*/.done`
on 2026-09-05 found **no `.done` files at all** there, so that observation is **not confirmed
here**. The mechanism above is reproduced; the specific sighting is not. Both facts are
recorded because acting on the mechanism is cheap and correct either way.

**Test (red now):** create a zero-byte `.done`, run the resume path, and assert the member is
**not** skipped. All ten sites currently skip it.

**Fix:** `[ -s "$d/.done" ] && python3 -c 'import json,sys;json.load(open(sys.argv[1]))' "$d/.done"`.
Existence is not integrity; a manifest that does not parse is not a manifest.

### P3 — Under `set -e`, `cmd; rc=$?` is dead code

```bash
# reproduced 2026-09-05
$ bash -c 'set -euo pipefail; false > /dev/null; _rc=$?; echo "GUARD REACHED"'
# prints nothing; outer rc=1
```

`collect_rollouts.sh:11` sets `set -euo pipefail`; line 181 runs inference redirected to a
log; line 182 reads `_rc=$?`; lines 186–189 are the guard, added with the comment "Without
this the old path's behaviour carried over: a crashed inference still got its `.done`".
**Under `set -e` the script exits at line 181 and lines 182–189 never run.**

The protection is real but it comes from `set -e`, not from the guard. That matters because
the guard reads as the protection: anyone who makes the call resilient (`|| true`, `set +e`,
wrapping it in an `if`) removes the actual protection while leaving the apparent one in place,
and the change looks safe in review.

Eight sites share the shape:
```
collect_rollouts.sh:182   sweep_cell.sh:31   parent_cell.sh:30   traj_cell.sh:37
supervise_round4_recheck.sh:69   e13b_slot.sh:65,98   traj3_fleet.sh:26
```
`sweep_cell.sh` and `parent_cell.sh` guard the call so their reads are live; the others must
each be classified before the guard is trusted.

**Test (red now):** assert that the guard's error message appears when inference exits
non-zero. It does not; the `set -e` message appears instead.

### P4 — Success is the artefact, never the exit code

`collect_rollouts.sh` has been observed returning non-zero with the member fully written.
Three call sites, three different behaviours:

| site | decides on | verdict |
|---|---|---|
| `sweep_cell.sh:30-35` | the product, with the reasoning in a comment | **correct** |
| `parent_cell.sh:31` | `[ -f member_0/.done ]` | artefact-based but uses `-f`, so P2 defeats it |
| `r3null_cell.sh:45`, `inter_cell.sh:60` | `\|\| { echo "[gen] failed"; exit 7; }` | **exit-code only — discards good data** |

The last two are the reported `[gen] failed … exit 7`. The failure is **intermittent**, and
intermittent discarding of good cells is worse than always failing: it produces a
**silently biased sample**, keeping the cells that happened to exit cleanly.

**Test (red now):** make `collect_rollouts.sh` return 7 after writing a complete member;
assert the cell is still counted. `r3null_cell.sh` and `inter_cell.sh` fail this.

### P5 — A failed cell must not count as a finished one

A failed cell writes `<result>.tmp.<pid>.bad` **into the results directory**, and the progress
counter tallies it as progress; `cell_json.py:190` records that this is exactly how 32 `.bad`
files were counted as 32 results. `crps_cell.sh:62` and `unifw_cell.sh:56` have since been
changed to record failures under a separate `failures/` tree.

**Test (red now):** plant one `.bad` in a results directory and assert the progress count does
**not** increase.

**Rule:** failures never live in the success namespace. `<results>/` holds successes;
`<results>/../failures/` holds failures; the counter reads only the first.

### P6 — A launcher must not end in `exec`

`exec` replaces the shell, so the `EXIT` trap that copies node-local results back never fires.
Two runs reached step 4800 over ~4h20m and left nothing, ~35 GPU-hours. **Eight launchers
still end this way:**

```
run_wmle_ft.sh   unifw_train.sh   repro_m3_cell.sh   k10_collect.sh
r1_msft_collect.sh   ft_arm.sh   cell2.sh   repro_m3_cell2.sh
```

**Test (red now):** send SIGTERM to a launcher mid-run and assert the partial results are on
shared storage. Also test normal exit and the `exec` path explicitly, because they fail
differently.

**Fix:** run the payload as a child (`"$@" & wait $!`) so the trap survives, and copy back
**incrementally** rather than once at the end. A trap is a single point of failure at exactly
the moment of failure.

---

## 2. Silent-failure inventory beyond the known four

### S1 — `$?` read after a command substitution **that precedes it on the same line**

The general form recorded in FACTS.md is slightly too broad. Position decides:

```bash
# reproduced 2026-09-05
$ bash -c 'false; echo "before-sub rc=$? then $(true)"'   # rc=1  correct
$ bash -c 'false; echo "$(true) then after-sub rc=$?"'    # rc=0  WRONG
```

`$?` is safe when it appears **textually before** any substitution in the same word list.
Five live sites match the loose pattern; classifying each by position leaves **exactly one real defect**:

| site | order | verdict |
|---|---|---|
| `run_onpolicy_loop.sh:61` | `$?` first | ok |
| `inter_node.sh:26` | `$?` first | ok |
| **`eval_shard.sh:12`** | **`$(date …)` first** | **BROKEN — every failure logs `rc=0`** |
| `node_run.sh:14` | `$?` first | ok |
| `node_batch.sh:19` | `$?` first | ok |

`eval_shard.sh:12` is the `|| echo "… FAILED rc=$?"` branch, so it is the *failure* logger
that reports success. **Fix:** capture `rc=$?` on its own line first.

### S2 — Existence tests standing in for integrity tests

P2 is one instance of a class: `-f`, `-d` and "file is present" are used where the question is
"is the content usable". Every such test on a file that carries data (`.done`, `*.json`,
`*.npz`, `daymap.json`) must be `-s` plus a parse.

### S3 — Node-local storage that dies with the allocation

`/local/user/$(id -u)/` is per-node and vanishes when the allocation ends; 7 fair-CRPS records
were lost when job 6266774 expired at 16:42Z. This is not a bug in any script — it is a
storage rule being violated (§4).

### S4 — Progress logs that are written by the thing they audit

`crps_panel.sh` recorded `rc=0` for every cell while cells were exiting 7 (S1's mechanism).
A progress log written by the dispatcher measures the dispatcher's belief, not the outcome.
**Rule: the completion count is computed by re-scanning artefacts, never accumulated.**

---

## 3. Provenance standard

A record that asserts parameters it did not use is worse than no record. Current records
assert the requested `K` while the estimator used 1.

**Every result record must carry, and the writer must refuse to write it if any is absent:**

| field | why |
|---|---|
| `k_effective` | the K the estimator actually used, computed from the member list it read |
| `arm_dirs` | the exact directory list read, absolute paths |
| `ckpt_path`, `ckpt_step` | the model, not the run label |
| `seeds` | the full seed list, not the count |
| `indices_sha256` | content hash of the context-index file |
| `code_sha` | git rev of the scoring code, plus `dirty` if the tree is modified |
| `node`, `jobid`, `stepid`, `generated_at` | where and when |

**Refusal rule:** the writer recomputes `k_effective` and `arm_dirs` from what it read and
**raises** if they disagree with what was requested. A record whose asserted parameters differ
from its actual ones must not exist, because it is indistinguishable from a correct one later.

`collect_rollouts.sh:142-167` already builds this manifest into `.done`. The gap is that
**nothing reads it** (the comment at line 139 says so explicitly: "nothing anywhere reads
`.done`'s contents, only its existence"). P2 closes that gap, and doing so costs no inodes,
which is the reason the manifest was put inside `.done` in the first place.

---

## 4. Storage rules that survive allocation expiry

| rule | reason |
|---|---|
| Node-local scratch is a **staging area, never a destination** | `/local/user/$(id -u)/` dies with the allocation, without warning |
| Copy back **incrementally**, per member, not per job | an expiry at 95% must leave 95%, not 0% |
| Never rely on an `EXIT` trap as the only copy-back | `exec` defeats it (P6), and so does SIGKILL |
| Results go to `/home/u6gb/kangli.u6gb/` | the Lustre project inode quota is at its cap |
| **But check the `/home` byte quota before every write campaign** | measured 2026-09-05: **100.20 GiB used of a 100.58 GiB hard limit, 99.62% full, 0.39 GiB free**, with inodes only 10.4% used. `df` reports the 15 PB filesystem and is irrelevant |
| Treat a successful write as **no evidence** of headroom | a 2 GiB write succeeded and only afterwards did `quota` report `107161140*` — over the hard limit, flagged. The accounting lags the write by seconds |
| One `OUT_ROOT` per seed, unique names, never reused | removes the need to clear anything, so no `rm` is ever required |

**The `/home` guarantee, corrected.** "The registry moved to `/home`, which has billions of
free inodes, so the full-filesystem failure cannot recur" is **half true**. It fixes the
*inode* failure, which is the one that occurred. It does not fix the *byte* failure, and the
byte quota is the one currently at 99.62%. The three questions are independent and all three
must be asked separately:

```bash
quota -u $(id -un)                 # 1. bytes AND inodes, per user -- the binding limit
: > "$TARGET_DIR/.probe" && echo w # 2. can I actually write here, now
findmnt -no SOURCE,FSTYPE /home    # 3. is it persistent, or scratch that gets purged
```

`df` answers none of them.

---

## 5. The smallest end-to-end validation run

Before any GPU measurement counts, one cell must pass this, and it is the **cheapest** thing
that exercises every precondition above:

**Shape:** one ticker, K = 4 seeds, one checkpoint, the smallest `n_sequences` that still
produces a finite fair-CRPS, run with the `sweep_cell.sh` pattern (one `OUT_ROOT` per seed).

**It passes only if all of these hold:**

1. The record reports `k_effective = 4` and lists four distinct `arm_dirs` (P1).
2. Deliberately corrupting one `.done` to zero bytes makes the run **fail loudly**, not skip
   the member (P2).
3. Forcing inference to exit non-zero produces the guard's message, not a bare `set -e`
   death (P3).
4. Forcing `collect_rollouts.sh` to exit 7 **after** writing a complete member still counts
   the cell (P4).
5. A planted `.bad` does not raise the progress count (P5).
6. SIGTERM mid-run leaves the already-generated members on shared storage (P6).
7. `eval_shard`-style failure logging reports the real non-zero code (S1).
8. The record's `code_sha`, `indices_sha256` and `seeds` are present and the writer refuses a
   record whose asserted `K` disagrees with `k_effective` (§3).

Items 2–7 are **negative** tests: they require deliberately breaking something and checking
that the pipeline notices. A validation run made only of positive tests certifies nothing,
because every one of the four known defects passes a positive test — that is what made them
silent.

**Ordering.** Run items 1–8 on CPU-only fixtures first; only item 1 and the finiteness of
fair-CRPS need a GPU. The GPU part of this validation is one cell, minutes; the rest costs
nothing and must be green before a card is requested.
