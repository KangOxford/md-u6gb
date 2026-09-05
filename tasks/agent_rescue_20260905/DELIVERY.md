# Subagent disconnect recovery — what landed, what was verified, what is left

Date 2026-09-05. Implementation lives in
`/lus/lfs1aip2/projects/public/u6gb/.claude/agent_registry/`, which is a symlink onto VAST
`/home` and therefore not subject to the Lustre inode quota. That placement is load-bearing;
see §1.

## 1. The finding that reframes the problem

Nine subagents were lost with their session on 2026-09-04. The next session's report said,
nine times identically:

```
progress: 1705 lines at progress.md
prompt:   prompt.txt
```

Both paths are relative, so they resolved against the working directory and reported one
unrelated file as nine agents' banked findings. Underneath that, three claims in the
existing skill turned out to be false when measured rather than read:

| claim | measurement (2026-09-05) | verdict |
|---|---|---|
| the transcript lives only in node-local tmpfs and does not survive | 603 durable `agent-*.jsonl` across 70 session dirs under `~/.claude/projects/*/*/subagents/`, appended after every message | **false** |
| the nine agents banked nothing | all nine hold a complete 416 KB–5.6 MB transcript | **false** |
| their prompts are gone | the full prompt (1,369–2,349 chars) sits in the same registry row, field `prompt_inline` | **false — the report never read it** |
| `sol_notebook_fixes`: "NO TRANSCRIPT, re-create" | 5.6 MB present in both locations | **false**, superseded by an appended correction row |

So subagents already checkpoint, at **per-message** granularity, written by the harness at
zero cost to the agent — no extra tool calls, no context, no tokens. Nothing was ever lost.
What was missing is the step that turns a transcript back into instructions.

Why the agent-written layer was empty is recorded in the registry rows themselves:
`"prompt.txt writes failed on a full filesystem at 18:30Z"`. The Lustre project inode quota
was at its hard cap (still 50,463,435 / 51,200,000 = 98.6% today) while VAST `/home` had
inodes only 10.4% used. The registry has since been moved to `/home`, so that specific
**inode** failure cannot recur.

That guarantee does not extend to bytes, and the byte quota is the one currently binding:
measured 2026-09-05, `/home` is **100.20 GiB used of a 100.58 GiB hard limit (99.62%)** while
inodes sit at 10.43%. `df` reports 15 PB free because it measures the filesystem, not the
user. A 2 GiB test write also **succeeded** and only afterwards did `quota` report
`107161140*` — over the hard limit — so a completed write is not evidence of headroom either.
Bytes, writability and persistence are three independent questions and each has its own
command (see §5 of `task_plan.md`).

## 2. What was built

Existing scripts were extended rather than replaced; one new file was added after checking
it duplicated nothing (`journal.sh` records prospectively, `rescue` only copies, and no
`.py` under `.claude/` read a transcript).

| file | change |
|---|---|
| `reg_report.py` (new, was inline) | `pending` now reports three sources per agent — banked `progress.md`, prompt (file **or** inline), durable transcript — and says UNRECOVERABLE only when all three are empty. A work dir inferred from the slug convention is printed as `[INFERRED from slug]`, never as recorded. |
| `extract_progress.py` (new) | Mechanical rebuild of a transcript: the agent's own text, its full tool trail with the file/command each call touched, and every tool result carrying `is_error`. No model involved, so output is a pure function of the transcript. Prefers the durable `/home` copy over the rescued tmpfs snapshot. |
| `reg_recover.py` (new) | `recover <id\|slug>` / `--all`. Materialises `prompt.txt` from the inline copy, runs the rebuild, writes `RESUME.md`, and stamps what it consumed so a second run is a no-op. |
| `reg_verify.py` (new) | Six checks, each a regression test for a measured defect, plus `--self-test` that breaks each one and requires it to go red. |
| `agent_reg.sh` | Added `fail`, `recover`, `verify`. `add` now writes the prompt to **both** the sibling file and the registry row. |
| `skills/agent-rescue/SKILL.md` | Corrected the central claim; documented the new commands and the durable-copy ordering. |

### Which transcript copy to read

They are not interchangeable. Across the nine agents the durable `/home` copy was **larger
in 6 cases, equal in 3, smaller in none**; the worst gap was 728,471 B vs 486,755 B. The
rescued tmpfs copy is a point-in-time snapshot, so taking it drops exactly the steps closest
to where the agent stopped — the ones a replacement most needs. The three equal pairs are
agents whose rescue ran after they had already stopped.

## 3. Verification

| what | how | result |
|---|---|---|
| all nine recoverable | `agent_reg.sh pending` | 9/9 `RECOVERABLE` |
| artefacts real and non-empty | `stat` on each | prompt 1,369–2,353 B; rebuild 11–26 KB; `RESUME.md` 2.8–3.9 KB |
| failures recorded explicitly | rebuild header | 3/3/3/0/0/0/0/1/1 `is_error` results |
| **re-entry does not re-execute** | `recover --all` twice | first run **9 built**; second run **0 built, 9 already current** |
| checks can actually fail | `reg_verify.py --self-test` | **4/4 turned red** |
| checks pass on real data | `agent_reg.sh verify` | **6/6 pass** |
| end-to-end add/done/fail | smoke agent | prompt in both places; gone from `pending` after `done`; `fail` reported in its own section |
| hook | `session_start_hook.sh` | rescued 20 transcripts, reported normally |

The self-test found two of its own checks unable to fail on the first run. The cause was
that the trials mutated **raw registry rows**, and a raw row superseded by a later `done`
row is not in the merged live set — so the mutation landed on an agent the checks never
inspect. The self-test reproduced, in itself, the defect class it exists to catch. Fixed by
selecting targets from the merged view; 4/4 red afterwards.

## 4. Remaining

| # | item | why not done |
|---|---|---|
| R1 | no replacement agent has been run from a generated packet | that needs spawning an agent; this round's scope was the mechanism. The packet is a static file whose correctness is covered by `verify` and by reading it |
| R2 | the mechanical rebuild cannot preserve judgement — it gives the whole tool trail, not which step mattered | that is what the agent-written layer is for; both layers are kept |
| R3 | this session runs inside `SLURM_JOB_ID=6324128` step 8 on `nid010561`, so it dies when the allocation ends | known, and it is case 3 in the skill's table, which now has a recovery path |
| R4 | `.recover_stamp.json` compares size and mtime only | an in-place same-length rewrite would go undetected; JSONL transcripts are append-only, so this cannot occur |
| R5 | the work inside the nine packets has not been resumed | recovery and resumption are separate; resumption was not requested this round |

## 5. Untouched

Other sessions: none touched. The existing 15-minute GPU sweep was checked read-only
(`gpu_watch_15min.log` last written 03:56:24, three minutes before the check) and was
neither restarted nor stopped. `registry.jsonl` is append-only — the incorrect
`sol_notebook_fixes` description was superseded by a new row, not edited. Replaced scripts
were copied to `.bak_<timestamp>` following the existing convention; nothing was deleted.

---

# Round 2 (2026-09-05) — from "a packet exists" to "the work is done"

## 6. Reconciliation before recovery

Duplicate recovery is avoided by checking, per agent, whether the declared deliverable exists
and whether its mtime falls before or after that agent's last transcript message.

| slug | declared deliverable | measured | verdict |
|---|---|---|---|
| `plan_measurement` | `plan_section_2_measurement.md` | 34,760 B @ 22:28; agent alive to 22:29 | **finished by the agent itself** |
| `plan_deliverable` | `plan_section_5_deliverable.md` | 48,238 B @ 18:22 = its last message | **finished by the agent itself** |
| `sol_corrected_inference` | `corrected_inference.py` | 93,432 B @ **09-05 02:40**, 8 h after this agent died | **another session owns it — do not re-recover** |
| `plan_analysis` | `plan_section_4_analysis.md` | absent | genuinely unfinished → done this round |
| `plan_infrastructure` | `plan_section_3_infrastructure.md` | absent | genuinely unfinished → done this round |
| `sol_decisive_experiment` | `plan_section_1_decisive.md` | absent | genuinely unfinished |
| `sol_history` | none declared | no output of any name | genuinely unfinished |
| `sol_pipeline_fixes` | tests under `pipefix_.../` | 32 entries, but "mid-flight correcting test expectations when killed" | genuinely unfinished |
| `sol_notebook_fixes` | four builder scripts | only one touched, at the minute it died | genuinely unfinished |

The two finished agents were closed on the status axis so a later round cannot pick them up
again.

## 7. The four-stage ledger

A packet existing says nothing about the task being done. The old registry had one axis
(running/done/failed) and therefore could not express "the packet is there but nobody ran it",
which was the true state of all nine agents.

```
prepared   a RESUME packet exists
submitted  the packet was handed to an executor
processed  that executor ran to completion and said what it did
artifact   the declared output exists and is non-empty   <- stat'd, not asserted
```

The first three are testimony; the fourth is a measurement, and the ledger keeps them apart
deliberately. The `artifact` column prints `ABSENT` rather than `yes` when the measurement
found nothing — the first version printed `yes` for both, which is the label saying what the
measurement did not.

Implemented in `reg_stage.py`, wired as `agent_reg.sh stage` / `stages`.

## 8. Two recoveries actually delivered, sequentially

### `plan_analysis` → `/home/u6gb/kangli.u6gb/plan_section_4_analysis.md` (17,169 B)

The packet paid for itself immediately: the dead agent had banked four verification scripts
under `/home/u6gb/kangli.u6gb/plan4_verify/`, and **all four reran and reproduced their
recorded values**:

| quantity | rerun |
|---|---|
| rung-3 null sd (8-ticker mean) | 0.019468 (recorded 0.019468) |
| GOOG variance share / Kish n_eff | 0.6276 / **2.3652** |
| correct 5% band multiplier | **1.8964** (the ±2 sd band is a 3.68% band) |
| sign-flip smallest attainable p | **2/2⁸ = 0.0078125** |
| sign-flip measured size inflation | ×1.35 to ×1.87 |

Two statements were tightened against the source rather than repeated: the "120%" attribution
is a derived ratio (119.9% = 0.0969/0.0808), not a recorded field; and under a context shuffle
`qL1` and `sd_ratio` are **bit-identical** in `fix_midtraining.json`, not different by 2.2e-16.

### `plan_infrastructure` → `/home/u6gb/kangli.u6gb/plan_section_3_infrastructure.md` (14,613 B)

The banked transcript stopped one step short of a root cause, and said the remaining step was
CPU-only. Four shell signatures were reproduced here:

| test | result |
|---|---|
| `set -u` + redirect + unset variable | **file created at 0 bytes, command never ran** — matches the observed 0-byte `.done` |
| variable set but empty | 1 byte, so **0 vs 1 byte distinguishes unset from empty** |
| `set -e`, then `cmd; _rc=$?` | the guard is **unreachable** (`collect_rollouts.sh:182`) |
| `$?` beside a command substitution | **position decides**: `$?` is correct only if it precedes the substitution |

That last one sharpens FACTS.md's rule. Of five sites matching the loose pattern, **exactly one
is a real defect**: `eval_shard.sh:12`, where `$(date …)` precedes `$?` inside the `|| echo
"… FAILED rc=$?"` branch, so every failure is logged as `rc=0`. Also inventoried: **10 sites**
testing `.done` with `-f` (a 0-byte file passes all ten), **8 launchers** ending in `exec` so
their EXIT trap never fires, and **8 sites** reading `rc=$?` that must each be classified for
reachability.

**Not confirmed:** the previous agent reported seeing a 0-byte `.done`. Scanning
`/lus/lfs1aip2/projects/public/u6gb/nb_build_pr22/crps_res_kcollapse_20260904T163807Z/*/member_*/.done`
found **no `.done` files there at all**, so that sighting could not be reproduced. The
mechanism is established, the sighting is not, and the deliverable says both.

## 9. Verification of this round

| what | result |
|---|---|
| packet reusable end to end | 4/4 banked scripts reran and reproduced recorded values |
| artifact 1 | `plan_section_4_analysis.md`, 17,169 B, stage `artifact` = PRESENT |
| artifact 2 | `plan_section_3_infrastructure.md`, 14,613 B, stage `artifact` = PRESENT |
| consumption stamp | `recover plan_analysis` after delivery → **0 built, 1 already current** |
| registry checks | `verify` 6/6 green |
| ledger | both items show prepared → submitted → processed → artifact PRESENT |
| concurrency | two items run **sequentially**; the other seven were not started |

## 10. Remaining, and what is off limits

| slug | what is missing |
|---|---|
| `sol_decisive_experiment` | the power/cost/prespecification section (its patch is already applied) |
| `sol_history` | the whole chronological reconstruction; needs cross-month `sacct` and several roots |
| `sol_pipeline_fixes` | a fail-before/pass-after test per defect; connects directly to §8's inventory |
| `sol_notebook_fixes` | three of four builder scripts untouched |
| `sol_corrected_inference` | **off limits — another session is working on it** |

Nothing was merged. Other sessions and the existing GPU sweep were not touched. The registry
remains append-only and every replaced script is preserved as `.bak_<timestamp>`.
