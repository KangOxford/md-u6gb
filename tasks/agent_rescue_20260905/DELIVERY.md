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
22.68 billion inodes free. The registry has since been moved to `/home`, so that specific
failure cannot recur.

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
