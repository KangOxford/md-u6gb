---
name: agent-rescue
description: Never lose a subagent's work to a disconnect, session limit, or node change. Registers every spawned agent, rebuilds a dead agent's work from its durable transcript, and lists what is unfinished so it can be resumed or re-created. Trigger on "agent rescue", "resume subagents", "recover agents", "lost agents", "抢救 agent", "续跑 agent", "agent 断了", or at the start of any session that spawned agents earlier.
user-invocable: true
arguments: "[pending|recover|rescue|add|done|fail|verify] — no argument runs rescue then pending"
allowed-tools: Bash, Read, Write, Edit
---

# agent-rescue

## What is actually true about where a subagent's work lives

A subagent is **not a separate process**. It is a concurrent conversation inside the parent
`claude` process, so it lives and dies with that process. But its transcript is written to
disk **as it goes**, in two places that are not equivalent:

```
~/.claude/projects/<project>/<SESSION>/subagents/agent-<id>.jsonl
    VAST /home. Durable, cross-node, appended after every message.   <-- the real checkpoint

/run/user/<uid>/claude-<uid>/<project>/<SESSION>/tasks/<id>.output
    tmpfs. RAM-backed, node-local, wiped on logout/reboot.           <-- a copy, not the copy
```

**Measured 2026-09-05**: 603 durable transcripts across 70 session dirs on `/home`. For the
nine agents lost the night before, the durable copy was **larger in 6 of 9 cases, equal in
3, and smaller in none** (worst gap `a6972058fc24931fa`: 728,471 B vs 486,755 B — the tmpfs
copy was a point-in-time snapshot and missed everything after it). Reading the rescued copy
when the durable one exists silently drops the steps closest to where the agent stopped,
which are the steps a replacement most needs. The three equal pairs are agents whose rescue
happened after they had already stopped, so the two copies agree — that is the best case,
not the general one.

An earlier version of this skill asserted that the transcript "lives in node-local tmpfs"
and does not survive. That was wrong, and it made the whole design pessimistic: it treated
transcripts as *evidence* rather than as *checkpoints*.

### The checkpoint you already have, for free

The harness writes that JSONL after **every message** — finer than any 3-to-5-minute
interval — and the agent pays nothing for it: no extra tool calls, no context, no tokens.
So the question is never "how do I make subagents checkpoint". They do. The question is
**what turns a transcript back into instructions**, and that is `recover` below.

## The layers, corrected

| layer | what it does | strength |
|---|---|---|
| 1 register | one line per spawn: id, slug, description, and the prompt **both** as a sibling file and inline in the row | after a disconnect you know what was running, and can re-issue it |
| 2 agent-banked notes | the agent appends findings to `work/<slug>/progress.md` as it goes | semantic and compact; **but it can fail** — see below |
| 3 durable transcript | written by the harness to `/home` after every message | **the guarantee.** Complete, free, survives node changes |
| 4 rescue | copy tmpfs `.output` files onto shared storage | redundancy only; routinely shorter than layer 3 |

Layer 2 is the one that fails, and it failed on 2026-09-04T18:30Z: the Lustre project inode
quota was at its hard cap, every `prompt.txt` sibling write failed, and none of the nine
agents banked a single line. Layer 3 was untouched, because `/home` is a different mount
with 22.6 billion inodes free. **Keep layer 2 — it is the agent's own judgement about what
mattered, which no mechanical rebuild can supply — but never let it be the only layer.**

## Commands

```bash
R=/lus/lfs1aip2/projects/public/u6gb/.claude/agent_registry

"$R/agent_reg.sh" pending            # unfinished agents + every recovery source per agent
"$R/agent_reg.sh" recover <id|slug>  # build the resume packet; re-running is a no-op
"$R/agent_reg.sh" recover --all
"$R/agent_reg.sh" verify             # 6 checks; `reg_verify.py --self-test` proves they can go red
"$R/agent_reg.sh" add <id> <slug> "<desc>" < prompt.txt
"$R/agent_reg.sh" done <id>
"$R/agent_reg.sh" fail <id> "<why>"  # a dead agent is not a finished one
"$R/agent_reg.sh" rescue             # layer 4; prints how many session dirs it scanned
```

`recover` writes, under `work/<slug>/`:

| file | what it is |
|---|---|
| `prompt.txt` | the spawn prompt, materialised from the inline copy in the registry row |
| `banked_from_transcript.md` | mechanical rebuild: the agent's own text, its full tool trail, and every tool result that carried `is_error` |
| `RESUME.md` | the packet to hand a replacement agent |
| `.recover_stamp.json` | transcript path + size + mtime consumed, so a second run skips |

**Re-entry is the point, not a nicety.** The reason to run `recover` is that something was
disconnected, and the likeliest next event is being disconnected again mid-recovery.
Verified 2026-09-05: first run built 9 packets, immediate second run built 0 and reported
9 already current. If the transcript has grown since the stamp, it rebuilds and says so.

## "Disconnected" is three different events

They differ in what survived, so they differ in what to do.

| event | parent process | agent id | durable transcript | what to do |
|---|---|---|---|---|
| **1. the user's SSH drops** | **alive** (if in tmux) | alive | alive | **nothing** — reconnect and carry on |
| **2. API error / session limit** | alive | alive | alive | `SendMessage` to the agent id |
| **3. teardown / node change / reboot** | dead | gone | **alive on /home** | `recover` → re-create from the packet |

Case 1 needs no action: Claude Code runs on the cluster, not on the user's laptop, and the
server's network to the API is unaffected by the laptop going away. **This holds only if
the parent is inside tmux on a login node.** A parent that is a direct child of `sshd` is
reaped with the connection; a parent running inside an `srun` step dies when the allocation
ends even though tmux held the connection.

Case 3 used to be the one that lost work. It no longer is — not because resumption got
better (resumption still needs the agent id and the runtime's in-memory state, and case 3
is defined by both being gone) but because layer 3 was there the whole time.

### Telling them apart

```bash
ls /run/user/$(id -u)/claude-$(id -u)/*/*/tasks/<agent-id>.output   # same session?
hostname                                                            # node changed?
```

## Spawning an agent so that losing it costs less

Layer 3 already covers you. This paragraph buys the *semantic* layer on top — the agent's
own judgement about what mattered, which a mechanical rebuild cannot reconstruct:

> **Write your findings as you go, not at the end.** After each meaningful step — a file
> read that settled something, a measurement, a conclusion you would not want to re-derive
> — append it to `<REGISTRY>/work/<SLUG>/progress.md`. Use `>>`, never rewrite the file.
> Each entry: what you did, what you found, and the file:line or command output backing it.
> Assume you will be killed without warning. Your final report is a summary of it, never a
> substitute for it.

Then register it:

```bash
"$R/agent_reg.sh" add <agent_id> <slug> "<one-line description>" < /path/to/prompt.txt
```

`add` now writes the prompt to **both** `work/<slug>/prompt.txt` and the registry row. A
prompt that exists in one place has one point of failure, and that failure already happened.

## The cost you cannot remove

Prompt cache expires on **wall-clock** (1 hour here), not on logical continuity:

| | input tokens | output tokens | wall clock |
|---|---|---|---|
| resume | re-read transcript (full price if cache is cold) | only what follows | continues immediately |
| re-create from packet | same re-read | **every prior tool call re-paid** | starts over |

Resume saves the output side and the wall clock, never the input side.

## Automatic firing

A `SessionStart` hook runs `rescue` then `pending`, and stays quiet when nothing is
unfinished:

```
/lus/lfs1aip2/projects/public/u6gb/.claude/agent_registry/session_start_hook.sh
```

wired in `/projects/public/u6gb/.claude/settings.json` under `hooks.SessionStart`. A rule
that has to be remembered is a rule that does not run, and the moment it is needed is
exactly the moment its context was lost.

## Cluster rules this skill obeys

Single-directory `ls`/`scandir` only — never recursive `ls -R`, `find`, `du -sh`, or `tree`.
Nothing is deleted: `rescue` uses `cp -n`, the registry is append-only (a wrong row is
superseded by a corrected one, never edited away), and `recover` overwrites only files it
owns under `work/<slug>/`. The registry itself sits on VAST `/home`, not Lustre, precisely
because the Lustre inode quota is the thing that broke layer 2.
