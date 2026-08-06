# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# 📒 Per-Round Recording Discipline (MANDATORY, 2026-05-28)

**Every conversation round MUST append timestamped, ID-prefixed entries to all four local task-record files in the workspace root (`/projects/public/u6gb/`):**

| File | ID prefix | Example |
|------|-----------|---------|
| `plans.md` | `P###` | `P001 UTC 2026-05-28T22:50:00Z: ...` |
| `findings.md` | `F###` | `F002 UTC 2026-05-28T22:50:00Z: ...` |
| `progress.md` | `PG###` | `PG003 UTC 2026-05-28T22:50:00Z: ...` |
| `learnt_lessons.md` | `L###` | `L004 UTC 2026-05-28T22:50:00Z: ...` |

- IDs are sequential per file, never reused, monotonically increasing.
- Always include a UTC timestamp after the ID (`UTC <ISO8601>`).
- Append with `echo >>` / Edit, never overwrite existing entries.
- This applies to EVERY round, including interrupts and corrections.

# 🚨🚨🚨 ISAMBARD-AI / LUSTRE Cluster Safety (Highest Priority, 2026-05-08)

**Isambard-AI is a BriCS national-level HPC, with 1000+ users sharing the Lustre parallel filesystem. Metadata storms triggered by LLM agents have already caused the entire group's jobs to be suspended once (BriCS administrators documented this explicitly in `/lus/lfs1aip2/projects/public/s5e/quant_team/quant/isambard-requirement`). Any violation of the rules below may lead to the team being banned again. These rules take precedence over all other content in this file, including "First Principles."**


## 1. Absolutely Prohibited Commands (DO NOT EXECUTE)

The following commands can create metadata storms for the Lustre MDT and are **forbidden, even for single "investigative" executions**:

```bash
ls -lR <path>            # Any recursive ls
ls -R <path>
ls -1td <dir>/*          # ls + time sort, hits every inode
ls | grep | sort         # ls followed by pipe (typical anti-pattern for finding checkpoints)
ls -la <large_dir>       # Directories with > 50 entries are considered large; do not try if size is unknown
find / ...               # Any broad find starting from /, /projects, or /home
find /projects ...
find /home ...
du -sh <large_path>      # Recursive stat
tree <lustre_path>       # Recursive listing
locate ...               # System-level search (depends on global index scanning)
**/*.ext                 # Deep recursive glob
*.py                     # Glob expansion in large directories (>10 files is unsafe)
watch ls /path           # Continuous metadata load
while true; do ls; sleep 5; done
inotifywait <lustre_path>
```

### 1.1 `scancel` is FORBIDDEN — no exceptions (2026-08-06)

**Claude must never run `scancel` in any form.** Not on a job, not on a job step,
not with `-s`, `-f`, `-u`, `--me`, or any other flag. If a job or step needs to be
stopped, **tell the user and let them do it**.

```bash
scancel <jobid>              # FORBIDDEN
scancel <jobid>.<stepid>     # FORBIDDEN — see below, the step form is not safe either
scancel -s KILL ...          # FORBIDDEN
scancel -f ...               # FORBIDDEN
scancel -u <user> / --me     # FORBIDDEN
```

**Why the step form is also forbidden.** `scancel <job>.<step>` looks scoped to one
step, but **adding any flag makes Slurm drop the `.<step>` suffix and act on the whole
job**. On 2026-08-06 a stuck training step was not responding to plain
`scancel 5924043.27`, so `scancel -s KILL 5924043.27` and `scancel -f 5924043.27` were
issued. Slurm answered
`scancel: error: Kill job error on job step id 5924043: No error` — note the missing
`.27` — and killed the entire 4-node allocation that had 22 hours left on it. It
survived only because that job happened to carry `Requeue=1`. The two spellings differ
by one flag and the blast radius differs by an entire allocation, so the whole family
is off-limits rather than "use it carefully."

**Do this instead**: report the stuck job/step id to the user with what is wrong, and
wait. A stuck step wastes minutes; a cancelled allocation wastes hours and re-queues
behind everyone else.

**Trigger Reflex Check**: Whenever I feel like using `find` / recursive `ls` / `du -sh` / `tree` / large directory glob to "understand project structure" or "find files," **STOP**. Use the safe alternatives below, or ask the user directly.

## 2. Safe Alternatives

```bash
# Lustre native metadata-batched find (far safer than GNU find)
lfs find <narrow_path> -name "*.py"

# Read known exact paths directly
cat /known/exact/path
head -N /known/path
tail -N /known/path
grep "pattern" /known/single/file        # Single file grep is always OK
grep -r "pattern" /small/src/tree        # ONLY if src tree has < 100 files
grep -r "pattern" /path --include='*.py' # Must use --include to narrow scope

# To find the latest checkpoint → read the breadcrumb file, do not ls the checkpoint dir
cat <run_dir>/latest_checkpoint.json

# If you don't know what's in a directory → ask the user, do not probe with ls
```

`ls /known/dir` should only be used if **fewer than 50 entries are expected**; if unsure, ask the user.

## 3. Checkpoint / Training Script Mandatory Requirements

Any batch script or training code must satisfy:

1. **Write to node-local storage, not Lustre**: All checkpoints, tensorboard logs, and training logs must be written to `$TMPDIR` or `/local/scratch/` during the run. Only `rsync -a $TMPDIR/... /projects/...` after the job finishes.
2. **Use Breadcrumb files for resume**: Rank 0 must **atomically write** a small `latest_checkpoint.json` (containing path + step) after every save. Auto-resume must read this file. **Strictly forbidden**: using `ls -1td` / `ls | grep | sort` on the checkpoint directory.
3. **Per-job isolated Output directories**: `OUTPUT_DIR=/projects/.../runs/${SLURM_JOB_ID}/` to avoid directory-level lock contention.
4. **Stage model weights first**: Rank 0 should copy weights to `$TMPDIR` and then broadcast. Multiple ranks are forbidden from reading large weight files from Lustre simultaneously.
5. **Use local cache for W&B, but keep runs online**: set `WANDB_DIR=$TMPDIR` and `WANDB_MODE=online` for training/eval jobs that should be tracked. Do not write W&B tokens into `CLAUDE.md`, Notion pages, shell scripts, or committed files; use the existing login state or a protected environment/secret source.
6. **Tighten Checkpoint intervals**: Save every N steps instead of every step. Choose N such that "recovering once wastes < 10% of job time."

## 4. Job Submission Pacing

- **There is no fixed upper limit on the number of submitted tasks**: It is not 5, nor 20. Whether to continue submitting is determined by explicit user instructions, current Slurm/QOS limits, startup/mount load, and current HPC status.
- Batch submissions must still be done one by one and staggered, with `sleep 30+` between batches. Observe `squeue` / startup logs to ensure no abnormal startup load.
- Low-risk compute jobs (helpers, plotting, smoke tests) can be submitted automatically based on user preference; large jobs must follow the submission rules in "Safe Operations."
- Exception: `scaled` experiments explicitly exempted in the existing `CLAUDE.md` follow original rules but do not have a fixed inflight limit.
- Smoke test → fix → resubmit can be done autonomously, but must still be staggered and monitored for startup load.
- **Reuse active allocations for evaluation:** Before submitting a new inference, evaluation, or LOB-Bench `sbatch`, inspect the user's RUNNING allocations. When one has compatible hardware and enough remaining walltime, prefer an attached job step such as `srun --jobid=<allocation> --overlap --exact ... --cpu-bind=none`; record the parent allocation, actual step ID, node, and timestamps. Do not create another queued allocation solely for the evaluation when a compatible active allocation is available.
- **Physical GPU gate before overlap:** Slurm `--overlap` does not make occupied GPU memory safe. Check active steps, compute PIDs, and per-GPU memory first. If the allocation is busy, an attached one-shot may wait for a predeclared zero-PID/near-baseline-memory gate, but it must never kill, retry, or overwrite the existing experiment without explicit authorization.
- **Queued-chain → attached-allocation handoff:** Hold the queued chain's root while establishing the replacement. Once the attached runner is confirmed live, cancel the entire superseded chain and verify every old job with both `squeue` and `sacct`. Never allow the queued and attached copies of the same experiment to run concurrently.

### 4.1 SquashFS SP500 Sweep Submission Constraints (2026-05-09)

SP500 scaling-law training must use compressed monthly SquashFS shards; falling back to raw `lob_preproc_sp500/<TICKER>/*.npy.zst` directory scanning is not allowed.

- `SQUASHFS_MULTI_MODE=1` must be set.
- Explicitly pass a month list like `SQUASHFS_MONTHS=2022-01,...,2025-12`; `SQUASHFS_MONTHS=all` is forbidden.
- Maintain `FORBID_RAW_NPYZST=1`.
- A single 8-node job will generate approximately `8 nodes * 48 months = 384` `squashfuse` mounts and read 48 shard-local `index.json` files per node.
- This is much safer than raw `.npy.zst` globs. System risk comes primarily from concurrent job and mount counts, not the 48 JSON files.
- For multi-experiment sweeps, no fixed inflight limit is set; however, jobs must be started one by one, staggered, and monitored for startup/mount load. Risk is estimated at `8-node jobs × 384 mounts/job`.

## 5. Login Node Taboos

- **Allowed**: Editing files, submitting jobs, `squeue` / `sacct` / `module list` / `sinfo`, light scripting.
- **Allowed (small CPU experiments, 2026-05-20)**: Lightweight CPU-only experiments such as linear regression / sklearn fits / pandas data prep / parquet conversion may run **directly on login nodes** when ALL of the following hold:
  - No GPU usage
  - Wall-clock under ~30 min, peak RAM under ~16 GB
  - Total bytes read+written under ~5 GB
  - Sequential file access only (no recursive `ls`/`find`, no parallel multi-worker pools)
  - Single-shot invocation, not a persistent loop or daemon
  - Rationale: a 5-min job behind a 30-min Priority queue wait wastes wall-clock; small CPU tasks have negligible Lustre metadata cost.
- **Prohibited**: Compiling, GPU inference/training, **heavy** data preprocessing (multi-process / >5 GB / >30 min), batch grep on large files, recursive ls / find.
- **Forbidden**: Persisting any agent on login nodes via `nohup` / `screen` / `tmux`.
- **Forbidden**: Auto-restarting any agent or daemon. If it is stopped, it was intentional; do not auto-recover.

## 6. Batch Script Review Checklist (Must check every line when writing or modifying batch scripts)

- [ ] Output dir isolated using `$SLURM_JOB_ID`
- [ ] Checkpoints written to `$TMPDIR`, `rsync` to `/projects` at end of job
- [ ] `latest_checkpoint.json` breadcrumb written (atomic by Rank 0)
- [ ] Auto-resume reads breadcrumb, does not `ls` checkpoint directory
- [ ] Tensorboard logs to `$TMPDIR`, synced at end of job
- [ ] `WANDB_DIR=$TMPDIR` and `WANDB_MODE=online` (if using W&B)
- [ ] No `find` / `ls -R` / `du -sh` / `tree` / large globs in the script
- [ ] Model weights staged to `$TMPDIR` before training
- [ ] Checkpoint interval is reasonable (not every step)

## 7. When in Doubt, Ask

If you are unsure whether a command will trigger metadata load, **do not try it**. Ask the user directly. Do not use "exploration" as an excuse to run `ls` / `find`.

> **Guiding Principle**: Act like a human researcher who understands that Lustre metadata operations are expensive. A human researcher would not `ls -lR /projects`, so neither will I.

## 8. AUP Accountability + Official 9 Anti-Patterns Table (2026-05-08 sync from HPC team / Aramis notes)

### 8.1 AUP Accountability (Most Important)

The Isambard-AI Acceptable Use Policy explicitly states: **All actions performed by an LLM / AI agent are attributed to the human account running it.** There is no "the agent did it" exemption.

- HPC audit logs record `kangli.s5e` / `aramis.s5e`, **not** "Claude."
- Every `sbatch` / `find` / `ls` / `rm` is treated as if "the human user typed it manually."
- Once BriCS attributes a violation, the penalty falls directly on the human account and may affect the entire team.
- **Batch scripts / training scripts generated by an LLM must be explicitly reviewed from a Lustre + HPC perspective before submission**—let the LLM self-check against the 9 anti-patterns below.

### 8.2 The 9 Anti-Patterns (Why + Fix)

The table below summarizes the 9 anti-patterns identified by the HPC team (Rich, Jakob) on 2026-05-08.

| # | Anti-pattern | Why it breaks Lustre | Fix |
|---|---|---|---|
| 1 | `ls -1td <ckpt_dir>` or `ls \| grep \| sort` to find latest checkpoint | Stats all inodes in the directory every call; metadata storm grows linearly with checkpoint count | Rank 0 atomically writes `latest_checkpoint.json` (path + step); auto-resume reads only this file |
| 2 | `find /` / `find /lus` / `ls -lR <broad path>` | Traverses the entire tree, floods the MDT, slows down everyone | Narrow the starting directory to the smallest possible subtree; use `lfs find` for non-recursive needs |
| 3 | Multiple jobs sharing a single output directory | Directory-level lock contention; concurrent jobs serialize on the same lock domain | Isolate each job using `${SLURM_JOB_ID}` or a unique experiment ID |
| 4 | Checkpoints / training logs written directly to Lustre | Continuous training triggers constant MDT locks | Write all intermediate output to `$TMPDIR` or NVMe scratch; `rsync` final checkpoint back to Lustre on success |
| 5 | Submitting a massive burst of jobs at once (e.g., 80) | Maximizes concurrent startup load; MDT cannot keep up | No fixed inflight limit for this repo; must submit one by one, stagger, and observe in batches |
| 6 | Checkpointing every step | Directory lock frequency follows the step count | Increase interval to at least 500 steps (model-dependent); target "recovering once wastes < 10% of job time" |
| 7 | Reading model weights from Lustre at startup | Multiple jobs starting simultaneously → startup lock storm | Pre-stage weights to `$TMPDIR` / NVMe before training actually starts |
| 8 | tensorboard / wandb / custom loggers writing to Lustre | Persistent metadata traffic throughout the entire run | Redirect local files to `$TMPDIR` (`WANDB_DIR=$TMPDIR`); keep tracked W&B runs online with `WANDB_MODE=online`; rsync non-W&B artifacts to Lustre at end-of-run |
| 9 | Running persistent AI agents + auto-restart wrappers on login/compute nodes | Explicitly forbidden by HPC team; auto-restart bypasses administrator intervention | Do not run resident agents on login/compute nodes; do not write "auto-restart if killed" scripts |

### 8.3 Origin + Original Quote from Rich (HPC Team) 2026-05-08

- **Triggering Event**: The team submitted a large volume of jobs simultaneously, with checkpoints written directly to Lustre and `ls`-based discovery for the latest checkpoint. This hit exactly the metadata storm pattern warned about by HPC.
- **Consequences**: On 2026-05-08, all group jobs were suspended by BriCS administrators, and AI agents were disabled service-wide on login nodes.
- **Symptoms**: `squeue` / `sbatch` began erroring. **Root Cause**: Lustre filesystem-level congestion.

> "Please note we are also disabling the use of AI agents on the login nodes. These are also prone to using the harmful patterns below, and could well be launching large numbers of compute node jobs that themselves contain those same patterns."
> — Rich, BriCS HPC team, 2026-05-08

---

## Project Overview

**LOBS5** — A token-level autoregressive generative model of limit order book (LOB) message flow using the S5 (Simplified Structured State Space) architecture in JAX/Flax. This experiment worktree (`exp/H1-scaling-law`) trains 5 model sizes (10M–120M params) on 8 tickers × 4 years to study neural scaling laws for LOB generation.

This is a **git worktree** of the main repo at `AlphaTrade/LOBS5/`.

## Architecture

```
LOBSTER .npy data
  │  (PyTorch DataLoader — CPU only, multi-ticker support)
  ▼
Tokenized messages (24 tok/msg, vocab=2112, base-100 encoding)
  + Order book volume images (500 depth levels)
  │
  ▼
┌─ PaddedLobPredModel (lob/lob_seq_model.py) ─────────────────┐
│  Message encoder: Embed → n_message_layers × SequenceLayer   │
│  Book encoder:    Dense → pre-layers → project → post-layers │
│  Padded fusion:   concat at message boundary                 │
│  Fused encoder:   n_layers × SequenceLayer(S5)               │
│  Decoder:         Dense → log_softmax (vocab_size)           │
└──────────────────────────────────────────────────────────────┘
  │
  ▼
Next-token cross-entropy loss → AdamW (optax) + cosine anneal
  → Orbax checkpoints + W&B logging
```

Each **SequenceLayer** (`s5/layers.py`): PreNorm → S5SSM (parallel scan via `jax.lax.associative_scan`) → half_glu1 activation → skip connection.

**S5SSM** (`s5/ssm.py`): HiPPO-LegS init, diagonal complex state matrix, ZOH discretization. Supports both parallel scan (training) and RNN mode (inference).

### Key source files

| File | Role |
|------|------|
| `run_train.py` | Entry point — CLI args, JAX distributed init, calls `lob.train.train()` |
| `lob/train.py` | Training loop — epochs, mini-epochs, validation, checkpointing, W&B |
| `lob/train_helpers.py` | JIT-compiled `train_step`/`eval_step`, optimizer, LR schedule, StepWatchdog |
| `lob/init_train.py` | Model/TrainState init, Orbax checkpoint load/save |
| `lob/sharding_utils.py` | JAX mesh creation (1D flat / 2D hierarchical), data & param sharding |
| `lob/dataloading.py` | Dataset factory, multi-ticker support, distributed sampler |
| `lob/lobster_dataloader.py` | `LOBSTER_Dataset` (PyTorch Dataset), file caching, masking |
| `lob/encoding.py` | `Message_Tokenizer` — LOB messages ↔ token sequences |
| `lob/lob_seq_model.py` | `PaddedLobPredModel` — Flax module (production model) |
| `s5/ssm.py` | S5 state space model core |
| `s5/layers.py` | `SequenceLayer` — single S5 layer with norm/activation/skip |
| `lob/inference.py` | Autoregressive generation with error correction |

## Common Commands

### Training (all runs via SLURM — never run on login node)

```bash
# Single model benchmark (30min, CURTAIL=300 steps)
D_MODEL=1024 N_LAYERS=12 BLOCKS=16 SSM_SIZE_BASE=1024 PER_GPU_BSZ=10 \
  CURTAIL_EPOCHS=300 sbatch --contiguous --nodes=64 --time=00:30:00 train_full_autoreg.batch

# Full training (15% of 1 epoch)
D_MODEL=1024 N_LAYERS=12 BLOCKS=16 SSM_SIZE_BASE=1024 PER_GPU_BSZ=10 \
  CURTAIL_EPOCHS=3179 NO_VALIDATION=1 sbatch --contiguous --nodes=64 --time=24:00:00 train_full_autoreg.batch

# Batch submit all 5 scaling law models
./submit_scaling_law.sh benchmark     # 5 sequential benchmarks
./submit_scaling_law.sh train         # 5 sequential 15%-epoch training runs
./submit_scaling_law.sh epoch         # 4 models × full epoch (120M skipped)
./submit_scaling_law.sh single 55M benchmark  # single model

# Dry run (preview commands)
DRY_RUN=1 ./submit_scaling_law.sh train

# Override node count
NODES=32 ./submit_scaling_law.sh benchmark
```

### Evaluation

```bash
# Post-training eval (single node)
RESTORE=/path/to/checkpoint sbatch eval_post_training.batch
```

### Tests (Docker-based, for local dev)

```bash
make test        # pytest ./tests/ inside Docker
make build       # build Docker image
```

## Scaling Law Model Configurations

| Label | d_model | n_layers | blocks | ssm_size | BSZ/GPU | Global BSZ (64N) |
|-------|---------|----------|--------|----------|---------|------------------|
| 10M   | 512     | 6        | 8      | 512      | 20      | 5,120            |
| 22M   | 768     | 6        | 12     | 768      | 14      | 3,584            |
| 55M   | 1024    | 12       | 16     | 1024     | 10      | 2,560            |
| 85M   | 1280    | 12       | 20     | 1280     | 6       | 1,536            |
| 120M  | 1536    | 12       | 24     | 1536     | 4       | 1,024            |

Default model (when no env overrides): **360M** (d_model=2048, n_layers=24, blocks=32, ssm_size=2048, BSZ=2/GPU).

## SLURM Job Structure

```
sbatch train_full_autoreg.batch
  └→ srun (1 task/node) → node_wrapper.sh (per-node)
       └→ conda activate, CUDA, NCCL/XLA flags → python run_train.py
```

- `train_full_autoreg.batch`: env vars, model config, data paths, srun launch, auto-resume logic
- `node_wrapper.sh`: conda env, CUDA 12.6, custom NCCL 2.29.3, AWS OFI NCCL 1.18.0, XLA flags
- `run_train.py`: argparse, `jax.distributed.initialize()`, calls `lob.train.train()`

Environment variables override model config (e.g., `D_MODEL`, `N_LAYERS`, `BLOCKS`, `SSM_SIZE_BASE`, `PER_GPU_BSZ`, `CURTAIL_EPOCHS`, `NO_VALIDATION`, `HIERARCHICAL`).

Multi-node (≥2 nodes) automatically enables hierarchical 2D mesh AllReduce.

## Key Design Decisions

- **PyTorch for data loading only** — DataLoader runs on CPU; all GPU computation is JAX/Flax
- **24-token encoding** (base-100, vocab=2112) — replaced 22-token encoding (base-10000, vocab=12012)
- **Hierarchical AllReduce** — 2D mesh splits into NVLink (intra-node) + Slingshot (inter-node); without it, 32N+ is ~2x slower
- **Custom NCCL 2.29.3** — source-built with GCC 12.3 for ARM CAS fix, loaded via LD_PRELOAD
- **Cosine annealing with warmup** — √κ LR scaling for AdamW when batch size changes
- **Mini-epochs** — split 1 data epoch into K sub-epochs for frequent eval checkpoints
- **Auto-resume** — on crash, batch script resubmits with `RESTORE_PATH` (up to 3 retries)

## Known Issue: Mid-Epoch Resume

**BUG**: When resuming from a mid-epoch checkpoint, `resume_from_step` is NOT auto-set from `state.step`. The dataloader restarts from batch 0, causing data duplication.

- **What works**: model weights, optimizer state, LR schedule (all keyed by `state.step`)
- **What breaks**: dataloader position — it replays already-seen batches
- **Root cause**: `lob/train.py:339` reads `args.resume_from_step` (always None) instead of computing `state.step % steps_per_epoch`
- **Fix needed**: auto-compute `resume_from_step = int(state.step) % steps_per_epoch` when restoring mid-epoch checkpoints. The `DistributedSampler(seed=42)` produces deterministic order per epoch, so skipping to the correct batch index is safe.
- **Curtail uses `batch_idx` not `state.step`**: so without fix, resume runs ALL curtail steps again (31k steps instead of remaining 8k), with wrong LR schedule (state.step keeps incrementing past curtail target).

## Logs and Checkpoints

- SLURM logs: `logs_lobs5/lobs5_{JOBID}.out`
- Per-node logs: `logs_lobs5/training_{JOBID}_node{N}.log`
- Checkpoints: `checkpoints/j{JOBID}_{WANDB_ID}_{JOBID}/{step}/`
- W&B project: `lobs5-scaling-law`
- Data split info: `logs_lobs5/data_split_j{JOBID}.json`

## HPC Environment

- **Platform**: ARM (Grace Hopper), 4× GPU/node, NV6 (6× NVLink bonded per GPU pair)
- **Conda**: `/projects/s5e/quant/miniforge3` (base env) — JAX 0.9.0.1, Python 3.12.11
- **Partition**: `workq`
- **Login node**: no GPU computation allowed; everything via `sbatch`

## Pre-Submit Checklist (MANDATORY)

**Every `sbatch` submission MUST be preceded by a `squeue` dedup check.** This is non-negotiable.

1. Run `squeue -u kangli.s5e -o "%.10i %.20j %.8T %.12M %.6D"` to list all running/pending jobs
2. For each running job, compare: **model config** (architecture, d_model, n_layers, params), **data** (tickers, date range), **encoding** (P1a/P1b/P1c), **seq_len**
3. If any existing job has the **same model + same data + same encoding**, do NOT submit — it's a duplicate
4. If the new job is a **resume** of a completed/timed-out job, confirm the old job is no longer running before submitting
5. Print the dedup check result to the user before proceeding with `sbatch`

**Why:** Job 3253421 (fresh start) ran in parallel with 3260152 (resume of same config) for 13 hours, wasting 16N × 13h = 208 node-hours because no dedup check was performed before submission.

**Anti-pattern to avoid:** Submitting a fresh-start job when a resume chain already exists for the same experiment. Always check if there's an active job or recent checkpoint for the same config before starting fresh.

## Session Search Protocol (MANDATORY)

**Use `/find-session-id` skill.** Core rule: **LARGEST JSONL = main session. Always.**

When searching for a historical conversation session by user-quoted text:

**Step 1 — Identify and exclude current session:**
```bash
# Current session ID is in the JSONL filename being written to right now
# ALWAYS determine it FIRST, then grep -v to exclude
```

**Step 2 — Use high-selectivity keys for first-pass filtering:**
- Commit hashes (e.g., `bae82954`) are ideal — nearly unique across all sessions
- SLURM job IDs, W&B run IDs, specific error messages also work well
- Avoid generic terms (KDA, GDN, smoke test) — too many false positives

**Step 3 — Multiple matches? Pick the LARGEST file (by size):**
- **LARGEST JSONL = main session** (user's interactive conversation, 1-5MB)
- Medium files (100KB-1MB) = continuation/child sessions (reference parent, partial context)
- Small files (<100KB) = subagent fragments (never return these)
- **DO NOT use "unique text" as discriminator** — subagent JSONLs contain full assistant output (including "Brewed for Xs"), leading to wrong (child) session

**Step 4 — Validate before returning:**
- Confirm result != current session ID
- Print session ID, JSONL path, modified date, size, and resume command

**Anti-patterns:**
- Grepping for common terms first, getting current session in results, returning it without noticing (caused errors 2026-03-26)
- Using rendered text ("Brewed for 44s") as discriminator instead of file size (caused error 2026-03-26, returned child session instead of parent)

## Post-Submit Job Monitoring (MANDATORY)

**Every `sbatch` must be followed by a multi-checkpoint monitor.** A single `sleep 300 && check` is NOT acceptable. Jobs can crash in seconds (j3492376: /dev/shm full, crashed at 21s) and a naive monitor won't detect it for minutes.

### Required Checkpoint Schedule

| Checkpoint | When | What to check |
|-----------|------|---------------|
| 1min | 60s | Log file exists? Any immediate crash (OOM, NCCL error, /dev/shm)? `sacct` exit code if job already gone. |
| 5min | 300s | Past XLA compile? tqdm output started? Step count > 0? |
| 15min | 900s | In stable regime? Speed (s/step)? Any NaN fatal? |
| 30min | 1800s | Final result. Total steps, stable speed, wandb URL. |

### Critical: Detect Crash vs Success

When a job is no longer in `squeue`, you MUST check `sacct -j $JOBID --format=ExitCode`:
- `0:0` = clean exit (success)
- `1:0` = crash (Python exception, OOM, NCCL failure)
- Other = signal kill, timeout, etc.

**Never assume "not in squeue" = success.** A job that crashes at second 21 also disappears from squeue.

### Critical: Check ALL Node Logs on Crash

When exit != 0:0, grep ALL `training_${JOBID}_node*.log` files for the FIRST error. The coordinator node (node0) often shows "Shutdown barrier timeout" which is a SYMPTOM. The root cause is usually on a different node.

```bash
for NLOG in $WT/training_${JOBID}_node*.log; do
    NODE=$(basename "$NLOG" | sed 's/.*node\([0-9]*\).*/\1/')
    ERR=$(grep -iE "RESOURCE_EXHAUSTED|No space left|OOM|NCCL WARN Error|SIGABRT|nan.*fatal|FATAL" "$NLOG" | grep -iv "CUDA_ERROR_NO_DEVICE" | head -1)
    [ -n "$ERR" ] && echo "node${NODE}: ${ERR:0:200}"
done
```

### Lesson (j3492376, 2026-03-30)

Job crashed at 21s (NCCL /dev/shm full on nid011191). Monitor script had `sleep 60` as first checkpoint, but only checked `squeue` and `grep tqdm` on node0. Node0 showed "Shutdown barrier timeout" (symptom), real error was on node4. Monitor ran for 47 minutes without surfacing the crash because it never checked `sacct` exit code or other node logs.
