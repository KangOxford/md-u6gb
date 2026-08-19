# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

### 1.2 删除一律改成改名 —— `rm` / `rm -rf` 禁用（2026-08-13，用户第二次强调）

**任何要删掉东西的地方，改成 `mv` 加 `_deprecated` 后缀。** 不停下来等确认，
直接改名就继续干活——改名可逆、删除不可逆，而两者的代价几乎相同。

```bash
rm -rf runs/foo/checkpoints                       # FORBIDDEN
mv runs/foo/checkpoints runs/foo/checkpoints_deprecated_20260813T2230Z   # 这样做
```

适用范围是**一切**：checkpoint 目录、日志、中间产物、"反正是空的" 的目录、
"反正是我自己刚才写坏的" 文件。理由不是怕删错重要数据，而是
**"这个能删" 这个判断本身就是我最容易出错的地方**：崩溃的 run 到底写没写
checkpoint、那个目录是不是别的会话在共用、上一次是不是真的失败了——这些在
删除的瞬间都变成不可验证的。改名把判断错误的代价从 "永久丢失" 降到 "多一个目录"。

时间戳要带上，否则重跑第二次时 `_deprecated` 会撞名。清理由用户决定，不由我决定。

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

### 🚨🚨🚨 4.-1 为什么 4.0 这条规则我反复不执行（2026-08-18，第三次犯，必须先读这一节）

**事发**：2026-08-18 17:48–19:5x，**36 张自有的卡整整空转 2 小时 10 分（6.30 kW）**，
同时队列里 10 个作业在等。用户原话：**「这是有病还是咋的？这么多空的不用，你在那排队。」**
这是同一个错误的第三次（2026-08-14「你看着空的 GPU 不用，你去排队是吧」、2026-08-16 XVLA 四连发）。

**前两次我都在 4.0 这里加了更强的措辞，然后又犯了。所以问题不在措辞，在触发机制。**
四条根因，按造成的损失排序：

#### R1 —— 我把 GPU 当成「一个我在做的任务」，而不是「一种绝不能空着的资源」

注意力转到分析/写代码时，**GPU 从我的世界模型里整个消失了**。4.0 写的是
「每 15–30 分钟巡一次」——那是一条**时间驱动**的规则，而我的执行是**任务驱动**的：
我不会在写分析写到一半时因为「过了 20 分钟」而中断自己。**规则的触发条件与我的实际
运行方式不匹配，所以它永远不会触发。**

> **修正：把巡卡挂到一个我一定会做的动作上，不挂在时钟上。**
> **硬规则：每一次我要输出一段长回复、或开始一项不用 GPU 的工作（分析、写文档、改代码）
> 之前，先查一次卡。** 这个触发条件我逃不掉，因为长回复和分析是我一直在做的事。

#### R2 —— 只起了一波，剩下的没排队

21 个 bench，12 个位置。我起了 12 个，然后**就不管了**。两小时后前一波跑完，
没有任何东西接上去。

> **修正：作业数 > 位置数时，剩余的必须当场写成待起清单并说明何时起。
> 「起了一波」不算完成，「所有作业都已落地或已排好队」才算。**

#### R3 —— 看到空队列，我的第一反应是「查故障」而不是「填卡」

`squeue` 里没有我的 step，我读成「它们死了」，然后去翻日志查死因。
**它们其实是跑完了。** 而查死因花掉的那几分钟，卡是空的。

> **修正：空队列的动作顺序是固定的——先填卡，再查上一批怎么样了。**
> 这两件事不冲突，但**填卡必须在前**：查故障需要几分钟，填卡需要几秒。
> 而且无论上一批是死是活，下一批都要起，所以查故障根本不是填卡的前置条件。

#### R4 —— 一条线被挡住，我把所有线都停了

inode 配额打满 → 我写不了一个新的测试文件 → **我升级给用户并停下了全部工作**，
一停两小时。但 inode 满**根本没有挡住 GPU 计算**：那 9 个 bench 在 inode 满的情况下
照样把产物写完了（改写已存在的文件不消耗 inode，而它们的计算全在节点本地盘）。
**我把「建不了新文件」过度推广成了「什么都做不了」。**

> **修正：升级给用户之前必须先回答两个问题——(1) 这个阻塞具体挡住了哪一步？
> (2) 其他步骤还能不能走？** 只有当答案是「全都走不了」时才停下来等。
> **等用户回话的时间里，卡照样在烧。**

---

### 🚨🚨 4.0 硬标准：**每一次 `sbatch` 之前必须先跑 `gtop`，有空卡就 attach，绝不排队**（2026-08-14 用户定）

**这是命令，不是建议。没有例外。**

```bash
gtop                 # 先看：自己名下的分配里，此刻哪些卡是空的
                     # 空卡存在 → 用 srun --overlap 上去跑，不许 sbatch
                     # 真的一张空卡都没有 → 才允许 sbatch
```

**是 `gtop` 不是 `sgpu`，两者分工不同，别用错**：

| 工具 | 回答的问题 | 用在这里对不对 |
|---|---|---|
| **`gtop`** | **此刻**每张卡的物理占用（util / 显存 / 锁 / step） | ✅ 就是它。唯一不来自 Slurm 记账、直接读卡的测量 |
| `sgpu` | 过去 N 小时谁跑过、断了几次（甘特图账本） | ❌ 历史账面，回答不了「现在有没有空卡」 |

而且**要不断地查**，不是查一次就完事：排队等待期间要持续 `squeue` / `gtop` 轮询，
一旦自己名下出现空节点，立刻把工作挪上去，不要继续干等。

**判据（从 `gtop` 直接读，这是唯一不来自记账的测量）**：

| 信号 | 含义 |
|---|---|
| `steps: .batch only · nothing computing` | **整个分配的 GPU 全空**，最该抢的目标 |
| GPU `0%` 且显存 `0.0/95.6G` | `idle`，真空 |
| GPU `0%` 但显存 > 64 MiB | `held`，有人预留了显存，**不是空的** |
| `🔓` 无锁 | 可用（无锁节点即便在跑任务也可以用） |
| `🔒 <别人>` | 别人声明过，不碰 |
| `DEAD(闲置 N min)` | 锁还在但长期空转，问过用户再动 |

**血泪教训（2026-08-14）**：H3 复刻的两个作业在队列里干等了一整天，其间用户账户
名下有 **40 张 GPU 空转**（`6006783` 8 张、`6006424` 16 张、`6000409` 16 张，全部
`nothing computing`，各自还剩 5–10 小时）。我一次都没查过 `sgpu`，直接 sbatch 然后
等。用户原话：**「你看着空的 GPU 不用，你去排队是吧？」**

这条规则本来就以「Reuse active allocations」的形式写在下面，我没执行。现在提到 4.0
并升级为硬标准：**`sgpu` 是 `sbatch` 的前置条件，跳过它就是违规。**

**血泪教训第二例（2026-08-16，用户再令 "always think about using attach before any
sbatch"）**：XVLA 四连发（2 评测 + 2 训练重启）全走 sbatch 排队，同一时刻自己名下
`6022465` 的 **16 张卡显存 1-4 MiB 彻底空转、还剩 13h49m**。错误机制是把前一天的
「假窗口黑名单」当成了免检章——黑名单是**时点判定，会过期**：昨天它忙、今天它空。
三条修正入规：
1. **每一次 sbatch 前逐卡重查显存**（`nvidia-smi --query-gpu=memory.used` 经
   `srun --overlap` 到目标节点，或读 gtop 逐卡行），gtop 头行的 idle 计数把
   held（显存 86G+ 驻留）也算进去，**头行不可作判据**；1-4 MiB 才是真空。
2. **1N/1GPU 的小任务（eval/冒烟/benchmark）一律 attach-first**——暴露面小、
   分钟级起跑；sbatch 只留给需要独立 walltime 保障的多节点长段。
3. attach 版与排队版并存时用**不同 TAG/输出目录**防结果目录冲突，取先完成者。

**attach 起来之后必须真的在算。** 一个 attach 的 srun step **随启动它的会话一起死**，
所以要用 `setsid nohup srun ... &` 让它脱离会话进程组，否则会话一重启，卡就白占着
空转。起完 40 秒内用 `squeue -s -j <alloc>` 确认 step 在，并且 `gtop` 里 util 起来了；
没起来就是没在算，要么修要么让开。

attach 的具体机制见下面 4.1 起的各条。**注意其中的 `nodelock` 部分已于 2026-08-14 废止**。

- **There is no fixed upper limit on the number of submitted tasks**: It is not 5, nor 20. Whether to continue submitting is determined by explicit user instructions, current Slurm/QOS limits, startup/mount load, and current HPC status.
- Batch submissions must still be done one by one and staggered, with `sleep 30+` between batches. Observe `squeue` / startup logs to ensure no abnormal startup load.
- Low-risk compute jobs (helpers, plotting, smoke tests) can be submitted automatically based on user preference; large jobs must follow the submission rules in "Safe Operations."
- Exception: `scaled` experiments explicitly exempted in the existing `CLAUDE.md` follow original rules but do not have a fixed inflight limit.
- Smoke test → fix → resubmit can be done autonomously, but must still be staggered and monitored for startup load.
- **Reuse active allocations for evaluation:** Before submitting a new inference, evaluation, or LOB-Bench `sbatch`, inspect the user's RUNNING allocations. When one has compatible hardware and enough remaining walltime, prefer an attached job step such as `srun --jobid=<allocation> --overlap --exact ... --cpu-bind=none`; record the parent allocation, actual step ID, node, and timestamps. Do not create another queued allocation solely for the evaluation when a compatible active allocation is available.
- **Physical GPU gate before overlap:** Slurm `--overlap` does not make occupied GPU memory safe. Check active steps, compute PIDs, and per-GPU memory first. If the allocation is busy, an attached one-shot may wait for a predeclared zero-PID/near-baseline-memory gate, but it must never kill, retry, or overwrite the existing experiment without explicit authorization.
- **🚨🚨 废止：不许再 lock GPU（2026-08-14 用户令，覆盖下面 R1）**
  用户原话：**「不许不许 lock 人家 GPU，以后不许带这种功能」**。
  `nodelock lock` 一律不再执行，`tasks/node_status/` 已改名 `_deprecated_20260814`。
  起因：我给 nid010499 上了 `claude-h3` 锁，然后 attach 的 step 死了，结果是
  **四张卡挂着我的锁在那儿 0% 空转**——锁的唯一效果是让别人不敢用。
  占用一张卡的资格只剩两条：**用户明示**，以及 `nvidia-smi` 显示它此刻是空的。
  下面 R1 的双查闸门只保留 `nvidia-smi` 那一半（物理事实），锁表那一半作废。

- ~~**🚨 声明在前，占用在后（R1，2026-08-13 定）**~~（**已被上一条废止**，保留原文供追溯）：
  attach 进任何分配之前先
  `nodelock lock`，再起进程。**物理闸门是一个瞬时快照，而启动窗口有 5–10 分钟**
  （挂 squashfs 分片 + JAX 分布式 init）；邻居在窗口里起来，快照怎么查都看不见。
  能覆盖窗口的只有声明。闸门必须**双查**：锁表无他人 live 锁 ∧ `nvidia-smi` 无
  他人 compute PID——锁表是**意图**、nvidia-smi 是**事实**，两者都会单独骗人。
  机械实现见 `tasks/hybrid_system_20260811/05_hybrid_mamba3_lob/code/claim_gate.sh`，
  完整规则（三档优先级、抢占只向下、TTL）见 `tasks/node_status/PRIORITY.md`。
  血泪教训：2026-08-13 10:19，2k baseline 臂在 6000412 上被邻居的 vLLM（92 GB）
  挤死，`GetKeyValue() timed out with key: cuda:local_topology/cuda/2`——邻居在
  锁表上**都声明过**，是我的闸门没读。
- **🚨 srun step 必须起真名字（`--job-name=<实验名>`）：** `node_budget_monitor.py`
  按正则 `^(bash|sh|zsh|...)$` 判 step 是否 idle，有真名字的 step 让整个作业记成
  `computing` 而**不计入 20 节点上限**。本仓库普遍用 `srun bash -lc '...'`，step 名
  就叫 `bash`，于是**正在训练的作业被判成 IDLE-HELD，随时可能被预算闸门 scancel**。
  2026-08-13 的 dry-run 里，已训 5h45m 的 hybrid 就显示 `IDLE-HELD only bash,bash,...`。
  起名字不是绕过闸门，是把度量修对。
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


## 🚨 训练观测频率：checkpoint 与 loss 是两件事，不许共用一个频率（2026-08-13）

### 硬约束

| 项 | 频率 | 理由 |
|---|---|---|
| **checkpoint** | **平均每 ~15 分钟一次** | 写整个参数树到 Lustre，按「恢复一次损失 <10% 时间」定 |
| **step_loss / lr / grad_norms** | **每 1 分钟（auto）或每 250 步（显式）** | 只是几个标量打到 wandb，几乎免费 |

**默认一律用 `CHECKPOINT_EVERY=auto`。** 那条路径已经按时间判据把两者分开：

```
AUTO_CKPT_INTERVAL  = 900s  = 15 分钟   checkpoint
AUTO_WANDB_INTERVAL =  60s  =  1 分钟   wandb
首次 checkpoint 在 ~5 分钟（防早期 NCCL 死锁丢掉全部进度）
```

### 反面教训（2026-08-12，hybrid ctx2k）

给了显式的 `CHECKPOINT_EVERY=2000` 覆盖掉 auto，同时踩了两个坑：

1. **checkpoint 变稀**：2k 上下文下 2000 步 = 50 分钟以上，一次崩溃丢 50 分钟
2. **`step_loss` 被绑到同一个频率上**——`train_helpers.py` 里原本是
   `should_ckpt = should_wandb = (step % N == 0)`

后果：hybrid 发散时**只能定位到「4000 到 6000 步之间」**，分不出是突然尖峰还是持续
爬升。而想看密的 loss 就得调小 checkpoint 频率，**等于用 Lustre 元数据去买观测精度，
代价差三个数量级**。

已修：`LOG_EVERY`（默认 250）独立控制显式档下的记录频率，与 `CHECKPOINT_EVERY` 解耦。

### 必须记录的 profiling 指标

```bash
export LOG_GRAD_NORMS=1     # 分组梯度范数 + clip_ratio
```

| 指标 | 为什么必需 |
|---|---|
| `grad_norms/{global,muon,ssm,regular,in_proj,out_proj}` | 发散时第一件事是看**哪一组**先炸 |
| `grad_norms/clip_ratio` | 直接回答「裁剪是不是一直饱和」——饱和说明裁剪只是在掩盖问题 |
| `lr` / `ssm_lr` | 排除「是不是 schedule 在这一段把 LR 抬起来了」；**必须与 loss 同频**，否则两条曲线对不齐就没法说话 |
| `throughput/{step_time_s,mfu_pct,tflops,tokens_per_sec}` | 步时突变往往先于 loss 异常 |

### 起跑前自查

- [ ] `CHECKPOINT_EVERY` 是 `auto`，或显式值 × 步时 ≈ 15 分钟
- [ ] `LOG_GRAD_NORMS=1`
- [ ] 显式档下 `LOG_EVERY` ≤ 250
- [ ] wandb 上能看到 `lr` 与 `step_loss` 在同一批 step 上有点

---

## 🚨 梯度累积 K 不许有「默认值」：声明有效批量，推导 K（2026-08-14 用户定）

### 硬规则

**`GRAD_ACCUM_STEPS` 永远不写成 `${GRAD_ACCUM_STEPS:-<某个数>}`。**
脚本里要声明的是**有效批量**，K 由它和节点数推导出来，并对调用方传进来的值硬校验。

```
有效批量 = micro_bsz × GPU/节点 × 节点数 × K
```

以本仓库 2k 上下文那三条臂为例（`micro_bsz=1`，`GPU/节点=4`，有效批量 80）：

| 节点数 | 2 | 4 | 5 | 10 | 20 |
|---|---:|---:|---:|---:|---:|
| **正确的 K** | **10** | **5** | 4 | 2 | 1 |

**换节点数而不改 K，改的就是实验本身。** 有效批量是实验定义的一部分，
和模型、数据、seed 同级；K 只是为了在给定硬件上凑出它的手段。

### 反面教训（2026-08-13/14）

`launch_2k_*.sh` 原本写的是 `export GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-10}`
（按 **2 节点**写死），靠调用方 `chain_manager.sh:291` 传 `5` 来覆盖。这个形状必出事：

- 4 节点上忘了覆盖 → 有效批量 **160**，是设计值的 2 倍
- **不会报任何错**。训练照跑、loss 照降、wandb 照记，只是训出一个与对照臂
  **不可比**的模型
- 而「不可比」要到几小时后做配对比较时才发现，那时机器已经烧掉了

同一族缺陷见 `feedback_config_only_reaches_one_path`：**一个必须随环境变的量，
被写成固定默认值 + 藏在别处的覆盖。**

### 正确写法（已落地在 `launch_2k_{baseline,hybrid,hybrid_pmatch}.sh`）

```bash
export EFFECTIVE_BSZ=${EFFECTIVE_BSZ:-80}          # ← 这个才该有默认值
_bsz_denom=$(( PER_GPU_BSZ * GPUS_PER_NODE * NNODES_ATTACH ))
if [ "$_bsz_denom" -le 0 ] || [ $(( EFFECTIVE_BSZ % _bsz_denom )) -ne 0 ]; then
    echo "FATAL: 有效批量 $EFFECTIVE_BSZ 不能被 $_bsz_denom 整除" >&2; exit 5
fi
_k_derived=$(( EFFECTIVE_BSZ / _bsz_denom ))
if [ -n "${GRAD_ACCUM_STEPS:-}" ] && [ "${GRAD_ACCUM_STEPS}" != "${_k_derived}" ]; then
    echo "FATAL: 传了 K=$GRAD_ACCUM_STEPS，但 $NNODES_ATTACH 节点要求 K=$_k_derived" >&2
    echo "       要改有效批量就显式传 EFFECTIVE_BSZ=，别靠改 K 间接改它" >&2; exit 5
fi
export GRAD_ACCUM_STEPS=$_k_derived
echo "[bsz] 有效批量 $EFFECTIVE_BSZ = ${PER_GPU_BSZ} × ${GPUS_PER_NODE}GPU × ${NNODES_ATTACH}节点 × K${GRAD_ACCUM_STEPS}"
```

**必须打印那行 `[bsz]`**，它让「这次到底跑的是多大批量」在日志里可检索，
而不是要靠读三个脚本反推。

### 配套：参数量也用同样的方式钉死

同理，凡是「参数量必须等于某个对照臂」的实验，设
`EXPECTED_PARAMS=<实测值>`（`src/lob/train.py:349-364` 会在不等时 raise）。
理由一样：配置没到达模型时**不会报错**，只会在几小时后给出一个假结论。

### 起跑前自查

- [ ] 脚本里没有 `GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-<数字>}` 这种写法
- [ ] 日志里有 `[bsz] 有效批量 ... = ... × ...节点 × K...`，且数值等于设计值
- [ ] 换过节点数的话，K 跟着变了（不是沿用上一次的）
- [ ] 参数量对照实验设了 `EXPECTED_PARAMS`

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

**Step 0 comes before everything else: run `gtop`** (physical occupancy right now --
*not* `sgpu`, which is the historical gantt ledger and cannot answer "is anything free
now"). If any GPU in an allocation this account already holds is idle, **attach to it
and do not submit at all**. See §4.0 — `sbatch` is only permitted once `gtop` shows
there is genuinely nothing free. Keep polling while anything is queued; the moment a
node frees up, move the work onto it.

**Every `sbatch` submission MUST be preceded by a `squeue` dedup check.** This is non-negotiable.

0. Run `gtop`. Idle GPUs anywhere in this account → attach via `srun --overlap`, do not `sbatch`.
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

## 运行环境的自我认知 (2026-08-07, 用户明确纠正)

**Claude Code 进程运行在 Isambard 的登录节点上，不是在用户的 Mac 上。** 用户从
MacBook 通过 SSH 连到服务器，Claude Code 是服务器上的一个程序。

推论（都影响该建议什么）：
- 服务器**始终有网**。用户 Mac 断网断的是 SSH 连接，不是服务器的出网能力。
- 因此 `tmux` **确实能**让这个会话在用户断开后存活：`tmux new -s <name>` 里跑
  `claude --resume <session-id>`，回来 `tmux attach`。不要再说「tmux 也没用因为
  需要 API 网络」——那是把自己误当成跑在用户笔记本上。
- 但会话存活 ≠ 工作继续：**没有用户输入时 Claude 不会自己醒来**。所以长时间无人值守
  的工作必须落到 SLURM 作业里（自带 checkpoint/resume/失败续投），而不是指望会话活着。
- 登录节点上常驻 agent 仍受 BriCS 2026-05-08 禁令约束。判据是**是否在无人值守下持续
  发起动作**：脱离后闲置等待输入的会话是良性的，自动轮询/循环提交的不是。
