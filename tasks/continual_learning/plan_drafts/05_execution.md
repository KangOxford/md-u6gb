# 05 — Execution, sizing, and the dependency graph

Facet owner: execution. Sister drafts cover the science; this one covers what runs,
where, in what order, at what size, and how an artefact survives an expiring allocation.

Every number below is labelled **measured** (with the file it came from) or
**provisional** (with the one measurement that would replace it). Nothing is presented
as measured that was not read off a file today.

## 中文速览

- **每步 token 数不需要 GPU 就能算出来**：`train_full_autoreg.batch:375` 把 `MSG_SEQ_LEN` 定为 500 条消息，
  `:385` 把每条消息定为 26 个 token，所以上下文 = 13,000 token。探针配置（1 卡、`PER_GPU_BSZ=4`、
  `GRAD_ACCUM_STEPS` 默认 1）每步 52,000 token。INVENTORY 的第 3 个待办项就此关掉一半——
  剩下的一半（0.565 s/step 是在什么配置上测的）也是零 GPU 成本，去 wandb 读 run 配置即可。
- **今天新发现的最大执行障碍**：`run/base_model/runtime/train.py:31-32` 无条件地把
  `CUDA_VISIBLE_DEVICES` 覆写成 `0,1,...,GPUS_PER_NODE-1`。也就是说 CLAUDE.md §4.0.1 的标准做法
  （要 4 张卡、进程内点名那张空卡）**到不了训练代码**。而 `node_wrapper.sh:38` 还会把你设的值打印出来——
  设了、打印了、没生效，正是本项目已记录在案的失效方式。**此刻 gtop 上 16 张真空卡里没有一张是 0 号卡**，
  所以不改这一行，一张都用不上（推理侧不受影响，`inference.py:425` 只打印不覆写）。
- **rollout 生成的速度是实测的**：`hp_v5me3_AMD_s97701/member_0/inference.log` 记录 500 个 context、
  batch 48、11 个批次，生成循环 246.4 秒（首批 77.3 秒是 XLA 编译，稳态每批约 16.8 秒）。
  一个成员约 4.1 分钟/卡。从 k=10 加到 k=20，8 只票共 80 个成员 = 5.5 GPU-小时。
- **inode 才是瓶颈，不是磁盘也不是卡**：项目配额 50,376,941 / 51,200,000（**只剩 823,059**，
  空间 127.1T/200T 还很宽）。一个未打包的成员 = 3,009 个文件（实测 `data_gen` 1500 + `data_real` 1500）。
  80 个成员 = 240,960 个 inode = 剩余额度的 29%。而且 **real 那一半在各个种子之间逐字节相同**
  （实测同一文件在 s97701 与 s97702 的 md5 都是 `85d5adb955b52e8135d1fea1a56c73d4`），
  写一次而不是写十次就能省下 108,000 个 inode。改名不释放 inode，所以这笔账只能在写之前算。
- **顺序由依赖图定，不由空卡定**。等卡的时候把 C1–C10 十项纯 CPU 的活做完；
  它们里有三项是当前 GPU 作业的前置条件。
- 两个现成脚本：`submit_adaptation_pair.sh` 直接可用（走 sbatch，跳过空卡，只在没有空卡时才该用）；
  `attach_adaptation.sh` **不能照原样用**，四处要改（gres、CUDA_VISIBLE_DEVICES、`setsid nohup`、
  `MAX_JOB_HOURS` 与分配剩余时间脱钩），详见 §8。

---

## 0. Two facts that shape everything below

**Fact A — the free cards are never card 0.** A `gtop --once` snapshot taken while writing
this draft (2026-09-04, login44) showed 16 genuinely idle cards, judged by per-card memory
`mem 0.0/95.6G` and not by the header count:

```
6317365 nid010234 [3]   6317365 nid010288 [3]   6317365 nid010308 [3]   6317365 nid010488 [3]
6282179 nid010901 [1,2,3]
6269978 nid010436 [1,2,3]   6269978 nid010437 [1,2,3]   6269978 nid010439 [1,2,3]
```

Sixteen idle cards, zero of them logical device 0. At the same moment two jobs were
PENDING (`6317366 u6gb-4-node-chain`, Resources; `6243552 l2c-s44-driver`, JobHeldUser).

**Fact B — the training entry point overwrites the card selection.**
`/lus/lfs1aip2/projects/public/u6gb/sigma-0/run/base_model/runtime/train.py:31-32`:

```python
_n_gpus = int(os.environ.get('GPUS_PER_NODE', '4'))
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(_n_gpus))
```

The assignment is unconditional, so a `CUDA_VISIBLE_DEVICES=3` set by the launcher is
discarded and the process runs on physical device 0. `node_wrapper.sh:38` prints
`[Wrapper] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES` **before** Python starts, so the
log shows the value you set while the process uses a different card. That is the project's
own documented failure mode — a knob that is set, printed, recorded, and never reaches the
code — and it is live in the one file every adaptation run goes through.

Together, A and B say: **no adaptation training job can attach to any currently free card
until `runtime/train.py` honours an explicit device list.** Rollout generation is not
affected — `run/base_model/runtime/inference.py:425` only prints
`os.environ.get('CUDA_VISIBLE_DEVICES', 'all')` and never assigns it, so generation can use
card 1, 2 or 3 today. This is why the ordering in §4 puts the code change before the first
training job and starts the GPU work with generation.

---

## 1. The dependency graph

### 1.1 Every job the plan implies

`kind` is **CPU** (login node or a CPU-only step), **GPU**, or **CODE** (an edit plus its
tests, no compute). "Blocks" names the downstream job that cannot start without it.

| id | name | kind | consumes | produces | blocks |
|---|---|---|---|---|---|
| C1 | tokens-per-step derivation | CPU | `train_full_autoreg.batch` lines 368/375/385-386, the chosen config | the identity in §2.2 | G1, G4, G5 sizing |
| C2 | recover the config behind 0.565 s/step | CPU + network | wandb run `wqgghoyj` (and `c86ghhsn`) config | (nodes, GPUs/node, `PER_GPU_BSZ`, `GRAD_ACCUM_STEPS`) for that timing | makes 0.565 s/step convertible; INVENTORY item 3 |
| C3 | write a resume breadcrumb for the selftrain chain | CPU | the 17 step directories under `/lus/lfs1aip2/projects/public/u6gb/sigma-0/checkpoints_selftrain/j5705912_b30675li_5705912/` | `latest_checkpoint.json` + a static `steps.json` | removes the directory scan from every later resume |
| C4 | freeze and hash the context set | CPU | `sample_indices_rank0.json` per member dir | one shared indices file + its sha256 | G2 (comparability of new members with old) |
| C5 | extend the reliability curve on existing rollouts | CPU | `results/failure_pool_reliability.json` inputs | split-half rho at k up to 5 (done), the k-needed fit | tells G2 how many members to generate |
| C6 | fidelity / autopsy replay scoring | CPU | a generation run dir (`data_gen`, `data_real`, `data_cond`) | exact-book-match fraction, first-divergence step, signature tally | the second scoring axis of issue #73 |
| C7 | offline weight probes over all 17 checkpoints | CPU | the selftrain chain | L2, mean abs, per-matrix spectral norms per step | the weight half of the plasticity readout |
| C8 | inode accounting and the real-arm dedupe decision | CPU | `lfs quota`, one `lfs find` per member dir | the budget in §5 and a write layout | G2 (must be settled before the first new member is written) |
| C9 | patch `runtime/train.py` to honour an explicit device list | CODE | Fact B | `GPU_IDS` honoured; a unit test that fails on the old behaviour | G1, G4, G5 on any node whose card 0 is busy |
| C10 | patch `code/attach_adaptation.sh` (four changes, §8) | CODE | C9 | an attach launcher that survives the session | G1, G4, G5 |
| G1 | short timing run: 60 optimizer steps at the probe config | GPU, 1 card | C1, C9, C10 | measured s/step and tokens/s for this exact config | converts every provisional wall clock in §2.4 |
| G2 | rollout regeneration to the target k | GPU, N cards | C4, C8, and C9/C10 only if card 0 is busy | new members under a new config tag | the corrected failure pool |
| G3 | activation and gradient probes at 3-5 checkpoint ages | GPU (or CPU, slow) | C7, C9 | dormant fraction, effective rank, optimization readiness, top Hessian eigenvalue | the co-moving diagnostic the plasticity claim needs |
| G4 | early-vs-late fixed-budget adaptation, 2 ages x 5 seeds | GPU, 1 card each | G1, C3, C9, C10 | validation-NLL trace per run, AUC per run | the PLAN Step 2 result |
| G5 | failure-pool continued training, mix grid x seeds | GPU, 1 card each | G2, G1 | one checkpoint per cell | the issue #73 result |
| G6 | post-training evaluation of every G4/G5 checkpoint | GPU inference, CPU scoring | G4, G5 | `bench` and `return-bench` numbers, C6 replay numbers | the two-coordinate report |

### 1.2 The graph

```
        CPU-only, runnable right now, no allocation needed
        ---------------------------------------------------
        C1 tokens/step ---------------------------.
        C2 wandb config for 0.565 s/step ---------|
        C3 breadcrumb ----------------------------|--> sizing + resume
        C5 reliability curve --> (how many k) ----|
        C7 weight probes ------------------------.|
        C8 inode budget -------------------.     ||
        C4 context set + sha256 ----------.|     ||
                                          ||     ||
        CODE                              ||     ||
        C9 train.py honours GPU_IDS --.   ||     ||
        C10 attach launcher ----------|   ||     ||
                                      |   ||     ||
        GPU                           v   vv     vv
                              +-- G1 timing (10 min, 1 card)
                              |      |
                              |      v
        C4,C8 --> G2 rollouts +   G4 early-vs-late adaptation (10 runs)
                    |                 |
                    v                 |
        C5 rescored pool              |
                    |                 |
                    v                 v
                  G5 failure-pool training (12 runs)
                    |                 |
                    +--------+--------+
                             v
                          G6 evaluation  ---> C6 replay scoring (CPU)
                             |
                             v
                       two-coordinate report
```

`G3` hangs off `C7` and `C9` and is independent of `G1`; it can fill any single free card
at any point.

### 1.3 The CPU-only claim, job by job

Today's three prerequisite measurements all turned out to be CPU-only after being assumed
to need GPUs, so this column is argued rather than asserted.

| job | why it is CPU-only | evidence |
|---|---|---|
| C1 | it is arithmetic over two constants in a shell script | `train_full_autoreg.batch:375` `MSG_SEQ_LEN=${MSG_SEQ_LEN:-500}`, `:385` `TOKENS_PER_MSG=26`, `:386` `context=$((MSG_SEQ_LEN * TOKENS_PER_MSG))`, `:368` `GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}` |
| C2 | a wandb run config is an HTTP GET; the login node has network | no local wandb directory exists under `/lus/lfs1aip2/projects/public/u6gb/sigma-0/` |
| C3 | writing one small JSON next to 17 existing directories | the 17 steps are already enumerated in §2.5 |
| C4 | reading and hashing a 11 KB JSON | `sample_indices_rank0.json` is 11,506 bytes |
| C5 | pure numpy over two `.npz` per member | `code/failure_pool_reliability.py` imports only `numpy`; it already produced `results/failure_pool_reliability.json` on CPU |
| C6 | replays a recorded message stream through the JaxOB matching engine; **no checkpoint is loaded and no network forward pass happens**. `jax` is imported lazily inside `replay_stream` only, and `describe_stream` is pure numpy | `src/post_training/heuristic_learning/fidelity.py` module imports are `argparse, json, os, re, dataclasses, pathlib, typing, numpy` plus `episode_builder`; `jax.numpy` and `matching_engine.jaxob` appear only at `fidelity.py:293-295`. `autopsy.py` and `mm_sim.py` are the same shape |
| C7 | Orbax restore with `restore_type=np.ndarray` under `JAX_PLATFORMS=cpu` | `code/probe_weights_offline.py` docstring gives that exact invocation; it produced `results/a1_step275.json`, `a1_step33575.json`, `a1_step69378.json` |
| C8 | `lfs quota` plus one `lfs find` on a single member directory | both run in seconds; numbers in §5 |
| C9, C10 | source edits and their tests | — |
| G3 | **claimed GPU, and this claim is soft.** The probes need activations and per-micro-batch gradients, i.e. forward and backward passes. A 78.5M-parameter model with a 13,000-token context on CPU is feasible in principle but has never been timed here | `code/plasticity_probes.py` is pure numpy and imports no framework; only its *inputs* need a forward pass. Whether one CPU forward pass at this context length finishes inside the login-node limit (30 min, 16 GB) is **not measured** — see Open questions |

The three that were wrongly assumed to need GPUs today (reliability, conditionality,
decomposition) are all instances of the same pattern: the *scoring* of a rollout is numpy
over two small arrays, while only the *generation* of the rollout is a GPU job. C6 is the
same pattern one step further out — replay through the matching engine is deterministic
simulation, not inference — and is the largest piece of work in this plan that has not yet
been recognised as free.

---

## 2. Sizing

### 2.1 What is actually measured

| quantity | value | source |
|---|---|---|
| context length | 500 messages x 26 tokens = 13,000 tokens | `train_full_autoreg.batch:375,385-386` |
| 75m preset | `D_MODEL=1024`, `N_LAYERS=6`, `BLOCKS=16`, `SSM_SIZE_BASE=1024`, `DEFAULT_PER_GPU_BSZ=10` | `train_full_autoreg.batch:238-244` |
| parameter count | 78,539,423 | `results/a1_step69378.json`, and printed as `[*] Trainable Parameters: 78539423` in the generation log |
| `GRAD_ACCUM_STEPS` default | 1 | `train_full_autoreg.batch:368` |
| rollout generation, 500 contexts, batch 48, 1 card | 246.44 s for the whole loop | `/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z/data/hp_v5me3_AMD_s97701/member_0/inference.log`, final line `Generation time for 500 sequences across 48 batch size: 246.44063305854797` |
| first batch vs steady state | 77.31 s then 16.8 s/batch | same log, tqdm timestamps `1/11 [01:17...]` through `11/11 [04:05...]` |
| model-only time per batch of 48 | 11.0 s | same log, `Generation time for batch of size 48: 11.03 seconds` (stable across all 11 batches) |
| one checkpoint on disk | 14 files, 0.55 GB | `lfs find` + `stat` on `checkpoints_selftrain/j5705912_b30675li_5705912/69378` |
| one unpacked rollout member | 1500 + 1500 CSV + 9 loose files, about 69 MB | `lfs find <member_0>/data_gen -type f \| wc -l` = 1500, same for `data_real`, where `<member_0>` is
`/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z/data/hp_v5me3_AMD_s97701/member_0`; mean file sizes in §5 |
| mamba3 training | 0.565 s/step | wandb `wqgghoyj`, per `results/INVENTORY.md` §4 — **config unknown** |
| ttt_linear training | 1.46 s/step | wandb `c86ghhsn` — config unknown |

### 2.2 Tokens per step, derived without a GPU

```
context_tokens = MSG_SEQ_LEN * TOKENS_PER_MSG
               = 500 * 26
               = 13,000

tokens_per_step = PER_GPU_BSZ * GPUS_PER_NODE * SLURM_NNODES * GRAD_ACCUM_STEPS * context_tokens
```

Instantiated at the two configurations that matter:

```
probe config (attach_adaptation.sh / dfm_smoke_1gpu.yaml):
tokens_per_step = 4 * 1 * 1 * 1 * 13,000
                = 52,000

main line (configs/train/mamba3_sp500.yaml, 32 nodes, GPUS_PER_NODE unset -> 4,
           PER_GPU_BSZ unset -> 75m default 10):
tokens_per_step = 10 * 4 * 32 * 1 * 13,000
                = 16,640,000
```

This closes the derivable half of INVENTORY open item 3. It required no GPU and no
measurement — it was available in the batch script the whole time.

Two consequences that are worth stating before anyone sizes a run from them:

1. `attach_adaptation.sh` sets neither `EFFECTIVE_BSZ` nor `GRAD_ACCUM_STEPS`, so it runs
   at the batch default `GRAD_ACCUM_STEPS=1` and an effective batch of 4 sequences. The
   repository rule is that `GRAD_ACCUM_STEPS` is derived from a declared effective batch
   size and never defaulted; the script does not do this, and `train_full_autoreg.batch`
   has no `EFFECTIVE_BSZ` variable at all (it only computes the product inline for an echo
   at line 371). This must be fixed in C10, not because 1 is the wrong value but because
   nothing would report it if the node count changed.
2. `CURTAIL_EPOCHS=1500` in `attach_adaptation.sh` gives 1500 x 52,000 = **78.0M tokens**,
   which is 13x to 26x short of the 1–2B token fixed budget PLAN.md §3 Step 3 specifies.
   Either the budget or the curtail value is wrong; §4 assumes the curtail value is the
   placeholder and sizes for the budget.

### 2.3 The one short measurement that pins the rest

`0.565 s/step` cannot be converted to a wall clock for any job in §1 because the tuple it
was measured at is not recorded. This is the same defect the project has already written
down as "a timing belongs to its batch count": a step time and a step size are one
measurement, and half of it was thrown away.

Two things recover it, in cost order:

**C2, free, seconds.** Read the wandb run config:

```bash
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/miniforge3/bin/python - <<'PY'
import wandb
api = wandb.Api()
for rid in ("wqgghoyj", "c86ghhsn"):
    for proj in ("sigma0-selftrain", "lobs5-scaling-law", "sigma0-dfm"):
        try:
            r = api.run(f"oxford-lob/{proj}/{rid}")
        except Exception:
            continue
        c = r.config
        print(rid, proj, {k: c.get(k) for k in
              ("per_gpu_bsz", "bsz", "n_nodes", "gpus_per_node", "grad_accum_steps",
               "msg_seq_len", "d_model", "n_layers")})
        break
PY
```

Even after C2, converting 0.565 s/step from its config to the probe config assumes linear
scaling across a change in node count and batch size. That assumption is exactly what
manufactured the wrong MFU number on the hybrid line, so it does not get to stand alone.

**G1, 10 minutes on one free card.** Run the probe config for 60 optimizer steps with
`CHECKPOINT_EVERY=auto` and read `step_time_s` out of the log. `MODEL_PRESET=75m` prints a
per-step timing line, and `train_helpers.py:1210` already emits `MFU` and `TFLOPS` beside
it. Sixty steps is enough because the first two or three carry XLA compilation and the rest
are flat — the generation log shows the same shape, 77.3 s then 16.8 s.

Until G1 lands, **every wall clock in §2.4 marked provisional is an assumption, not a
measurement**, and the plan must not schedule against it.

### 2.4 Per-job sizing

Provisional wall clocks use a single stated assumption: that 0.565 s/step was measured at
the main-line config (32 nodes x 4 GPU x `PER_GPU_BSZ=10`), giving

```
per-GPU throughput = 16,640,000 / (128 * 0.565)
                   = 230,100 tokens/s/GPU
probe step time    = 52,000 / 230,100
                   = 0.226 s/step
```

That assumption is unverified. If C2 shows 0.565 came from a 1-node run, the probe step
time is roughly 4x larger and every provisional row below moves with it.

| job | nodes x GPUs | budget | arithmetic | wall clock | status |
|---|---|---|---|---|---|
| G1 timing | 1 x 1 | 60 steps | 60 x 0.226 s + about 6 min of squashfs mount, checkpoint load and XLA compile | about 10 min | the compile and mount overhead is provisional; the point of G1 is to replace the 0.226 |
| G2 rollouts, k 10 to 20 | 1 x 1 per member, 80 members | 80 members | 80 x 246.4 s = 19,712 s = **5.5 GPU-hours** | 21 min on 16 cards, plus per-member process start | **measured** (the 246.4 s), except process start which is not separately timed |
| G2 rollouts, k 10 to 40 | as above, 240 members | 240 members | 240 x 246.4 s = **16.4 GPU-hours** | 62 min on 16 cards | measured on the same basis |
| G3 probes | 1 x 1 | 5 checkpoint ages x (1 forward for activations + K=32 micro-batch backwards + 20 HVPs) | not measured | unknown | see Open questions |
| G4 adaptation | 1 x 1 per run, 10 runs | 1B tokens each = 19,231 steps each | 19,231 x 0.226 s = 4,346 s = 1.21 h per run; 10 runs = **12.1 GPU-hours** | 73 min on 10 cards | provisional |
| G4 at 2B tokens | as above | 38,462 steps each | 24.2 GPU-hours | 2.4 h on 10 cards | provisional |
| G5 failure-pool training | 1 x 1 per run, 12 runs (4 mixes x 3 seeds) | 1B tokens each | 14.5 GPU-hours | 73 min on 12 cards | provisional |
| G6 evaluation | 1 x 1 per checkpoint | 22 checkpoints x one generation pass | 22 x 246.4 s = 1.5 GPU-hours if the evaluation set is the same 500 contexts | 6 min on 16 cards | measured basis, but the evaluation context count is a choice not yet made |

Two sizing notes that are not wall clock:

- **G4's seed count.** PLAN.md §3 Step 2 asks for at least 5 seeds per group before any
  claim. Ten runs is the minimum, not a target. The standing rule on this line is that no
  number enters a heading or gets called cleanest until n has stopped growing, so G4 should
  report the AUC difference as a trajectory against seed count, not as its value at n=5.
- **G2's k.** The measured k needed for corrected split-half reliability 0.80, per ticker,
  from `results/failure_pool_reliability.json`: NFLX 13.3, INTC 14.8, META 16.6, AMD 18.3,
  AMZN 18.4, MSFT 20.1, GOOG 22.0, JPM 23.6. For reliability 0.90 the same fit gives 29.9
  to 53.0. k=20 clears the 0.80 line for five of eight tickers and misses for MSFT, GOOG
  and JPM. k=24 clears all eight. The choice between k=20 (80 new members, 5.5 GPU-hours,
  240,960 inodes) and k=24 (112 new members, 7.7 GPU-hours, 337,344 inodes) is an inode
  decision, not a compute decision — see §5.

### 2.5 The checkpoint ladder G4 draws from

`/lus/lfs1aip2/projects/public/u6gb/sigma-0/checkpoints_selftrain/j5705912_b30675li_5705912/`
holds 17 step directories:

```
275  22495  24080  28830  30410  33575  52590  55773  57365  58949
60532  62113  63695  65275  66853  68435  69378
```

This settles INVENTORY open item 2 for the 78M size: one long run retained an early and a
late checkpoint 252x apart in steps, and three of them (275, 33575, 69378) already have
offline weight probes in `results/a1_step*.json`. There is **no** `latest_checkpoint.json`
in that directory, which is why C3 exists.

INVENTORY open item 1 is settled for the R1-era models by
`/lus/lfs1aip2/projects/public/u6gb/sigma-0/src/post_training/heuristic_learning/checkpoints.py:40-43`,
which pins

```
CHECKPOINT_ROOT = /lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_R1_Mamba3/checkpoints
  j4559297_u52a0g05_4559297  pinned step 150360, latest step in dir 168200
  j4553948_tz2tmn5z_4553948  pinned step 120000, latest step in dir 143150
```

### 2.6 A coherence problem between the two threads

The rollouts that every failure-pool measurement rests on were generated from

```
/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z/ckpt/wm_ft_multi3
```

at step 69378 (`inference.log` line 9), which is a **different run** from

```
/lus/lfs1aip2/projects/public/u6gb/sigma-0/checkpoints_selftrain/j5705912_b30675li_5705912
```

at step 69378, which is what `attach_adaptation.sh:56` restores. The step numbers coincide;
the weights do not. As the plan stands, Thread A adapts one set of weights and Thread B
selects failures from another. Either G2 regenerates from the selftrain chain, or G4/G5
restore from `wm_ft_multi3`, but not both. This is a decision, not a measurement, and it
belongs in the science drafts — flagged here because it changes what G2 must produce.

---

## 3. The attach recipe, exactly

### 3.1 The command

The proven form on this system, taken from the header of
`/lus/lfs1aip2/projects/public/u6gb/sigma-0-m3/baselines/m3/run_m3_scaling_attach.sh` and
matching CLAUDE.md §4.0.1:

```bash
# 1. Find which node has which idle card. Per-card memory is the judge; the gtop
#    header count includes held cards (0% util, tens of GB resident) and is not usable.
timeout 170 gtop --once 2>/dev/null | tr -d '\000' | awk '
  /^ ▸ job/{j=$3} /^   nid/{n=$1}
  /GH200/{ if ($0 ~ /idle/ && $0 ~ /mem +0\.0\//) {
    match($0,/\[[0-9]\]/); print j, n, substr($0,RSTART+1,1) } }'
# -> 6269978 nid010436 1

ALLOC=6269978; NODE=nid010436; CARD=1; RUNTAG=cl-adapt-e275

# 2. Take all four cards, disable CPU binding, give the step a real name,
#    and choose the card inside the process.
srun --overlap --jobid="$ALLOC" --nodelist="$NODE" \
     --nodes=1 --ntasks=1 --ntasks-per-node=1 \
     --gres=gpu:4 --cpu-bind=none \
     --job-name="$RUNTAG" \
     bash -lc "export CUDA_VISIBLE_DEVICES=$CARD; \
               export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5; \
               exec bash /path/to/step.sh" \
     < /dev/null
```

Run inside tmux, never bare on the login node:

```bash
TMUX="/tools/brics/apps/linux-sles15-neoverse_v2/gcc-12.3.0/tmux-3.4-5vcftkte724cekyuashr2ex65c5fpfxj/bin/tmux -S /tmp/tmux-$(id -u)/default"
$TMUX new-session -d -s cl-exec
$TMUX send-keys -t '=cl-exec:0' 'bash /lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/code/launch_one.sh 6269978 nid010436 1' Enter
```

### 3.2 Each constraint, and the job it binds

| constraint | why | which jobs |
|---|---|---|
| `--gres=gpu:4`, card chosen inside the process | `--gres=gpu:1` binds whatever logical device Slurm picks, which is device 0, and Slurm's gres accounting disagrees with physical occupancy. In the snapshot in §0 not one free card is device 0 | G1, G2, G3, G4, G5, G6 — every GPU job |
| `--cpu-bind=none` | the step dies with `Unable to satisfy cpu bind request` when the node's CPUs are busy, which is unrelated to GPU availability. `train_full_autoreg.batch:569` already sets `CPU_BIND="none"` whenever `GPUS_PER_NODE < 4`, and `:571` honours `CPU_BIND_OVERRIDE` | all GPU jobs; automatic for the training path because it runs with `GPUS_PER_NODE=1`, must be explicit for G2 and G6 which do not go through that batch |
| `--export=ALL` carries the login node's `TMPDIR` | `train_full_autoreg.batch:580` has `--export=ALL`. For the training path the damage is contained: `node_wrapper.sh:19` unconditionally overwrites `TMPDIR` with `/tmp/$USER/sigma0/${SLURM_JOB_ID}_${SLURM_PROCID}` and fails loudly at `:15-18` if that is not writable. **The residual hazard is `SIGMA0_JOB_TMPDIR`** (`node_wrapper.sh:13`), which *is* inherited and, if it points at a login-node-only path, kills the rank with `[Wrapper] FATAL: job-local temporary directory is unavailable`. `--export=ALL` also carries the outer step's `SLURM_*` variables, which is why `attach_adaptation.sh:62` unsets them all before setting four by hand | G1, G4, G5 (training path); G2 and G6 must not export a login-node `TMPDIR` |
| a step needs a real `--job-name` | the node budget monitor classifies a step named `bash` as idle and a running training job can be counted as idle-held. `train_full_autoreg.batch` puts no `--job-name` on its `srun` line at all (573-581) — the name only exists on the `#SBATCH` directive and in the resubmit | every attached step. `attach_adaptation.sh:70` already injects `--job-name=$RUNTAG` via its sed rewrite, which is the one thing that script gets exactly right |
| `srun` inside `while read` eats the loop's stdin | the fan-out loop over (ticker, card) reads a node list; the first `srun` consumes the rest of it and the remaining runs never start, silently. Redirect: `srun ... < /dev/null &` | G2 and G6 only — they are the jobs with a fan-out loop. G1/G4/G5 launch one step each |
| the `srun` client lives on the login node and dies with the session | `setsid nohup` is not enough; the client is the step's only control channel and a session teardown reaps it. Anything that must outlive the session goes in tmux (the server is node-local and owned by init) or in a batch job | G2 (about 20 min per fan-out wave), G4 and G5 (hours each). G1 is short enough to babysit |

### 3.3 A trap specific to this repository

`--gres` on the training path is written as `--gres=gpu:$GPUS_PER_NODE`
(`train_full_autoreg.batch:576`), and `GPUS_PER_NODE` simultaneously controls three things:
the gres request, whether `--cpu-bind=none` is emitted (`:569`), and the device list Python
builds (`runtime/train.py:32`, `local_device_ids` at `:353`). Setting it to 4 to get four
cards would silently drop `--cpu-bind=none` and make JAX claim all four. The four uses have
to be separated, which is what C9 and C10 do.

---

## 4. Ordering under uncertain GPU availability

The instruction is to run the plan and wait for GPUs, not to let GPU availability choose
the work. Two rules make that concrete:

1. **The order is the dependency graph's topological order.** A job never moves up because
   a card became free, and never moves down because one did not.
2. **Waiting is not idling.** There are ten CPU-only jobs and two code changes; three of
   them (C4, C8, C9) are hard prerequisites of the first GPU job. Free cards are free, so
   the moment a GPU job's prerequisites are met it starts — but the prerequisites are not
   skipped to reach a card sooner.

### 4.1 The order

```
wave 0 (no allocation needed, all of it in parallel)
  C1  tokens/step                    minutes, already done in §2.2
  C2  wandb config for 0.565         minutes
  C3  breadcrumb                     minutes
  C4  context set + sha256           minutes
  C8  inode budget                   minutes
  C5  reliability curve rerun        tens of minutes
  C7  weight probes, 17 checkpoints  tens of minutes
  C9  patch runtime/train.py         + test
  C10 patch attach_adaptation.sh     + a dry run that prints the rewritten srun

wave 1 (first card)
  G1  60-step timing                 10 min, 1 card
        -> replaces every provisional number in §2.4

wave 2 (fan out over whatever is free)
  G2  rollout regeneration           4.1 min per member, embarrassingly parallel
  G3  activation/gradient probes     independent, 1 card

wave 3 (needs G1's step time to size the budget)
  G4  early-vs-late adaptation       10 runs, 1 card each
  G5  failure-pool training          12 runs, 1 card each, needs G2

wave 4
  G6  evaluation                     1 card per checkpoint
  C6  replay scoring                 CPU, off the G6 outputs
```

### 4.2 What runs while waiting

Wave 0 in full, and then C6 on the rollouts that already exist. C6 is the largest
unrecognised free job in this plan: `fidelity.py report --episodes <run dir> --stock <T>`
and `autopsy.py --episodes <run dir> --stock <T>` both score existing generation
directories with no model and no card, and the second scoring axis of issue #73 (match to
the true system) currently has no measurement at all. There are 2308 arm directories under
`/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z/data/`;
the eight `hp_v5me3_*` families alone are 80 members already on disk.

### 4.3 Checkpoint and resume, per long job

The requirement is that an expiring allocation costs minutes, not hours.

| job | granularity | mechanism | worst-case loss |
|---|---|---|---|
| G2 rollouts | one member | `collect_rollouts.sh:95-98` skips a member whose `.done` exists, and `:186-190` refuses to write `.done` when inference exited non-zero, so a crashed member is retried rather than silently half-counted | one member, about 4 min |
| G1 timing | none needed | 10 min total | the whole run, 10 min |
| G4, G5 training | `CHECKPOINT_EVERY=auto` -> a save every 900 s, first save at about 300 s (`train_helpers.py:968-972`) | plus `train_helpers.py:1233-1238`: when `(max_job_hours - elapsed) * 60 <= save_before_timeout_minutes` it forces a save | up to 15 min |
| G6 evaluation | one checkpoint | same `.done` pattern as G2 if it reuses that driver | one checkpoint |

Three things must be true for the G4/G5 row to hold, and only the first is true today:

1. `CHECKPOINT_EVERY=auto` decouples loss logging from checkpointing:
   `AUTO_WANDB_INTERVAL = 60` and `AUTO_CKPT_INTERVAL = 900` (`train_helpers.py:968-969`),
   evaluated independently at `:1226-1231`. **An explicit numeric `CHECKPOINT_EVERY`
   couples them** — `:1221-1223` sets `should_ckpt` and `should_wandb` from the same
   modulo — and this repository has no `LOG_EVERY` to separate them again. So the auto mode
   is not a preference here, it is the only setting that satisfies the one-minute /
   fifteen-minute rule.
2. `MAX_JOB_HOURS` must be set to the **attached allocation's remaining walltime minus a
   margin**, not to a fixed 3.0. The timeout-imminent save at `:1233` keys off
   `max_job_hours`, which under attach has no relation to when the host allocation actually
   expires. `attach_adaptation.sh:54` hard-codes 3.0. If the allocation has 40 minutes left,
   the forced save never fires and the run dies between two 15-minute saves.
3. Auto-resume is off under attach (`NO_AUTO_RESUME=1`, `NO_AUTO_RESUME_DEPTH=99` in
   `attach_adaptation.sh:31-32`) and must stay off: the batch's resume path resubmits with
   `sbatch --dependency=afterany:${SLURM_JOB_ID}` (`train_full_autoreg.batch:726-729`), and
   under attach `SLURM_JOB_ID` is the **host allocation's** id, so a resume would chain a
   new job onto someone else's allocation. Resume therefore has to be manual: read the last
   step from the breadcrumb C3 writes and relaunch with `RESTORE_PATH` / `RESTORE_STEP`.

---

## 5. Artefact retrieval and the inode budget

### 5.1 The constraint, measured

```
$ lfs quota -h -p <project> /lus/lfs1aip2
  /lus/lfs1aip2  127.1T  0k  200T  -  50376941  0  51200000  -
```

Space: 127.1 T of 200 T, comfortable. Inodes: 50,376,941 of 51,200,000 — **823,059 free,
1.6 percent headroom**. Renaming does not free an inode, so the only lever is not creating
the file. `/home` (VAST) has 22.68 billion free inodes and is the escape hatch for logs and
watcher output, not for artefacts the repository needs.

### 5.2 What one rollout member costs

Measured on
`/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z/data/hp_v5me3_AMD_s97701/member_0/`:

| stream | files | mean bytes | total |
|---|---|---|---|
| `data_gen/` message | 500 | 9,134 | 4.6 MB |
| `data_gen/` orderbook | 500 | 58,138 | 29.1 MB |
| `data_gen/` provenance | 500 | 524 | 0.3 MB |
| `data_real/` message | 500 | 10,544 | 5.3 MB |
| `data_real/` orderbook | 500 | 58,292 | 29.1 MB |
| `data_real/` refcheck | 500 | 1,500 | 0.8 MB |
| loose (`.returns_*.npz` x4, `inference.log`, `sample_indices_rank0.json`, `.done`, dirs) | 9 + 3 dirs | — | 0.1 MB |
| **total** | **3,012 inodes** | | **about 69 MB** |

`collect_rollouts.sh:99-105` states the same arithmetic at the 2,000-context scale
(12,004 files per member) and records the cumulative cost: *"1,529 members collected this
way so far cost ~18.3M inodes, 36% of the project quota, and the quota is at its cap — one
more unpacked member does not fit."*

### 5.3 The budget for G2

| plan | members | inodes | fraction of the 823,059 free | bytes |
|---|---|---|---|---|
| k=20, unpacked, real duplicated (as today) | 80 | 240,960 | 29% | 5.4 GB |
| k=24, unpacked, real duplicated | 112 | 337,344 | 41% | 7.6 GB |
| k=40, unpacked, real duplicated | 240 | 722,880 | **88%** | 16.2 GB |
| k=20, real written once per ticker | 80 | 132,960 | 16% | 2.9 GB |
| k=24, real written once per ticker | 112 | 181,344 | 22% | 4.0 GB |

**The real arm is byte-identical across seeds.** Verified: the same filename in
`hp_v5me3_AMD_s97701/member_0/data_real/` and `hp_v5me3_AMD_s97702/member_0/data_real/`
has md5 `85d5adb955b52e8135d1fea1a56c73d4` in both. The seeds are consecutive
(97701..97710) and the context set is frozen, so this holds by construction, not by luck.
Writing it once per ticker and symlinking or referencing it saves 9/10 of 1,500 files per
member: **108,000 inodes on the 80 members already on disk**, and 108,000 more on any k=20
regeneration. That single change is worth more than the difference between k=20 and k=40.

### 5.4 Why `pack_member.py` is not the answer

`run/mid_training/pack_member.py` collapses a member from 12,004 files to 3. It is
tempting and it is wrong for this plan, for reasons already written down in
`collect_rollouts.sh:106-128`: packing discards `data_cond/` (the simulator's
initialisation input, without which a replay cannot reproduce the run), `data_tokens/`,
and the `message_*.csv` files that `onpolicy_hist.measure()` reads to build the
spread-regime histogram. That histogram is the input to the distributional-match axis —
the second of the two axes issue #73 asks for. The comment at `:107` says the default was
flipped from 1 to 0 precisely because *"the docstring in pack_member.py calls the discarded
files raw material that has already been consumed. That is true for CRPS and qL1 and false
for every other consumer, and the default made the false half authoritative."*

So: `PACK_MEMBER=0` stays, and the inode saving comes from not duplicating the real arm,
not from throwing away the generated one.

### 5.5 Retrieval, per job

Node-local scratch disappears when the allocation expires, and an expired allocation
refuses further `srun` — so retrieval is per run, never per batch.

| job | writes to | retrieved when | lands at |
|---|---|---|---|
| G1 timing | stdout only | immediately | `/lus/lfs1aip2/projects/public/u6gb/sigma-0/logs_cl_probe/<tag>/` |
| G2 rollouts | `$MEMBER_DIR` directly on Lustre with `PACK_MEMBER=0` (`collect_rollouts.sh:134`), so nothing to retrieve | — | `<OUT_ROOT>/hp_<config>_<TICKER>_s<SEED>/member_0/` |
| G3 probes | one JSON per checkpoint age | at the end of each age | `tasks/continual_learning/results/` |
| G4, G5 training | checkpoints go **straight to Lustre**, not to node-local. `train.py:455-462` builds `ckpt_dir` from `CHECKPOINT_BASE_DIR`, and there is **no `rsync` anywhere in `train_full_autoreg.batch` or `node_wrapper.sh`** | continuously | `CHECKPOINT_BASE_DIR/<run>_<wandb>_<jobid>/<step>/` |
| G6 evaluation | same as G2 | per checkpoint | as G2 |

Two corrections to assumptions the brief inherits:

- **Checkpoints are not staged to `$TMPDIR` and rsynced back.** They are written directly
  to Lustre with Orbax `save_interval_steps=1, max_to_keep=10, keep_period=5,
  enable_async_checkpointing=False` (`src/lob/train.py:479-486`). `TMPDIR` is used only for
  wandb and squashfs mount points (`node_wrapper.sh:20,370,421`). The cost is 14 files and
  0.55 GB per save, so at most 180 inodes and 5.5 GB per run — negligible in inodes,
  55 GB across ten G4 runs, which is worth stating because space is the one budget with
  room.
- **Resume does not read a breadcrumb.** `train_full_autoreg.batch:601,609` uses
  `lfs find --maxdepth 1` on the checkpoint base and takes the numerically largest step
  directory. That is the Lustre-native safe form and not a policy violation, but it is not
  a breadcrumb, and there is no `latest_checkpoint.json` in the selftrain chain. C3 writes
  one so that manual resume under attach (§4.3) needs no listing at all.

---

## 6. The monitoring contract

One resident watcher. It reports; it does not act.

### 6.1 What it checks, every 900 s

| check | how | why not the obvious way |
|---|---|---|
| idle cards | per-card `mem 0.0/95.6G` and `idle` from `gtop --once`, counting the per-card lines | the header `idle` count includes held cards (0% util with tens of GB resident) and would report cards that are not free |
| the pending queue | `squeue -u $USER -t PENDING -h -o "%.10i %.26j %.4D %R"` | one side alone is not a decision; "there is a free card" is not an event, "there is a free card **and** something waiting for one" is |
| dead dependencies | `Reason=DependencyNeverSatisfied` in the same `squeue` output | such a job will never run and cannot be cancelled by the watcher; it must be reported with its job id |
| its own probe | a `gtop` that produced no output, or a `Traceback`, is reported as a failed probe | `grep -c` on a crashed `gtop` returns 0, which reads as "no free cards" and hides exactly the waste the watcher exists to find |
| running step names | `squeue -s -j <alloc> -o "%.24j"` | the default `NAME` column is 8 characters wide, so grepping a 15-character job name against the default format returns nothing and reads as "the step died" |

Threshold: report when `idle_cards >= 4 AND pending >= 1`, **independently of whether this
session has work of its own**. The dispatch list is the account's pending queue.

### 6.2 What it must never do

Never `sbatch`, never `srun`, never attach, never `scancel`, never restart anything. It is
a read-only poll that writes a log line. That is what keeps it inside the site rule against
resident agents that take unattended actions, and it is what makes it incapable of causing
the problem it watches for.

### 6.3 Silence has to mean something

**Every round writes one line, including the rounds where nothing is actionable.** Without
that, an empty log is ambiguous between "the cluster is busy and there is nothing to
report" and "the watcher died an hour ago". The three previous watchers on this account
produced zero lines each and the failure was invisible for hours in every case.

A reader distinguishes the two cases by timestamp: if the newest line is older than about
two intervals (30 min), the watcher is dead and must be restarted. Concretely:

```
2026-09-04T21:52:07Z  idle=16 pending=2 allocs=9  ACTIONABLE idle>=4 pending>=1
    free: 6317365/nid010234[3] 6317365/nid010288[3] ... 6269978/nid010439[3]
    pend: 6317366 u6gb-4-node-chain 4N Resources | 6243552 l2c-s44-driver 1N JobHeldUser
2026-09-04T22:07:07Z  idle=0 pending=2 allocs=9   quiet
2026-09-04T22:22:08Z  PROBE FAILED gtop produced no per-card lines
```

### 6.4 Three mechanical requirements

1. **Singleton.** The script takes an `flock` at startup and a second invocation exits as a
   no-op. Without this, the rule "confirm the watcher is running at the start of every
   session" produces one watcher per session; seven concurrent copies were observed on this
   account on 2026-09-04, all polling `gtop` and appending to the same file.
2. **Log to `$HOME`, not to the project.** The project inode quota is at 98.4 percent and
   appending to a file that does not yet exist fails silently when it is full. `/home` has
   22.68 billion free inodes.
3. **Strip NUL before anything reads the log.** `gtop` output contains NUL bytes; once they
   are in the log, `grep` classifies the file as binary and prints `Binary file matches`
   instead of the matching lines, which is indistinguishable from no match. Pipe through
   `tr -d '\000'` on write and read the log with `awk`, not `grep`.

Never kill a watcher by matching a substring of its command line: that matches the wrapper
shell that launched it as well, and killing the wrapper kills the session. Match `argv`
exactly.

---

## 7. First-hour concrete schedule

Assuming the plan is approved at T+0. Times are elapsed minutes. Nothing here submits a
job to the queue; §0's snapshot shows 16 idle cards, and the first GPU job attaches.

| T+ | what | command |
|---|---|---|
| 0 | start the watcher, before anything else | `flock -n /home/u6gb/kangli.u6gb/.gpu_watch.lock bash /home/u6gb/kangli.u6gb/gpu_watch_15min.sh &` inside tmux; confirm a line appears in `/home/u6gb/kangli.u6gb/gpu_watch_15min.log` within 60 s |
| 1 | C1 — record the tokens-per-step identity in the plan file | already derived in §2.2; nothing to run |
| 2 | C2 — recover the config behind 0.565 s/step | the wandb snippet in §2.3 |
| 4 | C8 — re-read the inode budget so the numbers in §5 are current | `lfs quota -h -p $(lfs project -d /lus/lfs1aip2/projects/public/u6gb \| awk '{print $1}') /lus/lfs1aip2` |
| 5 | C3 — write the breadcrumb | one python heredoc writing `{"steps": [275, ..., 69378], "latest": 69378, "root": "/lus/lfs1aip2/projects/public/u6gb/sigma-0/checkpoints_selftrain/j5705912_b30675li_5705912"}` to `latest_checkpoint.json` in that directory |
| 8 | C4 — hash the frozen context set | `sha256sum /lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z/data/hp_v5me3_AMD_s97701/member_0/sample_indices_rank0.json` for each ticker; confirm the eight hashes are stable across seeds |
| 10 | C9 — patch `runtime/train.py` | make line 32 honour an explicit list: `os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["GPU_IDS"] if os.environ.get("GPU_IDS") else ",".join(str(i) for i in range(_n_gpus))`, and assert `len(GPU_IDS.split(",")) == _n_gpus`. Add a test that sets `GPU_IDS=3`, `GPUS_PER_NODE=1` and fails on the current behaviour |
| 20 | C10 — patch `attach_adaptation.sh` (the four changes in §8) and dry-run it | run with a `DRY_RUN=1` guard that prints the rewritten `srun` line and exits; check by eye that it carries `--gres=gpu:4`, `--cpu-bind=none`, `--job-name=$RUNTAG`, and `--jobid=$ALLOC` |
| 30 | pick a target card and gate it | the awk snippet in §3.1; then confirm with `srun --overlap --jobid=$ALLOC -w $NODE --gres=gpu:4 --cpu-bind=none --job-name=gate nvidia-smi --query-gpu=index,memory.used --format=csv` — 1–9 MiB is free, tens of GB is held |
| 35 | **G1** — 60-step timing on that card | `attach_adaptation.sh 275 t60 $ALLOC $NODE` with `CURTAIL_OVERRIDE=60 MAX_HOURS_OVERRIDE=0.4 GPU_IDS=$CARD` |
| 50 | read the step time out of the G1 log and replace every provisional number in §2.4 | grep the per-step timing line the training loop prints (`train_helpers.py:1206-1213`) |
| 55 | C5 and C7 in parallel on the login node while G1 runs | `python code/failure_pool_reliability.py --ks 1 2 3 5 --draws 40 --out results/failure_pool_reliability_d40.json`; `JAX_PLATFORMS=cpu python code/probe_weights_offline.py --root <chain> --steps 275 22495 24080 28830 30410 33575 52590 55773 57365 58949 60532 62113 63695 65275 66853 68435 69378 --out results/a1_all17.json` |
| 60 | decide k for G2 from §2.4 and the §5.3 budget, then start the first G2 wave in tmux | one `srun` per free card, each `< /dev/null &`, one member per invocation |

Everything before T+35 runs on the login node and needs no allocation. If no card is free
at T+30, the schedule does not change: C5, C7 and C6 continue, and G1 starts when a card
appears.

---

## 8. Verdict on the two existing scripts

### 8.1 `/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/code/submit_adaptation_pair.sh`

**Reusable as written, but it is the wrong tool most of the time.** It is a plain `sbatch`
of `run/base_model/train_full_autoreg.batch` with the probe environment set, one node,
`--gres=gpu:1`, `--time=03:30:00`. It is correct — sbatch gets a whole node, so device 0 is
free and Fact B does not bite — and it should be used only when `gtop` genuinely shows no
free card. Two things to fix:

- it sets neither `EFFECTIVE_BSZ` nor `GRAD_ACCUM_STEPS`, so the effective batch is 4
  sequences by default rather than by declaration (§2.2);
- `CURTAIL_EPOCHS=1500` is 78.0M tokens against a 1–2B budget (§2.2).

### 8.2 `/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/code/attach_adaptation.sh`

**Not reusable as written. Four changes, in order of how much time each has cost here.**

| # | problem | evidence | change |
|---|---|---|---|
| 1 | asks for `--gres=gpu:1` and lands on device 0 | its sed rewrites only line 573 of the batch; line 576 `--gres=gpu:$GPUS_PER_NODE` survives, and it exports `GPUS_PER_NODE=1`. In the §0 snapshot no free card is device 0 | extend the sed to rewrite line 576 to `--gres=gpu:${ATTACH_GRES:-$GPUS_PER_NODE}`, export `ATTACH_GRES=4`, keep `GPUS_PER_NODE=1` (which also keeps `CPU_BIND="none"` at batch line 569), and pass the card through the `GPU_IDS` hook that C9 adds |
| 2 | uses `setsid nohup ... &` | the `srun` client lives on the login node and is reaped with the session; `setsid` does not change that. 48 cards idled for 2h10m on this account for exactly this reason | launch inside tmux on the socket `/tmp/tmux-$(id -u)/default` |
| 3 | `MAX_JOB_HOURS=3.0` is unrelated to the host allocation's remaining walltime | the forced save at `train_helpers.py:1233` keys off `max_job_hours`; under attach that number is a guess | compute it from `squeue -j $ALLOC -h -o %L` minus a 20-minute margin, and refuse to launch if the remainder is under one checkpoint interval |
| 4 | no declared effective batch size | same as §8.1 | declare `EFFECTIVE_BSZ`, derive `GRAD_ACCUM_STEPS`, fail loudly if a caller passes an inconsistent `GRAD_ACCUM_STEPS`, and print one `[bsz]` line so the log records what actually ran |

Three things it already does right and that should be preserved: it unsets every inherited
`SLURM_*` before faking four (line 62), it verifies its own sed took effect and exits 5 if
not (lines 72-75), and it injects `--job-name=$RUNTAG` so the step is not counted as idle
(line 70). It also gives each run a unique squashfs mount root (line 40), which avoids the
stale-mount collision between the two members of a pair.

---

## Open questions

1. **Which config produced 0.565 s/step?** Until C2 answers it, every wall clock in §2.4
   marked provisional rests on an assumption (main line, 32 nodes) that is stated but not
   checked. If it was a 1-node run, the G4 estimates are roughly 4x too optimistic.
2. **Can the plasticity probes (G3) run on CPU?** `plasticity_probes.py` is pure numpy, so
   only its inputs need a forward and backward pass. A 78.5M-parameter model at a
   13,000-token context has never been timed on CPU here. If one forward pass finishes in
   under a few minutes, G3 joins the CPU-only column and stops competing for cards. This is
   measurable in about 15 minutes on the login node under `JAX_PLATFORMS=cpu` with a batch
   of 1, and nobody has done it.
3. **Which checkpoint does the plan adapt?** §2.6: the rollouts come from
   `/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z/ckpt/wm_ft_multi3`
   step 69378, the adaptation scripts restore from
   `/lus/lfs1aip2/projects/public/u6gb/sigma-0/checkpoints_selftrain/j5705912_b30675li_5705912` step 69378.
  
   Same step number, different weights. Until this is decided, G2's output cannot feed G5.
4. **What were `N_COND` and `N_GEN` for the `hp_v5me3_*` members?** The horizon list reaches
   250, so `N_GEN >= 250`, and the conditioning tensor is `(48, 13001)` which is 500
   messages. But the `.done` file for those members is **zero bytes** — the manifest that
   `collect_rollouts.sh:140-167` now writes was added after they were generated. New members
   generated today would carry a manifest; the existing ones do not, so a new member is
   only exchangeable with the old ones if every parameter is recovered from
   `inference.log` first. That recovery is a CPU job nobody has scheduled.
5. **Is `mm_sim.py` in scope at all?** It simulates a market-making policy over a frozen
   episode and writes nothing to disk. It is CPU and free, but no draft names a decision it
   feeds. If it is not in scope, `episode_builder.py`'s `build_episode` / `save_episode`
   have no driver either — nothing in that module calls them — and the episode machinery is
   dead code for this plan.
6. **How many contexts should G6 evaluate on?** §2.4's 1.5 GPU-hour figure assumes the same
   500 contexts as the failure-pool work. A larger evaluation set is cheap in GPU time and
   expensive in inodes (3,012 per member), so the two budgets have to be traded explicitly.
7. **Does `--exact` help or hurt on a busy node?** `attach_adaptation.sh:70` adds
   `--exact --cpus-per-task=64`; the proven m3 attach form does not. Neither has been tested
   against a node whose CPUs are saturated, which is the case the `--cpu-bind=none`
   requirement exists for.
8. **G5's mix grid is not yet a design.** Issue #73 floats 70/30. §2.4 sizes 4 mixes x 3
   seeds on the assumption that one of the four is a matched-size random pool serving as the
   zero-effect group. Whether that group is genuinely null — whether it shares an error term
   with the treatment — is a science question, not an execution one, and the sizing changes
   if the answer needs a different control.
