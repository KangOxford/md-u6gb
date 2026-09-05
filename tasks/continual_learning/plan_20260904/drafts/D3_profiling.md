# D3 — The profiling programme for the continual-learning line

> Draft 3 of 5, planning round 2026-09-04. Scope: the cost model, what must be
> profiled and in what order, one attached-step profiling specification, the standing
> instrumentation, and the token-budget to GPU-hour conversion table.
> No GPU job was run to produce this draft. Every number below is either read from an
> existing log on Lustre, derived from source, or left symbolic and marked.

## 中文速览

- **INVENTORY 的第 3 号未决项（tokens/step 从未记录）其实不需要 GPU 就能关掉。** 训练代码在起跑时
  就打印 `[FLOPs] Tokens/step:`，已有日志里全都在。实测：`micro_bsz x GPU 数 x 13,000`。
- **13,000 不是猜的**：`msg_seq_len=500` x `Message_Tokenizer.MSG_LEN=26`，两端都追到了最后一环。
- **s/step 也不必重测**：`j5705912` 的 693 个采样点给出 0.5687 s/step、sd 0.0044（cv 0.77%），
  与 INVENTORY 里那个来路不明的 0.565 吻合。四个 job 拼出 1/4/8/16 卡的扩展曲线。
- **换算常数：78.5M mamba3 上，每 10 亿 token 约 3.04 GPU-小时**（1 卡），8 卡时 3.29。
- **MFU 的分母是错的一半**：代码除的是 989 TFLOPS（GH200 稠密 BF16 张量核峰值），而 SSD 收缩
  默认跑 `Precision.HIGHEST`（`MAMBA3_CONTRACTION_PRECISION` 默认 `highest`），投影层跑 TF32。
  按 TF32 稠密峰值 494.5 读数正好翻倍：11.45% -> 22.9%。按 FP32 CUDA 核 67 读会超过 100%，
  这本身就证明活不在 FP32 核上。
- **三个必须先修的接线缺陷**（详见 §8）：`throughput/tokens_per_sec` 根本不是 token 每秒；
  `--val_split` 是死旋钮，这条线现在**完全没有训练中验证**，而 PLAN §2.4 要求双坐标；
  FLOPs 的 mamba3 判定挂在一个被别处副作用赋值的字段上。
- 分析产物按「图 > 表 > 文字」呈现的规矩这里不适用（这是规格文件），但下游的 profiling notebook 适用。

---

## 1. What this draft settles, and what it deliberately leaves open

| # | Question | Status after this draft |
|---|---|---|
| 1 | Tokens per step at the production setting | **Settled from existing logs, no GPU.** §2, §3 |
| 2 | s/step and its variance | **Settled for 1/4/8/16 GPU at 78.5M**, §3. Not settled for the 128-GPU 75m main line |
| 3 | MFU and its divisor | Divisor identified and shown to be ambiguous; §4 specifies the one measurement that resolves it |
| 4 | Peak memory, arms per card | **Not measured.** Only `bytes_in_use` between steps is logged (681 MB); peak during a step is nowhere. §4 P2 |
| 5 | Marginal cost of each plasticity probe | Derived analytically with the formulas in §5; the measurement that confirms them is P4 |
| 6 | Token budget to GPU-hours | Table in §7, filled where measured, symbolic elsewhere with the formula shown |

---

## 2. The cost model, derived from the configuration

### 2.1 The chain, end to end

The launch chain for the main line is

```
/lus/lfs1aip2/projects/public/u6gb/sigma-0/configs/train/mamba3_sp500.yaml
  -> /lus/lfs1aip2/projects/public/u6gb/sigma-0/run/base_model/train_base_model.py   (planner only, prints an sbatch line)
  -> /lus/lfs1aip2/projects/public/u6gb/sigma-0/run/base_model/train_full_autoreg.batch
  -> srun -> /lus/lfs1aip2/projects/public/u6gb/sigma-0/run/base_model/node_wrapper.sh
  -> /lus/lfs1aip2/projects/public/u6gb/sigma-0/run/base_model/runtime/train.py
  -> /lus/lfs1aip2/projects/public/u6gb/sigma-0/src/lob/train.py
```

### 2.2 The arithmetic

```
seq_len            = msg_seq_len * MSG_LEN
                   = 500 * 26
                   = 13,000 tokens per training sample

tokens_per_micro_step = micro_bsz * total_devices * seq_len
                      = PER_GPU_BSZ * (GPUS_PER_NODE * NNODES) * 13,000

tokens_per_opt_step   = tokens_per_micro_step * GRAD_ACCUM_STEPS

FLOPs_per_opt_step    = correction(d_model) * 6 * N_params * tokens_per_opt_step

MFU                   = FLOPs_per_opt_step / (s_per_opt_step * n_GPU * peak_FLOPS_per_GPU)
```

Instantiated at the two settings that matter for this line:

| Setting | PER_GPU_BSZ | GPUs | K | tokens/opt-step | N_params | correction |
|---|---:|---:|---:|---:|---:|---:|
| `mamba3_sp500.yaml` production (MODEL_PRESET=75m, 32 nodes) | 10 | 128 | 1 | **16,640,000** | 78,539,423 | 2.630 |
| Continual-learning probe (`attach_adaptation.sh`, 1 GPU) | 4 | 1 | 1 | **52,000** | 78,539,423 | 2.630 |

```
production:  10 * 128 * 13,000 * 1 = 16,640,000 tokens per optimizer step
CL probe:     4 *   1 * 13,000 * 1 =     52,000 tokens per optimizer step
```

The CL-probe number is not a derivation: it is printed verbatim as
`[FLOPs] Tokens/step: 52,000` in
`/lus/lfs1aip2/projects/public/u6gb/sigma-0/logs_lobs5/training_5705912_node0.log`.

**One corpus epoch, for scale.** 30.06e9 messages x 26 tokens = **781.6e9 tokens**.
Cross-checked independently: the tqdm total in
`/lus/lfs1aip2/projects/public/u6gb/sigma-0/logs_lobs5/training_5877859_node0.log`
is 939,147 per-process steps at 16 samples per process-batch over 4 processes, i.e.
60.1e6 windows x 13,000 = 7.81e11 tokens. The two routes agree to three digits.

### 2.3 Assignment-chain audit — the LAST link of every input

The repo has a documented history of knobs that are set, logged, and never reach the code,
so each value below is traced to the last assignment that survives, not the first.

| Value | First link | **Last link that survives** | Verdict |
|---|---|---|---|
| `msg_seq_len` | `MSG_SEQ_LEN=${MSG_SEQ_LEN:-500}` at `train_full_autoreg.batch:375` | `--msg_seq_len="${MSG_SEQ_LEN:-500}"` at `node_wrapper.sh:557`; argparse default 500 at `runtime/train.py:90` | **Live**, value 500 |
| `MSG_LEN` (tokens/message) | comment "26tok" | `TOK_LENS = (1,1,3,2,1,3,2,3,3,2,2,3)` summed at `/lus/lfs1aip2/projects/public/u6gb/sigma-0/src/lob/encoding.py:353`; consumed at `lobster_dataloader.py:945` | **Derived, = 26.** Verified by summation, not by reading the comment |
| `seq_len` | not a knob | `self.seq_len = self.n_messages * Message_Tokenizer.MSG_LEN` at `lobster_dataloader.py:945` -> `dataloading.py:264` -> `lob/train.py:103` | **13,000** |
| `micro_bsz` | `MODEL_PRESET` case at `train_full_autoreg.batch:236/244` | `--micro_bsz="$PER_GPU_BSZ"` at `node_wrapper.sh:588` | **Live**; 10 for 75m preset, 4 in both CL attach scripts |
| `num_devices` | `--num_devices="$GPUS_PER_NODE"` at `node_wrapper.sh:570` | **`args.num_devices = jax.local_device_count()` at `runtime/train.py:361`** | **Overridden whenever `SLURM_NNODES > 1`.** The CLI flag is dead on multi-node and live on single-node. Any single-card profiling run must therefore set `SLURM_NNODES=1` **and** `GPUS_PER_NODE=1` |
| `grad_accum_steps` | `GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}` at `train_full_autoreg.batch:368` | `${GRAD_ACCUM_STEPS:+--grad_accum_steps=...}` at `node_wrapper.sh:628`; argparse default 1 | **Live, = 1** on this line. The `:-1` default is the exact pattern the repo rule bans; see §8 item R3 |
| `token_mode` | `env_TOKEN_MODE: 26tok` in the yaml | **`--token_mode=26tok`, a string literal, at `node_wrapper.sh:598`** | **Dead knob.** Pinned in five independent places (batch guard, wrapper literal, argparse `choices=["26tok"]`, `init_train.py:356` raise, encoding module). Do not plan any experiment that varies it |
| `correction` gate | `_is_mamba3 = getattr(args,'ssm_type','gdn')=='mamba3'` at `/lus/lfs1aip2/projects/public/u6gb/sigma-0/src/lob/train.py:373` | `args.ssm_type` is assigned as a side effect by `stamp_architecture` at `/lus/lfs1aip2/projects/public/u6gb/sigma-0/src/s5/registry.py:360`, reached through `init_train_state` at `lob/train.py:145` | **Live but fragile**; see §8 item D3 |
| `total_devices` in the FLOPs block | — | `jax.device_count() if jax.process_count()>1 else args.num_devices` at `lob/train.py:324` | **Live** |
| `MAMBA3_CONTRACTION_PRECISION` | never set by batch or wrapper | env default `"highest"` at `/lus/lfs1aip2/projects/public/u6gb/sigma-0/src/s5/mamba3_jax.py:46`, consumed at lines 173/179/193/204/349 | **Live, = highest.** This is the value that decides which peak MFU should divide by, and nothing in the launch path mentions it |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | `os.environ.setdefault(...,"0.9")` at `runtime/train.py:34` | `export XLA_PYTHON_CLIENT_MEM_FRACTION=...` at `node_wrapper.sh:114-123` (0.80 at >=8 nodes hierarchical, else 0.90) | **Wrapper wins** (`setdefault` cannot override an exported value). 0.90 on a 1-node probe -> `bytes_limit` 85.50 GB, confirmed in the log |
| `val_split` | `--val_split="$VAL_SPLIT"` at `node_wrapper.sh:591`, `VAL_SPLIT=0.0` at `train_full_autoreg.batch:330` | **`val_split=0.0` hardcoded at `/lus/lfs1aip2/projects/public/u6gb/sigma-0/src/lob/train.py:121`** | **Dead knob**, pinned twice. There is no in-run validation on this path at all; see §8 item D2 |

---

## 3. What is already measured, at zero GPU cost

`[FLOPs] Tokens/step:` and `[Timing] ... recent N s/step, MFU X%` are printed by every run.
The lines live in per-node logs under
`/lus/lfs1aip2/projects/public/u6gb/sigma-0/logs_lobs5/`.
Extracting them requires no allocation. (Extraction note: those logs carry tqdm carriage
returns, so use `grep -o "\[Timing\][^[:cntrl:]]*"`, never a plain `grep`, which prints the
whole progress bar.)

### 3.1 Throughput, mamba3, 26tok, msg_seq_len=500, micro_bsz=4, K=1

`recent s/step` is seconds per **optimizer** step (`_recent_step_time` at
`/lus/lfs1aip2/projects/public/u6gb/sigma-0/src/lob/train_helpers.py:1194` divides by the
delta in `len(batch_losses)`, and `batch_losses` is appended once per optimizer step).
First 5 samples dropped from each series to exclude compilation.

| Job | N_params | d_model | GPUs | tokens/opt-step | n | s/step | sd | cv | tok/s | tok/s/GPU | GPU-h per 1e9 tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5705912 | 78,539,423 | 1024 | 1 | 52,000 | 688 | **0.5687** | 0.0044 | 0.77% | 91,437 | 91,437 | **3.038** |
| 5705913 | 78,539,423 | 1024 | 4 | 208,000 | 679 | 0.5762 | 0.0053 | 0.92% | 360,990 | 90,248 | 3.078 |
| 5705914 | 78,539,423 | 1024 | 8 | 416,000 | 364 | 0.6158 | 0.0104 | 1.69% | 675,499 | 84,437 | 3.290 |
| 5749206 | 78,539,423 | 1024 | 8 | 416,000 | 268 | 0.6217 | 0.0104 | 1.67% | 669,168 | 83,646 | 3.321 |
| 5877859 | 33,610,439 | 640 | 16 | 832,000 | 315 | 0.4508 | 0.0091 | 2.02% | 1,845,595 | 115,350 | 2.408 |

Log paths, in full:
```
/lus/lfs1aip2/projects/public/u6gb/sigma-0/logs_lobs5/training_5705912_node0.log
/lus/lfs1aip2/projects/public/u6gb/sigma-0/logs_lobs5/training_5705913_node0.log
/lus/lfs1aip2/projects/public/u6gb/sigma-0/logs_lobs5/training_5705914_node0.log
/lus/lfs1aip2/projects/public/u6gb/sigma-0/logs_lobs5/training_5749206_node0.log
/lus/lfs1aip2/projects/public/u6gb/sigma-0/logs_lobs5/training_5877859_node0.log
```

Readings:

1. **The 0.565 s/step in `results/INVENTORY.md` §4 is reproduced**: 0.5687 +/- 0.0044 over 688
   samples on job 5705912. The inventory attributes it to wandb `wqgghoyj`, which appears in
   none of these logs; job 5705912 is wandb `b30675li` in `oxford-lob/sigma0-selftrain`.
   The two agree numerically but their provenance is not the same, so the inventory line
   should be re-sourced to a run whose config is visible, not left pointing at `wqgghoyj`.
2. **Step time is remarkably stable** — cv 0.77% at 1 GPU, rising to 2.02% at 16.
   A profiling run therefore needs ~100 post-warmup samples, not thousands, for a 0.1%
   standard error on the mean. Do not report a step time from fewer than ~50 samples: the
   first few carry compilation and the early ones in these logs run 1.69 s (job 5877859,
   step 100) against a 0.45 s steady state.
3. **Weak-scaling loss is 8% out to 8 GPUs** (91,437 -> 84,437 tok/s/GPU) and the last
   column shows the whole practical range of the conversion constant: **3.04 to 3.32
   GPU-hours per billion tokens** at 78.5M.
4. Job 5877859 is a *different model* (33.6M, d_model 640, correction 3.090). Its higher
   per-GPU token rate is a size effect, not a scaling result; it is in the table only so
   that no one later reads the 16-GPU row as the 78.5M number.

### 3.2 What INVENTORY open item 3 should become

`results/INVENTORY.md` §4 currently says "Tokens per step were not recorded alongside these".
It is recorded, by `print_flops_summary` at
`/lus/lfs1aip2/projects/public/u6gb/sigma-0/src/s5/flops.py:200-209`, in every run's log and
in `wandb.config` (`tokens_per_step`, `num_params`, `flops_per_step`, `flops_correction`,
written at `/lus/lfs1aip2/projects/public/u6gb/sigma-0/src/lob/train.py:388-393`).
The item is closed by reading, not by measuring — with one caveat that must be carried
forward: **the field named `tokens_per_step` is tokens per MICRO step; it does not include
`grad_accum_steps`.** It is correct only because K=1 on this line.

---

## 4. What must be profiled, cheapest first

Each phase states what it buys and what decision it unblocks. Phases P1-P6 all fit inside a
single attached step on one GH200 card (§6). Costs are wall-clock on that one card.

| Phase | Cost | Measures | Unblocks |
|---|---|---|---|
| **P0** | 0 GPU, ~10 CPU-min | Recover tokens/step, N_params, correction, s/step, MFU from existing logs (done, §3) | Every budget line in §7 |
| **P1** | ~6 min | Tokens/step **assertion** at the probe setting; compile time; s/step over 200 post-warmup steps; per-step distribution incl. p99 | The conversion constant; the warmup length every later phase must skip |
| **P2** | ~6 min | Peak device bytes per arm; the micro_bsz ladder to the OOM edge | Arms per card; whether 5 seeds fit on one node |
| **P3** | ~5 min | Input-pipeline share: same step with a repeated batch vs the real SquashFS loader, at N_DATA_WORKERS in {4, 12, 24} | Whether a 1-GPU probe is compute-bound; whether 4 arms on a node will fight over 288 cores |
| **P4** | ~8 min | Marginal cost of each probe in `plasticity_probes.py`, in units of one train step | The probe cadence in §6; the overhead line in the §7 budgets |
| **P5** | ~3 min | `MAMBA3_CONTRACTION_PRECISION` in {highest, default} x step time | Which peak MFU divides by; and a free speed knob if the delta is large |
| **P6** | ~5 min | Forward-only (eval) step time, so validation passes can be budgeted | PLAN §2.4's second coordinate, which currently has no code path (§8 D2) |

Total ~33 minutes on one card. Order is strict: P1 fixes the warmup constant that P2-P6 use,
and P2 fixes the micro_bsz that P3-P6 run at.

### 4.1 MFU: state the divisor, always

`compute_mfu` at `/lus/lfs1aip2/projects/public/u6gb/sigma-0/src/s5/flops.py:171` defaults
`peak_flops_per_gpu = GH200_PEAK_BF16_FLOPS = 989e12`, the **dense BF16 tensor-core peak**.
The model does not run bf16: `DTYPE` is never set on this path, the parameters are fp32, the
Mamba3 SSD contractions request `jax.lax.Precision.HIGHEST` (full-f32 accumulation), and the
`nn.Dense` projections take XLA's default precision, which on Hopper is TF32.

So there is no single correct denominator, and the honest report is a triple:

| Denominator | GH200 dense peak per GPU | Job 5705912 reads | Comment |
|---|---:|---:|---|
| BF16 tensor core | 989 TFLOPS | **11.45%** | What the code prints today. A lower bound, since no bf16 is executed |
| TF32 tensor core | 494.5 TFLOPS | **22.9%** | The right divisor for the projection matmuls |
| FP32 CUDA cores | 67 TFLOPS | 169% | **Above 100%, which falsifies this peak as the binding one** — the work is not on the FP32 non-tensor path |

The FP32 row is not a joke entry: it is the cheapest available falsification test, and it is
what makes the claim "the model executes on tensor cores despite `Precision.HIGHEST`" evidence
rather than assertion. P5 turns this from an argument into a measurement: if switching
`MAMBA3_CONTRACTION_PRECISION=default` (TF32 in the contractions) moves step time materially,
the contractions are a real share of the step and the mixed denominator is unavoidable; if it
moves it by under a percent, the contractions are not the bottleneck and TF32 is the divisor
to standardise on.

A separate caution on the numerator: `achieved_tflops` already contains `correction = 2.630`,
which was calibrated against NVIDIA GPM tensor-pipe counters at BSZ=1 rather than derived. It
is a measured hardware-FLOP factor, not an analytical one. Any MFU quoted from this codebase
inherits that calibration and must say so.

### 4.2 Peak memory and arms per card

The only memory number in the logs is `bytes_in_use` from `print_memory_usage`
(`/lus/lfs1aip2/projects/public/u6gb/sigma-0/src/lob/train_helpers.py:875`), sampled at the
**top** of a step, before the forward pass:

```
Device cuda:0 Used: 681.56 MB / 85.50 GB      (job 5705912, 1 GPU, MEM_FRACTION 0.90)
CPU Memory: 23.94 GB
```

681 MB is parameters plus optimizer state; it says nothing about the peak inside a step, which
is what decides how many arms fit. `device.memory_stats()` exposes `peak_bytes_in_use`; it is
simply never read. P2 reads it.

Arms per card, once P2 lands:

```
n_arms = floor( (MEM_FRACTION * 95.6 GB - reserve) / peak_bytes_per_arm )
each arm must then set XLA_PYTHON_CLIENT_MEM_FRACTION ~ MEM_FRACTION / n_arms
```

Two caveats to attach to that number, both of which have bitten this account before:
co-resident arms contend for SMs, so `n_arms` arms do **not** finish in the time of one; and
`XLA_PYTHON_CLIENT_PREALLOCATE=true` (set at `node_wrapper.sh:106`) means the first arm grabs
its fraction and a second arm sees a card that looks full. **Default recommendation until P2
says otherwise: one training arm per card**, and use extra cards, not extra arms, for seeds.

### 4.3 Marginal cost of each plasticity probe

Definitions, with `t_step` = one optimizer step (fwd + bwd + update), `t_fwd ~ t_step/3`
(the repo's own measurement is backward = 2.00x forward, PR#32).

| Probe (`/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/code/plasticity_probes.py`) | Extra work | Cost per invocation | Overhead if run every N steps |
|---|---|---|---|
| `global_l2_norm` (weights) | tree reduction | < 1e-3 t_step | negligible |
| `global_l2_norm` (grads) | already computed under `LOG_GRAD_NORMS=1` | 0 | 0 |
| `dormant_fraction` | 1 forward with activation capture on S sampled positions | ~ t_step/3 | (1/3)/N |
| `effective_rank` | one d x d Gram per layer on the same captured activations | < 0.01 t_step if done on device | negligible |
| `OptimizationReadiness`, K micro-gradients, **K_accum = 1** | K extra forward+backward | K * t_step | K/N |
| `OptimizationReadiness`, **K_accum >= K** (grad accumulation on) | 2 scalar reductions per micro-batch; the mean gradient is already the accumulator | ~ 2e-4 t_step | negligible |
| `top_hessian_eigenvalue`, n_iters HVPs | forward-over-reverse, 1 HVP ~ 2 gradient evaluations | 2 * n_iters * t_step | 2*n_iters/N |

Worked numbers at the CL probe setting (t_step = 0.5687 s):

```
readiness, K=8, K_accum=1, every N=100 opt steps  ->  8/100  =  8.0% overhead
readiness, K=8, K_accum=1, every N=1000           ->  0.8%
readiness, K=8, under grad accumulation K_accum>=8 ->  ~0%      (the gradients already exist)
hessian, n_iters=20, every N=1000                 ->  40/1000 =  4.0%   (~23 s per invocation)
hessian, n_iters=20, every N=5000                 ->  0.8%
dormant + er2, every N=1000                       ->  ~0.03%
```

**Design consequence, and it is the cheapest finding in this draft:** optimization readiness
is nearly free if and only if the run uses gradient accumulation with `K_accum >= K`, because
the per-micro-batch gradients it needs are exactly the ones the accumulator is already
producing; all that is missing is `sum_i ||g_i||^2`, one extra reduction per micro-batch, and
`||gbar||^2`, one reduction per optimizer step. Today `GRAD_ACCUM_STEPS=1` on this line, so
readiness costs a full K extra backward passes. Either (a) run the plasticity probes at
`K_accum = K_readiness` and take readiness free, or (b) accept K/N and pick N.
Option (a) changes the effective batch size, which is part of the experiment definition, so it
must be chosen at design time and declared, not switched mid-ladder.

Two correctness notes on the probe module itself, both cost-relevant:

- `top_hessian_eigenvalue` takes and returns **flat numpy float64 vectors of length `dim`**.
  At `dim = 78,539,423` that is a 628 MB allocation per iteration plus a host round trip, and
  `np.asarray(hvp(v), dtype=np.float64)` at line 211 forces a float64 copy of the returned
  device array. Twenty iterations is ~6 s of pure host-side numpy on top of the ~23 s of HVPs.
  For a whole-model Hessian this should be a device-side power iteration; the numpy interface
  is right for a per-block probe (one layer at a time), which is arguably the more informative
  measurement anyway.
- The batch Hessian is a stochastic estimate. A single batch gives one draw, and this account
  has a documented history of claims made at n=1 and n=2 that reversed. **Budget >= 3 batches
  per checkpoint for the sharpness probe and report the spread**, or do not report sharpness.

---

## 5. The formulas the profiling run has to fill in

```
T_gpu(setting)    = tokens_per_opt_step / (s_per_opt_step * n_GPU)      [tokens per GPU-second]
GPU_hours(D)      = D / (3600 * T_gpu)
wallclock(D, G)   = D / (3600 * T_gpu * G)                              [hours on G GPUs]
overhead_probes   = (K_readiness/N_read) + (2*n_iters/N_hess) + (1/3)/N_dorm
GPU_hours_total   = GPU_hours(D) * (1 + overhead_probes) + n_ckpt * t_ckpt/3600
```

Measured today: `T_gpu = 91,437` tokens/GPU-second at 78.5M params, micro_bsz 4, 1 GPU,
`Precision.HIGHEST` contractions, giving `GPU_hours(1e9) = 3.038`.

---

## 6. Profiling run specification — one attached step, one GH200 card

### 6.1 Placement rules

- Attach only; do not queue. Pick the card with `gtop`'s per-card lines (`mem 0.0/95.6G` and
  `idle`; 1-9 MiB is genuinely free), never the header idle count, which counts held cards.
- Request `--gres=gpu:4 --cpu-bind=none` and select the free card inside the process with
  `CUDA_VISIBLE_DEVICES`; a `--gres=gpu:1` request is handed logical device 0, which is the
  card most likely to be occupied.
- Set `XLA_PYTHON_CLIENT_MEM_FRACTION` to leave room for the neighbour, e.g. 0.5, and record
  the value in the manifest, because it changes `bytes_limit` and therefore P2's ladder.
- Give the step a real `--job-name` (not the default `bash`), so the node-budget monitor does
  not read it as idle.
- The step must be robust to the session ending. On this cluster `setsid nohup srun ...` has
  been observed not to survive a real disconnect; a `tmux` session on the login node does.
  `/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/code/attach_adaptation.sh:78`
  currently uses `setsid nohup` and should be moved to tmux (§8 R4).

### 6.2 Environment, single-card

Derived from
`/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/code/attach_adaptation.sh`,
which already fakes the SLURM environment and rewrites the batch file's internal `srun`. The
profiling run reuses that mechanism unchanged. The deltas are:

```
SLURM_NNODES=1            # keeps is_distributed False so --num_devices stays live
GPUS_PER_NODE=1
PER_GPU_BSZ=4             # P1; P2 sweeps this
GRAD_ACCUM_STEPS=1        # P4 also runs a K_accum=8 cell
MODEL_PRESET=75m
ARCHITECTURE=mamba3
MSG_SEQ_LEN=500
CURTAIL_EPOCHS=<per phase, 60..260>
NO_AUTO_RESUME=1          # a profiling run must never resubmit itself
NO_VALIDATION=1
USE_WANDB=True            # WANDB_DIR must point at node-local scratch
SQUASHFS_MULTI_MODE=1, SQUASHFS_MONTHS=2024-08, FORBID_RAW_NPYZST=1
SQUASHFS_MULTI_MOUNT_ROOT=/tmp/kangli.u6gb/sigma0/profile_<RUNTAG>/sp500_squashfs
CHECKPOINT_EVERY=999999999   # no checkpoint writes during timing
MAMBA3_CONTRACTION_PRECISION=highest   # P5 flips this to default
```

Each phase is a fresh Python process, because XLA compilation caching and allocator state
otherwise leak between cells. Six processes, ~33 minutes.

### 6.3 What it writes, and where — the allocation-expiry rule

**Node-local storage disappears when the Slurm allocation ends, and on this cluster an expired
allocation cannot even be `srun` into to fetch results.** The rule for this run is therefore
the opposite of the usual "write to $TMPDIR, rsync at the end":

```
/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/results/profile_<UTC>/
    manifest.json          written BEFORE the first measurement
    p1_steptime.json       copied to Lustre the moment P1 ends
    p2_memory.json
    p3_dataloader.json
    p4_probes.json
    p5_precision.json
    p6_evalstep.json
    profile.log            the raw stdout of all six phases
```

where `<UTC>` is `$(date -u +%Y%m%dT%H%M%SZ)`, unique per run so nothing ever needs clearing
and no earlier profile is overwritten. Six small JSON writes over 33 minutes is negligible
Lustre metadata load; a single end-of-run rsync is the thing that gets lost.

`manifest.json` is written first, before any measurement, and records: git HEAD of
`/lus/lfs1aip2/projects/public/u6gb/sigma-0`, the full `argv` and the full resolved
environment of the run, node id, GPU UUID, `bytes_limit`, JAX/jaxlib versions, the checkpoint
path and step if one is restored, and the six phase specifications. The lesson it encodes:
a product whose invocation was not recorded cannot be extended later, and the absence never
raises at write time — only when someone tries to add a member.

### 6.4 Phase specifications

**P1 — step time and the tokens/step assertion.** 260 optimizer steps, discard the first 60.
Assert `tokens_per_step == PER_GPU_BSZ * num_devices * 13000` against the value the code
computed at `lob/train.py:369`, and fail loudly on mismatch rather than recording both.
Record: compile time (t of step 1 minus median), mean, sd, cv, p50, p95, p99, min, max, n.
Also record `jax.local_device_count()` and `args.num_devices` separately, so the override at
`runtime/train.py:361` is visible in the artifact rather than inferred.

**P2 — memory.** For `PER_GPU_BSZ` in {1, 2, 4, 6, 8, 10, 12}: 20 steps, then read
`jax.local_devices()[0].memory_stats()` and record `bytes_in_use`, `peak_bytes_in_use`,
`bytes_limit`, `largest_alloc_size`; on OOM record the failure and continue with the next
value rather than aborting. Run this phase with `XLA_PYTHON_CLIENT_PREALLOCATE=false` so the
allocator reports true peak, and cross-check the largest cell against
`nvidia-smi --query-gpu=memory.used` sampled during a step. Report `peak_bytes` versus
`PER_GPU_BSZ` as a fitted line; the intercept is the parameter-plus-optimizer footprint and
the slope is the per-sample activation cost, which is what generalises to other batch sizes.

**P3 — input pipeline.** Three cells at `N_DATA_WORKERS` in {4, 12, 24}, plus one cell with
`--debug_overfit` (a repeated in-memory batch, no SquashFS reads). The gap between the
repeated-batch cell and the best real-loader cell is the input-pipeline cost. If that gap is
above ~5%, four co-resident arms on one node will contend for the 288 cores and the arms-per-
node answer changes; if it is under 1%, the loader is irrelevant and P3 need never be rerun.

**P4 — probe marginal cost.** Six cells, each 40 steps: baseline; readiness at K in {4, 8, 16}
with `K_accum=1`; readiness at K=8 with `K_accum=8` (the free variant); one Hessian probe with
`n_iters` in {5, 10, 20} timed separately; activation capture plus `dormant_fraction` plus
`effective_rank` on the 6 fused layers at S in {512, 2048} sampled positions. Report every
cost as a **multiple of the baseline step**, not in seconds, so the table survives a change of
card, model size or batch size. Record the Hessian eigenvalue on three different batches, so
the noise floor of the sharpness probe is measured at the same time as its cost.

**P5 — precision.** Two cells, `MAMBA3_CONTRACTION_PRECISION` in {highest, default}, 100 steps
each. Report the step-time ratio and the implied share of the step spent in the SSD
contractions. This is also the cell that decides whether TF32 is a free speed knob or a
numerical change with no payoff. It must not be adopted on the strength of the timing alone:
if the delta is worth taking, a loss-curve comparison over a few thousand steps is a separate
decision, and the two cells here differ in numerics as well as speed.

**P6 — eval step.** 100 forward-only steps at the same batch, to get `t_eval/t_train`. PLAN
§2.4 requires an old-window NLL alongside every plasticity readout, and §8 D2 shows that path
does not currently exist, so its cost has never been in any budget.

### 6.5 Pre-registered expectations

Written down before the run so that a surprise is visible as a surprise:

| Quantity | Expected | Would be surprising |
|---|---|---|
| tokens/step at PER_GPU_BSZ=4, 1 GPU | exactly 52,000 | anything else at all |
| s/step, 260 steps | 0.569 +/- 0.01 | outside 0.55-0.60 |
| cv of s/step | < 1.5% | > 3% (means a noisy neighbour on the card) |
| peak_bytes at bsz 4 | 10-25 GB | < 3 GB (capture is wrong) or > 60 GB |
| readiness K=8, K_accum=1 | 8.0 +/- 0.5 x t_step | < 6x (the extra passes are not happening) |
| 1 HVP | 1.8 - 2.2 x t_step | < 1.2x (forward-over-reverse is not being used) |
| MAMBA3_CONTRACTION_PRECISION=default | 0 to -25% step time | a speed-up above 40% |

---

## 7. Standing instrumentation — measure this once, never again

Target: every future run in this line logs the cost model and the plasticity probes by
default, so no later question needs a dedicated profiling job.

### 7.1 Fields, and what is wrong with the ones that exist

| Field | Today | Change |
|---|---|---|
| `tokens_per_step` (wandb config) | `micro_bsz * total_devices * seq_len`, **excludes K** | rename to `tokens_per_micro_step`; add `tokens_per_opt_step = ... * grad_accum_steps` |
| `throughput/step_time_s` | seconds per **optimizer** step (correct) | add `throughput/micro_step_time_s`; the `[Timing]` print already uses two different denominators under two labels |
| `throughput/tokens_per_sec` | **wrong by a factor of `correction * N_params`** — see §8 D1 | fix to `tokens_per_opt_step / step_time_s` |
| `throughput/mfu_pct` | divides by 989e12 bf16 with no label | emit `mfu_bf16_dense_pct`, `mfu_tf32_dense_pct`, `achieved_tflops`, and a config field `peak_flops_basis` naming the peak and the dtype |
| device memory | `bytes_in_use` printed every 1000 batches | log `peak_bytes_in_use`, `bytes_in_use`, `bytes_limit` at the wandb cadence |
| grad norms | behind `LOG_GRAD_NORMS=1` | on by default for this line; it is two reductions |
| plasticity probes | not wired at all | see §7.2 |

### 7.2 Probe cadence

| Probe | Cadence | Justification |
|---|---|---|
| weight L2, non-embedding | every wandb log (1/min) | free |
| grad L2 by group + clip ratio | every wandb log | free, already computed |
| dormant fraction, effective rank | every 1000 optimizer steps | ~0.03% |
| optimization readiness, K=8 | every 1000, or free if K_accum>=8 | 0.8%, or 0% |
| top Hessian eigenvalue, n_iters=20, 3 batches | every 5000 optimizer steps | 2.4% |

Aggregate standing overhead at the CL-probe setting: **~3.2%** with K_accum=1, **~2.5%** with
accumulation on. That is the number that goes into the `(1 + overhead_probes)` factor in §5.

### 7.3 Where it is written

- **wandb**, project `oxford-lob/sigma0-continual`, with `WANDB_DIR` on node-local scratch and
  `WANDB_MODE=online` (repo rule; wandb's own files must not sit on Lustre).
- **A Lustre JSONL**, appended at the wandb cadence, one line per log point, at
  `/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/results/runs/<jobid>_<wandb_id>/metrics.jsonl`
  with a sibling `run_manifest.json` written at step 0 carrying the full resolved
  configuration and `argv`. One append per minute is negligible metadata load and it is the
  copy that survives both wandb outages and allocation expiry.
- Checkpoints and tensorboard stay on node-local storage and are rsynced at the end, unchanged
  from the repo rule. Only the small metrics stream is written through to Lustre live.

---

## 8. Defects found while tracing the chain

Ordered by how much they distort a number someone would quote.

**D1 — `throughput/tokens_per_sec` is not tokens per second.**
`/lus/lfs1aip2/projects/public/u6gb/sigma-0/src/lob/train_helpers.py:1249`:

```python
_tokens_per_sec = _effective_flops / _recent_step_time / 6   # rough: flops/(6*time) ~ N*tokens/time
```

`_effective_flops = correction * 6 * N_params * tokens_per_step * K`, so dividing by 6 leaves
`correction * N_params * tokens_per_step * K / t` — the correct expression divides by
`6 * correction * N_params`. The comment even states the residual (`~ N*tokens/time`) and the
name says otherwise. At 78.5M params and correction 2.630 the logged value is about
2.07e8 times the real token rate. Anyone reading that panel gets a number in the 1e13 range
and a metric name that says tokens per second.

**D2 — there is no in-run validation on this path, and PLAN Step 2 requires one.**
`VAL_SPLIT=0.0` at `train_full_autoreg.batch:330` (not even `${VAL_SPLIT:-0.0}`), and
independently `val_split=0.0` is hardcoded at
`/lus/lfs1aip2/projects/public/u6gb/sigma-0/src/lob/train.py:121`, so the `--val_split` flag
threaded through `node_wrapper.sh:591` never reaches the dataset. `train_only_validate_every_n_steps = 0`
at `lob/train.py:~422` completes the picture. This is deliberate for scaling-law production
runs (held-out loss is a separate job), but PLAN §2.4 requires **both** coordinates —
old-window NLL and probe AUC — at every adaptation stage, and §3 Step 2 says "log validation
NLL every fixed interval". **That code path does not exist**, so its cost has never appeared
in any budget. P6 measures the forward-step cost; a plan owner has to decide whether the
validation runs inside the training loop or as a separate scoring job over checkpoints.

**D3 — the FLOPs correction is gated on a field assigned by a side effect elsewhere.**
`/lus/lfs1aip2/projects/public/u6gb/sigma-0/src/lob/train.py:373` reads
`getattr(args, 'ssm_type', 'gdn') == 'mamba3'`, but `--ssm_type` is a deprecated alias whose
argparse default is `None`. The value only becomes `'mamba3'` because `stamp_architecture`
(`/lus/lfs1aip2/projects/public/u6gb/sigma-0/src/s5/registry.py:360`) assigns it inside
`init_train_state`, called forty lines earlier at `lob/train.py:145`. Two ways this silently
flips to `correction = 1.0`, understating FLOPs and MFU by 2.63x with no error:
`args.debug_loading = True` skips `init_train_state` entirely, and any refactor that moves the
FLOPs block above line 145 does the same. Fix: gate on
`architecture_from_args(args) == 'mamba3'`, which reads the canonical selector directly.
This is the same shape as the defect recorded for the hybrid line, where one arm logged
correction 1.0 and the other 3.09 and the entire MFU gap between them was bookkeeping.

**R1 — the throughput reference in `results/INVENTORY.md` §4 points at a run nobody can open.**
`wqgghoyj` appears in no log under
`/lus/lfs1aip2/projects/public/u6gb/sigma-0/logs_lobs5/`. Re-source that row to job 5705912 /
wandb `b30675li`, and add the columns that make it usable: GPUs, micro_bsz, tokens/step,
n samples, sd. A bare `0.565 s/step` cannot be converted into anything.

**R2 — `results/a1_step*.json` records the symlink shorthand.** The `root` field reads
`/projects/public/u6gb/sigma-0/checkpoints_selftrain/...`; it must be
`/lus/lfs1aip2/projects/public/u6gb/sigma-0/checkpoints_selftrain/...`. Same for
`/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/results/a1_probe.log`. Emit
`os.path.realpath` at write time so this cannot recur.

**R3 — `GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}` at `train_full_autoreg.batch:368` is the
exact pattern the repo's own rule bans**: a value that must change with node count, written as
a fixed default with the real value passed from elsewhere. It is currently benign because K=1
everywhere on this line, but the plasticity plan may turn accumulation on to get readiness for
free (§4.3), at which point the batch script must declare `EFFECTIVE_BSZ` and derive K, with a
hard failure if a caller passes an inconsistent K, and print the resolved
`[bsz] effective = micro x GPUs x nodes x K` line.

**R4 — `attach_adaptation.sh:78` launches with `setsid nohup`.** That has been observed on this
cluster not to survive a real disconnect, because the `srun` client lives on the login node and
is the step's only control channel. Move to a tmux session; note also that the tmux server is
node-local, so a login-node change loses it and every session must re-verify.

**R5 — rename candidates** (rename, never delete):
`results/failure_pool_reliability_total_superseded_20260904T212643Z.json` is already correctly
named; the `__pycache__` directory under
`/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/code/` should be left alone
rather than cleaned, since clearing it is exactly the "tidying my own scratch" reflex the
no-`rm` rule targets.

---

## 9. Token budget to GPU-hours

### 9.1 The conversion

```
GPU_hours(D) = D / (3600 * T_gpu)          T_gpu in tokens per GPU-second
```

with the measured anchor at 78.5M params, mamba3, 26tok, msg_seq_len 500, micro_bsz 4:

| Configuration | T_gpu | GPU-hours per 1e9 tokens |
|---|---:|---:|
| 1 GPU (job 5705912) | 91,437 | **3.038** |
| 4 GPU (job 5705913) | 90,248 | 3.078 |
| 8 GPU (job 5705914 / 5749206) | 84,437 / 83,646 | 3.290 / 3.321 |
| 128 GPU, 75m production, micro_bsz 10 | **unmeasured** | `= 128e9 / (3600 * tokens_per_opt_step / s_per_opt_step)` |

Every cell below uses the 1-GPU anchor, since the plasticity probes are 1-GPU runs.
`C = 3.038` GPU-hours per billion tokens; the probe overhead factor is `(1 + p)` with
`p = 0.032` from §7.2.

### 9.2 The table

| PLAN step | What runs | Token budget D | GPU-hours = D/1e9 * C * (1+p) | Value at C=3.038, p=0.032 | Status of inputs |
|---|---|---|---|---|---|
| Step 0 | inventory | 0 | 0 | **0** | done |
| Step 1 | probe module + tests | 0 (CPU) | 0 | **0** | done |
| **Profiling run (§6)** | 6 phases, 1 card | ~1.7e9 equivalent | 33 min / card | **0.55** | specified here |
| Step 2, as currently scripted | 2 groups x 5 seeds x 1500 steps x 52,000 | 7.80e8 | `0.78 * C * (1+p)` | **2.45** | **all inputs measured** |
| Step 2, at PLAN's stated budget (1e9/run) | 2 x 5 x 1e9 | 1.00e10 | `10 * C * (1+p)` | **31.4** | all inputs measured |
| Step 2, at PLAN's upper budget (2e9/run) | 2 x 5 x 2e9 | 2.00e10 | `20 * C * (1+p)` | **62.7** | all inputs measured |
| Step 2 validation passes | n_eval x D_eval | — | `(D_eval/1e9) * C * r_eval` | **symbolic**: `r_eval = t_eval/t_train` from P6 | **no code path exists** (§8 D2) |
| Step 3, CPT pilot | 6 cells x 1e9 | 6.00e9 | `6 * C * (1+p)` | **18.8** | all inputs measured |
| Step 3, upper | 6 cells x 2e9 | 1.20e10 | `12 * C * (1+p)` | **37.6** | all inputs measured |
| Step 4, mitigations | Step 2 rerun x n_variants | `n_v * 1e10` | `n_v * 10 * C * (1+p)` | **31.4 * n_v** | spectral regularisation adds ~14% step time (PLAN §3 Step 4.2), unverified here |
| Step 5, onset law, 3 sizes | `sum_s D_s / T_gpu(s)` | — | `sum_s (D_s/1e9) * C_s * (1+p)` | **symbolic**: `C_s` needs T_gpu at 34M and 300M | 34M anchor available from job 5877859 after correcting for d_model; 300M unmeasured |

Reference scale for the D column: **one corpus epoch is 781.6e9 tokens**, so PLAN's
"1-2B tokens" adaptation budget is 0.13% to 0.26% of a single epoch, and the currently scripted
1500-step probe is 0.010%.

### 9.3 The gap the table exposes

The scripted Step-2 budget (`CURTAIL_EPOCHS=1500` in both
`/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/code/attach_adaptation.sh:53` and
`/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/code/submit_adaptation_pair.sh:46`)
is **78M tokens per run, 13 to 26 times smaller than the 1-2B tokens PLAN §3 Step 2 asks for**.
Both numbers are defensible — 2.45 GPU-hours versus 31.4 — but they are not the same
experiment, and the AUC difference between an early and a late checkpoint is exactly the kind
of quantity that can shrink as the budget grows. The plan owner has to choose the budget
before the first seed is run, and the choice has to be recorded in the manifest, because
"we extended it later" is precisely the situation where the earlier members turn out not to be
exchangeable with the later ones.

### 9.4 How to complete the table with one more number

Only two cells are genuinely symbolic. `r_eval` from P6 fills the validation row. `T_gpu` at
the 128-GPU production setting fills the last row of §9.1 and requires nothing but reading
`[FLOPs] Tokens/step:` and the `[Timing]` line out of one production log — no GPU time at all,
once someone names the job id of a 32-node mainline run. That job id is INVENTORY open item 1,
and it is the same request that unblocks PLAN Step 2's checkpoint pair.

---

## 10. Adversarial checks on this draft

Applied to my own numbers, in the shape of the recorded failure modes.

| Failure mode | Where it could bite this draft | What I did |
|---|---|---|
| An effect that shrinks as n grows | the s/step means | reported n (688, 679, 364, 268, 315), sd and cv for every row; nothing here rests on n < 250 |
| Measuring one slice, writing the whole line's conclusion | "3.04 GPU-hours per billion tokens" | the sentence carries its qualifiers: at 78.5M params, mamba3, 26tok, msg_seq_len 500, micro_bsz 4, K=1, one GPU, `Precision.HIGHEST` contractions. The 8-GPU row shows it moves to 3.32 |
| A selection rule's consequence read as a property | dropping the first 5 timing samples | the drop is stated, and it is 5 of 688; the untrimmed 1-GPU mean is unchanged to three digits because compilation shows up only in the first two samples |
| A verdict read at a quantile with 8 events | p99 of step time in P1 | P1 requires 200 post-warmup samples and the spec says to print n next to every quantile |
| Dividing by a group-dependent constant | MFU, and per-token normalisations | this is exactly D1 and §4.1; the fix is to publish `achieved_tflops`, which has no chosen denominator, alongside every MFU percentage |
| A knob that is set, logged, and never reaches the code | the whole of §2.3 | audited to the last link; three dead or overridden knobs found (`token_mode`, `val_split`, `num_devices` on multi-node) |
| A default that may never have applied | `MSG_SEQ_LEN`, `GRAD_ACCUM_STEPS`, `MEM_FRACTION` | each traced to the assignment that wins, with the file and line |
| A plausible mechanism narrated without reading what happened | "the model runs TF32" | not asserted. §4.1 gives the falsifiable version (MFU > 100% against the FP32 peak rules that peak out) and P5 turns the rest into a measurement |
| Attributing an improvement to the wrong cause | the 16-GPU row's higher tok/s/GPU | flagged in §3.1 note 4 as a model-size effect, since job 5877859 is 33.6M at d_model 640, not the 78.5M model |
| A metric's name is not its semantics | `tokens_per_step`, `tokens_per_sec`, `step_time_s` | all three re-defined explicitly in §7.1; two of them were misnamed |

