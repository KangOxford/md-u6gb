# 04 — The continual-training experiment design

> Facet: the training experiment itself. Every number below is measured, with a `file:line`
> or a command behind it, or is flagged under `## Open questions`. Nothing is estimated
> silently.
>
> Base checkpoint chain under discussion:
> `/lus/lfs1aip2/projects/public/u6gb/sigma-0/checkpoints_selftrain/j5705912_b30675li_5705912`

## 中文速览

- **今天查出来、会改写整份设计的第一个事实**：基模 `j5705912` 停在 **warmup 上升段的 8.586%**
  （第 69,378 步，warmup 终点 808,053 步）。它的学习率**从来没有退火到零，也从来没到过峰值**。
  「检查点退火到零、必须重新升温」这个前提在这里不成立；反过来才是风险——按声明峰值起一段，
  等于把学习率一次抬高 11.6 倍。
- **第二个事实**：它只消费了一个 epoch 的 **0.0859%**（277,512 个窗口 / 3.23 亿）。
  所以「旧分布还对不对」的先验答案是**模型远远没学完**，不是**世界变了**。
  重放也不可能是「重看训练过的样本」——随机抽一个窗口撞上训练前缀的概率就是 0.0859%。
- **第三个事实（把比例读反了会毁掉整个矩阵）**：issue #73 原文写的是
  「70% historical data replay ... 30% failure scenarios」，**70 是重放、30 是失败样本**。
  本文统一用 ρ = 重放占比，于是 #73 floated 的那一格是 **ρ = 0.70**，
  「100% 失败样本」那一格是 **ρ = 0.00**。
- **实验组设计**：把 #73 的二选一写成 ρ ∈ {0.00, 0.35, 0.70} 一条曲线，
  再加三个让结论可解释的参照组：**同样大小的随机池**（挡「只是多训了」）、
  **按 |实际涨跌幅| 匹配的随机池**（挡「失败池其实是高波动池」——原始分数与 |涨跌幅| 相关 0.65）、
  **反向池**（取最容易的十分位；它若同样有效，选择就不含信息）。再加一个**空操作组**。
- **一个没人写下来的尺寸约束**：预算固定时 ρ 越小，池被重复的次数 R 越大——
  「100% 失败样本」同时也是「重复最多」的那一组。解法是让池的大小随 (1−ρ) 缩放，
  使 R 在各组相同，并另跑一版按曝光匹配作为复核。
- **五个必须从运行时行为断言、而不是从配置读出来的量**：有效批量、参数量 78,539,423、
  token 模式（**六处写死，配置里那个值一次都没到达代码**）、实际挂载的月份
  （**缺一个月是静默跳过，而且从不记录**）、实际交付的混合比例。
- **两个汇报坐标**：旧窗保留集的逐 token 交叉熵（稳定性）与池留存率 + 探针 AUC（可塑性），
  永远同时报，且池留存率必须配一个「未训练模型换种子重打分」的参照读数——
  否则回归均值会被当成训练效果。

---

## 0. What issue #73 actually asks

Retrieved verbatim (`gh issue view 73 --repo KangOxford/sigma-0`, state OPEN, author
`anjunming1202`, 0 comments, title *"The continual learning system. keep training on the
cases it failed to learn."*). The two sentences this draft has to answer:

> A failure might mean that market regimes have shifted, or it could mean we haven't learned
> that distribution properly yet, so it's worth retraining several times.

> Usually it's something like **70% historical data replay from past distributions and 30%
> failure scenarios** that the model struggled to learn. I'm not sure whether, during
> continual learning, we should put those failed cases or poorly learned trajectories in as
> 100% of the epoch's data, or keep the mix.

**Convention used throughout this document: `ρ` is the replay fraction.** The mix the issue
floats is therefore **ρ = 0.70**, and "100% of the epoch's data is failures" is **ρ = 0.00**.
Reading "70/30" the other way round would invert the whole arm matrix, so it is pinned here.

The issue also states the two scoring axes ("diverge hugely from the true data" and "whether
the outputs come from the same distribution") and names the underlearned-vs-regime-shift
ambiguity explicitly. §4 is the answer to that sentence.

---

## 1. What is already true (verified today)

| # | Fact | Where verified |
|---|---|---|
| 1.1 | Base run trained at `micro_bsz=4`, `num_devices=1`, `process_count=1`, `grad_accum_steps=1` → **effective batch 4 windows/step** | `.../j5705912_b30675li_5705912/69378/metadata/metadata` |
| 1.2 | `epochs=1`, `curtail_epochs=None`, `warmup_end=0.01`, `cosine_anneal=true`, `lr_min=0`, `jax_seed=42`, `msg_seq_len=500` | same file |
| 1.3 | `steps_per_epoch = 80,805,346`; run stopped at step **69,378** = **0.0859%** of one epoch (277,512 of 323,221,384 windows) | `memory/reference_j5705912_pretraining_forensics.md`, consistent with 323,221,384 ÷ 4 |
| 1.4 | Hence `warmup_end_step = int(80,805,346 × 0.01) = 808,053`; the run **never left the linear warmup ramp**; LR at the last checkpoint = 69,378/808,053 = **8.586% of peak** | `src/lob/init_train.py:498`; `src/lob/train_helpers.py:160-163` (`optax.linear_schedule(init_value=0.0, end_value=base_lr, transition_steps=warmup_end_step)`) |
| 1.5 | Three parameter groups: `ssm` → `optax.adam(ssm_lr)`, `regular` → `optax.adamw(lr, wd)`, `muon` (2-D kernels) → `scale_by_muon(nesterov=True)` → `add_decayed_weights` → `scale_by_learning_rate`; whole thing wrapped in `optax.clip_by_global_norm(MAX_GRAD_NORM=1.0)` | `src/lob/train_helpers.py:635-694` |
| 1.6 | `lr = lr_factor × ssm_lr_base = 1.0 × 5e-4`; `muon_lr = 0.01`; `weight_decay = 0.005` | `src/lob/train.py:65,72`; checkpoint metadata |
| 1.7 | Cosine tail floors at `LR_MIN_FRACTION = 0.05` of base LR whenever `lr_min <= 0` | `src/lob/train_helpers.py:25`; used at `src/lob/train.py:438-439`, `src/lob/init_train.py:500-501,530` |
| 1.8 | `RESTORE_RESET_SCHEDULE=True` sets `state.step → 0`. It **keeps** the Adam/Muon moment buffers and rewrites only the scalar `count` fields | `src/lob/train.py:273-275` → `src/lob/init_train.py:49-60` |
| 1.9 | **`val_split=0.0` is hardcoded at `src/lob/train.py:121` and `test_date_range=None` at `src/lob/train.py:135`.** `args.val_split` and `args.test_date_range` are never read on this path. `run/base_model/train_full_autoreg.batch:328-332` clamps `VAL_SPLIT=0.0` a second time | those lines |
| 1.10 | Consequence of 1.9: `src/lob/lobster_dataloader.py:1497` gives `n_val = 0` per ticker and `:1530` leaves `test_files` empty. **No data is held out by the training job at all.** Held-out evaluation must be a separate job | those lines |
| 1.11 | Parameter count = **78,539,423** | `/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/results/a1_step69378.json` (`n_params_total`); independently printed by `run/base_model/runtime/inference.py` in the rollout logs |
| 1.12 | 17 checkpoints retained on this chain: 275, 22495, 24080, 28830, 30410, 33575, 52590, 55773, 57365, 58949, 60532, 62113, 63695, 65275, 66853, 68435, 69378 | `ls` of the run directory |
| 1.13 | Monthly SquashFS shards exist for **2022-01 .. 2026-02** (50 shards). **2026-01 and 2026-02 are outside the `train_date_range` recorded in the checkpoint** | `ls /lus/lfs1aip2/projects/public/s5e/quant_team/lob_preproc_sp500_squashfs` |
| 1.14 | The checkpoint metadata has **102 config keys**, and **none** is `token_mode`, `architecture`, `squashfs_months`, `seq_len`, `vocab_size` or `n_params` | `json.load(...)['config'].keys()` on step 69378 |
| 1.15 | **A missing month shard is a silent skip**: `run/base_model/node_wrapper.sh:392-394` prints `WARN: shard missing ... skipping` and continues. A *failed mount* of a present shard is a hard `exit 1` (`:400-403`). The list of months actually mounted is **never recorded** — `:408` prints a count and `:416-417` prints only the first and last mount path | those lines |
| 1.16 | **`TOKEN_MODE` is pinned to `26tok` in six places, and the config value never reaches anything**: `src/base_model/training/train.py:78` overwrites it unconditionally right after reading the yaml; `run/base_model/node_wrapper.sh:598` types `--token_mode=26tok` literally; `run/base_model/runtime/train.py:136` has `choices=["26tok"]`; `src/lob/lobster_dataloader.py:20-27` states in a comment that the runtime selector was removed. `node_wrapper.sh:484-494` asserts `Message_Tokenizer.MSG_LEN == 26` — a **class constant**, not the loader's actual output | those lines |
| 1.17 | `COSINE_STEPS` is read once (`src/lob/train.py:431`) and the schedules built from it are used **only** for the wandb `lr`/`ssm_lr` traces (`:736-737`). The optimizer's real schedule is built earlier inside `init_train_state` (`:145` → `src/lob/init_train.py:496-541`), which never reads it | grep over `src`, `run`, `configs` |
| 1.18 | `run/base_model/train_full_autoreg.batch:368` contains `GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}` — the "defaulted K" form the project rule forbids | that line |
| 1.19 | **No replay-mixing code and no teacher-KL / distillation code** anywhere under `src/lob` or `src/base_model` | grep for `replay`, `teacher`, `kl_div`, `distill` |
| 1.20 | `LOBSTER_Dataset.__getitem__` maps a **global integer index** to `(file_idx, seq_idx)` via `self._seqs_cumsum`, so a window-level pool is expressible as a list of integers — valid only against a fixed file table | `src/lob/lobster_dataloader.py:1007-1015,1094-1099,1277` |
| 1.21 | `CHECKPOINT_EVERY=auto` already separates the two frequencies: `AUTO_CKPT_INTERVAL = 900s`, `AUTO_WANDB_INTERVAL = 60s`, first checkpoint at ~5 min | `src/lob/train_helpers.py:968-972` |
| 1.22 | `grad_norms/clip_ratio` is `global_grad_norm / MAX_GRAD_NORM`, **not** the fraction of steps clipped | `src/lob/train.py:575-580` |
| 1.23 | Held-out CE entry point exists: `run/base_model/evaluate_model_zoo_ce.py`, args `--checkpoint-root --data-root --tickers --date-range --batch-size --max-batches --steps --output`, plus `--expected-architecture/--expected-seed`. It refuses a checkpoint whose metadata `token_mode != "26tok"` (`:109-112`) and freezes one batch set across all steps (`:177-183`) | that file |
| 1.24 | An early-vs-late adaptation pair was attempted 2026-08-27 (job 6141106, steps 275 and 69378, `TRAIN_DATE_RANGE=2024-08-01,2024-08-31`, `CURTAIL_EPOCHS=1500`, `RESTORE_RESET_SCHEDULE=True`) and **both members exited 1 within one second** | `/lus/lfs1aip2/projects/public/u6gb/sigma-0/logs_cl_probe/{e275,l69378}/`; launcher `/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/code/submit_adaptation_pair.sh` |
| 1.25 | No `latest_checkpoint.json` breadcrumb exists in the base run directory, contrary to the repo convention | direct `cat`, file not found |
| 1.26 | Rollout generator is `run/base_model/runtime/inference.py`. Relevant args: `--n_cond_msgs` (default 250), `--n_gen_msgs` (250), `--n_sequences` (1024), `--batch_size` (32; the archived runs used 48), `--sample_indices_file`, `--seed` (base sampling seed; **index selection stays fixed at 42 regardless**), `--checkpoint_step` | that file, `:71-104` |
| 1.27 | 250 conditioning + 250 generated messages = **500 messages = exactly one training window** at `msg_seq_len=500`. A scored context maps 1:1 onto a trainable window | 1.26 with 1.2 |
| 1.28 | The archived rollouts record the context selection (`sample_indices_rank0.json`: 500 indices, their sha256, `dataset_length`, the index-file path) but **not** the generation parameters: `selection_seed` is literally `null`, no argv dump, no seed in any file's contents, and the batch size is recoverable only from a log line. How `v5m_eval_idx_<TICKER>.txt` itself was built is recorded nowhere | direct reads under `/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z/data/` |
| 1.29 | The archived rollouts were generated from `ckpt/wm_ft_multi3` at **step 69378** with `partial_restore=True` and 78,539,423 params — i.e. the same step number and parameter count as `j5705912`, but through a differently-named path | the `inference.log` files under the same root |

**Two consequences that reshape everything below.**

**(a) This checkpoint is undertrained, not overtrained.** 0.0859% of one epoch is 277,512
windows ≈ 3.6e9 tokens out of a 4.2e12-token corpus. "Forgetting the old distribution" is an
odd frame when the model never learned most of it. It also means replay cannot be re-showing
*seen* data: a uniformly drawn window has probability 8.586e-4 of having been in the consumed
prefix, and that prefix is not reconstructible (41 files were dropped at training time and
the 213,790-window discrepancy is self-consistent but not identifiable). Replay here is
**distribution replay**, and the write-up must say so rather than assume it.

**(b) The LR never annealed.** The last LR the weights experienced is 8.586% of the declared
peak: `ssm_lr = lr = 4.293e-5`, `muon_lr = 8.586e-4`. §3 is written around that, not around
the textbook annealed-to-zero case.

---

## 2. The arms

### 2.1 Held identical across every arm

| Held constant | Cycle-1 value | Why |
|---|---|---|
| starting weights | `.../j5705912_b30675li_5705912` step 69378 | only chain with a verified wide ladder (1.12) |
| optimizer-step budget `B` | 2,000 | §2.6 |
| effective batch `E` (windows/step) | 8, declared `EFFECTIVE_BSZ=8`, `K` derived | never defaulted (1.18) |
| sequence geometry | `msg_seq_len=500`, 26 tok/msg → 13,000 tok/window → 104,000 tok/step | asserted from the loader (§6) |
| LR policy | one rewarm shape, `m` swept identically (§3) | otherwise arms differ in two things |
| optimizer state handling | fresh moments (§2.5) | |
| data-order seed | same `JAX_SEED` across arms within a seed replicate | makes the contrast **paired** |
| eval sets | frozen manifests `oldwin_v1`, `newwin_v1`, `probe_v1` (§7) | |
| XLA autotuning | `--xla_gpu_autotune_level=0` on every **scoring** run | otherwise ~15% of rank agreement is lost to nondeterminism (two identical-seed regenerations agreed only at 0.81–0.87) |

### 2.2 The arms

`ρ` = replay fraction (§0). Pool fraction is `1 − ρ`.

| Arm | Pool source | ρ | Anchor | What it rules out |
|---|---|---|---|---|
| **A0** no-op | none — uniform draw over 2022-01..2025-12 | 1.00 | – | "any 2,000 steps on an undertrained model help". Without A0 nothing else is interpretable |
| **A1** random pool | uniformly random windows from the same calendar window as the failure pool | 0.70 | – | "the gain is from that month's data, not from failure selection" |
| **A2** move-matched random pool | random windows resampled to match the failure pool's marginal \|realised move\| decile **and** message-rate decile | 0.70 | – | "the failure pool is really a high-volatility pool" — required, because the raw score correlates **0.65** with \|realised move\| and A1 does not control for it |
| **A3** failures only | corrected-score top fraction | 0.00 | – | issue #73's "100% of the epoch's data" |
| **A4** the floated mix | corrected-score top fraction | **0.70** | – | issue #73's 70/30 — **the primary cell** |
| **A5** intermediate | corrected-score top fraction | 0.35 | – | turns two points into a curve; a knob is judged by monotonicity, not by which of two cells won |
| **A6** anti-pool | corrected-score **bottom** fraction | 0.70 | – | the strongest null for the selection step. If A6 ≈ A4 the score carries no trainable information and the premise of #73 is dead. One run |
| **A7** naive-score pool | **raw** (unstratified) score top fraction | 0.70 | – | the naive and corrected pools overlap only **40%** (per-ticker 0.30–0.56). If A7 ≈ A4, the `k=20` scoring budget buys nothing downstream |
| **A8** anchored | corrected-score top fraction | 0.70 | L2-SP `λ‖θ−θ₀‖²`, non-embedding | run only if A3/A4/A5 breach the old-window tolerance τ (§7). Off by default |

Pre-registered primary contrasts, declared before any run:

1. **`A4 − A2`** — does selecting by *model failure* beat selecting *matched-difficulty* data?
   This is what #73 is actually asking once the confounds are removed.
2. **`A3` vs `A5` vs `A4`** — is ρ a knob? Judged on **monotonicity across all three**, not
   on one significant pair.
3. **`A4 − A6`** — does the sign of the selection matter?
4. **`A4 − A7`** — does the stratification correction matter downstream?
5. **everything `− A0`** — the "training more" floor.

### 2.3 What replay is drawn from

| Definition | Status |
|---|---|
| the exact windows the base run consumed | **impossible** — the consumed prefix is not reconstructible (1.3 note). Do not attempt again |
| uniform over the full pre-training range 2022-01-01..2025-12-31 | **default**. This is the distribution the model is meant to retain, and 99.914% of it is unseen anyway |
| uniform over the base era only, 2022-01-01..2024-07-31 | **variant**, used in §5 where the question is specifically about old-era structure |

Replay uses the same `UniformTickerDistributedSampler` ticker balancing as pre-training
(`src/lob/dataloading.py:31-80`), so replay is not accidentally an AAPL-heavy stream.

### 2.4 The stay-close-to-previous-model term

None of the usual forms exists in the codebase (1.19). In cost order:

1. **L2-SP** `λ‖θ − θ₀‖²` over non-embedding parameters — two lines in the loss, one
   parameter-sized buffer, `θ₀` already resident at restore time.
2. **Teacher-KL** to `θ₀`'s next-token distribution — a second forward pass per step plus a
   frozen weight copy on device.
3. **EWC / Fisher-weighted L2** — needs a Fisher pass; not justified before 1 and 2.

**Default λ = 0 in A0–A7.** A nonzero anchor makes every arm's plasticity readout partly a
readout of λ. A8 enables it only after the unanchored arms establish there is something to
protect against, with λ from a 3-point pilot over `λ·‖θ₀‖² ∈ {1e-4, 1e-3, 1e-2}` (normalised
so λ is scale-free), taking the smallest λ that keeps old-window CE within τ.

### 2.5 Optimizer state at the stage boundary

| Option | Mechanism today | Assessment |
|---|---|---|
| (i) keep moments, reset schedule position | `RESTORE_RESET_SCHEDULE=True` (1.8): keeps `mu`/`nu`, zeroes `count` | **Do not use.** Zeroing `count` while keeping `nu` re-enables Adam bias correction on steady-state moments. With `optax.adamw` defaults (`b1=0.9`, `b2=0.999`) the correction decays over ≈1/(1−b2) = **1,000 steps** — half of a 2,000-step stage. Every arm would be measured largely inside a bias-correction transient |
| (ii) keep moments, keep counts | `RESTORE_RESET_SCHEDULE=False` | LR then continues up the original 808,053-step ramp: over 2,000 steps it moves from 8.586% to 8.834% of peak. Effectively a tiny constant LR. A legitimate **control** ("A0-noreset"), not a stage design |
| (iii) **fresh optimizer state, params only** | **does not exist** | **Default for all arms.** After `load_checkpoint`, replace `state.opt_state` with `tx.init(state.params)`. The moments were accumulated on a uniform 488-ticker stream; every arm deliberately changes that stream, so carrying second-moment estimates biases per-parameter step sizes toward the old data **by an amount that differs between arms** — a confound that scales with how different the arm is |

`PARTIAL_RESTORE=True` does not give (iii): it only sets `strict=False` on
`ocp.args.StandardRestore` (`src/lob/init_train.py:179`), relaxing tree matching; the
opt_state is still restored when present. A new flag `RESET_OPT_STATE=1` is required, with
the two-sided assertion of §6.8 proving which branch fired.

### 2.6 The sizing constraint nobody has written down

Let `N_pool` = distinct windows in the pool, `B` = optimizer steps, `E` = effective batch in
windows. The pool repeat count is

```
R(ρ) = (1 − ρ) · B · E / N_pool
```

**With `B·E` and the pool held fixed, `R` is largest exactly where ρ is smallest.** So arm A3
("100% failures") is *also* the arm that repeats its pool the most — by a factor of
`1/(1−0.70) = 3.33` relative to A4. Under naive budget matching, issue #73's question is
**inseparable from a question about repetition**: a "70/30 wins" headline could mean
"repeating a pool 3.3× more is worse".

There is no configuration holding budget, ρ and `R` simultaneously constant with one fixed
pool. Three matched variants exist; each holds two of the three:

| Variant | Holds constant | Varies | Reading if it disagrees with the others |
|---|---|---|---|
| **V-fixed-pool** | `B·E`, pool set | `R` (3.33× across ρ) | difference is repetition |
| **V-scaled-pool** *(primary)* | `B·E`, `R` | pool diversity: arm's pool is a random subset of size `(1−ρ)·N₀` | difference is diversity of distinct failures |
| **V-exposure** *(check)* | `R`, pool diversity, pool-derived windows `(1−ρ)·B·E` | total budget | difference is total compute |

Run **V-scaled-pool** as primary (repetition drives memorisation and is the more dangerous
confound) and **V-exposure** on A3/A4/A5 as the check. If the ρ ordering agrees across both,
the mixing conclusion stands; if it flips, the finding is about repetition or about budget
and must be reported that way.

**Cycle-1 worked point.** `N₀ = 4,000` distinct pool windows; `B = 2,000`; `E = 8` →
`B·E = 16,000` windows = 208M tokens (5.8% of the base run's token count). Then in
V-scaled-pool: A3 pool = 4,000 windows, `R = 4`; A4 pool = 1,200 windows, consumes
`0.30 × 16,000 = 4,800`, `R = 4`; A5 pool = 2,600, consumes 10,400, `R = 4`. `R` is constant
by construction and is asserted at run time (§6.6).

Pool cost at `N₀ = 4,000` and selection fraction `q = 0.25`: `S = 16,000` contexts scored at
`k = 20` = **320,000 rollouts**, which is 8× the 40,000 rollouts (8 tickers × 500 contexts ×
10 seeds) that the existing archive represents. At `q = 0.10` it is 800,000 rollouts, 20×.
**Rollout throughput has never been measured and is the binding constraint on this whole
design** (O-3).

### 2.7 Seeds, pairing, and when a number may be quoted

- **Pairing.** Within a seed replicate all arms share `JAX_SEED`, the eval manifests, and the
  replay index stream. The reported statistic is the **paired** per-replicate difference, and
  the noise floor is estimated on that same paired structure. An unpaired spread across arms
  is the wrong ruler here — on this codebase paired CE spreads measured 0.10% against
  unpaired 0.94–3.74%, a factor of 27.
- **Repeats are part of the design.** Arm A4 runs at `n = 6` from the start so the
  run-to-run sd of both coordinates is measured, not assumed.
- **Quoting rule.** No effect from this matrix enters a title, a bold line, or the word
  "best" until `n` has stopped growing. Report the **effect-versus-n trajectory**, not the
  value at the final `n`. This project produced three effects in one session that shrank or
  changed sign as `n` grew (+1.66%→+1.38% at n=8→13; −0.01%→−1.80% at n=1→6; 0.08pp→0.66pp).
- **Power.** Never compute required `n` from an effect size estimated on the same small
  sample; use the lower bound of the pilot effect's bootstrap CI, and report both.

---

## 3. Rewarm

### 3.1 Where this checkpoint actually sits

The standard worry — a checkpoint whose LR annealed to near zero looking dead without being
dead — is real and is **not this checkpoint's situation**:

```
warmup_end_step = int(80,805,346 × 0.01)  =  808,053 optimizer steps
last step reached                          =   69,378
position on the warmup ramp                =    8.586%
LR the weights last saw  ssm / regular     = 5e-4 × 0.08586 = 4.293e-5
                         muon              = 1e-2 × 0.08586 = 8.586e-4
LR at the early checkpoint (step 275)      = 0.034% of peak = 1.70e-7 / 3.40e-6
```

The failure mode to guard against here is the mirror image: "0.5 × pre-training peak" raises
the LR **5.8×** in one step and "1.0 × peak" raises it 11.6×. Damage from that reads exactly
like loss of plasticity.

### 3.2 The rewarm specification

| Parameter | Value | Note |
|---|---|---|
| **peak anchor** | `m × lr_last`, `lr_last` = 4.293e-5 (Adam groups) / 8.586e-4 (Muon) | anchored to what the weights experienced, not to a peak that was never reached |
| **`m` (swept, not tuned)** | {1, 2, 4} | `m = 1` means "continue at the LR it stopped at" |
| **warmup fraction** | 2% of stage steps. With `CURTAIL_EPOCHS = B·K − 1`, `steps_per_epoch = B`, so set `WARMUP_END = 0.02`; at `B = 2,000` that is **40 steps** | short, because the moments are fresh (§2.5) and there is no stale second-moment scale to rebuild |
| **schedule tail** | cosine to a **non-zero** floor. Leave `lr_min = 0` so the code's `LR_MIN_FRACTION = 0.05` applies (1.7): floor = 5% of the stage peak. Do **not** set `lr_min` explicitly | a zero floor would end each stage in exactly the "looks dead" state that motivates rewarm |
| **cosine period** | must equal the stage budget. Set it via `CURTAIL_EPOCHS`, **never** via `COSINE_STEPS` | `COSINE_STEPS` reaches only the wandb trace (1.17); setting it alone produces a logged LR curve that disagrees with the LR the optimizer used |
| **Adam / Muon moments** | **reset** (`RESET_OPT_STATE=1`, §2.5) | |
| **gradient clipping** | `MAX_GRAD_NORM = 1.0` unchanged; `LOG_GRAD_NORMS=1` | changing clip and LR together makes the result uninterpretable |

`m` is a nuisance axis: the arm contrast is computed at each `m` and reported as a 9 × 3
matrix. The claim is about the arms; the `m` axis exists to prove the claim is not about `m`.

### 3.3 What would show the rewarm choice contaminated a plasticity claim

Any one of these fires ⇒ the claim is withdrawn until re-run.

1. **Ordering instability in `m`.** The sign of `A4 − A2`, or of
   `AUC(θ_late) − AUC(θ_early)`, flips between `m = 1, 2, 4`. A conclusion that depends on
   the rewarm multiplier is a conclusion about the rewarm multiplier.
2. **The no-op arm moves with `m`.** Old-window CE for **A0** rises monotonically in `m`.
   A0 sees the pre-training distribution and nothing else; if it degrades, the damage is the
   learning rate, not the data mix. This is the null control that has to be genuinely null.
3. **The difference lives in the transient.** Recompute the AUC over `[0.1B, B]` instead of
   `[0, B]`. If the contrast disappears it was a startup transient, not a learning-speed
   difference. Report both windows always.
4. **Clipping saturates.** `grad_norms/clip_ratio` (= global grad norm ÷ 1.0, 1.22) exceeds 1
   on more than 20% of steps at the higher `m`. Under saturated clipping the effective step
   size is set by the clipper, and arms with larger gradients are silently given smaller
   steps.
5. **`m = 1` alone settles it.** If `A4 − A2` is already established at `m = 1` — no rewarm at
   all, LR continuing exactly where it stopped — the finding is rewarm-free by construction
   and the `m` sweep becomes a robustness annex rather than a dependency. This is the
   cheapest way to make the claim immune, and it is available precisely because 3.1 holds.

---

## 4. The pool, precisely

### 4.1 Sampling contexts

- **Calendar window.** Cycle 1: **2024-08-01..2024-08-31** (the stress slice fixed by the
  inventory and already targeted by the existing probe launcher). Cycle 2: 2025-04.
  Cycle 3: **2026-01**, which is provably outside the recorded `train_date_range` (1.13).
- **Tickers.** The same 8 as the reliability study (AMD, AMZN, GOOG, INTC, JPM, META, MSFT,
  NFLX) so the measured `k`, overlap and dispersion numbers transfer without recalibration.
  Scaling toward the 488 pre-training tickers is a separate decision (O-10).
- **Within a ticker-day.** Contexts drawn at uniformly spaced offsets from a fixed, recorded
  index file, then **stratified by \|realised move\| decile at the 100-message horizon** so
  every decile is represented. Without stratified drawing, the top fraction is a
  high-\|move\| set by construction and A2 has nothing to match against.
- **Geometry.** `--n_cond_msgs 250 --n_gen_msgs 250` (1.26) makes one scored context exactly
  one 500-message training window (1.27). No re-alignment is needed between scoring and
  training; that alignment must be asserted, not assumed (§6.10).
- **Count.** `S = 16,000` (2,000 per ticker) for the cycle-1 point of §2.6. Revise once
  throughput is measured (O-3).
- **Identity.** A pool member is `(message_file_path, window_start_message_index,
  global_dataset_index)`.

### 4.2 The score

Two distinct "corrections" exist in the code and must not be conflated
(`/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/code/failure_pool_reliability.py`):

- **Small-`k` bias correction inside `scores()` (`:126-148`)**: `bias2_raw = (x̄−y)²` still
  carries `σ²/k`; `bias2 = bias2_raw − spread/k`. The exact partition is
  `total = bias2_raw + spread_pop`.
- **Stratification, `stratify()` (`:181-200`)**: **10 quantile (equal-frequency) bins** of
  `|realised|` via `np.quantile(np.abs(realised), np.linspace(0,1,11))`, rank-normalised to
  [0,1] within each bin, neutral 0.5 for bins with ≤2 members.

**Use the stratified ranking of `total`.** Measured justification: the raw score correlates
0.65 with \|realised move\|, retains 0.46 of its ranking after rollouts are detached from
their contexts, and two independently permuted halves still agree at 0.43; stratification
drops those to 0.03 and 0.10 while keeping true signal at 0.46 against a 0.10 zero line.

- **`k = 20` rollouts per context — and it is extrapolated, not measured.** The split-half
  grid runs only to `k = 5` (10 seeds → two halves of 5); `split_half()` at `:264-301` and
  `rollouts_needed()` at `:238` fit a one-parameter Spearman-Brown curve. Per-ticker
  `k(ρ=0.80) ∈ {13.3, 14.8, 16.6, 18.3, 18.4, 20.1, 22.0, 23.6}`, mean 18.4. This caveat
  belongs in the write-up, not in a footnote.
- **The cross-pairing null is mandatory, not optional.** `pairing_nulls()` (`:203-235`)
  returns four readings — `true`, `shared` (one permutation applied to both halves),
  `independent`, `cross`. A consistently mis-paired score scored **0.49**, higher than the
  correct one at 0.46; only `cross ≈ independent` shows the ranking needs the correct
  pairing. Any new score proposed for this pool passes through the same four readings.
- **Every scoring run pins `--xla_gpu_autotune_level=0`** and records the flag. Without it
  two identical-seed regenerations agree only at rank correlation 0.81–0.87.
- **Dispersion is reported beside the score, never folded into it.** The exact decomposition
  `mean_i (x_i−y)² = (x̄−y)² + Var_i(x)` puts 26–34% of the top decile's score in the
  dispersion term (measured, lower inside the top decile in 8/8 tickers at every horizon but
  one), and the model is under-dispersed (reliability slope 0.46–0.59), so that share is a
  floor. A context can enter the pool for being **too narrow** rather than for being wrong.
  Store both terms per member (`dispersion_share()` at `:343`).
- **Do not use plug-in CRPS as the score.** Its bias is `E|X−X'|/(2K)`, proportional to
  ensemble width, so it rewards narrow sampling — the very failure the decomposition flags.

### 4.3 Selection fraction

| `q` | pool at `S = 16,000` | contrast | note |
|---:|---:|---|---|
| 0.10 | 1,600 | strongest | needs `S = 40,000` to reach `N₀ = 4,000` → 800,000 rollouts |
| **0.25** | **4,000** | **moderate — recommended** | matches the §2.6 worked point at 320,000 rollouts |
| 0.50 | 8,000 | weak | "failure pool" becomes "half the data"; the selection contrast is the thing being measured |

The trade is explicit: expected effect scales with (mean score inside the pool − mean score
overall), which falls monotonically as `q → 1`, while the compute for a given `N₀` falls as
`1/q`. **Report the realised ratio** (pool mean score ÷ overall mean score) for whichever `q`
is used, so the effect size can be read against it rather than against an assumption.

### 4.4 Regenerate or accumulate; staleness

**Regenerate every cycle. Do not accumulate.** A pool is on-policy at collection and
off-policy the instant training starts. An accumulated pool is a mixture over an unknown
number of past policies, and no readout can then attribute an effect to a cycle.

Staleness is measured, not argued, via **pool retention**:

```
ρ_ret = fraction of a frozen random subsample of the pool that still exceeds the
        pre-stage selection threshold when re-scored with the post-stage model
```

with three rules that make the number mean something:

1. **The threshold is frozen** at the pre-stage `q`-quantile. Recomputing it post-stage makes
   the top fraction `q` by construction — measuring the selection rule, not the data.
2. **A regression-to-the-mean floor is measured.** Re-score the *same* subsample with the
   *unchanged* `θ₀` using **fresh seeds**. At reliability 0.80 a large fraction drops below
   the threshold with no training at all. The trainable part is
   `ρ_ret(θ_stage) − ρ_ret(θ₀, fresh seeds)`, and the floor is reported alongside, always.
3. **Subsample size 300 at `k = 20`.** At `ρ_ret ≈ 0.5` the binomial se is 2.9pp, so a 10pp
   paired movement is ~3.5 se. Do not read `ρ_ret` on a subsample of 30 — this project has
   already published a verdict read at a quantile holding 8 events.

**Intra-stage staleness** is bounded: at `B = 2,000`, `E = 8`, the oldest pool member is
16,000 windows of training stale, against the 277,512 windows that produced `θ₀`.

### 4.5 The pool manifest — written before the first member

The archive this design builds on records *which* contexts were scored and **nothing** about
*how* the rollouts were generated (1.28): `selection_seed` is `null`, there is no argv dump,
the seed survives only in a directory name, and the construction of the index file itself is
unrecorded. That is precisely why extending it from `k = 10` to `k = 20` is blocked by
bookkeeping rather than by GPU time. The manifest is written **before the first member**, in
one atomic file:

```
pool_id, created_utc, git_sha, argv (verbatim), hostname, slurm_job_id
checkpoint: path, resolved realpath, step, params_sha256, n_params
generation: script path, n_contexts S, k, seeds (explicit list), seed_stride,
            n_cond_msgs, n_gen_msgs, batch_size, horizons [10,25,50,100,150,200,250],
            XLA_FLAGS (verbatim), index_file path + sha256, index_file_construction (how)
selection:  score = "stratified_rank(total)", n_move_bins = 10, q, threshold_value,
            pool_mean_score, overall_mean_score, ratio
data:       squashfs_months RESOLVED (not requested), ticker list,
            file_table_sha256 = sha256(sorted(message_files) ++ _seqs_per_file)
members:    [ {global_idx, file, window_start, total, bias2, bias2_raw, spread, move_bin} ]
```

`file_table_sha256` is load-bearing: a global window index (1.20) is meaningful only against
a fixed file table, and this project has already had 41 files disappear from one silently.
Anything unavailable is written as `null`, never as `"unknown"`.

---

## 5. Underlearned versus regime shift

Issue #73 names both possibilities in one sentence (§0). They need opposite treatments: if
the old distribution is still correct, forgetting is pure loss and replay should dominate; if
the world moved, part of the old distribution is now wrong and heavy replay fights the
update.

### 5.1 The prior this project actually has

State it honestly before designing a test: the base model consumed **0.0859% of one epoch**
(1.3). On the balance of evidence, the dominant explanation for any given failing context is
that the model never learned that structure. A design that assumes regime shift and finds it
will not be believed.

### 5.2 The gate measurement, which runs before any arm

Cheapest and most decisive, using only the model and the existing scoring code:

**Score `S' = 4,000` contexts from 2022-06 (deep inside the pre-training range) and
`S' = 4,000` from 2024-08, same `k = 20`, same stratified score, same seeds-by-construction.
Compare the two score distributions.**

- **Indistinguishable** ⇒ the pool is dominated by **underlearning**; "regime shift" is not
  supported at this level of training; replay is free; §5.3 becomes a refinement rather than
  a load-bearing step.
- **2024-08 systematically worse** ⇒ there is an era effect, and §5.3 localises it.

Contamination on the 2022-06 side is bounded and reported as a number: at 8.586e-4 per
window (1.3), the expected count of previously-seen windows among 4,000 is **3.4**. Do not
claim zero.

Cost: 8,000 contexts × 20 rollouts = 160,000 rollouts, 4× the existing archive. It runs
first because it can change the whole design.

### 5.3 The split: era-twin matching

For each pool member `c` from the recent window, find `T = 8` **twins** in the base era
(2022-01-01..2023-12-31) matched on covariates available without any rollout:

- same ticker,
- same \|realised move\| decile at the scoring horizon,
- same message-rate decile over the conditioning window,
- same time-of-day bucket (first 30 min / mid-day / last 30 min),
- same quoted-spread tercile.

Score the twins with the same model, same `k`, same stratified score, and define

```
Δ_c = score(c) − mean_over_twins score(twin)
```

- `Δ ≈ 0` → **underlearned**: the model fails on this *kind* of context regardless of era.
  Replay is pure gain; nothing has to be unlearned.
- `Δ ≫ 0` → **regime shift**: the same observable conditions now behave differently. Heavy
  replay actively fights the update.

The threshold is fixed **before** looking at any training outcome: at the value whose
bootstrap 95% CI lies entirely above zero.

**Do the split at stratum level, not per context.** The score needs `k ≈ 20` for reliability
0.80; a *difference* of two independently-estimated scores carries roughly twice the noise
variance, so per-context `Δ` would need `k ≈ 40` on each side. With 20 strata (ticker ×
coarsened move decile) the per-cell mean over ~200 members is stable at `k = 20`. Report `Δ`
per stratum with CIs and assign members by their stratum.

### 5.4 What falsifies the split

1. **The interaction is null.** Run the 2×2, `{underlearned, regime-shift} × {ρ = 0.0,
   ρ = 0.7}`, at fixed budget. The split *claims* an interaction: replay should help or be
   neutral on the underlearned subset and hurt on the shift subset. If the interaction CI
   includes zero at the pre-registered `n`, the split carries no decision-relevant
   information and is dropped. This is the falsifier that matters, because it is stated in
   the coordinate the split is supposed to control.
2. **`Δ` does not survive resampling.** Compute `Δ` on two disjoint halves of the 20 rollouts
   (`k = 10` each). If the split-half rank correlation of stratum-level `Δ` is below 0.5, `Δ`
   is noise and no threshold on it means anything.
3. **A calendar null gives the same `Δ`.** Repeat the twin matching entirely inside the base
   era, 2023-08 against 2022-01..2022-12 twins, at the same calendar distance. If `Δ` is as
   large there, `Δ` measures elapsed time, not a regime change.
4. **`Δ` is smooth in calendar time.** Compute `Δ` monthly over 2023-01..2025-12. A regime
   shift predicts a **step** at 2024-08-05; underlearning predicts no structure; a smooth
   drift supports neither and means the matching covariates are themselves drifting.
5. **`Δ` is a property of the matching rule.** Drop the spread tercile, or swap time-of-day
   for volume decile. If subset membership moves by more than ~20%, `Δ` is about the matcher.

---

## 6. What must be asserted from run-time behaviour, not read from configuration

This project has shipped a knob set in five places, printed correctly, recorded in metadata,
and still never reaching the code; and a default silently overwritten downstream. Two more
instances were found today (1.9, 1.16). The rule below is mechanical: **every quantity that
defines the experiment is derived from what the model or the loader actually did, and the run
dies if that disagrees with the declaration.** All checks run on rank 0 before step 1 except
6.6 and 6.11.

### 6.1 Effective batch size — declared, never defaulted

`run/base_model/train_full_autoreg.batch:368` currently violates this (1.18).

```bash
: "${EFFECTIVE_BSZ:?declare the effective batch in windows; do not default K}"
_denom=$(( PER_GPU_BSZ * GPUS_PER_NODE * SLURM_NNODES ))
[ "$_denom" -gt 0 ] || { echo "FATAL: bad denominator"; exit 5; }
[ $(( EFFECTIVE_BSZ % _denom )) -eq 0 ] || {
  echo "FATAL: EFFECTIVE_BSZ=$EFFECTIVE_BSZ not divisible by $_denom"; exit 5; }
_k=$(( EFFECTIVE_BSZ / _denom ))
if [ -n "${GRAD_ACCUM_STEPS:-}" ] && [ "$GRAD_ACCUM_STEPS" != "$_k" ]; then
  echo "FATAL: K=$GRAD_ACCUM_STEPS passed but $SLURM_NNODES nodes require K=$_k"; exit 5
fi
export GRAD_ACCUM_STEPS=$_k
echo "[bsz] effective=$EFFECTIVE_BSZ = ${PER_GPU_BSZ} x ${GPUS_PER_NODE}gpu x ${SLURM_NNODES}node x K${_k}"
```

and in Python, from the runtime rather than the environment:

```python
observed = args.micro_bsz * jax.local_device_count() * jax.process_count() * grad_accum_steps
declared = int(os.environ["EFFECTIVE_BSZ"])
assert observed == declared, f"effective batch {observed} != declared {declared}"
```

### 6.2 The batch the loader actually yields

```python
x = next(iter(trainloader))[0]
assert x.shape[0] == args.micro_bsz, f"loader gave micro batch {x.shape[0]}, args says {args.micro_bsz}"
```

6.1 multiplies a *declared* `micro_bsz` by device counts; if the loader silently yields a
different per-process batch, 6.1 still passes.

### 6.3 Parameter count

Already enforced at `src/lob/train.py:346-364`, but only when `EXPECTED_PARAMS` is set, and
the value is not persisted in the checkpoint (1.14). Set it always:

```bash
export EXPECTED_PARAMS=78539423     # measured: results/a1_step69378.json
```

and add the restore-side half, which does not exist today:

```python
n_init = sum(x.size for x in jax.tree_util.tree_leaves(params_before_restore))
n_rest = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
assert n_init == n_rest, f"restore changed parameter count {n_init} -> {n_rest}"
```

This catches a `partial_restore` that quietly dropped a subtree.

### 6.4 Token mode — from the data, not from the environment or a class constant

`token_mode` is absent from the checkpoint's 102 config keys (1.14) and is pinned in six
places (1.16). The existing guard at `node_wrapper.sh:484-494` asserts
`Message_Tokenizer.MSG_LEN == 26` — a class constant, which cannot catch a loader that emits
something else. Assert the loader's output instead:

```python
from lob.encoding_26tok import Message_Tokenizer as T26
x = next(iter(trainloader))[0]
tok_per_msg = x.shape[-1] // args.msg_seq_len
assert tok_per_msg == 26, f"loader yields {tok_per_msg} tok/msg; TOKEN_MODE claims 26tok"
assert tok_per_msg == T26.MSG_LEN, "loader disagrees with the tokenizer class it claims to use"
assert int(x.max()) < T26.VOCAB_SIZE, "token id outside the 26tok vocabulary"
manifest["tokens_per_message_observed"] = int(tok_per_msg)
```

Record `tok_per_msg` so the next run compares against a *recorded* value rather than a
declared one.

### 6.5 The data months actually mounted

A missing shard is a silent skip and the mounted list is never recorded (1.15). Derive it
from the file table the dataset built:

```python
import re
from pathlib import Path
months = sorted({m.group(0) for p in dataset.message_files
                 for m in [re.search(r"\d{4}-\d{2}", str(p))] if m})
requested = sorted(os.environ["SQUASHFS_MONTHS"].split(","))
assert months == requested, (
    f"mounted {months} != requested {requested}; "
    f"missing={sorted(set(requested)-set(months))} extra={sorted(set(months)-set(requested))}")
manifest["squashfs_months_resolved"] = months
manifest["file_table_sha256"]        = sha256_of(dataset.message_files, dataset._seqs_per_file)
manifest["train_size_windows"]       = int(train_size)
```

The assertion fails in **both** directions: a silently missing month makes the run a
different experiment; a silently extra month can contaminate an evaluation window.

### 6.6 The mix ratio, and the pool repeat count, actually delivered

The pool sampler emits an exact per-batch quota rather than a Bernoulli draw, so the realised
ratio is deterministic and checkable:

```python
# PoolMixSampler: for effective batch E and replay fraction rho, n_replay per batch
# alternates between floor(rho*E) and ceil(rho*E) so the running ratio hits rho exactly.
rho_obs = n_replay_consumed / (n_replay_consumed + n_pool_consumed)
assert abs(rho_obs - args.replay_ratio) < 1e-3, \
    f"declared rho={args.replay_ratio}, delivered {rho_obs}"
R_obs = n_pool_consumed / n_pool_distinct
assert abs(R_obs - EXPECTED_R) < 1e-6, f"pool repeats {R_obs} != designed {EXPECTED_R}"
manifest["rho_observed"], manifest["pool_repeats_R"] = rho_obs, R_obs
```

`R_obs` is asserted, not merely logged, because §2.6's V-scaled-pool design holds it constant
by construction and a violated `R` silently reintroduces the repetition confound.

### 6.7 The learning rate the optimizer actually used

Not the wandb trace, which is built from a **different** schedule object and is the only
consumer of `COSINE_STEPS` (1.17):

```python
# evaluate the schedule object handed to optax, not the one rebuilt in train.py
assert abs(actual_schedule(0)              - peak / warmup_steps)      < 1e-12
assert abs(actual_schedule(warmup_steps)   - peak)                     < 1e-12
assert abs(actual_schedule(B)              - LR_MIN_FRACTION * peak)   < 1e-9
assert os.environ.get("COSINE_STEPS", "0") == "0", \
    "COSINE_STEPS does not reach the optimizer (src/lob/train.py:431 vs init_train.py:496); " \
    "set the cosine period through CURTAIL_EPOCHS instead"
```

### 6.8 Optimizer state reset — both branches asserted

```python
if os.environ.get("RESET_OPT_STATE") == "1":
    state = state.replace(opt_state=tx.init(state.params))
    mu = state.opt_state[-1].inner_states['ssm'].inner_state[0].mu
    assert all(float(jnp.abs(m).max()) == 0.0 for m in jax.tree_util.tree_leaves(mu)), \
        "first moments non-zero after RESET_OPT_STATE=1"
else:
    mu = state.opt_state[-1].inner_states['ssm'].inner_state[0].mu
    assert any(float(jnp.linalg.norm(m)) > 0 for m in jax.tree_util.tree_leaves(mu)), \
        "RESET_OPT_STATE=0 but moments are zero — the restore did not carry optimizer state"
```

An assertion that can only fire in one configuration does not protect the other.

### 6.9 The restore actually happened

```python
assert restore_active, "RESTORE_PATH set but restore did not run"
d = sum(float(jnp.linalg.norm(a - b)) for a, b in
        zip(jax.tree_util.tree_leaves(state.params),
            jax.tree_util.tree_leaves(fresh_init_params)))
assert d > 0.0, "restored params identical to fresh init — restore silently no-oped"
assert int(step_before_remap) == int(os.environ["RESTORE_STEP"]), \
    "restored step is not the requested step"
```

A failed restore produces a perfectly healthy-looking fresh run. This is the assertion whose
absence is hardest to notice.

### 6.10 Pool, replay and evaluation-set identity

```python
assert pool_manifest["file_table_sha256"] == runtime_file_table_sha256, \
    "pool indices were built against a different file table"
assert pool_manifest["checkpoint"]["params_sha256"] == theta0_params_sha256, \
    "the pool was scored with a different model than the one being trained"
assert set(pool_idx).isdisjoint(eval_idx_oldwin | eval_idx_newwin | eval_idx_probe)
assert set(replay_idx).isdisjoint(eval_idx_oldwin | eval_idx_newwin | eval_idx_probe)
assert sha256_file(eval_manifest_path) == EXPECTED_EVAL_SHA, "evaluation set changed"
# scoring/training geometry alignment (1.27)
assert pool_manifest["generation"]["n_cond_msgs"] + \
       pool_manifest["generation"]["n_gen_msgs"] == args.msg_seq_len
```

The `params_sha256` check exists because the archived rollouts came from a path named
`ckpt/wm_ft_multi3` rather than from `j5705912` directly (1.29); identity of the scoring model
must be proven, not inferred from a step number.

### 6.11 Scoring-run determinism

For rollout-generation jobs only:

```python
assert "--xla_gpu_autotune_level=0" in os.environ.get("XLA_FLAGS", ""), \
    "scoring run without autotune pinned; ~15% of rank agreement will be lost"
manifest["xla_flags"] = os.environ["XLA_FLAGS"]
assert np.array_equal(first_rollout, reference_rollout), "generation is not reproducing"
```

### 6.12 Summary

| Quantity | Declared as | Asserted from | § |
|---|---|---|---|
| effective batch | `EFFECTIVE_BSZ` | `micro_bsz × jax.local_device_count() × jax.process_count() × K`, plus the loader's own leading dimension | 6.1, 6.2 |
| parameter count | `EXPECTED_PARAMS=78539423` | `tree_leaves(state.params)`, before and after restore | 6.3 |
| token mode | `TOKEN_MODE=26tok` | loader output shape, vocabulary bound, agreement with the tokenizer class | 6.4 |
| data months | `SQUASHFS_MONTHS` | months parsed from `dataset.message_files` | 6.5 |
| replay ratio ρ | `REPLAY_RATIO` | consumed-index counters | 6.6 |
| pool repeats `R` | designed constant | consumed-index counters | 6.6 |
| LR schedule | `WARMUP_END`, `CURTAIL_EPOCHS`, `m` | the schedule object handed to `optax` | 6.7 |
| optimizer reset | `RESET_OPT_STATE` | moment norms | 6.8 |
| restore | `RESTORE_PATH`, `RESTORE_STEP` | weight distance from fresh init; step value | 6.9 |
| pool / eval identity | manifests | `file_table_sha256`, `params_sha256`, disjointness, geometry sum | 6.10 |
| generation determinism | `XLA_FLAGS` | bitwise reproduction of a reference rollout | 6.11 |

---

## 7. The two reporting coordinates

Every stage of every arm reports both, together. One coordinate alone can look healthy while
the other collapses, and this line has already published a single-coordinate reading it had
to withdraw.

Because the training job holds nothing out (1.9, 1.10), **both coordinates come from separate
evaluation jobs**, built on `run/base_model/evaluate_model_zoo_ce.py` (1.23), which already
freezes one batch set across checkpoint steps and refuses a mismatched `token_mode`.

### 7.1 Coordinate 1 — stability: old-window held-out cross-entropy

| Item | Specification |
|---|---|
| **metric** | mean next-token cross-entropy, **nats per token**, and the same figure in **nats per message** (= 26 × per-token while the token mode is fixed). Both are reported, because per-token and per-message comparisons differ by a per-group constant, and dividing by a per-group constant shrinks noise without shrinking bias |
| **`oldwin_v1`** | 8,192 windows of 500 messages, fixed seed, from the pre-training file table restricted to **2022-01-01..2024-07-31**, stratified 128 windows × 64 randomly chosen tickers. Stored as a `(file, window_start, global_idx)` manifest with a content hash |
| **contamination** | expected **7.0** of 8,192 windows were in the consumed training prefix (8.586e-4 × 8,192). Reported as a number, never claimed to be zero |
| **`newwin_v1`** | 4,096 windows from the pool's calendar window, **disjoint from pool and replay** — the new-distribution side of the same coordinate |
| **`probe_v1`** | 4,096 windows from a slice touched by no arm: 2025-09, or **2026-01** for a provably out-of-corpus read (1.13) |
| **evaluated on** | `θ₀` once as reference, and `θ_stage` for every arm × seed × `m`, under identical eval code, batch and dtype |
| **tolerance τ** | pre-registered as 3 × the measured run-to-run sd of `CE(oldwin_v1)` over the 6 seeds of A4. **Do not set τ before that sd is measured** |
| **trap to avoid** | `evaluate_model_zoo_ce.py`'s default `--date-range 2025-12-01,2025-12-31` lies **inside** the base run's `train_date_range` (2022-01-01..2025-12-31) and nothing was held out of it. Run at defaults against this checkpoint family it scores in-sample data. Always pass `--date-range` explicitly |

### 7.2 Coordinate 2 — plasticity and pool shrinkage

Two readouts, both required, because they fail differently.

**(2a) Pool retention `ρ_ret`** — §4.4: frozen 300-member subsample, frozen threshold, and
mandatorily the `θ₀`-with-fresh-seeds floor. Report the pair
`(ρ_ret(θ_stage), ρ_ret(θ₀, fresh seeds))` and their difference. Never the first alone.

**(2b) Probe AUC** — the pre-registered plasticity measurement of `PLAN.md` §2.3, applied at
stage level. Take `θ_stage` and `θ₀`; give each an identical fixed-budget adaptation
(`B_probe = 1,500`, same `E`, fresh optimizer state, identical rewarm, identical seeds) on
the `probe_v1` slice that no arm trained on; log validation NLL every 50 steps; compare AUC.
Report AUC over `[0, B_probe]` **and** over `[0.1·B_probe, B_probe]` (§3.3 item 3).
Plasticity is judged **within one lineage** — `θ_stage` against `θ₀` — never against a fresh
initialisation, which would report "plasticity present" almost regardless of the truth.

**Diagnostics logged alongside, at no extra cost**, from
`/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/code/plasticity_probes.py`:
dormant fraction, Rényi-2 effective rank, weight/gradient norms, optimization readiness; plus
`grad_norms/{global,muon,ssm,regular,in_proj,out_proj}` and `grad_norms/clip_ratio` with
`LOG_GRAD_NORMS=1`. These decide nothing alone; they are required to **co-move** before a
plasticity claim is made.

### 7.3 The joint reading

| old-window CE | plasticity / retention | Reading | Next action |
|---|---|---|---|
| flat or ↓ | `ρ_ret` well below the `θ₀`-fresh-seed floor | the stage worked | proceed to the next cycle |
| ↑ beyond τ | `ρ_ret` ↓ | genuine stability–plasticity trade | raise ρ; if that costs the pool gain, enable A8's anchor |
| flat | `ρ_ret` ≈ floor | the pool carries no trainable signal at this budget | check A6: if the anti-pool moves as much, the **selection** is uninformative; if nothing moves, the **budget** is too small |
| ↑ beyond τ | `ρ_ret` ≈ floor | pure damage | check A0: if A0 also breached τ, it is the rewarm `m`, not the data (§3.3 item 2) |
| ↓ below `θ₀` | `ρ_ret` ↓ sharply | too good | suspect leakage between pool/replay and `oldwin_v1`; §6.10 should have caught it, so treat as infrastructure failure until re-verified |
| any | `ρ_ret` ↓ **and** probe AUC ↑ | the model got better at the pool and slower to learn | this is the plasticity-loss signature and the one result worth the exercise; needs `n ≥ 5` seeds and a co-moving diagnostic before it is stated |

---

## 8. Execution order

| Step | What | Cost | Gate to the next step |
|---|---|---|---|
| E-0 | Fix job 6141106's immediate exit (1.24); see O-5 for the leading hypothesis | tiny | a 1,500-step adaptation run completes |
| E-1 | Measure rollout throughput: contexts/GPU-hour at `k=1`, `n_gen_msgs=250` | small | fills §2.6 and §4.3 with real numbers |
| E-2 | **The gate of §5.2**: 4,000 contexts from 2022-06 vs 4,000 from 2024-08 | 160k rollouts | decides whether regime shift is on the table at all |
| E-3 | Build cycle-1 pool with the manifest of §4.5 | 320k rollouts | `file_table_sha256`, `params_sha256`, disjointness all asserted |
| E-4 | 6 seeds of A4 to measure the run-to-run sd of both coordinates | small | sets τ and the paired noise floor |
| E-5 | A0–A7 × 3 seeds × `m ∈ {1,2,4}`, **V-scaled-pool** | moderate | primary contrasts |
| E-6 | A3/A4/A5 × 3 seeds, **V-exposure** | moderate | confirms the ρ ordering is not a repetition or budget artefact |
| E-7 | A8, only if τ was breached | small | |

Every training run: `CHECKPOINT_EVERY=auto` (900 s checkpoints, 60 s wandb — already the
right two frequencies, 1.21), `LOG_GRAD_NORMS=1`, checkpoints to node-local `$TMPDIR` with an
rsync back at the end, per-job output directory keyed by `$SLURM_JOB_ID`, and a
`latest_checkpoint.json` breadcrumb — which the base run did **not** write (1.25) and which
must exist before any resume path is relied on.

---

## Open questions

**O-1 — Which run is `θ₀`?** `j5705912_b30675li_5705912` is the only chain verified to retain
a wide ladder (17 steps, 275 → 69,378) and is what the existing probe launcher targets. But
it trained on **a single GPU at effective batch 4** (1.1). If the production continual-learning
line is meant to run at a larger batch, the stage and the base sit on opposite sides of a
batch-size change and §2's budget matching compares things that are not comparable.
`j5705913_rhgz7lv6_5705913`, `j5705914_qweddnw7_5705914` and `j5749206_9o8um51n_5749206`
exist; their configs and ladders have not been read. **User decision.**

**O-2 — Which score realises issue #73's first axis?** #73 asks for "sequences that diverge
hugely from the true data". `src/post_training/heuristic_learning/fidelity.py:13-17` reports
exact-mid match 0.9614 (real replay) vs 0.5511 (generated) on ticker GS, 2026-01, 8 windows ×
900 messages — but that metric is `exact_mid_frac`, a mid-price match that the module's own
defect list (D-O2, `:75-78`) flags as *looser* than a full-book match, and it produces one
number per window, not a per-context ranking. It is also computed on **290M-class**
checkpoints (293,283,039 params, `checkpoints.py:54-66`), a different model family from the
78.5M line here. The score specified in §4.2 is the one with measured reliability; whether it
is the score #73 means is unconfirmed.

**O-3 — Rollout throughput.** Contexts per GPU-hour at `k = 1`, `n_gen_msgs = 250`, on this
backbone. Everything in §2.6 and §4.3 is parameterised by it, and nothing else in this
document is more likely to change the plan. Step E-1.

**O-4 — Are the 2025 and 2026 shards complete?** Files exist for 2022-01..2026-02 (1.13), but
per-month file counts and gaps are unchecked. 2026-01 is the most valuable evaluation slice
in the design (provably outside the recorded `train_date_range`) and its completeness is
unverified. A missing month there would be a **silent skip** (1.15).

**O-5 — Why did job 6141106 exit 1 in under a second?** Leading hypothesis, not confirmed:
`node_wrapper.sh:536` unconditionally launches `python -u -B "$WORKDIR/run/base_model/runtime/train.py"`,
and four yaml configs (`dfm_smoke_1gpu.yaml`, `selftrain12h_1gpu.yaml`,
`selftrain12h_2nodes.yaml`, `selftrain12h_2nodes_resume.yaml`) set
`legacy_workdir: /lus/lfs1aip2/projects/public/u6gb/openreview-v2`, where that file **does
not exist** (verified: no `run/` directory there at all; it has a flat `run_train.py` / `lob/`
layout). The 26-token guard three lines earlier does have dual-layout fallback logic; the
training-script path does not. Whether `WORKDIR` actually resolved to `openreview-v2` for job
6141106 has not been checked — the launcher `cd`s into the sigma-0 checkout first, so it may
not have. The per-node log `logs_cl_probe/*/training_6141106_node0.log` was not read. **One
`grep` settles it.**

**O-6 — Is `RESET_OPT_STATE` acceptable as a new flag?** §2.5 requires it and it does not
exist. The alternative (accepting a 1,000-step bias-correction transient inside a 2,000-step
stage) is not acceptable, so this is a blocking code change, not a preference.

**O-7 — The `COSINE_STEPS` defect.** Verified at 1.17. Whether any past LR curve published on
this line was drawn from it — and therefore disagrees with the LR that ran — is unchecked.
Two launchers set it (`model_zoo_matched_smoke_array.batch:49`,
`model_zoo_production_array.batch:193`); in the production array `CURTAIL_EPOCHS` happens to
be set consistently so the real schedule matches, but that is a coincidence of that launcher.

**O-8 — `k = 20` is extrapolated, not measured** (§4.2). Confirming it needs generation to
`k = 40`, blocked by the missing generation provenance on the existing archive (1.28). §4.5
fixes this going forward; it does not fix the archive.

**O-9 — What tolerance τ is scientifically meaningful, as opposed to detectable?** §7.1
defines τ from the seed-to-seed sd, which makes it *detectable*. Whether +0.005 nats/token on
the old window matters for any downstream use (LOBbench divergence ratio, return-bench IC) is
unestablished, and the 19-model zoo result — **every** model's divergence ratio already
exceeds 1 (1.24–1.60, last window at +40%) — suggests the downstream metrics may not resolve
it at all.

**O-10 — Does an 8-ticker pool generalise?** Everything measured (reliability, 40% overlap,
dispersion share) is on 8 tickers; pre-training ran on 488. A pool built on 8 against a
replay stream drawn from 488 are different distributions, and the arm contrast could be a
ticker-coverage effect. Not addressed here.

**O-11 — Identity of the scoring checkpoint.** The archived rollouts came from
`ckpt/wm_ft_multi3` at step 69378 with `partial_restore=True` and 78,539,423 params (1.29).
Step number and parameter count match `j5705912`, but "fine-tuned multi" in the name suggests
it may not be the same weights. §6.10 asserts this going forward via `params_sha256`; for the
existing archive it has to be checked by hand, and if the weights differ, today's reliability
numbers were measured on a different model than the one this design trains.
