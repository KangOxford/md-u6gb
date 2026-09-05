# 06 — Review inventory, item-by-item response, and the CPU evidence that followed

> 2026-09-05, main session. Covers: what review material already existed, what was verified
> independently before being accepted, what was fixed in code, what is blocked and on what
> evidence.

## 中文速览

- **`plan_20260904/reviews/` 是空的。五份独立对抗审查一份都不存在。** 但同目录的
  `drafts/D5_premortem.md` §6 是一份 333 行的、针对我已提交 commit `e8425cb1` 的审计——
  这是唯一落盘的对抗性材料，已逐条复用。
- **它找到的四个缺陷我全部独立复现了**，不是采信：横轴不一致（`n_pairs` 140 vs 20）、
  外推被自己的残差否定（7/8 只票 ratio 随 k 上升）、**我报的「零线 0.095」就是
  1/n_bins = 0.10 的解析泄漏**（四位小数吻合）、离散度份额是下偏估计且按票范围
  0.203–0.613 被压成了「26–34%」。
- **代码已修**，测试从 12 条加到 17 条，新增的每一条都在真实缺陷上变红。
- **M1 跑不了**：现有 80 个 member **没有 `data_cond/`**，`fidelity.py` 找不到初始化簿。
  更糟的是它**退出码 0**、只打表头不打行——静默失败。
- **17 个 step 已换算成 token**（52,000 tok/step），但 `num_devices=1` 这个记录值未经核实。
- **P1/P2/P6 的验收条件写死在 §5**，三项未齐不生成任何 rollout。

---

## 1. Review inventory — what exists, verified by reading it

| location | contents | status |
|---|---|---|
| `/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/plan_drafts/_REVIEW_BRIEF.md` | the brief for five reviewers | written, **never executed** |
| `/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/plan_20260904/reviews/` | — | **empty directory** |
| `/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/plan_20260904/drafts/D5_premortem.md` §6 | 333-line audit of commit `e8425cb1` | **the only adversarial material on disk; reused in full below** |
| same, `D1_failure_pool.md` §2 | "a defect in the stratification that must be fixed before anything is built on it" | present, overlaps §2.3 below |
| same, `D3_profiling.md` §8, §10 | defects found tracing the chain; adversarial checks on itself | present, not yet folded |
| same, `D4_deletion_refactor.md` | 964 lines, deletion and refactor sweep | present, not yet folded |

**So: one adversarial audit exists and has been used. Five independent reviews do not exist.**

### 1.1 The reviewer blocker, exactly

Five reviewer agents were launched on 2026-09-04 at approximately 21:20 UTC. All five, plus
the five drafting agents, terminated with:

```
Agent terminated early due to an API error: You've hit your session limit · resets 2am (UTC)
```

Three drafting agents had written their files before termination (`02`, `04`, `05`); two had
not (`01`, `03`, written afterwards by the main session). **No reviewer wrote anything.**
The limit is a per-session quota, not a per-agent one, so relaunching five reviewers is what
exhausted it. When quota returns, relaunch the five from `_REVIEW_BRIEF.md`, **one at a time
rather than as a batch of five**, and add `plan_drafts/01`, `03` and this file to their
reading list — `01` and `03` are the two written without any independent check.

---

## 2. Item-by-item response to the D5 §6 audit

Every claim was re-derived here before being accepted. Commands are in §2.6.

### 2.1 Horizon mismatch between the two reporting paths — **CONFIRMED, FIXED**

*Audit*: `split_half` was called with `horizon_idx=None` on the raw path (averaging all seven
horizons) and `horizon_idx=2` on the stratified path (H = 50 only), so the headline compared a
seven-horizon average against a single horizon.

*Verified independently*: `results/failure_pool_reliability.json` carried `n_pairs = 140` on
the raw rows and `n_pairs = 20` on the stratified rows — 20 draws × 7 horizons against
20 draws × 1. Exactly as the audit says.

*Fixed*: `main()` now passes `a.horizon_idx` to both paths, and asserts the two paths agree on
`n_pairs` before writing anything. The assertion is not decorative: it fails loudly on the
configuration that shipped.

*Effect on the published numbers* (both paths now at H = 50, 40 draws):

| | published | corrected |
|---|---|---|
| raw, k = 1 | 0.36–0.48 | **0.374–0.507** (mean 0.439) |
| raw, k = 5 | — | 0.631–0.748 (mean 0.692) |
| stratified, k = 1 | 0.15–0.25 | **0.144–0.250** (mean 0.190) |
| stratified, k = 5 | — | 0.330–0.545 (mean 0.462) |

The audit's own recomputation (0.370–0.507 raw k=1; 0.339–0.556 stratified k=5) matches to
within draw noise. The numeric damage was small, **which is why it survived** — a defect whose
effect is small is one nobody notices until it is applied where it matters.

### 2.2 The k ≈ 20 extrapolation — **CONFIRMED, FIXED, and the headline changes**

*Audit*: the one-parameter model is rejected by its own residuals; least squares through the
origin on `x = 1/k` is dominated by the `k = 1` point, so the fitted slope tracks the smallest
observed noise/signal ratio and **k is extrapolated too low**.

*Verified independently*: the implied ratio `(1/rho_k − 1)·k` must be constant if the model
holds. Measured on the stratified score:

```
ticker    k=1    k=2    k=3    k=5   trend
AMD      4.39   4.91   5.09   5.40  rising
AMZN     4.24   5.34   5.77   6.02  rising
GOOG     4.68   6.79   8.37   9.87  rising
INTC     3.52   3.85   4.52   4.93  rising
JPM      6.30   4.83   5.10   4.36  falling
META     3.98   4.24   4.97   5.38  rising
MSFT     4.54   5.75   6.57   7.85  rising
NFLX     3.12   3.74   3.97   4.02  rising
```

Rising in 7 of 8. The audit is right.

*Fixed*: `rollouts_needed` now returns the one-parameter fit, the largest-`k`-point fit, and a
two-parameter fit with its intercept, and **refuses to emit `k_for_rho_*` when the residual
exceeds 10% of the y-range**, returning `rejected_reason` instead. A point estimate the data
rejects is worse than none, because it is the thing that gets quoted.

*Corrected headline*: the fit is rejected in **5 of 8** tickers at the 10% threshold. From the
largest-`k` point alone, **k for ρ = 0.80 is 17–41, median 21, and still rising with k.**
~~"a one-parameter fit puts k at roughly 20"~~ — the honest statement is that no extrapolation
from `k ≤ 5` is supported and the error direction is that k is larger than quoted.

### 2.3 The stratification "zero line" is an analytic leak — **CONFIRMED, FIXED**

*Audit*: a score that is a pure monotone function of |y| still correlates exactly `1/n_bins`
with |y| after stratification, so the null floor is the leak, not noise. Production uses
`n_bins = 10`; the leak is 0.10; the reported `independent` and `cross` nulls sit at 0.06–0.11.

*Verified independently*, feeding `stratify` a pure |y| score:

```
n_bins   5      10      20      40
|rho|    0.2000 0.1000  0.0500  0.0250       exactly 1/n_bins, to four decimals
```

**This corrects a claim in the published notebook.** The reported "independent = 0.095, the
real zero line" was described as noise. It is the floor the 10-bin construction imposes.
The conclusion that genuine signal survives (true 0.46 against that floor) is unchanged, but
the floor must be stated as a construction artefact and quoted beside every stratified null.

*Also confirmed*: `test_stratify_removes_a_score_that_is_only_the_realised_move` asserted
`|rho| < 0.10` at `n_bins = 20`, where the leak is 0.05 — the test passed at twice the
production tolerance and could not have caught this.

*Fixed*: added `stratification_leak(n_bins) = 1/n_bins`, documented in `stratify`'s docstring,
emitted into the results JSON per ticker, and pinned by two new tests — one on the identity at
four bin counts, one at the production parameter.

### 2.4 The dispersion share — **CONFIRMED, FIXED**

*Audit*: `spread_share_top_decile` uses `spread_pop` (ddof = 0), which understates σ² by
(k−1)/k = 0.9 at k = 10, while the denominator is unbiased for bias² + σ². And "26–34%" is a
cross-ticker mean per horizon that hides the per-ticker range.

*Verified independently*, H = 50, top decile:

| ticker | published (ddof=0) | unbiased ×k/(k−1) | ceiling on any training gain |
|---|---|---|---|
| AMD | 0.266 | 0.296 | 70% |
| AMZN | 0.353 | 0.392 | 61% |
| **GOOG** | 0.613 | **0.681** | **32%** |
| INTC | 0.211 | 0.234 | 77% |
| JPM | 0.203 | 0.225 | 77% |
| META | 0.277 | 0.308 | 69% |
| MSFT | 0.287 | 0.319 | 68% |
| NFLX | 0.395 | 0.439 | 56% |

Per-ticker range at a fixed horizon is **0.203–0.613**, not 0.26–0.34. Compressing it into a
two-digit interval hid the ticker whose ceiling on training gain is 32%.

*Fixed*: `dispersion_share` now emits `spread_share_*_unbiased` and
`max_removable_share_top_decile`; a test pins the ×k/(k−1) relation.

### 2.5 Accepted but not yet acted on

| audit finding | why not yet |
|---|---|
| §6.3(a): every published null is a **single permutation draw**; with 60 draws `shared` exceeds `true` in **8/8** tickers | Accepted. The correct statement is stronger than the published one — a mis-paired score is *systematically* more self-consistent. `pairing_nulls` still returns one draw; making it return a spread is a small change and is listed in §6 |
| §6.3(b): a low `cross` is consistent with a pure generated-scale ranking; the discriminating null is a **partial against the rollouts' own dispersion**, from held-out seeds (85–92% of reliability survives) | Accepted and it strengthens the conclusion. The function does not exist yet |
| §6.4: `num_errors` in `inference.log` is **not an error count** — it is the number of generated messages that left the L2 book unchanged (34.5–77.0% of every rollout), is measured *more* reliably than the failure score, and **changes sign** against the score between tickers | Accepted. Must become a pre-registered covariate. Not yet joined |
| §6.4: cross-horizon leak — stratifying at H = 50 leaves ρ(corrected, \|y\|@H250) up to +0.216 | Accepted, unaddressed |
| §6.4: pool membership is 14–34% a consequence of the unregistered `n_bins` choice | Accepted; `n_bins` must be pre-registered |
| §6.7: `submit_adaptation_pair.sh` **exports no seed** although its header claims both members share one; `TEST_DATE_RANGE` is exported empty; "early" = step 275 is **pre-warmup** | Accepted; feeds `01` §4's rewrite of Step 2 |
| §6.8: the A1 weight probes disagree — `mean_abs_non_embed` falls 4.6% monotonically while `spectral_norm_median` rises 41%; top-5 spectral norms pinned to 63.94–64.01 across 69k steps | Accepted, unexplained, must not be quoted either way yet |

### 2.6 Reproducing this section

```bash
cd /lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/code
python3 -m pytest test_failure_pool_reliability.py -q          # 17 passed
python3 failure_pool_reliability.py --draws 40 --out ../results/failure_pool_reliability.json
python3 -c "import numpy as np, failure_pool_reliability as F
rng=np.random.default_rng(0); y=rng.normal(size=40000); s=np.abs(y)**2
print([round(abs(F.spearman(F.stratify(s,y,n_bins=b), np.abs(y))),4) for b in (5,10,20,40)])"
```

---

## 3. M1 — the record/replay consistency diagnostic: **blocked, with evidence**

M1 (`01` §2.2) was to run `fidelity.py`'s gen-arm replay per context, because the gen arm
reproduces its own recorded book in 0 of 8 windows against the real arm's 5 of 8, and nobody
knows how much of the failure score that contaminates.

**It cannot run on the existing archive.** `fidelity.py` needs `data_cond/` — the conditioning
window that initialises the replay (`episode_builder.py:263-269`). The 80 members hold
`data_gen/` and `data_real/` and **no `data_cond/`**:

```
$ ls .../hp_v5me3_AMD_s97702/member_0/
  data_gen  data_real  .returns_*.npz  sample_indices_rank0.json  inference.log  .done

$ python3 -c "... fidelity.main(['report','--episodes',M,'--stock','AMD','--n','2'])"
  FileNotFoundError: .../member_0/data_cond/AMD_2026-01-07_orderbook_real_id_27578.csv not found.
  window                 arm    msgs     mid    full  mean_tk  1st_div  1st_bad  onset  pos ...
  exit=0
```

Two things follow, and the second is worse than the first.

1. **The filename pattern matches** (`AMD_2026-01-02_message_real_id_1045_gen_id_0.csv` is
   exactly what `discover_windows` looks for), so this is not a layout mismatch. The
   initialisation input was simply never written. That is the P1-manifest failure in its
   purest form: the archive kept what one consumer needed and dropped what another needed,
   and the loss is invisible until a new consumer arrives.
2. **`fidelity.py` exits 0 having produced only a header row.** Every window failed and the
   process reported success. Any wrapper that checks the exit code would record M1 as done.
   This must be fixed before M1 is run anywhere: **a report with zero rows is a failure.**

**Consequence for the plan**: M1 moves behind G2, and G2's manifest (§5, P1) must require
`data_cond/`. Since a regeneration is needed for M1 anyway, and `01` §1.1 established that
the two threads can only be unified by regenerating from the selftrain chain, **the two
regenerations are the same regeneration.** That is the strongest argument yet for B-to-A.

---

## 4. The 17 selftrain steps in tokens

From `.../checkpoints_selftrain/j5705912_b30675li_5705912/metadata/_ROOT_METADATA`:
`micro_bsz = 4`, `num_devices = 1`, `grad_accum_steps = 1`, `msg_seq_len = 500`; token mode
26tok from the generation log (`[Encoding] Using 26tok encoding (MSG_LEN=26)`).

```
tokens/step = 4 x 1 x 1 x 500 x 26 = 52,000
```

| step | tokens | | step | tokens |
|---:|---:|---|---:|---:|
| 275 | 0.014B | | 58949 | 3.065B |
| 22495 | 1.170B | | 60532 | 3.148B |
| 24080 | 1.252B | | 62113 | 3.230B |
| 28830 | 1.499B | | 63695 | 3.312B |
| 30410 | 1.581B | | 65275 | 3.394B |
| 33575 | 1.746B | | 66853 | 3.476B |
| 52590 | 2.735B | | 68435 | 3.559B |
| 55773 | 2.900B | | 69378 | **3.608B** |
| 57365 | 2.983B | | | |

Candidate early/late pairs for M6:

| pair | separation | note |
|---|---:|---|
| 275 → 69378 | 3.593B | ~~the pair `attach_adaptation.sh` uses~~ — **step 275 is pre-warmup**; this is untrained-vs-trained, not early-vs-late (audit §6.7) |
| 22495 → 69378 | 2.438B | largest honest separation |
| **33575 → 69378** | **1.862B** | the audit's recommended "early"; already has offline weight probes |
| 52590 → 69378 | 0.873B | too close to be informative |

**Unverified input, flagged rather than hidden**: `num_devices = 1` is what the checkpoint
metadata records, and a single-GPU run of 69,378 steps is plausible but not confirmed. If the
run was multi-device, every figure above scales by the device count and the *ratios* between
steps are unaffected. What settles it: the wandb config for this run, or a `world_size` line
in the run's stdout log. Neither has been read. **Do not quote the absolute token counts
until it is.**

---

## 5. P1 / P2 / P6 acceptance — nothing generates until all three pass

These are the three preconditions from `03` §1 that cannot be retrofitted onto members that
already exist. Each is a script that exits non-zero, not a judgement.

### P1 — rollout manifest

**Accepted when**, for the run root about to be written:

1. `manifest.json` exists at the run root **before** the first member directory, i.e. its
   `written_at_utc` precedes the earliest member mtime.
2. Every field of the `03` §2 schema is present. Absent values are JSON `null`; the strings
   `"unknown"`, `"n/a"` and `""` fail.
3. `context_file_sha256` equals the hash of the file on disk.
4. `n_gen_msgs`, `n_cond_msgs`, `k`, `seed0`, `seed_stride`, `xla_flags` are all present —
   without `seed_stride`, "10 seeds" does not identify which 10.
5. `optimizer_state_present` is recorded. `wm_ft_multi3` restores with Muon **missing**; a
   manifest that does not say so lets an inference-only artefact be mistaken for a trainable one.
6. **`data_cond` is listed among the written streams** (§3). Without it the archive cannot be
   replayed and M1 cannot run.

### P2 — shared hashed context set

**Accepted when**:

1. One index file exists outside every member directory, and its sha256 is recorded in the
   manifest and in each member.
2. Every member's own `sample_indices_rank0.json` has the **same** content hash as that file.
3. The context ids join to `inference.log` (verified achievable: the id sets match exactly in
   8/8 tickers today), so per-rollout covariates such as the book-inert message count can be
   attached without a second pass.
4. A regeneration with the same manifest and the same context file reproduces the same id set.

### P6 — inode write plan

**Accepted when**:

1. `lfs quota` is read **immediately before** the run and its free-inode count is recorded in
   the manifest with its timestamp. The headroom is borrowed: the project sat **118 inodes**
   from its cap at 2026-09-04 17:54Z and the 741,511 free at 02:55Z were released by cleanup.
   A budget quoted without its timestamp is not a budget.
2. `inodes_planned = members × per_member` is written down, and `per_member` is measured, not
   assumed. Today it is **3,007** unpacked (500 contexts × ~6 files), **1,507** with the real
   arm written once per ticker (`05` §5.3, md5-verified byte-identical across seeds).
3. `inodes_planned < 0.5 × free_at_start`. Half, not all: the project is shared and the
   measured release rate is not under this plan's control.
4. The generated CSVs are written to node-local `$TMPDIR`, their CPU consumers run **in the
   same allocation**, and only `.npz` + manifest + derived scores return to Lustre — unless
   `data_cond`/`data_gen` are required downstream, in which case they count against (2).
5. `PACK_MEMBER=0` stays (`05` §5.4): packing discards `data_cond/`, `data_tokens/` and the
   `message_*.csv` that the spread-regime histogram reads.

**Current status: P1 absent, P2 partial, P6 absent. No generation may start.**

---

## 6. Next, in order

| # | what | kind | blocked by |
|---|---|---|---|
| 1 | Relaunch the five reviewers from `_REVIEW_BRIEF.md`, **one at a time**, reading list extended to `01`, `03`, `06` | agent | session quota |
| 2 | `pairing_nulls` returns a spread over draws, not one draw (§2.5) | CODE | nothing |
| 3 | Dispersion-partialled reliability + its known-answer test (§2.5) | CODE | nothing |
| 4 | Join `num_errors` (book-inert message count) to context ids as a pre-registered covariate | CPU | nothing |
| 5 | Fix `fidelity.py` to exit non-zero on a zero-row report (§3) | CODE | nothing |
| 6 | Read the wandb config for `j5705912` to confirm `num_devices` (§4) | CPU + network | nothing |
| 7 | Fold `D1` §2, `D3` §8/§10, `D4` into the merged plan | CPU | nothing |
| 8 | P1/P2/P6 acceptance scripts (§5) | CODE | nothing |
| 9 | G2 regeneration from the selftrain chain, **with `data_cond`** | GPU | 1–8 |

Items 2–8 are all CPU or code and none is blocked.

## Open questions

1. **The dilution assumption is still untested** (`02` §3.2): that a false positive in a
   training pool dilutes rather than contaminates. The whole `k = 3` budget rests on it, and
   `02` says explicitly that nothing should assume it. It is untested here too.
2. Whether `data_cond` can be reconstructed from `data_real` plus the conditioning offsets,
   which would unblock M1 without a regeneration. Not investigated.
3. Whether the 5-of-8 `rejected_reason` rate at a 10% residual threshold is the right
   threshold. It was chosen here, not derived, and it is exactly the kind of unregistered
   choice §2.5 criticises in `n_bins`.
