# Continual Learning for sigma-0 — task board

> **This board is the operational head of the plan.** The revision log below it is the
> evidence trail. A plan existing is not a task being done, and a pushed notebook is not a
> scientific conclusion; both are tracked separately here.
> Board last updated 2026-09-05T20:36Z.

## 1. Original goal

Answer GitHub issue KangOxford/sigma-0 **#73**: can a model be improved by mining rollouts
that diverge from the true data into a pool and continuing training on a mix of that pool and
historical replay — and if so, at what mix. Adjacent thread from `PLAN.md` §3: does the
network lose plasticity as it trains, measured by early-vs-late checkpoints under a fixed
adaptation budget.

## 2. Claims to adjudicate, and where each stands

| # | claim | state | evidence |
|---|---|---|---|
| C1 | A per-context failure score is reliable enough to build a pool | **partly** | k=1 unusable (0.144–0.250); k=5 gives 0.323–0.527. Required k not established — the only extrapolation is rejected by its own residuals |
| C2 | The score measures the model, not the size of the realised move | **established for the corrected score** | raw keeps 0.46 after pairing is destroyed; stratified drops to the binning floor. On held-out **contexts** the literal issue-#73 rule carries **2.88–2.98×** the population move vs 1.21–1.25 stratified, CI [+1.17,+2.15] excluding 0, 8/8 |
| C3 | The stratification is sound | **fixed, not free** | zero-move atom collapses 1–4 deciles; `stratify_v2` gives it its own stratum, leak 0.13 → 0.057 |
| C4 | `stratify_v2` edges can be fitted once and reused | **NOT established** | CI [−0.148,+0.099] SD units, up to 18% of the effect; non-significance is not equivalence |
| C5 | A false positive in a training pool dilutes rather than contaminates | **no departure detected at ~5% resolution** — nothing stronger | S1: deviations from linearity +0.0005…+0.0079 (primary), −0.0085…+0.0042 (sensitivity), every CI includes 0, widest bound ±0.037 SD units = **4.6% of the 0.81 effect**. That is the resolution of the test, not a property of the data. **Linearity is not shown**; a departure smaller than ~5% of the effect is invisible here, and no tolerance was pre-set. The `k=3` budget may be quoted only with this resolution attached |
| C6 | Plasticity declines with training | **not started** | M6 never ran; launcher defect now routed around |
| C7 | Training on the pool improves the model | **not started** | needs C5 and a training run |

## 3. Necessary steps, dependencies, and what each needs

| id | step | needs | depends on | state |
|---|---|---|---|---|
| S1 | Dilution test on the existing archive | CPU | — | $\color{green}{\textsf{DONE 2026-09-05}}$ — RESULTS addendum 9 |
| S2 | Export a seed per member in the adaptation launcher | **CPU** | — | not done; blocks M6's seed replication |
| S3 | Pre-register M6's same-age null pair | CPU | S8 | $\color{green}{\textsf{written}}$ — RESULTS addendum 10: entry traced, budget 78.0M tokens/member, decision rule fixed. **Not proposed for execution**: S8 is upstream |
| S4 | R3–R5 adversarial reviews, one at a time | CPU (agent) | — | R1, R2 done |
| S8 | **Decide which ladder M6 probes** — selftrain chain (3.608B tokens, 1.69% of onset) or the `wm_ft_multi3` ft ladder (250M tokens, on the rollouts' lineage) | CPU decision | X4 (closed) | **blocking S3's execution and S5** |
| S5 | M6 fixed-budget adaptation, early vs late | **GPU, 2 devices** | S2 ✓, S3 ✓ written, **S8** | launcher unblocked; `RESTORE_PATH` still pinned to the selftrain chain, and "early" should be 33575 not the pre-warmup 275 |
| S6 | G2 rollout regeneration with `data_cond` | **GPU** | S1 (for k), gate returning 0 | gate CLOSED on P1 for the historical archive |
| S7 | Training arms for issue #73 | **GPU** | S6, C5 | not started |

## 4. Cancelled or deferred, with the reason

| item | state | why |
|---|---|---|
| ~~M1 on the existing archive~~ | **impossible** | no `data_cond/` in any of the 80 members; folded into S6 |
| ~~PLAN Step 5, multi-size onset law~~ | **excluded on evidence** | the run is 1.69% of where the onset law places onset |
| ~~"A-to-B is impossible" (X4)~~ | **closed by measurement** | `wm_ft_multi3` is a 32-rung fine-tune descended from selftrain/69378 |
| ~~"edges may be fitted once and reused"~~ | **withdrawn** | non-significance read as equivalence |
| ~~"`num_errors` is not better measured"~~ | **withdrawn** | k mismatch; at matched k the original audit holds 8/8 |

## 5. Budget and stopping criteria

- **No GPU is requested for a step whose inputs are not settled.** S5 and S6 both have open
  inputs, so the current request is **0**.
- Rollout generation stops before it starts unless `code/generation_gate.py` returns 0. P1
  cannot pass on the historical archive and is satisfiable only by a new run writing its
  manifest first.
- Inode ceiling: a plan may use at most **half** the free inodes read at run time, never a
  figure quoted from a document. Successive readings moved 741,511 → 311,685 in one session.
- No number enters a title, is bolded, or is called "best" while n is still growing.
- A rejected extrapolation sets no budget in either direction.

## 6. Current GPU request

**0 devices.** Evidence: S5's two blocking inputs (S2, S3) are CPU work not yet done, and S6
is gated. Fresh `gtop` at **2026-09-05T20:35:53Z**: 64 cards, **50 truly free (0.0 MiB), 1
held, 14 occupied**; 36 named steps of other tasks are running on the shared allocations
`6324119/6324128/6324130/6324135`. Those cards should go to a task that can use them now.
S2 and S3 have landed; **S8 is now the blocker**. When it resolves the request is **2 devices
on one node**, reported from a sample taken immediately before launch. `nid010292` is excluded
— it reads 4/4 idle but **GPU0/1 there are granted to LDM**, and an idle reading is not a grant.

## 7. Next action

**S8, the ladder scope decision** — a CPU decision, not a measurement, and now the only thing
between here and a concrete GPU request. S2 and S3 are done (RESULTS addendum 10).

**Blocking everything that writes a file**: the project is at its inode hard cap
(**51,200,000 / 51,200,000**) and `$HOME` is at its space cap (**101G / 101G**). New files
cannot be created on either; this round's results were appended to an existing file. Clearing
one of the two is a prerequisite for S5/S6 artefacts and is not this line's to decide alone.

## 8. Readable results

| what | path |
|---|---|
| results, eight addenda | `results/RESULTS_20260905.md` |
| notebook, 7 figures | `failure_pool_reliability.html` / `.ipynb` |
| pre-registrations | `results/PREREG_selection_rules_20260905.md`, `results/PREREG_context_holdout_20260905.md` |
| reviews | `plan_drafts/R1_review_of_01_03_06.md`, `plan_drafts/R2_review_statistics.md` |
| the gate | `code/generation_gate.py`, `code/write_run_manifest.py` |
| worktree incident | `results/worktree_incident_20260905.md` |

---

# Continual Learning for sigma-0: Plasticity Measurement and Continual Pre-Training Plan

> Task dir: `tasks/continual_learning/` · Source research: `deep-reseach.md` (two-pass deep research, 2026-08-26)
> **Status: MERGED PLAN, revision 7 (2026-09-05).** Five facet drafts under `plan_drafts/`
> are the detail; this file is the spine and the decision record. Green marks what is done,
> ~~strikethrough~~ marks what measurement has overturned. Not yet adversarially reviewed —
> see §0.4.

---

## 0. Revision 2 — what changed on 2026-09-04/05 and why

Five facets were drafted (`plan_drafts/01`–`05`). Three were written by parallel agents,
two (`01_measurements.md`, `03_infrastructure.md`) by the main session after the agents for
those facets were killed by a session limit. Everything below is traceable to a measurement.

### 0.1 New measured facts

| # | fact | where |
|---|---|---|
| F1 | A single rollout per context cannot rank contexts: split-half rank correlation 0.36–0.48 raw, 0.15–0.25 corrected | `failure_pool_reliability.ipynb` |
| F2 | The obvious failure score is mostly not about the model: it correlates 0.65 with \|realised move\|, keeps 0.46 after the rollout-to-context pairing is destroyed, and two independently permuted halves still agree at 0.43. Stratifying within \|realised move\| bins drops these to 0.03 and 0.10. The naive and corrected top-decile pools overlap 40% | same |
| F3 | Split-half reliability certifies nothing on its own — a consistently mis-paired score scored 0.49 against the correct score's 0.46 | same |
| F4 | Error partitions exactly into systematic and dispersion terms; dispersion is 26–34% inside the top decile and the model is under-dispersed, so that is a floor | same |
| F5 | ~~Generation nondeterminism sets a ~15% ceiling on rank agreement~~ **Wrong in sign.** Two whole regenerations agree *more* than two disjoint seed sets, because a fraction φ of members never fork (0.976 at h=10, 0.191 at h=250). It behaves like a partial redraw, which 1/k averaging removes | `plan_drafts/02` §2.2 |
| F6 | On the corrected score the regeneration figure is 0.583–0.972, grand mean 0.742 — not the raw score's 0.846 | `plan_drafts/02` §2.2 |
| F7 | Doing nothing moves an arm-level endpoint by up to 28%; training-seed variance has **never been measured on this project** | `plan_drafts/02` §2.3 |
| F8 | `k >= 20` is a *selection* requirement for a per-context ranked list. For a **training pool**, false positives dilute rather than contaminate, and cost-per-power favours small k by 2.4× | `plan_drafts/02` §3.2 |
| F9 | For a paired arm comparison only `R = N·k` matters; the 1/sqrt(R) law holds to 1% | `plan_drafts/02` §3.3 |
| D1 §2 | **CONFIRMED and fixed.** `stratify()` is not decile stratification: 97–208 of 500 realised moves are exactly zero at H=50, so 1–4 bins are empty and one stratum holds 20–50% of the sample. ~~The leak floor is exactly 1/n_bins = 0.10~~ — that holds on a uniform \|y\|; measured it is **0.080–0.171**, mean 0.13. `stratify_v2` gives the zero mass its own stratum: leak → **0.045–0.072**, reliability 0.460 → 0.438, so the signal-to-floor margin goes from ≈3.5× to ≈**7.4×**. RESULTS addendum 5 §F |
| ~~F10~~ | **REFUTED by R1 (§0.11), then settled by measurement (RESULTS addendum 5 §E): `wm_ft_multi3` is a 32-rung fine-tune DESCENDED from the selftrain checkpoint at 69378 — weights differ, distance grows monotonically with fine-tune steps (0.261 → 0.346 → 0.525), and the byte-identical `_ROOT_METADATA` records that ancestor. X4 CLOSED.** ~~`wm_ft_multi3` holds one checkpoint (69378) with **no Muon optimizer state** (418.6 MB vs the selftrain chain's 499.5 MB); the selftrain chain holds 17 (275…69378) | measured 2026-09-05, `plan_drafts/01` §1.1 |
| F11 | One rollout member costs **3,007 inodes / 67 MB**; the analysis reads only 112 KB of `.npz` from it | `plan_drafts/03` §3.2 |
| F12 | The project sat **118 inodes** from its hard cap at 2026-09-04 17:54Z; the 741,511 now free were released by cleanup over nine hours, and are borrowed headroom rather than a baseline | `plan_drafts/03` §3.3 |
| F13 | The real arm is byte-identical across seeds (md5-verified), so writing it once per ticker halves the per-member inode cost | `plan_drafts/05` §5.3 |

### 0.2 The four cross-draft contradictions, and how they resolve

| # | contradiction | resolution |
|---|---|---|
| X1 | `04` builds the pool at `k = 20`; `02` §3.2 measures that small `k` is 2.4× better cost-per-power **for a training pool** | **Split by purpose.** `k = 20` only on the subsample reported as a ranked list; `k = 3` for the pool that feeds training, with the dilution assumption pre-registered and tested (it is untested — `02` open question) |
| X2 | `04` §8 E-3 wants 320,000 rollouts; `03` §3.4 measures that this is 130% of the free inode budget even after dedupe | X1 dissolves it: at `k = 3` the same 16,000 scored contexts cost **48,000 rollouts = 96 members = 20% of free inodes** |
| X3 | `05` §5.3 assumes the real-arm dedupe; `04`'s sizing does not | **The dedupe is a precondition, not an optimisation.** Without it the era gate alone is 130% over |
| X4 | `05` §2.6: the two threads operate on different weights at the same step number | **Settled by F10.** `wm_ft_multi3` has no early/late pair and no optimizer state, so it cannot be adapted. Either Thread B regenerates from the selftrain chain, or the threads stay separate |

### 0.3 The budget, after X1 and X3

| item | rollouts | members | inodes (deduped) | % of free |
|---|---:|---:|---:|---:|
| Era gate, 8,000 contexts at `k = 3` | 24,000 | 48 | 72,336 | 10% |
| Cycle-1 pool, 16,000 contexts at `k = 3` | 48,000 | 96 | 144,672 | 20% |
| Ranked-list subsample at `k = 20` | as needed | — | — | — |
| ~~Era gate at k = 20~~ | ~~160,000~~ | ~~320~~ | ~~482,240~~ | ~~65%~~ |
| ~~Cycle-1 pool at k = 20~~ | ~~320,000~~ | ~~640~~ | ~~964,480~~ | ~~130%, does not fit~~ |

### 0.4 What has **not** been done

**No adversarial review has run.** `plan_drafts/_REVIEW_BRIEF.md` specifies five reviewers;
none were launched — the session limit hit first. Facets `01` and `03` were additionally
written by the same session that produced the measurements they rest on, which is exactly
the arrangement the standing order exists to prevent. **Treat revision 2 as unreviewed.**

### 0.5 Revision 3 (2026-09-05) — the audit, and what it corrected

`plan_20260904/reviews/` is **empty**: the five independent reviewers of
`plan_drafts/_REVIEW_BRIEF.md` were never executed, all ten agents of that round having hit
`You've hit your session limit · resets 2am (UTC)`. One adversarial artefact does exist —
`plan_20260904/drafts/D5_premortem.md` §6, a 333-line audit of commit `e8425cb1` — and it
found four real defects in the published measurements. **All four were re-derived
independently before being accepted**; details and reproduction in `plan_drafts/06`.

| # | defect | corrected reading |
|---|---|---|
| A1 | The raw and stratified reliabilities were computed on **different horizon sets** (`n_pairs` 140 vs 20: seven horizons averaged against one) | Both now at H = 50. raw k=1 **0.374–0.507**, stratified k=1 **0.144–0.250**, stratified k=5 **0.330–0.545** |
| A2 | ~~"a one-parameter fit puts k at roughly 20"~~ The implied noise/signal ratio **rises with k in 7 of 8 tickers**, so the law is rejected by its own data and least squares through the origin extrapolates k **low** | From the largest-k point: **k for ρ=0.80 is 17–41, median 21, and still rising.** No extrapolation from k ≤ 5 is supported. `rollouts_needed` now refuses to emit a point estimate its residual rejects |
| A3 | ~~"independent = 0.095, the real zero line"~~ That is the **analytic leak of the 10-bin stratification**, `1/n_bins`, confirmed to four decimals (0.2000/0.1000/0.0500/0.0250 at 5/10/20/40 bins) | The floor is 0.10 by construction. Signal still survives above it (true 0.46), but the floor is not noise and must be quoted beside every stratified null |
| A4 | ~~"dispersion is 26–34 percent inside the top decile"~~ That is a cross-ticker mean per horizon of a **downward-biased** estimator (ddof = 0 numerator, unbiased denominator) | Per ticker at H = 50, unbiased: **0.225–0.681**. GOOG's ceiling on any training gain is **32%**, not ~70% |

Accepted and not yet acted on: every published null is a **single permutation draw** (with 60
draws `shared` exceeds `true` in 8/8 tickers, so the point is stronger than published); the
discriminating null is a partial against the rollouts' own dispersion (85–92% of reliability
survives it); `num_errors` in `inference.log` is **not an error count** but the number of
generated messages that leave the L2 book unchanged (34.5–77.0% of every rollout, measured
*more* reliably than the failure score, sign-flipping between tickers) and must become a
pre-registered covariate.

Tests: **12 → 17**, each new one red on a defect that actually shipped.

### 0.6 M1 is blocked, with evidence

`fidelity.py` cannot run on the existing 80 members: they hold `data_gen/` and `data_real/`
and **no `data_cond/`**, which is the conditioning window that initialises the replay
(`episode_builder.py:263-269`). The filename pattern matches, so this is not a layout
mismatch — the initialisation input was never written.

~~Worse, the tool exits 0 having printed only a header row.~~ **That was wrong and is
corrected in `results/RESULTS_20260905.md` §5**: the observed run exits **1**; the `exit=0`
originally reported was `tail`'s exit code, read as `$?` after a pipe. A narrower version of
the claim is real — a report in which every window is skipped *without raising* did return 0
— and a guard for it is now in `fidelity.py`, verified to return 2. The `data_cond` finding
is unaffected; it rested on the `FileNotFoundError`, not on the exit code.

Since M1 needs a regeneration and `plan_drafts/01` §1.1 established that the two threads can
only be unified by regenerating from the selftrain chain, **those are the same regeneration**.

### 0.7 The 17 selftrain steps in tokens

`micro_bsz 4 × num_devices 1 × K 1 × msg_seq_len 500 × 26 tok/msg = 52,000 tokens/step`
(from `checkpoints_selftrain/j5705912_b30675li_5705912/metadata/_ROOT_METADATA`).
Step 69378 = **3.608B tokens**. Candidate M6 pairs: ~~275 → 69378~~ (**step 275 is
pre-warmup** — untrained-vs-trained, not early-vs-late), **33575 → 69378 = 1.862B**,
22495 → 69378 = 2.438B. **`num_devices = 1` is recorded but unverified**; if the run was
multi-device every absolute figure scales, though the ratios do not. Do not quote the
absolute counts until the wandb config is read.

### 0.8 P1 / P2 / P6 acceptance — nothing generates until all three pass

Written out as exit-code conditions in `plan_drafts/06` §5. In one line each: **P1** a
manifest whose `written_at_utc` precedes the first member, every field present, `null` never
`"unknown"`, and `data_cond` among the written streams; **P2** one shared context file whose
sha256 matches every member's copy and which joins to `inference.log`; **P6** free inodes read
and timestamped immediately before the run, `per_member` measured (3,007 unpacked / 1,507
deduped), and `inodes_planned < 0.5 × free_at_start`.

**Current status: P1 absent, P2 partial, P6 absent. No generation may start.**

### 0.9 Still unverified, carried forward on purpose

The **dilution assumption** — that a false positive in a training pool dilutes rather than
contaminates — is what makes the `k = 3` budget of §0.3 affordable. `plan_drafts/02` §3.2
states plainly that it is untested and that nothing should assume it. It is still untested.
If it fails, §0.3 reverts to `k ≈ 21` and the cycle-1 pool stops fitting in the inode budget.

---

### 0.10 Revision 4 (2026-09-05) — CPU verification of the accepted-but-unimplemented items

One convention throughout: **horizon 50, 8 tickers, 500 contexts, 10 seeds, 60 draws,
stratified score at `n_bins = 10`, config `v5me3`.** Full tables in
`results/RESULTS_20260905.md`; machine-readable in `results/nulls_and_partials.json`.

| item | verdict | reading |
|---|---|---|
| Repeated pairing nulls | **SUPPORTED** | `shared > true` in **8/8** tickers at k=5 (mean gap +0.032), 7/8 at k=3. Every previously published null was one permutation draw; the ordering is systematic, so the point is stronger than published. `independent`/`cross` sit at 0.05–0.10, i.e. **at the 0.10 leak floor, not at zero** |
| Dispersion partial | **SUPPORTED, with a floor nobody had stated** | Kept **0.83–0.92, mean 0.87**. A score that is *entirely* dispersion keeps **0.48** under the identical procedure at 10 seeds (0.37 at 12 — the floor falls as the proxy improves). So the reading is "0.87 against a 0.48 floor", roughly half the margin that "0.87 survives" implies |
| `num_errors` semantics | **SUPPORTED** | Source-level: it counts generated messages that left the visible L2 book unchanged, under a comment calling them errors |
| `num_errors` magnitude | **SUPPORTED** | 34.0–76.4% per ticker, **pooled 60.6% of every rollout is book-inert**. Needs no join |
| `num_errors` correlations | **INSUFFICIENT** | 11×48 = **528 slots for 500 contexts**; the 28 surplus are a wrap of neither end (8/8 tickers). The position-to-context mapping is unestablished, so no per-context correlation is reproducible. Not refuted — the join may be recoverable from the batch construction, which has not been read |
| `fidelity.py` exit code | **REFUTED as stated** | See §0.6, corrected |

Tests **17 → 21**. Two new ones went red first and both failures were informative: one
assertion was wrong (a monotone nuisance leaves an exactly-zero residual, and `spearman` on
a constant is NaN by design); the other found the 0.48 floor above, which is the most
decision-relevant number of the round.

**Reliability budget**: per the standing instruction, the `k ≤ 5` extrapolation to `k ≈ 21`
is **not** used as a budget anywhere in this revision. Every figure above is quoted at the
`k` it was measured at.

---

### 0.11 Revision 5 (2026-09-05) — independent review R1 overturns four conclusions

`plan_drafts/R1_review_of_01_03_06.md`, the first of the five independent reviews (launched
alone; a batch of five exhausted the quota). 17 findings, four BLOCKING. Each re-derived here
before acceptance; full detail and commands in `results/RESULTS_20260905.md` addendum.

| finding | verdict | what it overturns |
|---|---|---|
| ~~"A-to-B is impossible"~~ | **REFUTED** | `wm_ft_multi3` has a **32-rung sibling ladder** `_step150…_step4800` that `01` never searched (it looked only *inside* the directory), Muon state **is** on disk (282 mentions, 835 entries, identical on both roots), and the two `_ROOT_METADATA` are **byte-identical** (md5 `028879b3…`) so step 69378 is inherited, not coincident. **F10, X4 and §0.6's "same regeneration" argument are struck** pending redo |
| Archive changed mid-round | **CONFIRMED, by content** | ~~Argued from mtimes~~ — mtime moves on a touch and cannot carry the claim. Established instead by running the code **recovered from `e8425cb1`** on today's data: it reproduces none of the four values it committed (AMD 0.2954→0.2661, GOOG 0.6359→0.6133, NFLX 0.3960→0.3948, JPM 0.2050→0.2029), and `dispersion_share` has no RNG. `06`'s "reproduces the audit" comparisons are across **two datasets**. No number here is anchored until the archive is pinned by content hash — which is exactly what P2 requires |
| Onset law vs run length | **PARTLY — the strong form is retracted** | `deep-reseach.md:2315`: `T = 1.3e-5·P^0.8269` in 5B-token instances. At P = 76M that is **214B tokens**; this run is 3.608B = **1.69%**. ~~M6 will read ABSENT for reasons unrelated to plasticity~~ — that overstates an extrapolation carried across architecture (pre-norm transformer → SSM), data (multilingual text → LOB flow) and task structure (cyclic 8-task with optimizer resets → none) at once. What holds: **an ABSENT reading here would say nothing about the law**, and the "first onset law for state-space models" framing is not reachable from this run. It does **not** show that no plasticity change exists at 3.6B |
| Step 2 never ran | **CONFIRMED** | `node_wrapper.sh:342` blanks `SQUASHFS_MULTI_MOUNT_ROOT` unconditionally, so line 370's `:-` default always fires and `attach_adaptation.sh`'s unique mount root never applies (`Transport endpoint is not connected` in both probe logs). It also makes both M6 members share one mount root. **A knob that never reaches the code** — this project's own documented failure shape |
| My A1 guard could not fire | **FIXED** | It compared two sets built from the same variable in the same function body. Now derives its expectation from the data; verified red. `regeneration_null` also still averaged 7 horizons on the raw score — at H=50 stratified it reads **0.726–0.875, mean 0.808**, not 0.846 |

R1 also **closes the `num_devices = 1` caveat** independently (a `[FLOPs] Tokens/step: 52,000`
line and `Peak BF16 (1 GPUs)`), so §0.7's token table stands.

**The lesson on my side**: `01` reported an absence found by a narrow search. An absence
found by a narrow search is a statement about the search, not about the world.

**X4 is now closed by measurement** (RESULTS addendum 5 §E): the two roots hold *different*
weights whose distance grows monotonically with fine-tuning steps, so `wm_ft_multi3` is a
fine-tune descended from the selftrain checkpoint and the shared metadata records an
**ancestor**, not a checkpoint. Both earlier positions were wrong — mine ("cannot be
unified") and the one R1's evidence invites ("may be the same object"). The threads share a
lineage, so unification needs no regeneration; ~~`PLAN` §0.6's "those are the same
regeneration"~~ is withdrawn. M1 still needs one, because `data_cond` is absent.

**The `num_errors` join is recovered** (addendum 5 §A): `num_errors[i] ↔ rank_indices[i]` for
`28 ≤ i < 500` and `num_errors[500+j] ↔ rank_indices[j]` for `j < 28`, verified by recomputing
the definition from the generated books on 3 members across 3 tickers. The 28 surplus slots
are the final partial batch padded by **wrapping to the start of the list**, so **28 of 500
contexts (5.6%) are generated twice** and their stored files come from the second pass. With
the join, the audit's claim that `num_errors` is measured *more* reliably than the failure
score is **REFUTED** (0.229–0.419 against the score's 0.330–0.545); the sign flip between
tickers is **CONFIRMED** (−0.136 to +0.195).

---

### 0.12 Revision 6 (2026-09-05) — acceptance, retractions, and P1/P2/P6 as actually stands

Detail in `results/RESULTS_20260905.md` addendum 2 and `results/worktree_incident_20260905.md`.

**The shared-worktree incident I caused is accepted and recorded.** All 75 stashed entries
reconciled by sha256: 56 modifications restored byte-for-byte, 1 (a live log) moved on after
the restore, **0 lost**; 18 uncommitted deletions belonging to another line were reverted by
the rebase and are **not** re-deleted. Stash kept, never dropped; no `reset`, no `clean`, no
further stash of the shared tree.

**Two retractions.** ~~mtime proves the archive was rewritten~~ — replaced by a content test
using the code recovered from `e8425cb1`, which reproduces none of the four values it
committed. ~~The 214B figure shows the probe cannot succeed~~ — it shows the run is 1.69% of
where that law places onset, which makes an ABSENT reading uninformative about the law; it
does not show that no plasticity change is present.

**R1-F6/F9 fixed**: `rollouts_needed` now emits the largest-k estimator for every ticker, so
an interval is one estimator throughout. Mixed 17–41 median 21 → uniform 17–41 median 22.
Neither is used as a budget. **R1-F8 fixed**: the notebook is rebuilt, 6 figures, 0 errors.

**Two attribution errors of mine, retracted** (addendum 3). ~~`nbconvert --execute` returns 0
on a failing notebook~~ — it returns **1**; my `rc=0` was `$?` read after a pipe, the third
time this session. What holds is that `--inplace` leaves a zero-output notebook on disk, so
counting outputs is still the check. ~~The archive was rewritten~~ — the supported claim is
narrower: same code, same input *set* (10 seeds, 500 contexts, matching the historical JSON),
deterministic function, different values, so **something in the unrecorded dependency set
changed**; which member is not establishable, because the historical record carries no input
hash. That gap is why the fingerprint file and the P1 hash requirement now exist.

**P1/P2/P6 are now one executable gate**, `code/generation_gate.py`, exit 0 only if all three
pass. **P2 PASSES**: the per-ticker context indices are promoted into
`results/context_sets_v5me3/` with a hashed manifest, so the shared set is a checked reference
rather than a coincidence. **P6 PASSES** against a live `lfs quota` reading (495,918 free,
plan 144,672 = 29%); free inodes fell 741,511 → 495,918 during this round, which is why the
gate reads at run time. **P1 is now two things, and neither substitutes for the other**: `code/write_run_manifest.py`
writes a manifest into a run root **before** any member exists and refuses otherwise (verified:
returncode 2 against the existing archive, which holds 2,403 member dirs), recording anything it
cannot read as `null` with a reason rather than inventing it; and
`generation_gate.py --mode historical` writes
`results/historical_attestation_v5me3.json`, schema `historical-attestation/1`, carrying
`derived_after_the_fact: true`, hashing all 81 `v5me3` members and listing **7 fields as
unrecoverable with their reasons**. **P1 still fails for the historical archive and should** —
but that no longer closes the gate forever: with a manifest written first the gate returns
**0, GATE: OPEN**. That was a demonstration into a scratch directory and is **not** an
authorisation to generate; nothing was generated. The next executable steps and the evidence
each must produce are in `results/RESULTS_20260905.md` addendum 4 §F. The gate found a defect in itself on first run — `check_p2` pooled
every config in the archive — which a checklist would not have surfaced.

#### P1 / P2 / P6 — what is actually done

| | done | outstanding |
|---|---|---|
| **P1** rollout manifest | The schema is written (`plan_drafts/03` §2) and the existing archive is fingerprinted: `results/archive_fingerprint_20260905.json` records sha256, bytes and mtime for `.returns_multih_{gen,real}.npz` and `sample_indices_rank0.json` across **80** member dirs | A manifest **written before the first member** cannot be retrofitted. P1 is satisfiable only by the next generation run |
| **P2** shared hashed context set | $\color{green}{\textsf{Condition 2 verified}}$ — the context index is **byte-identical across all 10 seeds within each ticker** (1 distinct sha256 per ticker; 8 distinct across tickers, as expected) | (1) a single index file *outside* the member dirs does not exist; (3) the join to `inference.log` is unestablished (528 slots vs 500 contexts); (4) regeneration reproducibility untested |
| **P6** inode write plan | The conditions are written and both anchors measured (3,007 inodes/member unpacked, 1,507 deduped) | The `lfs quota` read-and-record step, and the `inodes_planned < 0.5 × free_at_start` check, are not implemented as an executable gate |

**P1 partial, P2 partial (one of four conditions verified), P6 partial. No generation.**

---

### 0.13 Revision 7 (2026-09-05) — an invalid inference of mine, corrected

Detail in `results/RESULTS_20260905.md` addendum 8.

**Withdrawn.** The pre-registered reading rule at `4611383a` said "a cost indistinguishable
from zero means the edges transfer". That reads a failure to detect a difference as a
demonstration that there is none. Showing two things are close enough to substitute needs a
tolerance fixed in advance and an interval inside it, not a wide interval straddling zero.
~~"the edges may be fitted once and reused"~~ and its SUPPORTED label are withdrawn.

**The interval, target stated.** "Fit once and reuse" is a claim about a *new* ticker or
context set, so the ticker is the resampling unit, df = 7:

| analysis | mean | 95% CI | as a fraction of the contrast |
|---|---:|---|---|
| primary | −0.0242 | **[−0.1476, +0.0991]** | **[−18.0%, +12.1%]** |
| sensitivity | −0.0033 | **[−0.0718, +0.0653]** | **[−9.0%, +8.2%]** |

**The measured transfer difference is small, and reusability is NOT established.** A δ of
0.148 SD units (primary) or 0.072 (sensitivity) or larger would contain the interval — offered
as a description of the interval, **not** as pre-registered support, since no tolerance was
set in advance and choosing one now because it fits is the manoeuvre pre-registration exists
to prevent.

**What survives, and one of them is stronger than reported.** The realised-move confound
tested as the comparison actually claimed — the literal issue-#73 rule against the stratified
one, on `|bal − 1|`, paired with df = 7 — is **+1.661, 95% CI [+1.171, +2.151]** (primary) and
**+1.725, [+1.267, +2.184]** (sensitivity), **excluding zero, 8/8 tickers under both**. That
is a difference, not a failure to find one. The matched-`k` `num_errors` advantage stands
(8/8 at k=1 and at k=5). The 1.26–4.90× floor inflation is **scoped to contrasts whose two
sides genuinely cannot share draws**; it does not condemn matched pairing, which is the right
design when the two sides should share a nuisance.

---

## 1. Why this line of work

sigma-0 is a foundation model of NASDAQ limit-order-book message flow that must stay in service across years of non-stationary markets. Two failure modes threaten any such deployment:

1. **Loss of plasticity ("model aging")**: the network not only forgets, it stiffens — dormant units, representation collapse, gradients that stop carrying signal — until it can neither learn new structure nor discard old structure.
2. **Unlabeled regime drift**: markets never announce a task switch. Retraining cadence, learning-rate policy, and replay mix must be driven by measured distribution signals, not by task labels that do not exist.

Two passes of deep research (first over X/practitioner discourse, second over the primary papers) are archived in `deep-reseach.md`. This plan extracts the decisions.

## 2. What the research established

### 2.1 Plasticity loss is real for pre-trained sequence models, not just RL

- Hernandez-Garcia, Figliolia, Millidge (Zyphra, arXiv 2606.24752): GPT-style pre-norm Transformers, 5M–314M non-embedding params, multilingual next-token training. Plasticity measured as validation-loss AUC of a probing run (fixed 5B-token budget on held-out Vietnamese) from periodic checkpoints. Every size eventually loses plasticity; onset follows T ≈ 1.3e-5 · P^0.8269 (sublinear → scale delays, does not immunize). Critically, the effect appears **also under a stationary mixture** of all languages, so non-stationarity accelerates but is not required. "Scale alone cannot save us."
- Springer et al. (arXiv 2503.19206, ICML 2025): catastrophic overtraining — OLMo-1B pre-trained on 3T tokens fine-tunes *worse* than its 2.3T-token counterpart; progressive sensitivity formalized in a two-layer linear model.
- Lampinen (essay, 2026-05): scale plus pre-training substantially *reduce* interference and plasticity loss. Coheres with the above once "delayed" is separated from "defeated."

### 2.2 The "no mechanistic theory" claim is now half-outdated

- Joudaki et al. (arXiv 2510.00304): dynamical-systems definition — loss-of-plasticity manifolds (frozen-unit, cloned-unit / equitable partitions) proven invariant under GD, SGD, momentum, Adam; only symmetry-breaking perturbations (noisy SGD, dropout) escape. Simplicity bias and rank compression, which help static generalization, *steer networks onto* these manifolds.
- Wang et al. (arXiv 2605.09044): representation-rank and NTK-rank diagnostics can provably fail to predict trainability; proposes **optimization readiness** (gradient strength × gradient reliability), which lower-bounds one-step optimization gain and is cheap (a few forward/backward passes per checkpoint).
- Wang, Tissue et al. (arXiv 2505.07796, ICML 2025): closed-form continual-pre-training loss law — CPT loss decomposes into an LR-annealing term and a power-law distribution-shift term; replay ratio enters through the shift term; validated 106M–1.7B.
- Still missing (verified gaps): any plasticity measurement in SSM / linear-attention / GDN architectures, and any controlled LOB continual-learning study. **sigma-0 filling either gap is a first.**

### 2.3 The decisive design correction

The naive comparison — "continued pre-trained model vs fresh random init, matched compute; plasticity ABSENT if continued wins" — is broken. A pre-trained model beats scratch through representation transfer even while its plasticity decays, so that rule almost always reports ABSENT. The correct probe (following Zyphra):

- Compare **an early checkpoint θ_early and a late checkpoint θ_late of the same training run**, each given an identical fixed-budget adaptation on the same held-out slice, and compare validation-NLL AUC.
- PRESENT: AUC(θ_late) > AUC(θ_early) with CI excluding equality, plus at least one co-moving diagnostic (dormant fraction up, effective rank down, optimization readiness down).
- Scratch runs are kept only as a lower-bound reference; R(late/scratch) < 1 would be the far stronger "network nearly dead" statement, not the default expectation.

### 2.4 Reporting discipline

Every CPT or adaptation stage reports **both coordinates**: old-window validation NLL (stability / forgetting) and probe AUC or R_steps (plasticity). One coordinate alone can look healthy while the other collapses.

## 3. The experiment ladder

Ordered by cost; each step gates the next. No step assumes hardware or checkpoints that have not been inventoried in step 0.

### Step 0 — Inventory (no GPU) $\color{green}{\textsf{done, four open items remain}}$

List which sigma-0 checkpoints exist (run, size, step range, data window covered), which NASDAQ windows are tokenized and ready, and measured tokens/sec for the current model size. Output: a short table in `results/INVENTORY.md`. Everything below is parameterized by it.

### Step 1 — Diagnostics instrumentation $\color{green}{\textsf{done, not yet wired into training}}$

Land `code/plasticity_probes.py`: framework-agnostic implementations of

| Probe | Definition | Cost |
|---|---|---|
| dormant fraction | ReDo-style: unit-mean absolute activation, normalized by layer mean, below eps = 0.01 | free at eval |
| effective rank (Renyi-2) | er2(M) = (tr M)^2 / \|\|M\|\|_F^2 on the feature covariance | one small matmul |
| weight / gradient norms | global L2, non-embedding | free |
| optimization readiness | \|\|mean g\|\|^4 / mean \|\|g_i\|\|^2 over micro-batch gradients (strength × reliability; reconstruction of arXiv 2605.09044's descriptor) | K extra backward passes |
| top Hessian eigenvalue | power iteration over an injected HVP callable (Pearlmutter) | 10–20 HVPs, optional |

plus unit tests runnable on CPU. Wiring into the sigma-0 training loop is a follow-up commit; the target is that **every future long run logs these by default**, so plasticity evidence accumulates for free.

### Step 2 — Early-vs-late checkpoint probe $\color{green}{\textsf{weight probes done}}$ / **rest mis-specified, see `plan_drafts/01` §4**

- Take one existing long sigma-0 run; pick θ_early and θ_late checkpoints separated by as many tokens as the run allows.
- Fixed-budget adaptation of copies of both (identical tokens, batch, schedule, seeds) on a held-out later time slice; log validation NLL every fixed interval plus all Step-1 diagnostics.
- ≥ 5 seeds per group before any claim; 95% bootstrap CI on AUC difference and R_steps; declare a difference only if the CI excludes equality.
- Stress slices when data allows: the COVID window (2020-02..04) and the 2024-08 volatility spike (probe from a checkpoint trained through 2024-07 only — no leakage). **Superseded by Step 0**: tokenized data starts 2022-01 (`results/INVENTORY.md` §2), so the primary slice is 2024-08, the secondary 2025-04, and the base window 2022-01..2024-07.
- Decision: PRESENT / ABSENT / inconclusive per §2.3. If inconclusive, extend token budget before adding mechanisms.

### Step 3 — CPT pilot **superseded by `plan_drafts/04` §2–§3**

- D_cpt = (1−ρ)·D_new + ρ·D_old, old data stratified over time (not just the last month).
- Grid: peak LR ∈ {0.3, 0.5} × pre-training peak, ρ ∈ {0.05, 0.10, 0.25}; short pilots of 1–2B tokens.
- Rewarm is mandatory: short warmup (2–5% of stage steps) up to the chosen peak, then WSD or cosine with a **non-zero tail**; reset Adam moments at each stage boundary. A checkpoint whose LR annealed to ~0 and is continued without rewarm *looks* dead without being dead — that failure mode must not contaminate plasticity claims.
- Fit the arXiv 2505.07796 law's shift term from the pilots; choose the production (peak LR, ρ) to hit target new-window NLL while bounding the old-window NLL increase.
- Stage layout by calendar, not task ids: θ0 (base years) → CPT1 → CPT2 → ... with shock windows (2020-03, 2024-08) reported separately, never averaged into a calm-year replay bucket.
- Optional stabilizer, off by default: teacher-KL to the previous stage checkpoint, λ starting at 0, raised only if old-window NLL climbs beyond tolerance.

### Step 4 — Mitigations, in cost order (only if Step 2 finds decay)

1. Weight decay up (cheapest knob, first).
2. Spectral regularization toward sigma_max ≈ 1 (Lewandowski et al., ~14% step-time cost).
3. Local ReDo-style resets only where dormant fraction climbs; never periodic whole-network resets on a pre-trained model.
4. AltNet-style dual-network swaps and continual backprop reserved for a genuinely online setting.

### ~~Step 5 — Multi-size onset law~~ **excluded this round on evidence (`plan_drafts/01` §4)**

Cyclic year/regime schedule over 34M / 100M / 300M-class sigma-0 models, probe slice held out (a distinctive shock window), fit T = c · P^k for the LOB/SSM setting. This would be the first plasticity onset law for state-space / linear-attention models — the cell the verification table marks NOT FOUND. Only started after Step 2 produces a verdict-grade readout at one size.

## 4. Non-goals and traps (pre-registered)

- No per-request weight updates in serving; adaptation happens between stages.
- No regime-classifier-plus-EWC pipeline; drift signals (validation-NLL band, volatility state, plasticity probe) drive *stage-level* decisions only.
- Sliding-window fine-tuning on the most recent month is not CPT and will not be labeled as such.
- Fresh-vs-continued alone never decides a plasticity claim (§2.3).
- SEC Rule 612 half-penny tick: adopted 2024-09-18, compliance now first business day of Nov 2026 (extended by the Oct 2025 exemptive order) — slice boundaries must not assume the 2025 date.
- Nested Learning / HOPE, TTT layers, Titans: narrative-relevant, but not the backbone for the first measurement; no streaming-plasticity evidence yet.

## 5. Compute envelope (order of magnitude, to be refined by Step 0)

- Step 2 at the ~100M scale: two groups × 5 seeds × (1–2B tokens each) ≈ tens of short runs, each fitting on 1 node; embarrassingly parallel; attach-first.
- Step 3 pilots: 6 grid cells × 1–2B tokens at one size.
- Step 5 is the only multi-thousand-GPU-hour item and is explicitly deferred.

## 6. Deliverables checklist

- [x] PLAN.md (this file)
- [x] `code/plasticity_probes.py` + `code/test_plasticity_probes.py` (Step 1, this PR; 13 CPU tests pass)
- [x] `results/INVENTORY.md` (Step 0; partial — 4 open items listed there, checkpoint roots need the user)
- [x] $\color{green}{\textsf{Failure-pool prerequisites measured}}$ — `code/failure_pool_reliability.py`, 12 tests, `results/failure_pool_reliability.json`, `failure_pool_reliability.ipynb` (F1–F4)
- [x] $\color{green}{\textsf{Five facet drafts}}$ — `plan_drafts/01`–`05`
- [x] $\color{green}{\textsf{M0 checkpoint inventory}}$ — F10, which settles X4
- [x] $\color{green}{\textsf{Audit of commit e8425cb1 reused and answered}}$ — `plan_drafts/06`; four defects re-derived independently, all four fixed, tests 12 → 17 (§0.5)
- [x] $\color{green}{\textsf{17 selftrain steps converted to tokens}}$ — §0.7
- [x] $\color{green}{\textsf{P1/P2/P6 acceptance written as exit-code conditions}}$ — `plan_drafts/06` §5
- [ ] **Five independent adversarial reviews** (`plan_drafts/_REVIEW_BRIEF.md`) — blocked on session quota; relaunch **one at a time**, reading list extended to `01`, `03`, `06`
- [ ] ~~M1 on the existing archive~~ **impossible — no `data_cond/`** (§0.6); folded into the G2 regeneration
- [x] $\color{green}{\textsf{Items 2, 3, 4 measured on one convention}}$ — `results/RESULTS_20260905.md`, verdicts in §0.10
- [x] $\color{green}{\textsf{Item 5: zero-row report now fails}}$ — `fidelity.py` returns 2; **not yet committed to sigma-0** (separate repo, junming identity, PR#60 stack)
- [ ] Item 6: wandb config for `j5705912`, to confirm `num_devices = 1`
- [ ] Item 7: fold `D1` §2, `D3` §8/§10, `D4` into the merged plan
- [ ] Item 8: P1/P2/P6 acceptance **scripts** (the conditions are written; the executable checks are not)
- [ ] The three P-blockers before any generation: rollout manifest (P1), frozen hashed context set (P2), inode write plan (P6) — `plan_drafts/03` §1
- [ ] M4: arm-level repeat with several training seeds — the decisive rung (F7)
- [ ] Probe wiring into the sigma-0 training loop (follow-up)
- [ ] Step 2 readout: AUC(θ_late) vs AUC(θ_early), CI, diagnostics — needs the same-age null pair `plan_drafts/01` §2.6
