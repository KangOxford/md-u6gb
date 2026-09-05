# PLAN — CRPS return-alignment study: what the audit changed, and what remains measurable

> Task dir: `/lus/lfs1aip2/projects/public/u6gb/tasks/crps_analysis_20260905`
> Study root: `/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z`
> Worktree: `/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/crps-return-alignment-20260808`
> Branch: `feat/midtrain-return-alignment-evidence-20260818` (the base of sigma-0 PR#60)
> Evidence archive: `./all_findings.txt` — 236 findings from 6 adversarial + 7 solution + 11 planning agents
> Convention: $\color{green}{\textsf{done}}$ for completed, ~~strikethrough~~ for claims the evidence overturned.

## 中文速览

- 这条线原本的结论是「第四轮微调比第三轮更差」（CRPS +2.92%、R 效应 +0.0904、符号检验 p=0.0078）。
- 对抗审计推翻的不是某个数，是**推断的作用域**：效应点估计稳健（12 次交叉重配对 +0.0806 ± 0.0087，从不变号），
  但用来判它的误差棒**测错了对象**——它测的是同一个检查点重新生成 rollout 的散布，
  而结论要的是**换一次训练**的散布。后者实测 4.71%，**大于 +2.92% 的效应本身**。
- p=0.0078 不是测量值，是 8 个标的双边符号检验的**下限** 2/2⁸——这个设计能表达的最大证据量。
- 有两条被我裁掉的对抗发现：cosine 学习率混淆（不存在，见 §2.1）、以及 round-4 复制能收窄结论（不能，见 §3.1）。
- 现在唯一能推进的测量是**把 round-3 那侧的复制轨迹跑到 step 1200**，因为 SE 与自由度全被 n₃ 钉住。

---

## 1. What the study claimed

| # | Claim | Where it lives |
|---|---|---|
| C1 | Round 4 is worse than round 3 on return alignment: `dR = +0.0904`, `t = +3.54`, 8/8 tickers, sign-flip `p = 0.0078` | `/home/u6gb/kangli.u6gb/variance_ladder.json` → `the_claim.on_R` |
| C2 | On the endpoint-invariant transform `d|R-1| = +0.0336`, `t = 1.08`, 5/8, `p = 0.3047` | same file, `the_claim.on_abs_R_minus_1` |
| C3 | Fair CRPS panel: round 4 is `+2.92%` worse, sign-flip `p = 0.0938` | `crps_panel.jsonl`, 8 tickers |
| C4 | A "peak" at step 1200 survives selection correction, `P = 0.19` | `build_nb.py` analysis |
| C5 | The registered null band is `0.0195` | `fix_attribution.json` |

## 2. What the audit overturned, and what it left standing

### 2.1 ~~Cosine-anneal confound: the two run groups sit at 8.9x different LR at step 1200~~ — REFUTED BY ME, 2026-09-05

`adv_stats` and `sol_peak` both reported this as a serious confound; `sol_peak` even "confirmed" the
schedule shape `0.5(1+cos(pi s/S))` and the 8.9385 ratio. It does not exist.

The trainer builds its optimizer at
`/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/crps-return-alignment-20260808/run/mid_training/wmle_full_ft.py:186-187`:

```python
tx = optax.chain(optax.clip_by_global_norm(args.clip),
                 optax.adamw(args.lr, weight_decay=0.0))
```

`args.lr` is a scalar and appears exactly once in the file. There is no schedule object anywhere.
Four independent confirmations:

| Evidence | Reading |
|---|---|
| Source: `optax.adamw(args.lr, ...)` | scalar LR, no schedule |
| Runtime banner in every log | `[Optimizer] Using inject_hyperparams (legacy scalar LR)` |
| Parameter travel per step (`sol_design`) | 1.6137e-3 (150→300) vs 1.5968e-3 (1350→1500), ratio 1.011 — flat |
| s30 six-writer natural experiment (`sol_design`) | same seed, `max_step` 4800/1500/1050 print identical losses at the same step |

The `cosine_anneal=True` field lives in **checkpoint metadata** and never reaches the optimizer. This is
the house failure mode "a knob that is set, is logged, and never reaches the code", and two adversarial
agents were taken in by it. Recorded because the mistake is more instructive than the correction.

### 2.2 The effect is real; the error bar is measuring the wrong thing $\color{green}{\textsf{established}}$

The point estimate survives replication: 12 cross-replicate pairings of the same round4−round3 contrast
give `+0.0806 ± 0.0087` and never change sign (`adv_stats`). The failure is not a measurement artefact.

What fails is the **scope of the inference**. Every registered null in the study is a *generation*-realisation
null: two rollout regenerations from the **same** checkpoint (`adv_design`; `r3a`/`r3b`, null mean −0.0104,
per-ticker sd 0.0376). The claim being judged is a *training*-realisation claim. `sol_crps` measured the
right quantity on 12 replicate fine-tuning runs:

| Quantity | Value | 95% CI |
|---|---|---|
| Run-to-run sd of the panel cell statistic | **4.71%** | [1.78, 6.20] |
| The effect it is used to establish | **+2.92%** | — |
| Corrected panel se | 5.47% (was 1.44%) | — |
| Corrected panel result | `z = +0.53`, **`p = 0.594`** | CI [−7.81%, +13.64%] |

**The noise floor of one training run is larger than the effect.**

### 2.3 The p-value is the floor of its own test $\color{green}{\textsf{established}}$

Exact two-sided sign-flip on n=8 tickers has minimum attainable `p = 2/2^8 = 0.0078125`. All three
contrasts report exactly that (`adv_design`, `adv_stats`). It says "all 8 agreed in sign", which is the
most the design can express — it is not a measurement of effect size, and no Bonferroni correction over
the study's own 42-comparison `family.json` can leave it alive, because `alpha/m` drops below the floor
for `m >= 7`.

### 2.4 Other confirmed defects, with the corrected value

| Defect | Published | Corrected | Source |
|---|---|---|---|
| Peak refutation uses max-of-10-normals, not Grubbs | `P = 0.19` | `P = 0.0970` (2x) | `sol_peak` |
| Same, on the second arm (unifw, n=6) | `P = 0.1385` | `P = 0.0053` (26x) | `sol_peak` |
| Null band for a 2-seed vs 4-seed contrast | `0.0195` | `0.0195*sqrt(1.5) = 0.0238` | `adv_math`, `sol_inference` |
| Variance-ladder rungs mix LEVEL and CONTRAST sds | ratio 2.886 | commensurable 4.082 | `sol_inference` |
| Welch–Satterthwaite df from `n_eff = 2.365` | `1.37` | `2.301` | `adv_math`, `sol_inference` |
| maxT FWER (claimed 0.042) | 0.042 | measured 0.076–0.082 | `sol_inference` |
| Study's own module on the headline | — | `p = 0.357`, checkpoint position is 94.3% of variance | `adv_stats` |
| Fair CRPS over independent runs | unbiased | biased by `(Wbar−Dbar)/2`, flat in K; on this data −0.764% [−1.790,+0.247] | `sol_crps` |

### 2.5 Provenance defects that make numbers unquotable

- **Five** circulating values for R at multi4 step 1200: 0.9153 / 0.9578 / 0.9610 / 0.9704 / 0.9781 (`adv_results`).
- Three stores disagree on every one of 16 overlapping cells; mean 3-way range 0.0566 = **62.6%** of the headline effect (`adv_results`, `sol_notebook`).
- The code that produced every 2026-09-04 number is **uncommitted** — the scorer and trainer are both dirty in the worktree (`adv_results`).
- `one_run_many_numbers.ipynb` **contains zero executable code**: 8/8 code cells are a single `# name` comment with a pre-baked base64 PNG and `execution_count` hard-set to 1 (`adv_results`).

### 2.6 Design limits that no amount of compute removes

- **No held-out context exists.** All 80 generation seeds cover the identical 600 conditioning contexts, byte-identical `ids` arrays (`adv_design`, `sol_notebook`).
- `jax_seed` is pinned at 42 in every checkpoint, so all trajectories share one initialisation; the measurable `sigma_traj` is the data-order component only, a **lower bound** (`adv_stats`).
- `p_dropout = 0.0`, so `--train-seed` varies exactly one thing: the data-order permutation (`adv_math`, `adv_design`).
- One epoch is exactly 4800 items, so `--max-step 4800` is the whole epoch and every replicate has then seen the identical multiset (`adv_design`).

## 3. What is still worth measuring

### 3.1 ~~Add more round-4 replicates~~ — REJECTED, it buys nothing

`adv_stats` measured the sensitivity directly: 18 → 30 round-4 runs cuts the between-round SE by **4.4%**,
30 → infinity cuts it by **7.4%**, and Welch df stays at **5.42**. `adv_design` gives the same conclusion
analytically: with `sigma_A` bounded below by the within-round-3 checkpoint sd 0.0548, the SE floor is
`sqrt(0.0133^2 + 0.0548^2) = 0.0564`. **The uncertainty is pinned by n₃, not n₄.**

### 3.2 THE decisive measurement: round-3 replicates to step 1200

`adv_stats`: *"Round-3 has NO recorded metric at step 1200: no `ft_log.json` exists in any `wm_ft_traj3_*`
directory that reaches 1200."*

### ~~Verified independently, 2026-09-05, and it is worse than stated: 0 of 14 round-3 replicates reach step 1200.~~ — MY OWN MISREADING, corrected same day

I read the **logs** in `/home/u6gb/kangli.u6gb/traj3_s*.log`, whose tails stop at 650-750, and concluded the
runs had died. They had not. `adv_code` had already found the cause and I did not apply it: the RUNID fix
landed on the checkpoint path but **not on the log path**, so one log file holds several run instances and
its tail is not that run's last step. The state itself says otherwise --
`wm_ft_traj3_s1/latest_checkpoint.json` records step 1350 synced at `2026-09-05T00:07:13Z`, later than
anything in the log.

The general form: I read a derived artifact instead of the state, on a question the state answers directly.

### 3.2 The real imbalance, read from breadcrumbs $\color{green}{\textsf{measured}}$

| Round | Replicates with a `step_1200` checkpoint |
|---|---|
| Round 3 (`wm_ft_traj3_s*`) | **5** (s1, s2, s3, s4, s6) |
| Round 4 (`wm_ft_traj_s*`) | **30** |

This is exactly the `n3 = 5` that `adv_stats` found pinning the SE and the Welch df, now confirmed from the
filesystem rather than from a summary. Of the other nine round-3 replicates: s5 reached 1050, s22 is live at
600, s8 stopped at 450, and six (s10, s13, s19, s20, s21, s30) never took a step -- GPU OOM, because the
launcher requested `--gres=gpu:1` and Slurm always hands out logical device 0, which was occupied.

**Action taken 2026-09-05** $\color{green}{\textsf{launched}}$: eight further round-3 replicates
(`--train-seed` 40..47, `--max-step 1500`) on cards verified empty **inside** the job step, capped at three
per node because host RAM, not GPU, is the binding constraint (`adv_fleet_review`: four runs per node is the
configuration that host-OOM-killed s1). Runner:
`/lus/lfs1aip2/projects/public/u6gb/tasks/crps_analysis_20260905/r3_replicate.sh`. This takes n3 from 5 to 13.

Expected effect, using `adv_stats`'s own sensitivity: n3 = 5 -> 13 is the only move that shifts the between-round
SE and df at all; n4 = 18 -> 30 moved the SE by 4.4% and left df at 5.42.

### 3.3 Analysis-only items (no GPU)

- Recompute the panel with the run-to-run se from `sol_crps` (done: `p = 0.594`).
- Recompute the peak with exact Grubbs on both arms (done: 0.0970 / 0.0053).
- Rebuild the notebook so that its code actually executes (§2.5).

## 4. Order of work

1. $\color{green}{\textsf{done}}$ Consolidate 236 findings; settle the cosine contradiction against source.
2. $\color{green}{\textsf{done}}$ Diagnose why 0/14 round-3 replicates reach step 1200.
3. Launch round-3 replicates to step 1200 on verified-empty cards, long allocation.
4. Build an executable notebook of the corrected mathematics, statistics, and results.
5. Execute it, render HTML with embedded images, commit and push both.

## 5. Standing rules for this line

- The unit of replication for any training claim is a **training run**, never a generation seed.
- Any quantile or sign-flip statistic is printed next to its **event count** and its **attainable floor**.
- No number enters a heading, is bolded, or is called "best" until n has stopped growing.
- Every scored cell records `k_actual` from the estimator, not from a shell directory count.

---

## 6. Progress log

### 2026-09-05 03:5x -- eight round-3 replicates running, and a copy-back defect I wrote myself

| seed | node | step | step_1050 on Lustre | step_1200 on Lustre |
|---|---|---|---|---|
| s40 | nid010851 | 850  | no  | no |
| s41 | nid010851 | 1100 | yes | no |
| s42 | nid010234 | 1200 | yes | **yes** |
| s43 | nid010308 | 1150 | yes | no |
| s44 | nid010308 | 1150 | yes | no |
| s45 | nid010488 | 1100 | yes | no |
| s46 | nid010488 | 1100 | yes | no |
| s47 | nid010488 | 1100 | yes | no |

All eight alive. Throughput 5.0-6.6 s/it; s40 is one node-hour behind the rest.

**The defect.** `wmle_full_ft.py` saves to `"${--out}_step<N>"`, appending the suffix to the
`--out` path rather than creating a child of it, so every checkpoint is a **sibling** of `$LOCAL`.
My copy-back copied `"$LOCAL"/.` and so persisted exactly one file, the 260-byte
`ft_progress.json`, while reporting **zero errors** -- `cp` genuinely succeeded on the directory it
was given.

This is the same shape as the defect the notebook audits in someone else's code
(`adv_code`: *"sync_once line 106 makes a total copy-back failure indistinguishable from success"*).
I had read that finding, written a copy-back with explicit error reporting to avoid it, and still
shipped a check that passes because it measures the wrong thing. Reporting errors is not the same
as reporting the quantity that matters; the guard now counts what is **on the destination** and
warns when that count is zero.

Fixed in `r3_replicate.sh`. Recovery for the runs already in flight is
`harvest_r3.sh` (one `srun --overlap` per node, read-only on the trainer's side, never removes
anything), driven by `harvest_loop.sh` every 5 minutes until no run is left.

### Next condition

When all eight have `step_1200` on Lustre, score those checkpoints through the same cell pipeline
as the existing five round-3 replicates, recompute the between-round contrast at n3 = 13, and
update the notebook's figure 7 and the ledger. Until then the notebook stands as published; its
n3 = 5 panel is labelled with the count it was computed from.

**Risk to watch**: node-local storage is TMPFS and the checkpoints vanish with the allocation.
6317365 has ~19 h left and 6324130 ~23 h, against ~30 min of remaining training, so the margin is
wide -- but the harvest loop exists because that margin is not the thing that failed last time.

### 2026-09-05 05:0x -- all eight finished; persistence verified; n3 = 13 trained, 5 scored

**How the eight ended.** They reached step 1500 and wrote `complete: true` with
`last_saved_step: 1500`; step dirs are 9 minutes apart, matching 150 steps at 3.7 s/it. The wrapper
shell then exited 1 at 04:25, twelve minutes after training finished, because **I edited
`r3_replicate.sh` in place while it was running** and bash reads a script incrementally by byte
offset. The corruption cost nothing here, but only because it landed after the work was done.
Rule: never edit a running shell script; copy to a new name and launch that.

**Integrity, not exit status.** `cp` reported success while persisting one 260-byte file, so the
judgement is made on restores:

| Check | Result |
|---|---|
| Full restore, s40 / s47 / reference `wm_ft_traj3_s1` | 386 arrays, 159,374,987 elements, all finite &mdash; **identical across all three** |
| Total bytes, eight `step_1200` dirs | 594,031,383 .. 594,118,165 B against reference 594,092,900 B (<= 0.02%) |
| File counts 20-31 vs reference 28 | expected: OCDBT shard counts vary with write parallelism; `adv_code`'s clean range is 20-43, and the chimera signature is a **3x** size, absent here |

**What the five already-scored replicates say**, training seed as the independent unit, ticker as
the pairing basis, reference `multi3`:

| Metric | mean | sd | t (df 4) | sign agreement | sign-flip p | floor 2/2^5 |
|---|---|---|---|---|---|---|
| R (sd ratio) | **-0.0679** | 0.0330 | **-4.60** | **5/5 negative** | 0.0625 | **0.0625** |
| fair CRPS (%) | +1.68 | 1.81 | +2.07 | 4/5 | 0.1875 | 0.0625 |
| qL1 | +0.0003 | 0.0111 | +0.07 | 2/5 | 1.0000 | 0.0625 |

**Verdict: underpowered and confounded. Neither supported nor refuted.**

1. At n = 5 the attainable floor 0.0625 is above alpha = 0.05, so unanimity plus t = -4.60 still
   cannot reject. This is `sol_peak`'s "registered decision rule is unattainable at the n it
   headlines", now demonstrated on the data rather than argued.
2. The contrast reads replicates at step 1200 against `multi3` at its endpoint, so it carries the
   checkpoint-position term: one save interval moves R by up to 0.1647, **2.4x** the 0.0679 contrast.
3. Round 1 and the parent cannot enter the comparison. `parent_multi2` is scored on META only and
   `wm_ft_multi` on no ticker at all. **That is a gap in what was scored, not in compute.**

**K and calibration-member dependency.** K = 2 in **all 239** scored records, so no K-dependence is
estimable from them. Raising K would not license the comparison either: the fair-CRPS bias against
a single run is **flat in K** (-0.852%, CI [-1.734, +0.018]), and every run reuses generation seeds
97901/97902, which couples them -- cross-run correlation +0.389 when the seed is shared against
+0.139 when it is not, and the coupled Dbar (13.70e-5) falls **below** Wbar (14.93e-5), which
independent draws cannot do. A larger K is not evidence that the scoring is right.

### Next condition (unchanged in kind, now precise)

64 cells missing: 8 new seeds x 8 tickers at step 1200, plus `parent_multi2` and `wm_ft_multi` on
the seven tickers they lack. **0 idle cards at 05:0x**, so this waits on GPUs rather than on a
decision. When cards free: score those cells with K = 2 and generation seeds 97901/97902 to stay
commensurable with the existing 39, then recompute at n3 = 13 where the floor is 2/2^13 = 0.00024.
Also worth scoring `multi3` at step 1200 so the contrast stops carrying the checkpoint-position term.

### 2026-09-05 07:xx -- three of my own claims withdrawn; verdict table; 72 cells queued

**Withdrawn, with what replaces each.**

| Claim I made | Status | What is true instead |
|---|---|---|
| Coupled `Dbar < Wbar` is impossible for independent draws | **wrong** | Equal-distribution runs give `E|Xi-Xj| = E|Xi-Xi'|`, so the difference has expectation zero. A 2000-design simulation of this exact shape lands below zero in ~50% of them. The coupling is real; the evidence is the seed-pairing contrast: same-seed `-2.63e-5` against different-seed `+0.16e-5`, intervals far apart |
| Shared generation seeds invalidate the scoring | **conflated three things** | Within a cell the `K=2` members are two generation seeds of one checkpoint, exchangeable given it, so fair CRPS is unbiased **for that run**. Common random numbers bias only estimators assuming cross-run independence (a pooled `Dbar`). Their variance reduction on paired comparisons measures **0.4%** here. Evaluation Monte Carlo error is separate: **+/-7.5%** on `Wbar` |
| The fair-CRPS bias is flat in `K` | **algebra, reported as measurement** | `E[fair_K] - CRPS(mixture) = (Wbar-Dbar)/(2K)`; `E[fair_K] - mean single-run CRPS = (Wbar-Dbar)/2`. Which applies depends on the estimand. All 239 records are at `K=2`, so no `K`-dependence is estimable, and the magnitude's CI includes zero |

**Exchangeability, checked before pooling to n3 = 13.** Same parent (`wm_ft_multi2` at 69378), same
weights/prefix (`v5m3`), same training item set (`seed0_sha1 0f14669f2a4d`), same 4800 items, same
`max_step` 1500, same 78,539,423 parameters, same optimizer banner. **Only `order_sha1` differs**,
which is the replication unit. Two limits travel with any `n3 = 13` result: `jax_seed` is pinned at
42 so all thirteen share one initialisation (data-order component only, a lower bound), and both
sides must be read at step 1200 -- which is why `multi3` is being scored at 1200 rather than reused
at `final`.

**Verdict counts** (notebook section 11): 1 supported, 7 refuted, 2 underpowered, 1 algebra-only,
1 not yet estimable.

**Scoring queue**: 72 cells in `cell_queue.txt`, highest value first -- `multi3` at step 1200 on 8
tickers (removes the checkpoint-position term), then the 8 replicates on 8 tickers (n3 5 -> 13).
`drain_cells.sh` re-reads gtop each round, skips any cell whose `score.json` exists, so it is
idempotent and restartable. `score_cell.sh` passes `--assert-k 2 --assert-ckpt-step 69378
--assert-seeds 97901,97902`, so K comes from the estimator rather than a shell directory count.
Not queued: `parent_multi2` on its seven missing tickers and `wm_ft_multi` on all eight -- needed
for the round-1 and parent comparisons, but neither can change the verdicts above.

### 2026-09-05 08:xx -- read-back found the artifact half-updated

The repo, the artifact and the PR description are three separate surfaces and updating one does not
update the others. Read back, not assumed:

| Surface | State before this check | Action |
|---|---|---|
| Remote notebook | blob `6205dcfa` == local, current | none |
| **PR #76 body** | **still the pre-correction version** -- "Ten code cells, seven figures, 545 KB", no mention of any withdrawal | rewritten to the verdict table + the three withdrawals |
| **Artifact page** | ledger rows and commit hash current, but the provenance strip still read "10 code cells, 0 errors, 7 figures" and the note still read "Nine quantities moved" against 17 rows | both fixed, republished to the same URL |

`WebFetch` on the artifact caches for 15 minutes per URL and returns the whole page, so the final
check was string-level on the published file: `12 code cells`, `Seventeen quantities moved`,
commit `54cb5846`, three `withdrawn` rows all present; the stale `10 code cells` absent.

### Scoring queue

`score_v5_primary.py` defaults `--baseline base`, which is not among the scored arms when a cell
passes only `a=`, so the first cell exited 7 **after** computing its numbers. Fixed by passing
`--baseline a`; the rollouts survived on node-local storage, so `rescore_cell.sh` scored that cell
without regenerating anything.

First cell: `multi3` at step 1200, AMD -- `K = 2/2` asserted by the estimator, 500 contexts, 20 days,
CRPS `1.0540e-04`, qL1 `0.1932`, sd_ratio `0.7446`.

One drainer is running (PID confirmed by exact argv match, not by cmdline substring -- that trap
catches the wrapping shell too). 1 of 72 scored.

### The minimum pairing, before any verdict is updated

Fixed by construction, not by assumption:

| Element | Held fixed how |
|---|---|
| Checkpoint | step 1200 on **both** sides. `multi3` is being scored at 1200 rather than reused at `final`, because the endpoint comparison imports the checkpoint-position term |
| Independent unit | the **training seed**; 13 of them, shown exchangeable (same parent, weights, item set `seed0_sha1 0f14669f2a4d`, budget, parameter count) |
| Generation unit | `K = 2`, seeds 97901/97902 in **every** cell, so the pairing basis is the common context and the generation draw is matched across arms |
| Pairing basis | the ticker; 8 of them, the same 500 contexts and 20 days in each cell |

Nothing is judged until that set is complete. A green CI and a synchronised artifact are
housekeeping; neither is evidence about round 3.

### 2026-09-05 08:2x -- the drainer's probe was failing silently for eleven rounds

Eleven consecutive rounds logged `free cards: 0` while the cluster had 19-26 idle cards. The probe
was not fluctuating; it was returning nothing.

**Cause: terminal width.** `gtop` lays out to the terminal and emits **no GH200 lines at all** below
roughly 200 columns. The drainer ran in an 80-column tmux window; my own checks ran through a
tool with no TTY and therefore a wide default. Measured:

| COLUMNS | idle cards parsed |
|---|---|
| 80 | 0 |
| 100 | 0 |
| 120 | 0 |
| 200 | **25** |

Ruled out along the way: `gtop` missing from the drainer's `PATH` (it resolves), and the 50
inherited `SLURM_*` variables (gtop returns 25-26 with them set, unset, or partially set).

**This violates a rule already written down.** CLAUDE.md 4.-0.5 requirement 1 says a failed probe
must not be reported as zero idle cards, because `grep -c` on a crashed `gtop` returns 0 and hides
exactly the waste it is meant to surface. I wrote the drainer knowing that rule and still logged the
failure as a measurement.

Fixed in `drain_cells_v2.sh` (a **new file** -- editing a running shell script in place is what
corrupted the eight training wrappers earlier today):

- `export COLUMNS=200 LINES=200` before any probe;
- the raw `gtop` output is kept, and a round with no `GH200` line at all logs
  `PROBE FAILED: gtop emitted no GH200 lines (N bytes)` and skips rather than reporting zero.

The old drainer was stopped by exact PID -- not by `cmdline` substring, which also matches the
wrapping shell. v2's first round read 25 free cards and launched 11 cells.
