# Section 1 — The decisive experiment

Written 2026-09-05, before any new data exists. Every rate and cost below is **measured**
from checkpoint mtimes and `sacct`, not estimated; the derivation is shown so each can be
re-checked.

## 0. What makes it decisive

Every replicate in this study so far is a pseudo-replicate of **one** training trajectory per
arm. Tickers, days, contexts, generation seeds and checkpoint choice are all nested inside
that single run, so none of them can speak to the question actually being asked — would a
second fine-tuning run land in the same place. The variance ladder makes the gap explicit:

| rung | sd of the 8-ticker pooled R | status |
|---|---|---|
| registered null (checkpoint vs itself) | 0.0195 | **every published verdict used this** |
| generation seed, same checkpoint | 0.0368 | measured |
| where you stop on one trajectory | **0.0562** | measured; 2.89× the registered null |
| **which trajectory you are on** | **unmeasured** | one run per arm |

The decisive experiment is therefore the cheapest possible measurement of the missing rung:
**train k independent trajectories per arm and read the between-arm contrast against their
spread.**

The enabling change is already applied. `wmle_full_ft.py` had exactly two randomness sources,
both hardcoded literals (`jax.random.key(0)` for dropout, `onp.random.default_rng(0)` for data
order); `--train-seeds`/`--hold-seeds` only named rollout dump directories, and parameters are
restored rather than initialised so `jax_seed` was dead. The patch at
`/home/u6gb/kangli.u6gb/mt_seed_repl/apply_train_seed_patch.py` (69 insertions) routes
`_TRAIN_SEED` to line 200 and `args.train_seed` to line 332, with a self-proving probe at
lines 340-341 that compares the permutation under the given seed against seed 0. **Without
that patch, k trajectories would be k identical trajectories.**

## 1. Power, computed not asserted

Paired one-sample t on k independent trajectories, two-sided α = 0.05, 80% power. The relevant
sd is the **trajectory** rung, which has never been measured; the best available lower bound is
the checkpoint-position rung 0.0562, which is one of its components. **Every k below is
therefore a floor on what will actually be needed.**

| effect | sd 0.0562 (trajectory floor) | sd 0.0195 (the null that was wrongly used) |
|---|---|---|
| R +0.09 (the published headline) | **k = 6** | k = 3 |
| \|R−1\| +0.034 (the same result, correct transform) | **k = 24** | k = 5 |
| \|R−1\| +0.0726 (round 4 vs round 3) | **k = 7** | k = 3 |

Smallest effect detectable at 80% power, sd 0.0562:

| k | smallest detectable effect |
|---|---|
| 3 | 0.1834 |
| 4 | 0.1196 |
| 6 | 0.0806 |
| 8 | 0.0650 |
| 24 | 0.034 |

### What this table is, and what it is not

**It is a design estimate, not an authorisation and not a guarantee.** Three limits, all of
which must travel with the numbers:

1. **The variance is estimated, not known.** `sd = 0.0562` is the *checkpoint-position* rung
   measured on ten checkpoints of one trajectory. The trajectory rung it stands in for has
   never been measured. A power figure computed at an estimated sd is not an unconditional
   80% — the true power is a random quantity around it, and with the sd itself resting on a
   handful of points the interval on `k` is wide. **The correct reading is "k = 24 is the
   order of magnitude", not "24 runs will suffice".**
2. **The estimate is a floor.** The trajectory rung contains the checkpoint rung, so
   `sd_trajectory ≥ 0.0562` and every `k` above can only grow once it is measured.
3. **Nothing here authorises spending 354 node-h.** This section prices the experiment; it
   does not start it. The only thing the table licenses is the k = 6 staging step, whose
   purpose is to *measure* `s_trajectory` rather than to test any hypothesis, and even that
   is a proposal for the user to approve, not a decision taken here.

**The single most important number in this section is k = 24.** The quantity the study cares
about, |R−1|, at the effect size the study actually reported, needs twenty-four independent
fine-tuning runs per arm to reach conventional power — and that is a floor, because the
trajectory rung is at least as large as the checkpoint rung nested inside it. Reading the same
effect against the registered null gives k = 5, which is why every published interval looked
comfortable.

## 2. Cost, measured from the logs

Rates, derived from checkpoint mtimes under
`/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z/ckpt/`:

| quantity | measurement | value |
|---|---|---|
| seconds per training step | `wm_ft_multi3`, 32 checkpoints, steps 150→4800, span 4.92 h | **3.81 s** |
| same, cross-check | `wm_ft_multi4` clean segment, steps 450→4800 on 09-02, 4.33 h | 3.58 s |
| one trajectory, 4800 steps, 1 node × 4 GPU | 4800 × 3.81 s | **5.08 node-h = 20.3 GPU-h** |
| independent cross-check | `sacct` `unifw-train` COMPLETED | **04:18:04 = 4.30 h** |
| one evaluation cell (1 checkpoint × 1 ticker, K = 4 seeds) | `sacct` `crps-*` COMPLETED: 17:12, 17:06, 17:28 | **17.2 min** |

`wm_ft_multi4`'s checkpoint span is 157.25 h and must **not** be used: steps 150→300 land on
08-27 and step 450 on 09-02, so the run was interrupted and resumed six days later. Its clean
segment is what is quoted above. This is the kind of number that silently poisons a cost model.

Totals, two arms, one prespecified checkpoint per trajectory, 8 tickers:

| k per arm | trajectories | train | eval | total | GPU-h |
|---|---|---|---|---|---|
| 3 | 6 | 30.5 node-h | 48 cells, 13.8 node-h | 44.2 node-h | 177 |
| 4 | 8 | 40.6 | 64 cells, 18.3 | 59.0 | 236 |
| **6** | 12 | 60.9 | 96 cells, 27.5 | **88.5** | **354** |
| 8 | 16 | 81.3 | 128 cells, 36.7 | 118.0 | 472 |
| **24** | 48 | 243.8 | 384 cells, 110.1 | **353.9** | **1415** |

On four concurrent nodes: k = 6 is **22 h wall**, under four 6-hour allocations. k = 24 is
**88 h wall**, about fifteen allocations.

**The honest reading of this table:** k = 24 is affordable in GPU-hours and expensive in
calendar time and in babysitting. k = 6 is four allocations and detects 0.0806, which is larger
than every effect this study has claimed on |R−1|. So k = 6 **cannot** confirm the hypothesis;
it can only measure the trajectory rung and thereby tell us whether k = 24 is worth starting.
That staging is the recommendation.

## 3. Prespecification

Fixed now, before any data exists.

**Primary endpoint.** Δ|R − 1|, the change in distance from perfect calibration, averaged over
the eight tickers. Not R: on the published headline R gives +0.0904, 8/8, p = 0.0078 while
|R − 1| gives +0.0336, 5/8, p = 0.3047, because R is not monotone in quality when the target
is interior. **R = 1 is perfect, so overshoot is as wrong as undershoot.**

**Unit of replication.** The training trajectory, n = k per arm. Tickers are a within-trajectory
axis and are averaged, not counted as replicates. Effective n of the equal-weight 8-ticker mean
is 2.365 (Kish) because GOOG carries 62.8% of the null variance, and that number is reported
next to every ticker-level aggregate.

**Test.** Paired t on the k trajectory-level contrasts, plus a joint sign-flip maxT over the
family below. The sign-flip test on n = 8 has a smallest attainable p of 2/2⁸ = 0.0078125, so
**no family of ≥ 7 sign-flip tests on n = 8 can be Bonferroni-corrected to FWER 0.05** — α/m
falls below the floor. At the trajectory level with k ≤ 8 the floor is 2/2^k, which is 0.0078
at k = 8 and 0.0625 at k = 4; at k ≤ 4 the sign-flip test **cannot reach 0.05 at all** and only
the t is available.

**Multiplicity family, declared now:** three members — new arm vs round-3 frontier, new arm vs
round 4, new arm vs its uniform-weight control — one endpoint each, one prespecified checkpoint
each. Everything else is exploratory and carries no adjusted p.

**Checkpoint steps, fixed before data.** Step 4800 (the end of training) for every trajectory.
**Step 1200 is excluded as a confirmatory readout**: it was selected as the maximum over ten
checkpoints on the same seeds and contexts it would be scored on, so it cannot confirm itself.
`P(max of 10 noise ≥ observed) = 0.19` for multi4 and 0.14 for unifw, and the unpublished steps
1050 and 1350 take R through 0.806 → 0.915 → 0.819, so the whole excursion reverses in 150
steps. Step 1200 may enter only as the hypothesis to be tested at a fixed step on new
trajectories.

**Stopping rule.** Run k = 6 per arm to completion, then stop and read `s_trajectory` **only**.
No between-arm p-value is computed at that point. Continue to k = 24 only if
`s_trajectory ≤ 0.08`; if it exceeds 0.08 the comparison is unresolvable at any affordable k
and the study reports that instead.

**What falsifies the hypothesis.** Δ|R − 1| ≥ 0 at step 4800; or the uniform-weight control
moving at least as far (already measured at −0.0969 against round 4's −0.0808, so continued
fine-tuning accounts for 119.9% of the exit and the weights term −0.0161 sits inside the
±0.0389 null band); or the maxT-adjusted p exceeding 0.05 using the *measured* size of the
test, not its nominal size (nominal 0.0078 is truly 0.0146, an inflation of ×1.87).

**What a null looks like.** Δ|R − 1| within ±0.0389 of zero with a non-unanimous sign pattern
across trajectories and an adjusted p above 0.05. Written down now so it cannot be
re-described later.

## 4. Sequencing under allocations that expire without warning

Allocations are 5–8 h and a trajectory is 4.3–5.1 h, so **a trajectory does not reliably fit in
one allocation**. Non-negotiable consequences:

1. **Checkpoint and resume.** The patch adds `--max-step`; runs are launched as segments with
   `--start_step` reading the last checkpoint. A segment that dies costs at most the interval
   since the last checkpoint.
2. **Results land on shared storage as they are produced, never at the end.** Node-local
   `/local/user/$(id -u)/` dies with the allocation — seven fair-CRPS records were lost that
   way when job 6266774 expired at 16:42Z. Copy back per member, not per job.
3. **No launcher may end in `exec`.** It replaces the shell, so the EXIT trap that copies
   results back never fires; two runs reached step 4800 over ~4h20m and left nothing, about
   35 GPU-hours. Eight launchers still have this shape.
4. **`/home` byte headroom is checked before each wave, not assumed.** Measured 2026-09-05:
   100.21 GiB of a 100.58 GiB hard limit, **0.38 GiB free**, while inodes sit at 10.4%. `df`
   reports 15 PB and answers a different question.
5. **Attach rather than queue.** `--gres=gpu:4 --cpu-bind=none`, pinned with
   `CUDA_VISIBLE_DEVICES`; `--gres=gpu:1` binds whatever Slurm picks, usually the occupied
   logical device 0.

## 5. What this experiment cannot settle

- **Whether the objective helps at a different scale or budget.** k trajectories at 4800 steps
  measure the trajectory rung *at this budget*. Nothing here extrapolates.
- **Whether conditioning improves.** Two of the three registered metrics are provably blind to
  it: under a context shuffle `qL1_actual` and `qL1_after_shuffle` are bit-identical in
  `fix_midtraining.json`, as are `sd_ratio_actual` and `sd_ratio_after_shuffle`. `fair_crps` is
  the only conditional metric and has never produced a number.
- **Why any effect exists.** The uniform-weight control has already refuted the importance-weight
  mechanism. A positive result here would say "this training procedure moves calibration", not
  "the density ratio is what moved it".
- **Anything at k < 6.** At k = 3 the smallest detectable effect is 0.1834, larger than the
  entire span of everything this study has ever reported.

---

## 6. CORRECTION 2026-09-05T08:2xZ — the premise of §0 is false

**Everything above was written on the premise that there is one fine-tuning run per arm and
that the trajectory rung has never been measured. That is no longer true, and it was already
untrue when this section was written.** Measured on 2026-09-05 under
`/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z/ckpt/`:

| family | dirs | checkpoints per dir (min/med/max) | created |
|---|---|---|---|
| `wm_ft_traj_s*` | 45 | 0 / 10 / 32 | 09-04 21:18Z .. 09-05 02:48Z |
| `wm_ft_traj3_s*` | 14 | 0 / 7 / 10 | 09-04 23:50Z .. 09-05 03:30Z |
| `wm_ft_r3rep_s*` | 8 | 4 / 4 / 4 | 09-05 04:13Z .. 09-05 04:45Z |
| `wm_ft_traj_r*` | 6 | 0 / 5 / 7 | 09-04 22:57Z .. 09-05 00:07Z |
| **`wm_ft_traj1_r*`** | **24** | **0 / 0 / 0** | **09-05 06:53Z .. 09-05 08:04Z — in flight now** |

These are trajectory-seed replicates, not a different experiment. Their own progress files say
so: `wm_ft_traj_s30` records `"train_seed": 30, "max_step": 4800, "step": 4800,
"complete": true`; `wm_ft_traj3_s1` records `"train_seed": 1`; `wm_ft_r3rep_s40` records
`"train_seed": 40, "max_step": 1500, "complete": true`. That is exactly the rung-5 measurement
§4 says is missing.

**Three consequences, in order of importance:**

1. **Do not launch anything from §2's cost table.** Twenty-four `wm_ft_traj1_r*` runs were
   started within the last hour and have not yet written a checkpoint. Starting a k-trajectory
   sweep now would duplicate work already in flight and contend for the same cards.
2. **The right next action is to read what exists, not to spend 354 node-h.** `s_trajectory`
   can be estimated today from the completed `wm_ft_traj_s*` runs at step 4800, for zero GPU
   cost. That estimate then decides whether any further trajectories are needed at all.
3. **This section did not check for prior art before pricing new work.** The
   `ft_progress.json` files carrying `train_seed` were on disk and readable the whole time.
   The failure was not measurement, it was not looking — the same failure mode this study is
   documenting elsewhere.

**§1 and §2 stand as arithmetic and are withdrawn as a plan.** The power table remains a
correct statement about how many trajectories a given effect needs at a given sd; it is not a
statement about what should be run, and with replicates already accumulating it is not even a
statement about what is missing. Whoever owns the `wm_ft_traj*` line should be asked what they
are measuring before this section is acted on.

> **Withdrawn 2026-09-05 (round 5).** The sentence above saying `s_trajectory` "can be
> estimated today from the completed `wm_ft_traj_s*` runs at step 4800, for zero GPU cost"
> is **wrong**. Those runs have **no matching evaluation**: `sweep_results.jsonl` contains
> only `multi4`/`unifw`/`multi3` arms, `crps_panel.jsonl` has zero traj rows, and the task
> root's `data/` has no traj rollout directories. Twelve trajectories are trained and none is
> scored, so estimating the rung on the published estimand needs generation and scoring —
> GPU work, not a free read. See `round5/TRAJECTORY_LEDGER.md`.
