# Chronological reconstruction of the CRPS return-alignment line

Read-only. Every row cites the artefact it came from. Times are UTC. Training durations are
read from checkpoint mtimes; job states from `sacct`; artefact ordering from file mtimes.
Root: `/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z/`.

## 1. Training

| when | run | steps | ckpts | wall | note |
|---|---|---|---|---|---|
| 08-12 10:05→10:23 | `wm_ft_a` | 100..400 | 4 | 18 m | pilot |
| 08-12 13:29→13:35 | `wm_ft_b` | 100..200 | 2 | 6 m | pilot |
| 08-12 14:47→14:53 | `wm_ft_c` | 100..200 | 2 | 6 m | pilot |
| 08-13 09:58→10:07 | `wm_ft_mt` | 150..300 | 2 | 9 m | |
| 08-13 13:42→15:51 | `wm_ft_multi` | 200..2400 | 12 | 2.15 h | |
| 08-14 12:37→14:46 | `wm_ft_multi2` | 200..2400 | 12 | 2.15 h | |
| 08-14 15:53→17:44 | `wm_ft_multi2_b16` | 100..1200 | 12 | 1.85 h | batch variant |
| 08-14 18:54→20:58 | `wm_ft_multi2_t` | 200..2400 | 12 | 2.07 h | |
| 08-15 05:19→07:29 | `wm_ft_multi2_ak`, `_zk` | 200..2400 | 12 each | ~2.1 h | concurrent pair |
| **08-15 15:34→20:29** | **`wm_ft_multi3`** | **150..4800** | **32** | **4.92 h** | **round 3, the frontier** |
| **08-27 09:25 → 09-02 22:40** | **`wm_ft_multi4`** | **150..4800** | **32** | **see below** | **round 4** |
| **09-03 21:19→09-04 01:25** | **`wm_ft_unifw`** | **150..4800** | **32** | **4.10 h** | **uniform-weight control** |
| 09-04 01:42→02:06 | `wm_ft_multi3rep` | 150..600 | 4 | 24 m | replication, stopped early |

**`wm_ft_multi4`'s span is not its duration.** Steps 150→300 land on 08-27 09:25–09:34 and step
450 on 09-02 18:20 — a **152.8 h gap**. Its clean segment (450→4800 on 09-02, 4.33 h) gives
3.58 s/step; `wm_ft_multi3` gives 3.81 s/step over an uninterrupted 4.92 h. Any cost model that
reads 157 h off the directory is wrong by a factor of thirty.

`sacct` corroborates the control: `unifw-train` was **cancelled three times** (09-03 20:34,
20:54, 21:01, running 16:50 / 7:01 / 6:58) before the attempt at 21:08 **COMPLETED in
04:18:04** — matching the checkpoint span to within ten minutes.

## 2. Trajectory replicates — the part that changes the plan

| family | dirs | ckpts/dir min-med-max | created |
|---|---|---|---|
| `wm_ft_traj_s*` | 45 | 0 / 10 / 32 | 09-04 21:18 → 09-05 02:48 |
| `wm_ft_traj3_s*` | 14 | 0 / 7 / 10 | 09-04 23:50 → 09-05 03:30 |
| `wm_ft_r3rep_s*` | 8 | 4 / 4 / 4 | 09-05 04:13 → 09-05 04:45 |
| `wm_ft_traj_r*` | 6 | 0 / 5 / 7 | 09-04 22:57 → 09-05 00:07 |
| `wm_ft_traj1_r*` | **24** | **0 / 0 / 0** | **09-05 06:53 → 08:04, in flight** |

Their `ft_progress.json` files identify them: `wm_ft_traj_s30` has
`{"train_seed": 30, "max_step": 4800, "step": 4800, "complete": true}`, `wm_ft_traj3_s1` has
`"train_seed": 1`, `wm_ft_r3rep_s40` has `"train_seed": 40, "max_step": 1500, "complete": true`.

**These are the trajectory rung.** The claim repeated throughout this line — "one fine-tuning
run per arm, the trajectory rung has never been measured" — stopped being true on 09-04 21:18Z,
and the plan section that priced a new k-trajectory sweep was written after that. `s_trajectory`
can be estimated from the completed `wm_ft_traj_s*` runs at step 4800 for **zero GPU cost**.

## 3. Analysis artefacts, in the order they were written

| mtime | file | what it settled |
|---|---|---|
| 09-03 18:34:32 | `fix_stats.json` | test calibration, sign-flip floor |
| 09-03 18:34:50 | `fix_measurements.json` | the registered barometers |
| 09-03 21:47:54 | `fix_nulls.json` | the null ladder, 1,575 seed-disjoint contrasts |
| 09-03 21:57:28 | `fix_variance.json` | variance decomposition |
| 09-03 22:01:00 | `fix_midtraining.json` | context-shuffle blindness |
| **09-04 02:27:57** | **`fix_attribution.json`** | **the uniform-weight refutation** |
| 09-04 14:26:49 | `sweep_curve.json` | the step sweep |
| 09-04 14:41:34 | `sweep_results.jsonl` | per-cell sweep rows |

The ordering matters for one claim in particular: `fix_attribution.json` was written **1 h 2 m
after `wm_ft_unifw` finished** (01:25 → 02:27). The refutation of the importance-weight
mechanism is therefore downstream of a control that had just been trained, not a
reinterpretation of older data.

## 4. What was lost, and to what

- **Two runs left nothing.** `wm_ft_multi3rep` survives only to step 600 (4 checkpoints) and
  `wm_ft_multi4rep` has no directory at all, against a recorded ~35 GPU-hours. Cause on
  record: launchers ending in `exec`, which replaces the shell so the `EXIT` trap that copies
  node-local results back never fires. **Eight launchers still have this shape.**
- **Seven fair-CRPS records** died with allocation 6266774 when it expired at 16:42Z;
  `/local/user/$(id -u)/` is per-node and does not survive.
- **32 of 32 cells** in `crps_res_kcollapse_20260904T163807Z` were lost to K-collapse:
  `compare_arms.py:165` reads only `root/member_0` and `score_v5_primary.py:212` calls it once
  per arm *directory*, so K became the number of directories, `fair_crps` divided by
  `2k(k−1) = 0`, and every cell returned NaN. The `.bad` files they left were counted as
  progress, so the wave reported as finished.
- **14 checkpoint directories are genuinely empty**; the other 131 non-`_stepN` directories
  use the `step_NNNN`-inside layout and do hold checkpoints. An earlier count that treated all
  145 as empty was wrong and is withdrawn.

## 5. Claims and their fate

| claim | status | evidence |
|---|---|---|
| Round 4 ends below round 3 on R | **holds** | 18 round-3 single-seed values separate from 4 round-4 values, p = 2/C(22,4) = 2.7e-4 |
| The exit was caused by the importance weights | **refuted** | uniform-weight control moves −0.0969 vs round 4's −0.0808; weights term −0.0161 inside the ±0.0389 null band (`fix_attribution.json`, all four values verified verbatim) |
| Step 1200 is a peak worth stopping at | **not established** | P(max of 10 noise ≥ observed) = 0.19; unpublished steps 1050/1350 take R 0.806 → 0.915 → 0.819 |
| The registered effect at FWER 0.05 | **fails** | maxT-adjusted 0.0469, but the sign-flip test's measured size makes nominal 0.0078 truly 0.0146, so true FWER ≈ 0.071 |
| Two of three metrics see conditioning | **refuted** | `qL1_actual` == `qL1_after_shuffle` and `sd_ratio_actual` == `sd_ratio_after_shuffle`, bit-identical for every ticker and arm |
| The headline t = −7.8 came from a test that fires on noise | **wrong attribution** | −7.8362 is `naive_day_level_t` (df 19), and that test is the **conservative** one here (null spread 0.75, max 2.27). The tests that misbehave are ticker-level (null max 6.87) and crossed (53.65 at Satterthwaite df 3.24) — neither produced the headline |

## 6. What this reconstruction does not cover

- No GPU-hour total from `sacct` for the whole line: the training job steps of interest predate
  the window queried here and several were cancelled and resubmitted, so a total would be a
  guess dressed as a measurement.
- The `wm_ft_traj*` families are identified by their `ft_progress.json`, not by their launch
  scripts; **who is running them and what they are measuring has not been established** and
  should be asked before anything is launched against them.
- The two pipelines that scored overlapping cells are not compared here cell-by-cell; that
  needs `verify_sweep.py`'s two estimands to be joined, which is deliberately not done because
  they are different estimands (different K, no seed identity).
