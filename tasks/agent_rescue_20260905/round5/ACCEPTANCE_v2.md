# Acceptance conditions, v2 — counted from evidence

Supersedes `round4/ACCEPTANCE.md`, which had two faults: it said "10 conditions, 6 pass" while
listing seven as PASS, and it marked N7/N8/N9 OPEN from a defect list without reading the file.
All three were already satisfied. Counts below are the output of `round4/t_builders.py`.

## `sol_notebook_fixes` — 9 of 9 checked conditions pass

```
BEFORE (pre-fix backups, falling back to the live file where a script was never edited)
  3 passed, 6 failed
AFTER
  9 passed, 0 failed
```

| # | condition | before | after |
|---|---|---|---|
| B1 | the calibration figure labels the three tests by statistic | FAIL | PASS |
| B2 | the headline `t = −7.8` is not attributed to the ticker-level test | FAIL | PASS |
| B3 | the day-level test is named as the source of the headline | FAIL | PASS |
| B4 | the day-level test is described as conservative, with null max 2.27 | FAIL | PASS |
| B5 | every row quoting \|t\| = 53.65 carries its df | FAIL | PASS |
| B6 | the df quoted equals `fix_nulls.json`'s 3.24 | FAIL | PASS |
| N7 | the headline is reported on \|R−1\| as well as R, with the 5/8 split | **PASS** | PASS |
| N8 | steps 1050 and 1350 are published with the reversal stated | **PASS** | PASS |
| N9 | the uniform arm is labelled as not an independent confirmation, with Pearson r | **PASS** | PASS |

N7–N9 pass before *and* after because `make_step_budget_notebook.py` was never edited — it
already contained them. The test now prints `[note] … no backup … reading the live file`
rather than substituting silently, so a "before" that is really an "after" is visible.

**Still not covered by any check**: whether the *rendered notebooks* carry these corrections.
Only the builder sources were tested; nothing here executes them.

## `sol_history` — 5 of 8, two open, one declined

| # | condition | status |
|---|---|---|
| H1 | every training run with steps, checkpoints and wall-clock from measurement | PASS — 13 runs |
| H2 | `wm_ft_multi4`'s interrupted span flagged with its clean segment | PASS |
| H3 | artefacts ordered by mtime, with an ordering claim drawn from it | PASS |
| H4 | losses quantified with causes | PASS |
| H5 | claims listed with their fate and evidence file | PASS |
| **H6** | GPU-hours for the whole line from `sacct` | **DECLINED — see below** |
| **H7** | the two pipelines that scored overlapping cells compared, disagreement quantified | **PASS (this round)** |
| **H8** | registered claims quoted exactly and matched against the analysis run | **OPEN** |

### H6 — what is verifiable, and what is not

Not guessed, and not filled in. Verifiable from `sacct`:

| job / step | state | elapsed | nodes |
|---|---|---|---|
| `6258838.1439` `unifw-train` | COMPLETED | 04:18:04 | 1 |
| `6258838.1336/.1411/.1418` `unifw-train` | CANCELLED | 16:50 / 07:01 / 06:58 | 1 |
| `6258839.1963` `r3rep-train` | COMPLETED | 04:22:57 | 1 |
| `6258838.1902` `r3rep-train` | FAILED | 00:44:20 | 1 |
| `6269978.2441/.2443` `traj-s21/s20` | COMPLETED | 56:58 / 59:29 | 1 |
| `6269978.2438-.2440/.2442` `traj-s22/s25/s23/s24` | FAILED | ~59:37 each | 1 |
| `6317365` `u6gb-4-node-chain` | CANCELLED | 07:59:49 | 4 |
| `crps-*` scoring cells | COMPLETED | 17:12 / 17:06 / 17:28 | 1 |

**Unknown, and left unknown**: a total for the line. The training steps of `wm_ft_multi3` and
`wm_ft_multi4` fall outside the queried window; several runs were cancelled and resubmitted, so
summing the surviving rows would undercount the failures and overcount nothing consistently;
and node-hours for the 4-node chains cover work from more than one line. A single number here
would be a guess wearing the clothes of a measurement.

### H7 — the two pipelines disagree, quantified

Step 1200 of `multi4`, the headline checkpoint, has **three values from three pipelines**:

| pipeline | K / seeds | R at step 1200 |
|---|---|---|
| published curve (`sweep_curve.json`) | n = 2 seeds | **0.9610** |
| per-seed rows re-aggregated (`verify_sweep.py`, 4 seeds present) | 4 seeds | **0.9578** |
| pooled K ≥ 2 CRPS-scored cells | 8 cells | **0.9781** |

Spread across pipelines **0.0203**, against a registered null band of ±0.0389 — so the
disagreement is about half the width of the band the study reads effects against. The
neighbours that reverse the peak rest on **one seed each**. `verify_sweep.py` now prints both
estimands in separate tables with their cell counts and a note that they must never be merged.

### H8 — open

The registered claims have not been quoted verbatim and matched line-by-line against the
analysis actually run. Not attempted this round.

## Standing constraints observed

`sol_corrected_inference` untouched. No trajectory training started, attached to, or cancelled.
No copies deleted. No global `gh` identity change. Nothing launched from the withdrawn
354 node-h plan.
