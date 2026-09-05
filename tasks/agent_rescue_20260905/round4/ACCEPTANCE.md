# Acceptance conditions, per item

Written so that "done" is a test result, not a claim. Each row states the condition, how it is
checked, and today's status. **Neither task below is complete.**

## `sol_notebook_fixes`

Declared deliverable: the four builder scripts under
`/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/dfm-post-training-20260801/post_training/distributional_objectives/`.
The verifier scripts are tooling, not the deliverable — an earlier ledger row recorded a
`content PASS` against `verify_sweep.py`, which measured the tool and read as if the task were
finished. That row has been superseded.

| # | condition | check | status |
|---|---|---|---|
| N1 | the calibration figure labels the three tests by statistic, not by "what we used before/now" | `t_builders.py` B1 | **PASS** (red on the pre-fix backup) |
| N2 | no sentence attributes the headline `t = −7.8` to the ticker-level test | B2 | **PASS** |
| N3 | the day-level test is named as the source of the headline | B3 | **PASS** |
| N4 | the day-level test is described as conservative, with its measured null max 2.27 | B4 | **PASS** |
| N5 | every row quoting \|t\| = 53.65 carries its df | B5 | **PASS** |
| N6 | the df quoted equals `fix_nulls.json`'s 3.24 | B6 | **PASS** |
| N7 | the headline is reported on \|R−1\| as well as R, with the 5/8 split and p 0.3047 | **not written yet** | **OPEN** |
| N8 | steps 1050 and 1350 are published alongside 1200, with the reversal stated | **not written yet** | **OPEN** |
| N9 | the uniform-weight arm is labelled as sharing seeds, contexts and days (Pearson r 0.9348), so it is not independent confirmation | **not written yet** | **OPEN** |
| N10 | `verify_sweep.py` reports both estimands with their cell counts and never merges them | run it; two tables must appear | **PASS** |

**6 of 10 conditions pass. N7–N9 are untouched**, and they are the three that change what the
talk claims rather than how it labels a figure.

Before/after evidence for N1–N6: `t_builders.py <dir> --bak <suffix>` runs the same test
against the pre-fix backups. Measured 2026-09-05: **0 passed / 6 failed** before,
**6 passed / 0 failed** after.

## `sol_history`

Declared deliverable: a chronological table plus prose, citing each row's source. No output
path was named in the prompt; `plan_section_history.md` is used.

| # | condition | check | status |
|---|---|---|---|
| H1 | every training run appears with steps, checkpoint count and wall-clock from a measurement, not an estimate | read the table against ckpt mtimes | **PASS** — 13 runs |
| H2 | `wm_ft_multi4`'s interrupted span is flagged and its clean segment given | grep for the 152.8 h gap | **PASS** |
| H3 | analysis artefacts are ordered by mtime and at least one ordering claim is drawn from it | the `fix_attribution.json` ordering | **PASS** |
| H4 | what was lost is quantified with its cause | 2 runs, 7 records, 32 cells, 14 empty dirs | **PASS** |
| H5 | claims are listed with their fate and the evidence file | 6 claims | **PASS** |
| H6 | GPU-hours for the whole line from `sacct` | **not done** — cancelled/resubmitted steps make a total a guess | **OPEN, deliberately** |
| H7 | the two pipelines that scored overlapping cells are compared cell-by-cell and disagreements quantified | **not done** | **OPEN** |
| H8 | the registered claims are quoted exactly and matched against the analysis run | **not done** | **OPEN** |

**5 of 8 pass. H6 is declined with a reason; H7 and H8 are simply not done.**

## What this round did NOT establish

- The `wm_ft_traj*` families (97 directories with checkpoints, 24 more in flight) are
  **identified but not attributed**: their `ft_progress.json` shows `train_seed` values, so they
  are trajectory replicates, but who is running them and to what end is unknown. **Nothing
  should be launched against them until that is asked.**
- `sol_corrected_inference` was not touched.
