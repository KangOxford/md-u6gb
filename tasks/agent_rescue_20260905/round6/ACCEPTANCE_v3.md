# Acceptance status v3 — per item, from test output

Supersedes `round5/ACCEPTANCE_v2.md`. Nothing here is claimed complete because a document was
written; each row names the check and where its output comes from.

## `sol_notebook_fixes` — 9 of 9 checked conditions pass; the check set itself is incomplete

Test: `round5/t_builders.py <dir> [--bak <suffix>]`. Output on 2026-09-05:
**before `3 passed, 6 failed` → after `9 passed, 0 failed`.**

| # | condition | before | after | how verified |
|---|---|:---:|:---:|---|
| B1 | figure labels the three tests by statistic | FAIL | PASS | label literals parsed out of `f2()` |
| B2 | headline `t = −7.8` not attributed to the ticker-level test | FAIL | PASS | regex on the sentence |
| B3 | day-level test named as the headline's source | FAIL | PASS | regex |
| B4 | day-level test described as conservative with null max 2.27 | FAIL | PASS | regex |
| B5 | every row quoting \|t\| = 53.65 carries its df | FAIL | PASS | per-row scan |
| B6 | the df quoted equals `fix_nulls.json`'s 3.24 | FAIL | PASS | compared against the JSON |
| N7 | headline reported on \|R−1\| as well as R, with the 5/8 split | **PASS** | PASS | `make_step_budget_notebook.py`, never edited |
| N8 | steps 1050 and 1350 published with the reversal | **PASS** | PASS | same file, never edited |
| N9 | uniform arm labelled not an independent confirmation, with Pearson r | **PASS** | PASS | same file, never edited |

N7–N9 pass on both sides because that file was never edited; the runner prints
`[note] … no backup … reading the live file` so this is visible rather than silent.

**Not covered by any check, and therefore not claimed:** whether the *rendered* notebooks carry
these corrections. Only the builder sources are tested; nothing executes them.

## `sol_history` — 6 of 8, one declined with its evidence, one open

| # | condition | status | evidence |
|---|---|---|---|
| H1 | every training run with measured steps/checkpoints/wall-clock | PASS | 13 runs, `round4/plan_section_history.md` |
| H2 | `wm_ft_multi4`'s interrupted span flagged with its clean segment | PASS | 152.8 h gap between step 300 and 450 |
| H3 | artefacts ordered by mtime with an ordering claim drawn from it | PASS | `fix_attribution.json` 1 h 02 m after the control finished |
| H4 | losses quantified with causes | PASS | 2 runs, 7 records, 32 cells, 14 empty dirs |
| H5 | claims listed with fate and evidence file | PASS | 6 claims |
| H6 | GPU-hours for the whole line from `sacct` | **DECLINED** | verifiable rows listed in `round5/ACCEPTANCE_v2.md`; the total is left explicitly unknown |
| H7 | overlapping-cell pipelines compared, disagreement quantified | PASS | three values for step 1200: 0.9610 / 0.9578 / 0.9781, spread 0.0203 vs a ±0.0389 band |
| **H8** | **registered claims quoted and matched against the analysis run** | **PASS (this round)** | `round6/H8_REGISTERED_VS_RUN.md`: 5 matches, 1 gap, 1 qualification |

## What remains open, stated as open

- **H6** is declined, not done. A line-total GPU-hours figure does not exist here.
- **Rendered notebooks** are untested (see above).
- **`s_trajectory` is unestimated**, and cannot be estimated from the 12 trained trajectories
  because none is scored and their configuration is unrecorded (`round5/TRAJECTORY_LEDGER.md`).
- The **holdout-ratio anomaly** (multi3/multi4 outside the twelve's range) is a lead, not
  evidence, and has been demoted to that in the ledger.
- **R2 from H8** — which of six round-3 baselines the round-4 headline is read against — is an
  unregistered degree of freedom worth about the width of the null band. Not resolved here.

## Standing constraints observed this round

Read-only on all trajectory work. Nothing started, attached to, or cancelled. The 354 node-h
plan remains withdrawn and nothing was launched from it. `sol_corrected_inference` untouched.
No copies deleted. No global `gh` identity change.
