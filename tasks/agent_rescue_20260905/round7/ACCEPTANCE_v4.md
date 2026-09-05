# Acceptance v4 — the rendered notebooks, checked separately from the H8 document

Round 6 said in writing that nothing checked the rendered notebooks. This round checks them.
**"H8 written" and "notebooks render correctly" are two different results and are reported as
two different rows.**

## 1. Producers were run and their exit codes preserved

`rc` is captured on its own line, never beside a command substitution.

| builder | producer exit code | wrote | cells |
|---|---:|---|---:|
| `friday_talk.py` | **0** | `friday_talk.ipynb`, 327,238 B | 19 |
| `make_null_ladder_notebook.py` | **0** | `null_ladder_and_calibration.ipynb`, 368,238 B | 20 |

Logs: `producer_friday_talk.log`, `producer_make_null_ladder.log`.

**Two builders were deliberately NOT run.**
`make_step_budget_notebook.py` carries **uncommitted modifications from another session**
(`git status` shows `M` on it, on `step_budget.ipynb`, and on three figures) and I did not edit
it — running it would regenerate another line's output from their mid-edit source.
`make_verdict_audit_notebook.py` is clean and was never part of the fixes; running it would
touch a deliverable outside this scope. **Neither is claimed as accepted.**

## 2. Rendered-notebook checks

Test: `t_notebooks.py <notebook> [<phrase>] [<builder>]`. The same test runs against the
pre-run backups, which is the only way to know it detects anything.

| # | check | `friday_talk` before → after | `null_ladder` before → after |
|---|---|---|---|
| NB1 | parses as nbformat 4 | PASS → PASS | PASS → PASS |
| NB2 | every code cell carries output | PASS → PASS | PASS → PASS |
| NB3 | embeds figures (>50 KB of PNG) | PASS → PASS | PASS → PASS |
| NB4 | no cell carries an error output | PASS → PASS | PASS → PASS |
| **NB5** | the **rendered** notebook carries the correction | **FAIL → PASS** | **FAIL → PASS** |
| NB6 | every figure on disk is a non-empty valid PNG | PASS → PASS | PASS → PASS |
| **NB7** | the notebook is not older than its builder | **FAIL → PASS** | **FAIL → PASS** |
| | totals | **5/7 → 7/7** | **5/7 → 7/7** |

NB1–NB4 and NB6 pass on both sides. They confirm well-formedness and **do not detect the
fix**; only NB5 and NB7 do, and both were red before. Reporting them as evidence of the fix
would be the same error as counting a document's existence as a result.

**NB7 is the check whose absence produced round 6's gap.** It quantifies it: the shipped
`friday_talk.ipynb` was **1,685 minutes older** than the builder that was supposed to have
produced it, and `null_ladder_and_calibration.ipynb` **1,785 minutes older**. Every
source-level PASS in rounds 4–6 was therefore true of a file no reader was opening.

## 3. Figures

30 PNGs under `figs/`; **0 zero-byte, 0 with an unreadable header.** Written 2026-09-05 10:17Z
by the two producers. Pixel dimensions are consistent with the builders' `savefig.dpi = 300`
(e.g. `n2_null_ladder.png` at 2102 × 880, `t4_attribution.png` at 2125 × 938).

## 4. Status board, per item

| item | result | how it was verified |
|---|---|---|
| H8 document written | **PASS** | `round6/H8_REGISTERED_VS_RUN.md`, 7 items against the registered record |
| H8 arithmetic | **corrected this round** | 0.03921 is ≈ **half** the ±0.0389 band (total width 0.0778 → 50.4%), not the whole band; and 45.2% of the published effect |
| H8 R2 framing | **corrected this round** | "ranks 2nd of 6" no longer offered as evidence of an unbiased choice |
| builder sources B1–B6, N7–N9 | PASS 9/9 | `round5/t_builders.py`, before 3/9 |
| **rendered notebooks** | **PASS 7/7 on the two rebuilt** | this document |
| `step_budget.ipynb`, `verdict_audit.ipynb` | **NOT accepted** | producers not run; another session owns one, the other is out of scope |
| H6 line-total GPU-hours | **DECLINED** | verifiable rows listed in `round5/ACCEPTANCE_v2.md`; total left unknown |
| `s_trajectory` | **unestimated and unestimatable here** | 12 trained, 0 scored, configuration unrecorded |
| holdout-ratio anomaly | **lead, not evidence** | demoted in `round5/TRAJECTORY_LEDGER.md` |

## 5. Constraints observed

No training created, started, attached to, or cancelled. No other line's builder run. The
354 node-h plan remains withdrawn. `sol_corrected_inference` untouched. Both notebooks were
copied to `.bak_20260905T101704Z` before regeneration; nothing was deleted. No global `gh`
identity change.
