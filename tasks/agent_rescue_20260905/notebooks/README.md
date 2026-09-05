# The two delivered notebooks, with outputs

These are the artefacts themselves, not a description of them. Earlier rounds shipped
acceptance documents while the notebooks stayed outside the repository, so a reader of the
commit could not open what was being accepted. That is fixed here.

| file | what it is |
|---|---|
| `friday_talk.ipynb` | 19 cells, all with outputs, 5 embedded figures |
| `null_ladder_and_calibration.ipynb` | 20 cells, all with outputs, 7 embedded figures |
| `friday_talk.html` | self-contained render (`nbconvert --embed-images`), opens in a browser |
| `null_ladder_and_calibration.html` | same |
| `figs/*_FIXED.png` | the two figures that carried D-1/D-2/D-3, after repair |
| `producer_*.log` | the producers' own stdout, exit code 0 each |
| `SHA256SUMS.txt` | content hashes of the four deliverables |

## What was repaired in this round

All three defects from `round8/DISPLAY_ACCEPTANCE.md`. **Data, values, colours and every
conclusion are unchanged** — only layout and legends moved.

| defect | fix | verified by looking |
|---|---|---|
| **D-1** right panel's tick labels overprinted left panel's data | `wspace` 0.38 → 0.62, figure 7.0×2.9 → 7.6×3.3 in | `null_ladder…_FIXED.png`: the left panel's three diamonds at \|Δ\| ≈ 0.083 now stand clear of the right panel's labels |
| **D-2** legend drawn inside the data area beside real markers | both legends moved below the axes (`bbox_to_anchor`) | each row now carries exactly one data diamond; the friday_talk legend sits under the x-axis label |
| **D-3** grey diamonds drawn but never explained | figure-level legend with explicit proxy handles naming **both** colours | legend reads `the effect — beyond every null contrast` (blue), `the effect — inside the null` (grey), `largest of 1575 null contrasts` (red) |

Unchanged and re-checked on the rendered figures: `6.9`, `53.7`, `2.3` in friday_talk cell 06
at their original positions; the null-ladder bars and diamonds at their original values; the
corrected label `day-level t / (df 19) -- the headline came from HERE` still renders in full,
still unclipped.

## Acceptance

- Structural (`round7/t_notebooks.py`): **7/7 on both**, run against these files.
- Display: read as images after the rebuild; **D-1, D-2, D-3 all closed**.
- Producers: `friday_talk.py` rc=0, `make_null_ladder_notebook.py` rc=0 — captured on their
  own line, never beside a command substitution.

## Still out of scope, still untouched

`step_budget.ipynb` (another session holds uncommitted edits to its builder) and
`verdict_audit.ipynb`. Neither was opened, run, or accepted.
