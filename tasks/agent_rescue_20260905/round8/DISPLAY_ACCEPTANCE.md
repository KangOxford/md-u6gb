# Display acceptance — what the two notebooks actually look like

Scope: `friday_talk.ipynb` and `null_ladder_and_calibration.ipynb` only.
`step_budget.ipynb` (another session's uncommitted edits) and `verdict_audit.ipynb` remain
**out of scope and not accepted**. Figures were read as images, not inferred from bytes.
Nothing was rebuilt for this document.

Artefacts pinned by content, since the commit does not carry them (see §1):

```
14b7e5d06ff6599d2fc363b4b79ef1ffe731d720373d8bc37a577bdb22c6dd10  friday_talk.ipynb
224ff0b84c5877b4dae4d3869069e028bd830e8605f84d582ae09908dd627159  null_ladder_and_calibration.ipynb
```

## 1. First finding: the commit does not contain what it accepts

`git ls-tree -r 5d56c494498eea37942cb69e87bd7cfc51b4177b` lists six files, all under
`tasks/agent_rescue_20260905/round7/`. **Neither notebook is in it.** `ACCEPTANCE_v4.md`
asserts 7/7 on artefacts a reader of that commit cannot open, so the claim is not verifiable
from the commit it ships in. The two figures examined below are added to this round precisely
so that the display claims are checkable rather than asserted.

## 2. What NB1–NB7 do and do not cover

`ACCEPTANCE_v4` reports **7/7** on both notebooks. Restated precisely:

| covers | does not cover |
|---|---|
| the file parses (NB1) | whether a reader can read the figures |
| every code cell carries output (NB2) | whether labels collide with data |
| ≥ 50 KB of embedded PNG exists (NB3) | whether a legend is distinguishable from a data point |
| no error outputs (NB4) | whether every drawn marker is explained |
| the corrected sentence is present (NB5) | whether it is legible where it is drawn |
| PNG headers are valid (NB6) | anything about the pixels after byte 8 |
| the notebook is newer than its builder (NB7) | anything visual at all |

**7/7 is a statement about text, staleness and structural integrity. Display is a separate
result and is reported here.**

## 3. What renders correctly

- **No clipping anywhere.** The lengthened y-tick label in `friday_talk.ipynb` cell 06,
  `day-level t / (df 19) -- the headline came from HERE`, renders in full. `bbox_inches='tight'`
  widened that canvas to 2409 × 940 px (the other four friday_talk figures are 1666–2125 px
  wide), which is why it fits. The correction is legible at ~9.5 pt equivalent.
- **Axis labels and tick labels are readable** in both examined figures: `size of the reported
  result, |t| (log scale)` and the `10⁰ / 10¹ / 10²` decade ticks in friday_talk cell 06;
  `|Δ sd(gen)/sd(real)|` and `|t| (log scale)` in null_ladder cell 01.
- **The df annotations added this round are visible on the figure**, not only in the markdown:
  `(df 7)`, `(Satterthwaite df ~3.2)`, `(df 19)` all appear as tick labels.
- **The colour split reads correctly** in friday_talk cell 06: the two misbehaving tests are
  red bars (6.9 and 53.7) and the day-level row is grey (2.3), matching the corrected text.

## 4. Display defects found — three, all invisible to NB1–NB7

### D-1. Right-panel tick labels are overprinted on left-panel data
**`null_ladder_and_calibration.ipynb`, cell 01, both panels, all three rows.**
The right panel's y-tick labels are drawn far enough left to land inside the **left** panel's
data area, at roughly x ≈ 0.080–0.085 in the left panel's units. In the top row the label
`day-level t (df 19)` is **struck through by the left panel's blue diamond** (≈ x 0.083);
the same collision occurs on the middle row (`ticker-level t (df 7)`) and the bottom row
(`crossed ticker×day t`). In the source image this is around (1000, 175), (1000, 410) and
(1000, 655) px of 2047 × 909.
**Effect:** the left panel's three effect markers, which are the panel's only data, sit under
text belonging to the other panel.

### D-2. The legend is drawn inside the data area, beside real markers
**Both figures, bottom row.**
- `friday_talk.ipynb` cell 06: the legend's blue diamond sits at |t| ≈ 12 on the day-level
  row, ~4 units to the right of that row's real diamond at |t| ≈ 7.8 — around (1615, 680) px
  versus (1475, 690) px of 2409 × 940. **Two blue diamonds on one row, one data and one a
  legend swatch, with nothing distinguishing them.**
- `null_ladder_and_calibration.ipynb` cell 01: the same collision in both panels, and in the
  right panel it puts **three diamonds on the bottom row** — a blue at |t| ≈ 4.8, a grey at
  ≈ 5.5, and the legend's blue at ≈ 5.6.

### D-3. A drawn marker with no legend entry
**`null_ladder_and_calibration.ipynb`, cell 01, right panel.**
Grey diamonds appear on the middle and bottom rows; the legend names only `the effect` (blue)
and `largest of 1575 null contrasts` (red bar). **Nothing on the figure says what grey means.**

## 5. Alignment with ACCEPTANCE_v4

`ACCEPTANCE_v4` §2 should be read as follows, and this document supersedes its implication
that 7/7 closes the notebook question:

| row in v4 | stands? | amended by |
|---|---|---|
| NB1–NB4, NB6 pass both before and after | **stands** | — |
| NB5 and NB7 red before, green after | **stands** | — |
| "rendered notebooks: PASS 7/7" | **stands as written, but is not display acceptance** | §2 above |
| — | — | **new**: display acceptance is 3 defects open (D-1, D-2, D-3) |

## 6. Not done, and why

The three defects are **reported, not fixed**. Fixing D-1/D-2 means moving tick labels and the
legend, which requires re-running the producers and regenerating both notebooks; that was
explicitly out of scope for this pass. They are logged here so the next pass has a location and
a description rather than a re-discovery.

`step_budget.ipynb` and `verdict_audit.ipynb` were not opened and are not accepted. No training
was created, started, attached to or cancelled. Nothing was deleted.
