# Pre-registration — context-held-out validation, and the noise-floor correction

> Written **before** either measurement ran. Committed as its own commit; the run is a
> separate one.

## Why a second pre-registration

`PREREG_selection_rules_20260905.md` held out **rollouts**. That answers "does the pool keep
its score advantage when re-scored", and it was reported as such. It does **not** answer
whether anything transfers to contexts the selection never saw, because every context was in
both halves. Held-out rollouts are not context generalisation and are not to be presented as
if they were.

## Part 1 — what actually has to generalise

A selection rule does not generalise to new contexts by picking them; it picks from whatever
it is given. What *can* fail to transfer is the machinery the rule carries with it:

- the **stratum edges**, which are quantiles of |realised move| estimated on one set of contexts;
- the **zero-move atom's share**, which sets how much of the pool is unstratifiable.

So the test is: freeze the edges on one set of contexts, apply them unchanged to a disjoint
set, and measure what is lost against edges refit on that disjoint set.

### The context split, fixed here and derived from content, not from a seed

Contexts are sorted by their integer id. Even positions → `FIT`, odd positions → `HELD`.
Deterministic, reproducible from the archive alone, and independent of every score.

### Arms

| id | stratum edges | selects on | scored on |
|---|---|---|---|
| `T_transfer` | quantiles of \|y\| computed on **FIT** | `HELD` contexts, `SELECT` seeds | `HELD` contexts, `EVAL` seeds |
| `T_oracle` | quantiles of \|y\| computed on **HELD** | `HELD` contexts, `SELECT` seeds | `HELD` contexts, `EVAL` seeds |
| `T_global` | none (global top q on the raw score) | `HELD`, `SELECT` | `HELD`, `EVAL` |

Seed halves are the frozen ones already in use: `SELECT` = 97701–97705, `EVAL` = 97706–97710.

### Reported

`Y_v2` (contrast on held-out rollouts **and** held-out contexts, in population SD units) and
`bal`. **The transfer cost is `T_oracle − T_transfer`.** A cost indistinguishable from zero
means the edges transfer; a positive cost means they do not, and its size is the answer.

### Pre-registered reading

- transfer cost within the between-ticker spread ⇒ **edges transfer**; `stratify_v2` may be
  fitted once and reused.
- transfer cost above it ⇒ **edges must be refit per context set**, and any pool built with
  borrowed edges carries that cost.

## Part 2 — the 28 twice-generated contexts, uniformly and with lineage kept

Same rule as before, extended so the lineage is preserved rather than discarded:

- **Primary excludes** them, **sensitivity includes** them, identically for every arm.
- **In addition**, every emitted pool records which of its members are among the 28, and the
  ids themselves are written to `results/twice_generated_contexts.json` with the wraparound
  that produced them, so a future consumer can apply its own rule rather than inheriting mine.
- They are **not deleted from the archive** and no file of theirs is moved.

## Part 3 — the noise floor: measure the sharing, do not assume it

R2's third blocking finding is that the do-nothing floor from `repro`/`repB` is understated
because the two are seed-matched and bitwise-identical members cancel in a difference.
**The correction is a measurement, not a blanket rejection of seed-matched pairing.** Matched
pairing is the right design for many contrasts; it is wrong only to the extent that the two
sides actually share draws, and that extent is measurable.

To be measured, per ticker and per horizon:

1. `phi` — the fraction of (context, member) generated values that are **bitwise identical**
   between `repro` and `repB`. This is the sharing, stated as a number.
2. `cov` — the covariance between the two sides' per-context scores, and the variance of each.
   The floor for a matched pair is `Var(A) + Var(B) − 2·Cov(A,B)`; for an unmatched contrast
   the covariance term is absent. The inflation factor is therefore
   `sqrt((Var(A)+Var(B)) / (Var(A)+Var(B)−2·Cov(A,B)))`, computed rather than asserted.
3. The **crossed-seed** floor on the same data: pair `repro` seed *i* against `repB` seed *j≠i*,
   which removes the shared draws while keeping everything else.

**Reported as**: the matched floor, the crossed floor, the measured `phi`, and the ratio. If
the ratio tracks `1/sqrt(1−phi)` the mechanism is confirmed; if it does not, the mechanism is
not established and only the two floors are reported.

**Not done here**: no MDE and no budget is computed from any of this, because the only `k`
extrapolation available is rejected by its own residuals and cannot set one in either
direction.
