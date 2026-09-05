# Pre-registration — comparing selection rules on the existing archive

> Written **before** the comparison was run. Nothing below was chosen after seeing a result.
> Timestamped by its commit; the run that follows it is a separate commit.

## What is being compared

Three selection rules, all building a pool of the same size from the same contexts:

| id | rule |
|---|---|
| `R_v1_global` | global top `q` of `stratify(total)` — the production rule, whose binning collapses because of the zero-move atom |
| `R_v2_global` | global top `q` of `stratify_v2(total)` — corrected binning, selection still global |
| `R_v2_within` | top `q` **within each stratum** of `stratify_v2` — D1 §2's prescription, the piece not yet done |

## Selection and evaluation are disjoint by construction

The 10 seeds are split by seed number, not at random, so the split is frozen and stated
before any number is produced:

```
SELECT = 97701..97705      the rollouts a rule may use to choose the pool
EVAL   = 97706..97710      the rollouts every rule is scored on, and which no rule saw
```

No rule sees an EVAL rollout. A rule that selects well on rollouts it also gets graded on
would be measuring its own noise.

## Two yardsticks, because one of them would rig the comparison

Scoring the pools with `stratify_v2` alone would favour the two rules built on it. So each
pool is scored on EVAL under **both**, and both are reported:

- `Y_v1` — the pool's mean `stratify(total)` on EVAL minus the population mean, in
  population SD units
- `Y_v2` — the same with `stratify_v2`

A rule is preferred only if it is not worse on either yardstick.

## The 28 overwritten contexts — one rule, applied identically to every arm

Addendum 5 §A established that `rank_indices[0..27]` are generated twice, and that the stored
`data_gen` files come from the second pass. That affects 28 of 500 contexts in **every**
member, so it affects every rule equally — but "equally" is an assumption, not a fact, and it
is registered here rather than assumed later:

- **Primary analysis: exclude all 28** from both selection and evaluation, for **all three
  rules**, leaving 472 contexts.
- **Sensitivity analysis: include all 28**, again for all three rules.
- Report both. If the ranking of the rules differs between them, the ranking is not
  established and that is the finding.

## What is being measured, and what it is not

The reported quantity is a **selection contrast on held-out rollouts**: how much more of the
corrected failure score a rule's pool carries than the population, in SD units. It is a
proxy.

**It is not**: a p-value, a statistical significance statement, or evidence about training
benefit. A larger contrast means a rule concentrates more of the measured score into its
pool; whether a pool with a larger contrast trains a better model is a separate question that
no measurement on this archive can answer, because no training was run.

Auxiliary readout, reported beside the contrast because D1 §2's argument is about it:

- `bal` — the pool's mean |realised move| divided by the population's. A rule that has removed
  the realised-move confound has `bal ≈ 1`; the raw score's pool ran 2.3–3.7.

## Pre-registered decision

`R_v2_within` replaces `R_v2_global` for future pools **only if**, in the primary analysis, it
is no worse on either yardstick and its `bal` is closer to 1 in at least 6 of 8 tickers.
Otherwise the binning fix stands alone and the selection rule stays global.
