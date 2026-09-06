# Global verdicts — what is withdrawn, what is untested, what still stands

Durable record. Supersedes the verdict lines scattered through `RESULTS_20260905.md` where
they conflict.

## V1 — the dilution mechanism: **UNTESTED**

Not "supported", not "refuted". **The design cannot produce a positive result for any data**,
for two independent reasons (R3-F1):

1. **The treatment and its null are the same object.** The impurities are drawn at random from
   the complement of the pool (`PREREG`/addendum 5 §D step 2), and the hypothesis under test is
   that impurities are *worse than a random draw from the complement*. A test whose treatment
   arm and null arm are constructed identically cannot separate them.
2. **The outcome is exactly affine in the mixing fraction.** The reported contrast is a mean
   over pool members, so a pool that is a fraction `p` reference and `1−p` random has an
   expected contrast that is the two-endpoint interpolation by construction. R3 verified the
   published "linear prediction" column reproduces the endpoint interpolation to four
   decimals. **Linearity was not evidence of dilution; it was arithmetic.** The ±0.037
   "resolution" bounds draw noise, not contamination.

**Consequence.** `PLAN` §0.3's `k = 3` budget rests on this assumption and therefore rests on
nothing measured. §0.3 must not be quoted as a conclusion. The required `k` is **not
established**, and the archive cannot establish it: with 10 seeds the split-half curve stops
at `k = 5`, and the only extrapolation available is rejected by its own residuals, so it
cannot set a budget in either direction.

**Reachable substitute** (R3, CPU-only, not yet run): define the impurity as the rule's
**observed** false positives — contexts in the top decile on SELECT seeds but not on EVAL
seeds — rather than as a random draw, and compare slopes. That version has a treatment arm
distinguishable from its null.

## V2 — the 28 twice-generated contexts: **first-session duplication, not a 5.6% sample**

Addendum 7 reported them entering the pools "10 / 14 / 11 across the three arms, which is at
the population rate rather than concentrated". **The rate was the wrong check** (R3-F3).

The duplication is **concentrated in calendar time**: from `data_real` filename dates, with
zero id/date inversions in 500, all 28 fall on one or two sessions out of 20 — **AMD 28-of-35
and AMZN 28-of-29 on 2026-01-02 alone**. Their `|y|` is **0.791×** the rest (t = −2.75, CI
[0.647, 0.968], below 1 in 7/8 tickers); JPM's score rank-sum z = +3.17. Eleven of them in a
pool move `bal` by **−0.112**, about twice the between-rule difference addendum 6 called
established.

A count that matches the population rate says nothing about *which* population the members
are drawn from. **The check had to be on the calendar, and was not.**

**Mechanism, and it is deterministic.** `gen_driver.py:1698` chunks the index list as
`flat[i:i+batch_size]`; a final short chunk is padded by wrapping to the **start** of the
list, and the list is id-sorted, so the wrap always re-generates the **earliest** contexts.
**Every future run with `n_contexts % batch_size ≠ 0` double-generates its first session**,
and the stored files are the second pass. This is upstream of the manifest requirement: a
manifest that records `n_contexts` and `batch_size` without flagging a nonzero remainder
records the defect without naming it.

## V3 — X4: conclusion stands, **all published numbers withdrawn**

See addendum 9. The metric was `| ‖θ_A‖ − ‖θ_B‖ |`, a difference of norms computed from
separate single-checkpoint runs, not a distance.

### Raw pairwise values, recorded here so they need not be re-derived

Reference: `/lus/lfs1aip2/projects/public/u6gb/sigma-0/checkpoints_selftrain/j5705912_b30675li_5705912/69378`
Ladder: `/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z/ckpt/wm_ft_multi3_step<N>/69378`

| rung | ft steps | ‖θ_A−θ_B‖ | ‖θ_A−θ_B‖ non-embed | ‖θ_A‖ | ‖θ_B‖ | norm gap | distance / gap | shared arrays | bitwise identical |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `_step150` | 150 | 8.986009372319051 | 8.83960109144351 | 427.6297849052013 | 427.891083085933 | 0.2612981807317283 | 34.39 | 137 | **0** |
| `_step1500` | 1500 | — | 10.4827 | 427.6297849052013 | — | 0.3458 | 30.8 | 137 | **0** |
| `_step4800` | 4800 | — | 13.5858 | 427.6297849052013 | — | 0.5254 | 26.3 | 137 | **0** |

Full JSON: `results/x4/pairwise_step150.json`, `pairwise_step1500.json`, `pairwise_step4800.json`.

### Withdrawal scope, stated precisely

**Withdrawn**: addendum 5 §E's table in full (four rows of `l2_non_embed` and the `|delta|`
column); the sentence "the distance grows monotonically with fine-tuning steps" *as a
statement about distance*; and §C's decision rule, whose "identical to machine precision"
branch the tool used could not evaluate — `probe_weights_offline.py` has no pairwise mode.

**Not withdrawn**: that `wm_ft_multi3` is a fine-tune descended from the selftrain checkpoint.
Under the correct metric the ordering is still monotone (8.84 → 10.48 → 13.59) and **not one
of 137 shared arrays is bitwise identical at any rung**, so it is not the reference checkpoint
under another name.

**Also recorded**: R3 reports the withdrawn scalar is **non-monotone along the reference
chain** (427.19 → 423.49 → 425.09), an excursion of 3.69 against the ladder's entire 0.525
spread. The quantity a monotone trend was read from is not monotone where no fine-tuning
happens.

## Review provenance

Reviews are real files with reproduction artefacts, not summaries:

| review | file | size | findings | reproduction |
|---|---|---|---|---|
| R1 | `plan_drafts/R1_review_of_01_03_06.md` | — | — | — |
| R2 | `plan_drafts/R2_review_statistics.md` | — | 14, 3 blocking | `plan_drafts/R2_checks.py` |
| R3 | `plan_drafts/R3_review_causal_design.md` | 861 lines | 13, 3 blocking | recomputed inline, login node |

Subagent transcripts are session-local and do not survive the session; the review files and
their scripts are the durable record.

## V4 — the coverage check that was vacuous, and its corrected result

An earlier check printed `shard has 482; request had 0; missing from shard: []`. The request
set was **empty** — the glob for the config matched nothing — so the difference was trivially
empty and **the check established nothing**. It is not a coverage pass and was not treated as
one.

Recovered against the file the launcher actually reads (`attach_adaptation.sh:58`):

| | size | provenance |
|---|---:|---|
| **request universe** | **488** | `/lus/lfs1aip2/projects/public/u6gb/sigma-0/configs/train/dfm_smoke_1gpu.yaml`, key `env_TICKERS` |
| **shard universe** | **482** | `/lus/lfs1aip2/projects/public/s5e/quant_team/lob_preproc_sp500_squashfs/shard_2024-08.squashfs::index.json`, 20,069 files |
| requested, absent from shard | **6** | `BAC, EXE, PSKY, Q, SNDK, XYZ` |
| in shard, not requested | 0 | the shard is a strict subset of the request |
| trainable intersection | **482** | |

Recorded with both provenances in `results/ticker_coverage_2024-08.json`. The launcher reads
the pinned 482 from `results/tickers_2024-08.txt`, one file both members of a pair share,
because a matched pair must train on identical tickers.

**The request universe is recovered, not unavailable.** Had the YAML been missing or lacked
the key, the correct output was "request universe unavailable", not a comparison against an
empty set.

## What this does NOT mean

Correcting a table is not completing an experiment. **No comparative training result exists
for this line.**

What the completed run establishes is narrow and worth stating exactly: the Step 2 launcher,
which had never produced a live step (R1-F4), now runs end to end — mount, data, 200 training
steps, epoch-end checkpoint at `global_step=201`, written to
`/lus/lfs1aip2/projects/public/u6gb/sigma-0/checkpoints_cl_probe/j6324130_ffqshrmi_6324130`,
log preserved at `results/adapt_logs/cl-adapt-early2-s42.log`. **That is a working launcher,
not a result.** It is one member, at a curtailed 200-step budget, with no paired late member
and no same-age null control, so it answers no comparison at all.

The questions these verdicts touch — dilution, first-session duplication, lineage — are all
upstream of any training claim, and none of them is settled by a run that trained.
