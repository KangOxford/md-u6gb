# H8 — the registered claims, quoted, against the analysis that was run

Sources. The registration is the dated ledger in
`/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z/RESULTS.md`
(rows `DOSE3-2S`, `DOSE3-4S`, `TAILPOWER-10S`, `E13B-2S`) and the running log
`.../V5_LOOP.md`. The analysis is
`/lus/lfs1aip2/projects/public/u6gb/nb_build_pr22/fix_measurements.json`
(`task1_all_registered_baselines`, `task2_all_five_registered_bars`).

## R1 — which round-3 baseline the round-4 effect is read against

**Registered, verbatim** (`RESULTS.md`, `DOSE3-4S`, 2026-08-16):

> round-3 FINAL verdict: m3 (97701-4) vs m2 (97201-4), 4 seeds each, dose3_verdict.py …
> sd 0.8088→**0.8831** (+0.0743, z +5.65 …) … `v5me3_verdict.json`

**Run:** `task1_all_registered_baselines` reports
`published_baseline_file = "v5me3_verdict.json"`, `published_baseline_level = 0.8831`,
`published_baseline_seeds = [97701, 97702, 97703, 97704]`.

**Verdict: matches.** The file the record names is the file the analysis used.

## R2 — but the record names three baselines, and six exist

| file | seeds | round-3 level | round4 − this | rank by level | rank by most-negative |
|---|---:|---:|---:|---:|---:|
| `v5me3_verdict_2seed_preview.json` | 2 | 0.8895 | **−0.09315** | 1 | 1 |
| **`v5me3_verdict.json`** (published) | 4 | 0.8831 | **−0.08672** | **2** | **2** |
| `v5me3_verdict_tailpower.json` | 10 | 0.8723 | −0.07591 | 3 | 3 |
| `v5me3_verdict_e13b.json` | 2 | 0.8632 | −0.06687 | 4 | 4 |
| `v5me3_verdict_e13b6s.json` | 6 | 0.8546 | −0.05828 | 5 | 5 |
| `v5me3_verdict_fresh4.json` | 4 | 0.8503 | −0.05394 | 6 | 6 |

**The effect ranges from −0.05394 to −0.09315 across the six: a spread of 0.03921.**
The registered null band is ±0.0389, i.e. a **total width of 0.0778**, so the baseline-choice
spread is about **one half-width — 50.4% of the band — not the whole band.** An earlier draft
said "the width of the registered null band" and was wrong by a factor of two; corrected
2026-09-05 round 7. Half a band is still large enough to matter: it is 45% of the published
effect (−0.08672) and larger than the entire ±0.0195 registered-null sd the verdicts were read
against. Three of the six appear in the registered record as they were produced
(`v5me3_verdict.json` at DOSE3-4S, `v5me3_verdict_tailpower.json` at TAILPOWER-10S,
`v5me3_verdict_e13b.json` at E13B-2S).

**Verdict: partial — the gap stands.** Each baseline is registered *when it is made*;
**which one the round-4 comparison would be read against is not fixed in advance anywhere in
the record.** The published choice happens to rank 2nd of 6 rather than 1st, but **that ranking
is not evidence that the choice was unbiased**: with the selection rule unregistered, any
resulting rank is compatible with both an indifferent choice and a favourable one, and a rank
computed after the fact cannot separate them. The only thing the ranking rules out is the
narrow claim "the most extreme available baseline was chosen"; it says nothing about the
process that produced the choice.

## R3 — the two-seed preview and its own caveat

**Registered, verbatim** (`DOSE3-2S`, 2026-08-15):

> **ALL pre-registered bars PASS at direction grade** (B1.13: 2-seed = direction only;
> 4-seed final + isolated critic required before "ACCEPTED")

**Run:** the accepted verdict is the 4-seed `DOSE3-4S` row, with an isolated critic (V14,
independent bootstrap RNG 771155, 2000 draws).

**Verdict: matches.** The caveat the preview attached to itself was honoured.

## R4 — the five registered bars, as the round-4 analysis computed them

| bar | pooled delta | pooled z | reaches 2σ | per-ticker sign | moves in the worse direction |
|---|---:|---:|:---:|:---:|:---:|
| `qL1` | +0.04501 | +4.4481 | **yes** | 7/8 | yes |
| `sd` | −0.08672 | −5.5415 | **yes** | 8/8 | yes |
| `t998` | −0.40625 | −1.9117 | **no** | 5/8 | yes |
| `z3` | +0.20109 | +1.7005 | **no** | 5/8 | yes |
| `tilt_growth` | +0.03676 | +3.2701 | **yes** | — | yes |

**Verdict: partial, and the two halves point the same way.** Only **3 of 5** bars reach 2σ,
so "round 4 is worse on the registered bars" is a 3-of-5 statement at that threshold, not
5 of 5. But **5 of 5 move in the worse direction**, and two of the three that miss 2σ
(`t998`, `z3`) are the two the record itself had already flagged as underpowered — see R5.
The honest summary is: unanimous in direction, majority at 2σ.

## R5 — the record's own power retraction

**Registered, verbatim** (`TAILPOWER-10S`, 2026-08-16):

> **tail t998 STILL not shown: +0.2219 z +1.29 (4-seed was z +1.34 — seeds bought ~nothing)**
> ⇒ METHODS FINDING: the tail z's denominator is DAY-BLOCK variance — tail events cluster in a
> few of the 20 trading days, so power scales with EVAL DAYS, not seeds; my √n_seed power
> extrapolation used the wrong n (registered §8.1 power notes must decompose variance into day
> vs seed components). Verdict-grade tail test requires a longer eval period = next-cycle
> registered work.

**Run:** `t998` in the round-4 analysis reaches z −1.9117, still short of 2σ.

**Verdict: matches, and the record predicted it.** The registration contains its own
correction: seeds do not buy tail power, days do. That is consistent with `t998` remaining
below 2σ at round 4, and it means the two bars that miss 2σ were **already known to be
underpowered by construction** rather than being evidence of no effect.

## R6 — a count that moves with the seed budget

**Registered, verbatim:** `DOSE3-2S` records `toward-1 **8/8**` (2 seeds); `DOSE3-4S` records
`toward-1 **7/8** (GOOG a 0.0003 near-tie)`; `TAILPOWER-10S` records `toward-1 **8/8**` again
at 10 seeds.

**Verdict: matches, and the record is transparent about it.** The same count reads 8/8, 7/8
and 8/8 at 2, 4 and 10 seeds. The record names the near-tie rather than hiding it. It does
mean an 8/8 in this line is not a stable fact about the arms.

## R7 — a bar met on the letter and recorded as not shown

**Registered, verbatim** (`E13B-2S`, 2026-08-18):

> **BAR MET ON THE LETTER, RECORDED AS NOT SHOWN.** The pre-registered saturation risk
> materialised: leave-one-ticker-out drops the tail z to **+0.96 without JPM** … capping every
> ticker at ≤5× stream coverage — day-stratified — drops it to **+0.95** … JPM's stream is
> 24,981 messages, so 2,000 contexts cover it **40×**

**Verdict: matches.** The registration declines a result its own threshold had passed. This is
the strongest single piece of evidence that the registration was being used as a constraint
rather than as decoration.

## Summary of H8

| # | area | verdict |
|---|---|---|
| R1 | the named baseline file | **matches** |
| R2 | which of six baselines, fixed in advance | **not registered** — effect spread 0.03921 ≈ half the ±0.0389 band (total width 0.0778) |
| R3 | the preview's own 2-seed caveat | **matches** |
| R4 | five bars | **3 of 5 at 2σ, 5 of 5 in the worse direction** |
| R5 | tail power | **matches, and self-corrected in the record** |
| R6 | toward-1 count | **matches, but moves with seeds: 8/8, 7/8, 8/8** |
| R7 | a passed bar declined | **matches** |

**One genuine gap (R2), one qualification (R4), five matches.** The registration in this line
is better than the analysis practice around it: the record repeatedly declines results its own
thresholds allowed, and it contains its own power retraction. The unregistered degree of
freedom is not any single number but **which round-3 baseline the round-4 headline is read
against**; that choice moves the headline by 0.03921, about half the width of the ±0.0389
band and 45% of the published effect. Whether the choice actually made was biased is **not
determined here and cannot be, from the artefacts alone**.
