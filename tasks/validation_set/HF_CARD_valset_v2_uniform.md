---
license: other
license_name: lobster-derived-academic
tags:
- finance
- limit-order-book
- time-series
- language-modeling
- scaling-laws
size_categories:
- 10M<n<100M
---

# valset_v2_uniform: A Distribution-Matched Frozen Validation Set for S&P 500 Limit-Order-Book Modeling

*Built 2026-08-03. Supersedes nothing: this is a companion to [`sp500-lob-valset-v1`](https://huggingface.co/datasets/kangoxford/sp500-lob-valset-v1), not a replacement.*

> **Read this first.** `valset_v2_uniform` is held out from the **Mamba-3 48-month cohort only**.
> It is **NOT** held out for the Transformer queue, which trains on the 36-month (2023-2025)
> domain. For Transformer evaluation, use `valset_v1`. This is a deliberate trade, explained
> in §2.

## 1. What this is, in one paragraph

`valset_v2_uniform` is **12,094,981** window samples (**3.742%** of the 48-month training domain,
≈ 6.05 billion messages, ≈ 157.2 billion tokens), drawn so that it is a **uniform random subset**
of the training corpus. Unlike `valset_v1`, whose year composition is skewed 2.25x toward 2022,
v2 matches the training epoch on every marginal we can measure, to within 0.011 percentage points.
It is frozen, hashed, and reproducible from a single script.

| | `valset_v1` | **`valset_v2_uniform`** | training epoch |
|---|---:|---:|---:|
| Samples | 5,367,734 | **12,094,981** | 323,221,385 |
| % of 48-month domain | 1.661% | **3.742%** | 100% |
| Year shares (2022/23/24/25) | 55.25 / 13.51 / 15.10 / 16.13 | **24.52 / 22.80 / 25.53 / 27.16** | 24.53 / 22.80 / 25.52 / 27.16 |
| Max year deviation from epoch | **30.74 pp** | **0.0107 pp** | — |
| Per-year retention max/min | **3.80** | **1.0008** | 1.00 |
| Ticker Kish n_eff (of 488) | 163.19 | **128.07** | 128.01 |
| Held out for Mamba-3 | yes | **yes** | — |
| Held out for Transformer | yes | **no** | — |

## 2. Why v1 was skewed, and what changed

`valset_v1` is built in five steps. Steps 1 and 2 take the last 2% of each of three training
shuffle permutations and subtract the first 20% of each, both of which are prefix/suffix operations
on uniform random permutations and therefore **year-neutral**. Step 3 additionally subtracts the
36-month sub-domain's three 20% exclusion zones, so that the same set would also be held out for
the Transformer queue.

That sub-domain spans **2023-01 to 2025-12 only**. The cut can therefore only remove candidates
from 2023 onward, and leaves 2022 untouched. Measured retention in v1 is **3.744% for 2022 against
0.984% for 2023-2025**, a ratio of **3.80**, which is exactly `1 / 0.8⁶` for the 3 seeds × 2
adjacent-window guard = 6 independent 20% cuts. That single step is the entire cause of v1's
30.7 pp year skew.

**v2 is v1's recipe with step 3 removed and nothing else changed.** The build script is
`build_valset.py` with one constant flipped, and the three recipe sections (steps 1-2, steps 4-5,
and the zero-leakage verification block) were diffed and confirmed **byte-identical**.

**The trade.** Removing step 3 removes Transformer immunity. This is not a bug that can be fixed:
the two queues train on domains with **different supports** (48-month includes 2022, 36-month does
not), so no single uniform set can be same-distribution for both. `valset_v1` remains the
conservative ruler valid for both; `valset_v2_uniform` is the distribution-matched ruler for
Mamba-3.

## 3. Why uniformity beats balancing

An obvious alternative was to keep v1 and re-balance it by year. We did not, for a reason worth
stating: **balancing only fixes the marginals you think to name.** A uniform random subset is
same-distribution in *every* marginal at once, including ones nobody thought to check.

This is not hypothetical. v1's **ticker concentration** was off by 28% (Kish `n_eff` 163.19 against
the epoch's 128.01) and went unnoticed, because the published check was a *correlation* (0.9861,
which is genuinely high) rather than a *concentration* statistic. In v2 that deviation is 0.05%
without any ticker-specific code, because it was never introduced.

## 4. Measured acceptance gates

The build aborts before writing if any gate fails. All values below are from the build log and were
**independently recomputed** from `files_48mo.csv` + the decode file, matching digit-for-digit.

| Gate | Threshold | v2 measured | v1 for comparison |
|---|---|---:|---:|
| Per-year retention max/min | < 1.02 | **1.000806** | 3.80 |
| Max year share deviation | < 0.05 pp | **0.0107 pp** | 30.74 pp |
| Per-month retention max/min | < 1.05 | **1.0086** | — |
| Ticker Kish n_eff vs epoch | within 5% | **0.05%** | +27.9% |

Per-year retention: **2022 3.7435% · 2023 3.7448% · 2024 3.7465% · 2025 3.7456%**.

The epoch reference used by these gates **excludes the (GOOG, 2025-12) slice** (272,480 windows),
which the recipe excises and v2 therefore cannot contain. Comparing against an epoch that still
counted them is an apples-to-oranges reference; the first build run failed GATE2 at 0.0580 pp for
exactly that reason, and the fix corrects the reference rather than widening the threshold.

## 5. Zero leakage (Mamba-3)

Unchanged from v1 and re-verified in this build. For every seed s ∈ {5, 42, 137}, the hard assertion
`inv_s[V].min() >= ceil(0.20 · N48)` and `>= CONSUMED_CAPS[s]` passed, where the measured deepest
consumption is seed 5 at **16.63%** of the domain, against a 20% exclusion line.

Because **v1 ⊂ v2 is exact** (verified: 5,367,734 / 5,367,734 = 100.0000%), and v2 was built by
*removing* an exclusion rather than adding samples from a new region, v2's 48-month purity argument
is the same argument as v1's.

## 6. Nested subsets

Strictly nested, verified: `30,720 ⊂ 307,200 ⊂ 3,232,213 ⊂ V`. Every tier covers **488/488 tickers
and 48/48 months** (v1's 30,720 tier covered only 487). No stratification is applied or needed: the
pool is already uniform, so any uniform prefix of it is too.

| Tier | Samples | % of 48-mo epoch | Max year dev | Purpose |
|---|---:|---:|---:|---|
| `31Ksamples` | 30,720 | 0.0095% | 0.165 pp | Routine evaluation; divisible by all eval batch sizes |
| `307Ksamples` | 307,200 | 0.095% | 0.148 pp | High-precision comparison |
| `3.2Msamples` | 3,232,213 | 1.000% | 0.035 pp | Final / paper numbers |
| full pool | 12,094,981 | 3.742% | 0.011 pp | Superset of all the above |

Subset seed is `20260803` (v1 used `20260729`), so v2's tiers are not the same samples as v1's tiers
of the same size.

## 7. Files

```
indices/Val2022-2025sp500-12.1Msamples-3.742pct-of-48mo-epoch-uniform-mamba3only_pool_indices.npy
indices/Val2022-2025sp500-12.1Msamples-3.742pct-of-48mo-epoch-uniform-mamba3only_pool_decode.npz
indices/Val2022-2025sp500-3.2Msamples-1pct-of-48mo-epoch-uniform-mamba3only_subset.npy
indices/Val2022-2025sp500-307Ksamples-0.09504pct-of-48mo-epoch-uniform-mamba3only_subset.npy
indices/Val2022-2025sp500-31Ksamples-0.009504pct-of-48mo-epoch-uniform-mamba3only_subset.npy
meta/files_48mo.csv                    # 472,442 source files: file_idx, ticker, date, seqs, cum_start, offset
meta/manifest_valset_v2_uniform.json   # full recipe, distribution block, guarantees, applicability
meta/SHA256SUMS.txt
```

`pool_decode.npz` carries per-sample `global_idx`, `file_idx`, `seq_idx`, `seq_start_msg`, and
`flag_v1_8ticker`, which is what lets you derive year, month, and ticker for any sample without
re-running the build.

**Filenames deliberately carry the sample count and epoch fraction.** v2's 30,720 tier has the
*same size and different content* as v1's, so a bare `val_subset_30720.npy` would pass every
downstream length assertion while silently being the wrong data.

## 8. Materialization status

This repository currently ships **indices only** (230 MB). The sample data itself is not yet
materialized. At the measured 12,251 bytes per sample, the full pool would be ≈ 148 GB and the
1% tier ≈ 39.6 GB. `valset_v1`'s materialized SquashFS packs are in the v1 repository.

## 9. Reproduction

```bash
# 1 node, CPU only, ~50 min: mounts 48 monthly shards, rebuilds three index domains,
# reproduces the seed {5,42,137} permutations bit-for-bit, runs all gates, writes artifacts.
VALSET_RUN_ID=<id> sbatch build_valset_v2_uniform.sbatch
```

Environment is pinned in the manifest (`torch 2.8.0+cu129`, `numpy 2.3.3`, `python 3.12.11`). The
build asserts `N48 // 8 == 40,402,673` and `N36 // 2 == 122,000,461` against independent historical
training-log anchors before doing anything else, so an environment drift fails loudly rather than
silently producing a different set.

---

*If you need a validation set valid for both the Mamba-3 and Transformer queues, use
[`sp500-lob-valset-v1`](https://huggingface.co/datasets/kangoxford/sp500-lob-valset-v1) and apply
the per-year reweighting documented there. If you are evaluating Mamba-3 only, this set needs no
reweighting.*
