# valset_v1: A Frozen Validation Set for S&P 500 Limit-Order-Book Modeling

*Frozen Validation Set for the S&P 500 Limit-Order-Book Scaling-Law Study · 2026-07-29*

---

## Executive Summary

valset_v1 is a validation set built once from the S&P 500 limit-order-book pretraining corpus and permanently frozen thereafter, comprising **5,367,734 samples** (1.661% of the full domain, approximately 2.68 billion messages and 69.8 billion tokens). It solves a specific problem: the 33 training runs in the scaling-law study must be compared repeatedly against the same yardstick, and that yardstick must satisfy three conditions: **training has genuinely never seen it**, **it is drawn from the same distribution as the training data**, and **it never changes**.

There are three core guarantees, each backed by independent evidence. First, **zero leakage**. We audited every training record one by one (including retries after mid-run failures), confirming that each run consumed only a prefix of its data permutation, reaching at most 16.63% depth, while the validation set is drawn entirely from the region beyond 20%, with per-sample verification performed (§4); an independent behavioral experiment cross-confirmed this on two models, 78M and 350M: the model shows no measurable loss advantage on data it "definitely has seen," and the validation set is indistinguishable from data it "definitely has not seen" under a composition-matched comparison (§10). Second, ~~**same distribution**. Sample positions are determined by a random permutation and are independent of content, so the validation set is, by content, a uniform sample of the full domain: the ticker-level share correlates with the full-domain share at 0.9861, with all 488 tickers covered (§5).~~

<span style="color:red">**[CORRECTION 2026-08-03] This guarantee is wrong as stated, and is contradicted by §5.1 of this same document.** The inference "sample positions are random, therefore the set is a uniform sample of the full domain" is valid only for **step 1** of a **five-step** construction. Steps 3 to 5 are content-dependent removals, and step 3 (the 36-month sub-domain's three 20% exclusion zones) applies **only to 2023–2025** by construction. §5.1 measures the consequence: retention is **3.744% for 2022 against 0.984% for 2023–2025, a ratio of 3.80x**. The resulting validation set is **55.2% from 2022 while the full domain is 24.5%**, a deviation of **30.7 percentage points**. That is not a uniform sample of the full domain. The cited evidence (ticker correlation 0.9861) establishes representativeness along the **ticker dimension only** and is silent on the temporal dimension, which is precisely where the set departs from the domain. The defensible claim is narrower: **same corpus and same support; uniform along the ticker dimension; year-skewed by construction; recoverable to full-domain weighting only via the monthly re-weighting described in §5.1 and §7.** Independently re-measured 2026-08-03 from `files_48mo.csv` and `val_pool_decode.npz`: epoch 24.51 / 22.78 / 25.50 / 27.22% by year against pool 55.25 / 13.51 / 15.10 / 16.13%, with the per-month ratio a clean step function at the 2022/2023 boundary (2022 months 2.246–2.264, 2023–2025 months 0.570–0.600).</span>

Third, **permanently frozen**. The index, SHA-256 hashes, and the complete construction manifest were frozen once; as long as future training does not cross the reserved exclusion boundary, the validation set remains valid forever (§4).

Two forms are provided for use: an **index manifest** (43 MB, used together with the source data, with three nested subset tiers included) and a **materialized data package** (a standalone squashfs file carrying all sample data itself, in a format identical to the training data shards, readable by existing code with zero modification, §6). Two tiers of the materialized package have been delivered: the 30,720-sample tier (359 MB) and the 307,200-sample tier (3.51 GB), both of which passed byte-exact quality checks.

---

## 1. Background and Motivation

The dependent variable in scaling-law research is "held-out cross-entropy": models of different sizes are evaluated for loss on the same data that never participated in training, and the loss is then fit as a function of parameter count and data volume. This project's primary training matrix comprises 33 runs (12 model sizes, up to 3 data seeds per size), and subsequent experiments, including long-data-volume probes and a Transformer control matrix, must all repeatedly produce numbers against this same yardstick.

This imposes two requirements. First, the yardstick must be strictly independent of all training data: if any sample seen by any run were mixed in, the loss at the corresponding point would be systematically underestimated, directly distorting the fitted curve. Second, the yardstick must be fixed once and for all: if a fresh sample were drawn ad hoc for every evaluation, comparisons across batches would be contaminated by sampling noise, whereas the inter-run loss differences we need to resolve are only on the order of one part in a thousand.

The difficulty is that "independence" is not obvious: the three data seeds each correspond to a different shuffle order, and the regions consumed by the 33 runs interleave with one another; in addition, several side experiments (a Transformer training run over a different time window, one fine-tuning run, and a number of pilot runs) also touched the same corpus. The construction of valset_v1 clears every one of these consumption trails one by one, then samples only from the region that is "certain to have been untouched by anyone."

## 2. Dataset Overview

| Property | Value |
|---|---|
| Source corpus | S&P 500 LOB, 488 tickers, 2022-01 – 2025-12, 26-token encoding |
| Sample definition | non-overlapping 500-message window (message + order-book stream) |
| Domain size N | **323,221,385** windows (472,442 ticker-day files) |
| **Validation set V** | **5,367,734 windows = 1.661% of N** |
| Volume | ≈ 2.68 B messages ≈ 69.8 B tokens |
| Nested subsets | 30,720 ⊂ 307,200 ⊂ 3,232,213 (= 1% of N) |
| Ticker coverage | 488 / 488 (GOOG missing only its 2025-12 slice) |
| Files touched | 434,842 / 472,442 (92.0%) |
| Legacy-corpus flag | 847,533 samples (15.79%) carry the `v1_8ticker` flag |
| Freezing | index lists + SHA-256 + full construction manifest |
| Materialized form | standalone squashfs shards, layout-identical to training shards (§6.2) |

## 3. Construction Method

The starting point for the construction is a structural property of the training pipeline: data shuffling is driven by three seeds (5, 42, 137), each of which arranges all 323 million windows into one deterministic order, and whatever any run consumes is **a prefix** of that order: however many steps it runs, that is exactly how much it consumes (128 samples per step), never skipping ahead. An equally important property is that a window's starting position within its file is determined by a fixed random number independent of the training seed, so the "j-th window" seen by every run covers exactly the same messages. Together these two properties mean that, once the deepest consumption step under each seed is known, the question of "which samples have been touched" can be answered precisely, down to the message level.

Sampling then proceeds in five steps on this basis. Step one: take the **last 2%** of each seed's permutation, the segment farthest from the consumption frontier; merging and deduplicating across the three seeds yields 19,007,384 candidates. Step two: remove any candidate falling within the **first 20%** of any seed's permutation. The deepest point across all historical consumption (the 700B-token long run under seed 5) reaches only 16.63%, so the 20% line encloses all consumed regions with margin to spare; 12,106,704 samples remain. Step three: handle a second time window. The Transformer and positional-encoding experiments were trained on a 36-month sub-domain spanning 2023-2025, and this sub-domain is about to host a rerun of the Transformer matrix. We map the first 20% of that sub-domain's three seed permutations back onto full-domain windows (the window origins of the two domains are offset, so adjacent windows are removed together), removing 6,735,581 samples. Step four: a pilot run on a 466-ticker subset consumed 19,200 samples; the same mapping method removes 1,347 samples. Step five: GOOG's December 2025 was fully trained on once during a fine-tuning run, so that entire month, 10,377 samples, is removed. The final result is 5,367,734 samples.

| Step | Samples | % of N |
|---|---:|---:|
| Union of each seed's last 2% (deduplicated) | 19,007,384 | 5.881% |
| Remove samples inside any seed's first 20% | 12,106,704 | 3.746% |
| Remove mapped windows of the 36-month domain's three 20% zones (incl. adjacent windows) | −6,735,581 | |
| Remove windows consumed by the 466-ticker pilot run | −1,347 | |
| Remove the entire (GOOG, 2025-12) month | −10,377 | |
| **Final validation set** | **5,367,734** | **1.661%** |

It is worth emphasizing that "taking the last 2% of the permutation" introduces no bias in time or content: the permutation is uniformly random, and where a sample lands has nothing to do with what it contains, so the last 2% is, by content, a uniform random sample of the full domain. This is fundamentally different from the naive approach of "taking the last stretch of time in the dataset," ~~and it is also the source of the distributional consistency reported in §5.~~

<span style="color:red">**[CORRECTION 2026-08-03] The struck clause over-generalises from step 1 to the whole construction.** The statement above is true of **step 1 in isolation**, and step 1 alone would indeed yield an unbiased sample. But the final set is the output of **all five steps**, and step 3 removes the 36-month sub-domain's exclusion zones, which exist only over 2023–2025. Time bias is therefore introduced *after* step 1, by a step that is explicitly time-restricted. §5.1 quantifies it and even predicts it correctly in closed form (`3.746% × 0.8⁶ ≈ 0.981%`), so the construction is behaving as designed; what is wrong is only the claim that the design leaves the temporal distribution intact. Read "§5 shows distributional consistency" as **"§5.2 shows ticker-dimension consistency; §5.1 shows temporal inconsistency."**</span>

## 4. Zero-Leakage Guarantees

The evidence for zero leakage comes in three layers, each independent of the others.

**Layer one: full consumption audit.** We extracted, one by one from the experiment tracking system, the actual step counts of every training record on this corpus, including 270 primary-project runs and 23 long-training runs, and also retries after mid-run crashes (which likewise consumed data). The deepest consumption under each seed is as follows, all falling within the 20% exclusion line:

| Seed | Deepest consumer | Steps | Samples | % of N | Margin to 20% |
|---|---|---:|---:|---:|---:|
| 5 | 700B-token long-run | 420,000 | 53.76M | **16.63%** | 1.20× |
| 42 | 350M full-data run | 168,200 | 21.53M | 6.66% | 3.00× |
| 137 | primary matrix (6M size) | 106,909 | 13.68M | 4.23% | 4.72× |

A counterintuitive detail: the deepest consumer within the primary matrix is not a large model but the small 6M model (4.23%). Under a fixed wall-clock budget and fixed global batch size, the small model runs more steps and therefore consumes more; the largest model, 350M, consumes only 0.42%. But regardless of size, all runs fall far below 20%.

**Layer two: per-sample verification.** After construction was complete, we computed, for every single sample in the final set, its position within each of the three seed permutations, confirming that all positions lie beyond 20% (the minimum position under the three seeds is 0.2000003, 0.2000004, and 0.2000002 respectively, hugging the boundary), and that every sample belongs to the last 2% of at least one seed. Figure 3 visualizes this verification:

![seed positions](figures/fig3_seed_positions.png)

There are no samples in the gray region (the first 20% of the permutation); the uniform background in the middle segment reflects sample positions within the "other seeds'" permutations; the spike on the right corresponds to "last-2%-member" status. The three curves nearly coincide, showing that the three seeds are fully symmetric in construction.

**Layer three: behavioral experiment** (§10): trained models are used to empirically test whether "the loss on validation-set samples is consistent with that of data confirmed never seen," serving as independent cross-corroborating evidence beyond the constructive proof.

Zero leakage is a property that must be maintained. The manifest records a **future training budget**: on the full domain, the total step count under each seed must not exceed 505,033 (seed 5 has already used 420,000, leaving a margin of roughly 85,000 steps); on the 36-month sub-domain, it must not exceed 381,251 (5.8 times the per-run cap for the planned Transformer rerun, so the rerun plan is safe). Any training on a new data sub-domain must be re-audited before it begins. As long as the budget is not breached, the validation set remains valid forever.

## 5. Statistical Properties

### 5.1 Temporal Distribution

![monthly](figures/fig2_monthly_distribution.png)

Figure 2, top panel: the monthly-share curves of the validation set and the full domain have the same shape (both track market-activity fluctuations), but every month in 2022 has a systematically higher share in the validation set. The bottom panel gives the reason: coverage sits on two distinct plateaus:

| Year | Coverage (val / domain) | Mechanism |
|---|---:|---|
| 2022 | **3.744%** | only the main-domain exclusions apply |
| 2023–2025 | 0.984–0.985% | the 36-month domain's three 20% zones also apply |

2022 is affected only by the main-domain exclusion rule, giving a retention rate of 3.744%; from 2023 onward, the 36-month sub-domain exclusion is superimposed, dropping the retention rate to 0.984%. This figure matches the theoretical prediction exactly: for a sample to survive all 6 mutually independent 20% random exclusions (3 seeds times 2 adjacent windows), the survival rate should be 3.746% × 0.8⁶ ≈ 0.981%. The fact that the construction's behavior matches the mathematical expectation is itself a correctness check.

As a result, the ratio of 2022 to 2023–2025 within the validation set is roughly 55:45 (versus 24.5:75.5 in the full domain). When a conclusion requires the same weighting as the full domain, monthly re-weighting exactly restores it; per-sample month decoding is already provided with the deliverables.

### 5.2 Ticker Distribution

![ticker representativeness](figures/fig1_ticker_representativeness.png)

Figure 1 plots each ticker's share of the validation set against its share of the full domain (log scale): all 488 points hug the diagonal, with a correlation coefficient of **0.9861**. In other words, although the validation set is only 1.661% of the full domain, it is a faithful scaled-down copy of the full-domain distribution along the ticker dimension: highly active tickers are represented more, low-activity tickers less, with weights consistent with the actual message flow.

![top30](figures/fig4_top30_tickers.png)

| # | Ticker | Val samples | Val share | Domain share | v1 flag |
|---|---|---:|---:|---:|:---:|
| 1 | TSLA | 140,047 | 2.609% | 2.634% | ✅ |
| 2 | NVDA | 139,854 | 2.605% | 2.937% | ✅ |
| 3 | AAPL | 125,949 | 2.346% | 2.312% | ✅ |
| 4 | GOOGL | 119,141 | 2.220% | 2.211% | — |
| 5 | GOOG | 102,185 | 1.904% | 2.444% | ✅ |
| 6 | MSFT | 100,678 | 1.876% | 1.842% | ✅ |
| 7 | AMD | 97,544 | 1.817% | 1.797% | ✅ |
| 8 | AMZN | 94,293 | 1.757% | 1.741% | ✅ |
| 9 | MU | 50,128 | 0.934% | 0.911% | — |
| 10 | META | 46,983 | 0.875% | 0.652% | — |

The top-10 tickers together account for 18.94% of the validation set (23.03% of the full domain; the difference comes mainly from GOOG and NVDA being slightly more affected by the exclusion rules). The 8 flagged tickers in the table (GOOG, AAPL, NVDA, AMZN, META, TSLA, MSFT, AMD) total 847,533 samples (15.79%): their raw messages were touched, under different preprocessing, by experiments from an earlier era of the 8-ticker corpus. Evaluating the present S&P 500 training queue is unaffected in any way; only when evaluating that earlier batch of models must these samples be excluded using the flag column. The flag defaults to being retained rather than deleted, so as not to distort the activity-weighted distribution.

## 6. How to Use

### 6.1 Index-Based Usage (with Source Data)

Choose among the three nested subset tiers according to the required evaluation precision; the subsets are prefixes of a single fixed shuffle of the full pool, strictly nested, so conclusions from a smaller tier carry over seamlessly to a larger, more precise tier:

| Subset | Size | Purpose | CE std. err. (per checkpoint, est.) |
|---|---:|---|---|
| `val_subset_30720` | 30,720 | routine quick eval (same size as the pre-registered test set) | < 1e-4 nats |
| `val_subset_307200` | 307,200 | high-precision comparisons | ~3e-5 nats |
| `val_subset_3232213` | 3,232,213 (= 1% of N) | final eval / paper numbers | ~1e-5 nats |
| `val_pool_indices` | 5,367,734 | full pool (superset of the above) | |

Usage: reconstruct the dataset with a data configuration identical to that used in training (the manifest records every configuration item), and hand the indices in the subset file directly to the dataloader to fetch samples. The smallest tier, 30,720, is the same size as the pre-registered test set; the cross-entropy standard error for a single checkpoint is below 1e-4 nats, sufficient to resolve inter-run loss differences on the order of one part in a thousand.

### 6.2 Materialized Data Package (standalone squashfs)

For scenarios where mounting all the source data is inconvenient, or where the validation set needs to be sent to collaborators, we have materialized the sample data in its entirety into a standalone squashfs file. Once mounted, it behaves like an ordinary data directory: the directory structure, file naming, and index format are **fully identical** to the training data's monthly shards, and existing code can read it simply by pointing the data path at it:

```bash
mkdir -p /tmp/valset && squashfuse shard_valset_v1_30720.squashfs /tmp/valset
# Evaluation command: DATA_ROOT=/tmp/valset, plus the extra flag --random_offsets_train False
# When finished: fusermount -u /tmp/valset
```

The reason that flag is needed: during training, data reading randomly drops a stretch of messages at the start of each file (to avoid the window cut position being the same every epoch), whereas in the materialized package each file is exactly one 500-message sample, and not a single message at the start can be dropped. This is the only difference in usage.

Each sample is stored as a pair of files, one for the message stream and one for the order book, with filenames of the form `AAPL/AAPL_2023-05-17_message_val00123456.npy.zst`: the ticker and date are immediately visible, and the number after `val` is the sample's global index within the validation set, which can be used to look up, in the accompanying provenance archive, which source file and which rows the sample came from.

| Property | Value / behaviour |
|---|---|
| Self-contained | one file carries all sample data; no dependency on the 48 monthly source shards |
| Layout-compatible | identical directory / naming / index scheme as training shards; no code change |
| Read-only | squashfs is immutable after packing |
| Integrity | SHA-256 of the shard; per-sample provenance (`provenance_*.npz`) |
| Deterministic order | loader sorts by ticker → date → id; evaluation order reproducible |
| Verified | L1: 2,048 samples byte-compared against source rows; L2: read end-to-end by the training dataloader |

| Tier | File | Size | Status |
|---|---|---:|---|
| 30,720 | `squashfs/output/shard_valset_v1_30720.squashfs` | 359 MB | **verified & delivered** (sha256 `ffcb71d90d96…`) |
| 307,200 | `squashfs/output/shard_valset_v1_307200.squashfs` | 3.51 GB | **verified & delivered** (sha256 `c344f4c84cd0…`) |
| 3,232,213 (1% N) | ~~on demand~~ <span style="color:red">`squashfs/output/valset_v1_3232213_parts/` (13 sub-shards)</span> | ~~~38 GB (est.)~~ <span style="color:red">**39,595,597,213 B = 39.60 GB** (actual)</span> | ~~not built~~ <span style="color:red">**built & verified 2026-08-03**</span> |
| full pool (5,367,734) | ~~on demand~~ <span style="color:red">`squashfs/output/valset_v1_5367734_parts/` (13 sub-shards)</span> | ~~~63 GB (est.)~~ <span style="color:red">**65,721,429,210 B = 65.72 GB** (actual)</span> | ~~not built~~ <span style="color:red">**built & verified 2026-08-03**</span> |

<span style="color:red">**[UPDATE 2026-08-03] Both remaining tiers are now materialized.** They are delivered as **13 sub-shards each**, not as a single file, because `mksquashfs` requires every file of a shard to sit in one local tree while node-local `tmpfs` cannot span nodes. The sub-shards mount together with `SQUASHFS_MULTI_MODE=1`, the same mechanism by which training reads its 48 monthly shards. This sharding also keeps `index.json` at **76,310,559 B per sub-shard** (measured); a monolithic full-pool shard would have carried a **0.99 GB** `index.json` that every dataloader worker must parse. Build: 13 nodes in parallel, **57 minutes** wall clock, 13/13 workers clean, 26/26 SHA-256 files written and spot-checked. Measured materialization cost: **12,251 bytes per sample** (the two pre-existing tiers agree to within 0.2 B/sample, which is why the size estimates above were low by only ~4%). **Practical caveat:** a checkpoint takes ~155 s to evaluate on the 30,720 tier, so the full pool is ~175x that, roughly **7.5 hours per checkpoint**. These tiers are not drop-in replacements for the tier used in the published 436-point sweep.</span>

The 30,720 tier covers 487 tickers (the sole exception being the low-activity ticker Q, which happens to have no samples at this tier); the 307,200 tier covers all 488. Both tiers passed a two-layer quality check (2,048 samples each byte-compared, plus a full read-through by the dataloader). The quality check has two layers: L1 draws 2,048 samples and performs a **byte-by-byte comparison** between the data in the package and the corresponding rows in the source data; L2 mounts the entire package and reads it through with the same dataloader used in training (checking the total sample count and spot-reading three points). Only after both layers pass is the SHA-256 hash registered and the package delivered.

## 7. Quality Audit (Against Industry Standards)

Checked item by item against published validation-set quality standards (leakage protection, representativeness, statistical power, versioning, reuse discipline, behavioral verification, documentation; sources listed at the end of this report):

| # | Criterion | Standard | valset_v1 | Verdict |
|---|---|---|---|---|
| 1 | Disjointness / no leakage | disjoint from all training data; look-ahead care for time series | per-sample position proof, message-exact; cross-domain consumption subtracted with guards | **PASS** |
| 2 | Representativeness | i.i.d. with training distribution, all strata covered | ~~ticker-share corr 0.9861, 488/488 tickers, activity-weighted; monthly re-weighting supported~~ <span style="color:red">ticker stratum: corr 0.9861, 488/488 — **PASS**. Temporal stratum: 55.2% vs 24.5% for 2022, **30.7 pp off**, retention ratio 3.80x — **FAIL as sampled**, recoverable only by the monthly re-weighting of §5.1</span> | ~~**PASS**~~ <span style="color:red">**PARTIAL**</span> |
| 3 | Size / statistical power | judged by absolute count and metric SE (large-corpus norm 0.1–1%) | 5.37M samples (1.661%); CE SE < 1e-4 nats already at the 30,720 subset | **PASS** |
| 4 | Frozen & versioned | fixed once, hashed, reproducible | SHA-256 + manifest + deterministic build in the training environment | **PASS** |
| 5 | Reuse discipline | adaptive reuse wears a holdout out (Dwork et al.); keep an untouched final set | tiered subsets for routine use; pre-registered Feb-2026 test set untouched for final claims | **PASS** (policy) |
| 6 | Empirical leakage check | back construction proofs with a behavioral test | seen/held-out/val CE compared on 78M & 350M: no memorization gap; composition-adjusted VAL−MID CI contains 0 (§10) | **PASS** |
| 7 | Documentation | datasheet-style provenance, known biases disclosed | manifest + this report + §8 disclosures | **PASS** |

Three points must be kept in mind at all times when using this set. First, the validation set's annual weighting skews toward 2022 (55:45); when a conclusion needs the same weighting as the full domain, apply monthly re-weighting. Second, the missing (GOOG, 2025-12) month and the 8-ticker flag are known distributional gaps. Third, ~~this validation set measures held-out cross-entropy under the same distribution as training,~~ <span style="color:red">this validation set measures held-out cross-entropy on data from the **same corpus and same support** as training but at a **2022-enriched year weighting** (55.2% vs the domain's 24.5%); it equals the training distribution **only after** the monthly re-weighting of §5.1. Any `E` (irreducible-loss) figure fitted on this set without re-weighting therefore carries a composition term and must not be compared to an `E` fitted on a differently-weighted ruler.</span> which is the dependent variable scaling-law fitting needs; forward-time-shifted generalization ability is instead the responsibility of the separately pre-registered 2026 test set, and the two are complementary rather than interchangeable. In addition, the theoretical literature warns that a holdout set slowly degrades once it is reused adaptively and repeatedly, so paper-level conclusions should be produced on the largest subset touched the fewest times, or on the pre-registered test set.

## 8. Known Limitations and Disclosures

Beyond the three points in the preceding section, two residual uncertainties from the construction period remain, both quantified and recorded in the manifest. First, in the early period of the corpus (before the preprocessing pipeline was finalized), a batch of short-lived debugging runs occurred whose data-reading order can no longer be reconstructed; at most an estimated 0.3% of validation-set samples could have message-level exposure. The models from these debugging runs were discarded long ago and take part in no evaluation, so this poses no practical risk. Second, a set of pilot runs whose records have been deleted would already be covered by the existing exclusion zones if they used a standard seed; if they used a different seed, the expected impact is approximately 0.05%.

## 9. Deliverables

Index artifacts (`artifacts_valset_v1_j5790795/`, SHA-256 hashes in `SHA256SUMS.txt` within the directory):

```
val_pool_indices.npy          # full pool, 5,367,734 × int64 (sorted)
val_pool_decode.npz           # per-sample decode: global id / file / window / start row / flag
val_subset_{30720,307200,3232213}.npy   # three nested subset tiers (30720 also has a json copy)
files_48mo.csv                # metadata for 472,442 source files
manifest.json                 # full recipe, evidence chain, future budget, disclosures
```

Materialized data package (`squashfs/output/`): `shard_valset_v1_30720.squashfs` (delivered) and subsequent tiers, each accompanied by `provenance_*.npz` and `SHA256SUMS.txt`.

Accompanying materials: statistics file `stats_valset_v1.json`; construction and quality-check scripts `build_valset.py`, `valset_report_figs.py`, `squashfs/materialize_valset.py`, `squashfs/verify_valset_squashfs.py` (all checked into the repository, reproducible). All three correctness gates of the construction process passed: the reconstructed dataset size matches two independent historical log anchors; the equivalence test between the offline permutation and the training dataloader passed; and the final set's per-seed, per-sample position verification passed (Figure 3).

## 10. Leakage Test Experiment (Behavioral Verification)

Beyond the constructive proof, a separate, independent behavioral test was designed: if training genuinely never saw the validation set, then when tested with a trained model, the loss on validation-set samples should be indistinguishable from that of "data confirmed never seen," and should show a measurable gap relative to "data confirmed seen." The most common failure mode of this kind of test (known in the literature as dataset inference / memorization-gap testing) is that the control group and the tested group differ in distribution, producing a false signal; this design avoids that at the source: all three sample groups are uniform random subsets of the same full domain, and the only difference is training-exposure status:

| Group | Definition | Training exposure |
|---|---|---|
| SEEN | uniform sample from the consumed prefix of seed 5 | seen exactly once |
| MID | uniform sample from positions [20%, 98%], outside every tail | never seen, not in val |
| VAL | `val_subset_30720` | never seen (claimed) |

For two fully trained models (350M and 78M, seed 5), three groups of 30,720 samples each were drawn, mean cross-entropy was compared, and confidence intervals were estimated via bootstrap. Pre-registered criteria: H1 (detection power): the SEEN group's loss is significantly lower than the MID group's, demonstrating that the experiment can detect the trace of "having been seen once"; H2 (no leakage): the confidence interval of the loss difference between the VAL group and the MID group contains zero. Both criteria holding simultaneously constitutes experimental evidence of no leakage; if H1 does not hold, it indicates that the memorization effect of a single exposure is below the detection limit, in which case the effect of any residual leakage on the loss evaluation is likewise below the detection limit, and the conclusion equally supports the usability of the validation set.

**Results (78M, seed 5; 1,280 batches per group, bootstrap with 20,000 resamples)**:

| Metric | Mean CE (nats) | 95% CI |
|---|---|---|
| SEEN (seen once in training) | 0.559668 | [0.554663, 0.564737] |
| MID (never seen, mid-permutation) | 0.559874 | [0.554962, 0.564877] |
| VAL (validation set) | 0.604456 | [0.599929, 0.608963] |
| SEEN − MID | −0.000205 | [−0.007292, +0.006999] |
| VAL − MID | +0.044582 | [+0.037968, +0.051373] |

First finding: H1 is not detected. The loss difference between SEEN and MID is −0.0002 nats, with a confidence interval spanning zero and a detection limit of about ±0.007 nats. In other words, the model shows no measurable loss advantage on the 30,720 samples confirmed to have been seen once during training. Under the pre-registered interpretation path, this directly supports the usability of the validation set: even data that was "genuinely seen" leaves no trace exceeding 0.007 nats, so the effect of any hypothetical residual leakage on the loss evaluation must likewise fall below this detection limit.

The second finding requires one more step of analysis. The raw difference between VAL and MID is +0.045 nats, significantly positive, meaning the validation set is harder than the mid-permutation control group. This direction is itself the opposite of what leakage would produce: the signature of leakage is a lower loss on the leaked data. The true source is the year composition already documented in §5.1: the exclusion of the 36-month sub-domain (2023-01 through 2025-12) applies only to those three years, so 2022 samples survive entirely, and the validation set is consequently skewed 55.3% toward 2022, whereas the MID group, sampled uniformly across the full domain, has only about 24.6% from 2022. 2022 happens to be the year of highest market volatility and the highest intrinsic entropy of the message flow. After stratifying by year, the two groups nearly coincide:

| Year | CE(MID) | CE(VAL) | VAL − MID |
|---|---|---|---|
| 2022 | 0.634013 | 0.640580 | +0.006567 |
| 2023 | 0.596778 | 0.601371 | +0.004593 |
| 2024 | 0.503214 | 0.506493 | +0.003279 |
| 2025 | 0.507776 | 0.507641 | −0.000135 |

(Stratification classifies each batch by its majority year, with batch year purity of 72–76%.) The loss in 2022 is about 0.13 nats higher than in 2024–2025, which accounts for the entire raw difference; within any single year, the gap between the two groups does not exceed +0.0066 nats. After re-weighting the validation set's per-year losses using the MID group's year weights, the composition-adjusted VAL − MID is **+0.003551, 95% CI [−0.002575, +0.009780]**, a confidence interval spanning zero. Conclusion: under a composition-matched comparison, the validation set is statistically indistinguishable from "data confirmed never seen by any training run"; the raw +0.045 difference is fully explained by the documented constructional property of the 2022 share (see §5.1 and §7) and has nothing to do with leakage.

**Results (350M, seed 5; 5,120 batches per group, bootstrap with 20,000 resamples)**:

| Metric | Mean CE (nats) | 95% CI |
|---|---|---|
| SEEN (seen once in training) | 0.571704 | [0.568466, 0.574844] |
| MID (never seen, mid-permutation) | 0.573214 | [0.569990, 0.576425] |
| VAL (validation set) | 0.619364 | [0.616462, 0.622313] |
| SEEN − MID | −0.001511 | [−0.006019, +0.003007] |
| VAL − MID | +0.046150 | [+0.041881, +0.050457] |

The 350M model fully reproduces the 78M pattern. H1 is likewise not detected (−0.0015 nats, with a confidence interval that contains zero and is narrower). The raw VAL − MID difference of +0.0461 is nearly identical to the 78M model's +0.0446, which is itself further strong evidence for the compositional explanation: year composition is an intrinsic property of the sample groups, so the offset seen by the two models naturally agrees; if the difference came from memorization or leakage, the effect size would be expected to vary with model capacity. Stratifying by year (350M uses smaller batches, giving a year purity of 91%):

| Year | CE(MID) | CE(VAL) | VAL − MID |
|---|---|---|---|
| 2022 | 0.676993 | 0.680968 | +0.003975 |
| 2023 | 0.613392 | 0.605371 | −0.008021 |
| 2024 | 0.496920 | 0.501899 | +0.004978 |
| 2025 | 0.506774 | 0.509872 | +0.003098 |

The year-by-year differences are both positive and negative, with magnitude not exceeding 0.008 nats; the composition-adjusted VAL − MID is **+0.001285, 95% CI [−0.002490, +0.005097]**, a confidence interval spanning zero.

**Joint conclusion**: the two model sizes give consistent evidence. First, neither model can detect a memorization trace from "having been seen once" (78M: −0.0002 ± 0.007; 350M: −0.0015 ± 0.005), showing that under this training regimen a single exposure leaves no trace in the loss, and the effect of any residual leakage is below the detection limit. Second, the validation set is indistinguishable, after composition adjustment, from "same-distribution data confirmed never seen by training" (the adjusted confidence intervals for both models contain zero). Third, the raw difference that makes the validation set harder is numerically consistent across the two models and quantitatively predictable from the construction parameters, making it a documented property of year composition rather than leakage. The behavioral evidence and the constructive proof of §4 are mutually independent and reach the same conclusion: the validation set is clean.

Per-batch losses and analysis scripts are archived with the deliverables (`leakage_exp/results/`, `leakage_exp/analysis/`).

## References

[Unidata: Validation Dataset in ML](https://unidata.pro/blog/validation-dataset-in-ml/) · [IBM: What is Data Leakage in Machine Learning](https://www.ibm.com/think/topics/data-leakage-machine-learning) · [Google Research: The reusable holdout](https://research.google/blog/the-reusable-holdout-preserving-validity-in-adaptive-data-analysis/) · [Dwork et al. 2015, Generalization in Adaptive Data Analysis and Holdout Reuse](https://arxiv.org/pdf/1506.02629) · [mlbenchmarks.org: Test set reuse](https://mlbenchmarks.org/05-test-set-reuse.html) · [The Reliability Gap in Benchmark Auditing (arXiv 2606.03305)](https://arxiv.org/html/2606.03305) · [Gap-K%: Measuring Top-1 Prediction Gap for Detecting Pretraining Data (arXiv 2601.19936)](https://arxiv.org/pdf/2601.19936) · [awesome-data-contamination paper list](https://github.com/lyy1994/awesome-data-contamination)

*Report: 2026-07-29; §10 behavioral experiment results added: 2026-07-30. Built in a software environment and data pipeline fully identical to training (torch 2.8.0); statistics and figures generated by `valset_report_figs.py`, with figure text in English. This page is a complete English translation of the Chinese original, `VALSET_V1_REPORT.md`.*
