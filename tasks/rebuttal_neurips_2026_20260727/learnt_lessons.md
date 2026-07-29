# Learnt Lessons

- Notion attachment preservation requires downloading the actual bytes and recording size/hash provenance; a page snapshot is not enough.
- Reviewer-facing refits must read the exact historical Git blobs behind the submitted figures. Mixing in the later held-out/SP500 sweep would silently change the protocol.
- A successful Notion write is not sufficient verification. Re-fetching caught malformed LaTeX caused by JavaScript escape handling; plain, unambiguous notation was safer for this callout.
- Raw token counts are not cross-model data units when tokenizers encode 26 tokens/message versus one token/event. Message semantics, repeated exposure, parameter convention, and architecture-specific compute must all be aligned before declaring a model data-limited.
- A large fixed-`beta=0.28` fit penalty supports only a finite-range difference from that LM reference. The much smaller fixed-`beta=1.0` penalty prevents treating `beta>1` as tightly established.
- IsoFLOP point-estimate stability and identifiability are distinct. A modest slope change on 5 retained slices does not rescue the loss of 4 unbracketed slices.
- Generated figures must be checked against their CSV/JSON, not only viewed for plausibility. This caught a marker legend that contradicted the verified 5/9 bracketing count.
- Throughput profiling supplies the compute function, not a loss surface. It cannot substitute for completed, matched Transformer trajectories.

## Additional lessons — 2026-07-27 13:19 UTC

- A test-loss figure can still fail to validate scaling exponents when the exponents are locked; even a free fit does not validate the submitted result if it comes from a different sweep. Checkpoint-grid provenance is part of the estimand.
- “Training on the test set” is the wrong description here: network weights stay frozen during evaluation, while the low-dimensional scaling-law surface may be fitted to test CE afterward.
- Historical Git commits can resolve figure provenance that the current caption no longer contains. Here they established that the locked test-CE plot used an earlier 8M--78M six-size sweep.
- Review the NeurIPS checklist after manuscript revisions. It contained both a stale held-out description and an inverted prefactor formula even though the main proposition was correct.
- Generated audit JSON should use standard `null`, not implementation-specific `NaN`, so strict parsers can validate the artifact.
- When two agent tracks report incompatible held-out exponents, preserve both provenance trails and do not silently average or substitute one result for the other.

## Gamma clarification lessons — 2026-07-27 13:31 UTC

- In `q(N)=kN^gamma`, `gamma` is the slope of a log-log compute-profile regression; it must not be confused with the loss exponents `alpha` and `beta`.
- Rescaling the units of `N` or `q` changes `k` but not `gamma`, which is why the slope is the portable quantity.
- A high in-sample log-log `R^2` does not establish profiler measurement precision. With one measurement per size, repeated-measurement uncertainty remains unobserved.
- Notion exact-match updates must preserve stored escape characters such as `N\^gamma`; a failed no-match update is non-mutating and should be followed by a fetch-derived exact replacement.

## Exact-anchor lessons — 2026-07-28 00:12 UTC

- A Notion block anchor can be fetchable as an entity while still rejecting page-update operations. For a nested callout, fetch the parent page, identify the unique stored Markdown line, perform the smallest additive `update_content`, and then re-fetch the original block anchor.
- Rounded values are unsafe identifiers. Here `0.90` can mean the rounded compute-profile exponent `gamma=0.899729`, the near-mature training-loss exponent `beta=0.9438`, or the later interim long-D exponent `beta=0.9169`; every answer must name the symbol, data, and selection protocol.
- Fitting a low-dimensional scaling surface to frozen-checkpoint test CE is not training the neural network on the test set. It is still a test-set model-selection step and must be reported separately from an untouched final evaluation.
- “D-restricted,” “tail-25%,” “loss-cutoff,” and “long-D interim” are distinct estimands. Their numerical proximity or disagreement does not license substituting one for another in a reviewer response.

## Result-table audit lessons — 2026-07-29 14:26 UTC

- A manifest `COMPLETED` or `reached_curtail=1` flag is not sufficient completion evidence. Compare `final_step` with `target_step` and inspect the terminal training log before calling a run complete.
- “Terminal checkpoint” can silently mean “last available evaluated checkpoint.” This distinction matters when evaluation coverage stops before the intended training budget, as for `46M-s5`.
- Aggregate means must retain the per-seed raw values and exceptional `D` values. Otherwise the 46M row would hide that one seed was evaluated at 89.806B tokens while the other two were evaluated at 105.514B.
- Nominal size labels are not parameter counts. Reviewer-facing range claims must use checkpoint-introspected `N`; here the true 2.626M-to-293.283M span is 2.048 decades, not the 3.24 decades implied by plan labels.
- A current fit can be reproduced faithfully while still carrying a cohort-definition error. Report the existing 33-row result as-is for provenance, then label the strict 32-row refit as a new corrected estimand if it is later requested.
- Partial Transformer evaluations and throughput profiles are not completed Transformer scaling-law points. Cross-architecture tables should remain empty until matched terminal loss trajectories exist.

## Seed-ID clarification lessons — 2026-07-29

- A column containing `5, 42, 137` should be labelled as actual seed IDs/values, not merely `seeds`; its cardinality, not the largest numeric value, gives the number of seed-specific runs.
- Keep seed identity and seed count in separate columns when both appear in one audit table.
- Prefer `seed-specific training runs` over `fully independent runs`: the code confirms different initialization and data-shuffle randomness, but that alone does not establish statistical independence.

## 33-run trajectory-figure lessons — 2026-07-29

- A terminal-row CSV cannot support learning curves. Use the cohort-matched canonical checkpoint table, then assert that each curve's last point exactly equals the terminal-row source.
- A figure next to held-out results must say `held-out CE`; calling stochastic microbatch training CE the same “loss curve” would silently change the estimand.
- A 12-size × 3-seed contact sheet is more auditable than an arbitrary 33-panel order: the three absent cells line up directly with the seed-count table.
- “Terminal” in the marker legend must mean final available evaluation, with non-target-reaching runs visibly distinguished. Otherwise `46M-s5` looks falsely complete.
- After uploading Notion media, re-fetch the exact anchor and download the stored files to compare SHA256. A successful upload/append response alone does not prove the displayed image and attachments are intact.
- The first composite build failed because a multi-index lookup dropped the `seed` field. Keeping index columns with `drop=False` fixed it; the validation gate prevented an incomplete figure from reaching Notion.

## Failed→resume figure lessons — 2026-07-29

- A manifest's nominal `(size, seed)` label is not proof of checkpoint ancestry. Resume metadata can reveal a cross-seed restore even when the final job name and manifest row look consistent.
- A physical JID is not always a unique training history: one JID can contain multiple W&B run IDs. Select and record the intended W&B history before using JID-level gap fills or drawing a resume connector.
- “Complete trajectory” must name the observable. Sparse held-out checkpoints cannot reconstruct failure-to-resume optimization history when the predecessor was never evaluated; the complete diagnostic must use training loss and keep the held-out estimand separate.
- A dotted resume connector is provenance, not an observed loss line. Plot physical segments independently, mark failed endpoints and resume starts, and retain zero-data attempts without inventing values.
- W&B `finished` does not necessarily mean the intended training target was reached. A gracefully stopped or timeout-interrupted segment can still require a later resume.
- High-resolution delivery requires both explicit `dpi=300` and post-write pixel/DPI verification. Vector PDF/SVG remains the lossless companion, while Notion's stored PNG should be downloaded and checked rather than trusted from the upload response.
- Replacing a Notion attachment in place should preserve the previous bytes locally first, then retain the existing block ID and verify the replacement hash. This avoids duplicate media blocks without losing the superseded evidence.

## Log-x supplement lessons — 2026-07-29

- A log-scaled training-step axis materially expands the early optimization regime while compressing the late resume boundary. Keeping both linear and log versions avoids forcing one view to serve incompatible inspection goals.
- Log axes require strictly positive lower limits and should use their native logarithmic tick locator; a linear `MaxNLocator` should only be applied in linear mode.
- Attempt ribbons drawn in axes coordinates remain evenly readable under either data-axis transform, while loss segments, target lines, and resume markers continue to use the transformed training-step coordinates.
- A requested alternate rendering should be appended as a clearly labelled supplement when the user asks to keep the original, not silently replace the established primary artifact.
