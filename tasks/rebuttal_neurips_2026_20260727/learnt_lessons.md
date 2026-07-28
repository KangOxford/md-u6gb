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
