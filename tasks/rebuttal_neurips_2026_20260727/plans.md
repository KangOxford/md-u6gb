# Plans

- Completed: answer every reviewer question directly below the exact prompt, one Notion update and verification at a time.
- Preserve the submitted 8M-197M protocol separately from the later SP500/0.2M-350M artifacts in all paper edits and result tables.
- Revise the manuscript in `Overleaf_codex` to:
  - replace universal/causal language with finite-range logged-training-loss scope;
  - remove the categorical TradeFM/MarS “undertrained” assertion and the unqualified “nearly five times” headline;
  - add the common-D, leave-one-run-out, fixed-beta, profile, and IsoFLOP-deletion sensitivity results;
  - state the held-out, optimizer, repeated-profiler, and Transformer evidence gaps explicitly.
- Add only defensible descriptive TradeFM/MarS coordinates; do not present cross-tokenizer raw-token ratios as a measured frontier comparison.
- Compile and inspect the revised PDF, then commit each manuscript/figure modification batch immediately on `codex-rebuttal`.
- Do not push the Overleaf branch until explicitly requested, because another agent is editing a separate copy and remote conflict avoidance is part of the task.

## Status plan — 2026-07-27 13:19 UTC

- Completed: revise `Overleaf_codex` to distinguish the earlier locked six-size test-CE diagnostic, the later different-sweep free refit, and the still-missing checkpoint-matched refit of the submitted ten-run surface.
- Completed: add finite-range/common-D/fixed-beta/LOO/profile/IsoFLOP sensitivity evidence, theory caveats, optimizer/Transformer gaps, and descriptive TradeFM/MarS coordinates.
- Completed: rerun the sensitivity analysis and perform active-reference, citation, environment, brace, math-delimiter, JSON, and graphics-path static checks.
- Remaining external validation: compile on Overleaf and inspect floats, page limit, and warnings; no local TeX engine is installed.
- Remaining scientific validation: evaluate the frozen submitted-grid checkpoints on the Jan-2026 stream and freely refit all five scaling-law parameters. Do not describe the current partial result as that matched experiment.
- Keep `/lus/lfs1aip2/projects/public/u6gb/Overleaf_codex` isolated and do not push until the user explicitly requests it.
- Recommend rotating the plaintext Overleaf credentials visible on the Notion source block; never reproduce them in chat or records.

## Gamma clarification plan — 2026-07-27 13:31 UTC

- Completed: trace `gamma=0.90` to the exact historical profiler CSV, reproduce the log-log OLS fit, and record its input range, coefficient, intercept-derived prefactor, `R^2`, and SHA256.
- Completed: update and re-fetch the existing pXiP Q3 blue Notion callout with this provenance.
- No manuscript edit was required for the user’s explanatory question; the current paper already reports the 12-point profile and rounded exponent.
- If a measurement-error interval is required, collect repeated profiler measurements per size. The existing residual bootstrap and leave-one-profile-point-out checks quantify fit/design sensitivity, not repeated-measurement uncertainty.

## Sequential-answer plan — 2026-07-28 00:12 UTC

- Completed: write the consolidated Q1/common-D explanation into the exact task-list anchor and verify the rendered anchor rather than relying on the parent-page update response.
- Keep the Q1 checkbox unchecked until the user explicitly confirms understanding.
- Wait before starting pXiP Q2. The next experiment remains the checkpoint-matched free refit on Jan-2026 test CE, and its absence must not be hidden by the earlier locked-exponent figure or the later different-sweep fit.
- Make no further `Overleaf_codex` manuscript edits and do not push while this sequential Notion clarification is awaiting user review.

## Result-table plan — 2026-07-29 14:26 UTC

- Completed: locate the exact bracketed table instruction on the supplied Rebuttal page and update that location rather than appending elsewhere.
- Completed: use `aramis/results/selected_test_endpoint.csv` as the raw 33-row source and `aramis/results/main_fits.csv` plus bootstrap summaries for the compact fit table.
- Completed: preserve all 33 currently used observations in a 12-size audit table with actual `N`, evaluated `D`, per-seed January-2026 held-out CE, and cross-seed mean/sample SD.
- Completed: disclose that “terminal” currently means final available evaluated checkpoint and that `46M-s5` is included despite not reaching its target; distinguish 33 manifest-selected rows from 32 strict target-reaching rows.
- Completed: disclose the actual 2.048-decade parameter span, missing `23M-s42`, nominal `350M` label, excluded finisher, and absence of valid completed Transformer points.
- Completed: re-fetch the Notion page and verify the original instruction is struck through and the blue callout remains directly below it.
- No refit, manifest repair, experiment launch, manuscript edit, or Transformer result substitution is authorized by this round. If the user requests a strict terminal estimate, first correct the cohort to 32 rows and regenerate every dependent fit/bootstrap artifact as a separately committed analysis batch.
