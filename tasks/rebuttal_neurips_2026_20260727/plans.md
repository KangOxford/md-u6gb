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
