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
