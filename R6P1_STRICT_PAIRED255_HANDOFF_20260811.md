# R6.1 strict paired-255 handoff — 2026-08-11

## Outcome

The project goal is achieved by R6.1: the exact-lossless variable-length model
at optimizer step 9,000 is strictly better than the paired 26-token baseline on
all three aggregate LOB-Bench metrics over the same 255 sequences.

| Metric | R6.1 | 26-token | Relative improvement | Strict win |
|---|---:|---:|---:|:---:|
| WS-21 | 0.1620314331 | 0.2384903328 | 32.06% | yes |
| KS-21 | 0.1047934124 | 0.1164126896 | 9.98% | yes |
| L1-21 | 0.1670438261 | 0.1934201063 | 13.64% | yes |

The optimizer checkpoint step was fixed before scores were inspected.

## Evidence and gates

- Result JSON:
  `/lus/lfs1aip2/projects/public/u6gb/tasks/varlen_bench_subset_20260809/result_r6p1_optsteps_s9000_paired255_semanticreject_20260811T170910Z.json`
- Result SHA256:
  `96de13bfa9848f7e8bbfb3d866a01c456626bfa75cb6b7a9b85ca947eccea333`
- Exact generation runtime:
  `/lus/lfs1aip2/projects/public/u6gb/tasks/varlen_bench_subset_20260809/r6p1_bench_supervision/r6p1_optsteps_s9000_paired255_semanticreject_20260811T170910Z`
- Generation audit SHA256:
  `70da057ff077f700d19f07dae417f2bcceafd6b46f1dbf010bf174007dc6fd1b`
- Condition-pool SHA256:
  `9ae74b7d16466a3da2df5bde20aea941bc7274adff2c9fcaaca63f6c5b8ad648`
- Vocabulary SHA256:
  `5bb6af52f7ff3cd74e14fae16e86c8b4e471c6ac6725c180cababeb6eaf6f308`
- Vocabulary validation SHA256:
  `ef2bef2385431b824946f20ee5bdc1129b3b0ae75052eeb0c9c995d53034aaa0`
- Vocabulary validation covered 160,660,113,046 rows and 467,217 pairs;
  all lossless gates passed.
- Exact paired roundtrip audit covered 255 sequences, 63,750 messages, and
  401,613 tokens with zero exact failures. Audit SHA256:
  `919b797bacea61cd60c3ec96aa9eaf8284be305f39fe20174861a817975d4542`.
- Generation produced 255 message files, 255 order-book files, and 255
  provenance files; the audit reported no missing or failed sequences.

The final shard-log audit found 25 non-canonical candidate attempts and one
semantic negative-price candidate attempt. All 26 attempts were rejected before
state mutation and retried within the eight-attempt bound. This is 0.0408% of
63,750 accepted messages. There were zero hard-decode failures, zero stateful
rejections after the precheck, zero retry exhaustions, and zero failed
sequences. An earlier live monitor incorrectly globbed `n*.log`; the actual
files are `shard*.log`, and this handoff records the corrected count.

## Code and report

- Semantic-rejection implementation worktree:
  `/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/varlen-R6p1-bench-semantic-reject-20260811T1707Z`
- Implementation commit:
  `9147430bf4db948e7f1684dd95479a9c64b108e7`
- Report worktree:
  `/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/varlen-R6p1-report-20260811T044128Z`
- Report renderer commit:
  `5e02a5e6c02fbaf16316a3c63b8e8ddf46b427e0`
- Report artifacts:
  `/lus/lfs1aip2/projects/public/u6gb/tasks/varlen_bench_subset_20260809/r6p1_bench_supervision/r6p1_optsteps_s9000_paired255_semanticreject_20260811T170910Z/report_notion_20260811T1724Z`
- Notion page, updated and re-fetched successfully:
  `https://app.notion.com/p/3b812c4568fd8145a418c3dc0a0f42d6`

## R6.2 live state (not part of the R6.1 claim)

At `2026-08-11T17:26:42Z`, job `5980745` was running on four nodes and
sixteen GPUs at optimizer step 6,367 of 32,000, with checkpoint 6,000 present.
All four ranks were synchronized; the vocabulary, roundtrip, and loss-mask
gates were passed; the supervisor reported no alerts. R6.2 remains an ongoing
follow-up and must not be promoted without its own exact paired-255 strict
three-metric win.

Supervisor state:
`/lus/lfs1aip2/projects/public/u6gb/tasks/varlen_bench_subset_20260809/r6p2_train_supervision/r6p2_lossmask_longctx32k_optsteps_recovery_20260811T155248Z/state.json`

## Repository hygiene

The shared `findings.md`, `plans.md`, `learnt_lessons.md`, and `progress.md`
already contained unrelated uncommitted edits at handoff time. They were not
modified or staged. This dedicated record is intentionally isolated so its
commit cannot absorb those concurrent changes.
