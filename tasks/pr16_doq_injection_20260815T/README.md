# PR #16 do(Q) injection — GPU verification record (2026-08-16)

\skim: 四臂同节点实测 PR#16 两条 Not-verified 主张:off 臂逐字节=pre-PR(41 文件),
注入单在第 8 步精确落书(+777@mid−1tick,4/4 窗口),模型消息 9/10/10/11 步才分歧。
J0–J5 全过。产物已作为 evidence + GitHub Action 进入 PR 分支。

## What was measured

The two claims PR #16 itself listed as "Not verified": (1) the off arm reproduces
the pre-PR program byte-for-byte, and (2) the injected order appears in the book
at the flagged step. Four arms (P = merge-base, P2 = P rerun as determinism
control, O = PR off, I = PR inject step 8, qty 777, bid, mid−1tick) on one node,
one GPU (nid010434 / GPU0, D-X7: byte-identity is node-bound), checkpoint
m3-goog-78m-u6gb@28730 (pin hash verified), data GOOG 2026-02-27, seed 42,
4 windows × 16 cond × 32 gen.

Result: P = P2 = O identical tree digests (c07658e2…) over 41 files; I differs
(f7e3b455…); books first diverge exactly at step 8 in 4/4 windows; the L2 delta
between I's rows 7→8 is exactly +777 at the derived price (1 join, 3
improve-best, ask side untouched).

## Artifacts

- `code/run_four_arms.sh`, `code/launch_driver.sh` — the driver (final versions;
  runtime artifacts live in `/home/u6gb/kangli.u6gb/pr16_doq_artifacts_20260816/`)
- `injection_report.json` — the fixed-template log (sigma0.injection_verification.v1)
- In the PR branch: `ci/evidence/injection.json` (13 assertions),
  `.github/workflows/injection.yml` (4-guard action),
  `tools/diagnostics/verify_injection.py --check <report>` re-audits anywhere.

## Pointers

- PR: https://github.com/KangOxford/sigma-0/pull/16 (commits 41ec928 / 97b2db6 / 1788266)
- Results comment: …/pull/16#issuecomment-5305222317
- Action detail comment: …/pull/16#issuecomment-5305297855
- 42/42 per-file review comments posted 2026-08-16.

Gotchas hit on the way (memories written): u6gb project inode quota exhaustion
(51.2M files — new file creation fails project-wide, Edit tool included);
GOOG2016TO2021's 41-column books fail the wide-book contract ("width 38") — use
GOOG_data_npy (43-column); shared tmux pane send-keys is not a launch mechanism.
