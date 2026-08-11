# 26-token failure audit and 27th-pointer/L3 design

This is a read-only design audit. It did not modify model code, submit training,
cancel or alter Slurm jobs, or create an R7/27-token worktree.

## Existing 26-token resolver

On the paired 255-sequence generated set, cancel/delete resolution has:

- L1 exact timestamp: 23,466 / 31,142 = 75.3516%.
- L2 current-price nearest-millisecond fallback: 6,475 / 31,142 = 20.7919 percentage points.
- Final unresolved: 1,201 / 31,142 = 3.8565%.
- Combined live-ID success: 29,941 / 31,142 = 96.1435%.

L2 does not predict nanoseconds correctly. It first requires a live order at the
generated current price, then selects the closest timestamp after reducing time
to milliseconds, with no distance threshold. In 53.0502% of L2 hits only one
order exists at that price, so time does not affect the choice.

## What the 1,201 failures are

- 1,180 are delete events and 21 are partial cancels.
- All 1,201 miss exact timestamp and have no live order at the generated current
  price, although the declared-side L10 book is nonempty.
- 479 / 1,201 = 39.88% have a visible same-price NEW in condition or generated
  history, but that level is stale by cancellation time.
- 722 / 1,201 = 60.12% have no visible same-price NEW, consistent with a target
  before the visible window or a hallucinated/shifted price.
- Nearest displayed price is within 1 tick for 42.13%, within 10 ticks for
  53.79%, and within 100 ticks for 80.27%. Nearest-price L3 can produce a live
  ID, but it silently changes the referenced economic object and is therefore
  not a safe semantic resolver.

## Pointer oracle on real paired continuations

For 30,925 real cancel/delete events, the target NEW is visible in the condition
plus prior continuation for 27,269 events = 88.1779%. For 705 executions, it is
visible for 520 = 73.7589%.

Three one-token pointer labels were compared on visible cancel/delete events:

| Pointer label | Observed range | Entropy | Top-1 frequency | Assessment |
|---|---:|---:|---:|---|
| Absolute message/data index | 0..498 | 8.5421 bit | 0.433% | Fits one token, but position-specific and hard to learn |
| Backward message distance | 1..486 | 6.9938 bit | 8.126% | Better, but counts irrelevant non-NEW events |
| Reverse NEW rank | 1..241 | 6.0525 bit | 10.374% | Best of the three; stable semantic candidate |

The existing vocabulary already contains a 0..999 token range, so a 27th
position can reuse token IDs without enlarging the vocabulary. A 500-message
window becomes 13,500 rather than 13,000 tokens, a 3.846% increase. Holding the
13,000-token cap permits 481 complete 27-token messages.

## Success-rate bounds

If the pointer is used only after existing L1/L2, combined success is:

    96.1435% + 3.8565% * conditional_pointer_recovery

| Conditional recovery on the 1,201 misses | Combined live-ID success |
|---:|---:|
| 25% | 97.1076% |
| 50% | 98.0717% |
| 75% | 99.0359% |
| 88.1779% real-data visibility oracle | 99.5441% |
| 100% | 100.0000% |

The 99.5441% figure is an optimistic visibility oracle, not trained-model
accuracy. A trained predictor must be evaluated separately on the hard 1,201
events. Reaching 97%, 98%, 99%, and 99.5% overall requires conditional recovery
of 22.21%, 48.14%, 74.07%, and 87.03%, respectively.

## Candidate designs

### A. 27-token compatibility L3

Keep exact-time L1 and same-price/millisecond L2 unchanged. Invoke the pointer
only on the 1,201 final misses. This preserves the published 96.1435% resolver,
but cannot correct any wrong object selected among the 6,475 L2 hits.

### B. 27-token pointer-first (recommended)

Predict reverse NEW rank. Resolve it to the raw historical row and order ID,
require that ID to be live and on the declared side, repair event price from the
referenced live order, and validate or clamp cancellation/execution quantity.
Then fall back to exact timestamp and finally same-price closest millisecond.
This can improve semantic correctness as well as live-ID success.

### C. 26-token repurpose

Repurpose an existing resolver-unused reference field instead of adding a
position. This preserves the 13,000-token window but removes that field's old
meaning. It still requires retraining and a clean ablation, and is not preferred
if price and size are retained as consistency checks.

No implementation should start until the user selects A, B, or C.
