# T-nanosecond, price-size, and data-index reference cascade

Status: read-only diagnostics complete. No model code, worktree, training job, or
running Slurm allocation was changed.

## Direct answer

The proposed cascade is sound:

1. Resolve by exact reference creation time at nanosecond precision.
2. If that fails, require a unique candidate under declared side, reference
   price, and original reference size.
3. If that also fails, use a predicted historical data index constrained to a
   NEW row on the declared side whose order is still live.

However, "nearly 100%" must be split into two metrics:

- any-live-ID success can be forced to 100% on the frozen 26-token residual
  misses with a constrained pointer;
- exact-target-ID accuracy is not proven by that operation.

## Data-side oracle

The paired-255 real continuation has 31,630 cancel/delete/execution touches.
The target NEW is visible in condition plus earlier continuation for 27,789,
or 87.8565%.

Among those visible targets, exact reference time at nanosecond precision is
unique for 27,789/27,789 = 100%. Exact time alone therefore resolves every
target that an ordinary history-row pointer could address. The remaining 3,841
targets predate the condition window and have no historical row to point to.

Declared side plus reference price plus original reference size is unique for
23,982/27,789 = 86.3003% of visible touches. It is useful as an independent
error-correction channel when model-predicted time is wrong, but does not add
oracle coverage after a perfectly predicted nanosecond time.

For cancel/delete alone, visibility is 27,269/30,925 = 88.1779%.
Side-price-original-size is unique for 23,521/27,269 = 86.2555% of visible
cancel/delete targets.

## Frozen 26-token output

Current resolution is exact-nanosecond L1 at 75.3516%, followed by current-price
nearest-millisecond L2 at +20.7919 percentage points, for 96.1435% combined.
There are 1,201 residual misses.

Raw-history replay shows that every one of the 1,201 residual misses still has
at least five live same-side historical NEW candidates. Candidate count has
median 21, p90 42, and maximum 294. Hence a mask that permits only live
same-side NEW indices can always produce some live ID and mechanically lift the
operational metric to 100%.

That is not an exact-reference result: with typically 21 candidates, choosing
one legal index can cancel the wrong order. The frozen artifact also does not
persist the raw reference-size prediction, so the actual incremental recovery
of a price-plus-size L2 cannot be reconstructed. In 986/1,201 residual misses,
the persisted diagnostic says reference price differs from event price; this is
evidence of an independent signal or an independent error, not proof of a hit.

## Recommended representation

For BPE, preserve exact current-event delta time and add exact reference age in
nanoseconds, which is equivalent to reference creation time but session-shift
invariant. Keep touch quantity separate from original reference size.

For a conservative 27-token baseline, add one historical row index in range
0..498 plus an OUTSIDE_WINDOW sentinel. Apply it only after existing L1 and the
new unique price-plus-original-size L2. Mask invalid, non-NEW, wrong-side, and
non-live rows.

During the pilot, compute the index even when an earlier level succeeds and
record agreement, but do not let it override the old resolver. Report:

- exact target order-ID accuracy on paired real data;
- any-live-order-ID success;
- incremental L1, L2, and L3 recovery;
- invalid and outside-window indices;
- resolver conflicts and wrong-target rate.

Do not call masked-pointer 100% validity "100% reference prediction."
