"""Nemotron-H layer placement, scaled down to the sigma-0 LOB trunk.

Nemotron's own numbers are the only calibration we have: 4 attention layers out
of 31 sequence-mixing layers, sitting at relative depths {0.250, 0.375, 0.536,
0.696}.  Everything below is that recipe evaluated at our depth, plus the four
placement constraints read off the published pattern.
"""

NEMOTRON_ATTN_FRACTION = 4 / 31          # of sequence-mixing layers, not of all 56
DEPTH_BAND = (0.26, 0.71)                # first and last attention, relative depth


def attention_indices(n_layers):
    k = max(1, round(NEMOTRON_ATTN_FRACTION * n_layers))
    if k == 1:
        depths = [0.48]                  # centroid of Nemotron's four depths
    else:
        lo, hi = DEPTH_BAND
        depths = [lo + (hi - lo) * i / (k - 1) for i in range(k)]
    idx = [min(max(round(d * n_layers), 2), n_layers - 2) for d in depths]
    return sorted(set(idx))


def check(n_layers, idx):
    """The four constraints C1-C4 read off Nemotron's published pattern."""
    out = []
    lead = idx[0]
    need = max(2, -(-n_layers // 5))     # ceil(0.20 * L)
    out.append(("C1 首个 attention 前 >= %d 个递归块" % need, lead, lead >= need))
    tail = n_layers - 1 - idx[-1]
    out.append(("C2 末个 attention 后 >= 2 个递归块", tail, tail >= 2))
    gaps = [b - a - 1 for a, b in zip(idx, idx[1:])]
    out.append(("C3 无相邻 attention", min(gaps) if gaps else "n/a", all(g >= 1 for g in gaps)))
    out.append(("C4 attention 间隔 >= 3 个递归块", min(gaps) if gaps else "n/a", all(g >= 3 for g in gaps)))
    return out


if __name__ == "__main__":
    print(f"{'L':>3} {'k':>2}  {'attention idx':<20} {'C1':>4} {'C2':>4} {'C3':>4} {'C4':>4}")
    print("-" * 52)
    for L in (6, 8, 10, 12, 16, 24, 31, 56):
        idx = attention_indices(L)
        res = check(L, idx)
        flags = "".join(f"{'ok' if ok else 'FAIL':>5}" for _, _, ok in res)
        print(f"{L:>3} {len(idx):>2}  {str(idx):<20}{flags}")

    print()
    print("我们的 trunk: n_fused_layers = 6")
    idx = attention_indices(6)
    print(f"  attention at fused block {idx}")
    for name, val, ok in check(6, idx):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  (实际 {val})")
    print()
    print("如果想要 k=2 会怎样:")
    forced = [2, 4]
    for name, val, ok in check(6, forced):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  (实际 {val})")
