#!/usr/bin/env python3
"""A02 -- does DFM post-training reduce order-flow compound error across the
whole held-out ticker universe, not just GOOG?

Design note. A01 had one ticker and twenty trading days, so the only available
bootstrap block was the day. Here there are ~488 tickers with 8 sequences each,
and the right unit changes. Pooling every ticker into one histogram is wrong:
`price_rel` spans orders of magnitude between a utility and a mega-cap, so a
pooled divergence would charge cross-ticker dispersion as model error. Instead
each ticker gets its own excess slope, and the bootstrap resamples TICKERS.
One ticker's 4-vs-4 cross-half is noisy on its own; 488 of them are not, and
the CI says so honestly.

The estimator itself is A01's, imported rather than reimplemented: it is the
one whose no-op canary passes and whose floor arm is verified flat.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "A01_does_ce_exist", "code"))
from a01_ce_existence import (FIELDS, action_fields, build_counts,   # noqa: E402
                              codes, make_edges, D_curve, ols_slope)


def one_ticker(npz_l, npz_r, tick, bins, w, step, seed):
    """-> {field: {arm: slope_of_excess}}, or None if the file is unusable."""
    try:
        zl = np.load(npz_l, allow_pickle=True)
        F = action_fields(zl, tick)
        arms = {"draft": F["draft"], "corr": F["corr"]}
        if npz_r and os.path.exists(npz_r):
            arms["rand"] = action_fields(np.load(npz_r, allow_pickle=True),
                                         tick)["corr"]
    except Exception as e:
        return None, f"load: {type(e).__name__}"
    N, R = zl["real_msgs"].shape[0], zl["real_msgs"].shape[1]
    if N < 4 or R < 4 * w:
        return None, f"too small n={N} R={R}"
    grid = np.arange(w, R - w, step)
    if grid.size < 4:
        return None, "grid too short"
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    A, B = perm[: N // 2], perm[N // 2:]
    groups = [A, B]                       # one "block" per half; no day strata
    out = {}
    for name in FIELDS:
        try:
            e = make_edges(F["true"][name], F["true"]["_ok"], name, bins)
            cT, K = codes(F["true"][name], F["true"]["_ok"], name, e)
            CT = build_counts(cT, groups, grid, w, R, K)
            tA, tB = CT[0], CT[1]
            fl = D_curve(tA, tB, tA, tB, K)
            row = {}
            for k, Fa in arms.items():
                cG, _ = codes(Fa[name], Fa["_ok"], name, e)
                CG = build_counts(cG, groups, grid, w, R, K)
                row[k] = ols_slope(grid, D_curve(tA, tB, CG[0], CG[1], K) - fl)
            row["floor"] = ols_slope(grid, fl)
            row["illegal_draft"] = float(100 * (~arms["draft"]["_ok"]).mean())
            row["illegal_corr"] = float(100 * (~arms["corr"]["_ok"]).mean())
            out[name] = row
        except Exception as ex:
            out[name] = {"error": f"{type(ex).__name__}: {ex}"[:120]}
    return out, None


def boot_ci(v, n_boot, rng):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size < 8:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    m = v[idx].mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--prefix", default="dfm_a02_",
                    help="产物名前缀；换 checkpoint 时它带着 checkpoint 身份，"
                         "所以分析必须能指定，否则会去 glob 上一个残差的 npz")
    ap.add_argument("--month", default="2026-01")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tick", type=int, default=100)
    ap.add_argument("--bins", type=int, default=40)
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260813)
    a = ap.parse_args()

    pat = os.path.join(a.rollouts, f"{a.prefix}*_{a.month}_learned.npz")
    per, skipped = {}, {}
    files = sorted(glob.glob(pat))
    print(f"[A02] {len(files)} learned npz under {a.rollouts}", flush=True)
    for i, f in enumerate(files):
        tk = os.path.basename(f).replace(a.prefix, "").replace(
            f"_{a.month}_learned.npz", "")
        r = f.replace("_learned.npz", "_random.npz")
        res, err = one_ticker(f, r, a.tick, a.bins, a.window, a.step, a.seed)
        if res is None:
            skipped[tk] = err
        else:
            per[tk] = res
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(files)}  ok={len(per)} skipped={len(skipped)}",
                  flush=True)
    print(f"[A02] usable {len(per)} tickers, skipped {len(skipped)}", flush=True)

    rng = np.random.default_rng(a.seed + 1)
    agg = {}
    hdr = (f"\n{'field':12s}{'n_tk':>6s}{'draft':>10s}{'corr':>10s}{'rand':>10s}"
           f"{'floor':>9s}{'corr-draft':>12s}{'95% CI':>22s}{'verdict':>9s}"
           f"{'better':>8s}")
    print(hdr, flush=True)
    print("-" * (len(hdr) - 1), flush=True)
    def _g(t, name, k):
        # A ticker whose random arm has not been generated yet yields None,
        # and np.isfinite(None) raises rather than returning False. Coerce
        # here so a half-finished control arm degrades to "fewer paired
        # tickers" instead of killing the whole aggregation.
        x = per[t][name].get(k)
        return float("nan") if x is None else float(x)

    for name in FIELDS:
        d = [_g(t, name, "draft") for t in per if "error" not in per[t][name]]
        c = [_g(t, name, "corr") for t in per if "error" not in per[t][name]]
        rd = [_g(t, name, "rand") for t in per if "error" not in per[t][name]]
        fl = [_g(t, name, "floor") for t in per if "error" not in per[t][name]]
        pair = np.array([(x, y) for x, y in zip(d, c)
                         if np.isfinite(x) and np.isfinite(y)])
        pr = np.array([(x, y) for x, y in zip(d, rd)
                       if np.isfinite(x) and np.isfinite(y)])
        dc, lo, hi = boot_ci(pair[:, 1] - pair[:, 0], a.n_boot, rng) \
            if pair.size else (np.nan,) * 3
        drd, rlo, rhi = boot_ci(pr[:, 1] - pr[:, 0], a.n_boot, rng) \
            if pr.size else (np.nan,) * 3
        better = float((pair[:, 1] < pair[:, 0]).mean()) if pair.size else np.nan
        v = "REDUCED" if hi < 0 else ("WORSE" if lo > 0 else "n.s.")
        agg[name] = {"n_tickers": int(len(pair)),
                     "mean_draft": float(np.nanmean(d)) if d else None,
                     "mean_corr": float(np.nanmean(c)) if c else None,
                     "mean_rand": float(np.nanmean(rd)) if rd else None,
                     "mean_floor": float(np.nanmean(fl)) if fl else None,
                     "corr_minus_draft": dc, "ci": [lo, hi], "verdict": v,
                     "frac_tickers_improved": better,
                     "rand_minus_draft": drd, "rand_ci": [rlo, rhi]}
        print(f"{name:12s}{len(pair):6d}{np.nanmean(d):+10.4f}{np.nanmean(c):+10.4f}"
              f"{(np.nanmean(rd) if rd else np.nan):+10.4f}"
              f"{np.nanmean(fl):+9.4f}{dc:+12.4f}"
              f"{f'[{lo:+.4f},{hi:+.4f}]':>22s}{v:>9s}{100*better:7.1f}%",
              flush=True)

    json.dump({"month": a.month, "prefix": a.prefix, "n_tickers": len(per), "skipped": skipped,
               "window": a.window, "step": a.step, "bins": a.bins,
               "n_boot": a.n_boot, "seed": a.seed,
               "aggregate": agg, "per_ticker": per},
              open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
