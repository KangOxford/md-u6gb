#!/usr/bin/env python3
"""A01b -- why did A01 report no compound error on `price_rel`?

A01 measured price_rel and found no significant slope at any horizon. That
contradicts the prior that price, being a level relative to a FIXED anchor
(ref_mid, one constant per sequence), should accumulate more readily than any
other field. This file checks the instrument rather than the model.

Three measurements, none of which pass through the binned divergence:

  1. out-of-support fraction of price_rel vs position m
  2. median |price_rel| vs m, model against truth -- does the model TRACK the
     drift of the real price at all?
  3. the arithmetic of how much a binned forward KL attenuates (1)

Nothing here is a divergence. That is the point: A01's verdict on price_rel
came from a divergence, so a control for it must not use one.
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0] + "/../../A01_does_ce_exist/code")
from a01_ce_existence import action_fields, make_edges  # noqa: E402


def curves(F, arm, grid, w, lo, hi):
    v, ok = F[arm]["price_rel"], F[arm]["_ok"]
    oos, med, p99, sd = [], [], [], []
    for m in grid:
        sl = slice(m - w, m + w + 1)
        x = v[:, sl][ok[:, sl]]
        x = x[np.isfinite(x)]
        if not x.size:
            oos.append(np.nan); med.append(np.nan); p99.append(np.nan)
            sd.append(np.nan); continue
        oos.append(float(((x < lo) | (x > hi)).mean()))
        a = np.abs(x)
        med.append(float(np.median(a))); p99.append(float(np.percentile(a, 99)))
        sd.append(float(x.std()))
    return dict(oos=oos, med_abs=med, p99_abs=p99, sd=sd)


def slope(grid, y, per=100.0):
    g, y = np.asarray(grid, float), np.asarray(y, float)
    m = np.isfinite(y)
    return float(np.polyfit(g[m], y[m], 1)[0] * per) if m.sum() > 3 else np.nan


def attenuation(d0, d1, span, K, n_true, H):
    """How much of an out-of-support blow-up survives a binned divergence?"""
    p = (1.0 / K) / (n_true + 1.0)          # smoothed TRUE prob of the OOS bin
    return {
        "kl_term_at_d0": float(p * np.log(p / d0)),
        "kl_term_at_d1": float(p * np.log(p / d1)),
        "p_true_oos": float(p),
        # the only residual in forward KL is renormalisation of the in-support
        # bins by (1-delta); the OOS bin itself is weighted by p_true ~ 0
        "kl_slope": float(100 * (-np.log(1 - d1) + np.log(1 - d0)) / H / span),
        # JS charges q*log(q/m) even where p = 0
        "js_slope": float(100 * (0.5 * np.log(2) * (d1 - d0)) / H / span)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tick", type=int, default=100)
    ap.add_argument("--bins", type=int, default=40)
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--step", type=int, default=20)
    a = ap.parse_args()

    z = np.load(a.npz, allow_pickle=True)
    F = action_fields(z, a.tick)
    R, N = z["real_msgs"].shape[1], z["real_msgs"].shape[0]
    w = a.window
    grid = np.arange(w, R - w, a.step)
    e = make_edges(F["true"]["price_rel"], F["true"]["_ok"], "price_rel", a.bins)
    lo, hi = float(e[0]), float(e[-1])

    print(f"[A01b] {a.npz.rsplit('/',1)[-1]}  n_seq={N} n_gen={R}")
    print(f"       true support (quantile edges) = [{lo:.1f}, {hi:.1f}] ticks\n")

    res = {"support": [lo, hi], "grid": grid.tolist(), "arms": {}}
    print(f"{'arm':7s}{'oos@start':>11s}{'oos@end':>10s}{'oos slope':>12s}"
          f"{'med|p| start':>14s}{'med|p| end':>12s}{'med slope':>12s}")
    for arm in ("true", "draft", "corr"):
        c = curves(F, arm, grid, w, lo, hi)
        c["oos_slope_pp"] = slope(grid, c["oos"]) * 100
        c["med_slope"] = slope(grid, c["med_abs"])
        c["p99_slope"] = slope(grid, c["p99_abs"])
        res["arms"][arm] = c
        print(f"{arm:7s}{100*c['oos'][0]:10.2f}%{100*c['oos'][-1]:9.2f}%"
              f"{c['oos_slope_pp']:+11.4f}pp{c['med_abs'][0]:14.0f}"
              f"{c['med_abs'][-1]:12.0f}{c['med_slope']:+12.2f}")

    t = res["arms"]["true"]["med_slope"]
    print(f"\n  drift tracking (median |price_rel| slope / the true slope):")
    for arm in ("draft", "corr"):
        print(f"    {arm:6s} {res['arms'][arm]['med_slope']:+8.2f} / {t:+.2f} "
              f"= {100*res['arms'][arm]['med_slope']/t:6.1f}%")

    d0 = res["arms"]["draft"]["oos"][0]
    d1 = res["arms"]["draft"]["oos"][-1]
    res["attenuation"] = attenuation(d0, d1, grid[-1] - grid[0], 42, N * (2 * w + 1),
                                     3.52)
    at = res["attenuation"]
    print(f"\n  a binned divergence sees this much of the {100*d0:.2f}% -> "
          f"{100*d1:.2f}% blow-up:")
    print(f"    p_true(out-of-support bin) = {at['p_true_oos']:.2e}  (0 by "
          f"construction, the support IS the true quantiles)")
    print(f"    forward-KL slope {at['kl_slope']:+.5f} /100    "
          f"JS slope {at['js_slope']:+.5f} /100")

    json.dump({"npz": a.npz, "n_seq": int(N), "n_gen": int(R), "res": res},
              open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
