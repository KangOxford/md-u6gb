#!/usr/bin/env python3
"""A01 self-check: does the estimator do what PLAN.md claims?

Controls, each isolating one way the pipeline could manufacture or hide a
result. All run on the SAME primitives as the real measurement, on a real npz
(so marginals and day structure are realistic), with the generated arm
replaced by a synthetic stream whose defect is known exactly.

  C6   no-op canary    gen := true                    -> excess must be 0
  C4   constant        shift d0 of bins, flat in m     -> level up, slope ~ 0
  C5   linear drift    shift d(m)=d*(m/R) of bins      -> slope detectable;
                                                          yields threshold d*
  C3   shuffle         true, permuted along m per row  -> slope back to floor
  C4b  over-dispersion replace d0 by a uniform draw     -> demonstrates that
                                                          forward KL can go
                                                          NEGATIVE

Injection acts in BIN-CODE space: with probability d(m) an element's bin code
moves one bin up, cyclically among the valid bins. That is a known-strength,
support-preserving deformation that works identically for ordinal and
categorical fields. The first version of this file injected "replace with a
uniform draw over the support" instead, which is nearly the identity for a
near-uniform binary field like `direction` (undetectable at d=0.40) and which
WIDENS the generated support -- and forward KL rewards that, so the measured
excess went negative. C4b keeps that probe deliberately, as the demonstration.
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from a01_ce_existence import (FIELDS, KIND, action_fields, build_counts,  # noqa: E402
                              codes, make_edges, D_curve, ols_slope)


def shift_codes(c, K_total, frac_m, rng):
    """With prob frac_m(m), move a valid bin code one bin up, clamped at the top.

    NOT cyclic. A cyclic shift preserves uniformity, and quantile bins make the
    true distribution uniform BY CONSTRUCTION, so the cyclic version is a
    distributional no-op on every quantile-binned field -- it was undetectable
    at delta=0.40 for price_rel and log10_dt for exactly that reason. Clamping
    piles mass at the top bin, which is what a drifting generator actually does.
    """
    kv = K_total - 2                      # last two codes: out-of-support, illegal
    hit = (rng.random(c.shape) < frac_m[None, :]) & (c < kv)
    out = c.copy()
    out[hit] = np.minimum(c[hit] + 1, kv - 1)
    return out


def collapse_codes(c, K_total, frac_m, rng):
    """With prob frac_m(m), replace a valid code by bin 0 (mode collapse).

    Used for the power scan. A one-bin shift of a uniform distribution is a
    boundary effect only (KL ~ delta^2 / K), so it is a very weak probe; mode
    collapse moves mass non-locally and is the realistic failure mode for a
    generator that has drifted off-distribution.
    """
    kv = K_total - 2
    hit = (rng.random(c.shape) < frac_m[None, :]) & (c < kv)
    out = c.copy()
    out[hit] = 0
    return out


def widen_codes(c, K_total, frac_m, rng):
    """C4b: with prob frac_m(m), replace a valid code by a uniform valid code."""
    kv = K_total - 2
    hit = (rng.random(c.shape) < frac_m[None, :]) & (c < kv)
    out = c.copy()
    out[hit] = rng.integers(0, kv, size=int(hit.sum()))
    return out


def estimate(cT, cG, K, groups, nd, grid, w, R, draws):
    CT = build_counts(cT, groups, grid, w, R, K).reshape(nd, 2, len(grid), K)
    CG = build_counts(cG, groups, grid, w, R, K).reshape(nd, 2, len(grid), K)
    tA, tB, gA, gB = CT[:, 0], CT[:, 1], CG[:, 0], CG[:, 1]
    sT = (tA.sum(0), tB.sum(0))
    exc = (D_curve(*sT, gA.sum(0), gB.sum(0), K) - D_curve(*sT, *sT, K))
    exc_js = (D_curve(*sT, gA.sum(0), gB.sum(0), K, div="js")
              - D_curve(*sT, *sT, K, div="js"))
    acc = np.empty(len(draws)); acc_js = np.empty(len(draws))
    for b, d in enumerate(draws):
        a_, b_ = tA[d].sum(0), tB[d].sum(0)
        c_, d_ = gA[d].sum(0), gB[d].sum(0)
        acc[b] = ols_slope(grid, D_curve(a_, b_, c_, d_, K)
                           - D_curve(a_, b_, a_, b_, K))
        acc_js[b] = ols_slope(grid, D_curve(a_, b_, c_, d_, K, div="js")
                              - D_curve(a_, b_, a_, b_, K, div="js"))
    return {"slope": ols_slope(grid, exc), "level": float(np.nanmean(exc)),
            "ci": [float(np.nanpercentile(acc, 2.5)),
                   float(np.nanpercentile(acc, 97.5))],
            "max_abs": float(np.nanmax(np.abs(exc))),
            "slope_js": ols_slope(grid, exc_js),
            "level_js": float(np.nanmean(exc_js)),
            "ci_js": [float(np.nanpercentile(acc_js, 2.5)),
                      float(np.nanpercentile(acc_js, 97.5))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tick", type=int, default=100)
    ap.add_argument("--bins", type=int, default=40)
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260812)
    a = ap.parse_args()

    z = np.load(a.npz, allow_pickle=True)
    F = action_fields(z, a.tick)["true"]
    N, R = z["real_msgs"].shape[0], z["real_msgs"].shape[1]
    days = np.asarray(z["dates"]); uday = np.unique(days); nd = len(uday)
    w = a.window
    grid = np.arange(w, R - w, a.step)
    rng = np.random.default_rng(a.seed)
    half = np.zeros(N, np.int64)
    for d in uday:
        r = np.flatnonzero(days == d); r = r[rng.permutation(r.size)]
        half[r[: r.size // 2]] = 1
    groups = [np.flatnonzero((days == d) & (half == h)) for d in uday for h in (0, 1)]
    draws = np.random.default_rng(a.seed + 1).integers(0, nd, size=(a.n_boot, nd))
    cr = np.random.default_rng(a.seed + 77)

    CODE = {}
    for name in FIELDS:
        e = make_edges(F[name], F["_ok"], name, a.bins)
        CODE[name] = codes(F[name], F["_ok"], name, e)      # (codes, K)

    out, fail = {}, []
    print(f"[selfcheck] {a.npz.rsplit('/',1)[-1]}  n_seq={N} n_gen={R} "
          f"days={nd} grid={grid.size}  K=" +
          " ".join(f"{n[:5]}:{CODE[n][1]}" for n in FIELDS), flush=True)

    # ---------------------------------------------------------------- C6
    print("\nC6  no-op canary (gen := true) -- excess must be identically 0")
    for name in FIELDS:
        cT, K = CODE[name]
        r = estimate(cT, cT, K, groups, nd, grid, w, R, draws[:20])
        ok = r["max_abs"] < 1e-12
        print(f"    {name:12s} max|excess|={r['max_abs']:.3e}  "
              f"slope={r['slope']:+.3e}  {'PASS' if ok else 'FAIL'}")
        out.setdefault("C6", {})[name] = {**r, "pass": ok}
        if not ok:
            fail.append(f"C6:{name}")
    if fail:
        json.dump({"fail": fail, "res": out}, open(a.out, "w"), indent=1)
        print(f"\nCANARY FAILED {fail} -- refusing to report numbers")
        return 3

    # ---------------------------------------------------------------- C4
    print("\nC4  constant bin-shift -- level must rise, slope must stay ~0")
    for d0 in (0.02, 0.05, 0.10, 0.20):
        row = {}
        for name in FIELDS:
            cT, K = CODE[name]
            r = estimate(cT, shift_codes(cT, K, np.full(R, d0), cr), K,
                         groups, nd, grid, w, R, draws)
            r["false_positive"] = bool(r["ci"][0] > 0 or r["ci"][1] < 0)
            row[name] = r
        out.setdefault("C4", {})[str(d0)] = row
        bad = [n for n in FIELDS if row[n]["false_positive"]]
        print(f"    d0={d0:<5} level=" +
              " ".join(f"{n[:5]}:{row[n]['level']:+.4f}" for n in FIELDS) +
              f"   false-positive slope: {bad if bad else 'NONE'}")

    # ---------------------------------------------------------------- C5
    print("\nC5  linear mode-collapse drift -- scan strength, get the threshold d*")
    ramp = np.arange(R) / float(R - 1)
    for dd in (0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40):
        row = {}
        for name in FIELDS:
            cT, K = CODE[name]
            r = estimate(cT, collapse_codes(cT, K, dd * ramp, cr), K,
                         groups, nd, grid, w, R, draws)
            r["detected"] = bool(r["ci"][0] > 0)
            r["detected_js"] = bool(r["ci_js"][0] > 0)
            row[name] = r
        out.setdefault("C5", {})[str(dd)] = row
        det = [n for n in FIELDS if row[n]["detected"]]
        print(f"    delta={dd:<6} KLslope=" +
              " ".join(f"{n[:5]}:{row[n]['slope']:+.4f}" for n in FIELDS) +
              f"   detected {len(det)}/5 {det}")

    thr, thr_js = {}, {}
    for name in FIELDS:
        h = [float(k) for k in out["C5"] if out["C5"][k][name]["detected"]]
        hj = [float(k) for k in out["C5"] if out["C5"][k][name]["detected_js"]]
        thr[name] = min(h) if h else None
        thr_js[name] = min(hj) if hj else None
    out["detection_threshold_kl"] = thr
    out["detection_threshold_js"] = thr_js
    print(f"\n    d* (KL): {thr}")
    print(f"    d* (JS): {thr_js}")

    # --------------------------------------------------------------- C4b
    print("\nC4b over-dispersion probe -- forward KL can go NEGATIVE, JS cannot")
    for d0 in (0.05, 0.20):
        row = {}
        for name in FIELDS:
            cT, K = CODE[name]
            r = estimate(cT, widen_codes(cT, K, np.full(R, d0), cr), K,
                         groups, nd, grid, w, R, draws)
            row[name] = r
        out.setdefault("C4b", {})[str(d0)] = row
        print(f"    d0={d0:<5} KL level=" +
              " ".join(f"{n[:5]}:{row[n]['level']:+.4f}" for n in FIELDS))
        print(f"{'':11s} JS level=" +
              " ".join(f"{n[:5]}:{row[n]['level_js']:+.5f}" for n in FIELDS))

    # ---------------------------------------------------------------- C3
    print("\nC3  positional shuffle of the TRUE stream -- slope must fall to floor")
    row = {}
    prng = np.random.default_rng(a.seed + 5)
    for name in FIELDS:
        cT, K = CODE[name]
        cS = np.array(cT)
        for i in range(N):
            cS[i] = cS[i][prng.permutation(R)]
        r = estimate(cT, cS, K, groups, nd, grid, w, R, draws)
        r["slope_excludes_0"] = bool(r["ci"][0] > 0 or r["ci"][1] < 0)
        row[name] = r
        print(f"    {name:12s} slope={r['slope']:+.4f} "
              f"[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]  level={r['level']:+.4f}"
              f"   JS level={r['level_js']:+.5f}")
    out["C3_shuffle"] = row

    json.dump({"npz": a.npz, "n_seq": int(N), "n_gen": int(R), "n_days": int(nd),
               "grid": grid.tolist(), "window": w, "bins": a.bins,
               "n_boot": a.n_boot, "seed": a.seed, "fail": fail, "res": out},
              open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
