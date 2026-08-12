#!/usr/bin/env python3
"""A01 -- is there compound error in the generated ORDER FLOW at all?

Self-contained by design: it imports nothing from the earlier eval tree. That
tree has no floor arm, no power calibration, and its position grid includes
points whose window is truncated at the right edge; reusing it would inherit
all three, and a later edit there would silently change this result.

Estimator (see PLAN.md section 3). Sequences are split into two disjoint
halves A, B, stratified by trading day. At each position m,

  D_m^arm   = 1/2 [ KL(true_A || arm_B) + KL(true_B || arm_A) ] / H_m
  D_m^floor = 1/2 [ KL(true_A || true_B) + KL(true_B || true_A) ] / H_m
  excess_m  = D_m^arm - D_m^floor

Both sides compare n/2 against n/2, so the finite-sample KL bias is identical
by construction and cancels exactly in the difference -- not asymptotically,
identically. The model side never shares a sequence with the true side it is
scored against, which otherwise lets the two empirical histograms move
together and deflates the model's inflation while leaving the floor alone.

Bins are quantiles of the TRUE distribution plus two out-of-support bins and
one ILLEGAL bin. Illegal messages are charged, not dropped: dropping them
erases "the model emitted something invalid" from the metric entirely.

The day-block bootstrap works on binned counts, which are an exact sufficient
statistic, so a resample is a weighted sum over the day axis. All arms share
one draw matrix, so excess is a paired difference per replicate.
"""
import argparse
import json
import os

import numpy as np

FIELDS = ["event_type", "direction", "price_rel", "size", "log10_dt"]
KIND = {"event_type": "cat", "direction": "cat",
        "price_rel": "ord", "size": "ord", "log10_dt": "ord"}
CATS = {"event_type": [1, 2, 3, 4], "direction": [-1, 1]}


# ---------------------------------------------------------------- extraction
def action_fields(z, tick):
    """(n_seq, n_gen) arrays per field, per arm, plus a legality mask."""
    cols = [str(v) for v in z["msg_cols"]]
    out = {}
    for arm, key in (("true", "real_msgs"), ("draft", "draft_msgs"),
                     ("corr", "corr_msgs")):
        m = z[key]
        ev = m[:, :, cols.index("event_type")]
        dr = m[:, :, cols.index("direction")]
        sz = m[:, :, cols.index("size")]
        px = m[:, :, cols.index("price")]
        t = m[:, :, cols.index("time")]
        dt = np.full_like(t, np.nan)
        dt[:, 1:] = np.diff(t, axis=1)
        ok = (np.isin(ev, [1, 2, 3, 4]) & np.isin(dr, [-1, 1]) & (sz >= 1)
              & (px > 0) & np.isfinite(ev) & np.isfinite(sz) & np.isfinite(px))
        out[arm] = {"event_type": ev, "direction": dr, "size": sz,
                    "price_rel": (px - z["ref_mid"][:, None]) / float(tick),
                    "log10_dt": np.log10(np.where(dt > 0, dt, np.nan)),
                    "_ok": ok}
    return out


def make_edges(v_true, ok, name, n_bins):
    if KIND[name] == "cat":
        return np.asarray(CATS[name], float)
    v = np.asarray(v_true)[np.asarray(ok, bool)]
    v = v[np.isfinite(v)]
    return np.unique(np.quantile(v, np.linspace(0.0, 1.0, n_bins + 1)))


def codes(v, ok, name, edges):
    """Integer code per element. Last two codes: OUT-OF-SUPPORT, ILLEGAL."""
    v = np.asarray(v, float)
    msk = np.asarray(ok, bool)
    if KIND[name] == "cat":
        K = len(edges)
        c = np.full(v.shape, K, np.int64)
        for i, g in enumerate(edges):
            c[v == g] = i
    else:
        K = len(edges) - 1
        c = np.searchsorted(edges, v, side="right") - 1
        c = np.where(np.isfinite(v), c, K)
        c = np.where((c < 0) | (c >= K), K, c).astype(np.int64)
    c = c.copy()
    c[~msk] = K + 1
    return c, K + 2


# ------------------------------------------------------------------- counts
def build_counts(c, groups, grid, w, R, K):
    """counts[g, gi, bin] for each row-group g (here: day x half)."""
    out = np.zeros((len(groups), len(grid), K), np.int64)
    for gi, m in enumerate(grid):
        sl = slice(m - w, m + w + 1)          # always full width by construction
        blk = c[:, sl]
        for g, rows in enumerate(groups):
            if rows.size:
                out[g, gi] = np.bincount(blk[rows].ravel(), minlength=K)
    return out


def _probs(a, K):
    return (a + 1.0 / K) / (a.sum(-1, keepdims=True) + 1.0)


def kl_from_counts(a, b, K):
    """KL(p||q) with add-1/K smoothing, on the last axis."""
    p, q = _probs(a, K), _probs(b, K)
    return np.sum(p * (np.log(p) - np.log(q)), axis=-1)


def js_from_counts(a, b, K):
    """Jensen-Shannon divergence, same smoothing.

    Reported beside the KL because forward KL REWARDS over-dispersion: it
    penalises q being small where p is large, so smearing q's mass over a
    wider support lowers it. A model that is merely over-dispersed can
    therefore score a low D_m and look free of compound error. JS is
    symmetric and bounded, so it cannot be bought that way.
    """
    p, q = _probs(a, K), _probs(b, K)
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * (np.log(p) - np.log(m)), -1)
                  + np.sum(q * (np.log(q) - np.log(m)), -1))


def entropy_from_counts(a):
    tot = np.maximum(a.sum(-1, keepdims=True), 1)
    p = a / tot
    return -np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-300)), 0.0), -1)


def D_curve(tA, tB, gA, gB, K, h_eps=0.05, div="kl"):
    """Symmetric cross-half D_m, normalised by the pooled true entropy."""
    H = entropy_from_counts(tA + tB)
    f = kl_from_counts if div == "kl" else js_from_counts
    d = 0.5 * (f(tA, gB, K) + f(tB, gA, K))
    return np.where(H > h_eps, d / np.maximum(H, 1e-12), np.nan)


def ols_slope(grid, y, per=100.0):
    ok = np.isfinite(y)
    if ok.sum() < 4:
        return np.nan
    return float(np.polyfit(grid[ok].astype(float), y[ok], 1)[0] * per)


# --------------------------------------------------------------- mean drift
def drift_moments(v, ok, day_rows, grid, w):
    """Per-day S1/S2/N of the legal, finite values inside each window."""
    nd, ng = len(day_rows), len(grid)
    S1 = np.zeros((nd, ng)); S2 = np.zeros((nd, ng)); N = np.zeros((nd, ng))
    for gi, m in enumerate(grid):
        sl = slice(m - w, m + w + 1)
        vv, oo = v[:, sl], ok[:, sl]
        for d, rows in enumerate(day_rows):
            if not rows.size:
                continue
            x = vv[rows][oo[rows]]
            x = x[np.isfinite(x)]
            if x.size:
                S1[d, gi] = x.sum(); S2[d, gi] = (x * x).sum(); N[d, gi] = x.size
    return S1, S2, N


def z_curve(sT, qT, nT, sG, nG, min_n=40):
    """z(m) = (mean_gen - mean_true) / std_true, from resampled moments."""
    with np.errstate(invalid="ignore", divide="ignore"):
        mt = sT / np.maximum(nT, 1)
        var = qT / np.maximum(nT, 1) - mt ** 2
        sd = np.sqrt(np.maximum(var, 0.0))
        mg = sG / np.maximum(nG, 1)
        z = (mg - mt) / np.where(sd > 1e-9, sd, np.nan)
    return np.where((nT >= min_n) & (nG >= min_n), z, np.nan)


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--control", default=None, help="random-P arm (optional)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tick", type=int, default=100)
    ap.add_argument("--bins", type=int, default=40)
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--truncated-grid", action="store_true",
                    help="C2: include right-edge points whose window is cut")
    a = ap.parse_args()

    z = np.load(a.npz, allow_pickle=True)
    F = action_fields(z, a.tick)
    N, R = z["real_msgs"].shape[0], z["real_msgs"].shape[1]
    days = np.asarray(z["dates"])
    uday = np.unique(days)
    w, step = a.window, a.step

    # G5: only full-width windows. A window cut at the right edge has fewer
    # samples, KL bias ~ K/(2n) rises, and the curve tilts up for free.
    hi = R if a.truncated_grid else R - w
    grid = np.arange(w, hi, step)
    if grid.size < 4:
        raise SystemExit(f"grid too short: R={R} w={w}")

    # day-stratified disjoint halves, fixed seed
    # Day-stratified disjoint halves. The odd sequence of each day goes to
    # whichever half is currently smaller, otherwise every odd day sends its
    # remainder the same way and the global split drifts (64 seqs over 20 days
    # gave 40/24 before this). The imbalance does not bias `excess` -- model
    # and floor use the same A/B on the same sides, so the bias is identical
    # either way -- but it wastes power.
    rng = np.random.default_rng(a.seed)
    half = np.zeros(N, np.int64)
    n0 = n1 = 0
    for d in uday:
        r = np.flatnonzero(days == d)
        r = r[rng.permutation(r.size)]
        k = r.size // 2
        if r.size % 2 and n1 < n0:
            k += 1
        half[r[:k]] = 1
        n1 += k
        n0 += r.size - k
    day_rows = [np.flatnonzero(days == d) for d in uday]
    groups = [np.flatnonzero((days == d) & (half == h))
              for d in uday for h in (0, 1)]          # 2 groups per day
    nd = len(uday)

    arms = [("draft", F["draft"]), ("corr", F["corr"])]
    if a.control:
        arms.append(("rand", action_fields(np.load(a.control, allow_pickle=True),
                                           a.tick)["corr"]))

    print(f"[A01] {os.path.basename(a.npz)}", flush=True)
    print(f"      n_seq={N} n_gen={R} days={nd} grid={grid.size} pts "
          f"[{grid[0]}..{grid[-1]}] w={w} "
          f"({(500+R)/500:.1f}x the 500-message training window)", flush=True)
    print(f"      halves: A={int((half==0).sum())} B={int((half==1).sum())} "
          f"(day-stratified, seed={a.seed})", flush=True)
    for k, Fa in arms:
        print(f"      illegal {k:5s} {100*(~Fa['_ok']).mean():6.3f}%", flush=True)

    draws = np.random.default_rng(a.seed + 1).integers(0, nd, size=(a.n_boot, nd))

    res = {}
    hdr = (f"\n{'field':11s}{'arm':6s}{'KLslope':>10s}{'95% CI':>19s}"
           f"{'CE':>4s}{'JSslope':>10s}{'JS 95% CI':>19s}{'CE':>4s}"
           f"{'level':>9s}{'floor':>8s}{'z0':>8s}{'zEnd':>8s}{'zSlope':>9s}")
    print(hdr, flush=True); print("-" * len(hdr.strip()), flush=True)

    for name in FIELDS:
        edges = make_edges(F["true"][name], F["true"]["_ok"], name, a.bins)
        cT, K = codes(F["true"][name], F["true"]["_ok"], name, edges)
        CT = build_counts(cT, groups, grid, w, R, K)      # (2*nd, ng, K)
        CT = CT.reshape(nd, 2, len(grid), K)
        # floor uses the same construction as the model arms
        tA_all, tB_all = CT[:, 0], CT[:, 1]

        S1T, S2T, NT = drift_moments(F["true"][name], F["true"]["_ok"],
                                     day_rows, grid, w)

        res[name] = {"K": int(K), "n_bins_used": int(len(edges)), "arms": {}}
        boot_exc, point_exc = {}, {}

        # ---- floor (shared by every arm) -------------------------------
        def stack(sel):
            return sel.sum(0)

        # floor: gen side IS the true side, so D_curve(tA,tB,gA=tA,gB=tB)
        # expands to 1/2[KL(tA||tB) + KL(tB||tA)].  Passing (tB,tA) instead
        # would give 1/2[KL(tA||tA)+KL(tB||tB)] == 0, i.e. no floor at all.
        Dfl_pt = D_curve(stack(tA_all), stack(tB_all),
                         stack(tA_all), stack(tB_all), K)
        Jfl_pt = D_curve(stack(tA_all), stack(tB_all),
                         stack(tA_all), stack(tB_all), K, div="js")
        fl_boot = np.empty(a.n_boot)
        for b in range(a.n_boot):
            d = draws[b]
            tA, tB = tA_all[d].sum(0), tB_all[d].sum(0)
            fl_boot[b] = ols_slope(grid, D_curve(tA, tB, tA, tB, K))
        res[name]["floor"] = {
            "slope": ols_slope(grid, Dfl_pt),
            "slope_ci": [float(np.nanpercentile(fl_boot, 2.5)),
                         float(np.nanpercentile(fl_boot, 97.5))],
            "level": float(np.nanmean(Dfl_pt)), "curve": Dfl_pt.tolist(),
            "level_js": float(np.nanmean(Jfl_pt)), "curve_js": Jfl_pt.tolist()}

        for k, Fa in arms:
            cG, _ = codes(Fa[name], Fa["_ok"], name, edges)
            CG = build_counts(cG, groups, grid, w, R, K).reshape(nd, 2, len(grid), K)
            gA_all, gB_all = CG[:, 0], CG[:, 1]

            Darm = D_curve(stack(tA_all), stack(tB_all),
                           stack(gA_all), stack(gB_all), K)
            Jarm = D_curve(stack(tA_all), stack(tB_all),
                           stack(gA_all), stack(gB_all), K, div="js")
            exc = Darm - Dfl_pt
            exc_js = Jarm - Jfl_pt
            s_pt = ols_slope(grid, exc)

            acc = np.empty(a.n_boot)
            acc_js = np.empty(a.n_boot)
            for b in range(a.n_boot):
                d = draws[b]
                tA, tB = tA_all[d].sum(0), tB_all[d].sum(0)
                gA, gB = gA_all[d].sum(0), gB_all[d].sum(0)
                acc[b] = ols_slope(grid, D_curve(tA, tB, gA, gB, K)
                                   - D_curve(tA, tB, tA, tB, K))
                acc_js[b] = ols_slope(grid, D_curve(tA, tB, gA, gB, K, div="js")
                                      - D_curve(tA, tB, tA, tB, K, div="js"))
            lo, hi_ = np.nanpercentile(acc, 2.5), np.nanpercentile(acc, 97.5)
            boot_exc[k], point_exc[k] = acc, s_pt

            S1G, S2G, NG = drift_moments(Fa[name], Fa["_ok"], day_rows, grid, w)
            Z = z_curve(S1T.sum(0), S2T.sum(0), NT.sum(0), S1G.sum(0), NG.sum(0))
            zb = np.empty(a.n_boot)
            for b in range(a.n_boot):
                d = draws[b]
                zb[b] = ols_slope(grid, z_curve(S1T[d].sum(0), S2T[d].sum(0),
                                                NT[d].sum(0), S1G[d].sum(0),
                                                NG[d].sum(0)))
            zf = Z[np.isfinite(Z)]
            ce = bool(lo > 0)
            res[name]["arms"][k] = {
                "slope_excess": s_pt, "slope_excess_ci": [float(lo), float(hi_)],
                "has_CE": ce, "mc_sd": float(np.nanstd(acc)),
                # G4: what this setup could have detected, stated whether or
                # not anything was. "CI covers zero" alone cannot distinguish
                # no effect from no power.
                "min_detectable_slope": float(1.96 * np.nanstd(acc)),
                "resolution_sigma": (float(abs(s_pt) / np.nanstd(acc))
                                     if np.nanstd(acc) > 0 else None),
                "level_excess": float(np.nanmean(exc)),
                "level_D": float(np.nanmean(Darm)),
                "curve_D": Darm.tolist(), "curve_excess": exc.tolist(),
                "slope_excess_js": ols_slope(grid, exc_js),
                "slope_excess_js_ci": [float(np.nanpercentile(acc_js, 2.5)),
                                       float(np.nanpercentile(acc_js, 97.5))],
                "has_CE_js": bool(np.nanpercentile(acc_js, 2.5) > 0),
                "level_excess_js": float(np.nanmean(exc_js)),
                "curve_excess_js": exc_js.tolist(),
                "z": Z.tolist(), "z_first": float(zf[0]) if zf.size else None,
                "z_last": float(zf[-1]) if zf.size else None,
                "z_slope": ols_slope(grid, Z),
                "z_slope_ci": [float(np.nanpercentile(zb, 2.5)),
                               float(np.nanpercentile(zb, 97.5))],
                "illegal_pct": float(100 * (~Fa["_ok"]).mean())}
            r = res[name]["arms"][k]
            jlo, jhi = r["slope_excess_js_ci"]
            print(f"{name if k=='draft' else '':11s}{k:6s}{s_pt:+10.4f}"
                  f"{f'[{lo:+.4f},{hi_:+.4f}]':>19s}{'YES' if ce else '-':>4s}"
                  f"{r['slope_excess_js']:+10.5f}"
                  f"{f'[{jlo:+.5f},{jhi:+.5f}]':>19s}"
                  f"{'YES' if r['has_CE_js'] else '-':>4s}"
                  f"{r['level_excess']:9.4f}{res[name]['floor']['level']:8.4f}"
                  f"{(r['z_first'] if r['z_first'] is not None else np.nan):+8.3f}"
                  f"{(r['z_last'] if r['z_last'] is not None else np.nan):+8.3f}"
                  f"{r['z_slope']:+9.4f}", flush=True)

        fs = res[name]["floor"]
        flo, fhi = fs["slope_ci"]
        print(f"{'':11s}{'FLOOR':6s}{fs['slope']:+10.4f}"
              f"{f'[{flo:+.4f},{fhi:+.4f}]':>19s}{'--':>4s}"
              f"{'':>10s}{'':>19s}{'':>4s}{0.0:9.4f}{fs['level']:8.4f}", flush=True)

        for k in [x[0] for x in arms if x[0] != "draft"]:
            d = boot_exc[k] - boot_exc["draft"]
            lo, hi_ = np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5)
            res[name][f"contrast_{k}_vs_draft"] = {
                "point": point_exc[k] - point_exc["draft"],
                "ci": [float(lo), float(hi_)],
                "verdict": "REDUCED" if hi_ < 0 else ("WORSE" if lo > 0 else "n.s.")}

    ce_fields = [n for n in FIELDS if res[n]["arms"]["draft"]["has_CE"]]
    print(f"\nSTEP 1  fields with compound error in the pretrained draft: "
          f"{ce_fields if ce_fields else 'NONE'}", flush=True)
    bad_floor = [n for n in FIELDS
                 if not (res[n]['floor']['slope_ci'][0] <= 0 <= res[n]['floor']['slope_ci'][1])]
    print(f"        floor arms whose own slope EXCLUDES zero (pipeline artefact!): "
          f"{bad_floor if bad_floor else 'NONE'}", flush=True)

    json.dump({"npz": a.npz, "control": a.control, "n_seq": int(N),
               "n_gen": int(R), "n_days": int(nd), "grid": grid.tolist(),
               "window": w, "step": step, "bins": a.bins, "seed": a.seed,
               "n_boot": a.n_boot, "truncated_grid": bool(a.truncated_grid),
               "res": res}, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
