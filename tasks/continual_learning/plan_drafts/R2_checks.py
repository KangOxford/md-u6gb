"""Reproduces every measured number in `R2_review_statistics.md`. CPU only, no GPU, no sbatch.

    cd /lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning
    python3 plan_drafts/R2_checks.py            # all checks
    python3 plan_drafts/R2_checks.py f1 f3      # only these

Every check prints the published figure beside the measured one, so a disagreement is visible
without cross-referencing the review. Runtime is about eight minutes for all of them; the
individual checks are 20 s to 3 min.
"""
from __future__ import annotations

import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np

CODE = Path("/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/code")
sys.path.insert(0, str(CODE))
import failure_pool_reliability as F  # noqa: E402

ROOT = F.DEFAULT_ROOT
TK = ["AMD", "AMZN", "GOOG", "INTC", "JPM", "META", "MSFT", "NFLX"]
SEEDS = list(range(97701, 97711))
H = 2                       # horizon index 2 = 50 messages, the convention throughout
HN = ["H10", "H25", "H50", "H100", "H150", "H200", "H250"]


# ---------------------------------------------------------------------------- helpers

def trim(v: np.ndarray, p: float = 0.05) -> float:
    v = np.sort(v)
    n = v.size
    c = int(np.floor(p * n))
    return float(v[c:n - c].mean())


def relgap(a: float, b: float) -> float:
    return (a - b) / ((a + b) / 2)


def spear(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = F._rank(np.asarray(a, float)), F._rank(np.asarray(b, float))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def num_errors(ticker: str, seed: int) -> np.ndarray:
    """The 528 logged slots for one member, in log order."""
    p = ROOT / f"hp_v5me3_{ticker}_s{seed}" / "member_0" / "inference.log"
    return np.array([int(x) for m in re.finditer(r"num_errors \[([^\]]*)\]", p.read_text())
                     for x in m.group(1).split()])


def slot_order() -> np.ndarray:
    """Slot index for each context, in `rank_indices` order.

    Addendum 5 §A: the final partial batch is padded by wrapping to the start, so contexts
    0..27 are generated twice and only the second pass survives on disk. The surviving
    generation of context j (j < 28) is slot 500+j; contexts 28..499 are slot i.
    """
    return np.concatenate([np.arange(500, 528), np.arange(28, 500)])


def npz_rows(ticker: str) -> np.ndarray:
    """Row in the (string-sorted) `load_arm` id order, for each context in rank_indices order.

    `.returns_multih_*.npz` stores ids as strings and `load_arm` sorts them lexicographically,
    so row position is NOT monotone in dataset index. Anything index-based must map explicitly.
    """
    ri = np.array(json.load(
        (ROOT / f"hp_v5me3_{ticker}_s{SEEDS[0]}" / "member_0" / "sample_indices_rank0.json")
        .open())["rank_indices"])
    pos = {int(s): i for i, s in enumerate(sorted({str(i) for i in ri}))}
    return np.array([pos[int(v)] for v in ri])


def nulls_with(real, gen, k, h, rng, strat_fn, n_draws):
    """`pairing_nulls_repeated` with the stratifier as an argument, which production lacks."""
    keys = ("true", "shared", "independent", "cross")
    acc = {q: [] for q in keys}
    S, N = gen.shape[0], real.shape[0]
    for _ in range(n_draws):
        p = rng.permutation(S)
        A, B = p[:k], p[k:2 * k]
        p1, p2 = rng.permutation(N), rng.permutation(N)
        y = real[:, h]

        def sc(g, idx):
            return strat_fn(F.scores(real, g[idx])["total"][:, h], y)

        acc["true"].append(spear(sc(gen, A), sc(gen, B)))
        acc["shared"].append(spear(sc(gen[:, p1], A), sc(gen[:, p1], B)))
        acc["independent"].append(spear(sc(gen[:, p1], A), sc(gen[:, p2], B)))
        acc["cross"].append(spear(sc(gen, A), sc(gen[:, p1], B)))
    return {q: float(np.mean(acc[q])) for q in keys}


# ---------------------------------------------------------------------------- checks

def f1():
    """F1 (BLOCKING): the `num_errors` refutation compares k=1 against k=5."""
    print("F1  published: num_errors 0.229-0.419 'below the score's 0.330-0.545 at k=5'")
    order = slot_order()
    ne1, ne5, s1, s5, raw1, raw5 = [], [], [], [], [], []
    print(f"  {'tk':5} | {'ne k=1':>7} {'ne k=5':>7} | {'strat k=1':>9} {'strat k=5':>9} |"
          f" {'raw k=1':>7} {'raw k=5':>7}")
    for tk in TK:
        real, gen, seeds = F.load_arm(ROOT, "v5me3", tk)
        A = np.array([num_errors(tk, s)[order] for s in seeds], dtype=float)
        rng = np.random.default_rng(19)
        r1 = [spear(A[p[0]], A[p[1]]) for p in (rng.permutation(len(seeds)) for _ in range(60))]
        rng = np.random.default_rng(19)
        r5 = [spear(A[p[:5]].mean(0), A[p[5:10]].mean(0))
              for p in (rng.permutation(len(seeds)) for _ in range(60))]
        sh = lambda k, st: F.split_half(real, gen, k, 60, np.random.default_rng(19), "total",
                                        horizon_idx=H, stratified=st)["rho_mean"]
        ne1.append(np.mean(r1)); ne5.append(np.mean(r5))
        s1.append(sh(1, True));  s5.append(sh(5, True))
        raw1.append(sh(1, False)); raw5.append(sh(5, False))
        print(f"  {tk:5} | {ne1[-1]:7.3f} {ne5[-1]:7.3f} | {s1[-1]:9.3f} {s5[-1]:9.3f} |"
              f" {raw1[-1]:7.3f} {raw5[-1]:7.3f}")
    print(f"  ranges: num_errors k=1 {min(ne1):.3f}-{max(ne1):.3f}   k=5 {min(ne5):.3f}-{max(ne5):.3f}")
    print(f"          stratified k=1 {min(s1):.3f}-{max(s1):.3f}   k=5 {min(s5):.3f}-{max(s5):.3f}")
    print(f"          raw        k=1 {min(raw1):.3f}-{max(raw1):.3f}   k=5 {min(raw5):.3f}-{max(raw5):.3f}")
    print(f"  num_errors beats the STRATIFIED score at matched k=1 in "
          f"{sum(a > b for a, b in zip(ne1, s1))}/8, at matched k=5 in "
          f"{sum(a > b for a, b in zip(ne5, s5))}/8")


def f2():
    """F2 (BLOCKING): E1a's target Drho >= 0.70 is unreachable at any k."""
    print("F2  02 §1.1 E1a target: rho_true - rho_indep >= 0.70")
    for k in (3, 5):
        d = []
        for tk in TK:
            real, gen, _ = F.load_arm(ROOT, "v5me3", tk)
            r = F.pairing_nulls_repeated(real, gen, k, H, np.random.default_rng(0), True, 60)
            d.append(r["true_mean"] - r["independent_mean"])
        d = np.array(d)
        print(f"  k={k}: " + " ".join(f"{x:.3f}" for x in d) +
              f"   max {d.max():.3f}   pass {(d >= 0.70).sum()}/8")
    inf = []
    for tk in TK:
        real, gen, _ = F.load_arm(ROOT, "v5me3", tk)
        ks = np.array([1., 2., 3., 5.])
        rh = np.array([F.split_half(real, gen, int(k), 60, np.random.default_rng(17), "total",
                                    horizon_idx=H, stratified=True)["rho_mean"] for k in ks])
        c, *_ = np.linalg.lstsq(np.vstack([np.ones_like(ks), 1 / ks]).T, 1 / rh, rcond=None)
        inf.append(1 / c[0])
    inf = np.array(inf)
    print(f"  two-parameter rho_inf: " + " ".join(f"{x:.3f}" for x in inf))
    print(f"  below 0.80 in {(inf < 0.80).sum()}/8 (02 §3.1 published 7/8); "
          f"Drho>=0.70 needs rho_true>=0.78, reachable for {(inf >= 0.78).sum()}/8")


def f3():
    """F3 (BLOCKING): the do-nothing floor is seed-matched, the arm contrast cannot be."""
    print("F3  phi = fraction of (context, member) forward returns bitwise identical between")
    print("    the two regenerations.  Published: 0.976 0.873 0.661 0.393 0.281 0.225 0.191")
    phis = []
    for tk in TK:
        rb, gb, sb = F.load_arm(ROOT, "v5me3repro", tk)
        rc, gc, sc = F.load_arm(ROOT, "v5me3repB", tk)
        cm = sorted(set(sb) & set(sc))
        B, C = gb[[sb.index(s) for s in cm]], gc[[sc.index(s) for s in cm]]
        phis.append([float((B[:, :, h] == C[:, :, h]).mean()) for h in range(7)])
    print("    measured : " + " ".join(f"{x:.3f}" for x in np.mean(phis, axis=0)))
    print()
    print("    matched  = repro[{a,b}] vs repB[{a,b}]  (same seed labels, shares phi bitwise)")
    print("    crossed  = repro[{a,b}] vs repB[{c,d}]  (different labels, shares nothing)")
    print("    statistic: 5%-trimmed-mean relative gap, RMS over all seed pairs then over tickers")
    print(f"    {'h':>5} {'phi':>6} {'matched%':>9} {'crossed%':>9} {'ratio':>7} {'1/sqrt(1-phi)':>14}")
    for h in (0, 2, 6):
        M, X, P = [], [], []
        for tk in TK:
            rb, gb, sb = F.load_arm(ROOT, "v5me3repro", tk)
            rc, gc, sc = F.load_arm(ROOT, "v5me3repB", tk)
            cm = sorted(set(sb) & set(sc))
            B, C = gb[[sb.index(s) for s in cm]], gc[[sc.index(s) for s in cm]]
            P.append(float((B[:, :, h] == C[:, :, h]).mean()))
            mt, xt = [], []
            for pair in itertools.combinations(range(len(cm)), 2):
                A_ = list(pair)
                rest = [i for i in range(len(cm)) if i not in pair]
                mB = trim(F.scores(rb, B[A_])["total"][:, h])
                mt.append(relgap(mB, trim(F.scores(rb, C[A_])["total"][:, h])))
                if len(rest) >= 2:
                    xt.append(relgap(mB, trim(F.scores(rb, C[rest[:2]])["total"][:, h])))
            M.append(np.sqrt(np.mean(np.square(mt))))
            X.append(np.sqrt(np.mean(np.square(xt))))
        M, X, P = np.array(M), np.array(X), np.array(P)
        print(f"    {HN[h]:>5} {P.mean():6.3f} {100*M.mean():9.2f} {100*X.mean():9.2f}"
              f" {X.mean()/M.mean():7.2f} {1/np.sqrt(1-P.mean()):14.2f}")


def f4():
    """F4 (MAJOR): 1/n_bins is neither an upper nor a lower bound on a stratified null."""
    print("F4  synthetic: score = |y| + noise on both halves, 10 bins, 40 draws")
    print(f"    {'noise':>7} {'leak=corr(strat,|y|)':>22} {'independent null':>18}")
    rng = np.random.default_rng(7)
    y = rng.normal(size=500)
    ay = np.abs(y)
    for noise in (0.0, 0.5, 1.0, 4.0):
        lk, nl = [], []
        for _ in range(40):
            a = ay + noise * rng.normal(size=500)
            b = ay + noise * rng.normal(size=500)
            sa, sb = F.stratify(a, y), F.stratify(b, y)
            lk.append(abs(spear(sa, ay)))
            nl.append(spear(sa, sb))
        print(f"    {noise:7.1f} {np.mean(lk):22.3f} {np.mean(nl):18.3f}")
    print("    -> leak 0.100 with a null of 1.000 (not an upper bound); leak 0.014 with a")
    print("       null of 0.041 (not a lower bound).")
    ind, lk = [], []
    for tk in TK:
        real, gen, _ = F.load_arm(ROOT, "v5me3", tk)
        r = F.pairing_nulls_repeated(real, gen, 5, H, np.random.default_rng(0), True, 60)
        ind.append(r["independent_mean"])
        a = np.abs(real[:, H])
        lk.append(abs(spear(F.stratify(a, real[:, H]), a)))
    ind, lk = np.array(ind), np.array(lk)
    se = ind.std(ddof=1) / np.sqrt(8)
    print(f"    real data k=5: independent mean {ind.mean():.4f} (se {se:.4f}); "
          f"{(ind < 0.10).sum()}/8 strictly below the claimed 0.10 floor")
    print(f"      t vs 0.100 = {(ind.mean()-0.10)/se:+.2f};  measured leak {lk.min():.3f}-{lk.max():.3f} "
          f"(mean {lk.mean():.3f}), t vs it = {(ind.mean()-lk.mean())/se:+.2f}")
    rng = np.random.default_rng(21)
    real, gen = np.zeros((300, 1)), rng.normal(size=(9, 300, 1))
    out = F.pairing_nulls_repeated(real, gen, 3, 0, rng, True, 60)
    print(f"    the leak-floor TEST's own fixture: leak_floor {out['leak_floor']}, "
          f"independent {out['independent_mean']:+.4f}  -> the docstring's bound is violated "
          f"and the test still passes")


def f5():
    """F5 (MAJOR): 'the margin roughly doubles' is a denominator swap."""
    print("F5  published (RESULTS addendum 5 §F): 0.460/0.13 = 3.5x  ->  0.438/0.059 = 7.4x")
    a = []
    for tk in TK:
        real, gen, _ = F.load_arm(ROOT, "v5me3", tk)
        ay = np.abs(real[:, H])
        n1 = nulls_with(real, gen, 5, H, np.random.default_rng(11), F.stratify, 40)
        n2 = nulls_with(real, gen, 5, H, np.random.default_rng(11), F.stratify_v2, 40)
        a.append([n1["true"], n1["independent"], abs(spear(F.stratify(ay, real[:, H]), ay)),
                  n2["true"], n2["independent"], abs(spear(F.stratify_v2(ay, real[:, H]), ay))])
    a = np.array(a)
    m = a.mean(0)
    print(f"    measured: rel_v1 {m[0]:.3f} ind_v1 {m[1]:.3f} leak_v1 {m[2]:.3f} | "
          f"rel_v2 {m[3]:.3f} ind_v2 {m[4]:.3f} leak_v2 {m[5]:.3f}")
    print(f"    ratio over the leak            : v1 {m[0]/m[2]:.2f}x  v2 {m[3]/m[5]:.2f}x")
    print(f"    ratio over the independent null: v1 {m[0]/m[1]:.2f}x  v2 {m[3]/m[4]:.2f}x")
    d1, d2 = a[:, 0] - a[:, 1], a[:, 3] - a[:, 4]
    dd = d2 - d1
    print(f"    EXCESS true-independent (the E1a quantity): v1 {d1.mean():.3f}  v2 {d2.mean():.3f}")
    print(f"      paired over 8 tickers: {dd.mean():+.4f} sd {dd.std(ddof=1):.4f} "
          f"t {dd.mean()/(dd.std(ddof=1)/np.sqrt(8)):+.2f} on 7 df, positive in {(dd>0).sum()}/8")


def f6():
    """F6 (MAJOR): the dispersion-partial floor is degenerate and is a one-point calibration."""
    print("F6  dispersion_partial_floor sets real=0, so stratify() collapses:")
    z = np.zeros(500)
    edges = np.quantile(np.abs(z), np.linspace(0, 1, 11))
    edges[-1] += 1e-12
    b = np.clip(np.digitize(np.abs(z), edges[1:-1]), 0, 9)
    print(f"    non-empty strata in the floor fixture: {len(np.unique(b))} "
          f"(production has 6-9, RESULTS addendum 5 §F)")
    rng = np.random.default_rng(3)
    print(f"    {'planted signal sd':>18} {'kept (real=0)':>14} {'kept (real=y)':>14}")
    for sig in (0.0, 0.5, 1.0, 2.0, 4.0):
        w = np.exp(rng.normal(size=(1, 500, 1)))
        bias = sig * rng.normal(size=(1, 500, 1))
        g0 = bias + w * rng.normal(size=(10, 500, 1))
        k0 = F.dispersion_partialled_reliability(np.zeros((500, 1)), g0, 3, 0,
                                                 np.random.default_rng(4), 40)["fraction_kept"]
        y = rng.normal(size=(500, 1))
        g1 = y[None] + bias + w * rng.normal(size=(10, 500, 1))
        k1 = F.dispersion_partialled_reliability(y, g1, 3, 0,
                                                 np.random.default_rng(4), 40)["fraction_kept"]
        print(f"    {sig:18.2f} {k0:14.3f} {k1:14.3f}")
    print("    score-nuisance rank correlation on the REAL data (k=3, 60 draws):")
    print(f"    {'tk':5} {'rho':>6} {'kept':>6} {'a_spread':>9} {'a_move':>8}")
    for tk in TK:
        real, gen, _ = F.load_arm(ROOT, "v5me3", tk)
        rng2 = np.random.default_rng(0)
        keep, base, als, alm = [], [], [], []
        for _ in range(60):
            p = rng2.permutation(gen.shape[0])
            A, B, Hh = p[:3], p[3:6], p[6:]
            ns = F.scores(real, gen[Hh])["spread_pop"][:, H]
            nm = np.abs(gen[Hh].mean(axis=0)[:, H])
            sc = lambda idx: F.stratify(F.scores(real, gen[idx])["total"][:, H], real[:, H])
            sa, sb = sc(A), sc(B)
            base.append(spear(sa, sb))
            keep.append(spear(F.partial_out(F.partial_out(sa, ns), nm),
                              F.partial_out(F.partial_out(sb, ns), nm)))
            als.append(spear(sa, ns)); alm.append(spear(sa, nm))
        print(f"    {tk:5} {np.mean(base):6.3f} {np.mean(keep)/np.mean(base):6.3f} "
              f"{np.mean(als):9.3f} {np.mean(alm):8.3f}")
    print("    (the synthetic floor sits at a_spread 0.81 / a_move 0.64 -- a different regime)")
    print("    direct alternative -- split-half reliability of each term, k=5, stratified:")
    row = {q: [] for q in ("total", "spread_pop", "bias2_raw", "bias2")}
    for tk in TK:
        real, gen, _ = F.load_arm(ROOT, "v5me3", tk)
        for q in row:
            row[q].append(F.split_half(real, gen, 5, 60, np.random.default_rng(13), q,
                                       horizon_idx=H, stratified=True)["rho_mean"])
    print("      " + "  ".join(f"{q} {np.mean(v):.3f}" for q, v in row.items()))


def f7():
    """F7 (MAJOR): 'only R matters' holds on the null and fails under a heterogeneous effect."""
    print("F7  sd of the trimmed-mean relative gap, 60 draws, mean over 8 tickers, h=50")
    print(f"    {'design':>12} {'R':>5} " + " ".join(f"{'het=%.2f' % h:>9}"
                                                    for h in (0.0, 0.25, 0.5, 1.0)))
    for (N, k) in [(500, 1), (250, 2), (100, 5)]:
        out = []
        for het in (0.0, 0.25, 0.5, 1.0):
            s = []
            for tk in TK:
                real, gen, _ = F.load_arm(ROOT, "v5me3", tk)
                rng = np.random.default_rng(5)
                g = []
                for _ in range(60):
                    cs = rng.choice(real.shape[0], size=N, replace=False)
                    p = rng.permutation(gen.shape[0])
                    m1 = F.scores(real[cs], gen[p[:k]][:, cs])["total"][:, H]
                    m2 = F.scores(real[cs], gen[p[k:2 * k]][:, cs])["total"][:, H]
                    e = np.maximum(0.0, 1.0 + het * rng.normal(size=N))
                    g.append(relgap(trim(m1 * e), trim(m2)))
                s.append(np.std(g, ddof=1))
            out.append(100 * np.mean(s))
        print(f"    {'N=%d k=%d' % (N, k):>12} {N*k:5d} " + " ".join(f"{x:9.2f}" for x in out))


def f9():
    """F9 (MAJOR): the design-effect table is computed on the largest ticker only."""
    print("F9  dataset_length is per ticker; 02 §1.1 uses 226,002 (GOOG) for all eight")
    print(f"    {'tk':5} {'dataset_length':>15} {'m @ N=500':>10} {'DEFF (ICC .10)':>15} {'eff N':>7}")
    for tk in TK:
        d = json.load((ROOT / f"hp_v5me3_{tk}_s{SEEDS[0]}" / "member_0"
                       / "sample_indices_rank0.json").open())
        m = 500 * 500 / d["dataset_length"]
        deff = 1 + (m - 1) * 0.10
        print(f"    {tk:5} {d['dataset_length']:15d} {m:10.2f} {deff:15.2f} {500/deff:7.0f}")


def f10():
    """F10 (MAJOR): 02 §2.3's floor table does not reproduce; the headline moves most."""
    print("F10 published pool-restricted mean-of-M gaps: "
          "+19.91 -1.25 -7.91 +6.22 -7.64 +11.43 +27.82 +15.31 (mean +7.99, t=1.74)")
    gm = lambda v: float(np.exp(np.log(v[v > 0]).mean()))
    stats = {"mean": np.mean, "gmean": gm, "median": np.median, "trim05": trim}
    allr = {k: [] for k in stats}
    pool = {k: [] for k in stats}
    zeros = []
    for tk in TK:
        rb, gb, sb = F.load_arm(ROOT, "v5me3repro", tk)
        rc, gc, sc = F.load_arm(ROOT, "v5me3repB", tk)
        cm = sorted(set(sb) & set(sc))
        MB = F.scores(rb, gb[[sb.index(s) for s in cm]])["total"][:, H]
        MC = F.scores(rb, gc[[sc.index(s) for s in cm]])["total"][:, H]
        zeros.append(int((MB == 0).sum()))
        real, gen, seeds = F.load_arm(ROOT, "v5me3", tk)
        sel = [seeds.index(s) for s in seeds if s >= 97706]
        s = F.stratify(F.scores(real, gen[sel])["total"][:, H], real[:, H])
        top = np.argsort(-s)[:max(1, round(0.10 * MB.size))]
        for k, f in stats.items():
            allr[k].append(relgap(f(MB), f(MC)))
            pool[k].append(relgap(f(MB[top]), f(MC[top])))
    for k in stats:
        a, b = np.array(allr[k]), np.array(pool[k])
        print(f"    {k:7} all RMS {100*np.sqrt((a**2).mean()):6.2f}%  "
              f"pool RMS {100*np.sqrt((b**2).mean()):6.2f}%  pool max {100*np.abs(b).max():6.2f}%")
    b = np.array(pool["mean"])
    print("    pool mean-of-M per ticker: " + " ".join(f"{100*x:+.2f}" for x in b))
    print(f"      mean {100*b.mean():+.2f}%  sd {100*b.std(ddof=1):.2f}%  "
          f"t {b.mean()/(b.std(ddof=1)/np.sqrt(8)):.2f}   positive {(b>0).sum()}/8")
    print(f"    exact zeros in M, per 500 contexts: {zeros}  "
          f"-> the geometric mean depends entirely on an unstated zero-handling rule")


def f13():
    """F13 (MINOR): the ticker, not the draw, is the unit for the `shared > true` claim."""
    print("F13 published: `shared > true` in 8/8 tickers at k=5, mean gap +0.032, "
          "per-ticker +- 0.02-0.05 (that +- is a WITHIN-ticker draw sd)")
    for k in (3, 5):
        g = np.array([F.pairing_nulls_repeated(*F.load_arm(ROOT, "v5me3", tk)[:2], k, H,
                                               np.random.default_rng(0), True, 60)
                      ["shared_minus_true_mean"] for tk in TK])
        se = g.std(ddof=1) / np.sqrt(8)
        loo = [np.delete(g, i) for i in range(8)]
        ts = [x.mean() / (x.std(ddof=1) / np.sqrt(7)) for x in loo]
        print(f"    k={k}: " + " ".join(f"{x:+.3f}" for x in g) +
              f"  mean {g.mean():+.4f} se {se:.4f} t {g.mean()/se:+.2f} on 7 df, "
              f"positive {(g>0).sum()}/8, leave-one-out t {min(ts):+.2f}..{max(ts):+.2f}")


def f14():
    """F14 (MINOR) + the clean join check: the wrap mapping, in all eight tickers."""
    print("F14 the wrapped 28 are rank_indices[0..27] = the 28 LOWEST dataset indices")
    for tk in TK:
        d = json.load((ROOT / f"hp_v5me3_{tk}_s{SEEDS[0]}" / "member_0"
                       / "sample_indices_rank0.json").open())
        ri = np.array(d["rank_indices"])
        print(f"    {tk:5} span of the first 28 = {ri[27]-ri[0]:6d} indices = "
              f"{(ri[27]-ri[0])/d['dataset_length']:.2%} of the corpus")
    print()
    print("    join check: slot j and slot 500+j should be the same context (seed-mean over 10)")
    print(f"    {'tk':5} {'rho(first28,surplus28)':>23} {'same-slot 5v5 ceiling':>22} {'28 random slots':>17}")
    for tk in TK:
        A = np.array([num_errors(tk, s) for s in SEEDS], dtype=float)
        rng = np.random.default_rng(2)
        idx = rng.choice(np.arange(28, 500), 28, replace=False)
        print(f"    {tk:5} {spear(A[:, :28].mean(0), A[:, 500:528].mean(0)):23.3f} "
              f"{spear(A[:5, 500:528].mean(0), A[5:, 500:528].mean(0)):22.3f} "
              f"{spear(A[:, idx].mean(0), A[:, 500:528].mean(0)):17.3f}")
    print()
    print("    effect of dropping the 28 (k=5 nulls, k=3 partial, 60 draws):")
    print(f"    {'tk':5} {'true 500':>9} {'true 472':>9} {'shr-tru 500':>12} {'shr-tru 472':>12}"
          f" {'kept 500':>9} {'kept 472':>9}")
    acc = []
    for tk in TK:
        real, gen, _ = F.load_arm(ROOT, "v5me3", tk)
        rows = npz_rows(tk)[:28]
        keep = np.setdiff1d(np.arange(real.shape[0]), rows)
        out = []
        for R_, G_ in ((real, gen), (real[keep], gen[:, keep])):
            n = F.pairing_nulls_repeated(R_, G_, 5, H, np.random.default_rng(0), True, 60)
            dsp = F.dispersion_partialled_reliability(R_, G_, 3, H, np.random.default_rng(0), 60)
            out.append((n["true_mean"], n["shared_minus_true_mean"], dsp["fraction_kept"]))
        acc.append(out)
        print(f"    {tk:5} {out[0][0]:9.3f} {out[1][0]:9.3f} {out[0][1]:+12.4f} "
              f"{out[1][1]:+12.4f} {out[0][2]:9.3f} {out[1][2]:9.3f}")
    a = np.array([[x for x in o[0]] for o in acc])
    b = np.array([[x for x in o[1]] for o in acc])
    print(f"    means: true {a[:,0].mean():.4f} -> {b[:,0].mean():.4f} | "
          f"shared-true {a[:,1].mean():+.4f} -> {b[:,1].mean():+.4f} | "
          f"kept {a[:,2].mean():.4f} -> {b[:,2].mean():.4f}")
    print(f"    tickers with shared > true: {(a[:,1]>0).sum()}/8 -> {(b[:,1]>0).sum()}/8")


CHECKS = {"f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5, "f6": f6, "f7": f7,
          "f9": f9, "f10": f10, "f13": f13, "f14": f14}


def main(argv):
    want = [a.lower() for a in argv[1:]] or list(CHECKS)
    for name in want:
        if name not in CHECKS:
            print(f"unknown check {name!r}; available: {' '.join(CHECKS)}")
            return 2
    for name in want:
        print("=" * 78)
        CHECKS[name]()
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
