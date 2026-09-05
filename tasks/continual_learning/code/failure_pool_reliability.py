"""Is a "failure scenario" a property of the context, or a property of one draw?

Issue #73 proposes mining rollouts that "diverge hugely from the true data" into
a pool and continuing training on it. Two things have to hold before that pool
means anything, and both are measurable on rollouts that already exist:

  (R) Reliability. Score a context from one set of rollouts, score it again from
      a disjoint set. If the two rankings do not agree, "this context is a
      failure" is a statement about the draw, not about the context. Generation
      is not even seed-reproducible here (XLA autotuning; see
      feedback_a_seed_that_does_not_reproduce), so this ceiling is not free.

  (D) Decomposition. Total squared error against the single realised path splits
      exactly into a systematic part and a dispersion part

          mean_i (x_i - y)^2  =  (xbar - y)^2  +  var_i(x)

      The left side is what "diverges from the truth" selects on. Only the first
      term on the right is learnable: the second says the model spread its mass
      wide and the realised path fell off centre, which is a correct conditional
      forecast meeting one draw. Selecting on the left over-weights exactly the
      high-dispersion regimes.

      Both right-hand terms are biased at small k. With s^2 the ddof=1 sample
      variance over k members, E[(xbar-y)^2] = bias^2 + sigma^2/k, so the
      bias-corrected estimate is (xbar-y)^2 - s^2/k. That correction is the
      difference between "systematic error" and "systematic error plus a copy of
      the dispersion I was trying to remove", and at k=1 it is undefined --- one
      rollout per context cannot separate the two terms at all.

Data (already on disk, no GPU): frozen contexts, one realised future each, and
S independently seeded rollouts per context.

    <root>/hp_<config>_<TICKER>_s<SEED>/member_0/.returns_multih_{real,gen}.npz
    ids  (500,)    context ids, unsorted, differing order between arms
    vals (500, 7)  forward returns at HS = 10,25,50,100,150,200,250 messages

The real arm is written once, on the lowest seed of each config. Everything is
joined by id, never by row order.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HORIZONS = (10, 25, 50, 100, 150, 200, 250)
DEFAULT_ROOT = Path(
    "/lus/lfs1aip2/projects/public/u6gb/tasks/"
    "crps_return_alignment_20260808T025024Z/data"
)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def _npz(root: Path, config: str, ticker: str, seed: int, arm: str) -> Optional[Path]:
    p = root / f"hp_{config}_{ticker}_s{seed}" / "member_0" / f".returns_multih_{arm}.npz"
    return p if p.exists() else None


def seeds_for(root: Path, config: str, ticker: str) -> List[int]:
    """Seeds present for one (config, ticker), ascending.

    Directory names are matched exactly rather than globbed loosely: a bare
    ``hp_v5me3_*`` also catches ``hp_v5me3repro_*`` and the ``_deprecated_``
    renames that this repo uses in place of deletion.
    """
    out = []
    for d in root.glob(f"hp_{config}_{ticker}_s*"):
        tail = d.name[len(f"hp_{config}_{ticker}_s"):]
        if tail.isdigit():
            out.append(int(tail))
    return sorted(out)


def load_arm(root: Path, config: str, ticker: str) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Return (real, gen, seeds) with real (N,H) and gen (S,N,H), id-aligned.

    Contexts are the intersection of every seed's id set, sorted, so that a seed
    which scored a different subset cannot silently shift rows.
    """
    seeds = seeds_for(root, config, ticker)
    if not seeds:
        raise FileNotFoundError(f"no seeds for {config}/{ticker} under {root}")

    real_path = next((_npz(root, config, ticker, s, "real") for s in seeds
                      if _npz(root, config, ticker, s, "real")), None)
    if real_path is None:
        raise FileNotFoundError(f"no real arm for {config}/{ticker}")
    zr = np.load(real_path)

    gens = {}
    for s in seeds:
        p = _npz(root, config, ticker, s, "gen")
        if p is not None:
            gens[s] = np.load(p)
    seeds = sorted(gens)

    common = set(zr["ids"].tolist())
    for z in gens.values():
        common &= set(z["ids"].tolist())
    ids = np.array(sorted(common))
    if ids.size == 0:
        raise ValueError(f"no shared context ids for {config}/{ticker}")

    def take(z) -> np.ndarray:
        pos = {cid: i for i, cid in enumerate(z["ids"].tolist())}
        return z["vals"][[pos[c] for c in ids]]

    real = take(zr)
    gen = np.stack([take(gens[s]) for s in seeds])
    return real, gen, seeds


# --------------------------------------------------------------------------
# per-context scores
# --------------------------------------------------------------------------

def scores(real: np.ndarray, gen: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-context, per-horizon scores from k = gen.shape[0] rollouts.

    total       mean_i (x_i - y)^2     what "diverges from the truth" selects on
    spread_pop  var_i(x), ddof=0       the dispersion half of the exact identity
    spread      var_i(x), ddof=1       unbiased estimate of sigma^2, for the correction
    bias2_raw   (xbar - y)^2           systematic part, still carrying sigma^2/k
    bias2       bias2_raw - spread/k   bias-corrected; may go negative on noise

    Two variances on purpose. ``total = bias2_raw + spread_pop`` holds exactly,
    so shares computed from ``spread_pop`` partition the selected-on quantity and
    sum to one. The correction needs an unbiased sigma^2 instead, which is the
    ddof=1 form; mixing the two makes a share that can exceed 1 by k/(k-1).
    """
    k = gen.shape[0]
    xbar = gen.mean(axis=0)
    total = ((gen - real[None]) ** 2).mean(axis=0)
    spread_pop = gen.var(axis=0, ddof=0)
    spread = gen.var(axis=0, ddof=1) if k > 1 else np.zeros_like(xbar)
    bias2_raw = (xbar - real) ** 2
    bias2 = bias2_raw - spread / k if k > 1 else bias2_raw
    return {"total": total, "spread_pop": spread_pop, "spread": spread,
            "bias2_raw": bias2_raw, "bias2": bias2}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, ties averaged. Local so the module has no scipy dep."""
    good = np.isfinite(a) & np.isfinite(b)
    a, b = a[good], b[good]
    if a.size < 3:
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    r = np.empty(x.size, dtype=float)
    r[order] = np.arange(x.size, dtype=float)
    # average ties so that a score with many equal values is not ordered by
    # array position, which would manufacture agreement between two halves.
    _, start, cnt = np.unique(x[order], return_index=True, return_counts=True)
    for s, c in zip(start, cnt):
        if c > 1:
            r[order[s:s + c]] = r[order[s:s + c]].mean()
    return r


# --------------------------------------------------------------------------
# the correction: score a context against others whose future moved as much
# --------------------------------------------------------------------------

def stratify(score: np.ndarray, realised: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Rank ``score`` inside bins of |realised move|, returned on [0, 1].

    Raw squared error is dominated by how far the realised path actually
    travelled: a context whose future moved a lot scores as a failure no matter
    which rollouts are paired with it. Ranking inside a |y| bin asks the only
    question a pool should be built on --- given that the future moved this far,
    were these rollouts unusually wrong?

    |y| is a property of the context and is byte-identical across seeds, so
    binning on it leaks nothing about which rollouts were drawn.
    """
    out = np.empty(score.size, dtype=float)
    edges = np.quantile(np.abs(realised), np.linspace(0.0, 1.0, n_bins + 1))
    edges[-1] += 1e-12
    b = np.clip(np.digitize(np.abs(realised), edges[1:-1]), 0, n_bins - 1)
    for i in range(n_bins):
        m = b == i
        out[m] = _rank(score[m]) / max(m.sum() - 1, 1) if m.sum() > 2 else 0.5
    return out


def stratification_leak(n_bins: int = 10) -> float:
    """The residual |y| correlation that stratification cannot remove, = 1 / n_bins.

    Analytic, and confirmed to four decimals against a synthetic pure-|y| score. Any null
    measured on a stratified score has this as its floor; 0.095 measured at n_bins = 10 is
    the floor reached, not noise.
    """
    return 1.0 / n_bins


def pairing_nulls(real: np.ndarray, gen: np.ndarray, k: int, horizon_idx: int,
                  rng: np.random.Generator, stratified: bool) -> Dict[str, float]:
    """Four readings that separate "reliable" from "measuring what we meant".

    true         both halves scored with the correct rollout-to-context pairing
    shared       one permutation applied to both halves --- consistently wrong
    independent  a separate permutation per half --- the actual zero line
    cross        true half against permuted half

    ``shared`` is the one that matters. A consistently mis-paired score is as
    reliable as the correct one, sometimes more so, because reliability only
    asks whether two halves agree and a shared error is something they can agree
    on. So split-half reliability on its own certifies nothing; ``cross`` near
    ``independent`` is what shows the ranking actually needs the pairing.
    """
    S, N = gen.shape[0], real.shape[0]
    if 2 * k > S:
        return {}
    p = rng.permutation(S)
    A, B = p[:k], p[k:2 * k]
    p1, p2 = rng.permutation(N), rng.permutation(N)

    def sc(g: np.ndarray, idx: np.ndarray) -> np.ndarray:
        v = scores(real, g[idx])["total"][:, horizon_idx]
        return stratify(v, real[:, horizon_idx]) if stratified else v

    return {
        "k": k, "stratified": stratified,
        "true": spearman(sc(gen, A), sc(gen, B)),
        "shared": spearman(sc(gen[:, p1], A), sc(gen[:, p1], B)),
        "independent": spearman(sc(gen[:, p1], A), sc(gen[:, p2], B)),
        "cross": spearman(sc(gen, A), sc(gen[:, p1], B)),
    }


def rollouts_needed(ks: Sequence[int], rhos: Sequence[float],
                    targets: Sequence[float] = (0.80, 0.90),
                    max_resid_frac: float = 0.10) -> Dict[str, object]:
    """Turn a measured reliability curve into a required-k prediction, or refuse to.

    Two independent k-member estimates of one context's score correlate as
    rho_k = s2 / (s2 + n2/k), so ``(1/rho_k - 1) * k`` is the constant ``n2/s2`` if the
    model holds. It does not hold here: that quantity **rises with k in 7 of 8 tickers**
    (e.g. GOOG 4.68 -> 9.87 from k=1 to k=5). Least squares through the origin on x = 1/k
    is dominated by the k=1 point, so the fitted slope tracks the *smallest* observed
    ratio and the extrapolated k comes out **too low**.

    So this returns three things and lets the caller see the disagreement: the
    one-parameter fit, the fit through the largest-k point alone, and a two-parameter fit
    whose intercept measures the curvature. ``k_for_rho_*`` is emitted only when the
    one-parameter residual is small relative to the y-range; otherwise the key is absent
    and ``rejected_reason`` says why. A point estimate the data rejects is worse than no
    point estimate, because it gets quoted.
    """
    ks = np.asarray(ks, dtype=float)
    rhos = np.asarray(rhos, dtype=float)
    ok = np.isfinite(rhos) & (rhos > 0)
    ks, rhos = ks[ok], rhos[ok]
    x, y = 1.0 / ks, 1.0 / rhos - 1.0

    slope = float((x * y).sum() / (x * x).sum())
    resid = float(np.abs(y - slope * x).max())
    y_range = float(y.max() - y.min()) if y.size > 1 else float("inf")

    out: Dict[str, object] = {
        "noise_over_signal": slope,
        "max_abs_resid": resid,
        "y_range": y_range,
        "implied_ratio_by_k": {int(k): float((1.0 / r - 1.0) * k) for k, r in zip(ks, rhos)},
        "ratio_rises_with_k": bool(((1.0 / rhos[-1] - 1.0) * ks[-1]) > ((1.0 / rhos[0] - 1.0) * ks[0])),
    }
    # the largest-k point alone: the honest slope for extrapolating to large k
    out["noise_over_signal_largest_k"] = float((1.0 / rhos[-1] - 1.0) * ks[-1])
    if ks.size >= 3:                       # two-parameter fit; intercept measures curvature
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        out["two_param_slope"], out["two_param_intercept"] = float(coef[0]), float(coef[1])
        out["two_param_max_abs_resid"] = float(np.abs(y - A @ coef).max())

    if y_range > 0 and resid / y_range > max_resid_frac:
        out["rejected_reason"] = (
            f"one-parameter fit residual {resid:.3f} exceeds {max_resid_frac:.0%} of the "
            f"y-range {y_range:.3f}; the linear law does not describe this regime, so no "
            f"extrapolation from these k is supported. The direction of the error is that "
            f"k is LARGER than a one-parameter fit would say."
        )
        for t in targets:
            out[f"k_for_rho_{t:.2f}_largest_k_only"] = out["noise_over_signal_largest_k"] / (1.0 / t - 1.0)
    else:
        for t in targets:
            out[f"k_for_rho_{t:.2f}"] = slope / (1.0 / t - 1.0)
    return out


# --------------------------------------------------------------------------
# the two measurements
# --------------------------------------------------------------------------

def split_half(real: np.ndarray, gen: np.ndarray, k: int, n_draws: int,
               rng: np.random.Generator, key: str, horizon_idx: Optional[int] = None,
               stratified: bool = False) -> Dict[str, float]:
    """Score each context twice from disjoint halves of k rollouts each.

    Returns the mean rank correlation between the halves, plus the persistence
    of a top-decile selection: contexts picked by half A, re-scored on half B,
    expressed as the fraction of half A's own gap that survives. Persistence is
    the null control --- selecting on a pure-noise score gives 0.
    """
    S = gen.shape[0]
    if 2 * k > S:
        return {}
    rhos, persist = [], []
    for _ in range(n_draws):
        perm = rng.permutation(S)
        A, B = perm[:k], perm[k:2 * k]
        sa = scores(real, gen[A])[key]
        sb = scores(real, gen[B])[key]
        if stratified:
            sa = np.stack([stratify(sa[:, j], real[:, j]) for j in range(sa.shape[1])], 1)
            sb = np.stack([stratify(sb[:, j], real[:, j]) for j in range(sb.shape[1])], 1)
        hs = range(real.shape[1]) if horizon_idx is None else [horizon_idx]
        for h in hs:
            rhos.append(spearman(sa[:, h], sb[:, h]))
            n_top = max(1, round(0.10 * sa.shape[0]))
            top = np.argsort(-sa[:, h])[:n_top]
            gap_a = sa[top, h].mean() - sa[:, h].mean()
            gap_b = sb[top, h].mean() - sb[:, h].mean()
            if gap_a > 0:
                persist.append(gap_b / gap_a)
    return {
        "k": k,
        "rho_mean": float(np.nanmean(rhos)),
        "rho_sd": float(np.nanstd(rhos)),
        "top_decile_persistence": float(np.nanmean(persist)) if persist else float("nan"),
        "n_pairs": len(rhos),
    }


def regeneration_null(root: Path, cfg_a: str, cfg_b: str, ticker: str,
                      key: str) -> Dict[str, float]:
    """Same config, same seeds, two independent generations.

    This is the ceiling on any per-context ranking: whatever agreement two full
    regenerations of the identical setup fail to reach, no selection rule can
    recover, because the disagreement is in the generation and not in the data.
    """
    ra, ga, sa_seeds = load_arm(root, cfg_a, ticker)
    rb, gb, sb_seeds = load_arm(root, cfg_b, ticker)
    common = sorted(set(sa_seeds) & set(sb_seeds))
    if not common:
        return {}
    ga = ga[[sa_seeds.index(s) for s in common]]
    gb = gb[[sb_seeds.index(s) for s in common]]
    if not np.array_equal(ra, rb):
        # the realised future must be identical or the two sides are not
        # scoring the same question
        return {"error": "real arms differ between configs"}
    sa = scores(ra, ga)[key]
    sb = scores(rb, gb)[key]
    rhos = [spearman(sa[:, h], sb[:, h]) for h in range(ra.shape[1])]
    per = []
    for h in range(ra.shape[1]):
        n_top = max(1, round(0.10 * sa.shape[0]))
        top = np.argsort(-sa[:, h])[:n_top]
        ga_ = sa[top, h].mean() - sa[:, h].mean()
        gb_ = sb[top, h].mean() - sb[:, h].mean()
        if ga_ > 0:
            per.append(gb_ / ga_)
    return {
        "k": len(common),
        "seeds": common,
        "rho_per_horizon": [float(r) for r in rhos],
        "rho_mean": float(np.nanmean(rhos)),
        "top_decile_persistence": float(np.nanmean(per)) if per else float("nan"),
    }


def dispersion_share(real: np.ndarray, gen: np.ndarray) -> Dict[str, object]:
    """How much of the selected-on quantity is dispersion rather than bias.

    Reported both over all contexts and over the top decile by total error,
    because the pool is built from the top decile and the two can differ.
    """
    sc = scores(real, gen)
    out = {"k": int(gen.shape[0]), "per_horizon": []}
    for h, H in enumerate(HORIZONS[:real.shape[1]]):
        tot, spr, b2 = sc["total"][:, h], sc["spread_pop"][:, h], sc["bias2"][:, h]
        n_top = max(1, round(0.10 * tot.size))
        top = np.argsort(-tot)[:n_top]
        k = gen.shape[0]
        # spread_pop (ddof=0) underestimates sigma^2 by (k-1)/k while `total` is unbiased
        # for bias^2 + sigma^2, so the raw share understates the irreducible part by ~11%
        # at k=10. Report both; the unbiased one is the ceiling-on-gain number.
        share_all = float(spr.sum() / tot.sum()) if tot.sum() > 0 else float("nan")
        share_top = float(spr[top].sum() / tot[top].sum()) if tot[top].sum() > 0 else float("nan")
        corr = k / (k - 1) if k > 1 else float("nan")
        out["per_horizon"].append({
            "horizon": H,
            "spread_share_all": share_all,
            "spread_share_top_decile": share_top,
            "spread_share_all_unbiased": share_all * corr,
            "spread_share_top_decile_unbiased": share_top * corr,
            "max_removable_share_top_decile": 1.0 - share_top * corr,
            "frac_top_decile_bias2_nonpositive": float((b2[top] <= 0).mean()),
            "rho_total_vs_bias2": spearman(tot, b2),
        })
    return out


def pool_overlap(real: np.ndarray, gen: np.ndarray, horizon_idx: int,
                 frac: float = 0.10) -> float:
    """How much of the naive top-decile pool survives the correction.

    Both pools are the same size, so this is the fraction of the naive pool that
    the corrected rule also picks. Low overlap means the correction is not a
    refinement of the naive pool, it is a different pool.
    """
    v = scores(real, gen)["total"][:, horizon_idx]
    n = max(1, round(frac * v.size))
    a = set(np.argsort(-v)[:n].tolist())
    b = set(np.argsort(-stratify(v, real[:, horizon_idx]))[:n].tolist())
    return len(a & b) / n


# --------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--config", default="v5me3")
    ap.add_argument("--tickers", nargs="+",
                    default=["AMD", "AMZN", "GOOG", "INTC", "JPM", "META", "MSFT", "NFLX"])
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 3, 5])
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--key", default="total", choices=["total", "spread", "spread_pop", "bias2", "bias2_raw"])
    ap.add_argument("--null-configs", nargs=2, default=["v5me3repro", "v5me3repB"])
    ap.add_argument("--horizon-idx", type=int, default=2,
                    help="index into HORIZONS for the single-horizon readouts (default 2 = 50)")
    ap.add_argument("--null-k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)

    rng = np.random.default_rng(a.seed)
    report: Dict[str, object] = {"config": a.config, "key": a.key, "root": str(a.root),
                                 "tickers": {}, "null": {}}

    for tk in a.tickers:
        try:
            real, gen, seeds = load_arm(a.root, a.config, tk)
        except (FileNotFoundError, ValueError) as e:
            report["tickers"][tk] = {"error": str(e)}
            continue
        # Both reporting paths take the same horizon. Before this, the raw path passed
        # horizon_idx=None and averaged all seven horizons while the stratified path used
        # one, so the headline compared a 7-horizon average against a single horizon
        # (n_pairs 140 vs 20 in the emitted JSON). The numeric effect was small, which is
        # exactly why it survived.
        sh_raw = [r for k in a.ks
                  if (r := split_half(real, gen, k, a.draws, rng, a.key,
                                      horizon_idx=a.horizon_idx))]
        sh_str = [r for k in a.ks
                  if (r := split_half(real, gen, k, a.draws, rng, a.key,
                                      horizon_idx=a.horizon_idx, stratified=True))]
        entry: Dict[str, object] = {
            "n_contexts": int(real.shape[0]), "n_seeds": len(seeds), "seeds": seeds,
            "split_half_raw": sh_raw,
            "split_half_stratified": sh_str,
            "k_needed_raw": rollouts_needed([r["k"] for r in sh_raw],
                                            [r["rho_mean"] for r in sh_raw]),
            "k_needed_stratified": rollouts_needed([r["k"] for r in sh_str],
                                                   [r["rho_mean"] for r in sh_str]),
            "nulls_raw": pairing_nulls(real, gen, a.null_k, a.horizon_idx, rng, False),
            "nulls_stratified": pairing_nulls(real, gen, a.null_k, a.horizon_idx, rng, True),
            "dispersion": dispersion_share(real, gen),
            "horizon_idx": a.horizon_idx, "horizon": HORIZONS[a.horizon_idx],
            "pool_overlap_raw_vs_stratified": pool_overlap(real, gen, a.horizon_idx),
        }
        n_raw = {r["n_pairs"] for r in sh_raw}
        n_str = {r["n_pairs"] for r in sh_str}
        assert n_raw == n_str, (
            f"reporting paths disagree on the horizon set: raw n_pairs={n_raw}, "
            f"stratified n_pairs={n_str}. A reliability quoted from one path against the "
            f"other is not a comparison.")
        entry["stratification_leak"] = stratification_leak()
        report["tickers"][tk] = entry
        print(f"[{tk}] n={real.shape[0]} S={len(seeds)}", flush=True)
        for lab, rows in (("raw", sh_raw), ("strat", sh_str)):
            for r in rows:
                print(f"    {lab:>5} k={r['k']}  rho={r['rho_mean']:+.3f}+-{r['rho_sd']:.3f}"
                      f"  top-decile persistence={r['top_decile_persistence']:+.3f}")
        n = entry["nulls_stratified"]
        print(f"    nulls(strat, k={n['k']}): true {n['true']:+.3f}  shared {n['shared']:+.3f}"
              f"  independent {n['independent']:+.3f}  cross {n['cross']:+.3f}")
        print(f"    top-decile overlap raw vs stratified: "
              f"{entry['pool_overlap_raw_vs_stratified']:.1%}")

    ca, cb = a.null_configs
    for tk in a.tickers:
        try:
            n = regeneration_null(a.root, ca, cb, tk, a.key)
        except (FileNotFoundError, ValueError) as e:
            n = {"error": str(e)}
        report["null"][tk] = n
        if "rho_mean" in n:
            print(f"[null {tk}] {ca} vs {cb} k={n['k']} rho={n['rho_mean']:+.3f}"
                  f"  persistence={n['top_decile_persistence']:+.3f}")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=2))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
