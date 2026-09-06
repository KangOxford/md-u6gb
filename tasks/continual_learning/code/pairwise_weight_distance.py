"""True pairwise distance between two checkpoints' parameter trees.

X4 was declared settled using `l2_non_embed` from `probe_weights_offline.py`, which is
`sqrt(sum(theta**2))` for ONE checkpoint -- a norm. The quantity compared across roots was
therefore `| ||theta_A|| - ||theta_B|| |`, which is not a distance: two different parameter
vectors can have identical norms, and the triangle inequality makes the norm gap a lower
bound on the distance rather than an estimate of it. This computes `||theta_A - theta_B||`
parameter by parameter, which is what the lineage claim needed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from probe_weights_offline import flatten, load_params


def pairwise(a_dir: Path, b_dir: Path) -> dict:
    fa = dict(flatten(load_params(a_dir)))
    fb = dict(flatten(load_params(b_dir)))
    common = sorted(set(fa) & set(fb))
    if not common:
        raise SystemExit(f"no shared parameter names between {a_dir} and {b_dir}")
    d_sq = ne_sq = a_sq = b_sq = 0.0
    n_ident = n_tot = 0
    for k in common:
        x, y = np.asarray(fa[k], dtype=np.float64), np.asarray(fb[k], dtype=np.float64)
        if x.shape != y.shape:
            continue
        d = x - y
        s = float((d * d).sum())
        d_sq += s
        a_sq += float((x * x).sum())
        b_sq += float((y * y).sum())
        if "embed" not in k.lower():
            ne_sq += s
        n_tot += 1
        n_ident += int(np.array_equal(x, y))
    na, nb = np.sqrt(a_sq), np.sqrt(b_sq)
    return {
        "a": str(a_dir), "b": str(b_dir),
        "n_shared_arrays": n_tot, "n_bitwise_identical_arrays": n_ident,
        "true_distance_l2": float(np.sqrt(d_sq)),
        "true_distance_l2_non_embed": float(np.sqrt(ne_sq)),
        "norm_a": float(na), "norm_b": float(nb),
        "norm_gap": float(abs(na - nb)),
        # the gap is a LOWER bound on the distance; how loose it is says how badly the
        # norm-difference stood in for the distance
        "distance_over_norm_gap": float(np.sqrt(d_sq) / abs(na - nb)) if abs(na - nb) > 0 else None,
        "relative_distance": float(np.sqrt(d_sq) / na) if na > 0 else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=Path, required=True, help="root/<step> directory")
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    r = pairwise(args.a, args.b)
    print(json.dumps(r, indent=1))
    if args.out:
        args.out.write_text(json.dumps(r, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
