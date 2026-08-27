"""A1: offline weight-space plasticity diagnostics across checkpoint ages.

Reads three checkpoints of one run (early / mid / late) on CPU and reports the
weight-space correlates of plasticity loss that need no data or gradients:
global L2 norm, mean absolute parameter magnitude (Zyphra's rising-magnitude
correlate), and per-matrix spectral norms (whose growth is the mechanism the
spectral-regularization line targets).

Usage:
  JAX_PLATFORMS=cpu python probe_weights_offline.py \
      --root /path/to/checkpoints_selftrain/j5705912_b30675li_5705912 \
      --steps 275 33575 69378 --out results/a1_weight_probes.json
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_params(step_dir: Path):
    import jax
    import orbax.checkpoint as ocp

    ckptr = ocp.PyTreeCheckpointer()
    meta = ckptr.metadata(str(step_dir / "state"))
    tree_meta = getattr(meta, "item_metadata", None) or getattr(meta, "tree", None) or meta
    restore_args = jax.tree_util.tree_map(
        lambda _: ocp.RestoreArgs(restore_type=np.ndarray), tree_meta
    )
    state = ckptr.restore(str(step_dir / "state"), restore_args=restore_args)
    # TrainState layout: prefer the params subtree; fall back to the whole tree.
    if isinstance(state, dict) and "params" in state:
        return state["params"]
    return state


def flatten(tree, prefix=""):
    out = []
    if isinstance(tree, dict):
        for k in sorted(tree):
            out += flatten(tree[k], f"{prefix}/{k}" if prefix else str(k))
    elif tree is None:
        pass
    else:
        arr = np.asarray(tree)
        if np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.complexfloating):
            out.append((prefix, arr))
    return out


def probe(root: Path, step: int) -> dict:
    leaves = flatten(load_params(root / str(step)))
    is_embed = lambda name: "embed" in name.lower()
    total_sq = 0.0
    total_abs = 0.0
    total_n = 0
    ne_sq = 0.0
    ne_abs = 0.0
    ne_n = 0
    spec = {}
    for name, a in leaves:
        if np.issubdtype(a.dtype, np.complexfloating):
            mag = np.abs(a)
        else:
            mag = np.abs(a.astype(np.float64))
        sq = float(np.square(mag).sum())
        ab = float(mag.sum())
        n = int(mag.size)
        total_sq += sq
        total_abs += ab
        total_n += n
        if not is_embed(name):
            ne_sq += sq
            ne_abs += ab
            ne_n += n
        if a.ndim == 2 and min(a.shape) >= 8 and not is_embed(name):
            m = a.astype(np.complex128) if np.issubdtype(a.dtype, np.complexfloating) else a.astype(np.float64)
            spec[name] = float(np.linalg.svd(m, compute_uv=False)[0])
    svals = np.array(sorted(spec.values()))
    return {
        "step": step,
        "n_params_total": total_n,
        "l2_total": float(np.sqrt(total_sq)),
        "l2_non_embed": float(np.sqrt(ne_sq)),
        "mean_abs_total": total_abs / max(total_n, 1),
        "mean_abs_non_embed": ne_abs / max(ne_n, 1),
        "n_matrices": len(spec),
        "spectral_norm_mean": float(svals.mean()) if len(svals) else None,
        "spectral_norm_median": float(np.median(svals)) if len(svals) else None,
        "spectral_norm_max": float(svals.max()) if len(svals) else None,
        "spectral_top5": {k: round(v, 4) for k, v in sorted(spec.items(), key=lambda kv: -kv[1])[:5]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--steps", required=True, type=int, nargs="+")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = []
    for s in args.steps:
        r = probe(args.root, s)
        rows.append(r)
        print(json.dumps(r, indent=None))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"root": str(args.root), "rows": rows}, indent=2))
        print(f"wrote {args.out}")

    if len(rows) >= 2:
        a, b = rows[0], rows[-1]
        print("\n=== early -> late ratios ===")
        for k in ["l2_non_embed", "mean_abs_non_embed", "spectral_norm_mean", "spectral_norm_median", "spectral_norm_max"]:
            if a.get(k) and b.get(k):
                print(f"{k}: {a[k]:.4f} -> {b[k]:.4f}  (x{b[k]/a[k]:.3f})")


if __name__ == "__main__":
    main()
