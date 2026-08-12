"""Four-arm report: {baseline, hybrid} x {500-message, 2,000-message} context.

The question the grid answers is not "is the hybrid better" but "does the
hybrid's advantage survive a longer window". Reading either row alone cannot
tell those apart, which is why the table is built as a grid and the headline
number is a difference of differences.

Every figure reads the benchmark JSON directly, so a number in the report and a
number in the artifact cannot drift apart.

Age bins: the 2k runs use the extended set, whose first five bins are identical
to the 500 set by construction. Only those five are compared across contexts;
the three long bins exist only at 2k and are reported on their own.
"""
import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/lus/lfs1aip2/projects/public/u6gb/tasks/hybrid_system_20260811/05_hybrid_mamba3_lob"
SHARED_BINS = ["1-10", "11-25", "26-50", "51-100", "101-250"]
LONG_BINS = ["251-500", "501-1000", "1001-2000"]
C_BASE, C_HYB = "#4C72B0", "#DD8452"


def load(spec):
    """Metrics from one or more benchmark directories, comma separated.

    Multiple paths exist because a run's pieces do not always land together: the
    published baseline keeps LOB-Bench in the bench directory and the reference
    and return artefacts in a backfill directory produced later. Later paths win
    on conflict, so the caller controls precedence by ordering.
    """
    o = {}
    if not spec:
        return o
    for d in [x for x in str(spec).split(",") if x]:
        o.update(_load_one(d))
    return o


def _load_one(d):
    o = {}
    if not os.path.isdir(d):
        return o
    s = os.path.join(d, "summary.json")
    if os.path.exists(s):
        o.update({k: v for k, v in json.load(open(s)).items() if isinstance(v, float)})
    for f in glob.glob(os.path.join(d, "refer_success_*.json")):
        j = json.load(open(f))
        c = j["cancel_delete"]
        o["L1"] = 100.0 * c["exact"] / c["n"]
        o["MISS"] = 100.0 * c["miss"] / c["n"]
        o["n_cd"] = c["n"]
        o["age"] = {k: 100.0 * v["exact"] / v["n"] for k, v in j["by_age"].items()}
        o["age_n"] = {k: v["n"] for k, v in j["by_age"].items()}
    for f in glob.glob(os.path.join(d, "return_bench*.csv")):
        for r in csv.DictReader(open(f)):
            o["IC" + r["horizon"]] = float(r["IC"])
            o["DA" + r["horizon"]] = float(r["direction_acc"])
    return o


def fmt(v, nd=5):
    return "—" if v is None else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base500", default=None)
    ap.add_argument("--hyb500", default=None)
    ap.add_argument("--base2k", required=True)
    ap.add_argument("--hyb2k", required=True)
    ap.add_argument("--ce", default=None, help="json: {arm: ce} for (7.1)")
    ap.add_argument("--out", default=os.path.join(ROOT, "figures", "ctx2k_grid.png"))
    a = ap.parse_args()

    A = {k: load(getattr(a, k)) for k in ("base500", "hyb500", "base2k", "hyb2k")}
    ce = json.load(open(a.ce)) if a.ce else {}

    METRICS = [("ws21", "(7.3) WS-21 ↓", 5, -1), ("ks21", "(7.3) KS-21 ↓", 5, -1),
               ("l1_21", "(7.3) L1-21 ↓", 5, -1), ("L1", "(7.4) L1 exact % ↑", 4, +1),
               ("MISS", "(7.4) unresolved % ↓", 4, -1),
               ("IC250", "(7.2) IC h=250 ↑", 4, +1)]

    print("=" * 86)
    print("  four-arm grid   {baseline, hybrid} x {500, 2000} messages")
    print("=" * 86)
    hdr = f"{'metric':<22}" + "".join(f"{k:>15}" for k in A)
    print(hdr); print("-" * len(hdr))
    for key, name, nd, _ in METRICS:
        print(f"{name:<22}" + "".join(f"{fmt(A[k].get(key), nd):>15}" for k in A))
    if ce:
        print(f"{'(7.1) CE ↓':<22}" + "".join(f"{fmt(ce.get(k), 6):>15}" for k in A))

    # ---- the headline: does the hybrid advantage survive the longer window ---
    print()
    print("hybrid minus baseline, within each context, and the change between them")
    hdr2 = f"{'metric':<22}{'at 500':>13}{'at 2000':>13}{'change':>13}{'verdict':>26}"
    print(hdr2); print("-" * len(hdr2))
    for key, name, nd, sign in METRICS:
        d5 = d2 = None
        if A["base500"].get(key) is not None and A["hyb500"].get(key) is not None:
            d5 = A["hyb500"][key] - A["base500"][key]
        if A["base2k"].get(key) is not None and A["hyb2k"].get(key) is not None:
            d2 = A["hyb2k"][key] - A["base2k"][key]
        v = "—"
        if d5 is not None and d2 is not None:
            g5, g2 = sign * d5 > 0, sign * d2 > 0
            if g5 and g2:
                v = "holds" if abs(d2) >= abs(d5) * 0.5 else "holds, weaker"
            elif g5 and not g2:
                v = "LOST at 2000"
            elif not g5 and g2:
                v = "appears at 2000"
            else:
                v = "absent in both"
        ch = "—" if (d5 is None or d2 is None) else f"{d2 - d5:+.{nd}f}"
        print(f"{name:<22}{('—' if d5 is None else f'{d5:+.{nd}f}'):>13}"
              f"{('—' if d2 is None else f'{d2:+.{nd}f}'):>13}{ch:>13}{v:>26}")

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))
    fig.subplots_adjust(wspace=0.26)

    ax = axes[0]
    x = np.arange(len(SHARED_BINS))
    for ctx, mk, ls in (("500", "o", "--"), ("2k", "s", "-")):
        b, h = A[f"base{ctx}"].get("age", {}), A[f"hyb{ctx}"].get("age", {})
        if not b or not h:
            continue
        d = [h.get(k, np.nan) - b.get(k, np.nan) for k in SHARED_BINS]
        ax.plot(x, d, mk + ls, lw=2.4, ms=8,
                color=C_HYB if ctx == "500" else C_BASE,
                label=f"context {ctx} messages")
        for xi, v in zip(x, d):
            if not np.isnan(v):
                ax.annotate(f"{v:+.1f}", (xi, v), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=8.5)
    ax.axhline(0, color="black", lw=1.1)
    ax.set_xticks(x); ax.set_xticklabels(SHARED_BINS, fontsize=9)
    ax.set_xlabel("reference age (messages back)", fontsize=10)
    ax.set_ylabel("exact recall, hybrid − baseline (pp)", fontsize=10)
    ax.set_title("Does the recall gain survive a longer window", fontsize=12.5, pad=12)
    ax.legend(fontsize=9.5, frameon=False); ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.5, -0.21, "Only the five bins both contexts share are plotted; the 2k runs\n"
                        "also carry 251-500, 501-1000 and 1001-2000, shown at right.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color="#333")

    ax = axes[1]
    b2, h2 = A["base2k"].get("age", {}), A["hyb2k"].get("age", {})
    bins = [k for k in SHARED_BINS + LONG_BINS if k in b2 and k in h2]
    if bins:
        xs = np.arange(len(bins)); w = 0.38
        ax.bar(xs - w / 2, [b2[k] for k in bins], w, color=C_BASE, label="baseline 2k")
        ax.bar(xs + w / 2, [h2[k] for k in bins], w, color=C_HYB, label="hybrid 2k")
        ax.set_xticks(xs); ax.set_xticklabels(bins, fontsize=8.5, rotation=20)
        ax.set_ylabel("exact reference recall (%)", fontsize=10)
        ax.set_title("Reach at 2,000 messages, including the new far bins",
                     fontsize=12.5, pad=12)
        ax.legend(fontsize=9.5, frameon=False); ax.grid(alpha=0.22, axis="y")
        ax.spines[["top", "right"]].set_visible(False)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nfigure: {a.out}")


if __name__ == "__main__":
    main()
