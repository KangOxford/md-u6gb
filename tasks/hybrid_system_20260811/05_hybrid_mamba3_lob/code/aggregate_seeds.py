"""Aggregate the two arms across generation seeds, paired.

Pairing matters more than averaging here. Most of the published 16.6% cross-seed
spread in WS-21 is a common factor, whether a given draw of sequences happens to
be easy or hard, and both arms see the same draw under the same seed. Taking
hybrid(s) - baseline(s) per seed cancels that factor; averaging each arm first
and subtracting does not.

So the criterion is: are the paired differences the same sign across seeds, and
how large are they relative to their own spread. That is the pre-registered P1
shape (paired difference, interval away from zero), with seeds as the resampling
unit rather than blocks within one run.
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics as st

DIRPAT = re.compile(r"bench_\d{8}T\d{6}Z_j\d+_(?P<arm>[a-z0-9_]+?)_s(?P<seed>\d+)_\d+$")
AGE_ORDER = ["1-10", "11-25", "26-50", "51-100", "101-250", "before-window"]


def load_run(d):
    out = {}
    s = os.path.join(d, "summary.json")
    if os.path.exists(s):
        j = json.load(open(s))
        out.update({k: v for k, v in j.items() if isinstance(v, float)})
    for f in glob.glob(os.path.join(d, "refer_success_*.json")):
        j = json.load(open(f))
        c = j["cancel_delete"]
        out["L1"] = 100.0 * c["exact"] / c["n"]
        out["n_cd"] = c["n"]
        out["age"] = {k: 100.0 * v["exact"] / v["n"] for k, v in j["by_age"].items()}
        ev = j.get("by_event", {})
        for name in ("cancel", "delete", "execution"):
            if name in ev:
                out[f"n_{name}"] = ev[name]["n"]
    for f in glob.glob(os.path.join(d, "return_bench*.csv")):
        for r in csv.DictReader(open(f)):
            out["IC" + r["horizon"]] = float(r["IC"])
            out["DA" + r["horizon"]] = float(r["direction_acc"])
    return out


def fmt(v, nd=5):
    return "—" if v is None else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/lus/lfs1aip2/projects/public/u6gb/tasks/"
                                      "hybrid_system_20260811/05_hybrid_mamba3_lob")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="arm=seed=dir for runs whose directory name does not encode them")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    runs = {}                                  # (arm, seed) -> metrics
    for d in sorted(glob.glob(os.path.join(a.root, "bench_2026*"))):
        m = DIRPAT.search(os.path.basename(d))
        if not m:
            continue
        r = load_run(d)
        if r:
            runs[(m["arm"], int(m["seed"]))] = r
    for spec in a.extra:                       # e.g. hybrid_m3=2026=/path/to/dir
        arm, seed, d = spec.split("=", 2)
        r = load_run(d)
        if r:
            runs[(arm, int(seed))] = r

    arms = sorted({k[0] for k in runs})
    seeds = sorted({k[1] for k in runs})
    if len(arms) != 2:
        print(f"arms found: {arms} — need exactly two to pair")
    print(f"runs: {len(runs)}   arms: {arms}   seeds: {seeds}\n")

    METRICS = [("ws21", "WS-21 ↓", 5), ("ks21", "KS-21 ↓", 5), ("l1_21", "L1-21 ↓", 5),
               ("L1", "L1 exact % ↑", 4), ("IC250", "IC h=250 ↑", 4),
               ("DA250", "DirAcc h=250 ↑", 4)]

    print("per run")
    hdr = f"{'metric':<16}" + "".join(f"{a_ + '@' + str(s):>18}" for a_ in arms for s in seeds)
    print(hdr); print("-" * len(hdr))
    for key, name, nd in METRICS:
        row = f"{name:<16}"
        for a_ in arms:
            for s in seeds:
                row += f"{fmt(runs.get((a_, s), {}).get(key), nd):>18}"
        print(row)

    if len(arms) != 2:
        return
    base, hyb = arms if arms[0].startswith("base") else arms[::-1]

    print()
    print("paired differences   hybrid(seed) - baseline(seed)")
    hdr2 = f"{'metric':<16}" + "".join(f"{'s' + str(s):>13}" for s in seeds) + \
           f"{'mean':>13}{'spread':>10}{'same sign':>11}"
    print(hdr2); print("-" * len(hdr2))
    summary = {}
    for key, name, nd in METRICS:
        diffs = []
        row = f"{name:<16}"
        for s in seeds:
            b = runs.get((base, s), {}).get(key)
            h = runs.get((hyb, s), {}).get(key)
            d = None if (b is None or h is None) else h - b
            diffs.append(d)
            row += f"{('—' if d is None else f'{d:+.{nd}f}'):>13}"
        ok = [d for d in diffs if d is not None]
        if ok:
            mean = st.fmean(ok)
            spread = (max(ok) - min(ok)) if len(ok) > 1 else 0.0
            same = "yes" if all(d > 0 for d in ok) or all(d < 0 for d in ok) else "NO"
            row += f"{mean:+13.{nd}f}{spread:10.{nd}f}{same:>11}"
            summary[key] = {"per_seed": ok, "mean": mean, "spread": spread,
                            "same_sign": same == "yes", "n_seeds": len(ok)}
        print(row)

    print()
    print("L1 exact by reference age, paired difference in percentage points")
    hdr3 = f"{'age':<16}" + "".join(f"{'s' + str(s):>11}" for s in seeds) + f"{'mean':>11}{'same sign':>11}"
    print(hdr3); print("-" * len(hdr3))
    age_summary = {}
    for k in AGE_ORDER:
        diffs, row = [], f"{k:<16}"
        for s in seeds:
            b = runs.get((base, s), {}).get("age", {}).get(k)
            h = runs.get((hyb, s), {}).get("age", {}).get(k)
            d = None if (b is None or h is None) else h - b
            diffs.append(d)
            row += f"{('—' if d is None else f'{d:+.2f}'):>11}"
        ok = [d for d in diffs if d is not None]
        if ok:
            same = "yes" if all(d > 0 for d in ok) or all(d < 0 for d in ok) else "NO"
            row += f"{st.fmean(ok):+11.2f}{same:>11}"
            age_summary[k] = {"per_seed": ok, "mean": st.fmean(ok), "same_sign": same == "yes"}
        print(row)

    print()
    print("Reading: a metric whose paired differences agree in sign across every seed is")
    print("not explained by which draw of sequences a seed happened to produce. A metric")
    print("whose mean difference is smaller than its own spread across seeds is not")
    print("separable from that draw, whatever its point value looks like.")

    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump({"arms": arms, "seeds": seeds,
                   "runs": {f"{k[0]}@{k[1]}": {kk: vv for kk, vv in v.items() if kk != "age"}
                            for k, v in runs.items()},
                   "paired": summary, "paired_by_age": age_summary},
                  open(a.out, "w"), indent=1)
        print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
