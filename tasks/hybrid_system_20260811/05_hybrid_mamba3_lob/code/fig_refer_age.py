"""The baseline's reference-recall decay, which is the mechanism criterion.

One panel per question: how does exact recall decay with distance, where does
the resolver's fallback take over, and how much of the total volume sits in
each regime. Reads the JSON that refer_success.py writes, so the figure cannot
drift from the numbers.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/lus/lfs1aip2/projects/public/u6gb/tasks/hybrid_system_20260811/05_hybrid_mamba3_lob/figures"
ORDER = ["1-10", "11-25", "26-50", "51-100", "101-250", "before-window"]
LABEL = {"1-10": "1–10", "11-25": "11–25", "26-50": "26–50",
         "51-100": "51–100", "101-250": "101–250",
         "before-window": "before\nthe window"}

C_EXACT = "#4C72B0"
C_FALL = "#DD8452"
C_MISS = "#C44E52"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", nargs="+", required=True,
                    help="refer_success_*.json, one per arm")
    ap.add_argument("--names", nargs="+", default=None)
    ap.add_argument("--out", default=os.path.join(OUT, "refer_recall_by_age.png"))
    a = ap.parse_args()

    arms = [json.load(open(p)) for p in a.json]
    names = a.names or [d["label"] for d in arms]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    fig.subplots_adjust(wspace=0.30)
    counts = "  ".join(f"{n}: {sum(v['n'] for v in a['by_age'].values()):,} cancel/delete"
                       for a, n in zip(arms, names))
    fig.suptitle(f"Reference recall, {arms[0]['sequences']:,} sequences   |   {counts}",
                 fontsize=10.5, y=1.05)

    # -------------------------------------------------- panel 1: the decay
    ax = axes[0]
    xs = list(range(len(ORDER)))
    for arm, name, mk in zip(arms, names, ["o", "s", "^"]):
        ys = [100.0 * arm["by_age"][k]["exact"] / arm["by_age"][k]["n"]
              if k in arm["by_age"] else None for k in ORDER]
        ax.plot(xs, ys, marker=mk, lw=2.2, ms=8, label=name)
        for x, y in zip(xs, ys):
            if y is not None:
                ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                            xytext=(0, 9), ha="center", fontsize=8.5)
    ax.axvspan(4.5, 5.5, color="#EEEEEE", zorder=0)
    ax.text(5.0, 62, "no amount of recall\nreaches this bin",
            ha="center", va="top", fontsize=8, color="#666")
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL[k] for k in ORDER], fontsize=9)
    ax.set_xlabel("age of the referenced order (messages back)", fontsize=10)
    ax.set_ylabel("exact nanosecond recall, L1 (%)", fontsize=10)
    ax.set_title("Recall decays monotonically with distance", fontsize=12.5, pad=12)
    ax.set_ylim(25, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9, frameon=False, loc="lower left",
              bbox_to_anchor=(-0.02, -0.03))
    ax.spines[["top", "right"]].set_visible(False)

    # ------------------------------------ panel 2: what fills the gap, arm 0
    ax = axes[1]
    arm = arms[0]
    keys = [k for k in ORDER if k in arm["by_age"]]
    ex = [100.0 * arm["by_age"][k]["exact"] / arm["by_age"][k]["n"] for k in keys]
    fa = [100.0 * arm["by_age"][k]["fallback"] / arm["by_age"][k]["n"] for k in keys]
    mi = [100.0 * arm["by_age"][k]["miss"] / arm["by_age"][k]["n"] for k in keys]
    xs = list(range(len(keys)))
    ax.bar(xs, ex, color=C_EXACT, label="L1 exact")
    ax.bar(xs, fa, bottom=ex, color=C_FALL, label="L2 fallback (no threshold)")
    ax.bar(xs, mi, bottom=[e + f for e, f in zip(ex, fa)], color=C_MISS, label="unresolved")
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL[k] for k in keys], fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_ylabel("share of referencing messages (%)", fontsize=10)
    ax.set_xlabel("age of the referenced order (messages back)", fontsize=10)
    ax.set_title("What fills the gap", fontsize=12.5, pad=12)
    ax.legend(fontsize=8.5, frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.5, -0.22,
            "The fallback never fails, so it can cancel a different order.\n"
            "A high combined rate is a statement about the resolver, not the model.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color="#333")

    # -------------------------------------------- panel 3: where the volume is
    ax = axes[2]
    ns = [arm["by_age"][k]["n"] for k in keys]
    tot = sum(ns)
    bars = ax.barh(range(len(keys)), [100.0 * n / tot for n in ns], color="#8C8C8C", height=0.6)
    for b, n in zip(bars, ns):
        ax.text(b.get_width() + 0.6, b.get_y() + b.get_height() / 2,
                f"{n:,}", va="center", fontsize=9)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([LABEL[k].replace("\n", " ") for k in keys], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("share of all cancel/delete messages (%)", fontsize=10)
    ax.set_title("Where the volume sits", fontsize=12.5, pad=12)
    ax.set_xlim(0, 42)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.5, -0.22,
            "A third of all references point before the conditioning window.\n"
            "That bin bounds what any architecture can win here.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color="#333")

    os.makedirs(OUT, exist_ok=True)
    fig.savefig(a.out, dpi=150, bbox_inches="tight", facecolor="white")
    print(a.out)


if __name__ == "__main__":
    main()
