"""(7.1) held-out CE learning curves for both arms.

Reads the JSON evaluate_model_zoo_ce.py writes, so the figure cannot drift from
the numbers. Right axis restates the same quantity per message, which is what
the task asks for: 26-token messages count every position in the loss, so
nats/message = nats/token x 26 exactly. That identity does NOT survive a change
of encoding, and is only used here because both arms are 26-token.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/lus/lfs1aip2/projects/public/u6gb/tasks/hybrid_system_20260811/05_hybrid_mamba3_lob/figures"
TOKENS_PER_MSG = 26


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", default=None)
    ap.add_argument("--out", default=os.path.join(OUT, "ce_curves.png"))
    a = ap.parse_args()

    arms = [json.load(open(p)) for p in a.json]
    names = a.names or [d.get("architecture", "?") for d in arms]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    fig.subplots_adjust(wspace=0.28)

    ax = axes[0]
    series = []
    for arm, name, mk in zip(arms, names, ["o", "s", "^"]):
        rows = sorted(arm["results"], key=lambda r: r["step"])
        xs = [r["step"] for r in rows]
        ys = [r["validation_ce"] for r in rows]
        series.append(dict(zip(xs, ys)))
        ax.plot(xs, ys, marker=mk, lw=2.2, ms=7, label=name)
    ax.set_xlabel("training step", fontsize=10)
    ax.set_ylabel("held-out CE (nats / token)", fontsize=10)
    ax.set_title("Held-out cross entropy, GOOG 2026-01", fontsize=12.5, pad=12)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    sec = ax.secondary_yaxis(
        "right", functions=(lambda v: v * TOKENS_PER_MSG, lambda v: v / TOKENS_PER_MSG))
    sec.set_ylabel("nats / message  (x26)", fontsize=10)

    # ------------------------------------------------- panel 2: the difference
    ax = axes[1]
    if len(series) >= 2:
        common = sorted(set(series[0]) & set(series[1]))
        d = [100.0 * (series[1][s] - series[0][s]) / series[0][s] for s in common]
        colors = ["#55A868" if v < 0 else "#C44E52" for v in d]
        ax.bar([str(s) for s in common], d, color=colors, width=0.6)
        for x, v in zip(range(len(common)), d):
            ax.annotate(f"{v:+.2f}%", (x, v), textcoords="offset points",
                        xytext=(0, 6 if v > 0 else -14), ha="center", fontsize=9)
        ax.axhline(0, color="black", lw=1)
        ax.set_xlabel("training step", fontsize=10)
        ax.set_ylabel(f"{names[1]} relative to {names[0]} (%)", fontsize=10)
        ax.set_title("Lower is better for the hybrid", fontsize=12.5, pad=12)
        lo, hi = min(d), max(d)
        pad = max(2.0, 0.25 * (hi - lo))
        ax.set_ylim(lo - pad, hi + pad)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.5, -0.20,
                "CE is a weak gate here: a change once moved CE 0.8% while moving\n"
                "LOB-Bench KS 102%. Read it as 'did anything break', not as quality.",
                transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color="#333")

    os.makedirs(OUT, exist_ok=True)
    fig.savefig(a.out, dpi=150, bbox_inches="tight", facecolor="white")
    print(a.out)


if __name__ == "__main__":
    main()
