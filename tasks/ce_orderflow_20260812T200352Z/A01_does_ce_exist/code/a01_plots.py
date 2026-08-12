#!/usr/bin/env python3
"""Figures for A01. Reads only out/a01_g*.json -- no recomputation, so a figure
can never disagree with the number in the report."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402

FIELDS = ["event_type", "direction", "price_rel", "size", "log10_dt"]
HOR = [("500", "2x"), ("1000", "3x"), ("1500", "4x"), ("3500", "8x")]
ARMC = {"draft": "#1f77b4", "corr": "#2ca02c", "rand": "#d62728"}
ARML = {"draft": "pretrained draft", "corr": "DFM corrected", "rand": "random-P"}
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load():
    out = {}
    for h, _ in HOR:
        p = f"{HERE}/out/a01_g{h}.json"
        if os.path.exists(p):
            out[h] = json.load(open(p))
    return out


def fig_curves(D, arms=("draft",), fname="fig1_excess_curves.png"):
    hs = [h for h, _ in HOR if h in D]
    fig, ax = plt.subplots(len(FIELDS), len(hs),
                           figsize=(3.4 * len(hs), 2.3 * len(FIELDS)),
                           squeeze=False)
    for i, f in enumerate(FIELDS):
        for j, h in enumerate(hs):
            a, d = ax[i][j], D[h]
            g = np.asarray(d["grid"], float)
            for k in arms:
                if k not in d["res"][f]["arms"]:
                    continue
                r = d["res"][f]["arms"][k]
                y = np.asarray(r["curve_excess"], float)
                a.plot(g, y, lw=1.4, color=ARMC[k], label=ARML[k])
                ok = np.isfinite(y)
                if ok.sum() > 3:
                    p = np.polyfit(g[ok], y[ok], 1)
                    a.plot(g, np.polyval(p, g), ls="--", lw=1.0,
                           color=ARMC[k], alpha=.75)
            a.axhline(0, color="0.35", lw=.9)
            a.axhspan(-.02, .02, color="0.85", zorder=0)
            if i == 0:
                a.set_title(f"{int(h)+500} msgs = {dict(HOR)[h]} window", fontsize=9)
            if j == 0:
                a.set_ylabel(f, fontsize=9)
            if i == len(FIELDS) - 1:
                a.set_xlabel("generation position m", fontsize=8)
            a.tick_params(labelsize=7)
    ax[0][0].legend(fontsize=7, frameon=False)
    fig.suptitle("excess = $D_m$ - floor, order-flow fields.  dashed = OLS fit; "
                 "the SLOPE is the compound error, the height is not.",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, .975])
    fig.savefig(f"{HERE}/figs/{fname}", dpi=140)
    plt.close(fig)


def fig_slopes(D):
    hs = [h for h, _ in HOR if h in D]
    x = np.arange(len(hs))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for panel, (arm, ttl) in enumerate([("draft", "pretrained draft: is there "
                                         "compound error?"),
                                        ("corr", "after DFM correction")]):
        a = ax[panel]
        for i, f in enumerate(FIELDS):
            ys, los, his, sig = [], [], [], []
            for h in hs:
                r = D[h]["res"][f]["arms"].get(arm)
                if r is None:
                    ys.append(np.nan); los.append(np.nan); his.append(np.nan)
                    sig.append(False); continue
                ys.append(r["slope_excess"])
                los.append(r["slope_excess_ci"][0]); his.append(r["slope_excess_ci"][1])
                sig.append(r["has_CE"])
            ys, los, his = map(np.asarray, (ys, los, his))
            off = (i - 2) * .07
            a.errorbar(x + off, ys, yerr=[ys - los, his - ys], fmt="none",
                       ecolor=f"C{i}", elinewidth=1.2, capsize=2.5, alpha=.8)
            a.scatter(x + off, ys, s=42, c=[f"C{i}" if s else "white" for s in sig],
                      edgecolors=f"C{i}", linewidths=1.4, zorder=3,
                      label=f if panel == 0 else None)
        a.axhline(0, color="0.3", lw=1)
        a.set_xticks(x); a.set_xticklabels([f"{int(h)+500}\n{dict(HOR)[h]}" for h in hs])
        a.set_xlabel("total sequence length / training window")
        a.set_ylabel("slope of excess (per 100 messages)")
        a.set_title(ttl, fontsize=10)
    ax[0].legend(fontsize=8, frameon=False, ncol=2)
    fig.suptitle("filled = 95% day-block CI excludes zero", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, .95])
    fig.savefig(f"{HERE}/figs/fig2_slope_vs_horizon.png", dpi=140)
    plt.close(fig)


def fig_z(D):
    hs = [h for h, _ in HOR if h in D]
    fig, ax = plt.subplots(1, len(FIELDS), figsize=(3.0 * len(FIELDS), 3.2),
                           squeeze=False)
    for i, f in enumerate(FIELDS):
        a = ax[0][i]
        for j, h in enumerate(hs):
            d = D[h]; g = np.asarray(d["grid"], float)
            for k in ("draft", "corr"):
                r = d["res"][f]["arms"].get(k)
                if r is None:
                    continue
                a.plot(g, np.asarray(r["z"], float), lw=1.1, color=ARMC[k],
                       alpha=.45 + .18 * j,
                       label=ARML[k] if (i == 0 and j == 0) else None)
        for s in (-1, 1):
            a.axhline(s, color="0.6", ls=":", lw=.9)
        a.axhline(0, color="0.3", lw=.9)
        a.set_title(f, fontsize=9); a.set_xlabel("m", fontsize=8)
        a.tick_params(labelsize=7)
    ax[0][0].set_ylabel(r"$z(m)=(\mu_{gen}-\mu_{true})/\sigma_{true}$", fontsize=9)
    ax[0][0].legend(fontsize=7, frameon=False)
    fig.suptitle("mean drift. $D_m$ normalises by the TRUE moments, so it is "
                 "nearly blind to a distribution that keeps its shape and "
                 "walks its centre away.", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, .92])
    fig.savefig(f"{HERE}/figs/fig3_mean_drift.png", dpi=140)
    plt.close(fig)


def main():
    D = load()
    if not D:
        raise SystemExit("no results in out/")
    os.makedirs(f"{HERE}/figs", exist_ok=True)
    fig_curves(D, ("draft",), "fig1_excess_curves_draft.png")
    fig_curves(D, ("draft", "corr", "rand"), "fig1b_excess_curves_all_arms.png")
    fig_slopes(D)
    fig_z(D)
    print("wrote", sorted(os.listdir(f"{HERE}/figs")))


if __name__ == "__main__":
    main()
