#!/usr/bin/env python3
"""Direction accuracy and IC across horizons, before vs after mid-training.

Complements the distribution figures: those judge the law, these judge the
point forecast the same rollouts imply (3-seed ensemble mean per context).
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = Path("/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/crps-return-alignment-20260808")
T = Path("/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z")
OVL = Path("/projects/public/u6gb/overleaf/mid-train-kang-neurips-workshop2026/figures")
sys.path.insert(0, str(W / "run" / "mid_training"))
sys.path.insert(0, str(W / "src"))
from horizon_profile import member_multih, HS  # noqa: E402

BLUE, RED = "#4878b0", "#d1495b"


def arm_pred(roots):
    """id -> mean r_h vector over the seed members that carry the id."""
    ms = [member_multih(T / "data" / r / "member_0", "gen") for r in roots]
    ids = set(ms[0])
    for m in ms[1:]:
        ids &= set(m)
    return {i: np.mean([m[i] for m in ms], axis=0) for i in ids}


def main():
    real = member_multih(T / "data" / "hp_mdbase_s91001" / "member_0", "real")
    base = arm_pred([f"hp_mdbase_s9100{i}" for i in (1, 2, 3)])
    ft = arm_pred([f"hp_mdwmftb_s9100{i}" for i in (1, 2, 3)])
    ids = sorted(set(real) & set(base) & set(ft), key=int)
    R = np.stack([real[i] for i in ids])
    B = np.stack([base[i] for i in ids])
    F = np.stack([ft[i] for i in ids])
    n = len(ids)
    print(f"contexts: {n}")

    da_b, da_f, ic_b, ic_f = [], [], [], []
    for j in range(len(HS)):
        r, b, f = R[:, j], B[:, j], F[:, j]
        ok = np.isfinite(r) & np.isfinite(b) & np.isfinite(f)
        r, b, f = r[ok], b[ok], f[ok]
        nz = r != 0
        da_b.append((np.sign(b[nz]) == np.sign(r[nz])).mean())
        da_f.append((np.sign(f[nz]) == np.sign(r[nz])).mean())
        ic_b.append(np.corrcoef(b, r)[0, 1])
        ic_f.append(np.corrcoef(f, r)[0, 1])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    a1.plot(HS, np.array(da_b) * 100, "o-", color=BLUE, lw=2, label="before (base)")
    a1.plot(HS, np.array(da_f) * 100, "o-", color=RED, lw=2, label="after (wm_ft_b)")
    a1.axhline(50, color="gray", ls="--", lw=1.1)
    a1.text(HS[0], 50.4, "coin flip", fontsize=8, color="gray")
    a1.set_ylabel("direction accuracy (%)")
    a1.set_title("sign of the ensemble-mean return vs realised sign", fontsize=10)
    a1.legend(fontsize=9)

    se = 100 / np.sqrt(n)
    a2.plot(HS, np.array(ic_b) * 100, "o-", color=BLUE, lw=2)
    a2.plot(HS, np.array(ic_f) * 100, "o-", color=RED, lw=2)
    a2.axhline(0, color="gray", ls="--", lw=1.1)
    a2.fill_between(HS, -2 * se, 2 * se, color="gray", alpha=0.12)
    a2.text(HS[0], 0.3, "$\\pm 2$ null s.e.", fontsize=8, color="gray")
    a2.set_ylabel("Pearson IC ($\\times 100$)")
    a2.set_title("correlation of ensemble-mean return with realised return",
                 fontsize=10)

    for ax in (a1, a2):
        ax.set_xscale("log")
        ax.set_xticks(HS)
        ax.set_xticklabels(map(str, HS), fontsize=8)
        ax.minorticks_off()
        ax.set_xlabel("horizon $h$ (messages)")
        ax.grid(alpha=0.25)
    fig.suptitle("Directional skill by horizon, before vs after mid-training "
                 f"(GOOG 2026-01, {n} contexts $\\times$ 3 seeds)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for out in (T / "figs" / "fig_v5m_direction.pdf", OVL / "fig_v5m_direction.pdf",
                T / "figs" / "fig_v5m_direction_preview.png"):
        fig.savefig(out, dpi=140)
        print("wrote", out)


if __name__ == "__main__":
    main()
