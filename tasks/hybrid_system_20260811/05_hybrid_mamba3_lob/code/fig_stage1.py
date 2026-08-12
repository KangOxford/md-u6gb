"""Stage-1 figure: what was built, what it costs, and what it has to beat."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = "/lus/lfs1aip2/projects/public/u6gb/tasks/hybrid_system_20260811/05_hybrid_mamba3_lob/figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams["axes.unicode_minus"] = False

C_REC = "#4C72B0"
C_ATT = "#C44E52"
C_BASE = "#7F7F7F"
C_GAIN = "#55A868"

MEASURED = {
    "parameters": (33_610_439, 35_435_423),
    "s / step": (0.3703, 0.4092),
    "peak HBM (GB)": (71.6, 67.1),
}
ATTN_AT, N_LAYERS = 3, 6

fig = plt.figure(figsize=(16, 5.6))
gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.1, 1.5], wspace=0.30)

# ------------------------------------------------------- panel 1: the stack
ax = fig.add_subplot(gs[0, 0])
ax.set_title("Fused trunk composition", fontsize=12.5, pad=14)
for col, (name, attn) in enumerate([("baseline", None), ("hybrid", ATTN_AT)]):
    x = col * 1.35
    for i in range(N_LAYERS):
        hit = attn is not None and i == attn
        ax.add_patch(Rectangle((x, i), 1.15, 0.80,
                               facecolor=C_ATT if hit else C_REC,
                               edgecolor="white", linewidth=1.8))
        ax.text(x + 0.575, i + 0.40, "ATTN" if hit else "mamba3",
                ha="center", va="center", color="white", fontsize=8.5,
                fontweight="bold" if hit else "normal")
    ax.text(x + 0.575, -0.75, name, ha="center", va="center", fontsize=11)
for i in range(N_LAYERS):
    ax.text(-0.22, i + 0.40, str(i), ha="center", va="center",
            fontsize=8, color="#666")
ax.annotate("", xy=(1.32, ATTN_AT + 0.40), xytext=(1.20, ATTN_AT + 0.40),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.8))
ax.text(0.60, N_LAYERS + 0.55,
        "Nemotron rule:  k = max(1, round(4/31 x L))\n"
        "L = 6  ->  exactly one, at position 3\n"
        "(k = 2 would break the spacing constraints)",
        ha="center", va="bottom", fontsize=8.5, color="#333")
ax.set_xlim(-0.55, 2.75)
ax.set_ylim(-1.3, N_LAYERS + 1.6)
ax.axis("off")

# -------------------------------------------------- panel 2: cost of the swap
ax = fig.add_subplot(gs[0, 1])
ax.set_title("Cost of the swap\n1x GH200, batch 4 x 13k tokens", fontsize=12.5, pad=14)
labels = list(MEASURED)
deltas = [100 * (hi / lo - 1) for lo, hi in MEASURED.values()]
ypos = range(len(labels))
bars = ax.barh(list(ypos), deltas,
               color=[C_ATT if d > 0 else C_GAIN for d in deltas], height=0.42)
ax.axvline(0, color="black", lw=1.1)
for y, b, d, (lo, hi) in zip(ypos, bars, deltas, MEASURED.values()):
    ax.text(d + (0.7 if d > 0 else -0.7), y, f"{d:+.1f}%", va="center",
            ha="left" if d > 0 else "right", fontsize=12, fontweight="bold")
    fmt = (lambda v: f"{v:,.0f}") if lo > 1000 else (lambda v: f"{v:g}")
    ax.text(-10.5, y - 0.30, f"{fmt(lo)}  ->  {fmt(hi)}",
            va="center", ha="left", fontsize=8.5, color="#666")
ax.set_yticks(list(ypos))
ax.set_yticklabels(labels, fontsize=10.5)
ax.set_xlim(-11, 16)
ax.set_xlabel("change vs baseline", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax.text(0.5, -0.30,
        "The memory drop is not a typo. Flash attention is O(L) in memory,\n"
        "cheaper than the per-chunk intermediates of the mamba3 layer it replaced.",
        transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color="#333")

# ------------------------------------------- panel 3: the pre-registered target
ax = fig.add_subplot(gs[0, 2])
ax.set_title("Primary criterion: exact reference resolution (L1)",
             fontsize=12.5, pad=14)
base, oracle, ceiling = 75.3516, 88.1779, 100.0
ax.barh([0], [base], color=C_BASE, height=0.30, label="reached by baseline")
ax.barh([0], [oracle - base], left=base, color=C_GAIN, alpha=0.40,
        height=0.30, label=f"winnable gap: {oracle - base:.2f} pp")
ax.barh([0], [ceiling - oracle], left=oracle, color="#E0E0E0",
        height=0.30, label="target predates the window: unreachable")
for v, txt, col in [(base, f"{base:.2f}%\nbaseline", "#333"),
                    (oracle, f"{oracle:.2f}%\nvisibility oracle", "#2A6B45")]:
    ax.axvline(v, color=col, ls="--", lw=1.3, ymin=0.30, ymax=0.72)
    ax.text(v, 0.20, txt, ha="center", va="bottom", fontsize=9.5, color=col)
ax.annotate("", xy=(oracle, -0.24), xytext=(base, -0.24),
            arrowprops=dict(arrowstyle="<->", color="#2A6B45", lw=2.0))
ax.text((base + oracle) / 2, -0.30, "how much does hybrid close?\n(pre-registered)",
        ha="center", va="top", fontsize=9.5, color="#2A6B45")
ax.set_xlim(60, 102)
ax.set_ylim(-0.75, 0.85)
ax.set_xlabel("cancel/delete resolved to a live order by exact ns timestamp (%)",
              fontsize=9.5)
ax.set_yticks([])
ax.spines[["top", "right", "left"]].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1,
          fontsize=8.5, frameon=False)
ax.text(0.02, 0.99,
        "The key itself is perfect: on visible targets, ns-precision reference\n"
        "time is unique for 27,789 / 27,789 = 100%. So the whole 12.83 pp is the\n"
        "model failing to recall the key, not the encoding failing to provide one.",
        transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color="#333")

path = os.path.join(OUT, "stage1_build_and_cost.png")
fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
print(path)
