#!/usr/bin/env python3
"""Plot sustainable full-node counts against the award's remaining GPU hours."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT = Path(__file__).parent / "assets" / "sustainable_nodes_to_award_end.png"
OUTPUT_SVG = Path(__file__).parent / "assets" / "sustainable_nodes_to_award_end.svg"
BALANCE_GPU_HOURS = 134_873.73
HOURS = 57 * 24
GPUS_PER_NODE = 4
NODE_COUNTS = np.array([16, 17, 20, 22, 24, 25])
COSTS = NODE_COUNTS * GPUS_PER_NODE * HOURS


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#3e4854",
            "axes.labelcolor": "#20262e",
            "xtick.color": "#3e4854",
            "ytick.color": "#20262e",
        }
    )

    fig, ax = plt.subplots(figsize=(14, 8), facecolor="#f6f7f9")
    ax.set_facecolor("white")
    y = np.arange(len(NODE_COUNTS))
    colors = ["#23867b", "#23867b", "#3478a8", "#3478a8", "#d57b16", "#b8453d"]
    bars = ax.barh(y, COSTS, color=colors, height=0.64)

    ax.axvline(BALANCE_GPU_HOURS, color="#252b33", linewidth=2.5, linestyle="--")
    ax.axvspan(0, BALANCE_GPU_HOURS, color="#23867b", alpha=0.045)
    ax.axvspan(BALANCE_GPU_HOURS, 145_000, color="#b8453d", alpha=0.055)

    for bar, nodes, cost in zip(bars, NODE_COUNTS, COSTS):
        delta = BALANCE_GPU_HOURS - cost
        status = f"{delta:,.0f} left" if delta >= 0 else f"{abs(delta):,.0f} short"
        put_inside = nodes >= 20
        ax.text(
            cost - 1_300 if put_inside else cost + 1_200,
            bar.get_y() + bar.get_height() / 2,
            f"{cost:,.0f}  |  {status}",
            va="center",
            ha="right" if put_inside else "left",
            color="white" if put_inside else "#1f2832",
            fontweight="bold",
            fontsize=11,
        )

    ax.set_yticks(y, [f"{n} nodes  ({n * GPUS_PER_NODE} GPUs)" for n in NODE_COUNTS])
    ax.invert_yaxis()
    ax.set_xlim(0, 147_000)
    ax.set_xlabel("GPU hours consumed from 2026-07-16 through 2026-09-10")
    ax.set_title("Sustainable Full-Node Concurrency Until Award End", fontsize=20, loc="left", pad=18)
    ax.text(
        0,
        1.015,
        "Conservative 57-day window; one Isambard-AI node = 4 GPUs",
        transform=ax.transAxes,
        color="#5d6874",
        fontsize=11,
    )
    ax.grid(axis="x", color="#dce1e6", linewidth=0.9)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.text(
        BALANCE_GPU_HOURS - 1_200,
        -0.48,
        f"Available balance  {BALANCE_GPU_HOURS:,.2f} GPUHr",
        ha="right",
        va="center",
        color="#252b33",
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "#c9d0d8", "boxstyle": "round,pad=0.35"},
    )
    fig.text(
        0.01,
        0.015,
        "Balance used: 150,000 allocation - 15,126.26 used = 134,873.74 GPUHr (portal estimate: 134,873.73).",
        color="#697480",
        fontsize=9.5,
    )
    fig.subplots_adjust(left=0.18, right=0.98, top=0.88, bottom=0.12)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(OUTPUT_SVG, format="svg", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
