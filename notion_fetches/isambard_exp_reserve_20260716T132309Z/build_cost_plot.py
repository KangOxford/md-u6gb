#!/usr/bin/env python3
"""Build the allocation-cost figure used on the Notion reserve-fleet page."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).parent / "assets"
OUTPUT_PATH = OUTPUT_DIR / "u6gb_exp_reserve_cost.png"

DAYS = np.arange(0, 366)
NHR_PER_NODE_DAY = 24
GPU_HOURS_PER_NHR = 4
BASE_NODES = 4
MAX_NODES = 5


def cumulative_nhr(nodes: int) -> np.ndarray:
    return DAYS * nodes * NHR_PER_NODE_DAY


def main() -> None:
    base = cumulative_nhr(BASE_NODES)
    maximum = cumulative_nhr(MAX_NODES)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#3f4854",
            "axes.labelcolor": "#20262e",
            "xtick.color": "#3f4854",
            "ytick.color": "#3f4854",
        }
    )

    fig = plt.figure(figsize=(14, 8), facecolor="#f7f8fa")
    grid = fig.add_gridspec(2, 1, height_ratios=[4.4, 1.25], hspace=0.32)
    ax = fig.add_subplot(grid[0])
    ax.set_facecolor("white")

    base_color = "#167d74"
    spare_color = "#d97706"
    max_color = "#a63f35"

    ax.fill_between(
        DAYS,
        base,
        maximum,
        color=spare_color,
        alpha=0.18,
        label="Cost of 5th node (+25%)",
    )
    ax.plot(DAYS, base, color=base_color, linewidth=3, label="4-node baseline")
    ax.plot(
        DAYS,
        maximum,
        color=max_color,
        linewidth=2.5,
        linestyle="--",
        label="5 nodes running continuously",
    )

    for day in (30, 365):
        ax.axvline(day, color="#a7afb9", linewidth=1, linestyle=":")
        base_value = day * BASE_NODES * NHR_PER_NODE_DAY
        max_value = day * MAX_NODES * NHR_PER_NODE_DAY
        ax.scatter([day, day], [base_value, max_value], s=45, color=[base_color, max_color], zorder=5)
        x_offset = -7 if day == 365 else 7
        horizontal = "right" if day == 365 else "left"
        ax.annotate(
            f"{day}d: {base_value:,} NHR",
            (day, base_value),
            xytext=(x_offset, -18),
            textcoords="offset points",
            ha=horizontal,
            color=base_color,
            fontweight="bold",
        )
        ax.annotate(
            f"{day}d: {max_value:,} NHR",
            (day, max_value),
            xytext=(x_offset, 10),
            textcoords="offset points",
            ha=horizontal,
            color=max_color,
            fontweight="bold",
        )

    ax.set_xlim(0, 365)
    ax.set_ylim(0, 47_000)
    ax.set_xlabel("Continuous operating days")
    ax.set_ylabel("Cumulative allocation cost (NHR)")
    ax.set_title("4-Node Experiment Reserve: Allocation Cost", fontsize=20, loc="left", pad=16)
    ax.text(
        0,
        1.015,
        "Full physical nodes on Isambard-AI; 1 node = 4 GH200 GPUs = 1 NHR per hour",
        transform=ax.transAxes,
        color="#59636f",
        fontsize=11,
    )
    ax.grid(axis="y", color="#dce1e6", linewidth=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncol=3, bbox_to_anchor=(0, 0.94))

    table_ax = fig.add_subplot(grid[1])
    table_ax.axis("off")
    periods = [("1 day", 1), ("30 days", 30), ("365 days", 365)]
    rows = []
    for label, days in periods:
        base_nhr = days * BASE_NODES * NHR_PER_NODE_DAY
        extra_nhr = days * NHR_PER_NODE_DAY
        rows.append(
            [
                label,
                f"{base_nhr:,}",
                f"{base_nhr * GPU_HOURS_PER_NHR:,}",
                f"+{extra_nhr:,}",
                f"{base_nhr + extra_nhr:,}",
            ]
        )

    table = table_ax.table(
        cellText=rows,
        colLabels=["Period", "4-node NHR", "4-node GPU-hours", "5th-node NHR", "5-node NHR"],
        loc="center",
        cellLoc="center",
        colLoc="center",
        bbox=[0, 0.05, 1, 0.9],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.5)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#dce1e6")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#303844")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("white" if row % 2 else "#eef2f4")

    fig.text(
        0.01,
        0.012,
        "Accounting cost only. No public GBP/NHR price is published. Source: Isambard Accounting documentation.",
        color="#68727d",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.9, bottom=0.08)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
