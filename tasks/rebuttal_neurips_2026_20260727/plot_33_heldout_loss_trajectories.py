#!/usr/bin/env python3
"""Plot the current 33-run Jan-2026 held-out loss trajectories.

The source is the canonical held-out table produced by the rebuttal analysis.
The composite combines an all-run overview with a 12-size x 3-seed contact
sheet. It also writes one standalone PNG per logical run and CSV/JSON audit
artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "path.simplify": True,
        "path.simplify_threshold": 0.5,
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
    }
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


DEFAULT_RESULT_ROOT = Path(
    "/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/"
    "scaling_law_plots/aramis/results"
)
SIZE_ORDER = [
    "0p2M",
    "1M",
    "4M",
    "6M",
    "10M",
    "14M",
    "23M",
    "46M",
    "78M",
    "120M",
    "200M",
    "350M",
]
SEED_ORDER = [5, 42, 137]
SEED_STYLE = {
    5: {"marker": "o", "linestyle": "-"},
    42: {"marker": "s", "linestyle": "--"},
    137: {"marker": "^", "linestyle": ":"},
}
EXPECTED_MISSING = {
    ("23M", 42): "incomplete; excluded",
    ("200M", 137): "no completed run",
    ("350M", 137): "no completed run",
}
INCOMPLETE_TARGET_RUN = "46M-s5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def human_params(value: int) -> str:
    millions = value / 1e6
    if millions < 10:
        return f"{millions:.1f}M"
    return f"{millions:.0f}M"


def load_and_validate(
    canonical_path: Path, endpoint_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    points = pd.read_csv(canonical_path)
    endpoints = pd.read_csv(endpoint_path)
    required = {
        "label",
        "seed",
        "run_id",
        "source_jid",
        "step",
        "target_step",
        "N",
        "D",
        "L",
        "n_tickers",
        "source",
    }
    for name, frame in [("canonical", points), ("endpoint", endpoints)]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} input is missing columns: {sorted(missing)}")

    for frame in (points, endpoints):
        frame["seed"] = frame["seed"].astype(int)
        frame["step"] = frame["step"].astype(int)
        frame["target_step"] = frame["target_step"].astype(int)
        frame["N"] = frame["N"].astype(int)

    if len(points) != 285:
        raise ValueError(f"expected 285 held-out checkpoints, found {len(points)}")
    if points["run_id"].nunique() != 33:
        raise ValueError(
            f"expected 33 logical runs, found {points['run_id'].nunique()}"
        )
    if points["label"].nunique() != 12:
        raise ValueError(f"expected 12 sizes, found {points['label'].nunique()}")
    if set(points["n_tickers"].unique()) != {487}:
        raise ValueError(
            f"expected uniform 487-ticker coverage, found {points['n_tickers'].unique()}"
        )
    if len(endpoints) != 33 or endpoints["run_id"].nunique() != 33:
        raise ValueError("endpoint table must contain one row for each of 33 runs")
    if set(points["run_id"]) != set(endpoints["run_id"]):
        raise ValueError("canonical and endpoint run cohorts differ")
    if points.duplicated(["run_id", "step"]).any():
        raise ValueError("canonical input contains duplicate logical checkpoints")
    if set(points["label"]) != set(SIZE_ORDER):
        raise ValueError("canonical size labels differ from the locked display order")

    expected_grid = {(label, seed) for label in SIZE_ORDER for seed in SEED_ORDER}
    observed_grid = set(zip(points["label"], points["seed"], strict=True))
    missing_grid = expected_grid - observed_grid
    extra_grid = observed_grid - expected_grid
    if missing_grid != set(EXPECTED_MISSING) or extra_grid:
        raise ValueError(
            f"unexpected seed grid: missing={sorted(missing_grid)}, extra={sorted(extra_grid)}"
        )

    endpoint_lookup = endpoints.set_index("run_id")
    for run_id, group in points.groupby("run_id", sort=False):
        last = group.sort_values(["step", "source_jid"]).iloc[-1]
        endpoint = endpoint_lookup.loc[run_id]
        if int(last["step"]) != int(endpoint["step"]):
            raise ValueError(f"{run_id}: endpoint step is not final available step")
        if not np.isclose(float(last["D"]), float(endpoint["D"]), rtol=0, atol=0):
            raise ValueError(f"{run_id}: endpoint D differs from canonical final D")
        if not np.isclose(float(last["L"]), float(endpoint["L"]), rtol=0, atol=1e-12):
            raise ValueError(f"{run_id}: endpoint L differs from canonical final L")

    reached = endpoints.assign(
        reached_target=endpoints["step"] >= endpoints["target_step"]
    )
    not_reached = set(reached.loc[~reached["reached_target"], "run_id"])
    if not_reached != {INCOMPLETE_TARGET_RUN}:
        raise ValueError(f"unexpected target-reaching exceptions: {sorted(not_reached)}")

    order = {label: index for index, label in enumerate(SIZE_ORDER)}
    points["size_order"] = points["label"].map(order)
    endpoints["size_order"] = endpoints["label"].map(order)
    points = points.sort_values(
        ["size_order", "seed", "step", "source_jid"], kind="stable"
    ).reset_index(drop=True)
    endpoints = endpoints.sort_values(
        ["size_order", "seed", "step", "source_jid"], kind="stable"
    ).reset_index(drop=True)
    return points, endpoints


def build_summary(
    points: pd.DataFrame, endpoints: pd.DataFrame
) -> pd.DataFrame:
    grouped = (
        points.groupby(
            ["size_order", "label", "seed", "run_id"], sort=False, observed=True
        )
        .agg(
            source_jids=("source_jid", lambda values: ";".join(map(str, pd.unique(values)))),
            n_checkpoints=("step", "size"),
            first_step=("step", "min"),
            final_step=("step", "max"),
            target_step=("target_step", "first"),
            N=("N", "first"),
            D_first=("D", "min"),
            D_final=("D", "max"),
            L_first=("L", "first"),
            L_min=("L", "min"),
            L_final=("L", "last"),
            n_tickers=("n_tickers", "first"),
            source=("source", "first"),
        )
        .reset_index()
    )
    endpoint_values = endpoints[
        ["run_id", "step", "D", "L", "protocol"]
    ].rename(
        columns={
            "step": "endpoint_step",
            "D": "endpoint_D",
            "L": "endpoint_L",
            "protocol": "endpoint_protocol",
        }
    )
    summary = grouped.merge(
        endpoint_values, on="run_id", how="left", validate="one_to_one"
    )
    summary["reached_target"] = summary["final_step"] >= summary["target_step"]
    summary["D_first_B"] = summary["D_first"] / 1e9
    summary["D_final_B"] = summary["D_final"] / 1e9
    summary["endpoint_D_B"] = summary["endpoint_D"] / 1e9
    summary["standalone_plot"] = summary.apply(
        lambda row: (
            f"runs/{int(row['size_order']) + 1:02d}_{row['run_id']}.png"
        ),
        axis=1,
    )
    return summary.sort_values(["size_order", "seed"], kind="stable").reset_index(
        drop=True
    )


def row_limits(frame: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    x_min = float(frame["D"].min() / 1e9)
    x_max = float(frame["D"].max() / 1e9)
    x_low = x_min / 1.16
    x_high = x_max * 1.16
    y_min = float(frame["L"].min())
    y_max = float(frame["L"].max())
    y_pad = max(0.025, (y_max - y_min) * 0.12)
    return (x_low, x_high), (max(0.0, y_min - y_pad), y_max + y_pad)


def plot_curve(
    ax: plt.Axes,
    frame: pd.DataFrame,
    endpoint: pd.Series,
    color: tuple[float, float, float, float],
    *,
    compact: bool,
) -> None:
    seed = int(endpoint["seed"])
    style = SEED_STYLE[seed]
    x = frame["D"].to_numpy(dtype=float) / 1e9
    y = frame["L"].to_numpy(dtype=float)
    ax.plot(
        x,
        y,
        color=color,
        linestyle=style["linestyle"],
        marker=style["marker"],
        markersize=3.5 if compact else 5,
        linewidth=1.45 if compact else 1.9,
        alpha=0.92,
        markeredgecolor="white",
        markeredgewidth=0.35,
    )

    final_x = float(endpoint["D"]) / 1e9
    final_y = float(endpoint["L"])
    reached_target = int(endpoint["step"]) >= int(endpoint["target_step"])
    if reached_target:
        ax.scatter(
            [final_x],
            [final_y],
            s=42 if compact else 62,
            marker=style["marker"],
            facecolor=color,
            edgecolor="black",
            linewidth=1.0,
            zorder=5,
        )
    else:
        ax.scatter(
            [final_x],
            [final_y],
            s=58 if compact else 78,
            marker="o",
            facecolor="white",
            edgecolor="#c62828",
            linewidth=2.0,
            zorder=6,
        )
        tokens_per_step = final_x / int(endpoint["step"])
        target_x = tokens_per_step * int(endpoint["target_step"])
        ax.axvline(target_x, color="#c62828", linestyle="--", linewidth=1.1, alpha=0.8)
        ax.annotate(
            "final available\n53,970 / 63,407",
            xy=(final_x, final_y),
            xytext=(-3, 11),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=6.5 if compact else 8,
            color="#a11b1b",
            fontweight="bold",
        )


def make_standalone_plots(
    points: pd.DataFrame,
    endpoints: pd.DataFrame,
    colors: dict[str, tuple[float, float, float, float]],
    output_dir: Path,
) -> list[Path]:
    run_dir = output_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    endpoint_lookup = endpoints.set_index("run_id")
    created: list[Path] = []
    for index, (run_id, frame) in enumerate(
        points.groupby("run_id", sort=False), start=1
    ):
        endpoint = endpoint_lookup.loc[run_id]
        label = str(endpoint["label"])
        seed = int(endpoint["seed"])
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        plot_curve(ax, frame, endpoint, colors[label], compact=False)
        x_limits, y_limits = row_limits(frame)
        if run_id == INCOMPLETE_TARGET_RUN:
            target_x = (
                float(endpoint["D"])
                / 1e9
                / int(endpoint["step"])
                * int(endpoint["target_step"])
            )
            x_limits = (x_limits[0], max(x_limits[1], target_x * 1.08))
        ax.set_xscale("log")
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.grid(True, which="major", color="#d9d9d9", linewidth=0.6, alpha=0.8)
        ax.set_xlabel("Cumulative tokens D (billions; log scale)")
        ax.set_ylabel("Jan-2026 held-out CE (487-ticker macro mean)")
        status = (
            "target reached"
            if int(endpoint["step"]) >= int(endpoint["target_step"])
            else "final available; target not reached"
        )
        fig.suptitle(
            f"{run_id}  |  N≈{human_params(int(endpoint['N']))}",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_title(
            f"{len(frame)} evaluated checkpoints  |  endpoint L={float(endpoint['L']):.6f}"
            f"  |  {status}",
            fontsize=9,
        )
        fig.text(
            0.01,
            0.01,
            "Source: aramis/results/canonical_test.csv; terminal marker = final available evaluation.",
            fontsize=7,
            color="#555555",
        )
        fig.tight_layout(rect=(0, 0.035, 1, 0.94))
        path = run_dir / f"{index:02d}_{run_id}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        created.append(path)
    return created


def make_composite(
    points: pd.DataFrame,
    endpoints: pd.DataFrame,
    colors: dict[str, tuple[float, float, float, float]],
    output_dir: Path,
    canonical_sha: str,
    endpoint_sha: str,
) -> list[Path]:
    endpoint_lookup = endpoints.set_index(["label", "seed"])
    fig = plt.figure(figsize=(18, 38))
    grid = fig.add_gridspec(
        13,
        3,
        height_ratios=[3.8] + [1.0] * 12,
        hspace=0.62,
        wspace=0.22,
    )

    overview = fig.add_subplot(grid[0, :])
    for run_id, frame in points.groupby("run_id", sort=False):
        endpoint = endpoints.loc[endpoints["run_id"] == run_id].iloc[0]
        plot_curve(
            overview,
            frame,
            endpoint,
            colors[str(endpoint["label"])],
            compact=True,
        )
    overview.set_xscale("log")
    overview.set_xlim(
        float(points["D"].min() / 1e9) / 1.15,
        float(points["D"].max() / 1e9) * 1.15,
    )
    overview.set_ylim(0.50, float(points["L"].max()) + 0.12)
    overview.grid(True, which="major", color="#d7d7d7", linewidth=0.65, alpha=0.8)
    overview.set_xlabel("Cumulative tokens D (billions; log scale)", fontsize=11)
    overview.set_ylabel("Jan-2026 held-out CE\n(487-ticker macro mean)", fontsize=11)
    overview.set_title(
        "All 33 seed-specific logical runs (285 evaluated checkpoints)",
        loc="left",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )

    size_handles = [
        Patch(facecolor=colors[label], edgecolor="none", label=label)
        for label in SIZE_ORDER
    ]
    seed_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            marker=SEED_STYLE[seed]["marker"],
            linestyle=SEED_STYLE[seed]["linestyle"],
            linewidth=1.5,
            markersize=5,
            label=f"seed {seed}",
        )
        for seed in SEED_ORDER
    ]
    legend_sizes = overview.legend(
        handles=size_handles,
        title="Nominal size (color)",
        ncol=6,
        loc="upper right",
        frameon=True,
        fontsize=8,
        title_fontsize=8,
    )
    overview.add_artist(legend_sizes)
    overview.legend(
        handles=seed_handles,
        title="Seed ID (style)",
        ncol=3,
        loc="center right",
        bbox_to_anchor=(1.0, 0.68),
        frameon=True,
        fontsize=8,
        title_fontsize=8,
    )
    overview.text(
        0.012,
        0.035,
        "Filled endpoint = target reached. Red hollow endpoint = final available checkpoint "
        "before intended target (46M-s5 only).",
        transform=overview.transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
    )

    for row_index, label in enumerate(SIZE_ORDER):
        size_points = points.loc[points["label"] == label]
        x_limits, y_limits = row_limits(size_points)
        n_value = int(size_points["N"].iloc[0])
        for column_index, seed in enumerate(SEED_ORDER):
            ax = fig.add_subplot(grid[row_index + 1, column_index])
            key = (label, seed)
            if key in EXPECTED_MISSING:
                ax.set_facecolor("#f2f2f2")
                ax.text(
                    0.5,
                    0.54,
                    f"{label}-s{seed}",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="#666666",
                )
                ax.text(
                    0.5,
                    0.39,
                    EXPECTED_MISSING[key],
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#8b0000",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color("#c8c8c8")
                if column_index == 0:
                    ax.set_ylabel(
                        f"{label}\nN≈{human_params(n_value)}",
                        rotation=0,
                        ha="right",
                        va="center",
                        labelpad=38,
                        fontsize=9,
                        fontweight="bold",
                    )
                continue

            frame = size_points.loc[size_points["seed"] == seed]
            endpoint = endpoint_lookup.loc[key]
            plot_curve(ax, frame, endpoint, colors[label], compact=True)
            if str(endpoint["run_id"]) == INCOMPLETE_TARGET_RUN:
                target_x = (
                    float(endpoint["D"])
                    / 1e9
                    / int(endpoint["step"])
                    * int(endpoint["target_step"])
                )
                x_limits = (x_limits[0], max(x_limits[1], target_x * 1.08))
                ax.set_facecolor("#fff4f2")
            ax.set_xscale("log")
            ax.set_xlim(*x_limits)
            ax.set_ylim(*y_limits)
            ax.grid(True, which="major", color="#dddddd", linewidth=0.45, alpha=0.8)
            ax.yaxis.set_major_locator(MaxNLocator(3))
            ax.tick_params(axis="both", labelsize=7)
            ax.set_title(
                f"seed {seed}  |  {len(frame)} ckpts  |  final L={float(endpoint['L']):.4f}",
                fontsize=8,
                fontweight="bold",
                color="#a11b1b"
                if str(endpoint["run_id"]) == INCOMPLETE_TARGET_RUN
                else "#222222",
            )
            if column_index == 0:
                ax.set_ylabel(
                    f"{label}\nN≈{human_params(n_value)}",
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=38,
                    fontsize=9,
                    fontweight="bold",
                )
            else:
                ax.set_yticklabels([])
            if row_index == len(SIZE_ORDER) - 1:
                ax.set_xlabel("D (B tokens; log)", fontsize=8)

    fig.suptitle(
        "Current 33-run cohort: Jan-2026 held-out loss trajectories",
        x=0.5,
        y=0.996,
        fontsize=20,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.986,
        "One composite: all-run overview + 12 nominal sizes × 3 actual seed IDs "
        "(33 populated cells; 3 absent from the completed cohort)",
        ha="center",
        fontsize=11,
        color="#333333",
    )
    fig.text(
        0.01,
        0.004,
        "Primary outcome: forward-time Jan-2026 CE, equal macro-average over 487 tickers. "
        "Each line connects only checkpoints from one logical (size, seed) run. "
        f"Inputs: canonical_test.csv sha256={canonical_sha[:12]}…; "
        f"selected_test_endpoint.csv sha256={endpoint_sha[:12]}….",
        fontsize=7.5,
        color="#555555",
    )

    stem = output_dir / "heldout_loss_trajectories_33run_composite"
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    svg_path = stem.with_suffix(".svg")
    fig.savefig(png_path, dpi=190, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path, svg_path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical-test",
        type=Path,
        default=DEFAULT_RESULT_ROOT / "canonical_test.csv",
    )
    parser.add_argument(
        "--endpoint",
        type=Path,
        default=DEFAULT_RESULT_ROOT / "selected_test_endpoint.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "artifacts"
        / "heldout_loss_trajectories_33run",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    points, endpoints = load_and_validate(args.canonical_test, args.endpoint)
    summary = build_summary(points, endpoints)
    colors = {
        label: mpl.colormaps["viridis"](position)
        for label, position in zip(
            SIZE_ORDER, np.linspace(0.06, 0.94, len(SIZE_ORDER)), strict=True
        )
    }

    points_out = args.output_dir / "heldout_loss_trajectories_33run_points.csv"
    summary_out = args.output_dir / "heldout_loss_trajectories_33run_summary.csv"
    points.drop(columns=["size_order"]).assign(D_B=points["D"] / 1e9).to_csv(
        points_out, index=False
    )
    summary.drop(columns=["size_order"]).to_csv(summary_out, index=False)

    canonical_sha = sha256(args.canonical_test)
    endpoint_sha = sha256(args.endpoint)
    standalone = make_standalone_plots(points, endpoints, colors, args.output_dir)
    composite = make_composite(
        points,
        endpoints,
        colors,
        args.output_dir,
        canonical_sha,
        endpoint_sha,
    )

    artifact_paths = [points_out, summary_out, *standalone, *composite]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "figure_title": "Current 33-run cohort: Jan-2026 held-out loss trajectories",
        "protocol": {
            "outcome": "Jan-2026 held-out CE",
            "aggregation": "equal macro-average over 487 tickers",
            "x_axis": "cumulative tokens D",
            "cohort": "manifest-selected current 33 logical runs",
            "terminal_marker": "final available held-out evaluation",
        },
        "audit": {
            "n_sizes": int(points["label"].nunique()),
            "n_runs": int(points["run_id"].nunique()),
            "n_checkpoints": int(len(points)),
            "points_per_run_min": int(summary["n_checkpoints"].min()),
            "points_per_run_max": int(summary["n_checkpoints"].max()),
            "points_per_run_median": float(summary["n_checkpoints"].median()),
            "n_target_reaching_runs": int(summary["reached_target"].sum()),
            "non_target_reaching_runs": summary.loc[
                ~summary["reached_target"], "run_id"
            ].tolist(),
            "missing_size_seed_cells": [
                {"label": label, "seed": seed, "reason": reason}
                for (label, seed), reason in EXPECTED_MISSING.items()
            ],
        },
        "inputs": {
            str(args.canonical_test): {
                "sha256": canonical_sha,
                "bytes": args.canonical_test.stat().st_size,
            },
            str(args.endpoint): {
                "sha256": endpoint_sha,
                "bytes": args.endpoint.stat().st_size,
            },
        },
        "artifacts": {
            str(path.relative_to(args.output_dir)): {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in artifact_paths
        },
    }
    manifest_path = args.output_dir / "heldout_loss_trajectories_33run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "n_sizes": manifest["audit"]["n_sizes"],
                "n_runs": manifest["audit"]["n_runs"],
                "n_checkpoints": manifest["audit"]["n_checkpoints"],
                "n_target_reaching_runs": manifest["audit"]["n_target_reaching_runs"],
                "standalone_plots": len(standalone),
                "composite": [str(path) for path in composite],
                "svg_bytes": composite[-1].stat().st_size,
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
