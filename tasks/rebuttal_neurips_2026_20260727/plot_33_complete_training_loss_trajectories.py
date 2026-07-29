#!/usr/bin/env python3
"""Plot complete recoverable training-loss job histories for the 33-run cohort.

Unlike the sparse held-out checkpoint plot, this diagnostic uses W&B step_loss
history so that a failed job and every observed resume segment can appear in
the same size-by-seed subplot. Physical jobs are never joined by a solid loss
line. Zero-data attempts are retained in an attempt ribbon.

Two 6M chains have cross-seed checkpoint ancestry. They are explicitly
reassigned to the ancestry of the final nominal run and labelled as seed
switches instead of being presented as seed-pure trajectories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 8.5,
        "path.simplify": True,
        "path.simplify_threshold": 0.35,
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
    }
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter


DEFAULT_ROOT = Path(
    "/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/"
    "scaling_law_plots"
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
EXPECTED_MISSING = {
    ("23M", 42): "incomplete; excluded",
    ("200M", 137): "no completed run",
    ("350M", 137): "no completed run",
}
EXPECTED_RUNS = 33
EXPECTED_SEGMENTS = 54
EXPECTED_POINTS = 10_403
INCOMPLETE_RUN = "46M-s5"
INCOMPLETE_RELAY = "j4514373"
CROSS_SEED = {
    "6M-s5": {
        "origin_seed": 137,
        "final_seed": 5,
        "predecessor": "j4499369",
        "relay": "j4507467",
        "final": "j4508786",
    },
    "6M-s137": {
        "origin_seed": 5,
        "final_seed": 137,
        "predecessor": "j4499362",
        "relay": "j4507453",
        "final": "j4509365",
    },
}
EXCLUDED_LONG_D_JIDS = {"j4501060", "j4506444"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_label(value: object) -> str:
    label = str(value)
    return {
        "0.2M": "0p2M",
        "50M": "46M",
        "300M": "350M",
    }.get(label, label)


def normalized_jid(value: object) -> str:
    raw = re.sub(r"\.0$", "", str(value).strip())
    if not raw or raw.lower() == "nan":
        return ""
    return raw if raw.startswith("j") else f"j{raw}"


def unique_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def parse_attempts(value: object) -> list[str]:
    return unique_preserving_order(
        normalized_jid(item) for item in str(value).split(",")
    )


def select_wandb_history_per_jid(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep the W&B run reaching the largest step for each physical JID."""
    if raw.empty:
        return raw.copy()
    candidates = (
        raw.groupby(["jid", "wandb_run_id"], dropna=False, observed=True)
        .agg(max_step=("global_step", "max"), n_rows=("global_step", "size"))
        .reset_index()
        .sort_values(
            ["jid", "max_step", "n_rows", "wandb_run_id"], kind="stable"
        )
        .drop_duplicates("jid", keep="last")
    )
    selected = raw.merge(
        candidates[["jid", "wandb_run_id"]],
        on=["jid", "wandb_run_id"],
        how="inner",
        validate="many_to_one",
    )
    return (
        selected.sort_values(
            ["jid", "wandb_run_id", "global_step"], kind="stable"
        )
        .drop_duplicates(
            ["jid", "wandb_run_id", "global_step"], keep="last"
        )
        .reset_index(drop=True)
    )


def build_cohort(
    raw_path: Path, manifest_path: Path, endpoint_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(raw_path, low_memory=False)
    manifest = pd.read_csv(manifest_path, low_memory=False)
    endpoints = pd.read_csv(endpoint_path, low_memory=False)

    raw_required = {
        "jid",
        "wandb_run_id",
        "wandb_state",
        "global_step",
        "step_loss",
    }
    manifest_required = {
        "arch",
        "ablation_type",
        "label",
        "seed",
        "curtail_epochs",
        "reached_curtail",
        "all_attempts_jids",
        "n_params_M",
    }
    endpoint_required = {"label", "seed", "run_id", "N"}
    for name, frame, required in (
        ("raw loss", raw, raw_required),
        ("manifest", manifest, manifest_required),
        ("endpoint", endpoints, endpoint_required),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} input missing columns: {sorted(missing)}")

    raw = raw.copy()
    raw["jid"] = raw["jid"].map(normalized_jid)
    raw["global_step"] = pd.to_numeric(raw["global_step"], errors="coerce")
    raw["step_loss"] = pd.to_numeric(raw["step_loss"], errors="coerce")
    raw = raw.dropna(subset=["global_step", "step_loss", "wandb_run_id"])
    raw["global_step"] = raw["global_step"].astype(int)

    production = manifest[
        (manifest["arch"] == "Mamba3")
        & (manifest["ablation_type"] == "production")
    ].copy()
    production["label_norm"] = production["label"].map(normalize_label)
    production["seed_int"] = pd.to_numeric(
        production["seed"], errors="raise"
    ).astype(int)
    if production.duplicated(["label_norm", "seed_int"]).any():
        dup = production.loc[
            production.duplicated(["label_norm", "seed_int"], keep=False),
            ["label_norm", "seed_int"],
        ]
        raise ValueError(f"duplicate production manifest rows:\n{dup}")

    endpoints = endpoints.copy()
    endpoints["label"] = endpoints["label"].map(normalize_label)
    endpoints["seed"] = pd.to_numeric(
        endpoints["seed"], errors="raise"
    ).astype(int)
    endpoints["run_id"] = (
        endpoints["label"].astype(str)
        + "-s"
        + endpoints["seed"].astype(str)
    )
    if len(endpoints) != EXPECTED_RUNS or endpoints["run_id"].nunique() != EXPECTED_RUNS:
        raise ValueError(
            f"expected {EXPECTED_RUNS} endpoint rows, found {len(endpoints)}"
        )

    endpoint_pairs = set(zip(endpoints["label"], endpoints["seed"], strict=True))
    expected_grid = {
        (label, seed) for label in SIZE_ORDER for seed in SEED_ORDER
    }
    if expected_grid - endpoint_pairs != set(EXPECTED_MISSING):
        raise ValueError("endpoint size-seed grid differs from the frozen 33-run cohort")

    manifest_lookup = production.set_index(["label_norm", "seed_int"])
    run_specs: dict[str, dict] = {}
    for endpoint in endpoints.itertuples(index=False):
        key = (str(endpoint.label), int(endpoint.seed))
        if key not in manifest_lookup.index:
            raise ValueError(f"missing production manifest row for {key}")
        row = manifest_lookup.loc[key]
        run_id = f"{key[0]}-s{key[1]}"
        run_specs[run_id] = {
            "run_id": run_id,
            "label": key[0],
            "seed": key[1],
            "N": int(endpoint.N),
            "target_step": int(float(row["curtail_epochs"])),
            "manifest_reached_curtail": bool(int(row["reached_curtail"])),
            "attempts": parse_attempts(row["all_attempts_jids"]),
            "manifest_label": str(row["label"]),
        }

    # The final 6M job IDs were recorded under the opposite nominal seed row.
    # Reassign the full attempt lists to the actual checkpoint ancestry.
    original_s5 = list(run_specs["6M-s5"]["attempts"])
    original_s137 = list(run_specs["6M-s137"]["attempts"])
    run_specs["6M-s5"]["attempts"] = unique_preserving_order(
        [jid for jid in original_s137 if jid != "j4509365"] + ["j4508786"]
    )
    run_specs["6M-s137"]["attempts"] = unique_preserving_order(
        [jid for jid in original_s5 if jid != "j4508786"] + ["j4509365"]
    )

    point_frames: list[pd.DataFrame] = []
    attempt_rows: list[dict] = []
    segment_rows: list[dict] = []
    summary_rows: list[dict] = []
    size_order = {label: index for index, label in enumerate(SIZE_ORDER)}

    for run_id in sorted(
        run_specs,
        key=lambda rid: (
            size_order[run_specs[rid]["label"]],
            run_specs[rid]["seed"],
        ),
    ):
        spec = run_specs[run_id]
        attempts = spec["attempts"]
        selected = select_wandb_history_per_jid(
            raw[raw["jid"].isin(attempts)].copy()
        )
        observed_jids = [
            jid for jid in attempts if jid in set(selected["jid"])
        ]
        if not observed_jids:
            raise ValueError(f"{run_id} has no observable W&B loss history")
        final_observed_jid = observed_jids[-1]
        attempt_rank = {jid: rank for rank, jid in enumerate(attempts)}
        observed_rank = {jid: rank for rank, jid in enumerate(observed_jids)}

        for jid in attempts:
            jid_frame = selected[selected["jid"] == jid]
            attempt_rows.append(
                {
                    "size_order": size_order[spec["label"]],
                    "label": spec["label"],
                    "seed": spec["seed"],
                    "run_id": run_id,
                    "attempt_rank": attempt_rank[jid],
                    "jid": jid,
                    "has_loss_data": not jid_frame.empty,
                    "wandb_run_id": (
                        str(jid_frame["wandb_run_id"].iloc[0])
                        if not jid_frame.empty
                        else ""
                    ),
                    "wandb_state": (
                        str(jid_frame["wandb_state"].iloc[0])
                        if not jid_frame.empty
                        else ""
                    ),
                    "n_points": int(len(jid_frame)),
                    "step_min": (
                        int(jid_frame["global_step"].min())
                        if not jid_frame.empty
                        else np.nan
                    ),
                    "step_max": (
                        int(jid_frame["global_step"].max())
                        if not jid_frame.empty
                        else np.nan
                    ),
                }
            )

        for jid in observed_jids:
            segment = selected[selected["jid"] == jid].copy()
            state = str(segment["wandb_state"].iloc[0]).lower()
            is_final = jid == final_observed_jid
            if run_id == INCOMPLETE_RUN and is_final:
                role = "timeout_final"
            elif is_final:
                role = "final"
            elif state == "crashed":
                role = "crashed_pre_resume"
            else:
                role = "interrupted_pre_resume"
            segment["size_order"] = size_order[spec["label"]]
            segment["label"] = spec["label"]
            segment["nominal_seed"] = spec["seed"]
            segment["run_id"] = run_id
            segment["target_step"] = spec["target_step"]
            segment["attempt_rank"] = attempt_rank[jid]
            segment["observed_segment_rank"] = observed_rank[jid]
            segment["segment_role"] = role
            segment["is_final_observed_segment"] = is_final
            segment["progress"] = (
                segment["global_step"] / float(spec["target_step"])
            )
            segment["cross_seed_restore"] = run_id in CROSS_SEED
            point_frames.append(segment)
            segment_rows.append(
                {
                    "size_order": size_order[spec["label"]],
                    "label": spec["label"],
                    "seed": spec["seed"],
                    "run_id": run_id,
                    "observed_segment_rank": observed_rank[jid],
                    "attempt_rank": attempt_rank[jid],
                    "jid": jid,
                    "wandb_run_id": str(segment["wandb_run_id"].iloc[0]),
                    "wandb_state": str(segment["wandb_state"].iloc[0]),
                    "segment_role": role,
                    "n_points": int(len(segment)),
                    "step_min": int(segment["global_step"].min()),
                    "step_max": int(segment["global_step"].max()),
                    "loss_first": float(segment["step_loss"].iloc[0]),
                    "loss_last": float(segment["step_loss"].iloc[-1]),
                    "target_step": spec["target_step"],
                    "cross_seed_restore": run_id in CROSS_SEED,
                }
            )

        selected_run = selected[selected["jid"].isin(observed_jids)]
        actual_reached = run_id != INCOMPLETE_RUN
        cross = CROSS_SEED.get(run_id)
        summary_rows.append(
            {
                "size_order": size_order[spec["label"]],
                "label": spec["label"],
                "seed": spec["seed"],
                "run_id": run_id,
                "N": spec["N"],
                "target_step": spec["target_step"],
                "final_logged_step": int(selected_run["global_step"].max()),
                "n_loss_points": int(len(selected_run)),
                "n_observed_segments": len(observed_jids),
                "n_manifest_attempts": len(attempts),
                "n_zero_data_attempts": len(attempts) - len(observed_jids),
                "n_crashed_segments": int(
                    sum(
                        str(
                            selected.loc[
                                selected["jid"] == jid, "wandb_state"
                            ].iloc[0]
                        ).lower()
                        == "crashed"
                        for jid in observed_jids
                    )
                ),
                "observed_jids": ";".join(observed_jids),
                "all_attempt_jids": ";".join(attempts),
                "reached_target": actual_reached,
                "status_note": (
                    "timeout at step 53970; resume attempt j4514373 has no loss rows"
                    if run_id == INCOMPLETE_RUN
                    else (
                        f"cross-seed restore {cross['origin_seed']}→{cross['final_seed']}"
                        if cross
                        else ""
                    )
                ),
                "cross_seed_origin": cross["origin_seed"] if cross else np.nan,
                "cross_seed_final": cross["final_seed"] if cross else np.nan,
            }
        )

    points = pd.concat(point_frames, ignore_index=True)
    points = points.sort_values(
        [
            "size_order",
            "nominal_seed",
            "observed_segment_rank",
            "global_step",
        ],
        kind="stable",
    ).reset_index(drop=True)
    attempts = pd.DataFrame(attempt_rows).sort_values(
        ["size_order", "seed", "attempt_rank"], kind="stable"
    )
    segments = pd.DataFrame(segment_rows).sort_values(
        ["size_order", "seed", "observed_segment_rank"], kind="stable"
    )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["size_order", "seed"], kind="stable"
    )

    validate_cohort(points, attempts, segments, summary)
    return points, attempts, segments, summary


def validate_cohort(
    points: pd.DataFrame,
    attempts: pd.DataFrame,
    segments: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    if summary["run_id"].nunique() != EXPECTED_RUNS:
        raise ValueError(
            f"expected {EXPECTED_RUNS} logical runs, found {summary['run_id'].nunique()}"
        )
    if len(segments) != EXPECTED_SEGMENTS:
        raise ValueError(
            f"expected {EXPECTED_SEGMENTS} physical loss segments, found {len(segments)}"
        )
    if len(points) != EXPECTED_POINTS:
        raise ValueError(
            f"expected {EXPECTED_POINTS} selected raw loss points, found {len(points)}"
        )
    if points.duplicated(
        ["run_id", "jid", "wandb_run_id", "global_step"]
    ).any():
        raise ValueError("duplicate physical loss points remain after selection")
    if set(points["jid"]) & EXCLUDED_LONG_D_JIDS:
        raise ValueError("the unrelated 6M long-D chain entered the production plot")
    selected_wandb = (
        segments.set_index("jid")["wandb_run_id"].astype(str).to_dict()
    )
    if selected_wandb.get("j4499378") != "9o13kj41":
        raise ValueError("10M-s137 selected the wrong same-JID W&B predecessor")
    if selected_wandb.get("j4508677") != "28bcfloa":
        raise ValueError("46M-s42 selected the wrong same-JID W&B predecessor")
    for run_id, cross in CROSS_SEED.items():
        observed = segments.loc[
            segments["run_id"] == run_id, "jid"
        ].tolist()
        if observed != [cross["predecessor"], cross["final"]]:
            raise ValueError(
                f"{run_id} does not follow the audited cross-seed ancestry: {observed}"
            )
        relay = attempts[
            (attempts["run_id"] == run_id)
            & (attempts["jid"] == cross["relay"])
        ]
        if len(relay) != 1 or bool(relay.iloc[0]["has_loss_data"]):
            raise ValueError(f"{run_id} relay audit differs from expectation")
    incomplete = summary.set_index("run_id").loc[INCOMPLETE_RUN]
    if int(incomplete["final_logged_step"]) != 53_970:
        raise ValueError("46M-s5 final logged step differs from audited timeout")
    relay = attempts[
        (attempts["run_id"] == INCOMPLETE_RUN)
        & (attempts["jid"] == INCOMPLETE_RELAY)
    ]
    if len(relay) != 1 or bool(relay.iloc[0]["has_loss_data"]):
        raise ValueError("46M-s5 zero-data resume relay is missing")
    if set(summary.loc[~summary["reached_target"], "run_id"]) != {
        INCOMPLETE_RUN
    }:
        raise ValueError("unexpected non-target-reaching run set")


def rolling_median(frame: pd.DataFrame) -> np.ndarray:
    window = min(11, max(3, (len(frame) // 25) * 2 + 1))
    return (
        frame["step_loss"]
        .rolling(window=window, min_periods=1, center=True)
        .median()
        .to_numpy()
    )


def segment_style(role: str, base_color: object) -> dict:
    if role == "crashed_pre_resume":
        return {"color": "#d95f02", "linestyle": "--", "linewidth": 1.15}
    if role == "interrupted_pre_resume":
        return {"color": "#b8860b", "linestyle": "--", "linewidth": 1.15}
    if role == "timeout_final":
        return {"color": "#c62828", "linestyle": "-", "linewidth": 1.35}
    return {"color": base_color, "linestyle": "-", "linewidth": 1.35}


def plot_run(
    ax: plt.Axes,
    run_points: pd.DataFrame,
    run_attempts: pd.DataFrame,
    run_summary: pd.Series,
    base_color: object,
    *,
    compact: bool,
) -> None:
    run_id = str(run_summary.get("run_id", run_summary.name))
    grouped = list(
        run_points.groupby("observed_segment_rank", sort=True, observed=True)
    )
    previous: tuple[float, float, str] | None = None
    for _, segment in grouped:
        segment = segment.sort_values("global_step", kind="stable")
        role = str(segment["segment_role"].iloc[0])
        style = segment_style(role, base_color)
        smooth = rolling_median(segment)
        ax.plot(
            segment["global_step"],
            segment["step_loss"],
            color=style["color"],
            alpha=0.20,
            linewidth=0.45 if compact else 0.6,
            solid_capstyle="round",
            zorder=2,
        )
        ax.plot(
            segment["global_step"],
            smooth,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"] if compact else style["linewidth"] + 0.25,
            alpha=0.98,
            zorder=3,
        )
        start = (
            float(segment["global_step"].iloc[0]),
            float(smooth[0]),
            str(segment["jid"].iloc[0]),
        )
        end = (
            float(segment["global_step"].iloc[-1]),
            float(smooth[-1]),
            str(segment["jid"].iloc[-1]),
        )
        if previous is not None:
            is_cross = run_id in CROSS_SEED
            ax.plot(
                [previous[0], start[0]],
                [previous[1], start[1]],
                color="#7b1fa2" if is_cross else "#78909c",
                linestyle=":",
                linewidth=1.0,
                alpha=0.9,
                zorder=2,
            )
            ax.scatter(
                [start[0]],
                [start[1]],
                marker="D",
                s=22 if compact else 34,
                facecolors="white",
                edgecolors="#1565c0",
                linewidths=1.1,
                zorder=6,
            )
        if role == "crashed_pre_resume":
            ax.scatter(
                [end[0]],
                [end[1]],
                marker="X",
                s=30 if compact else 48,
                color="#c62828",
                linewidths=0.7,
                zorder=7,
            )
        elif role == "interrupted_pre_resume":
            ax.scatter(
                [end[0]],
                [end[1]],
                marker="s",
                s=18 if compact else 28,
                facecolors="#ffca28",
                edgecolors="#6d4c41",
                linewidths=0.7,
                zorder=7,
            )
        elif role == "timeout_final":
            ax.scatter(
                [end[0]],
                [end[1]],
                marker="o",
                s=34 if compact else 54,
                facecolors="white",
                edgecolors="#c62828",
                linewidths=1.3,
                zorder=7,
            )
        else:
            ax.scatter(
                [end[0]],
                [end[1]],
                marker="o",
                s=28 if compact else 44,
                facecolors=[base_color],
                edgecolors="black",
                linewidths=0.8,
                zorder=7,
            )
        previous = end

    target = int(run_summary["target_step"])
    ax.axvline(
        target,
        color="#263238",
        linestyle=(0, (3, 2)),
        linewidth=0.9,
        alpha=0.85,
        zorder=1,
    )

    # Equally spaced attempt ribbon. Hollow gray squares are submitted attempts
    # without recoverable W&B loss rows, including zero-data resume relays.
    attempt_count = len(run_attempts)
    if attempt_count:
        xs = np.linspace(0.04, 0.96, attempt_count)
        for x, (_, attempt) in zip(xs, run_attempts.iterrows(), strict=True):
            if not bool(attempt["has_loss_data"]):
                face = "none"
                edge = "#9e9e9e"
            elif str(attempt["jid"]) == str(run_points["jid"].iloc[-1]):
                face = base_color
                edge = "#263238"
            elif str(attempt["wandb_state"]).lower() == "crashed":
                face = "#ef6c00"
                edge = "#b71c1c"
            else:
                face = "#fbc02d"
                edge = "#6d4c41"
            ax.scatter(
                [x],
                [0.035],
                transform=ax.transAxes,
                marker="s",
                s=12 if compact else 22,
                facecolors=face,
                edgecolors=edge,
                linewidths=0.7,
                zorder=8,
            )

    ax.text(
        0.015,
        0.97,
        (
            f"{int(run_summary['n_observed_segments'])} loss seg / "
            f"{int(run_summary['n_manifest_attempts'])} attempts"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.3 if compact else 8,
        color="#455a64",
    )
    if run_id in CROSS_SEED:
        cross = CROSS_SEED[run_id]
        ax.text(
            0.985,
            0.97,
            f"CROSS-SEED {cross['origin_seed']}→{cross['final_seed']}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.0 if compact else 7.5,
            color="#6a1b9a",
            fontweight="bold",
            bbox={
                "facecolor": "#f3e5f5",
                "edgecolor": "#ab47bc",
                "boxstyle": "round,pad=0.18",
                "linewidth": 0.6,
            },
        )
    elif run_id == INCOMPLETE_RUN:
        ax.text(
            0.985,
            0.97,
            "TIMEOUT 53,970 / 63,407",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.0 if compact else 7.5,
            color="#b71c1c",
            fontweight="bold",
            bbox={
                "facecolor": "#ffebee",
                "edgecolor": "#ef5350",
                "boxstyle": "round,pad=0.18",
                "linewidth": 0.6,
            },
        )


def padded_limits(values: pd.Series, fraction: float = 0.08) -> tuple[float, float]:
    low = float(values.min())
    high = float(values.max())
    span = max(high - low, 0.03)
    return max(0.0, low - span * fraction), high + span * fraction


def make_standalone(
    points: pd.DataFrame,
    attempts: pd.DataFrame,
    summary: pd.DataFrame,
    colors: dict[str, object],
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    run_dir = output_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for index, row in enumerate(summary.itertuples(index=False), start=1):
        run_id = str(row.run_id)
        frame = points[points["run_id"] == run_id]
        run_attempts = attempts[attempts["run_id"] == run_id]
        run_summary = summary.set_index("run_id").loc[run_id]
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        plot_run(
            ax,
            frame,
            run_attempts,
            run_summary,
            colors[str(row.label)],
            compact=False,
        )
        x_max = max(
            float(row.target_step) * 1.045,
            float(frame["global_step"].max()) * 1.03,
        )
        ax.set_xlim(0, x_max)
        ax.set_ylim(*padded_limits(frame["step_loss"], 0.07))
        ax.grid(True, color="#dfe5e8", linewidth=0.55, alpha=0.8)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.set_xlabel("Training step")
        ax.set_ylabel("W&B step_loss (raw + per-segment rolling median)")
        fig.suptitle(
            f"{run_id}: complete recoverable job history",
            fontsize=13,
            fontweight="bold",
        )
        fig.text(
            0.01,
            0.012,
            "Hollow ribbon square = attempt with no recoverable loss rows; "
            "dotted connector = resume handoff, not a loss observation.",
            fontsize=7,
            color="#546e7a",
        )
        fig.tight_layout(rect=(0, 0.035, 1, 0.94))
        path = run_dir / f"{index:02d}_{run_id}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        created.append(path)
    return created


def make_composite(
    points: pd.DataFrame,
    attempts: pd.DataFrame,
    summary: pd.DataFrame,
    colors: dict[str, object],
    output_dir: Path,
    raw_sha: str,
    manifest_sha: str,
    dpi: int,
) -> list[Path]:
    fig = plt.figure(figsize=(18, 34))
    grid = fig.add_gridspec(
        nrows=14,
        ncols=3,
        height_ratios=[1.6, 0.12] + [1.0] * 12,
        left=0.065,
        right=0.985,
        top=0.946,
        bottom=0.035,
        hspace=0.70,
        wspace=0.20,
    )
    overview = fig.add_subplot(grid[0, :])
    for run_id, run_frame in points.groupby("run_id", sort=False):
        label = str(run_frame["label"].iloc[0])
        for _, segment in run_frame.groupby(
            "observed_segment_rank", sort=True, observed=True
        ):
            segment = segment.sort_values("global_step", kind="stable")
            role = str(segment["segment_role"].iloc[0])
            style = segment_style(role, colors[label])
            overview.plot(
                segment["progress"],
                rolling_median(segment),
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=0.85,
                alpha=0.52,
            )
    overview.axvline(
        1.0,
        color="#263238",
        linestyle=(0, (3, 2)),
        linewidth=1.0,
        alpha=0.9,
    )
    overview.set_xlim(0, 1.045)
    overview.set_ylim(*padded_limits(points["step_loss"], 0.04))
    overview.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    overview.set_xlabel("Progress toward intended target step")
    overview.set_ylabel("W&B step_loss")
    overview.grid(True, color="#dfe5e8", linewidth=0.55, alpha=0.8)
    overview.set_title(
        "Overview: all 54 physical loss segments (per-segment rolling median; no solid cross-job joins)",
        fontsize=10,
        fontweight="bold",
    )

    summary_lookup = summary.set_index(["label", "seed"], drop=False)
    for row_index, label in enumerate(SIZE_ORDER):
        row_points = points[points["label"] == label]
        row_y = padded_limits(row_points["step_loss"], 0.075)
        row_target = int(
            summary.loc[summary["label"] == label, "target_step"].iloc[0]
        )
        for col_index, seed in enumerate(SEED_ORDER):
            ax = fig.add_subplot(grid[row_index + 2, col_index])
            key = (label, seed)
            if key not in summary_lookup.index:
                ax.set_facecolor("#f3f4f5")
                ax.text(
                    0.5,
                    0.56,
                    f"{label}-s{seed}",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="#7b8790",
                )
                ax.text(
                    0.5,
                    0.40,
                    EXPECTED_MISSING[key],
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=6.8,
                    color="#8a949b",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color("#c7cdd1")
                continue
            run_summary = summary_lookup.loc[key]
            run_id = str(run_summary["run_id"])
            frame = points[points["run_id"] == run_id]
            run_attempts = attempts[attempts["run_id"] == run_id]
            plot_run(
                ax,
                frame,
                run_attempts,
                run_summary,
                colors[label],
                compact=True,
            )
            ax.set_xlim(
                0,
                max(
                    row_target * 1.045,
                    float(row_points["global_step"].max()) * 1.025,
                ),
            )
            ax.set_ylim(*row_y)
            ax.grid(True, color="#e1e6e8", linewidth=0.45, alpha=0.75)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.set_title(
                f"{run_id}  |  {int(run_summary['n_loss_points']):,} pts",
                fontsize=7.4,
                fontweight="bold",
                pad=3,
            )
            if col_index == 0:
                ax.set_ylabel("step_loss", fontsize=7)
            else:
                ax.tick_params(labelleft=False)
            if row_index == len(SIZE_ORDER) - 1:
                ax.set_xlabel("training step", fontsize=7)
            else:
                ax.tick_params(labelbottom=False)
            ax.tick_params(labelsize=6.2)

    legend = [
        Line2D(
            [0],
            [0],
            color="#607d8b",
            linewidth=0.6,
            alpha=0.25,
            label="all raw loss points",
        ),
        Line2D(
            [0],
            [0],
            color="#1565c0",
            linewidth=1.4,
            label="final observed segment",
        ),
        Line2D(
            [0],
            [0],
            color="#d95f02",
            linestyle="--",
            linewidth=1.3,
            marker="X",
            markerfacecolor="#c62828",
            label="crashed pre-resume",
        ),
        Line2D(
            [0],
            [0],
            color="#b8860b",
            linestyle="--",
            linewidth=1.3,
            marker="s",
            markerfacecolor="#ffca28",
            label="interrupted then resumed",
        ),
        Line2D(
            [0],
            [0],
            color="#78909c",
            linestyle=":",
            marker="D",
            markerfacecolor="white",
            markeredgecolor="#1565c0",
            label="resume handoff/start",
        ),
        Line2D(
            [0],
            [0],
            color="#263238",
            linestyle=(0, (3, 2)),
            linewidth=1.0,
            label="intended target step",
        ),
        Patch(
            facecolor="none",
            edgecolor="#9e9e9e",
            label="zero-data attempt (ribbon)",
        ),
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=4,
        frameon=False,
        fontsize=8,
        handlelength=2.3,
    )
    fig.suptitle(
        "33 training-loss job histories: failed → resume segments in the same subplot",
        fontsize=17,
        fontweight="bold",
        y=0.993,
    )
    fig.text(
        0.5,
        0.962,
        (
            "300-DPI composite • raw W&B step_loss + within-segment rolling median • "
            "54 observed physical segments / 10,403 raw points • "
            "6M cross-seed checkpoint ancestry is explicitly marked"
        ),
        ha="center",
        fontsize=9.5,
        color="#37474f",
    )
    fig.text(
        0.012,
        0.010,
        (
            "Every physical JID/W&B run is plotted separately; dotted handoffs are provenance connectors, "
            "not interpolated loss. Hollow gray ribbon squares retain attempts with no recoverable loss rows. "
            f"Inputs sha256: all_loss={raw_sha[:12]}…; manifest={manifest_sha[:12]}…."
        ),
        fontsize=7.2,
        color="#546e7a",
    )
    stem = output_dir / "training_loss_job_histories_33run_composite"
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    svg = stem.with_suffix(".svg")
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    fig.savefig(svg)
    plt.close(fig)
    return [png, pdf, svg]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-loss",
        type=Path,
        default=DEFAULT_ROOT / "all_loss_curves.v2_clean.csv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_ROOT / "scaling_law_data.csv",
    )
    parser.add_argument(
        "--endpoint",
        type=Path,
        default=DEFAULT_ROOT / "aramis" / "results" / "selected_test_endpoint.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "artifacts"
        / "training_loss_job_histories_33run_complete",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG output resolution (default: 300 DPI)",
    )
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    points, attempts, segments, summary = build_cohort(
        args.raw_loss, args.manifest, args.endpoint
    )
    colors = {
        label: mpl.colormaps["viridis"](position)
        for label, position in zip(
            SIZE_ORDER, np.linspace(0.06, 0.94, len(SIZE_ORDER)), strict=True
        )
    }

    stem = "training_loss_job_histories_33run"
    points_path = args.output_dir / f"{stem}_points.csv"
    attempts_path = args.output_dir / f"{stem}_attempts.csv"
    segments_path = args.output_dir / f"{stem}_segments.csv"
    summary_path = args.output_dir / f"{stem}_summary.csv"
    points_out = points[
        [
            "size_order",
            "label",
            "nominal_seed",
            "run_id",
            "attempt_rank",
            "observed_segment_rank",
            "jid",
            "wandb_run_id",
            "wandb_state",
            "segment_role",
            "global_step",
            "step_loss",
            "progress",
            "target_step",
            "cross_seed_restore",
        ]
    ]
    points_out.to_csv(points_path, index=False)
    attempts.to_csv(attempts_path, index=False)
    segments.to_csv(segments_path, index=False)
    summary.to_csv(summary_path, index=False)

    raw_sha = sha256(args.raw_loss)
    manifest_sha = sha256(args.manifest)
    endpoint_sha = sha256(args.endpoint)
    standalone = make_standalone(
        points, attempts, summary, colors, args.output_dir, args.dpi
    )
    composite = make_composite(
        points,
        attempts,
        summary,
        colors,
        args.output_dir,
        raw_sha,
        manifest_sha,
        args.dpi,
    )
    artifacts = [
        points_path,
        attempts_path,
        segments_path,
        summary_path,
        *standalone,
        *composite,
    ]
    manifest_payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "figure_title": (
            "33 training-loss job histories: failed to resume segments "
            "in the same subplot"
        ),
        "protocol": {
            "outcome": "W&B training step_loss",
            "raw_points": "all selected loss rows are plotted with low alpha",
            "smooth": "rolling median computed separately inside each physical segment",
            "cross_job_policy": "no solid cross-job loss line",
            "resume_connector": "gray or purple dotted provenance handoff",
            "zero_data_policy": "shown as hollow square in attempt ribbon",
            "png_dpi": args.dpi,
            "six_m_policy": CROSS_SEED,
            "excluded_unrelated_long_d_jids": sorted(EXCLUDED_LONG_D_JIDS),
        },
        "audit": {
            "n_sizes": int(summary["label"].nunique()),
            "n_logical_runs": int(summary["run_id"].nunique()),
            "n_observed_physical_segments": int(len(segments)),
            "n_unique_observed_jids": int(segments["jid"].nunique()),
            "n_raw_loss_points": int(len(points)),
            "n_multi_segment_runs": int(
                (summary["n_observed_segments"] > 1).sum()
            ),
            "n_zero_data_attempts": int((~attempts["has_loss_data"]).sum()),
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
            str(args.raw_loss): {
                "sha256": raw_sha,
                "bytes": args.raw_loss.stat().st_size,
            },
            str(args.manifest): {
                "sha256": manifest_sha,
                "bytes": args.manifest.stat().st_size,
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
            for path in artifacts
        },
    }
    manifest_out = args.output_dir / f"{stem}_manifest.json"
    manifest_out.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                **manifest_payload["audit"],
                "standalone_pngs": len(standalone),
                "composite": [str(path) for path in composite],
                "manifest": str(manifest_out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
