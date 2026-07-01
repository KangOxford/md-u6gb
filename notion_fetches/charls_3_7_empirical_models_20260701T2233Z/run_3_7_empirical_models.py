from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


HELPER_DIR = Path("/lus/lfs1aip2/projects/public/u6gb/notion_fetches/miao_old_heterogeneity_20260619T180435Z")
sys.path.insert(0, str(HELPER_DIR))

import run_coresidence_heterogeneity as rc  # noqa: E402


def fit(label: str, data: pd.DataFrame, y: str, exog: list[str], extra: dict[str, object] | None = None) -> pd.DataFrame:
    extra = extra or {}
    res, err, fit_data = rc.fit_panel(data, y, exog)
    if res is None:
        return pd.DataFrame(
            [
                {
                    "model": label,
                    "dv": y,
                    "error": err,
                    "n": len(fit_data),
                    "n_ids": fit_data["ID"].nunique() if "ID" in fit_data else np.nan,
                    **extra,
                }
            ]
        )
    return pd.DataFrame(rc.result_rows(res, label, y, len(fit_data), fit_data["ID"].nunique(), extra))


def load_work() -> pd.DataFrame:
    panel = pd.read_csv(rc.PANEL)
    panel["ID"] = rc.normalize_id(panel["ID"])
    panel["Q_equal_fixed"] = rc.build_q_equal_fixed(panel)
    family = rc.load_harmonized_family_vars()
    work = panel.merge(family, on=["ID", "wave"], how="left", validate="many_to_one")
    work = work[work["wave"].between(1, 4)].copy()
    work["has_living_child"] = np.where(work["child"].notna(), (work["child"] > 0).astype(float), np.nan)
    work["weekly_inperson"] = work["kcntf"]
    work["low_contact"] = np.where(
        work["has_living_child"].eq(1) & work["weekly_inperson"].isin([0, 1]),
        1 - work["weekly_inperson"],
        np.nan,
    )
    work["low_transfer_zero"] = np.where(work["transfer"].notna(), (work["transfer"] <= 0).astype(float), np.nan)

    wide = pd.read_stata(
        str(rc.HCHARLS),
        columns=["ID"] + [f"r{wave}flonel" for wave in range(1, 5)],
        convert_categoricals=False,
    )
    lon_rows = []
    for wave in range(1, 5):
        lon_rows.append(
            pd.DataFrame(
                {
                    "ID": rc.normalize_id(wide["ID"]),
                    "wave": wave,
                    "loneliness": pd.to_numeric(wide[f"r{wave}flonel"], errors="coerce"),
                }
            )
        )
    loneliness = pd.concat(lon_rows, ignore_index=True)
    loneliness.loc[~loneliness["loneliness"].between(1, 4), "loneliness"] = np.nan
    return work.merge(loneliness, on=["ID", "wave"], how="left", validate="many_to_one")


def print_rows(title: str, df: pd.DataFrame, cols: list[str]) -> None:
    print("\n" + title)
    out = df[[c for c in cols if c in df.columns]].copy()
    for col in ["coef", "se", "p", "within_r2"]:
        if col in out:
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
    print(out.to_string(index=False))


def main() -> None:
    work = load_work()
    base = rc.BASE_EXOG
    psy = ["cesd10", "loneliness", "life_satisfaction"]
    body = ["adl_basic"]
    dvs = ["Q_equal_fixed"] + psy + body

    main_effects = pd.concat(
        [
            fit("main_cashflow_w1w4", work, y, base, {"channel": "q" if y == "Q_equal_fixed" else "psych" if y in psy else "body"})
            for y in dvs
        ],
        ignore_index=True,
    )

    closures = []
    for mediator in psy + body:
        closures.append(
            fit(
                "closure_q_plus_mediator",
                work,
                "Q_equal_fixed",
                base + [mediator],
                {"mediator": mediator, "channel": "psych" if mediator in psy else "body"},
            )
        )
    closure = pd.concat(closures, ignore_index=True)

    contact = work[work["low_contact"].notna()].copy()
    contact["ln_pension_x_low_contact"] = contact["ln_pension"] * contact["low_contact"]
    het_contact = pd.concat(
        [
            fit("het_low_contact", contact, y, base + ["low_contact", "ln_pension_x_low_contact"], {"moderator": "low_contact"})
            for y in dvs
        ],
        ignore_index=True,
    )

    transfer = work[work["low_transfer_zero"].notna()].copy()
    transfer["ln_pension_x_low_transfer_zero"] = transfer["ln_pension"] * transfer["low_transfer_zero"]
    transfer_exog = [
        "ln_pension",
        "ln_housing_price",
        "medical_cpi",
        "married_binary",
        "has_insurance",
        "low_transfer_zero",
        "ln_pension_x_low_transfer_zero",
    ]
    het_transfer = pd.concat(
        [fit("het_low_transfer_zero", transfer, y, transfer_exog, {"moderator": "low_transfer_zero"}) for y in dvs],
        ignore_index=True,
    )

    print_rows(
        "MAIN_EFFECTS_TERMS",
        main_effects[main_effects["term"].isin(["ln_pension", "ln_transfer"])],
        ["model", "channel", "dv", "term", "coef", "se", "p", "sig", "n", "n_ids", "within_r2"],
    )
    print_rows(
        "CLOSURE_MEDIATOR_TERMS",
        closure[closure["term"].eq(closure["mediator"])],
        ["model", "channel", "mediator", "dv", "term", "coef", "se", "p", "sig", "n", "n_ids"],
    )
    print_rows(
        "HET_LOW_CONTACT_INTERACTIONS",
        het_contact[het_contact["term"].isin(["low_contact", "ln_pension_x_low_contact"])],
        ["model", "moderator", "dv", "term", "coef", "se", "p", "sig", "n", "n_ids", "within_r2"],
    )
    print_rows(
        "HET_LOW_TRANSFER_INTERACTIONS",
        het_transfer[het_transfer["term"].isin(["low_transfer_zero", "ln_pension_x_low_transfer_zero"])],
        ["model", "moderator", "dv", "term", "coef", "se", "p", "sig", "n", "n_ids", "within_r2"],
    )


if __name__ == "__main__":
    main()

