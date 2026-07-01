from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


HELPER_DIR = Path("/lus/lfs1aip2/projects/public/u6gb/notion_fetches/miao_old_heterogeneity_20260619T180435Z")
sys.path.insert(0, str(HELPER_DIR))

import run_coresidence_heterogeneity as rc  # noqa: E402


def fit_rows(data: pd.DataFrame, model: str, y: str, exog: list[str], extra: dict[str, object] | None = None) -> pd.DataFrame:
    extra = extra or {}
    res, err, fit_data = rc.fit_panel(data, y, exog)
    if res is None:
        return pd.DataFrame(
            [
                {
                    "model": model,
                    "dv": y,
                    "error": err,
                    "n": len(fit_data),
                    "n_ids": fit_data["ID"].nunique() if "ID" in fit_data else np.nan,
                    **extra,
                }
            ]
        )
    return pd.DataFrame(rc.result_rows(res, model, y, len(fit_data), fit_data["ID"].nunique(), extra))


def load_work_panel() -> pd.DataFrame:
    panel = pd.read_csv(rc.PANEL)
    panel["ID"] = rc.normalize_id(panel["ID"])
    panel["Q_equal_fixed"] = rc.build_q_equal_fixed(panel)
    family = rc.load_harmonized_family_vars()
    work = panel.merge(family, on=["ID", "wave"], how="left", validate="many_to_one")
    work = work[work["wave"].between(1, 4)].copy()
    work["has_living_child"] = np.where(work["child"].notna(), (work["child"] > 0).astype(float), np.nan)
    work["weekly_inperson"] = work["kcntf"]
    work["weekly_any"] = work["kcnt"]
    work["weekly_phone_email"] = work["kcntpm"]
    work["child_near"] = work["lvnear"]

    cols = ["ID"] + [f"r{wave}flonel" for wave in range(1, 5)]
    wide = pd.read_stata(str(rc.HCHARLS), columns=cols, convert_categoricals=False)
    rows = []
    for wave in range(1, 5):
        rows.append(
            pd.DataFrame(
                {
                    "ID": rc.normalize_id(wide["ID"]),
                    "wave": wave,
                    "loneliness": pd.to_numeric(wide[f"r{wave}flonel"], errors="coerce"),
                }
            )
        )
    loneliness = pd.concat(rows, ignore_index=True)
    loneliness.loc[~loneliness["loneliness"].between(1, 4), "loneliness"] = np.nan
    return work.merge(loneliness, on=["ID", "wave"], how="left", validate="many_to_one")


def main() -> None:
    work = load_work_panel()
    base = rc.BASE_EXOG
    contacts = ["weekly_inperson", "weekly_any", "weekly_phone_email"]
    dvs = ["Q_equal_fixed", "Q", "cesd10", "loneliness", "life_satisfaction", "adl_basic"]

    sample_rows = []
    for contact in contacts:
        sample = work[work["has_living_child"].eq(1) & work[contact].notna()].copy()
        sample_rows.append(
            {
                "contact_var": contact,
                "n": len(sample),
                "n_ids": sample["ID"].nunique(),
                "share_weekly_1": sample[contact].mean(),
                "within_sd": (sample[contact] - sample.groupby("ID")[contact].transform("mean")).std(),
            }
        )
    print("SAMPLE")
    print(pd.DataFrame(sample_rows).to_string(index=False))

    main_tables = []
    for contact in contacts:
        sample = work[work["has_living_child"].eq(1) & work[contact].notna()].copy()
        for y in dvs:
            main_tables.append(fit_rows(sample, "dual_core_contact_main", y, [contact] + base, {"contact_var": contact}))
    main_df = pd.concat(main_tables, ignore_index=True)
    print("\nMAIN CONTACT TERMS")
    print(main_df[main_df["term"].isin(contacts)].to_string(index=False))

    sample = work[work["has_living_child"].eq(1) & work["weekly_inperson"].notna()].copy()
    comp_models = {
        "A_transfer_only_plus_controls": ["ln_transfer", "ln_housing_price", "medical_cpi", "married_binary", "has_insurance"],
        "B_transfer_plus_weekly_inperson": ["ln_transfer", "weekly_inperson", "ln_housing_price", "medical_cpi", "married_binary", "has_insurance"],
        "C_transfer_weekly_inperson_pension": ["ln_transfer", "weekly_inperson", "ln_pension", "ln_housing_price", "medical_cpi", "married_binary", "has_insurance"],
    }
    comp = pd.concat(
        [fit_rows(sample, name, "Q_equal_fixed", exog, {"contact_var": "weekly_inperson"}) for name, exog in comp_models.items()],
        ignore_index=True,
    )
    print("\nCOMPETITION TERMS")
    print(comp[comp["term"].isin(["ln_transfer", "weekly_inperson", "ln_pension"])].to_string(index=False))

    lag = work.sort_values(["ID", "wave"]).copy()
    for contact in contacts:
        lag[f"L1_{contact}"] = lag.groupby("ID")[contact].shift(1)
    lag_rows = []
    for contact in contacts:
        sample = lag[lag["has_living_child"].eq(1) & lag[f"L1_{contact}"].notna() & lag["wave"].between(2, 4)].copy()
        for y in ["Q_equal_fixed", "cesd10", "loneliness", "life_satisfaction", "adl_basic"]:
            lag_rows.append(fit_rows(sample, "lagged_contact", y, [f"L1_{contact}"] + base, {"contact_var": contact}))
    lag_df = pd.concat(lag_rows, ignore_index=True)
    print("\nLAGGED CONTACT TERMS")
    print(lag_df[lag_df["term"].isin([f"L1_{contact}" for contact in contacts])].to_string(index=False))

    near_sample = work[
        work["has_living_child"].eq(1) & work["weekly_inperson"].notna() & work["child_near"].notna()
    ].copy()
    distance = pd.concat(
        [
            fit_rows(near_sample, "distance_first_stage", "weekly_inperson", ["child_near"] + base),
            fit_rows(near_sample, "distance_reduced_form", "Q_equal_fixed", ["child_near"] + base),
        ],
        ignore_index=True,
    )
    print("\nDISTANCE TERMS")
    print(distance[distance["term"].eq("child_near")].to_string(index=False))


if __name__ == "__main__":
    main()

