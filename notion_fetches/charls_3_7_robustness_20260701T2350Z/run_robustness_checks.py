from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


HELPER_DIR = Path("/lus/lfs1aip2/projects/public/u6gb/notion_fetches/miao_old_heterogeneity_20260619T180435Z")
PANEL_PATH = Path(
    "/lus/lfs1aip2/projects/public/s5e/quant_team/quant/miao/second/第二篇 养老质量/cleaned_data/panel_urban_elderly.csv"
)

sys.path.insert(0, str(HELPER_DIR))
import run_coresidence_heterogeneity as rc  # noqa: E402


def minmax_index(df: pd.DataFrame, cols: dict[str, pd.Series], directions: dict[str, int]) -> pd.Series:
    norm = pd.DataFrame(index=df.index)
    for name, series in cols.items():
        x = pd.to_numeric(series, errors="coerce")
        if directions[name] < 0:
            x = -x
        xmin = x.min(skipna=True)
        xmax = x.max(skipna=True)
        if pd.isna(xmin) or pd.isna(xmax) or xmax <= xmin:
            norm[name] = 0.5
        else:
            norm[name] = (x - xmin) / (xmax - xmin)
        norm[name] = norm[name].fillna(norm[name].mean())
    return norm.mean(axis=1)


def add_result(rows: list[dict[str, object]], model: str, data: pd.DataFrame, outcome: str, exog: list[str], term: str) -> None:
    res, err, used = rc.fit_panel(data, outcome, exog)
    if res is None or term not in res.params.index:
        rows.append(
            {
                "model": model,
                "outcome": outcome,
                "term": term,
                "coef": np.nan,
                "se": np.nan,
                "p": np.nan,
                "n": len(used),
                "ids": used["ID"].nunique() if len(used) else 0,
                "err": err,
            }
        )
        return
    rows.append(
        {
            "model": model,
            "outcome": outcome,
            "term": term,
            "coef": float(res.params[term]),
            "se": float(res.std_errors[term]),
            "p": float(res.pvalues[term]),
            "n": len(used),
            "ids": used["ID"].nunique(),
            "within_r2": float(res.rsquared_within),
            "err": "",
        }
    )


def prepare_panel() -> pd.DataFrame:
    panel = pd.read_csv(PANEL_PATH)
    panel["ID"] = rc.normalize_id(panel["ID"])
    panel["wave"] = pd.to_numeric(panel["wave"], errors="coerce")
    panel["Q_equal_fixed"] = rc.build_q_equal_fixed(panel)
    panel["income_log"] = np.log1p(pd.to_numeric(panel["income_total"], errors="coerce").clip(lower=0))
    panel["housing_log"] = np.log1p(pd.to_numeric(panel["housing_value"], errors="coerce").clip(lower=0))
    panel["Q_no_income"] = minmax_index(
        panel,
        {
            "adl_basic": panel["adl_basic"],
            "cesd10": panel["cesd10"],
            "housing_log": panel["housing_log"],
            "life_satisfaction": panel["life_satisfaction"],
        },
        {"adl_basic": -1, "cesd10": -1, "housing_log": 1, "life_satisfaction": 1},
    )
    panel["Q_non_economic"] = minmax_index(
        panel,
        {
            "adl_basic": panel["adl_basic"],
            "cesd10": panel["cesd10"],
            "life_satisfaction": panel["life_satisfaction"],
        },
        {"adl_basic": -1, "cesd10": -1, "life_satisfaction": 1},
    )

    fam = rc.load_harmonized_family_vars()
    work = panel.merge(fam, on=["ID", "wave"], how="left", validate="many_to_one")
    work = work[work["wave"].between(1, 4)].copy()

    pension_raw = pd.to_numeric(work["pension"], errors="coerce").fillna(0).clip(lower=0)
    work["pension_positive"] = (pension_raw > 0).astype(float)
    work["ln_pension_w99"] = np.log1p(pension_raw.clip(upper=pension_raw.quantile(0.99)))
    work = work.sort_values(["ID", "wave"])
    work["L1_ln_pension"] = work.groupby("ID")["ln_pension"].shift(1)
    work["living_alone"] = np.where(work["hhres"].notna(), (work["hhres"] == 1).astype(float), np.nan)
    return work


def main() -> None:
    work = prepare_panel()
    rows: list[dict[str, object]] = []

    base_with_transfer = list(rc.BASE_EXOG)
    base_no_transfer = ["ln_pension", "ln_housing_price", "medical_cpi", "married_binary", "has_insurance"]
    receipt_exog = ["pension_positive", "ln_transfer", "ln_housing_price", "medical_cpi", "married_binary", "has_insurance"]
    winsor_exog = ["ln_pension_w99", "ln_transfer", "ln_housing_price", "medical_cpi", "married_binary", "has_insurance"]
    lag_exog = ["L1_ln_pension", "ln_transfer", "ln_housing_price", "medical_cpi", "married_binary", "has_insurance"]

    for outcome in ["Q_equal_fixed", "Q", "life_satisfaction", "cesd10", "adl_basic"]:
        add_result(rows, "baseline_with_transfer", work, outcome, base_with_transfer, "ln_pension")
        add_result(rows, "baseline_no_transfer", work, outcome, base_no_transfer, "ln_pension")

    for outcome in ["Q_no_income", "Q_non_economic"]:
        add_result(rows, "alternative_quality_index", work, outcome, base_with_transfer, "ln_pension")

    for outcome in ["Q_equal_fixed", "life_satisfaction", "cesd10", "adl_basic"]:
        add_result(rows, "pension_receipt_binary", work, outcome, receipt_exog, "pension_positive")
        add_result(rows, "pension_winsor99", work, outcome, winsor_exog, "ln_pension_w99")
        add_result(rows, "lagged_pension", work[work["L1_ln_pension"].notna()].copy(), outcome, lag_exog, "L1_ln_pension")

    add_result(
        rows,
        "living_alone_available_sample",
        work[work["living_alone"].notna()].copy(),
        "Q_equal_fixed",
        base_with_transfer,
        "ln_pension",
    )

    for drop_wave in [1, 2, 3, 4]:
        add_result(
            rows,
            f"drop_wave_{drop_wave}",
            work[work["wave"] != drop_wave].copy(),
            "Q_equal_fixed",
            base_with_transfer,
            "ln_pension",
        )

    out = pd.DataFrame(rows)
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.width", 180)
    pd.set_option("display.float_format", lambda x: f"{x:.6g}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
