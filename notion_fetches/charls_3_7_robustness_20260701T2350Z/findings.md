# Findings

- Target Notion page: `https://app.notion.com/p/38512c4568fd81a6b472cf09a5e009f1`
- User instruction preserved on Notion as a struck-through prompt, with a Notion callout response immediately below it.
- Data source used for checks: `/lus/lfs1aip2/projects/public/s5e/quant_team/quant/miao/second/第二篇 养老质量/cleaned_data/panel_urban_elderly.csv`
- Helper code used: `/lus/lfs1aip2/projects/public/u6gb/notion_fetches/miao_old_heterogeneity_20260619T180435Z/run_coresidence_heterogeneity.py`

Key robustness evidence:
- Baseline with child transfer controlled: `ln_pension -> Q_equal_fixed = 0.00294882`, `p=2.77e-25`, `n=11521`.
- Baseline without child transfer: `ln_pension -> Q_equal_fixed = 0.00295438`, `p=2.14e-25`, `n=11521`.
- Legacy outcome `Q`: `ln_pension = 0.00173882`, `p=2.13e-05`.
- Life satisfaction: `ln_pension = 0.00668216`, `p=0.00523`.
- CES-D: `ln_pension = -0.0581621`, `p=0.000768`.
- ADL: `ln_pension = -0.00584448`, `p=0.0619`; weaker than psychological mechanisms.
- No-income quality index: `ln_pension = 0.00120171`, `p=0.000159`.
- Non-economic quality index: `ln_pension = 0.00144299`, `p=1.90e-05`.
- Pension receipt dummy on `Q_equal_fixed`: `0.0226197`, `p=2.98e-21`.
- 99% winsorized pension on `Q_equal_fixed`: `0.00294999`, `p=2.86e-25`.
- Living-alone-available sample: `ln_pension -> Q_equal_fixed = 0.00288749`, `p=9.05e-19`.
- Leave-one-wave-out checks keep `Q_equal_fixed` coefficients between `0.00289624` and `0.00305783`, all `p<0.001`.
- Lagged pension is not suitable as a main robustness result: W2-W4 sample shrinks, `L1_ln_pension -> Q_equal_fixed = -0.000755846`, `p=0.0442`, while psychological and ADL outcomes are not significant.

2026-07-02 follow-up:
- User questioned whether robustness item `(4) 控制子女转移支付与样本限制` should be deleted or moved to appendix.
- Decision recorded on Notion: do not delete completely, but demote it.
- Recommended placement: one short sentence in the main text, full coefficients in an appendix robustness/sample-consistency table.
- Rationale: the check supports model stability and heterogeneity-sample consistency, but it is not a new mechanism and should not compete with the stronger robustness checks.
