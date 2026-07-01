# Findings

## CHARLS 3.7 empirical model rerun, 2026-07-01 UTC

- Target Notion page: `https://app.notion.com/p/38512c4568fd81a6b472cf09a5e009f1`.
- Bracketed instruction on the page requested a CHARLS rerun for main effects, mechanisms, and heterogeneity significance.
- Local cleaned panel used: `/lus/lfs1aip2/projects/public/s5e/quant_team/quant/miao/second/第二篇 养老质量/cleaned_data/panel_urban_elderly.csv`.
- Harmonized CHARLS source used: `/lus/lfs1aip2/projects/public/s5e/quant_team/quant/miao/second/第二篇 养老质量/数据/charls原始数据及社区代码/Harmonized_CHARLS_D/H_CHARLS_D_Data.dta`.
- Existing validated helper code reused: `/lus/lfs1aip2/projects/public/u6gb/notion_fetches/miao_old_heterogeneity_20260619T180435Z/run_coresidence_heterogeneity.py`.
- Main effect on `Q_equal_fixed`: `ln_pension` is positive and significant, coefficient `0.002949`, `p<0.001`; `ln_transfer` is not significant, coefficient `0.000215`, `p=0.427`.
- Psychological first stage: `ln_pension -> cesd10` coefficient `-0.0582`, `p=0.0008`; `ln_pension -> life_satisfaction` coefficient `0.00668`, `p=0.0052`; `ln_pension -> loneliness` is not significant, coefficient `-0.00386`, `p=0.263`.
- Child transfer is not significant in the psychological first-stage outcomes checked here.
- Mechanism closure: `cesd10`, `loneliness`, and `life_satisfaction` are each significant when included in `Q_equal_fixed` models, so the psychological outcomes themselves are strongly associated with养老质量.
- Body mechanism: `ln_pension -> adl_basic` is only marginal, coefficient `-0.00584`, `p=0.062`.
- Low-contact heterogeneity: `ln_pension x low_contact` is not significant for `Q_equal_fixed`, coefficient `-0.000376`, `p=0.595`; not significant for checked mechanism outcomes.
- Low-transfer heterogeneity: `ln_pension x low_transfer_zero` is not significant for `Q_equal_fixed`, coefficient `-0.000718`, `p=0.150`; not significant for checked mechanism outcomes.
- Notion was updated in place: the original bracketed prompt is struck through and a green result callout is directly below it.

