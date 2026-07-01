# Findings

## 2026-07-01 3.7 pension-centered rewrite

- Target Notion page: `https://app.notion.com/p/38512c4568fd81a6b472cf09a5e009f1`.
- User instruction: keep the earlier main idea of pension affecting elderly care quality, especially life satisfaction; drop child intergenerational transfer from the main line if insignificant; keep psychological and body-function mechanisms, with emphasis on psychological state; if child companionship heterogeneity is insignificant, use whether the elderly person lives alone as the heterogeneity test.
- Page was rewritten around the pension-centered model rather than the child-transfer/contact framing.
- Child intergenerational transfer was demoted to optional control or appendix robustness because the latest rerun found `ln_transfer -> Q_equal_fixed = 0.000215, p=0.427`.
- Psychological mechanism kept as the main channel. Current evidence: `ln_pension -> cesd10 = -0.0582, p=0.0008`; `ln_pension -> life_satisfaction = 0.00668, p=0.0052`; `ln_pension -> loneliness` is not significant.
- Body-function mechanism kept as a contrast channel. Current evidence: `ln_pension -> adl_basic = -0.00584, p=0.062`, weaker than psychological-state evidence.
- Heterogeneity was changed from child companionship to living alone because child-contact interaction was not significant. Prior living-alone evidence: `ln_pension x living_alone -> Q_equal_fixed = 0.004588, p<0.001`; life-satisfaction interaction is marginal, `coef=0.01397, p=0.0805`.
- Notion verification confirmed the user instruction is struck through and the page now contains the revised 3.7.1-3.7.4 structure.

