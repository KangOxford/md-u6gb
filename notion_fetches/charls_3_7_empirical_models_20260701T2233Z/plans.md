# Plans

## Completed plan

- Fetch the exact Notion page `https://app.notion.com/p/38512c4568fd81a6b472cf09a5e009f1`.
- Treat the bracketed instruction as the active request and update that exact location in Notion.
- Rerun the 3.7 empirical components on CHARLS W1-W4 using the existing cleaned panel and Harmonized family variables.
- Use the prior W1-W4 cash-flow fixed-effect grammar: individual fixed effects, wave fixed effects, and individual-clustered standard errors.
- Estimate main effects for `ln_pension` and `ln_transfer` on `Q_equal_fixed`.
- Estimate psychological and body first-stage models for `cesd10`, `loneliness`, `life_satisfaction`, and `adl_basic`.
- Estimate closure models for `Q_equal_fixed` with each mediator included.
- Estimate heterogeneity using two moderators:
  - `low_contact = 1 - h?kcntf`, available for elders with living children.
  - `low_transfer_zero = 1(transfer <= 0)`.
- Write the result as a Notion callout directly below the struck-through prompt.
- After Notion verification, write task-local markdown records and commit only those files.

## Recommended writing plan

- Keep 3.7.1 as a strong pension main-effect model.
- Keep 3.7.2 as a psychological mechanism section, but phrase it as CES-D and life-satisfaction supported rather than all psychological outcomes supported.
- Keep 3.7.3 body function as a contrast channel, not the main mechanism.
- Reframe 3.7.4 as exploratory heterogeneity or robustness because neither `low_contact` nor `low_transfer_zero` interactions are significant.

