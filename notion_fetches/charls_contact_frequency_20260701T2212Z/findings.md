# Findings

## CHARLS contact-frequency validation, 2026-07-01 UTC

- Target Notion page: `https://app.notion.com/p/a48bfd93f6c64bb2bb4172a19764289d`.
- Local cleaned panel used: `/lus/lfs1aip2/projects/public/s5e/quant_team/quant/miao/second/第二篇 养老质量/cleaned_data/panel_urban_elderly.csv`.
- Harmonized CHARLS source used: `/lus/lfs1aip2/projects/public/s5e/quant_team/quant/miao/second/第二篇 养老质量/数据/charls原始数据及社区代码/Harmonized_CHARLS_D/H_CHARLS_D_Data.dta`.
- Existing validated helper code reused: `/lus/lfs1aip2/projects/public/u6gb/notion_fetches/miao_old_heterogeneity_20260619T180435Z/run_coresidence_heterogeneity.py`.
- Variable-label check found that `h?kcntf` is the closest available "visit" proxy: any weekly contact with children in person.
- `h?kcnt` is combined in-person/phone/email weekly contact, not a pure visit count.
- Main result: `weekly_inperson -> Q_equal_fixed` coefficient is `0.002682`, `p=0.481`, with `N=8,764` and `3,219` IDs.
- Lag robustness does not support promotion to a core variable: `L1_weekly_inperson -> Q_equal_fixed` coefficient is `0.002846`, `p=0.595`.
- Distance proxy check: `child_near -> weekly_inperson` is strong (`coef=0.2141`, `p=2.3e-34`), but `child_near -> Q_equal_fixed` reduced form is not significant (`coef=0.00191`, `p=0.622`).
- Mechanism checks do not form the expected psychological channel: weekly in-person contact is not significant for `cesd10`, `loneliness`, or `life_satisfaction`.
- Notion reportback was created as four child pages under the target page:
  - `https://app.notion.com/p/39012c4568fd81d5ad22f1d58326dfe0`
  - `https://app.notion.com/p/39012c4568fd81b7b2ddc52577a5935b`
  - `https://app.notion.com/p/39012c4568fd814abc7dcf237dbbbcc7`
  - `https://app.notion.com/p/39012c4568fd815a8f15f3fe3ae57c06`

