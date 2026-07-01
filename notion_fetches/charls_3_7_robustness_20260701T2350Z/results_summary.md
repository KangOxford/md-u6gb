# CHARLS 3.7 Robustness Summary

The Notion page now recommends a four-table empirical structure:

1. Baseline regressions.
2. Psychological and body-function mechanisms.
3. Living-alone heterogeneity.
4. Robustness checks.

Recommended main robustness table columns:

| Check | Outcome | Pension coefficient | P-value | Interpretation |
| --- | --- | ---: | ---: | --- |
| Baseline with transfer | Q_equal_fixed | 0.00294882 | 2.77e-25 | Main result |
| No child-transfer control | Q_equal_fixed | 0.00295438 | 2.14e-25 | Not driven by transfer controls |
| Legacy Q | Q | 0.00173882 | 2.13e-05 | Alternative outcome |
| No-income index | Q_no_income | 0.00120171 | 0.000159 | Not mechanical income effect |
| Non-economic index | Q_non_economic | 0.00144299 | 1.90e-05 | Psych/body/satisfaction only |
| Pension receipt dummy | Q_equal_fixed | 0.0226197 | 2.98e-21 | Extensive-margin pension access |
| 99% winsorized pension | Q_equal_fixed | 0.00294999 | 2.86e-25 | Not driven by outliers |
| Living-alone-available sample | Q_equal_fixed | 0.00288749 | 9.05e-19 | Compatible with heterogeneity sample |

Recommended appendix checks:

| Check | Main finding | Placement |
| --- | --- | --- |
| Leave-one-wave-out | Q_equal_fixed coefficient remains 0.00289624 to 0.00305783, all p<0.001 | Appendix robustness |
| Lagged pension | Mixed/weak; Q turns negative and psychological outcomes are not significant | Appendix diagnostic only |
