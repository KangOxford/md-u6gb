# Learnt Lessons

- `run_coresidence_heterogeneity.py::build_q_equal_fixed` returns a Series, so callers must assign it as `df["Q_equal_fixed"]`; replacing the whole dataframe causes an AttributeError downstream.
- `run_coresidence_heterogeneity.py::fit_panel` returns `(result, error, used_data)`, not a result object alone.
- The most persuasive robustness check for this CHARLS paper is not merely replacing the dependent variable; it is rebuilding the quality index without the income dimension to address mechanical-income concerns.
- The lagged pension check is empirically weak in the current W2-W4 sample and should not be framed as supporting evidence.
- Notion fetch still rejects `notion://docs/enhanced-markdown-spec` as an invalid URL in this environment, so updates should reuse markdown constructs already verified in the target page.
