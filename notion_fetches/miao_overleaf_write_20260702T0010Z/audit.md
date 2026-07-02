# Markdown Audit for Notion and Overleaf Drafting

## Verdict

The current 3.7 Notion content is ready to serve as the source outline for the empirical section, but it should not be copied into Overleaf verbatim. It needs a light academic rewrite and LaTeX conversion.

## Keep

- Keep the main line: pension income improves elderly-care quality and life satisfaction.
- Keep child transfer as a control or appendix robustness item, not as a core explanatory variable.
- Keep psychological status as the main mechanism because CES-D and life satisfaction are stronger than ADL.
- Keep body function as an auxiliary mechanism.
- Keep living-alone heterogeneity using `hhres == 1`.
- Keep robustness checks that directly address measurement and mechanical-index concerns.

## Revise Before Overleaf

- Replace Notion inline math style like `$`Q_{it}`$` with normal LaTeX math such as `$Q_{it}$`.
- Avoid phrases like "old Q4" in the paper body; write "previous specification" or "alternative quality index" instead.
- Do not overstate causality. The current design supports fixed-effect associations with strong controls, not a clean natural experiment.
- Add sample size, individual fixed effects, wave fixed effects, and clustered standard errors notes under each table.
- Make the no-income quality index robustness prominent because it addresses the key mechanical concern.
- Move child-transfer-control and living-alone-sample checks to the appendix or a short footnote.
- Keep lagged pension as a diagnostic appendix check only; do not use it to support the main hypothesis.
- Use consistent terminology: "pension income", "elderly-care quality", "psychological status", "body function", and "living alone".

## Main-Text Table Plan

| Table | Placement | Content | Status |
| --- | --- | --- | --- |
| Table 1 | Main text | Baseline `ln_pension -> Q_equal_fixed` and life satisfaction | Ready |
| Table 2 | Main text | CES-D, life satisfaction, ADL mechanism tests | Ready with wording caution |
| Table 3 | Main text | Living-alone heterogeneity | Ready |
| Table 4 | Main text | Alternative outcome, no-income index, pension dummy, 99% winsorization | Ready |
| Appendix A | Appendix | Child-transfer control and living-alone sample restriction | Demoted |
| Appendix B | Appendix | Leave-one-wave-out and lagged pension diagnostics | Demoted |

## Risks

- Overleaf Git access is currently blocked by connection timeouts, so no `.tex` file has been edited yet.
- The exact Overleaf project structure is unknown for the requested repository.
- If the repository contains an existing paper draft, section numbers and terminology may need to be aligned before writing.
- If the paper uses a journal or university template, table formatting must follow that template.

## Recommendation

Once Overleaf access works, write the empirical section first. The first LaTeX commit should only add or revise the empirical strategy/results section and associated tables, without restructuring the full paper.
