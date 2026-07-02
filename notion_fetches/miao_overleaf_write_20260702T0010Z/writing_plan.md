# Overleaf Writing Plan

Target page: `https://app.notion.com/p/38512c4568fd81a6b472cf09a5e009f1`

Notion audit page: `https://app.notion.com/p/39112c4568fd81b2bffee957937c908f`

Target Overleaf remote:

```text
https://git@git.overleaf.com/6a45abc0a2fd90b8e04523f6
```

## Current Status

- The requested Overleaf repository could not be cloned from this environment.
- `git clone https://git@git.overleaf.com/6a45abc0a2fd90b8e04523f6 ...` was started and then manually interrupted after it hung without remote output.
- `git ls-remote https://git@git.overleaf.com/6a45abc0a2fd90b8e04523f6` timed out after 45 seconds.
- `git ls-remote https://git.overleaf.com/6a45abc0a2fd90b8e04523f6` timed out after 35 seconds.
- No usable checkout exists at `/lus/lfs1aip2/projects/public/u6gb/notion_fetches/miao_overleaf_write_20260702T0010Z/overleaf`.

## Writing Scope

Start with the empirical part of the paper, because this is the section with the clearest validated evidence:

1. Baseline effect: pension income improves elderly-care quality.
2. Psychological mechanism: pension income reduces CES-D and improves life satisfaction.
3. Body-function mechanism: keep as auxiliary because ADL is weaker.
4. Heterogeneity: use living alone, defined by `hhres == 1`.
5. Robustness: prioritize alternative outcomes, no-income quality index, pension receipt dummy, and 99% winsorized pension.
6. Appendix robustness: child-transfer control, living-alone-identifiable sample restriction, leave-one-wave-out checks, and lagged pension diagnostics.

## Proposed Paper Structure

1. Introduction
   - Research question: whether stable pension income improves elderly-care quality among urban elderly people.
   - Core contribution: separate institutional pension cash flow from unstable family transfer support.
   - Mechanism emphasis: psychological security and life satisfaction.

2. Literature Review and Hypotheses
   - Pension income and elderly welfare.
   - Economic security and psychological well-being.
   - Family support, living arrangement, and heterogeneous vulnerability.
   - Hypotheses: main effect, psychological mechanism, body-function auxiliary mechanism, living-alone heterogeneity.

3. Data, Variables, and Empirical Strategy
   - CHARLS W1-W4 sample.
   - Outcome: `Q_equal_fixed`, life satisfaction, CES-D, ADL.
   - Core explanatory variable: `ln_pension = ln(1 + pension)`.
   - Controls: child transfer as control only, housing price, medical CPI, marital status, insurance.
   - Model: individual fixed effects, wave fixed effects, individual-clustered standard errors.

4. Empirical Results
   - Table 1: baseline pension effect.
   - Table 2: psychological and body-function mechanisms.
   - Table 3: living-alone heterogeneity.
   - Table 4: main robustness checks.
   - Appendix tables: sample-consistency checks and lagged diagnostic checks.

5. Conclusion and Policy Implications
   - Pension stability matters for elderly-care quality.
   - Psychological welfare is the stronger channel.
   - Pension policy may matter more for elderly people without co-resident family support.

## Immediate Overleaf Tasks After Access Works

1. Clone the repository into a clean directory.
2. Identify the main `.tex` file and bibliography file.
3. Check whether the project uses `ctex`, `biblatex`, `natbib`, or a journal template.
4. Create an empirical-section draft from the current Notion 3.7 content.
5. Convert Notion formulas/tables into LaTeX syntax.
6. Compile locally if TeX tooling is available; otherwise commit and push to Overleaf for compilation.
7. Commit only the intended `.tex`, `.bib`, or figure/table files.

## First Writing Target

The first Overleaf writing target should be Section 3.7 / empirical strategy and result interpretation, not the whole paper. This reduces risk because the validated evidence is already concentrated there.
