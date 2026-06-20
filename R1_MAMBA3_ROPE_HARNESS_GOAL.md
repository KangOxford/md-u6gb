# /goal: R1 Mamba3 RoPE Fraction Harness, Figures, and Notion Subpages

## 语言要求

除非引用英文原文、代码、文件路径、命令或 Alex 的原始问题，否则所有最终说明、报告正文、Notion 子页面内容、图注和给用户的总结都使用中文。

## Objective

Act as a co-scientist / harness engineer for the `exp_R1_Mamba3` RoPE question. Do **not** directly answer from intuition. Build a small, reproducible evidence pack that explains why the local R1 Mamba3 implementation uses partial RoPE by default, whether that matches or diverges from the Mamba-3 paper / upstream implementations, and what should be said back to Alex only after the checks are complete.

Original Notion page:

```text
https://www.notion.so/r1-mamba3-rope-36612c4568fd809d9082d4cf23663001?source=copy_link
```

Prompt on the page:

```text
这个有可能是个bug

Alex Bismuth [9:59 PM]
Does anyone know why in exp_R1 with Mamba-3 the RoPE only affects 50% of the planes? This seems to follow what was done by some implementations online which have default 50% of planes rotating and optionally 100% of planes but in the paper I only see the case with 100% of planes rotating.
[9:59 PM]
(I’m working on 2D RoPE where we would have a fraction of planes rotating on time and a fraction on “message content” and I stumbled upon this)

想办法理解他的问题
回答他的问题
convert this into a goal markdown file that i can use to do with /goal xxxxxx.md
```

## Required Posture

- Treat this as harness engineering, not a chat answer.
- Produce falsifiable evidence: exact files, exact knobs, minimal tests, and a short conclusion with confidence level.
- 最终产物必须回写到 Notion，作为原始 Notion 页下面的结构化子页面，而不是只留在本地 markdown。
- 用 `image2` 生成关键解释图和流程图。不要只用文字或手写 ASCII 图替代；若当前执行环境没有 `image2`，必须记录阻塞原因并用可用的等价图像生成工具作为 fallback，同时在报告中说明替代。
- Separate these claims:
  - What the local implementation does.
  - What the paper specifies.
  - What external/upstream implementation(s) do.
  - What R1 experiments actually trained/evaluated.
  - Whether there is a bug, an intentional hyperparameter choice, or an undocumented divergence.
- Do not run expensive training jobs unless the user explicitly asks. Prefer static audit, tiny shape tests, and existing result extraction.

## Local Experiment Root

```text
/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_R1_Mamba3
```

Start there.

Known relevant files from the first pass:

```text
run_train.py
node_wrapper.sh
train_full_autoreg.batch
s5/mamba3.py
s5/mamba3_jax.py
s5/flops.py
lob/init_train.py
phase_a_sweep.sh
phase_b_sweep.sh
phase_b_extended_sweep.sh
submit_rope_confirmation.sh
submit_rope_dstate_46m.sh
extract_rope_downstream.py
extract_rope_tex_numbers.py
extract_rope_conf_latest.py
extract_n3136_ws21.py
how_much_history_section.tex
tasks/R1-Mamba3/session-chain.md
```

## Already Verified Starting Evidence

Use these as starting points, but re-open the files and verify line numbers before final reporting.

- `run_train.py` exposes `--mamba3_rope_fraction`, default `0.5`.
- `node_wrapper.sh` forwards `MAMBA3_ROPE_FRACTION` into `--mamba3_rope_fraction`.
- `train_full_autoreg.batch` defaults `MAMBA3_ROPE_FRACTION="${MAMBA3_ROPE_FRACTION:-0.5}"`.
- `s5/mamba3.py` documents `rope_fraction` as “fraction of d_state used for RoPE (0.5 or 1.0)”, defaults it to `0.5`, then computes:

```text
split_tensor_size = int(N * rope_fraction)
num_rope_angles = split_tensor_size // 2
```

- `s5/mamba3_jax.py` applies RoPE to Q/K using cumulative `dt`-scaled angles and pads the unrotated dimensions through unchanged for partial RoPE.
- `phase_a_sweep.sh` explicitly swept `rope_fraction in {0.0, 0.25, 0.75, 1.0}` against the `0.5` baseline.
- `phase_b_sweep.sh` and `phase_b_extended_sweep.sh` say they locked `rope=1.0` after Phase A.
- Existing extraction scripts compare `rope=0.5` and `rope=1.0`; do not assume their printed conclusions are correct until the underlying result directories are checked.

Session-chain clues:

```text
2026-04-22 session:
aa226a66-7b54-4e6b-b396-a85b0ba67b26

2026-05-12 session:
7bb4e925-b0e0-408f-b568-866a76546de5
```

## Harness Work Plan

### 1. Source Audit

Create a concise table with:

```text
file | knob/default | where used | effect on tensor shape | evidence line
```

Minimum files to audit:

```text
run_train.py
node_wrapper.sh
train_full_autoreg.batch
lob/init_train.py
s5/mamba3.py
s5/mamba3_jax.py
s5/flops.py
```

Answer mechanically:

- Is the 50% default real?
- Is the fraction applied over `d_state`, feature pairs, or “planes”?
- Which dimensions pass through unchanged?
- Does `rope_fraction=1.0` rotate all Q/K state planes in the local code?

### 2. Minimal Executable Shape Test

Add or run a tiny non-training check that instantiates the Mamba3 layer or directly checks the shape arithmetic for several values:

```text
N = 128
rope_fraction = 0.0, 0.25, 0.5, 0.75, 1.0
expected split_tensor_size
expected num_rope_angles
expected rotated pair count
expected pass-through pair count
```

Preferred deliverable:

```text
agent_outputs/r1_mamba3_rope_shape_check_YYYYMMDD.md
```

or a small script if existing imports are clean:

```text
agent_outputs/check_mamba3_rope_shapes.py
```

Keep it CPU/JAX-light if possible; do not require SLURM.

### 3. Paper / Upstream Diff

Verify against primary sources.

Required checks:

- Mamba-3 paper linked by Alex: identify the exact section/equations/code-level description of RoPE and whether it implies full rotation.
- Online implementation Alex mentioned:

```text
https://github.com/rishikksh20/mamba3-pytorch
```

Check whether it defaults to partial RoPE and whether it exposes full RoPE as an option.

Record the finding as:

```text
source | default | full-RoPE option? | evidence | interpretation
```

Use official paper / repository evidence. If browsing is needed, cite links in the final evidence pack. Do not overquote.

### 4. Existing Experiment Evidence

Audit existing R1 RoPE experiments, without rerunning training:

```text
phase_a_sweep.sh
phase_b_sweep.sh
phase_b_extended_sweep.sh
submit_rope_confirmation.sh
submit_rope_dstate_46m.sh
extract_rope_downstream.py
extract_rope_tex_numbers.py
extract_rope_conf_latest.py
extract_n3136_ws21.py
how_much_history_section.tex
```

Find and verify:

- Which checkpoints/results used `rope=0.5`.
- Which used `rope=1.0`.
- Whether Phase A really made `rope=1.0` the follow-up default.
- Whether the paper/model checkpoint referred to as “paper 46M” or “paper 78M” used `rope=0.5`.
- Whether any written claim says partial RoPE is for robustness/content preservation versus simply inherited from an implementation.

Do not trust comments alone if result files/logs exist. Prefer logs, command lines, checkpoint configs, W&B metadata, or extraction output tied to result directories.

### 5. Bug Classification

Classify the issue using this decision table:

```text
case | condition | classification
local default is 0.5, paper says 1.0, no local experiments justify 0.5 | likely undocumented divergence / possible bug
local default is 0.5, paper allows fraction or upstream default is 0.5 | implementation choice, not necessarily bug
local default is 0.5, but R1 later sweeps/locks 1.0 | old default remains stale; experimental recipe may have moved to full RoPE
local local implementation's `1.0` still leaves planes unrotated | actual implementation bug
local shape/test code contradicts comments | documentation bug or implementation bug depending on behavior
```

Be precise: “possible bug” is acceptable only if a concrete invariant fails.

### 6. Co-Scientist Output

Write the final evidence pack locally:

```text
/lus/lfs1aip2/projects/public/u6gb/R1_MAMBA3_ROPE_HARNESS_REPORT.md
```

The report should include:

- One-page executive finding.
- Evidence table with file paths and line references.
- Tiny shape-test output.
- Paper/upstream comparison.
- Existing experiment comparison.
- Bug classification.
- Open questions for Alex or the R1 owner.
- A short proposed reply draft to Alex, clearly labeled as a draft.

The draft reply should not be casual speculation. It should say what was verified, what remains uncertain, and whether the recommended next action is changing the default, documenting the divergence, or running a controlled `0.5` vs `1.0` confirmation.

### 7. Image2 Figure Generation

Use `image2` to generate at least three figures for the final Notion subpages and local report:

```text
figures/r1_mamba3_rope_tensor_geometry.png
figures/r1_mamba3_rope_evidence_pipeline.png
figures/r1_mamba3_rope_bug_classification.png
```

Figure requirements:

- 图 1：展示 `d_state=128` 下 `rope_fraction=0.5` 与 `rope_fraction=1.0` 的 Q/K state planes：哪些维度旋转、哪些维度保持 pass-through。
- 图 2：展示证据链：local code audit -> shape test -> paper/upstream diff -> existing experiment evidence -> bug classification -> Alex reply draft。
- 图 3：展示 bug classification decision table，从“paper says full RoPE?”、“local default 0.5?”、“experiments justify?”、“rope=1.0 really rotates all planes?”流向最终分类。
- 所有图内文字使用中文，但保留必要英文术语如 `RoPE`, `d_state`, `Q/K`, `rope_fraction`。
- 生成图像后，把图像路径、生成 prompt、生成时间写入本地报告和 Notion 子页面。
- 如果 `image2` 支持多轮编辑，先生成草图，再根据报告中的最终分类更新图 3，避免图示与结论不一致。

Suggested `image2` prompt skeleton:

```text
生成一张清晰的科研报告图，中文标注，主题是 Mamba3 RoPE fraction 的张量几何。
画出 d_state=128 的 Q/K state planes，被分成 64 个二维旋转平面。
左侧显示 rope_fraction=0.5：前 32 个平面参与 RoPE 旋转，后 32 个平面 pass-through。
右侧显示 rope_fraction=1.0：64 个平面全部参与 RoPE 旋转。
风格：简洁、论文附图、浅色背景、矢量感、无装饰性元素。
```

### 8. Notion Subpage Publication

After the local report and figures are complete, publish the final output into Notion under the original page:

```text
parent page: https://www.notion.so/36612c4568fd809d9082d4cf23663001
title: R1 Mamba3 RoPE Fraction Harness Report
```

Create structured Notion child pages, not one giant unstructured page:

```text
R1 Mamba3 RoPE Fraction Harness Report
R1 Mamba3 RoPE Source Audit
R1 Mamba3 RoPE Paper and Upstream Diff
R1 Mamba3 RoPE Existing Experiment Evidence
R1 Mamba3 RoPE Figure Pack
R1 Mamba3 RoPE Reply Draft for Alex
```

Minimum Notion content:

- 主报告子页：中文 executive finding、bug classification、最终建议、所有子页链接。
- Source Audit 子页：本地代码表格，包含文件路径、行号、knob/default、shape effect。
- Paper and Upstream Diff 子页：paper / repo 证据、链接、差异解释。
- Existing Experiment Evidence 子页：Phase A/B/confirmation 结果、checkpoint/result 证据。
- Figure Pack 子页：嵌入或链接 `image2` 生成的三张图，附 prompt 和生成时间。
- Reply Draft 子页：给 Alex 的英文回复草稿，以及中文解释版。

Publication rules:

- Notion 子页面正文用中文；Alex reply draft 可用英文，但要配中文说明。
- 每个 Notion 子页开头写明数据核查时间和本地报告路径。
- 回写完成后，在原始 Notion 页追加一个简短中文索引，列出新建子页链接。
- 如果 Notion 工具不可用，不要假装已发布；在本地报告中记录“Notion 发布阻塞”，并给出待执行的 create-pages/update-page 操作清单。

## Acceptance Criteria

- A local report exists at:

```text
/lus/lfs1aip2/projects/public/u6gb/R1_MAMBA3_ROPE_HARNESS_REPORT.md
```

- The report references exact local files and line numbers.
- It includes a minimal executable or computed shape check for the RoPE fraction behavior.
- It distinguishes local implementation behavior from paper/upstream behavior.
- It includes `image2`-generated figures or an explicit recorded fallback/blocker.
- It publishes the final report into Notion child pages under the original Notion page, or records a concrete Notion-publication blocker.
- It does not launch expensive jobs.
- It contains a final answer draft only after the evidence is assembled.

## Suggested Commands

```bash
cd /lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_R1_Mamba3

rg -n "mamba3_rope_fraction|MAMBA3_ROPE_FRACTION|rope_fraction|num_rope_angles|split_tensor_size|apply_rope|angle_dt_cumsum" \
  run_train.py node_wrapper.sh train_full_autoreg.batch lob/init_train.py s5/mamba3.py s5/mamba3_jax.py s5/flops.py

sed -n '140,155p' run_train.py
sed -n '50,110p' s5/mamba3.py
sed -n '60,125p' s5/mamba3_jax.py
sed -n '205,230p' s5/mamba3_jax.py
sed -n '1,190p' phase_a_sweep.sh
sed -n '1,220p' phase_b_sweep.sh
sed -n '1,140p' phase_b_extended_sweep.sh
```

## Important Constraint

The user wants harness engineering work as a co-scientist. Do not stop with a direct prose answer to the RoPE question. Build the evidence pack first.
