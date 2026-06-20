# R1 Mamba3 RoPE Fraction Harness Report

核查时间：2026-05-20T21:52:39Z  
本地 harness 目录：`/lus/lfs1aip2/projects/public/u6gb/r1_mamba3_rope_harness`  
原始 Notion 页：<https://www.notion.so/36612c4568fd809d9082d4cf23663001>

## 执行摘要

Alex 的问题成立：`exp_R1_Mamba3` 的本地默认确实是 `rope_fraction=0.5`，这意味着在 `d_state=128` 时只旋转 32/64 个二维 Q/K state planes，剩余 32 个 plane 作为内容通道 pass-through。

当前证据不支持把它归类为“`rope_fraction=1.0` 实现错误”。最小 shape check 证明本地 `rope_fraction=1.0` 会旋转全部 64/64 个二维 plane，没有残留未旋转 plane。

2026-05-20T22:10Z 追加核查：Aramis 的 Slack 回复提供了关键线索。官方 `state-spaces/mamba` Mamba3 module 和 Alex 提到的 PyTorch repo 都把 `rope_fraction=0.5` 作为实现默认，并允许 `1.0`。本地 R1 的 `rope_ablation_section.tex` 也明确把 0.5 作为 paper/scaling-law default；46M paper-scale retest 之后选择 `rope=0.5` 作为 scaling-law ladder 和后续 ablation 的默认。

因此修正后的分类是：**不是 tensor bug；是官方实现默认 + R1 paper-scale 实验选择，但 arXiv Mamba-3 数学表述本身没有清楚暴露 partial-RoPE knob**。需要避免两个过度简化：不能说 Alex 误读，因为 paper 公式确实看起来像对 B/C 全部应用 RoPE；也不能说 R1 应改成 1.0，因为本地 46M retest 和 78M paper baseline 支持 0.5 作为更稳健的 LOBbench operating point。

建议动作：答复 Alex 时说清楚三层事实：**(1) 代码默认 0.5 是有意的 `rope_fraction` hyperparameter；(2) official/reference code 默认也是 0.5，paper 正文更像 full B/C RoPE、没有把这个 knob 解释清楚；(3) R1 的 8M sweep 不支持 0.5 作为 WS winner，但 46M paper-scale retest 让 0.5 成为 metric-robust default，因此当前 R1 default 不是 typo。**

## 产物

| 产物 | 路径 |
|---|---|
| 本报告 | `/lus/lfs1aip2/projects/public/u6gb/R1_MAMBA3_ROPE_HARNESS_REPORT.md` |
| shape check 脚本 | `/lus/lfs1aip2/projects/public/u6gb/r1_mamba3_rope_harness/agent_outputs/check_mamba3_rope_shapes.py` |
| shape check 输出 | `/lus/lfs1aip2/projects/public/u6gb/r1_mamba3_rope_harness/agent_outputs/r1_mamba3_rope_shape_check_20260520.md` |
| RoPE 结果抽取脚本 | `/lus/lfs1aip2/projects/public/u6gb/r1_mamba3_rope_harness/agent_outputs/summarize_rope_results.py` |
| RoPE 结果抽取输出 | `/lus/lfs1aip2/projects/public/u6gb/r1_mamba3_rope_harness/agent_outputs/rope_results_summary_20260520.md` |
| 图 1 | `/lus/lfs1aip2/projects/public/u6gb/r1_mamba3_rope_harness/figures/r1_mamba3_rope_tensor_geometry.png` |
| 图 2 | `/lus/lfs1aip2/projects/public/u6gb/r1_mamba3_rope_harness/figures/r1_mamba3_rope_evidence_pipeline.png` |
| 图 3 | `/lus/lfs1aip2/projects/public/u6gb/r1_mamba3_rope_harness/figures/r1_mamba3_rope_bug_classification.png` |

## `image2` 状态

goal 要求用 `image2` 生成 figures。当前 shell 中 `image2` 不存在：

```text
which: no image2 in PATH
```

因此按 goal 的 fallback 规则执行：使用本地可复现 matplotlib 绘图脚本生成 PNG。为了保证中文渲染，下载并使用开源字体：

```text
/lus/lfs1aip2/projects/public/u6gb/r1_mamba3_rope_harness/assets/NotoSansCJKsc-Regular.otf
```

生成脚本：

```text
/lus/lfs1aip2/projects/public/u6gb/r1_mamba3_rope_harness/agent_outputs/generate_rope_figures.py
```

图像尺寸验证：

```text
r1_mamba3_rope_bug_classification.png (1919, 1006)
r1_mamba3_rope_evidence_pipeline.png (2007, 867)
r1_mamba3_rope_tensor_geometry.png (1849, 936)
```

## 图像 prompts / 生成说明

图 1 prompt：

```text
生成一张清晰的科研报告图，中文标注，主题是 Mamba3 RoPE fraction 的张量几何。
画出 d_state=128 的 Q/K state planes，被分成 64 个二维旋转平面。
左侧显示 rope_fraction=0.5：前 32 个平面参与 RoPE 旋转，后 32 个平面 pass-through。
右侧显示 rope_fraction=1.0：64 个平面全部参与 RoPE 旋转。
风格：简洁、论文附图、浅色背景、矢量感、无装饰性元素。
```

图 2 prompt：

```text
生成一张中文科研流程图，主题是 R1 Mamba3 RoPE fraction 的证据链。
流程从源码审计、最小 shape check、paper/upstream diff、现有实验结果、bug 分类，到 Alex 回复草稿。
强调“不从直觉回答，每个结论必须有文件路径、行号、运行输出或外部来源链接”。
```

图 3 prompt：

```text
生成一张中文 bug classification decision tree。
节点包括：paper 是否显示 full RoPE、本地默认是否 0.5、本地 1.0 是否全旋转、实验是否支持继续用 0.5、upstream 是否也默认 0.5。
最终分类是：不是实现错误；更像遗留默认 + 未文档化 paper divergence。
```

## Source Audit

| 文件 | knob/default | 使用位置 | shape effect | 证据 |
|---|---|---|---|---|
| `/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_R1_Mamba3/run_train.py` | `--mamba3_rope_fraction`, default `0.5` | CLI 参数 | 默认把 `d_state` 的 50% dims 送入 RoPE | lines 140-150 |
| `/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_R1_Mamba3/node_wrapper.sh` | `MAMBA3_ROPE_FRACTION` | 环境变量转 CLI | 训练 batch 可覆盖默认 | lines 538-543 |
| `/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_R1_Mamba3/train_full_autoreg.batch` | `MAMBA3_ROPE_FRACTION="${MAMBA3_ROPE_FRACTION:-0.5}"` | batch 默认值 | 未显式覆盖时仍走 0.5 | lines 640-645 |
| `/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_R1_Mamba3/lob/init_train.py` | `getattr(args, ..., 0.5)` | 传入 `init_Mamba3SSM` | CLI/default 进入模型构造 | lines 351-385 |
| `/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_R1_Mamba3/s5/mamba3.py` | `rope_fraction: float = 0.5` | 模型字段和 projection size | `split_tensor_size=int(N*rope_fraction)`；`num_rope_angles=split/2` | lines 58-69, 98-109 |
| `/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_R1_Mamba3/s5/mamba3_jax.py` | `apply_rope` | Q/K 旋转 | angle 少于 plane 数时 pad `cos=1/sin=0`，剩余 dims pass-through | lines 65-100, 216-218, 246-247 |
| `/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_R1_Mamba3/s5/flops.py` | `rope_fraction` | FLOPs 估计 | `rope_dim=int(N*rope_fraction)//2` | lines 100-120 |

结论：

- 50% 默认是真的，不是 Alex 误读。
- fraction 作用在 `d_state` 维度上；由于 RoPE 是二维 even/odd plane，真正 angle 数是 `int(N*fraction)/2`。
- `rope_fraction=0.5`：`N=128` 时旋转 64 dims = 32 planes，剩余 64 dims = 32 planes pass-through。
- `rope_fraction=1.0`：`N=128` 时旋转 128 dims = 64 planes，没有 pass-through plane。

## 最小 shape check

脚本输出：

```text
| N | rope_fraction | split_tensor_size | num_rope_angles | total_planes | rotated_planes | pass_through_planes |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.00 | 0 | 0 | 64 | 0 | 64 |
| 128 | 0.25 | 32 | 16 | 64 | 16 | 48 |
| 128 | 0.50 | 64 | 32 | 64 | 32 | 32 |
| 128 | 0.75 | 96 | 48 | 64 | 48 | 16 |
| 128 | 1.00 | 128 | 64 | 64 | 64 | 0 |

Invariant checks:
- all invariants passed
```

这证明 `rope_fraction=1.0` 在本地 arithmetic 层面确实全旋转，不是“配置写了 1.0 但仍只旋转一半”的实现错误。

## Paper / Upstream Diff

### Mamba-3 paper

外部来源：Mamba-3 paper, arXiv 2603.15569, <https://arxiv.org/pdf/2603.15569>

相关证据：

- paper 在 Proposition 3 中把 complex SSM 等价成对 SSM 的 B/C components 应用 data-dependent rotary embedding；公式展示 `R_i` 累乘作用到 B/C。
- paper Figure 2 的 Mamba-3 block 图示标注 `B C -> RoPE`，并在 architecture 段落说 imaginary-valued Θ 通过 RoPE trick 计算。
- paper 的数学表述和图示没有暴露 `rope_fraction` 或 partial-RoPE knob；本次用 PDF text search 检查 `rope/fraction/default/half/50`，未在正文中找到显式 `rope_fraction` 配置说明。从 Alex 的角度读成 full RoPE 是合理的。

保守解释：paper 正文支持“理论表述是 B/C RoPE”，但它不是逐项列出实现 hyperparameter 的配置文件。0.5 默认应归因到 official/reference implementation 和 R1 paper-scale 实验选择，而不是只从 arXiv 正文推出。

### Official Mamba3 implementation

外部来源：`state-spaces/mamba`, <https://github.com/state-spaces/mamba>

相关证据：

- README 的 Mamba-3 用法指向 `mamba_ssm/modules/mamba3.py`。
- official `mamba3.py` 默认参数是 `rope_fraction=0.5`。
- official `mamba3.py` assert `rope_fraction in [0.5, 1.0]`。
- official `mamba3.py` 使用同样的 shape 逻辑：`split_tensor_size = int(d_state * rope_fraction)`，`num_rope_angles = split_tensor_size // 2`。

解释：Aramis 说“这是 Mamba3 默认”在 implementation 层面是对的。更精确说法是 official implementation default，而不是 arXiv 正文明确解释了 partial-RoPE default。

### Alex 提到的 PyTorch repo

外部来源：`rishikksh20/mamba3-pytorch`, <https://github.com/rishikksh20/mamba3-pytorch>

相关证据：

- README 说明该 repo 是 Mamba-3 的 clean readable implementation，并说明 RoPE applied to B/C projections。
- `mamba3.py` 里明确写 `rope_fraction` 是 state dims 中使用 rotary 的比例。
- 该 repo 默认 `rope_fraction=0.5`，并 assert 只支持 `{0.5, 1.0}`。
- 注释写明：`0.5 -> first d_state/2 dims rotate -> d_state/4 angles`；`1.0 -> all d_state dims rotate -> d_state/2 angles`。

保守解释：R1 的 `0.5` 默认与 official/reference implementation 一致。它说明 partial RoPE 是一个实现选择，不是 R1 独有 typo；但仍然应该在 R1 文档中解释 paper 正文为什么看起来像 full B/C RoPE。

## Existing Experiment Evidence

### 2026-05-20T22:10Z 修正

前一版只看到早期 Phase B scripts，因此把 `rope=1.0` 当作 R1 后续 recipe。新增 Slack 线索后重新检查 `rope_ablation_section.tex`、`submit_rope_confirmation.sh` 和 `scaling_law_runs.md`，结论需要修正：

- 早期 8M Phase A/B 确实一度把 `1.0` 当作 winner。
- 后续 46M paper-scale confirmation 重新比较 `{0.25, 0.5, 1.0}`。
- `rope_ablation_section.tex:229` 明确写最终选择 `rope=0.5` for scaling-law ladder and every subsequent ablation。
- `scaling_law_runs.md:19` 的 common config 也确认 scaling-law registry 使用 `rope_fraction=0.5`。

因此 Phase B scripts 是历史中间态，不是最终 R1 paper-scale default。

### Phase A script design

`phase_a_sweep.sh` 明确说 baseline 是 `rope=0.5`，同时扫 `{0.0, 0.25, 0.75, 1.0}`。脚本注释还写“rope_fraction 对 LOB data genuinely unknown”，因此这是一个真实实验变量，不是死代码。

证据：

```text
phase_a_sweep.sh lines 4-6, 18-20, 152-164
```

### Phase B / extended recipe drift (historical)

`phase_b_sweep.sh` 和 `phase_b_extended_sweep.sh` 都把 `rope_fraction=1.0` 标为 Phase A winner 并锁定。这说明早期 Phase A/B 曾经使用 `1.0`，但后续 46M confirmation 和 scaling-law registry 已经覆盖这个中间结论。

证据：

```text
phase_b_sweep.sh lines 17-20, 140-143, 170-173
phase_b_extended_sweep.sh lines 15-18, 92-101
```

### 历史报告

`agent_outputs/phase_a_rope_lobbench_results.md` 的结论是 full RoPE 在 h250/h500 的 WS-21 最优，并建议 Phase B 锁定 `rope_fraction=1.0`。`agent_outputs/phase_b_memory_cluster_results.md` 明确 Phase B 训练配置为 `rope=1.0 (Phase A winner)`。

### 重新抽取的 public mirror 结果

抽取脚本：`/lus/lfs1aip2/projects/public/u6gb/r1_mamba3_rope_harness/agent_outputs/summarize_rope_results.py`

结果输出：`/lus/lfs1aip2/projects/public/u6gb/r1_mamba3_rope_harness/agent_outputs/rope_results_summary_20260520.md`

8M Phase A n=3136 结果：

| run | WS-21 | 95% CI | IC Spearman | DirAcc | Sharpe |
|---|---:|---|---:|---:|---:|
| rope=0.00 | 0.2014 | [0.1976, 0.2052] | 0.1322 | 0.5590 | 0.0982 |
| rope=0.25 | 0.1215 | [0.1177, 0.1271] | 0.1119 | 0.5584 | 0.1061 |
| rope=0.50 | 0.1975 | [0.1931, 0.2029] | 0.1018 | 0.5409 | 0.0739 |
| rope=0.75 | 0.2131 | [0.2089, 0.2174] | 0.1129 | 0.5536 | 0.0895 |
| rope=1.00 | 0.1225 | [0.1184, 0.1263] | 0.1141 | 0.5522 | 0.0932 |

可访问的 46M confirmation 结果：

| run | WS-21 | 95% CI | n_feat |
|---|---:|---|---:|
| rope=0.25 step 13350 | 0.1159 | [0.1123, 0.1196] | 20 |
| rope=1.00 step 7400 | 0.0866 | [0.0826, 0.0922] | 19 |
| rope=1.00 step 27510 | 0.0916 | [0.0875, 0.0956] | 21 |
| rope=0.50 d256 step 13270 | 0.0791 | [0.0750, 0.0844] | 20 |

### 本地论文章节中的 46M paper-scale retest

`rope_ablation_section.tex:208-229` 给出更完整的 paper-scale 选择逻辑：

| rope_fraction | WS | KS | L1 | IC Pearson | Sharpe | DirAcc | RV50 | 解释 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.25 | 0.060 | 0.097 | 0.167 | 0.136 | 0.097 | 0.552 | 0.89 | WS/IC winner，但 under-dispersed |
| 0.50 | 0.068 | 0.080 | 0.157 | 0.092 | 0.086 | 0.554 | 1.00 | KS/L1/DirAcc/RV50 winner，唯一 never-worst |
| 1.00 | 0.088 | 0.112 | 0.181 | 0.088 | 0.093 | 0.551 | 0.83 | 46M distribution metrics last |

这解释了 Aramis 的“0.5 best on LOBbench”：严格说不是 0.5 赢所有指标，而是 0.5 在 paper-scale 多指标 LOBbench 上最稳健，因此被选作 scaling-law 和后续 ablation 默认。

注意：历史 extraction scripts hard-code `/projects/s5e/lob_pipeline`，当前会话不可读；本次重新抽取使用 public mirror：

```text
/lus/lfs1aip2/projects/public/s5e/quant_team/lob_pipeline
```

## Bug Classification

| 条件 | 证据 | 判定 |
|---|---|---|
| 本地默认是 0.5 | CLI、batch、model 默认全部为 0.5 | true |
| arXiv paper 正文看起来是 full B/C RoPE | 公式/图示无 partial knob，显示 B/C RoPE | true |
| official/reference implementation 默认是 0.5 | `state-spaces/mamba` 与 rishikksh20 repo 默认 0.5，可选 1.0 | true |
| 本地 `1.0` 是否全旋转 | shape check：64/64 planes rotated | true |
| R1 paper-scale 实验是否支持默认 0.5 | 46M retest 后 `rope_ablation_section.tex` 选择 0.5 作为 scaling-law ladder/default | true |

分类：

```text
不是实现错误：rope_fraction=1.0 在本地会全旋转。
不是 Alex 误解：默认 0.5 真实存在，paper 正文也确实没有清楚说明 partial knob。
最准确：official/reference default + R1 paper-scale metric-robust choice；文档需要把 paper 正文和实现默认的差异写清楚。
```

建议工程动作：

1. 保留 `MAMBA3_ROPE_FRACTION=0.5` 可以成立，因为它有 official/reference default 和 46M paper-scale R1 证据支持。
2. 在 README / config 注释里补一句：`rope_fraction` 控制 B/C state channels 中应用 data-dependent RoPE 的比例；0.5 会保留一半 position-free channels，1.0 会全旋转。
3. 文档里区分 arXiv Mamba-3 数学正文、official implementation default、R1 LOBbench paper-scale default，避免把 “paper full RoPE” 和 “code default 0.5” 混为一个说法。
4. 给 Alex 的回复中明确说明 8M 与 46M sweep 的 ranking 不同；0.5 是 metric-robust default，不是 8M headline WS winner。

## Open Questions

- “original Mamba3 paper recommended 0.5”这句话需要追到 official paper appendix / code release 的精确出处；当前 arXiv 正文搜索未看到 `rope_fraction` 这个显式 knob。
- R1 文档应决定术语：把 0.5 叫 “Mamba3 official implementation default” 更稳，比直接叫 “paper default” 更不容易引发误解。
- 46M `rope=0.5 d256` 在 public mirror 中 WS-21 很强，说明 rope fraction 与 `d_state` 有交互；如果要进一步优化 production choice，应做同规模、同 `d_state`、同训练步数的 controlled comparison。

## Draft Reply To Alex

```text
I checked the implementation and the local ablation notes. The 50% behavior is real, but it is not a tensor-shape bug. `rope_frac` controls what fraction of the B/C state channels get the data-dependent RoPE; with `d_state=128`, `rope_frac=0.5` rotates 32 of the 64 even/odd planes and leaves the other 32 as position-free channels. Setting `rope_frac=1.0` does rotate all planes in our code.

The reason it is 0.5 by default is that this follows the Mamba-3 reference/official implementation default. The paper text is a bit easy to misread because the math describes RoPE on B/C without spelling out this implementation knob, so your “paper looks like full RoPE” reading is reasonable.

For R1, we did sweep it. The short version is: 8M was non-monotonic and did not favor 0.5 on headline WS; then a 46M paper-scale retest compared 0.25 / 0.5 / 1.0. At 46M, 0.25 won WS/IC, but 0.5 won KS, L1, DirAcc, and realized-volatility match, and was the only setting that was never worst. That is why the scaling-law ladder and later ablations use 0.5 as the robust default. So I would call this an intentional/default hyperparameter choice that should be documented better, not a bug.
```

中文解释版：

```text
Alex 的观察是对的。本地默认 0.5 确实只旋转一半 Q/K state planes；这不是他误读。当前证据显示 `rope_fraction=1.0` 实现本身可以全旋转，所以不是“1.0 也只转一半”的实现 bug。修正后的解释是：0.5 来自 official/reference implementation default，并且 R1 46M paper-scale retest 后把它选成 metric-robust default；但 paper 正文没有把 partial-RoPE knob 讲清楚，所以需要补文档。
```

## Sources

- Mamba-3 paper: <https://arxiv.org/pdf/2603.15569>
- rishikksh20 Mamba3 PyTorch repo: <https://github.com/rishikksh20/mamba3-pytorch>
- Raw repo file: <https://raw.githubusercontent.com/rishikksh20/mamba3-pytorch/master/mamba3.py>
