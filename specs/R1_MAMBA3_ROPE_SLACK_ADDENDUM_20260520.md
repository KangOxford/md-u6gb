# R1 Mamba3 RoPE Slack Addendum

核查时间：2026-05-20T22:10Z

本 addendum 修正 `R1_MAMBA3_ROPE_HARNESS_REPORT.md` 的第一版结论。新输入是 Slack 线程中 Aramis 的回复：

```text
It's the default in the mamba3 paper, and we swept it — rope_frac=0.5 was the best performing on lobbench
(Rope_frac controls how many ‘channels’ get RoPE)
There's a section on this in my thesis I think, I'll send it
```

## 结论更新

Aramis 的方向基本正确，但需要更精确地表述证据层级：

1. `rope_fraction=0.5` 确实是 Mamba3 official/reference implementation 的默认，不是 R1 本地 typo。
2. arXiv Mamba-3 paper 的数学正文没有清楚暴露 `rope_fraction` partial-RoPE knob；它把 complex SSM 等价成 B/C data-dependent RoPE，Alex 从 paper 读成 full B/C RoPE 是合理的。
3. R1 本地论文/章节把 0.5 称为 paper default，并在 46M paper-scale confirmation 后选择 `rope=0.5` 作为 scaling-law ladder 和后续 ablation 的默认。
4. “0.5 best on LOBbench”要加限定：8M 上 0.5 不是 WS/IC/Sharpe winner；46M paper-scale 上 0.5 输给 0.25 的 WS/IC，但赢 KS、L1、DirAcc、RV50，并且是唯一 never-worst 的 metric-robust cell。因此它是 robust default，不是单一 headline WS winner。

## 新证据

| 来源 | 证据 | 解释 |
|---|---|---|
| official `state-spaces/mamba` Mamba3 module | `rope_fraction=0.5`; assert `[0.5, 1.0]`; `split_tensor_size=int(d_state * rope_fraction)` | official implementation 默认 partial RoPE |
| local `rope_ablation_section.tex:12` | 78M baseline uses `rope_fraction=0.5` | R1 的 78M +RoPE baseline 是 0.5 |
| local `rope_ablation_section.tex:161-180` | 8M sweep over `{0.0,0.25,0.5,0.75,1.0}` | 8M landscape non-monotonic；0.5 worst on Uncon but best on Cond |
| local `rope_ablation_section.tex:208-229` | 46M retest `{0.25,0.5,1.0}`; choose `0.5` for scaling-law ladder and subsequent ablations | paper-scale 选择 0.5 是实验决策 |
| local `scaling_law_runs.md:15-20` | common config lists `rope_fraction=0.5` | scaling-law registry confirms 0.5 default |
| local `submit_rope_confirmation.sh:1-8` | compares 0.25/1.0 against 0.5 paper baseline | 0.5 was baseline before confirmation runs |

## 修正后的 Alex 回复草稿

```text
I checked the implementation and the local ablation notes. The 50% behavior is real, but it is not a tensor-shape bug. `rope_frac` controls what fraction of the B/C state channels get the data-dependent RoPE; with `d_state=128`, `rope_frac=0.5` rotates 32 of the 64 even/odd planes and leaves the other 32 as position-free channels. Setting `rope_frac=1.0` does rotate all planes in our code.

The reason it is 0.5 by default is that this follows the Mamba-3 reference/official implementation default. The paper text is a bit easy to misread because the math describes RoPE on B/C without spelling out this implementation knob, so your “paper looks like full RoPE” reading is reasonable.

For R1, we did sweep it. The short version is: 8M was non-monotonic and did not favor 0.5 on headline WS; then a 46M paper-scale retest compared 0.25 / 0.5 / 1.0. At 46M, 0.25 won WS/IC, but 0.5 won KS, L1, DirAcc, and realized-volatility match, and was the only setting that was never worst. That is why the scaling-law ladder and later ablations use 0.5 as the robust default. So I would call this an intentional/default hyperparameter choice that should be documented better, not a bug.
```

## 与第一版报告的差异

第一版报告说“R1 后续 recipe 已从 0.5 迁移到 1.0”。这只对早期 Phase B script 注释成立；它不是最终 paper-scale recipe。更完整的时间线是：

```text
early Phase A/B: 8M sweep, scripts temporarily mark 1.0 as Phase A winner
later paper-scale confirmation: 46M retest compares 0.25/0.5/1.0
final local paper/scaling-law choice: 0.5 as metric-robust default
```

所以最终建议不应是“改默认到 1.0”，而应是“保留 0.5 可以成立，但必须把 `rope_fraction` 的含义和 paper 正文没有显式说明 partial knob 这点写清楚”。
