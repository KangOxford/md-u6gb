# Findings

- 2026-07-27 12:05 UTC: The supplied Notion URL resolves to `⚖️ Rebuttal NeurIPS 2026 (1)` (page ID `3aa12c45-68fd-80ab-89b5-d0d597bad64c`).
- The page asks for Chinese rebuttal work based on update material, with long answers in subpages and short answers in callouts; Codex text should be blue and Claude text red.
- The supplied deep anchor `3aa12c45-68fd-8040-bca1-fc7613009b23` is the paper-source callout. It contains an Overleaf source URL, sensitive credentials that were not reproduced, and the attachment `quant_neurips26_clean_(18).pdf`.
- The PDF bytes were preserved at `notion_fetches/rebuttal_neurips_2026_20260727T1210Z/assets/quant_neurips26_clean_(18).pdf`: 598,646 bytes, 31 pages, SHA256 `68bdf56e8a4b6f6247907aeb410309d1aa4bd1d7cfa28d99d71953ffb0ae045f`.
- The paper was cloned into the requested conflict-isolated directory `/lus/lfs1aip2/projects/public/u6gb/Overleaf_codex` and checked out on branch `codex-rebuttal`.
- Twelve reviewer answers were inserted and re-fetched one at a time as blue Notion callouts: pXiP Q1-Q5, WHZQ Q1-Q4, and 8P5h Q1-Q3. The final page audit found exactly 12 `Codex 回答` markers and 12 blue callouts.
- Submitted-sweep sensitivity analysis uses the historical 10-run / 1,846-checkpoint blobs, not the later 0.2M-350M protocol. The baseline is `alpha=0.3711`, `beta=1.3325`, `E=0.50592`, with frontier slopes `0.8487/0.2364`.
- A common-D refit on `[4.472e9, 2.817152e10]` tokens gives `alpha=0.9935`, `beta=1.4336`, `E=0.53115`; this is support sensitivity on logged training loss, not held-out validation.
- The late half of common-D gives `alpha=1.2132`, `beta=1.1405`; full-support run balancing gives `beta=1.1823`, while common-D plus run balancing gives `beta=1.4968`. The magnitude is weighting/support-sensitive even though these point estimates remain above one.
- Fixing `beta=0.28` increases the same log-Huber objective by 50.16%, raw-loss RMSE by 80.43%, and changes the frontier slopes to `0.1764/0.8413`. Fixing `beta=1.0` increases the objective by only 6.11%, so the submitted interval rejects 0.28 more clearly than it identifies `beta>1`.
- The published TradeFM coordinate is 524.4M parameters and 42.8B cumulative trade-event tokens; MarS's largest order model is 1.02B parameters and 32B order tokens. Once message semantics are respected, these do not support a categorical “undertrained” claim; architecture-specific compute profiles and comparable tokenization are also unavailable.
- The only verified LR sensitivity is a single 78M, seed-5 sweep. At step 10,800, CE for multipliers `0.25/0.5/1/2/4` is `0.572858/0.577215/0.570238/0.583513/0.594320`; the central range spans about 2.3%, not 0.5%.
- No defensible matched Transformer scaling fit exists. Current held-out rows are Mamba-3 only, and the historical Transformer-v3 snapshot has zero valid completed scaling points.
- Removing the lowest-N point from every IsoFLOP slice changes slopes from `0.7295/0.4234` to `0.6916/0.4055`, but only 5 of 9 slices remain bracketed by at least two points per side. The latter is a reduced-subset sensitivity, not a like-for-like robustness confirmation.
- The sensitivity figure now distinguishes the five bracketed drop-lowest candidates from the four unbracketed candidates; the earlier figure legend had incorrectly labelled every red candidate as unbracketed.

- 2026-07-27T13:04:27Z: beta panorama across protocols: 1.333 (full-curve) / 0.944 (cutoff) / 0.616 (tail-25) / 0.917 (SP500 v6 interim, long-D) / 0.978 (Jan-2026 held-out CE, new fit). Conclusion for rebuttal: beta>1 is a fitting-protocol artifact; corrected range 0.9-1.0 stays ~3x LM value; core claim survives, headline toned down.
F106 UTC 2026-07-27T13:12:03Z: unseen-val 方案代码级验证（回应用户复问）：s5e_mamba3 证据链 lob/train.py:94-96 create_lobster_prediction_dataset(seed=args.jax_seed) → dataloading.py:178/199 DistributedSampler(seed=seed)——数据顺序 seed 就是 --jax_seed(5/42/137)，同 seed run 的 seen 集严格嵌套（=该 seed 最长 run 前缀），三 seed 为三条独立序列；seen 全集按全局 batch 前缀 perm[0:S×128] 重放（per-rank 交错合并恰为前缀）；window→ticker-day 由 seqs_per_file 累积索引确定，per-file offset 由同 seed 链播种；train/val split 独立 seed(lobster_dataloader.py:1441 默认42)在 VAL_SPLIT=0.0 下不生效。结论：方案由纸面可行升级为代码级确认。已写入 Notion 解答 3 补充 callout（3aa12c45-68fd-817a）。

## Codex evidence update — 2026-07-27 13:19 UTC

- “有 test loss”与“投稿主 surface 已被独立验证”不是同一件事。历史提交 `77c9228`/`4ff719a` 证明旧 `test_ce_scaling_presentation` 来自更早的 six-size 8M--78M sweep，锁定该 sweep 的 `alpha=1.178, beta=1.054`；它不是最终投稿 8M--197M / ten-run surface 的 checkpoint-matched test-CE refit。
- Test-set evaluation does not update neural-network weights. Saved checkpoints are frozen, test CE is measured, and only the five-parameter scaling surface `(E,A,B,alpha,beta)` is fitted to `(N,D,L_test)`.
- A later but different Mamba-3 sweep supplies partial forward evidence: 44 runs, 254 retained checkpoints, 487 tickers, approximately 2.625M--293M parameters. The all-checkpoint free fit is `(E,alpha,beta)=(0.4467,2.1037,0.1558)`; the run-final 44-point fit is `(0.5080,2.0008,0.3017)`. These do not reproduce the submitted logged-loss `beta≈1.33`, but cannot substitute for a matched refit of the submitted grid.
- The separate Claude-track note above reporting `beta≈0.978` refers to a different claimed artifact/protocol and is not reconciled with the Codex-verified 254-checkpoint artifact. It was not merged into the Codex manuscript.
- The pXiP Q2 blue Notion callout now states all three evidence layers explicitly and was re-fetched successfully: earlier locked six-size diagnostic, later different-sweep free refit, and missing checkpoint-matched refit of the submitted ten-run surface.
- The external-model figure uses open descriptive markers for TradeFM and MarS and a counterfactual Mamba-3 allocation curve; it makes no undertrained/overtrained classification.
- A full rerun of `rebuttal_analysis/run_sensitivity.py` reproduced the common-D, fixed-beta, LOO, profile, joint-propagation, and 5/9 IsoFLOP-deletion results without numerical drift.

## Gamma provenance clarification — 2026-07-27 13:31 UTC

- `gamma=0.899728665943256` is the slope from an ordinary least-squares regression with intercept, `log(q_i)=log(k)+gamma log(N_i)`, implemented by `np.polyfit(log_n, log_q, 1)`.
- The inputs are 12 unique Mamba-3 profiler rows from historical Git object `b661f145...:reproduce_with_codex/flops_profile.csv`: `N=2,625,923` to `293,283,039` parameters and `q=98.4M` to `8.701B` FLOPs/token.
- The same regression gives `k=170.6179807715654` and log-log `R^2=0.991824295393419`. The paper rounds the dimensionless slope to `gamma=0.90`; `k` is unit-dependent.
- Source CSV SHA256 is `bb09af777f4ce2ddcc79137d7d9855c4b8f48835d16bd766ac4e322cfaf8a2e9`. The reproducing code is `Overleaf_codex/rebuttal_analysis/run_sensitivity.py:253-305`.
- `gamma` is a compute-profile exponent, not a network training parameter and not either loss-surface exponent `alpha` or `beta`.

## Sequential Q1 anchor update — 2026-07-28 00:12 UTC

- The user-supplied anchor `3aa12c45-68fd-8072-b6d4-ef5b09d2f9ad` resolves to the nested gray task-list callout, not an independently updatable Notion page. The targeted write therefore used the parent page `3aa12c45-68fd-80ab-89b5-d0d597bad64c` and an exact replacement of the unique pXiP Q1 checklist line.
- A new blue callout titled `Codex 回答｜第 1 项：D-restricted β 与三个“0.90”口径` now sits directly under the unchecked Q1 line. Re-fetching the exact anchor found one marker and verified `beta=1.4336`, `beta=0.6161`, `beta=0.9438`, and `gamma=0.899729`.
- The common-D result remains the literal answer to Q1: full support gives `(alpha,beta)=(0.3711,1.3325)`, common-D gives `(0.9935,1.4336)`, and late common-D gives `(1.2132,1.1405)`. Common-D restriction alone does not force `beta<1`, but the movement across window and weighting choices rules out treating `1.33` as a universal or identified causal exponent.
- `beta=0.6161` is a nonlinear log-Huber loss-surface fit on tail-25% logged training losses (9 runs, 90 points); `beta=0.9438` is the analogous fit on near-mature logged training-loss checkpoints selected by `loss <= 1.2 * terminal_loss` (10 runs, 1,448 points). Neither is a test-loss result or the literal common-D refit.
- `gamma=0.899729` is a separate ordinary least-squares log-log fit of FLOPs/token against parameter count over 12 profiler points, with `k=170.618` and `R^2=0.991824`. It contains no loss observations and has no train/test split.
- The Jan-2026 forward stream is an evaluation protocol: neural-network checkpoints remain frozen, test CE is computed, and only the five-parameter scaling surface may be freely refit. The earlier six-size paper diagnostic locked exponents; the later 44-run free fit is a different sweep. Neither is the requested checkpoint-matched free refit of the submitted ten-run grid.
- Primary hashes recorded in the callout: submitted sensitivity JSON `88e895...c7176`, tail fit `8ee578...8a2`, near-mature fit `7608eb...669`, profiler CSV `bb09af...a2e9`, and later interim long-D fit `8fb4c6...4fce`.
