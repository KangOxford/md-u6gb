# PLAN — CRPS return-alignment study: what the audit changed, and what remains measurable

> Task dir: `/lus/lfs1aip2/projects/public/u6gb/tasks/crps_analysis_20260905`
> Study root: `/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z`
> Worktree: `/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/crps-return-alignment-20260808`
> Branch: `feat/midtrain-return-alignment-evidence-20260818` (the base of sigma-0 PR#60)
> Evidence archive: `./all_findings.txt` — 236 findings from 6 adversarial + 7 solution + 11 planning agents
> Convention: $\color{green}{\textsf{done}}$ for completed, ~~strikethrough~~ for claims the evidence overturned.

## 中文速览

- 这条线原本的结论是「第四轮微调比第三轮更差」（CRPS +2.92%、R 效应 +0.0904、符号检验 p=0.0078）。
- 对抗审计推翻的不是某个数，是**推断的作用域**：效应点估计稳健（12 次交叉重配对 +0.0806 ± 0.0087，从不变号），
  但用来判它的误差棒**测错了对象**——它测的是同一个检查点重新生成 rollout 的散布，
  而结论要的是**换一次训练**的散布。后者实测 4.71%，**大于 +2.92% 的效应本身**。
- p=0.0078 不是测量值，是 8 个标的双边符号检验的**下限** 2/2⁸——这个设计能表达的最大证据量。
- 有两条被我裁掉的对抗发现：cosine 学习率混淆（不存在，见 §2.1）、以及 round-4 复制能收窄结论（不能，见 §3.1）。
- 现在唯一能推进的测量是**把 round-3 那侧的复制轨迹跑到 step 1200**，因为 SE 与自由度全被 n₃ 钉住。

---

## 1. What the study claimed

| # | Claim | Where it lives |
|---|---|---|
| C1 | Round 4 is worse than round 3 on return alignment: `dR = +0.0904`, `t = +3.54`, 8/8 tickers, sign-flip `p = 0.0078` | `/home/u6gb/kangli.u6gb/variance_ladder.json` → `the_claim.on_R` |
| C2 | On the endpoint-invariant transform `d|R-1| = +0.0336`, `t = 1.08`, 5/8, `p = 0.3047` | same file, `the_claim.on_abs_R_minus_1` |
| C3 | Fair CRPS panel: round 4 is `+2.92%` worse, sign-flip `p = 0.0938` | `crps_panel.jsonl`, 8 tickers |
| C4 | A "peak" at step 1200 survives selection correction, `P = 0.19` | `build_nb.py` analysis |
| C5 | The registered null band is `0.0195` | `fix_attribution.json` |

## 2. What the audit overturned, and what it left standing

### 2.1 ~~Cosine-anneal confound: the two run groups sit at 8.9x different LR at step 1200~~ — REFUTED BY ME, 2026-09-05

`adv_stats` and `sol_peak` both reported this as a serious confound; `sol_peak` even "confirmed" the
schedule shape `0.5(1+cos(pi s/S))` and the 8.9385 ratio. It does not exist.

The trainer builds its optimizer at
`/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/crps-return-alignment-20260808/run/mid_training/wmle_full_ft.py:186-187`:

```python
tx = optax.chain(optax.clip_by_global_norm(args.clip),
                 optax.adamw(args.lr, weight_decay=0.0))
```

`args.lr` is a scalar and appears exactly once in the file. There is no schedule object anywhere.
Four independent confirmations:

| Evidence | Reading |
|---|---|
| Source: `optax.adamw(args.lr, ...)` | scalar LR, no schedule |
| Runtime banner in every log | `[Optimizer] Using inject_hyperparams (legacy scalar LR)` |
| Parameter travel per step (`sol_design`) | 1.6137e-3 (150→300) vs 1.5968e-3 (1350→1500), ratio 1.011 — flat |
| s30 six-writer natural experiment (`sol_design`) | same seed, `max_step` 4800/1500/1050 print identical losses at the same step |

The `cosine_anneal=True` field lives in **checkpoint metadata** and never reaches the optimizer. This is
the house failure mode "a knob that is set, is logged, and never reaches the code", and two adversarial
agents were taken in by it. Recorded because the mistake is more instructive than the correction.

### 2.2 The effect is real; the error bar is measuring the wrong thing $\color{green}{\textsf{established}}$

The point estimate survives replication: 12 cross-replicate pairings of the same round4−round3 contrast
give `+0.0806 ± 0.0087` and never change sign (`adv_stats`). The failure is not a measurement artefact.

What fails is the **scope of the inference**. Every registered null in the study is a *generation*-realisation
null: two rollout regenerations from the **same** checkpoint (`adv_design`; `r3a`/`r3b`, null mean −0.0104,
per-ticker sd 0.0376). The claim being judged is a *training*-realisation claim. `sol_crps` measured the
right quantity on 12 replicate fine-tuning runs:

| Quantity | Value | 95% CI |
|---|---|---|
| Run-to-run sd of the panel cell statistic | **4.71%** | [1.78, 6.20] |
| The effect it is used to establish | **+2.92%** | — |
| Corrected panel se | 5.47% (was 1.44%) | — |
| Corrected panel result | `z = +0.53`, **`p = 0.594`** | CI [−7.81%, +13.64%] |

**The noise floor of one training run is larger than the effect.**

### 2.3 The p-value is the floor of its own test $\color{green}{\textsf{established}}$

Exact two-sided sign-flip on n=8 tickers has minimum attainable `p = 2/2^8 = 0.0078125`. All three
contrasts report exactly that (`adv_design`, `adv_stats`). It says "all 8 agreed in sign", which is the
most the design can express — it is not a measurement of effect size, and no Bonferroni correction over
the study's own 42-comparison `family.json` can leave it alive, because `alpha/m` drops below the floor
for `m >= 7`.

### 2.4 Other confirmed defects, with the corrected value

| Defect | Published | Corrected | Source |
|---|---|---|---|
| Peak refutation uses max-of-10-normals, not Grubbs | `P = 0.19` | `P = 0.0970` (2x) | `sol_peak` |
| Same, on the second arm (unifw, n=6) | `P = 0.1385` | `P = 0.0053` (26x) | `sol_peak` |
| Null band for a 2-seed vs 4-seed contrast | `0.0195` | `0.0195*sqrt(1.5) = 0.0238` | `adv_math`, `sol_inference` |
| Variance-ladder rungs mix LEVEL and CONTRAST sds | ratio 2.886 | commensurable 4.082 | `sol_inference` |
| Welch–Satterthwaite df from `n_eff = 2.365` | `1.37` | `2.301` | `adv_math`, `sol_inference` |
| maxT FWER (claimed 0.042) | 0.042 | measured 0.076–0.082 | `sol_inference` |
| Study's own module on the headline | — | `p = 0.357`, checkpoint position is 94.3% of variance | `adv_stats` |
| Fair CRPS over independent runs | unbiased | biased by `(Wbar−Dbar)/2`, flat in K; on this data −0.764% [−1.790,+0.247] | `sol_crps` |

### 2.5 Provenance defects that make numbers unquotable

- **Five** circulating values for R at multi4 step 1200: 0.9153 / 0.9578 / 0.9610 / 0.9704 / 0.9781 (`adv_results`).
- Three stores disagree on every one of 16 overlapping cells; mean 3-way range 0.0566 = **62.6%** of the headline effect (`adv_results`, `sol_notebook`).
- The code that produced every 2026-09-04 number is **uncommitted** — the scorer and trainer are both dirty in the worktree (`adv_results`).
- `one_run_many_numbers.ipynb` **contains zero executable code**: 8/8 code cells are a single `# name` comment with a pre-baked base64 PNG and `execution_count` hard-set to 1 (`adv_results`).

### 2.6 Design limits that no amount of compute removes

- **No held-out context exists.** All 80 generation seeds cover the identical 600 conditioning contexts, byte-identical `ids` arrays (`adv_design`, `sol_notebook`).
- `jax_seed` is pinned at 42 in every checkpoint, so all trajectories share one initialisation; the measurable `sigma_traj` is the data-order component only, a **lower bound** (`adv_stats`).
- `p_dropout = 0.0`, so `--train-seed` varies exactly one thing: the data-order permutation (`adv_math`, `adv_design`).
- One epoch is exactly 4800 items, so `--max-step 4800` is the whole epoch and every replicate has then seen the identical multiset (`adv_design`).

## 3. What is still worth measuring

### 3.1 ~~Add more round-4 replicates~~ — REJECTED, it buys nothing

`adv_stats` measured the sensitivity directly: 18 → 30 round-4 runs cuts the between-round SE by **4.4%**,
30 → infinity cuts it by **7.4%**, and Welch df stays at **5.42**. `adv_design` gives the same conclusion
analytically: with `sigma_A` bounded below by the within-round-3 checkpoint sd 0.0548, the SE floor is
`sqrt(0.0133^2 + 0.0548^2) = 0.0564`. **The uncertainty is pinned by n₃, not n₄.**

### 3.2 THE decisive measurement: round-3 replicates to step 1200

`adv_stats`: *"Round-3 has NO recorded metric at step 1200: no `ft_log.json` exists in any `wm_ft_traj3_*`
directory that reaches 1200."*

### ~~Verified independently, 2026-09-05, and it is worse than stated: 0 of 14 round-3 replicates reach step 1200.~~ — MY OWN MISREADING, corrected same day

I read the **logs** in `/home/u6gb/kangli.u6gb/traj3_s*.log`, whose tails stop at 650-750, and concluded the
runs had died. They had not. `adv_code` had already found the cause and I did not apply it: the RUNID fix
landed on the checkpoint path but **not on the log path**, so one log file holds several run instances and
its tail is not that run's last step. The state itself says otherwise --
`wm_ft_traj3_s1/latest_checkpoint.json` records step 1350 synced at `2026-09-05T00:07:13Z`, later than
anything in the log.

The general form: I read a derived artifact instead of the state, on a question the state answers directly.

### 3.2 The real imbalance, read from breadcrumbs $\color{green}{\textsf{measured}}$

| Round | Replicates with a `step_1200` checkpoint |
|---|---|
| Round 3 (`wm_ft_traj3_s*`) | **5** (s1, s2, s3, s4, s6) |
| Round 4 (`wm_ft_traj_s*`) | **30** |

This is exactly the `n3 = 5` that `adv_stats` found pinning the SE and the Welch df, now confirmed from the
filesystem rather than from a summary. Of the other nine round-3 replicates: s5 reached 1050, s22 is live at
600, s8 stopped at 450, and six (s10, s13, s19, s20, s21, s30) never took a step -- GPU OOM, because the
launcher requested `--gres=gpu:1` and Slurm always hands out logical device 0, which was occupied.

**Action taken 2026-09-05** $\color{green}{\textsf{launched}}$: eight further round-3 replicates
(`--train-seed` 40..47, `--max-step 1500`) on cards verified empty **inside** the job step, capped at three
per node because host RAM, not GPU, is the binding constraint (`adv_fleet_review`: four runs per node is the
configuration that host-OOM-killed s1). Runner:
`/lus/lfs1aip2/projects/public/u6gb/tasks/crps_analysis_20260905/r3_replicate.sh`. This takes n3 from 5 to 13.

Expected effect, using `adv_stats`'s own sensitivity: n3 = 5 -> 13 is the only move that shifts the between-round
SE and df at all; n4 = 18 -> 30 moved the SE by 4.4% and left df at 5.42.

### 3.3 Analysis-only items (no GPU)

- Recompute the panel with the run-to-run se from `sol_crps` (done: `p = 0.594`).
- Recompute the peak with exact Grubbs on both arms (done: 0.0970 / 0.0053).
- Rebuild the notebook so that its code actually executes (§2.5).

## 4. Order of work

1. $\color{green}{\textsf{done}}$ Consolidate 236 findings; settle the cosine contradiction against source.
2. $\color{green}{\textsf{done}}$ Diagnose why 0/14 round-3 replicates reach step 1200.
3. Launch round-3 replicates to step 1200 on verified-empty cards, long allocation.
4. Build an executable notebook of the corrected mathematics, statistics, and results.
5. Execute it, render HTML with embedded images, commit and push both.

## 5. Standing rules for this line

- The unit of replication for any training claim is a **training run**, never a generation seed.
- Any quantile or sign-flip statistic is printed next to its **event count** and its **attainable floor**.
- No number enters a heading, is bolded, or is called "best" until n has stopped growing.
- Every scored cell records `k_actual` from the estimator, not from a shell directory count.
