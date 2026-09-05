# R1 — Adversarial review of `01_measurements.md`, `03_infrastructure.md`, `06_review_response.md` and `PLAN.md` §0.1–§0.9

> Reviewer 1 of five, launched alone on 2026-09-05. Lens: **the three files the main session
> wrote about its own measurements, with no independent check.** Everything below was
> re-derived on this machine; every command is CPU-only and shown.
>
> **The code moved while I was reading it.** `code/failure_pool_reliability.py` and
> `code/test_failure_pool_reliability.py` were edited at 2026-09-05 06:43Z (items 2 and 3 of
> `06` §6 landing) and `sigma-0/src/post_training/heuristic_learning/fidelity.py` at 06:42Z.
> All code findings are stated against the 06:44Z snapshot kept at
> `/run/user/1483804540/claude-1483804540/-lus-lfs1aip2-projects-public-u6gb/a6d1353d-1003-4f01-b416-12a6339ff99d/scratchpad/fpr_snapshot_20260905T0644Z.py`.
> The four documents under review (`01` 03:13Z, `03` 03:15Z, `06` 04:02Z, `PLAN.md` 04:03Z) did not change.

## 中文速览

- **`01` §1.1 的核心结论「A-to-B 不可能」是错的，两条依据都不成立。** `wm_ft_multi3` 其实有
  **18 个兄弟检查点** `wm_ft_multi3_step{150…4800}`（都在上一级 `ckpt/` 目录里，`01` 只
  `lfs find` 了 `wm_ft_multi3/` 内部所以看不见）；Muon 优化器状态**就在盘上**（141 个 muon 叶子、
  36 个真数组，与 selftrain 链逐项相同），总字节数两边都是 **566.4 MB**，`01` 引的
  「418.6 vs 499.5 MB」只是 OCDBT 分块方式不同。两份 `_ROOT_METADATA` **md5 完全相同**，
  所以 step 69378 撞车不是巧合，是继承。PLAN 的 F10、X4、§0.6 全部建立在这条上。
- **所有已发布数字所依赖的 npz 在 2026-09-05 02:36–02:37Z 被静默重写过。** `scores()` 与
  e8425cb1 逐字节相同，AMD 的 `spread_share_top_decile` 却从 0.2954 变成 0.2661、
  池重叠 0.380→0.300、重生成 ρ 0.8048→0.8642。没有任何记录说明改了什么。
  于是 `06` 说的「独立复现」比对的其实是另一批输入。
- **计划自己引的 Zyphra onset 律预测这个模型规模的起点在约 200B token，而唯一有早/晚检查点
  的那条 run 只跑到 3.608B（1.7%）。** M6 按现在的设计几乎必然读出 ABSENT，而原因与可塑性无关。
- **适应跑一直起不来的真因就写在日志里：`SQUASHFS_MULTI_MOUNT_ROOT` 这个旋钮从来没到达代码**
  —— `node_wrapper.sh:342` 无条件把它置空。同一个缺陷还让 M6 两个成员共用一个挂载根。
- `06` §2.1 说的 A1 修复**没修完**：`regeneration_null` 仍在平均全部七个视野，`split_half` 的
  `horizon_idx` 默认值仍是 `None`，而新加的那个 `assert` **任何命令行输入都触发不了**。
- `06` §2.2 的比值表**用 §2.6 给的命令复现不出来**，而模块 docstring 与新测试里钉的常数
  （GOOG 4.68→9.87）正是那批复现不出来的数（已提交 JSON 是 5.93→10.16）。
- `03` §3.4「E-3 跑不了」在算术上对，但它是「坚持把每个 member 的 3,000 个 CSV 写到 Lustre」
  的后果；`03` 自己的第三个选项把每 member 的 Lustre 占用降到约 6 个 inode。这条却一路
  传到 PLAN §0.2 X2，把 k 从 20 压到 3 —— 一个科学参数被一个有零成本解法的存储约束改掉了。

---

## Findings

| # | Severity | File / section | Claim attacked | Defect |
|---|---|---|---|---|
| F1 | **BLOCKING** | `01` §1.1; `PLAN` §0.1 F10, §0.2 X4, §0.6 | "`wm_ft_multi3` carries exactly one checkpoint, so … A-to-B is impossible. Not expensive — impossible." | 18 sibling checkpoints exist; Muon state is on disk; the size argument is an OCDBT packing artefact; the two roots share a byte-identical `_ROOT_METADATA` |
| F2 | **BLOCKING** | `06` §2 (all); `PLAN` §0.5 | "Every claim was re-derived here before being accepted" | The `.returns_multih_*.npz` inputs were rewritten 2026-09-05 02:36–02:37Z with no record; deterministic quantities moved by up to 10% relative while the code that computes them is byte-identical |
| F3 | **BLOCKING** | `PLAN` §0.7, §3 Step 2; `01` §2.6 M6 | The M6 early/late pair is `33575 → 69378 = 1.862B tokens` | The plan's own onset law (`PLAN` §2.1) puts onset for this size at ~200B tokens. The whole run is 1.7% of that. M6 reads ABSENT by construction |
| F4 | **BLOCKING** | `01` §4 defect (a); `PLAN` Step 2 | "the adaptation job 6141106 exits immediately" | 6141106 is a 24-hour host allocation, not the adaptation job. The real cause is in the log: `SQUASHFS_MULTI_MOUNT_ROOT` is cleared by `node_wrapper.sh:342` and never applies, so the mount is stale **and both M6 members share one mount root** |
| F5 | MAJOR | `06` §2.1 | "`main()` now passes `a.horizon_idx` to both paths, and asserts … The assertion is not decorative" | The assert compares two sets built from the same variable in the same function body — no CLI input can make it fire. `regeneration_null` still averages 7 horizons. `split_half(horizon_idx=None)` is still the default. No test for A1 |
| F6 | MAJOR | `06` §2.2; `PLAN` §0.5 A2, §0.9 | "From the largest-k point alone, k for ρ = 0.80 is 17–41, median 21" | 3 of the 8 numbers come from the one-parameter fit the same section calls biased low; the pure largest-k set is 16.7–40.6, median 22.4; and it is still a `k ≤ 5` extrapolation, hence a lower bound. `PLAN` §0.9 then uses "k ≈ 21" as a budget |
| F7 | MAJOR | `code/failure_pool_reliability.py` `rollouts_needed` | The rejection gate implements the audit's fix | `ratio_rises_with_k` is computed and never used. AMD and AMZN rise monotonically and still get a point estimate emitted |
| F8 | MAJOR | `06` §2.3; `PLAN` §0.5; deliverables | "This corrects a claim in the published notebook" | The notebook still carries all four retracted claims, and `_nb_build_failure_pool.py:336` now raises `KeyError: 'k_for_rho_0.80'` on GOOG. No item anywhere schedules the rebuild |
| F9 | MAJOR | `06` §2.2, §2.6; module docstring; new test | "Commands are in §2.6" | Those commands are deterministic and produce a different table. The constants landed in the docstring and in `test_rollouts_needed_refuses_to_extrapolate…` (`GOOG 4.68 → 9.87`) match no committed artefact |
| F10 | MAJOR | `PLAN` §0.1 F1, F4 | The measured-facts table | F1 and F4 state numbers that §0.5 A1 and A4 retract four paragraphs later, unmarked, while F5 uses the document's own strikethrough convention |
| F11 | MAJOR | `PLAN` §0.5 | "Tests: 12 → 17, each new one red on a defect that actually shipped" | Run against the e8425cb1 module, 2 of the 5 pass unchanged. The defect with a numeric consequence (A1) has no test at all |
| F12 | MAJOR | `03` §3.4; `PLAN` §0.2 X2, §0.3 | "E-3 as written cannot run" ⇒ `k = 3` | True only if the CSVs go to Lustre. `03`'s own option 3 costs ~6 inodes per member. A storage constraint with a zero-science fix was allowed to change a scientific parameter that rests on the untested dilution assumption |
| F13 | MAJOR | `code/attach_adaptation.sh`; `PLAN` §5 | "Step 2 … 1–2B tokens each" | The launcher's budget is `1500 × 52,000 = 78M` tokens. The uncommitted `CURTAIL_EPOCHS=${CURTAIL_OVERRIDE:-1500}` reintroduces the `:-` default anti-pattern on the one quantity a fixed-budget comparison must hold equal, with nothing asserting the pair agrees |
| F14 | MAJOR | `01` §3; `PLAN` §0.2 | The score is stratified `total` | `plan_20260904/drafts/D1_failure_pool.md` §1.4–§1.5 decides for `bias2` against `total`. Not in `PLAN` §0.2's contradiction table |
| F15 | MINOR | `06` §4, §6 item 6; `PLAN` §0.7 | "`num_devices = 1` … Neither has been read. Do not quote the absolute token counts" | Both settling artefacts are on disk and one is in a file `06` §1 already lists |
| F16 | MINOR | `03` §3.2, §3.4, §4.1; `06` §5 P6 | inode arithmetic and code inventory | `3,007` is a file count (3,010 inodes); `240,640` should be `240,560`; `03`'s "482,240 = 65%, fits" contradicts `06` §5 P6 clause 3; line counts and "12 tests" are stale |
| F17 | MINOR | `01` open question 3 | "the rollout length … is not recorded … a one-line CPU check that has not been done" | `wc -l` on any generated CSV = 250 |

---

## F1 (BLOCKING) — `01` §1.1's "A-to-B is impossible" is false, and both legs of the evidence fail

**The claim.** `/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/plan_drafts/01_measurements.md` §1.1:

> "**`wm_ft_multi3` carries exactly one checkpoint, so it has no early-and-late pair and
> A-to-B is impossible.** Not expensive — impossible. A second, independent check confirms it
> from the other direction: the largest data blob under `wm_ft_multi3/69378/state/` is
> **418.6 MB** against **499.5 MB** for the selftrain chain's step 69378 … so `wm_ft_multi3`
> is an **inference-only artefact with no Muon optimizer state**."

and

> "That also confirms draft 05 §2.6's assertion directly: the two step-69378 checkpoints differ
> in size and in content, so the coinciding step numbers are a coincidence, not an identity."

`PLAN.md` §0.1 F10, §0.2 X4 and §0.6 all rest on this, and §0.6 turns it into the strongest
stated argument for spending a GPU regeneration run ("those are the same regeneration").

**Why it fails, leg 1 — the checkpoints exist, one directory up.** `01` §2.1's method is
`lfs find <run_root> -maxdepth 1 -type d`, which looks only *inside*
`/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z/ckpt/wm_ft_multi3/`.
The fine-tune wrote its ladder as **sibling directories**:

```
$ lfs find /lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z/ckpt \
      -maxdepth 1 -name 'wm_ft_multi3*'
  wm_ft_multi3_step150   _step300   _step450   _step750   _step900   _step1200
  _step1500  _step1650  _step2100  _step2250  _step2850  _step3000  _step3150
  _step3750  _step3900  _step4350  _step4500  _step4800     (+ wm_ft_multi3rep_step150/300)
```

Two files sitting in the directory `01` did read say the same thing:

```
$ cat /lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z/ckpt/wm_ft_multi3/ft_progress.json
{"last_saved_step": 4800, "arm": "wm_ft_multi3",
 "ckpt": "/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z/ckpt/wm_ft_multi3_step4800",
 "complete": true, ...}
```

`ft_log.json` logs held-out `gen`/`real` every 50 steps from 3451 to 4800. So `wm_ft_multi3`
is a 4,800-step fine-tune with 18 retained ages — an early-and-late ladder.

**Why it fails, leg 2 — the Muon state is on disk and the sizes are equal.** Parsing the
on-disk `_METADATA` of each checkpoint:

| checkpoint | total leaves | leaves under `muon` | muon leaves with `skip_deserialize=False` | total bytes | largest OCDBT blob |
|---|---:|---:|---:|---:|---:|
| `wm_ft_multi3/69378` | 835 | 141 | 36 | 593,951,889 | 418.6 MB |
| `wm_ft_multi3_step4800/69378` | 835 | 141 | 36 | 593,954,934 | 363.8 MB |
| `wm_ft_multi3_step150/69378` | 835 | 141 | 36 | 593,960,130 | 313.4 MB |
| selftrain `69378` | 835 | 141 | 36 | 594,011,454 | 476.3 MB |
| selftrain `33575` | 835 | 141 | 36 | 594,083,834 | 350.5 MB |
| selftrain `275` | 835 | 141 | 36 | 594,144,353 | 276.2 MB |

Every checkpoint is **566.4 MB, identical to within 0.03%**, and every one declares the same
36 real Muon arrays. The "largest blob" varies from 276 MB to 476 MB purely because OCDBT
packs the same tree into different numbers of files — and it varies by the same amount
*within* the selftrain chain, so by `01`'s own metric the selftrain chain's step 275 would
also be "inference-only".

The log line `01` quotes says the opposite of what `01` read into it. In that Orbax message
`Target` is the on-disk tree (it holds the full muon subtree, with
`ValueMetadataEntry(value_type='jax.Array', skip_deserialize=False, write_shape=(1024, 4480))`
entries) and `Source` is the restore item the *inference* script built — which is why the
same log first prints `[Optimizer] Using inject_hyperparams (legacy scalar LR)` and then also
reports `regular.inner_state: Source dict, Target list`, a type mismatch that cannot mean
"the checkpoint is missing data".

**Why the "coincidence" reading fails.**

```
$ md5sum /lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z/ckpt/wm_ft_multi3/metadata/_ROOT_METADATA \
         /lus/lfs1aip2/projects/public/u6gb/sigma-0/checkpoints_selftrain/j5705912_b30675li_5705912/metadata/_ROOT_METADATA
028879b3aa96f727b3fb94a6f894d072  ...wm_ft_multi3/metadata/_ROOT_METADATA
028879b3aa96f727b3fb94a6f894d072  ...j5705912_b30675li_5705912/metadata/_ROOT_METADATA
```

Byte-identical, same mtime (Jul 18 21:50). `wm_ft_multi3` is a descendant of the selftrain
chain and inherits its step key. The step numbers coincide **because they are the same step
number**, which is the opposite of a coincidence. Note the second-order trap this creates and
that `06` §4 walks into unknowingly: `wm_ft_multi3/metadata/_ROOT_METADATA` describes the
*pre-training* run, not the fine-tune, so nothing in it can be read as a property of the
fine-tune.

**The scenario in which the plan produces a wrong answer.** `PLAN.md` §0.6 concludes that M1
and thread unification need the same regeneration and calls it "the strongest argument yet
for B-to-A". That argument is void: A-to-B is available, `wm_ft_multi3` has 18 ages and a
resumable optimizer state, and the choice between B-to-A and A-to-B has to be re-taken on
scientific grounds (which lineage the plasticity question is about) rather than on a
nonexistent impossibility. A GPU regeneration is currently justified by a false premise.

**Cheapest check that settles it.** The three commands above: one `lfs find -maxdepth 1
-name 'wm_ft_multi3*'`, one `cat ft_progress.json`, one `md5sum`. Seconds, no GPU.

**One thing `01` got right and should be kept**: the fine-tune ladder is in *fine-tune* steps
(150…4800), not pre-training tokens, so the A-to-B early/late separation is ~4,650 fine-tune
steps and has to be converted before it can be compared with the selftrain chain's ladder.
That is a real limitation — but it is a sizing question, not an impossibility.

---

## F2 (BLOCKING) — the archive under every published number was rewritten, silently, between the audit and the response

**The claim.** `06` §2 opening: "Every claim was re-derived here before being accepted."
`PLAN.md` §0.5: "**All four were re-derived independently before being accepted.**"

**Why it fails.** `scores()` is byte-identical between commit `e8425cb1` and the current
working tree (`diff` of the function: no output), and the expression computing
`spread_share_top_decile` is unchanged (commit `49694537` only *adds* fields). Yet:

| ticker (H = 50) | `spread_share_top_decile` at e8425cb1 | now | pool overlap then → now | regeneration ρ then → now |
|---|---:|---:|---|---|
| AMD | 0.2954 | **0.2661** | 0.380 → **0.300** | 0.8048 → **0.8642** |
| AMZN | 0.3386 | 0.3531 | 0.560 → 0.580 | 0.8177 → 0.8302 |
| GOOG | 0.6359 | 0.6133 | 0.380 → 0.360 | 0.8417 → 0.8391 |
| INTC | 0.2143 | 0.2110 | 0.340 → 0.320 | 0.8671 → 0.8651 |
| JPM | 0.2050 | 0.2029 | 0.300 → 0.320 | 0.8453 → 0.8452 |
| META | 0.2816 | 0.2770 | 0.360 → 0.340 | 0.8681 → 0.8678 |
| MSFT | 0.2852 | 0.2870 | 0.360 → 0.380 | 0.8659 → 0.8632 |
| NFLX | 0.3960 | 0.3948 | 0.520 → 0.560 | 0.8611 → 0.8592 |

These are deterministic functions of the input arrays — no `rng` touches any of them, and
both runs report the same `n_contexts = 500` and the same ten seeds. The inputs changed:

```
$ D=/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z/data
$ stat -c%y $D/hp_v5me3_AMD_s97701/member_0/.returns_multih_gen.npz   2026-09-05 02:36:37
$ stat -c%y $D/hp_v5me3_AMD_s97701/member_0/.returns_multih_real.npz  2026-09-05 02:36:41
$ stat -c%y $D/hp_v5me3_AMD_s97701/member_0/.returns_gen.npz          2026-09-03 18:29:30
```

Every `.returns_multih_*.npz` across every ticker and seed carries an mtime of 02:36–02:37 on
2026-09-05 — after the audited commit (Sep 4 21:27) and before the corrected results JSON
(Sep 5 03:59). Nothing records what changed or why; these are dot-files with no manifest,
which is exactly the failure `03` §2 is written to prevent, occurring inside the review that
argues for `03` §2.

**Scenario in which the plan produces a wrong answer.** `06` §2.4 presents a "published
(ddof = 0)" column of AMD 0.266 / GOOG 0.613. Those are *not* the published values —
`e8425cb1` published 0.2954 / 0.6359, which is exactly what
`plan_20260904/drafts/D5_premortem.md` §6.5 reports. So `06` silently replaced the audited
numbers with its own re-run on different inputs and then reported agreement. The same applies
to A1's table and A2's ratios. Every "matches to within draw noise" in `06` is a comparison
between two different datasets.

Downstream, AMD's naive-vs-corrected pool overlap moved from 0.380 to 0.300 — the "top-decile
pools overlap 40%" headline (`_BRIEF` fact 2, `PLAN` §0.1 F2) is a mean whose per-ticker
range widened from 0.30–0.56 to 0.30–0.58, and whose worst cell moved by 8 percentage points.
An arm comparison sized against a 40% overlap and run against a 30% one is mis-sized.

**Cheapest check.** `stat -c%y` on any `.returns_multih_gen.npz` versus
`git log -1 --format=%ci e8425cb1`; then
`git show e8425cb1:tasks/continual_learning/results/failure_pool_reliability.json` and diff
the `dispersion` blocks. Under a minute.

**What must happen before anything else.** Record what regenerated those files and whether
the change was intended. If it was a fix, the audit's numbers and the response's numbers are
not comparable and `06` should say so; if it was not intended, every number in `PLAN` §0.1 and
§0.5 is provisional.

---

## F3 (BLOCKING) — the plan's own onset law says this run is ~60× too short for M6 to see anything

**The claim.** `PLAN.md` §0.7 and `06` §4 convert the 17 selftrain steps to tokens and pick

> "**33575 → 69378 = 1.862B**, the audit's recommended 'early'"

as the M6 pair, and `PLAN.md` §3 Step 2 says "pick θ_early and θ_late checkpoints separated by
as many tokens as the run allows."

**Why it fails.** `PLAN.md` §2.1 cites the Zyphra onset law and
`/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/deep-reseach.md:2315` states it
with units:

> "Onset scaling law: T = 1.3e-5 * P^0.8269, where T = task-instance number at onset and
> P = non-embedding parameters."

and `deep-reseach.md:2317` calibrates the instance size: "a 1B non-embedding model would show
onset near 360 task instances = 1.8T tokens; a 7B model near 9T tokens". Both anchors give
**5B tokens per task instance** (1.3e-5·(1e9)^0.8269 = 359.7; 1.3e-5·(7e9)^0.8269 = 1799).

For this model, `[*] Trainable Parameters: 78539423` with a (2112, 1024) embedding, so
P_non-embed ≈ 7.64e7:

```
T = 1.3e-5 * (7.64e7)^0.8269  =  42.8 task instances  =  2.14e11 tokens  =  214B tokens
```

The entire selftrain run is **3.608B tokens** (`PLAN` §0.7), i.e. **1.7% of predicted onset**,
about 60× short. The widest available pair (275 → 69378 = 3.593B) is also 1.7%. Even the
*smallest* model in the reference study (5M non-embedding) has predicted onset at 22.5B
tokens — six times this run's total.

**The concrete scenario.** M6 runs, AUC(θ_late) ≈ AUC(θ_early), the CI includes equality, and
`PLAN` §2.3's rule reports ABSENT or inconclusive. `PLAN` §2.2 already frames the deliverable
as "the first plasticity onset law for SSM/linear-attention models" — so an ABSENT reading
from a run 60× short of predicted onset is at serious risk of being written up as evidence
that state-space models resist plasticity loss. The design cannot see the phenomenon it is
built to detect.

**Cheapest check.** Two lines of arithmetic on the plan's own formula, above. No filesystem
access needed.

**What to do with it.** Not "abandon Step 2" — the honest move is to put the predicted onset
next to the available budget **in §0.7, at the point where the pair is chosen**, so the reader
sees the gap when the decision is made rather than in a footnote. Then either (a) state Step 2
as a *negative-result-with-known-power* measurement and pre-register that ABSENT is
uninformative, or (b) find a run with a larger token span, or (c) reduce the effective P by
probing the 33.6M model (`D3` §3.1 row `5877859`), for which predicted onset is ~99B — still
27× short, which is itself the finding. Note also that the LOB/SSM constant is exactly what
Step 5 proposes to measure, so the law may not transfer; but the plan cannot both cite it as
the reason the work matters and ignore it when sizing the probe.

---

## F4 (BLOCKING) — the reason Step 2 has never run is a knob that never reaches the code, and the same defect makes the M6 pair collide

**The claim.** `01` §4 lists as defect (a) of Step 2: "the adaptation job 6141106 exits
immediately (05 §8.2)", and on that basis marks Step 2 "~~as written~~ **mis-specified,
rewrite**".

**Why the evidence pointer is wrong.** `6141106` is not the adaptation job:

```
$ sacct -j 6141106 --format=JobID,JobName,State,Elapsed -P | head -2
6141106|u6gb-4-node-chain|TIMEOUT|23:59:23
```

It is a 24-hour four-node host allocation that carried hundreds of unrelated steps.
`code/attach_adaptation.sh:12` merely defaults `ALLOC=6141106`. The actual failures are:

```
6141106.254|cl-adapt-e275   |FAILED|1:0|00:00:07
6141106.255|cl-adapt-l69378 |FAILED|1:0|00:00:08
6141106.259|cl-adapt-e275   |FAILED|1:0|00:00:01
6141106.260|cl-adapt-l69378 |FAILED|1:0|00:00:01
6153294     |cl-adapt-e275   |FAILED|1:0|00:00:32     (the sbatch form, same day)
6153297     |cl-adapt-l69378 |FAILED|1:0|00:00:32
```

**The root cause has been sitting in a 1,899-byte log for nine days.** Both members, in
`/lus/lfs1aip2/projects/public/u6gb/sigma-0/logs_cl_probe/{e275,l69378}/training_6141106_node0.log:23-24`:

```
mkdir: cannot stat '/tmp/kangli.u6gb/sigma0/6141106_0/sp500_squashfs_6141106_0/2024-08':
       Transport endpoint is not connected
[squashfs] FATAL: squashfuse failed for 2024-08
```

`code/attach_adaptation.sh:38-40` exports a unique mount root precisely to prevent this:

```bash
# Unique node-local mount root per run: avoids the stale dead mount left under
# the default ${SLURM_JOB_ID}-derived path, and keeps the pair from colliding.
export SQUASHFS_MULTI_MOUNT_ROOT=/tmp/kangli.u6gb/sigma0/cl_probe_${SHORT}/sp500_squashfs
```

and `/lus/lfs1aip2/projects/public/u6gb/sigma-0/run/base_model/node_wrapper.sh` throws it away:

```
342:  SQUASHFS_MULTI_MOUNT_ROOT=""                                    # unconditional
370:  SQUASHFS_MULTI_MOUNT_ROOT="${SQUASHFS_MULTI_MOUNT_ROOT:-$TMPDIR/sp500_squashfs_${SLURM_JOB_ID:-manual}_${SLURM_PROCID:-0}}"
```

Line 342 runs before line 370, so the `:-` default always wins. This is the project's
signature defect twice over: a knob set, exported, and documented in a comment that never
reaches the code, and a default that applies because something upstream cleared the override.
The log's mount path proves it: `sp500_squashfs_6141106_0`, the default form.

**The second consequence, worse than the first.** Because the fallback path is keyed on
`${SLURM_JOB_ID}_${SLURM_PROCID}`, two members attached into the same allocation on the same
node get **the same mount root** — the logs show both `e275` and `l69378` at
`/tmp/kangli.u6gb/sigma0/6141106_0/sp500_squashfs_6141106_0`. `cleanup_squashfs`
(`node_wrapper.sh:343-359`) unmounts everything under that root, so one member finishing
unmounts the other member's data mid-run. **A paired early-vs-late comparison cannot be run
concurrently on one allocation at all**, and the failure mode is a mid-run data disappearance
rather than a clean error.

**Cheapest check.** `sed -n '340,372p'` on `node_wrapper.sh` and `grep squashfs` on the two
probe logs. Under a minute; both paths are already written in `01` §4 and `05` §8.2.

**Effect on the plan.** Defect (a) is a five-minute infrastructure fix (delete line 342's
unconditional reset, or move it above the export read), not grounds for a Step-2 rewrite.
Defects (b) (wrong lineage) and (c) (no null pair) remain, and (b) is itself affected by F1.

---

## F5 (MAJOR) — the horizon fix is incomplete and the assertion added to guard it cannot fire

**The claim.** `06` §2.1: "`main()` now passes `a.horizon_idx` to both paths, and asserts the
two paths agree on `n_pairs` before writing anything. **The assertion is not decorative: it
fails loudly on the configuration that shipped.**"

**Why it fails.** The assertion (snapshot lines 618-622) is

```python
n_raw = {r["n_pairs"] for r in sh_raw}
n_str = {r["n_pairs"] for r in sh_str}
assert n_raw == n_str, (...)
```

and both lists are built four lines above from the *same* `a.horizon_idx`, the same `a.ks`
and the same `a.draws`. `n_pairs` is `n_draws × len(hs)` and `hs` is identical on both paths,
so `n_raw == n_str == {a.draws}` for every possible command line. **No CLI input can make it
fire.** It fires only if someone edits the source back to `horizon_idx=None` — a guard against
a text edit, not against a configuration. It is also an `assert`, so it vanishes under `-O`.
This is the "a self-test that cannot fail" pattern in its runtime form.

**Two reporting paths still mix horizon sets.**

1. `split_half`'s signature is still `horizon_idx: Optional[int] = None` (snapshot line 312),
   and line 333 still does `hs = range(real.shape[1]) if horizon_idx is None else [horizon_idx]`.
   The dangerous default that produced the shipped defect is intact for every caller except
   the one that was fixed. The audit's instruction was "make `horizon_idx` mandatory on the
   reporting path."
2. `regeneration_null` (snapshot line 351) has **no `horizon_idx` parameter at all** and its
   `rho_mean` is the mean over all seven horizons. That number is in the emitted JSON next to
   single-horizon split-half numbers, and it is *the ceiling number* the plan quotes. Measured:

```
ticker  7h mean   H=10   H=25   H=50   H=100  H=150  H=200  H=250
AMD      0.864   0.965  0.945  0.899  0.844  0.805  0.804  0.787
...
MEAN     0.854                 H=50 mean 0.888, range 0.851-0.912
```

So the "regenerations agree at 0.81–0.87" (`_BRIEF` fact 5, `PLAN` §0.1 F5/F6, `02` §2.2)
is a seven-horizon average being compared against `split_half` numbers that are now H = 50
alone — **the identical defect A1 describes, unfixed, on a different pair of paths.**

**Concrete scenario.** Anyone reading `results/failure_pool_reliability.json` reads
`split_half_stratified[k=5].rho_mean = 0.462` (H = 50) beside `null.<ticker>.rho_mean = 0.854`
(7-horizon) and concludes "regeneration ceiling 0.85, achieved 0.46". At the matched horizon
the ceiling is 0.888, and the horizon dependence is 0.965 → 0.787 across the range, so any
statement about how far below the ceiling the score sits is off by an unknown amount.

**Cheapest check.** `grep -n "horizon_idx" code/failure_pool_reliability.py` — `regeneration_null`
does not appear. Then the per-horizon table above, five lines of Python on data already loaded.

**Also unaddressed.** `D5` §6.6 gap 1 ("No test that the raw and stratified reporting paths use
the same horizon set") is still open. Four other gaps got tests; the one belonging to the
defect `06` lists first did not.

---

## F6 (MAJOR) — "17–41, median 21" is the rejected extrapolation with a bigger constant, and it mixes two estimators

**The claim.** `06` §2.2: "*Corrected headline*: … From the largest-`k` point alone, **k for
ρ = 0.80 is 17–41, median 21, and still rising with k.** ~~'a one-parameter fit puts k at
roughly 20'~~ — the honest statement is that no extrapolation from `k ≤ 5` is supported."
`PLAN.md` §0.5 A2 repeats it; `PLAN.md` §0.9 then writes "**If it fails, §0.3 reverts to
`k ≈ 21`**".

**Why it fails, mechanically.**

*(a) Three of the eight numbers are not from the largest-k point.* `rollouts_needed` emits
`k_for_rho_0.80_largest_k_only` **only in the rejected branch**. From the committed JSON, the
fit is rejected for GOOG, INTC, META, MSFT, NFLX (5 of 8 — `06`'s count is right) and *not*
rejected for AMD, AMZN, JPM, which get the one-parameter `k_for_rho_0.80` instead. So the set
`06` quotes is a mixture:

| ticker | emitted key | value | pure largest-k (`4 × ratio_{k=5}`) |
|---|---|---:|---:|
| AMD | `k_for_rho_0.80` (one-param) | 18.8 | 22.2 |
| AMZN | `k_for_rho_0.80` (one-param) | 21.6 | 24.7 |
| GOOG | `..._largest_k_only` | 40.6 | 40.6 |
| INTC | `..._largest_k_only` | 20.1 | 20.1 |
| JPM | `k_for_rho_0.80` (one-param) | 20.8 | 18.7 |
| META | `..._largest_k_only` | 22.5 | 22.5 |
| MSFT | `..._largest_k_only` | 28.7 | 28.6 |
| NFLX | `..._largest_k_only` | 16.7 | 16.7 |
| | **range / median** | **16.7–40.6 / 21.2** | **16.7–40.6 / 22.4** |

Two of the three substitutions push the number *down*, in exactly the direction `06` identifies
as the error of the one-parameter fit.

*(b) It is still a `k ≤ 5` extrapolation.* `4 × ratio_{k=5}` is the one-parameter formula
refitted through the last point; it assumes the ratio is constant for `k > 5`, which the same
section's table says it is not (rising in 7 of 8). So 17–41 is a **lower bound**, not an
estimate, and quoting a median of it as a headline is the same act the paragraph retracts.

*(c) `06` dropped the audit's third consequence.* `D5` §6.2 consequence 3 reports that the
two-parameter fit puts ρ = 0.80 **out of reach at any k in 7 of 8 tickers** (implied ceilings
0.40–0.78). `rollouts_needed` computes `two_param_intercept` and `06` never reports what it
implies. If the ceiling reading is even partly right, "k for ρ = 0.80" is not a well-posed
quantity and no number should be attached to it.

*(d) `PLAN` §0.9 then uses it as a budget.* "reverts to `k ≈ 21`" is a number the document
itself says is unsupported and biased low.

**Cheapest check.** `python3 -c "import json; d=json.load(open('results/failure_pool_reliability.json'));
print({t:sorted(v['k_needed_stratified']) for t,v in d['tickers'].items()})"` — the presence or
absence of `rejected_reason` per ticker is immediately visible.

**The honest form.** Report the two-parameter ceiling per ticker alongside a *bound*:
"reliability 0.80 needs at least 17–41 rollouts per context and possibly is unreachable; the
measurement does not support a point estimate." Then size the pool on the quantity the
downstream comparison consumes (`02` §3.3: only `R = N·k` matters for a paired arm
comparison), not on a ρ target.

---

## F7 (MAJOR) — the trend diagnostic is computed and never used in the decision it exists for

**The claim.** `rollouts_needed`'s docstring (snapshot lines 250-266) says the model "does not
hold here: that quantity **rises with k in 7 of 8 tickers**", and the function returns
`ratio_rises_with_k`.

**Why it fails.** The gate is `if y_range > 0 and resid / y_range > max_resid_frac`.
`ratio_rises_with_k` is placed in the output dict and never read. Consequence, from the
committed JSON:

```
AMD   4.56 → 4.86 → 5.18 → 5.54   rising monotonically   NOT rejected → k_for_rho_0.80 emitted
AMZN  5.30 → 5.53 → 5.66 → 6.17   rising monotonically   NOT rejected → k_for_rho_0.80 emitted
JPM   5.29 → 5.00 → 4.87 → 4.68   falling                NOT rejected → k_for_rho_0.80 emitted
```

So two tickers whose data falsify the constant-ratio model in exactly the way the docstring
describes still receive a point estimate from the fit the docstring calls biased low. The
denominator `y_range = y.max() - y.min()` also grows with how much ρ improves with k, so a
ticker with a wider dynamic range passes the same absolute curvature more easily — the
threshold is not a curvature criterion.

**Cheapest check.** `grep -n "ratio_rises_with_k" code/failure_pool_reliability.py` — two hits,
both writes, no read.

**Fix.** Reject whenever `ratio_rises_with_k` is true, or gate on the two-parameter intercept
being distinguishable from zero. `06`'s own open question 3 already flags the 10% threshold as
unregistered; the trend test needs no threshold at all.

---

## F8 (MAJOR) — the published notebook still carries all four retracted claims, and it can no longer be rebuilt

**The claim.** `06` §2.3: "**This corrects a claim in the published notebook.**"

**Why it matters and why nothing catches it.** The notebook at
`/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/failure_pool_reliability.ipynb`
(and its `.html`, both built 2026-09-04 21:26, before the fix) still contains, in its own
"What this settles" section:

- "the one-parameter fit puts $k \approx 18$ … and $k \approx 21$ (range 16-35) … **Call it 20
  or more.**" — retracted by A2
- "**26-34 percent**" (two occurrences) — retracted by A4
- the `0.03 and 0.10` zero-line framing — retracted by A3
- Table 1's raw-vs-stratified columns, built from the pre-fix JSON — retracted by A1

And the builder can no longer regenerate it:

```
$ sed -n '336p' _nb_build_failure_pool.py
          f"{kn['noise_over_signal']:>5.2f} {kn['k_for_rho_0.80']:>6.0f} {kn['k_for_rho_0.90']:>6.0f} | "
$ python3 -c "... tickers whose k_needed_stratified has no k_for_rho_0.80 ..."
['GOOG', 'INTC', 'META', 'MSFT', 'NFLX']
```

`KeyError: 'k_for_rho_0.80'` on the first rejected ticker. So the corrected results cannot be
published without editing the builder, and neither `06` §6's ordered list nor `PLAN.md`'s
deliverables checklist contains an item for it. Under the standing order that a notebook is
only delivered once it is pushed, the artefact a reader will actually open is the one with the
four retracted claims in it.

**Cheapest check.** The two commands above, plus `grep -c "26-34 percent" failure_pool_reliability.ipynb`
(returns 2).

---

## F9 (MAJOR) — `06` §2.2's table cannot be produced by `06` §2.6's commands, and the module was pinned to those unreproducible numbers

**The claim.** `06` §2: "Commands are in §2.6", and §2.6 gives
`python3 failure_pool_reliability.py --draws 40 --out ../results/failure_pool_reliability.json`.

**Why it fails.** That command is deterministic (`--seed 0`; my re-run reproduces the committed
JSON bit for bit on every field I checked). It does not produce `06` §2.2's table, and neither
does `--draws 20`:

| ticker | `06` §2.2 (k=1,2,3,5) | `--draws 20` | `--draws 40` (committed JSON) |
|---|---|---|---|
| AMD | 4.39 4.91 5.09 5.40 | 4.24 5.09 5.19 5.55 | 4.56 4.86 5.18 5.54 |
| GOOG | **4.68** 6.79 8.37 **9.87** | 5.03 7.41 8.89 10.30 | **5.93** 7.80 9.07 **10.16** |
| MSFT | 4.54 5.75 6.57 7.85 | 4.24 5.42 5.87 7.13 | 4.15 5.37 6.32 7.16 |

GOOG's k = 1 ratio differs by 21% between `06`'s table and the artefact `06` §2.6 tells you to
generate. Worse, **the unreproducible numbers were landed in the code**:

```python
# failure_pool_reliability.py, rollouts_needed docstring
#   "(e.g. GOOG 4.68 -> 9.87 from k=1 to k=5)"
# test_failure_pool_reliability.py
    rising = [1.0 / (1.0 + r / k) for k, r in zip(ks, [4.68, 6.79, 8.37, 9.87])]  # GOOG
```

so the guard test is calibrated against a run that no committed artefact contains. (Most likely
this is F2 again — the table predates the 02:36 rewrite — but nothing in `06` says so.)

**Cheapest check.** Run §2.6's command to a scratch path and diff `implied_ratio_by_k` against
the table. Six seconds.

---

## F10 (MAJOR) — `PLAN` §0.1's fact table still states the numbers §0.5 retracts, unmarked

**The claim.** `PLAN.md`'s header sets the convention: "Green marks what is done,
~~strikethrough~~ marks what measurement has overturned."

**Why it fails.** §0.1 F5 uses the convention correctly. F1 and F4 do not:

| row | states | retracted by | marked? |
|---|---|---|---|
| F1 | "split-half rank correlation 0.36–0.48 raw, 0.15–0.25 corrected" | §0.5 A1 (0.374–0.507 / 0.144–0.250, and the two were on different horizon sets) | **no** |
| F4 | "dispersion is 26–34% inside the top decile … so that is a floor" | §0.5 A4 (per-ticker 0.225–0.681; GOOG's ceiling 32%) | **no** |
| F3 | "a mis-paired score scored 0.49 against 0.46" | §0.5 (with 60 draws, `shared` > `true` in 8/8) — strengthened, not contradicted | no, and that is fine |

F1 additionally cites `failure_pool_reliability.ipynb`, the artefact of F8.

**Why it matters.** §0.1 is a numbered fact table at the top of the spine document. It is the
part that gets quoted into PRs and slides; §0.5 is nine sections of prose below it. A reader
who quotes F4's "26–34%" and builds a training-gain estimate on it will over-estimate the
headroom on GOOG by a factor of two.

**Cheapest fix.** Strike F1 and F4 and point them at A1 and A4, which is what the document's
own convention already prescribes.

---

## F11 (MAJOR) — "each new test red on a defect that actually shipped" is false for 2 of 5

**The claim.** `PLAN.md` §0.5: "Tests: **12 → 17**, each new one red on a defect that actually
shipped." `06` §2 header: "tests from 12 to 17, and the new ones each go red on a real defect."

**Verified by running the current test file against the pre-fix module** (module from
`git show e8425cb1:...`, tests from HEAD, in a scratch directory):

| new test | red on e8425cb1? | why |
|---|---|---|
| `test_rollouts_needed_refuses_to_extrapolate_a_curve_that_is_not_linear` | **yes** | old `rollouts_needed` never rejects |
| `test_dispersion_share_reports_the_unbiased_value` | **yes** | old output lacks the fields |
| `test_stratification_leak_is_exactly_one_over_n_bins` | red, but only via `AttributeError` on the new helper `F.stratification_leak`; the assertion it makes about `stratify` was already true | documents a fact, does not guard a regression |
| `test_stratify_at_production_bins_leaves_the_leak_not_zero` | **no — passes unchanged** | uses only `stratify` and `spearman`, both untouched |
| `test_rollouts_needed_still_answers_when_the_law_does_hold` | **no — passes unchanged** | old code always emitted the key |

And the defect with an actual numeric consequence — A1, the horizon mismatch, listed first in
both `06` §2 and `PLAN` §0.5 — has **no test**, only the assertion of F5 that cannot fire.

**Cheapest check.** The three commands in the paragraph above; 0.3 s of pytest.

**Why this is not pedantry.** The claim "each new test is red on a shipped defect" is the
evidence offered that the fix is real. Two of five are documentation, one is an import guard,
and the headline defect is untested — so the test count is not evidence about A1 at all.

---

## F12 (MAJOR) — "E-3 cannot run" is a placement problem, and it was allowed to change a scientific parameter

**The claim.** `03` §3.4: "**E-3 as written cannot run.** 320,000 rollouts do not fit even
deduped." `PLAN.md` §0.2 X2 records this as a contradiction and resolves it by
"X1 dissolves it: at `k = 3` …"; §0.3 makes `k = 3` the budget; §0.9 records that the whole
`k = 3` budget rests on the untested dilution assumption.

**The arithmetic is right** (I checked all of it): per member 3,007 files; deduped 1,507;
E-3 = 640 members = 964,480 deduped inodes; against 741,511 free at 02:55Z that is 130%. And
the conclusion is if anything stronger than `03` states — over 274 samples of the watcher log
spanning 2026-09-04 01:29Z to 2026-09-05 06:27Z, the **maximum free inodes ever observed is
830,586** (2026-09-04 21:13:33Z), so E-3-as-written exceeds the all-time observed headroom by
16%, not just the current reading. `03`'s two anchors reproduce exactly (118 free at
2026-09-04 17:54:08Z; 741,511 at 2026-09-05 02:55:14Z; now 732,330 at 06:49Z).

**Why the conclusion is still wrong as used.** "Cannot run" is a property of one write layout,
not of the experiment. The analysis reads only `.returns_multih_{real,gen}.npz` — `03` §3.2 says
so itself. `03` §3.4's option 3 (generate into node-local `$TMPDIR`, run the CSV consumers in
the same allocation, return only the `.npz` + manifest + derived scores) costs per member:

```
.returns_multih_{real,gen}.npz, .returns_{real,gen}.npz, sample_indices_rank0.json,
inference.log, .done, 1 directory   ≈  8 inodes
```

E-3 at k = 20 then costs 640 × 8 ≈ **5,120 inodes, 0.7% of free** — 190× less than the deduped
figure, and 0.6% of the number that the plan called impossible. `03` names this option and
calls it "the only one that does not change the science", then the 中文速览 line and the
summary table both carry "E-3 仍是 130%（放不下）" forward, and that is the form that reached
`PLAN.md` §0.2 X2.

**The concrete scenario in which this produces a wrong answer.** `PLAN.md` §0.3 sets `k = 3`,
§0.9 states that the `k = 3` budget rests on the dilution assumption, and `02` §3.2 says
nothing should assume it. So the plan changed a parameter that governs pool purity from 20 to
3, on an untested statistical assumption, in order to satisfy a storage constraint that has a
zero-science fix. If the dilution assumption fails, `PLAN` §0.9's fallback ("reverts to
`k ≈ 21`") is also infeasible under `06` §5's own P6 rule (see F16) — so the plan has no
feasible path at large k *and* no evidence that small k is sound.

**Cheapest check.** Count what the analysis actually opens:
`grep -n "returns_multih\|sample_indices\|inference.log" code/failure_pool_reliability.py` —
one file pattern. Then multiply.

**What to do.** Make the node-local write layout a P6 clause rather than a fallback, and
re-take the `k` decision on statistical grounds alone.

---

## F13 (MAJOR) — the launcher's adaptation budget is 78M tokens, and the budget knob was just given a `:-` default

**The claim.** `PLAN.md` §5: "Step 2 at the ~100M scale: two groups × 5 seeds × (**1–2B tokens
each**)". `PLAN.md` §3 Step 2: "Fixed-budget adaptation of copies of both (identical tokens,
batch, schedule, seeds)".

**Why it fails.** `code/attach_adaptation.sh` sets `PER_GPU_BSZ=4`, `GPUS_PER_NODE=1`, no
`GRAD_ACCUM`, no `MSG_SEQ_LEN` override (so 500, as in the checkpoint metadata), `TOKEN_MODE=26tok`,
and `CURTAIL_EPOCHS=1500`. Using `06` §4's own conversion, `4 × 1 × 1 × 500 × 26 = 52,000`
tokens/step:

```
1500 steps × 52,000 tok/step = 78,000,000 tokens = 0.078B
```

That is **13–25× below `PLAN` §5's stated 1–2B** and **64× below the 5B-token probe budget of
the Zyphra design `PLAN` §2.1 copies**. It is also far below what the job could do:
`MAX_JOB_HOURS=3.0` at the measured 0.5687 s/step (`D3` §3.1) allows ~19,000 steps ≈ 0.99B
tokens, so the cap is the binding constraint and it was never converted to tokens.

**And the budget knob was just made defaultable.** The uncommitted diff to
`code/attach_adaptation.sh` (present in the working tree, not in `06`):

```diff
-export CURTAIL_EPOCHS=1500
+export CURTAIL_EPOCHS=${CURTAIL_OVERRIDE:-1500}
+export MAX_JOB_HOURS=${MAX_HOURS_OVERRIDE:-3.0}
```

This is exactly the shape `03` §6's guard `test_effective_batch_is_derived_not_defaulted`
forbids ("`GRAD_ACCUM_STEPS` appears with a `:-` default anywhere in a launch script") — but
the guard names one variable, and the defect reappeared on a different one. In a fixed-budget
early-vs-late comparison the adaptation budget is *the* quantity that must be identical
between the two members, and nothing asserts it: the two members are launched by two separate
invocations, each reading `CURTAIL_OVERRIDE` from whatever environment it inherits, and
neither logs the token count. `MAX_JOB_HOURS` is now exported twice (line 30 and line 54), so
line 30 is dead — a second instance of the same shape in the same file.

**Cheapest check.** `grep -n ':-' code/attach_adaptation.sh code/submit_adaptation_pair.sh`
and the one-line multiplication above.

**Fix.** Declare `ADAPT_TOKENS` and derive `CURTAIL_EPOCHS` from it and the measured
tokens/step, print the derived value the way `CLAUDE.md`'s `[bsz]` rule requires, and have the
pair script assert both members got the same number — rather than allowing an override that
silently redefines the experiment.

---

## F14 (MAJOR) — `total` versus `bias2` is an unlisted cross-draft contradiction on the score itself

**The claim.** `01` §3: "Draft 04 §4.2 settles the **return-based** score: stratified ranking of
`total` … That is adopted here without change." `PLAN.md` §0.2's contradiction table lists four
contradictions (X1–X4) and this is not among them.

**Why it fails.** `/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/plan_20260904/drafts/D1_failure_pool.md`
carries a section headed **"### 1.4 Why `bias2` and not `total`: the dispersion floor is real
and measured"** followed by **"### 1.5 DECISION 1"**. `06` §1's table lists `D1` §2 only
("overlaps §2.3 below") and marks the rest "not yet folded"; `06` §6 puts folding `D1`/`D3`/`D4`
at item 7, behind four code items.

This is not a stylistic disagreement. Every published number was computed with
`--key total` (the argparse default, and the JSON's `"key": "total"`), and A4's finding is
precisely that the dispersion share of `total` inside the top decile ranges 0.225–0.681 —
i.e. that on GOOG two thirds of the selected-on quantity is not learnable. `D1` §1.4 uses that
same fact to argue the pool must be built on `bias2`. If `D1` is right, every reliability and
null number in `PLAN` §0.1 and §0.5 describes a score the pool will not use.

**Cheapest check.** `sed -n '99,150p' plan_20260904/drafts/D1_failure_pool.md` and compare with
`04` §4.2. Minutes, no compute. `failure_pool_reliability.py` already accepts
`--key bias2`, so the reliability comparison between the two keys is one CPU run of six seconds.

---

## F15 (MINOR) — `num_devices = 1` is settled twice over on disk, and `06` §6 item 6 is already done

**The claim.** `06` §4: "**Unverified input, flagged rather than hidden**: `num_devices = 1` …
What settles it: the wandb config for this run, or a `world_size` line in the run's stdout log.
**Neither has been read. Do not quote the absolute token counts until it is.**" `06` §6 item 6
schedules "Read the wandb config for `j5705912`" as CPU + network work.

**Why it is already answered.**

1. The run's own stdout log prints it:

```
$ tr -d '\000' < /lus/lfs1aip2/projects/public/u6gb/sigma-0/logs_lobs5/training_5705912_node0.log | awk ...
665: [FLOPs] Params: 78,539,423
666: [FLOPs] Tokens/step: 52,000
670: [FLOPs] Peak BF16 (1 GPUs): 989 TFLOPS
```

`Tokens/step: 52,000` is printed by the code, not derived, and `(1 GPUs)` states the device count.

2. `plan_20260904/drafts/D3_profiling.md` §3.1 — a file `06` §1's own table lists as present —
already tabulates job `5705912` at **GPUs = 1, tokens/opt-step = 52,000, 0.5687 s/step over 688
samples**, and §3.2 records that `tokens_per_step` is written to every log and to `wandb.config`
by `print_flops_summary` at
`/lus/lfs1aip2/projects/public/u6gb/sigma-0/src/s5/flops.py:200-209`.

3. A third, independent path already exists in this repository:
`/lus/lfs1aip2/projects/public/u6gb/tasks/unseen_manifest_j5705912_20260831/FORENSICS.md:20-24`
records `steps_per_epoch = int(train_size / (micro_bsz × num_devices × process_count)) = 80,805,346`
against `train_size = 323,221,385`, whose ratio is exactly 4 — so `micro_bsz × num_devices ×
process_count = 4` and with `micro_bsz = 4`, `num_devices = 1`. It also records "windows
consumed 277,512 = 69,378 × 4".

`06` §4's arithmetic is right (69,378 × 52,000 = 3.608B; 33,575 × 52,000 = 1.746B; separation
1.862B). The caveat should be removed and §6 item 6 struck.

**One trap `06` §4 avoided by luck and should state explicitly**: `wm_ft_multi3/metadata/_ROOT_METADATA`
is byte-identical to the selftrain chain's (F1), so reading `num_devices` from *that* copy
would describe the pre-training run, not the fine-tune. `06` read the selftrain copy, which is
the right one — but nothing in the document says why that matters.

---

## F16 (MINOR) — inode arithmetic, an acceptance-rule conflict, and stale inventory in `03`

All small, all in `03` and `06` §5:

1. **`3,007` is a file count, not an inode count.** Measured on
   `/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z/data/hp_v5me3_AMD_s97702/member_0/`: 3,007 files + 3 directories = 3,010 inodes. `03` §3.2's
   own itemisation lists "3 dirs" inside a column that sums to 3,007. 0.1% — but the label
   "3,007 inodes" propagates into `06` §5 P6 clause 2 as a measured per-member figure.
2. **`03` §3.4's G2 row**: `80 × 3,007 = 240,560`, not the tabulated `240,640`. The deduped
   figure (120,560) is right.
3. **Acceptance rules conflict.** `03` §3.4 marks the era gate at k = 20 deduped
   "482,240 = 65%, **fits**"; `06` §5 P6 clause 3 requires `inodes_planned < 0.5 × free_at_start`.
   65% > 50%, so the same number is "fits" in one document and a P6 failure in the other. It
   matters for `PLAN` §0.9's fallback: at k ≈ 21 the era gate is 336 × 1,507 = 506,352 = 68% of
   free (fails P6) and the cycle-1 pool is 672 × 1,507 = 1,012,704 = 137% (impossible). §0.9
   mentions only the cycle-1 pool.
4. **`03` §4.1's inventory is stale**: it lists `failure_pool_reliability.py` at 459 lines and
   "12 tests"; the file was 531 lines when `03` was written, is 658 now, and the tests are 17
   (21 as of 06:43Z). §3.1's "12 passed in 0.15 s" likewise.
5. **A docstring that is false**: `failure_pool_reliability.py:38` says "The real arm is written
   once, on the lowest seed of each config." All ten seeds carry `.returns_multih_real.npz`
   (verified identical by md5 across seeds 97701/97702, and three real CSVs likewise). The
   dedupe claim of `05` §5.3 / `PLAN` X3 is confirmed by that same check — it is the docstring's
   description of the *current* archive that is wrong.
6. **`code/attach_adaptation.sh:34` sets `PYTHONPATH=/lus/lfs1aip2/projects/public/u6gb/openreview-v2`**, a different tree
   from the `sigma-0` checkout it then runs from. `03` §2's manifest schema records
   `code_commit` as "the 40-hex of the sigma-0 checkout that generated this"; if imports resolve
   from `openreview-v2`, that field is wrong by construction. `03` §4.1 item 3 warns that
   "`PYTHONPATH` hacks are how `TOKEN_MODE` came to be pinned in five places" — and one is
   sitting in the launcher Step 2 depends on.

---

## F17 (MINOR) — `01` open question 3 is a one-second check

`01`'s open question 3: "**The rollout length used to generate the existing 80 members is not
recorded in this file** and the length-stratification question in §3 cannot be closed without
it."

```
$ wc -l /lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z/data/hp_v5me3_AMD_s97702/member_0/\
        data_gen/AMD_2026-01-02_message_real_id_597_gen_id_0.csv
250
```

250 generated messages per rollout, matching `D5` §6.4's `--n_gen_msgs` default of 250 at
`/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/crps-return-alignment-20260808/run/base_model/runtime/inference.py:112`.
Because the length is **constant across the archive**, `01` §3's rollout-length-stratification
concern does not apply to the existing 80 members — it applies only to any future run that
varies the length, and that is the sentence `01` §3 should carry.

---

## What I checked and found clean

**Reproduced exactly, from the committed artefacts:**

- `python3 -m pytest test_failure_pool_reliability.py -q` → 17 passed (at 06:35Z; 21 after the
  06:43Z edit, all passing).
- `python3 failure_pool_reliability.py --draws 40` reproduces
  `results/failure_pool_reliability.json` field-for-field on every quantity I compared —
  the pipeline is deterministic at `--seed 0`, which is what makes F9 diagnosable.
- `06` §2.6's leak snippet returns exactly `[0.2, 0.1, 0.05, 0.025]`. The analytic identity
  `stratification_leak(n_bins) = 1/n_bins` holds to four decimals. **A3 is fully confirmed.**
- `06` §2.1's A1 evidence: `git show e8425cb1:...json` gives `n_pairs` 140 on every raw row and
  20 on every stratified row. **The horizon mismatch was real and is correctly diagnosed.**
- `06` §2.1's corrected table: raw k=1 0.374–0.507 (mean 0.439), raw k=5 0.631–0.748 (0.692),
  stratified k=1 0.144–0.250 (0.190), stratified k=5 0.330–0.545 (0.462) — all four match the
  JSON exactly, as do the audit's own recomputed endpoints (0.370–0.507; 0.339–0.556).
- `06` §2.2's "rejected in 5 of 8": GOOG, INTC, META, MSFT, NFLX carry `rejected_reason`;
  AMD, AMZN, JPM do not. Correct.
- `PLAN` §0.5 A4's numbers against the current JSON: unbiased range 0.2255–0.6815, GOOG
  `max_removable_share_top_decile` = 0.3185 → "32%". Correct **for the current data** (see F2).
- `06` §4's token arithmetic: 52,000 tok/step; 69,378 → 3.608B; 33,575 → 1.746B; separation
  1.862B; 22,495 → 2.438B. All correct, and `Tokens/step: 52,000` is printed by the code itself.
- `06` §3: the 80 members hold `data_gen/` and `data_real/` and **no `data_cond/`**; the
  filename pattern does match `episode_builder.fixture_paths` (`episode_builder.py:255-275`),
  so M1's blocker is correctly identified as a missing stream rather than a layout mismatch.
- `PLAN` §0.1 F13 / `05` §5.3's dedupe premise: three real CSVs and `.returns_multih_real.npz`
  are byte-identical across seeds 97701 and 97702 (md5).
- `03` §3.3's two inode anchors, from
  `/home/u6gb/kangli.u6gb/gpu_watch_15min.log`: 51,199,882 used (118 free) at 2026-09-04
  17:54:08Z and 50,458,489 (741,511 free) at 2026-09-05 02:55:14Z. Both exact. Over 274 samples
  the maximum free is 830,586 and the current reading is 732,330 (06:49Z), so the "borrowed
  headroom" warning is right and, for E-3-as-written, understated.
- `03` §3.2's per-member breakdown: 500 each of `message_real`/`orderbook_real`/`refcheck_real`
  in `data_real/` and 500 each of `message`/`orderbook`/`provenance` `_gen_id_0` in `data_gen/`.
- `03` §3.4's inode multiplications (era gate 962,240 / 482,240; E-3 1,924,480 / 964,480;
  G2 120,560 deduped) and `PLAN` §0.3's k = 3 rows (72,336 = 10%; 144,672 = 20%). All correct
  apart from the 240,640 slip of F16.
- `06` §1's inventory: `plan_20260904/reviews/` is empty; `D5_premortem.md` §6 spans lines
  309–641 (333 lines, as stated); `D4_deletion_refactor.md` is 964 lines; `D1` §2 and `D3` §8/§10
  exist. `01` §1.1's citation of `attach_adaptation.sh:56` is exact.
- `01` §1.1's blob sizes (418.6 MB / 499.5 MB) are literally correct as *largest-blob* readings —
  it is the inference drawn from them that fails (F1).

**Lenses I exercised and found nothing further:** the null-control construction in
`pairing_nulls` (permutes the context axis of `gen` only, `shared`/`independent`/`cross` are
built exactly as documented — the audit is right that this part is sound); the exact
decomposition `total = bias2_raw + spread_pop` and the `spread/k` bias correction (pinned by
tests, verified against a planted answer); tie handling in `_rank`; the id-intersection join in
`load_arm` (joins by id, never by row order); the `spread_pop`/`spread` two-variance choice
(correct and documented). I found no metric-name-versus-semantics defect inside
`failure_pool_reliability.py` itself, and no divide-by-a-per-group-constant defect in the
reliability path.

**Lenses I did not exercise** (out of my remit or not reachable without GPU): the statistical
content of `02`, the arm design of `04`, the execution recipe of `05`, `plasticity_probes.py`
and its tests, and anything requiring a training or generation run.
