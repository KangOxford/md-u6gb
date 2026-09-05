# 01 — The measurement ladder

> Written by the main session on 2026-09-05 after the drafting agent for this facet was
> killed by a session limit before it wrote anything. Drafts 02, 04 and 05 landed complete
> and are treated here as fixed inputs; where this file and one of them disagree, the
> disagreement is called out rather than silently reconciled, because a contradiction
> between drafts is a defect in the merged plan.

## 中文速览

- **两条线现在算的不是同一个模型。** 失败池的全部读数来自 `wm_ft_multi3` 的 step 69378，
  而 `attach_adaptation.sh` 恢复的是 selftrain 链的 step 69378 —— 步数撞车、权重不同。
  这不是测量问题，是**必须先做的一个决定**，它决定 G2 到底要生成什么。
- **阶梯的第一级不花 GPU，而且可能把整个 #73 的前提推翻**：`fidelity.py` 的 gen arm
  在 8 个窗口里 0/8 复现记录的订单簿（real arm 5/8）。如果生成的消息流本身不自洽，
  那么「这个 context 是失败案例」的排序里有多少是模型的问题、有多少是记录的问题，
  现在**无人知道**，而这一级是纯 CPU 的。
- **决定性的那一级是 M4（同配置整组重复）**，不是任何一个新指标。02 §2.3 已经量到：
  什么都不做，端点也会动到 28%。在那个数出来之前，任何组间差都不可判读。
- **PLAN.md 的 Step 2 现在是错的**：它把「早/晚检查点」的对比建立在一条
  `NO_AUTO_RESUME` 的适应跑上，而那条跑的作业 6141106 立刻退出了（05 §8.2、04 §1.24）。
  Step 5（多规模 onset 律）在本轮**按证据排除**，理由是 Step 2 尚未产出任何结论级读数。
- 打分器不是一个，是**三条互不等价的轴**，本文件给出选它们的判据与代价。

---

## 1. Are issue #73 and `PLAN.md` the same question?

**No, and they must not be merged before one decision is taken.**

`PLAN.md` asks whether the network has *stiffened* — whether a late checkpoint, given the
same adaptation budget as an early one, learns less. Issue #73 asks whether *targeted data*
recovers behaviour the model gets wrong. Those are different objects: the first is a property
of the parameters, the second is a property of the data mix. They share a training loop and
almost nothing else.

They can share a plan only if they operate on the same weights, and today they do not.

### 1.1 The coherence problem, restated as a decision

Draft 05 §2.6 established it and correctly refused to settle it. The two thread heads are

```
Thread B (issue #73, all of today's reliability numbers)
    /lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z/ckpt/wm_ft_multi3
    step 69378   (inference.log line 9)

Thread A (PLAN.md Step 2, what attach_adaptation.sh restores)
    /lus/lfs1aip2/projects/public/u6gb/sigma-0/checkpoints_selftrain/j5705912_b30675li_5705912
    step 69378   (attach_adaptation.sh:56)
```

The step numbers coincide and the weights do not. Three ways out, with their costs:

| option | what it costs | what it buys | what it loses |
|---|---|---|---|
| **B-to-A**: regenerate rollouts from the selftrain chain | one G2-sized generation run | one model, both threads, all of today's method transfers | today's 80 members become a method demonstration, not a result about the model being trained |
| **A-to-B**: adapt `wm_ft_multi3` instead | rewrite `attach_adaptation.sh:56`; needs the early checkpoint of *that* run to exist | today's 80 members stay live | the selftrain chain's 17 checkpoint ages (05 §2.5) are the early-vs-late ladder; `wm_ft_multi3` may not have one |
| **Keep separate** | nothing | honest | no shared conclusion; two half-projects |

**M0 was run while writing this file, and it settles the choice.** Two measurements,
both narrow `lfs find`, both seconds:

```
wm_ft_multi3          checkpoint steps: 69378                                    (1 step)
selftrain chain       checkpoint steps: 275 22495 24080 28830 30410 33575 52590
                                        55773 57365 58949 60532 62113 63695
                                        65275 66853 68435 69378                 (17 steps)
```

**`wm_ft_multi3` carries exactly one checkpoint, so it has no early-and-late pair and
A-to-B is impossible.** Not expensive — impossible. A second, independent check confirms it
from the other direction: the largest data blob under `wm_ft_multi3/69378/state/` is
**418.6 MB** against **499.5 MB** for the selftrain chain's step 69378, and the generation
log for every existing member records

```
[Checkpoint] Loading step=69378 from .../ckpt/wm_ft_multi3 (partial_restore=True)
[load_checkpoint] StandardRestore failed (... opt_state[1].inner_states.muon: - Source: MISSING
```

so `wm_ft_multi3` is an **inference-only artefact with no Muon optimizer state**. It cannot
be resumed for training under the optimizer this project uses, whatever its checkpoint count.

That also confirms draft 05 §2.6's assertion directly: the two step-69378 checkpoints differ
in size and in content, so the coinciding step numbers are a coincidence, not an identity.

**Therefore the choice is between B-to-A and keeping the threads separate.** B-to-A means
G2 regenerates rollouts from a selftrain-chain checkpoint, which changes what G2 produces and
therefore the whole of draft 05 §2.4's sizing. Whether that is worth one generation run
depends on M1 (§2.2): if the gen-arm fidelity defect turns out to contaminate the failure
score, today's 80 members need regenerating anyway and B-to-A is free at the margin.

### 1.2 What the merged question would be, if they merge

> Does a checkpoint that has been trained further both (a) learn less from new data at fixed
> budget, and (b) fail on an identifiable, reproducible set of contexts that targeted
> training recovers without damaging the rest?

(a) alone is a paper about plasticity. (b) alone is an engineering result about a data mix.
Only together do they say anything about *continual* learning, which is why the decision in
§1.1 is worth taking rather than deferring.

---

## 2. The ladder

Ordered by cost. `kind` follows draft 05's convention. Every rung names the null control it
requires; **a rung without a null control is not on the ladder**, because this project has
already recorded a case where a mis-paired score was *more* reliable than the correct one
(0.49 against 0.46).

| # | measurement | kind | gates | null control |
|---|---|---|---|---|
| **M0** | Checkpoint-age inventory of both roots | CPU, seconds | the §1.1 decision, therefore everything | none needed — it is an inventory, not an inference. $\color{green}{\textsf{DONE 2026-09-05}}$ |
| **M1** | `fidelity.py` gen-arm replay on the existing 80 members | CPU, hours | whether the failure score is measuring the model or the recording | the real arm, already at 0.9614 exact match, is the control and it is already built |
| **M2** | Corrected-score regeneration null at the horizon actually used | CPU, minutes | how many independent draws `k` really buys | the matched-`k` split-half of 02 §2.2 |
| **M3** | Era gate: base-window contexts vs 2024-08 contexts | GPU generation, CPU scoring | whether "regime shift" is on the table at all | contexts from a second base-window month, same size |
| **M4** | Arm-level repeat: one arm, several training seeds, nothing else changed | GPU | **every** between-arm comparison | it *is* the null; that is the point |
| **M5** | Failure-pool arms vs random-pool arms at matched volume | GPU | the issue #73 answer | the random-pool arm |
| **M6** | Early-vs-late fixed-budget adaptation, AUC | GPU | the `PLAN.md` Step 2 answer | same-age pair: two checkpoints at the *same* step, different seeds |

### 2.1 M0 — checkpoint-age inventory

**Measures**: which steps exist under each of the two candidate run roots, and how many
tokens separate the earliest from the latest.

**Decision rule, pre-registered**: Thread A moves onto `wm_ft_multi3` if that run has at
least two checkpoints whose token separation is at least as large as the selftrain chain's
earliest-to-latest separation. Otherwise Thread B regenerates from the selftrain chain.

**How, without violating the Lustre rules**: read the breadcrumb if one exists; if not,
`lfs find <run_root> -maxdepth 1 -type d`, which is a single narrow directory, and write the
result to a static `steps.json` (draft 05's C3) so nothing scans it again.

$\color{green}{\textsf{Result, 2026-09-05}}$: `wm_ft_multi3` = {69378}; selftrain chain =
17 steps from 275 to 69378. Draft 05 §2.5's enumeration is confirmed. The decision rule above
fires for "otherwise": **Thread B regenerates from the selftrain chain, or the threads stay
separate.** The `steps.json` breadcrumb of draft 05's C3 should be written now so nothing
scans either directory again.

### 2.2 M1 — is the failure score measuring the model, or the recording?

This is the rung that could invalidate issue #73's premise, and it is free.

`src/post_training/heuristic_learning/fidelity.py` reports, on GS 2026-01 over eight windows
of 900 messages: the **real arm** replays the recorded continuation and reproduces the
recorded book exactly in **5 of 8** windows (mean exact match 0.9614); the **gen arm** replays
the *generated* messages against the book the generator recorded while producing them and
reproduces it in **0 of 8** (mean 0.5511). Both arms run identical code over identically
shaped inputs, so an engine defect cannot produce that asymmetry.

**What that means for the pool.** Every failure score computed so far reads a forward return
off a generated path. If the generated message stream does not reproduce the book the
generator itself recorded, then part of the divergence attributed to "the model got this
context wrong" is an artefact of the recording, and it is not known which part.

**Measures**: exact-book-match fraction and first-divergence step, per context, over the same
500 contexts the reliability numbers use, on at least two tickers.

**Decision rule, pre-registered, on the correlation between per-context exact-match fraction
and per-context corrected failure score**:

- |Spearman| < 0.15 → the two are separable; the failure score stands; proceed to M3.
- |Spearman| ≥ 0.30 → **stop**. The pool is partly a ranking of recording defects. Fix
  `fidelity.py`'s class-R/class-O defect first; no continual-training arm is interpretable
  until then.
- in between → report both and split the pool by exact-match fraction, treating the two
  halves as separate strata everywhere downstream.

**Null control**: the real arm, which shares every line of code and differs only in whose
messages are replayed. It is already measured at 0.9614.

**Cost**: draft 05 §1.3 establishes that this is CPU-only — `fidelity.py` imports `jax` lazily
at `:293-295` and `describe_stream` is pure numpy. Draft 05 calls it "the largest piece of
work in this plan that has not yet been recognised as free" and that assessment is correct.

### 2.3 M2 — what `k` actually buys

Draft 02 §2.2 changed this rung. The brief's claim that two regenerations agreeing at
0.81–0.87 sets a ceiling was **wrong in sign**: two whole regenerations agree *more* than two
disjoint seed sets, because a fraction `phi` of members never forked at all (`phi` = 0.976 at
horizon 10, 0.191 at horizon 250). Nondeterminism behaves like a *partial redraw*, which
`1/k` averaging removes; it is not an irreducible floor.

**Measures**: on the corrected score at the horizon the pool will actually use, the matched-`k`
split-half and the regeneration agreement, per ticker, with `phi` reported beside them.

**Decision rule**: the number quoted in any write-up is the **corrected-score** figure
(grand mean 0.742, 0.583 at horizon 250), never the raw-score 0.846. If the pool's horizon is
≥ 150, the fork rate exceeds 70% and the effective number of independent draws is closer to
`k` than to `2k` even when two generations are pooled — say that, do not average it away.

**Null control**: the four pairing nulls of `pairing_nulls()`. Any *new* score proposed for
the pool passes through the same four readings before it is used.

**Cost**: CPU, minutes, on data already on disk. Draft 02 already computed most of it.

### 2.4 M3 — the era gate

Specified in draft 04 §5.2 and adopted here unchanged in design. Two changes to its **size**,
both forced by measurement:

1. **It does not fit in the inode budget as written.** 8,000 contexts × `k` = 20 is 320
   members at 3,007 inodes each = 962,240 inodes against 741,511 free (130%). With draft 05
   §5.3's real-arm dedupe it is 482,240 (65%) and fits. **The dedupe is therefore a
   precondition of M3, not an optimisation.** See 03 §3.
2. Draft 04's contamination bound (3.4 expected previously-seen windows out of 4,000) is
   reported as a number rather than claimed to be zero. Keep that.

**Decision rule** as drafted: indistinguishable ⇒ underlearning dominates and replay is free;
2024-08 systematically worse ⇒ era effect, and 04 §5.3's era-twin matching localises it.

**Null control**: a second base-window month, same size, scored identically. Without it,
"2024-08 differs from 2022-06" cannot be separated from "any two months differ".
**Draft 04 does not specify this control.** Adding it costs one more month of contexts and
it is the difference between an era effect and a month effect.

### 2.5 M4 — the decisive rung

**If only one measurement could run, it is this one.**

Draft 02 §2.3 measured that doing nothing moves an arm-level endpoint by up to 28%, and
recorded that training-seed variance "has **never been measured on this project**". Every
between-arm claim the plan wants to make — every mix ratio, every replay fraction, the whole
of §5 — is a difference between numbers whose own repeat spread is unknown.

**Measures**: one arm, held fixed, run with several training seeds; both reporting coordinates
recorded per seed.

**Decision rule**: the between-arm effect the plan is powered for must exceed the measured
within-arm spread by the margin draft 02 §3.4 specifies. If it does not, the arm grid shrinks
before it runs, not after.

**Null control**: this measurement is the null control. That is why it is decisive and why it
must precede M5.

**Cost**: draft 04 E-4 puts it at 6 seeds of one arm and calls it "small". It is the cheapest
thing on the ladder that can invalidate the most expensive thing on it.

### 2.6 M5 and M6

M5 (the issue #73 answer) and M6 (the `PLAN.md` Step 2 answer) are fully specified in drafts
04 §2 and 05 §2.4 respectively. Two things this file adds:

- **M5's random-pool arm is not optional.** Draft 04 §2.2 includes it. Without it, any
  improvement is confounded with having trained on more data, and this project has already
  published a case where an improvement was fully explained by a mechanical relationship
  rather than by the intervention.
- **M6 needs a same-age control pair.** `PLAN.md` §2.3 correctly rejects fresh-vs-continued,
  but its replacement — early checkpoint against late checkpoint — still has no null. Two
  checkpoints at the *same* step under different seeds, given the same adaptation budget,
  bound how much AUC difference is attributable to nothing at all. `PLAN.md` does not specify
  this and should.

---

## 3. What the failure score should be

Draft 04 §4.2 settles the **return-based** score: stratified ranking of `total`, with the four
pairing nulls mandatory and plug-in CRPS explicitly rejected because its bias is proportional
to ensemble width. That is adopted here without change.

What 04 does not settle, because it was not asked to, is whether a return-based score is the
right axis-1 score at all. There are three candidates and they are not equivalent:

| candidate | what it scores | cost per context | measured status |
|---|---|---|---|
| **return-based** (today) | forward mid-return at 7 horizons vs the realised one | 2 small arrays | fully characterised: reliability, conditionality, decomposition, four nulls |
| **`fidelity.py` exact-book-match** | whether the generated message stream reproduces its own recorded book | one replay through the matching engine, CPU | gen arm 0.5511 over 8 windows; **never computed per context at scale** |
| **LOBbench divergence** | distance between generated and real order-flow statistics over a window | the `bench` pipeline | 19-model zoo: every ratio > 1 (1.24–1.60), last window +40%; **never computed per context** |

**They answer different questions.** The return-based score asks "was the price path wrong".
Exact-book-match asks "is this rollout even self-consistent". LOBbench divergence asks "does
this rollout look like market data at all". Issue #73's two axes map onto the first and the
third; the second is a validity check on both.

**What would decide between them.** Compute all three on the same 500 contexts on two tickers
and cross-correlate them per context. That is one CPU job (M1 already computes the second).

- Rank correlation below ~0.3 between any pair ⇒ they are measuring different failures and
  the pool must be built on the axis the training is meant to fix, named in advance.
- High correlation ⇒ use the cheapest, which is the return-based score.

**A caution that applies to the third candidate specifically.** The zoo result — every model's
divergence ratio above 1, rising to +40% at the last window — says autoregressive drift grows
with rollout length in every model of the family. A pool selected on a length-growing quantity
will fill up with long rollouts. Either fix the rollout length across the pool, or stratify on
it the same way the current score stratifies on realised move size. **Draft 04 does not
mention rollout-length stratification and it is the same defect as the one already fixed.**

---

## 4. Reconciliation with `PLAN.md`

| Step | Status | Why |
|---|---|---|
| **Step 0** — inventory | $\color{green}{\textsf{done, incomplete}}$ | `results/INVENTORY.md` exists; its four open items remain, and M0 adds a fifth (checkpoint ages of `wm_ft_multi3`) |
| **Step 1** — probe code | $\color{green}{\textsf{done}}$ | `code/plasticity_probes.py` + 13 tests. Not yet wired into the training loop, which was always the follow-up |
| **Step 2** — early-vs-late probe | ~~as written~~ **mis-specified, rewrite** | Three defects: (a) the adaptation job 6141106 exits immediately (05 §8.2); (b) it restores the selftrain chain while every failure-pool number comes from `wm_ft_multi3` (§1.1); (c) it has **no null control** — see M6 |
| **Step 3** — CPT pilot, replay × rewarm | superseded by draft 04 | 04 §2–§3 specifies the arms, rewarm, and optimizer-state handling concretely. `PLAN.md`'s grid (peak LR ∈ {0.3, 0.5}× , ρ ∈ {0.05, 0.10, 0.25}) is a subset of 04's and its ρ convention must be checked against 04 §0, which pins ρ as the **replay** fraction |
| **Step 4** — mitigations | unchanged, still correctly gated | Only runs if Step 2 finds decay. Nothing measured since bears on it |
| **Step 5** — multi-size onset law | ~~this round~~ **excluded on evidence** | It is gated on "Step 2 produces a verdict-grade readout at one size", and Step 2 currently produces nothing. It is also the only multi-thousand-GPU-hour item, and the inode budget (03 §3) cannot hold its artefacts |

Two `PLAN.md` claims that survive intact and should be carried into the merged plan verbatim:

- **§2.3** — fresh-vs-continued almost always reports "plasticity absent" through representation
  transfer, so it never decides a plasticity claim. Still true, still the reason M6 exists.
- **§2.4** — every stage reports both coordinates. Draft 04 §7 specifies them exactly.

---

## Open questions

1. ~~`wm_ft_multi3`'s checkpoint ages are unknown.~~ $\color{green}{\textsf{resolved 2026-09-05}}$ —
   one checkpoint, no Muon optimizer state, 418.6 MB vs the selftrain chain's 499.5 MB.
   A-to-B is impossible; see §1.1. **What remains open is which of the 17 selftrain steps
   becomes the early member of the M6 pair**, which depends on token separation that nobody
   has converted from step numbers (draft 05 C1/C2 supply the conversion).
2. **Whether the gen-arm fidelity defect affects the return-based score is genuinely open.**
   The 0.5511 figure is from GS 2026-01, a different ticker and month from the 8 tickers the
   reliability numbers use. M1 measures it where it matters. Nobody should assume the defect
   transfers, and nobody should assume it does not.
3. **The rollout length used to generate the existing 80 members is not recorded in this
   file** and the length-stratification question in §3 cannot be closed without it. It is in
   `inference.log` per member; reading it is a one-line CPU check that has not been done.
4. **M3's second base-window month is proposed here and costed nowhere.** Draft 05's sizing
   does not include it. It changes M3's inode footprint by 50%.
5. **Whether M4 can be run on the selftrain chain before §1.1 is decided.** If Thread A moves
   to `wm_ft_multi3`, an M4 measured on the selftrain chain does not transfer. Either decide
   §1.1 first, or accept that M4 may need repeating.
