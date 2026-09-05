# 03 — Preconditions, provenance, profiling, and the delete/refactor sweep

> Written by the main session on 2026-09-05 after the drafting agent for this facet was killed
> by a session limit before it wrote anything. Every number below was measured while writing,
> with the command shown. Drafts 02, 04 and 05 landed complete and are fixed inputs; draft 05
> §5 already covers part of the inode question and is extended rather than repeated here.

## 中文速览

- **inode 是整份计划最硬的约束，而且现在的余量是清理工作腾出来的，不是常态。** 2026-09-04
  17:54Z 项目距硬顶只剩 **118 个** inode；到 09-05 02:55Z 释放出 741,511 个。
- **04 §5.2 的 gate 和 §8 的 E-3 都放不下。** 一个 rollout member = **3,007 inode / 67 MB**
  （实测）。gate 要 320 个 member = 962,240 inode = 余量的 130%；E-3 要 640 个 = 260%。
  用 05 §5.3 的 real-arm 去重后 gate 降到 65%（放得下），E-3 仍是 130%（**放不下**）。
- **分析真正读的只有每个 member 里 112 KB 的 `.npz`**，那 3,000 个 CSV 是给 `fidelity.py`
  和直方图用的。所以布局的答案是「CSV 留在节点本地、只把 npz 和一份打包回 Lustre」，
  不是「打包丢掉」（05 §5.4 已论证 `PACK_MEMBER=0` 必须保持）。
- **代码没有实质重复**——三个模块关注点不相交。真正的缺口是
  `probe_weights_offline.py` **没有任何测试**，而它产出了 `results/a1_step*.json`。
- **确定性的边界被 02 §2.2 改写了**：非确定性不是天花板，是「部分重抽」，`1/k` 能平均掉它。
  所以 `--xla_gpu_autotune_level=0` 该开在**要做逐位比较**的地方，不是所有评测路径。

---

## 1. What must exist before any measurement counts

Each row names the file or command that satisfies it. A row with no artefact is not satisfied
by intention.

| # | precondition | satisfied by | status |
|---|---|---|---|
| P1 | A rollout manifest written **at generation time**, before the first member | §2 below; nothing writes it today | **absent** |
| P2 | A frozen, hashed context set shared by every member | draft 05 C4; `sample_indices_rank0.json` exists per member but is not shared or hashed | **partial** |
| P3 | A checkpoint breadcrumb so no resume path scans a directory | draft 05 C3; `steps.json` + `latest_checkpoint.json`; 01 §2.1 measured the steps, nothing has written them down | **absent, inputs now known** |
| P4 | A held-out benchmark frozen across cycles | draft 04 §7; not yet cut | **absent** |
| P5 | Deterministic generation wherever a bitwise comparison is claimed | §5 below | **absent** |
| P6 | An inode write plan that fits the budget | §3 below | **absent, and the plan as drafted does not fit** |
| P7 | Run-time assertions for every knob the design depends on | draft 04 §6 specifies twelve; none are wired | **absent** |
| P8 | A test for every module that produces a committed result | §4 below; `probe_weights_offline.py` has none | **partial** |

**P1, P2 and P6 are the ones that block G2**, because all three are properties of how the
first new member is written, and none can be retrofitted onto members that already exist.

---

## 2. The rollout manifest, field by field

Draft 04 §4.5 specifies a **pool** manifest — which contexts were selected and why. This is
the layer below it: what a **generation run** must record so that a later run can be proved
exchangeable with it. The two are different objects and both are needed.

The failure this prevents has already happened on this project: an archive recorded which
contexts were scored and nothing at all about how the rollouts were produced, so when the
analysis concluded "k = 16 would settle it", the extension could not be done — not for want
of GPUs, for want of a record.

```
manifest_version    int        1
written_at_utc      str        ISO-8601, written BEFORE the first member, not after
run_tag             str        e.g. "v5me3"
code_commit         str        40-hex of the sigma-0 checkout that generated this
code_dirty          bool       true if the working tree differed from that commit

checkpoint_root     str        absolute path, /lus/... expanded, never /projects/...
checkpoint_step     int
params_sha256       str        of the restored parameter tree, not of the files
optimizer_state_present  bool  wm_ft_multi3 restores with muon MISSING; that must be recorded
partial_restore     bool

token_mode          str        asserted from what the loader yields, not from the environment
n_cond_msgs         int        conditioning window length
n_gen_msgs          int        rollout length -- the pool can be biased by this alone
horizons            list[int]  [10, 25, 50, 100, 150, 200, 250]

context_file        str        path to the shared frozen index file
context_file_sha256 str        content hash; two runs with different hashes are not comparable
n_contexts          int

seed0               int
seed_stride         int        without it, "10 seeds" does not identify which 10
k                   int        members per context in THIS run
batch_size          int
xla_flags           list[str]  verbatim, including whether autotune was disabled
jax_version         str
platform            str

real_arm_written    bool       false when the real arm is referenced from a shared copy
real_arm_path       str|null   where the shared copy lives, when not written here
```

**Two rules about absent values.** Write `null`, never the string `"unknown"`: `null` is
falsy and fails a check, while `"unknown"` is a truthy string that passes every `if
manifest.get(field):` and silently certifies a run that recorded nothing. And write the
manifest **before** the first member, so a run killed halfway still says what it was doing.

**Where it goes**: one manifest per generation run at the run root, plus a copy of its
sha256 inside every member directory, so a member that gets moved still points at its origin.

---

## 3. Profiling — measured, not estimated

### 3.1 The CPU analysis

```
$ /usr/bin/time -f "%e s" python3 code/failure_pool_reliability.py --draws 20 --out /dev/null
    3.95 s, 4.00 s, 3.96 s        (three runs; 8 tickers x 500 contexts x 10 seeds x 7 horizons)
$ python3 -m pytest code/test_failure_pool_reliability.py -q
    12 passed in 0.15 s
```

The entire measurement that produced yesterday's three conclusions is **four seconds of CPU**.
That is the ratio worth keeping in view when sizing anything else in this plan.

### 3.2 What a rollout member costs

Measured on `.../crps_return_alignment_20260808T025024Z/data/hp_v5me3_AMD_s97702/member_0/`:

```
     500  *_refcheck_real.csv        \
     500  *_orderbook_real.csv        |  the real arm: byte-identical across seeds (05 5.3)
     500  *_message_real.csv         /
    ~500  *_orderbook_*_gen.csv      \
    ~500  *_message_*_gen.csv         |  the generated arm
    ~500  *_provenance_*_gen.csv     /
       4  .returns{,_multih}_{real,gen}.npz      112 KB total
       1  sample_indices_rank0.json              11,506 bytes
       1  inference.log,  1 .done,  3 dirs
  ------
   3,007  inodes,  67 MB apparent
```

**The reliability analysis reads only `.returns_multih_{real,gen}.npz` — 112 KB of the 67 MB.**
The CSVs are not dead weight: `fidelity.py`, `autopsy.py` and the spread-regime histogram all
read them, which is exactly why draft 05 §5.4 is right that `PACK_MEMBER=0` must stay. But
they do not all need to live on Lustre for the lifetime of the project.

### 3.3 The inode budget, and the fact that the headroom is borrowed

The watcher log records the project inode count every 900 s, which makes it a time series.
Extracted over the last 40 samples:

```
2026-09-04 17:54Z   51,199,882 / 51,200,000      <- 118 inodes from the hard cap
2026-09-05 02:55Z   50,458,489 / 51,200,000      <- 741,511 free
```

The direction matters and is easy to misread: inodes are being **released**, not consumed.
The 741,511 free inodes are the product of an active cleanup effort over those nine hours,
not a stable baseline. Planning against them as if they were permanent headroom is the same
error as planning against a GPU that happens to be idle.

`/home` (VAST) has 22.68 billion free inodes and is the escape hatch for logs and watcher
output. It is **not** an escape hatch for artefacts the repository must keep.

### 3.4 The plan does not fit

```
per member, as written today        3,007 inodes
per member, real arm deduped        1,507 inodes      (05 5.3: real arm is byte-identical
                                                       across seeds, md5-verified)
free now                             741,511 inodes
```

| what | rollouts | members | as written | deduped |
|---|---:|---:|---|---|
| 04 §5.2 era gate (8,000 contexts × k=20) | 160,000 | 320 | 962,240 = **130% OVER** | 482,240 = 65%, **fits** |
| 04 §8 E-3 cycle-1 pool | 320,000 | 640 | 1,924,480 = **260% OVER** | 964,480 = **130% OVER** |
| 05 §5.3 G2 at k=20, 8 tickers | 40,000 | 80 | 240,640 = 32% | 120,560 = 16% |

Three consequences, all forced:

1. **The real-arm dedupe is a precondition, not an optimisation.** Without it the era gate
   alone exceeds the budget. Draft 05 §5.3 proposes it and draft 04's sizing does not assume
   it; that is a contradiction between drafts and this is its resolution.
2. **E-3 as written cannot run.** 320,000 rollouts do not fit even deduped. Either the pool
   shrinks, or contexts-per-member rises above 500 so the same rollouts cost fewer inodes,
   or the generated CSVs stop being written to Lustre. The third is the only one that does
   not change the science, and it is available: generate into node-local `$TMPDIR`, run the
   CSV consumers (`fidelity.py`, the histogram) **in the same allocation**, and rsync back
   only the `.npz`, the manifest, and the derived per-context scores. That is the pattern
   already mandated for checkpoints by the repository's own rules.
3. **Any sizing table in drafts 04 and 05 that predates this section is provisional.** The
   free-inode figure moved by 741,393 in nine hours; a budget quoted without its timestamp
   is not a budget.

---

## 4. Delete and refactor

### 4.1 There is no significant duplication, which is itself the finding

```
code/failure_pool_reliability.py   459 lines   scoring, ranking, nulls, extrapolation
code/plasticity_probes.py          222 lines   dormant fraction, effective rank, readiness, Hessian
code/probe_weights_offline.py      123 lines   Orbax restore + spectral norms
code/attach_adaptation.sh           79 lines
code/submit_adaptation_pair.sh      54 lines
code/test_failure_pool_reliability.py  146     12 tests
code/test_plasticity_probes.py         129     13 tests
```

The three Python modules share no functions. The expected duplication — a rank routine, a
loader, a JSON writer — is present in exactly one of them each. **So the refactor to do is
not de-duplication.** Four things that are worth doing, in order:

1. **`probe_weights_offline.py` has no test file, and it produced committed results**
   (`results/a1_step275.json`, `a1_step33575.json`, `a1_step69378.json`). Those three JSONs
   are the entire weight half of the plasticity readout. A module whose output is quoted and
   whose behaviour is unasserted is the shape of every silent-defect story in this project's
   history. Write `code/test_probe_weights_offline.py` with, at minimum: a synthetic
   parameter tree whose L2 norm and spectral norms are known in closed form; a check that
   `flatten` visits every leaf exactly once; and a check that the embedding matrix is
   excluded from the non-embedding norms.
2. **`spearman` and `_rank` belong in a shared module the moment a second consumer appears.**
   Today there is one, so moving them now is speculative. The trigger to write down: when the
   plasticity readout needs a rank correlation, extract both into `code/rankstats.py` together
   with their four existing tie/endpoint tests, and leave a re-export so nothing breaks.
3. **`failure_pool_reliability.py` has to become importable from inside `sigma-0`.** Draft 04
   §6 requires run-time assertions in the training loop that reference the same scoring and
   stratification code. A scoring module that lives only in this notes repository cannot be
   imported by a training job without a `PYTHONPATH` hack, and `PYTHONPATH` hacks are how
   `TOKEN_MODE` came to be pinned in five places. Move the module to
   `sigma-0/src/post_training/` and keep this repository's copy as a thin import.
4. **Rename away, do not delete** (this repository's rule, and `mv` needs no approval):
   - `results/failure_pool_reliability_total_superseded_20260904T212643Z.json` — already
     renamed, keep as the audit trail of the pre-stratification run.
   - `code/__pycache__/` — regenerated, harmless, but it is inodes in a project at its cap.
   - `figs/fp_fig*.png` — six PNGs that are also embedded in the notebook. They are the
     inspection exports, not deliverables. Rename them under `figs/_inspection_<stamp>/`
     so it is clear they are not the published figures.

### 4.2 What must not be deleted

`PACK_MEMBER=0` stays, for the reason draft 05 §5.4 gives: packing discards `data_cond/`,
`data_tokens/` and the `message_*.csv` files, and the last of those feeds the spread-regime
histogram that is the second of issue #73's two axes. The docstring calling those files
"raw material that has already been consumed" is true for CRPS and false for every other
consumer, and the default made the false half authoritative.

---

## 5. The determinism boundary

Draft 02 §2.2 changed what this decision is about, and the boundary has to move with it.

**Old reading** (the brief's, and mine yesterday): nondeterminism destroys ~15% of rank
agreement irreducibly, so pin `--xla_gpu_autotune_level=0` on the whole evaluation path.

**What was actually measured**: two whole regenerations agree *more* than two disjoint seed
sets, at every horizon. A fraction `phi` of members never fork at all — 0.976 at horizon 10,
0.191 at horizon 250 — and the rest behave like a partial redraw, which `1/k` averaging
removes exactly as it removes sampling noise. It is not a floor.

**So the flag is needed where a claim is bitwise, and not where a claim is statistical.**

| where | flag | why |
|---|---|---|
| Any run whose output is compared **bitwise** to another run — reproducing an archived result, proving two members are exchangeable, a guard test that asserts equality | **on** | without it the comparison is meaningless; with it, 13/50 becomes 50/50 |
| A generation run whose members are **averaged** into a per-context score | **off** | `1/k` already removes the fork noise, and the flag costs 1.49× wall clock, which at 320 members is the difference between fitting in an allocation and not |
| Training | **off** | training averages over far more noise than this; and 02's finding means the flag buys nothing statistical |

**What breaks if the boundary is drawn wrong.** Drawn too tight (flag off everywhere): any
claim of the form "these two archives are the same" is unfalsifiable, and the exchangeability
check in §2 cannot run. Drawn too loose (flag on everywhere): every generation costs 49%
more for a benefit that averaging already provides, and the era gate stops fitting in a
single allocation.

**The flag must be recorded in the manifest either way** (§2, `xla_flags`), because a member
generated with it and a member generated without it are not bitwise comparable even though
both are statistically valid.

---

## 6. Guard tests that can go red

One per precondition. Each is written so that the violation it targets makes it fail, which is
the only property that matters — this project has shipped a self-test that could not fail
because `from x import NAME` early-bound past the monkeypatch, so **every guard below
references `module.NAME`, never a name imported into the test's own namespace.**

| guard | fails when |
|---|---|
| `test_manifest_written_before_first_member` | the manifest's `written_at_utc` is later than the mtime of any member directory |
| `test_manifest_has_no_unknown_strings` | any field equals `"unknown"`, `"n/a"`, or `""` instead of `null` |
| `test_context_file_hash_matches` | the recorded `context_file_sha256` differs from the hash of the file on disk |
| `test_members_are_exchangeable` | two runs' manifests differ in any of `checkpoint_step`, `params_sha256`, `token_mode`, `n_cond_msgs`, `n_gen_msgs`, `context_file_sha256`, `horizons` — the fields whose disagreement makes pooling invalid |
| `test_seed_set_is_identified` | `seed0` and `seed_stride` are absent, so "10 seeds" does not say which |
| `test_real_arm_reference_resolves` | `real_arm_written` is false and `real_arm_path` does not exist — the dedupe of §3.4 turning into data loss |
| `test_effective_batch_is_derived_not_defaulted` | `GRAD_ACCUM_STEPS` appears with a `:-` default anywhere in a launch script |
| `test_token_mode_matches_the_loader` | the manifest's `token_mode` disagrees with the token count the dataloader actually yields |
| `test_no_rm_in_any_script` | a tracked script under `code/` contains `rm ` outside a comment |
| `test_probe_weights_norms_are_exact` | the closed-form L2 and spectral norms of a synthetic tree are not reproduced |

The first four are the ones that would have prevented the archive that could not be extended.

---

## Open questions

1. **Whether generated CSVs can be consumed inside the generating allocation.** §3.4's third
   option — the only one that does not shrink the science — assumes `fidelity.py` and the
   spread-regime histogram can run in the same allocation that generated the rollouts, on
   node-local storage. Draft 05 §1.3 establishes both are CPU-only, which makes it plausible,
   but nobody has run them against a node-local `data_gen/` and the wall clock is unmeasured.
2. **The real-arm dedupe's effect on the CSV consumers.** Draft 05 verified the real arm is
   byte-identical across seeds by md5 on one file pair. It does not say whether `fidelity.py`
   resolves a symlinked or referenced `data_real/`, or whether it assumes a local directory.
   One read of its path handling settles it; it has not been done.
3. **Which of the 17 selftrain steps is the early member of the M6 pair.** 01 §2.1 measured
   the step numbers; converting them to tokens needs draft 05's C1 and C2, and C2 requires a
   wandb config fetch that has not been made.
4. **Whether `params_sha256` is cheap on an Orbax OCDBT checkpoint.** The manifest asks for a
   hash of the restored parameter tree rather than of the files, because the file layout
   differs between the two roots (14 files each, but 418.6 MB vs 499.5 MB largest blob). The
   cost of restoring on CPU purely to hash has not been measured.
5. **Nothing here has been reviewed adversarially.** The five reviewers specified in
   `_REVIEW_BRIEF.md` were never launched. This file and 01 were written by the same session
   that produced the numbers they rest on, which is precisely the arrangement the standing
   order exists to prevent.
