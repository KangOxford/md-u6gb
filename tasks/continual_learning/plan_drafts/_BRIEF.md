# Shared brief — read this before drafting. Every fact here is measured, not assumed.

You are drafting one facet of a research plan for **continual learning on sigma-0**, a
foundation model of NASDAQ limit-order-book message flow (JAX/Flax, S5/mamba3 state-space
backbone, 26-token message encoding, ~78M params for the run in question). Another four
agents are drafting other facets in parallel; five more will then attack all five drafts.
Write for that adversary.

## The two threads that must merge

**Thread A — plasticity and continual pre-training.** Existing file:
`/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/PLAN.md`. Steps 0-5, of which
Step 0 (inventory) and Step 1 (probe code) are done, Step 2 has only its offline weight
half done. Key design decision already made: plasticity must be judged by comparing an
early and a late checkpoint **of the same run** under identical fixed-budget adaptation,
never fresh-vs-continued.

**Thread B — GitHub issue #73, failure-driven continual learning.** Generate rollouts,
score them on two axes (gap to the true system; distributional match), select the failures
into a pool, continue training on a mix of pool plus historical replay. The issue asks
whether the epoch should be 100% failures or a mix, and floats 70/30.

## What was measured today (2026-09-04), all on CPU, all reproducible

Code: `code/failure_pool_reliability.py` (+ 12 passing tests in
`code/test_failure_pool_reliability.py`). Results: `results/failure_pool_reliability.json`.
Notebook: `failure_pool_reliability.ipynb`. Data: 8 tickers x 500 frozen contexts x 10
independently seeded rollouts x 7 forward horizons, under
`/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z/data/`.

1. **A single rollout per context cannot rank contexts.** Split-half rank correlation of a
   per-context failure score: 0.36-0.48 on raw squared error, 0.15-0.25 on the corrected
   score. A one-parameter fit puts k ~ 20 rollouts per context for reliability 0.80.
2. **The obvious score is mostly not about the model.** Raw `mean_i (x_i - y)^2` correlates
   0.65 with |realised move| and retains 0.46 of its ranking after rollouts are detached
   from their contexts; two independently permuted halves still agree at 0.43. Ranking
   within |realised move| bins drops these to 0.03 and 0.10 while keeping real signal
   (true 0.46 vs zero line 0.10). The naive and corrected top-decile pools **overlap 40%**.
3. **Split-half reliability certifies nothing on its own.** A consistently mis-paired score
   scores 0.49, higher than the correct one at 0.46. Only a cross-pairing null separates
   them.
4. **Error decomposes exactly**: `mean_i (x_i - y)^2 = (xbar - y)^2 + Var_i(x)`. Dispersion
   is 37-45% of total over all contexts and 26-34% inside the top decile (paired, lower in
   8/8 tickers at every horizon but one). The model is under-dispersed (reliability slope
   0.46-0.59) so 26-34% is a floor.
5. **Generation is not seed-reproducible.** Same checkpoint, same seed, same contexts, three
   runs: sd_ratio 1.156/0.916/1.016. Root cause is XLA autotuning; `--xla_gpu_autotune_level=0`
   makes it bitwise reproducible at 1.49x wall clock. Two complete regenerations of one
   config with identical seeds agree at rank correlation 0.81-0.87, so ~15% of rank
   agreement is lost to nondeterminism alone.

## Existing machinery you must plan around, not reinvent

    /lus/lfs1aip2/projects/public/u6gb/sigma-0/src/post_training/heuristic_learning/
        fidelity.py         real-arm vs gen-arm replay; exact book match 0.9614 vs 0.5511
        autopsy.py          locates the step where a replay first disagreed
        episode_builder.py  freezes one minute of exogenous background flow so every
                            candidate faces a byte-identical market (paired comparison)
        mm_sim.py, checkpoints.py, README.md

Benchmarks already wired as skills: `bench` (LOBbench), `return-bench` (IC / ranked IC /
direction accuracy). A 19-model zoo evaluation exists in which **every** model's divergence
ratio exceeds 1 (1.24-1.60) with the last window at +40%, i.e. autoregressive drift is
universal across the family and grows with rollout length.

Known from PR#22: destroying context pairing leaves W1 bitwise unchanged while CRPS moves
12.8-20.0%, so the acceptance criteria used on this line so far are **unconditional**
statistics. Plug-in CRPS from K members is biased by `E|X-X'|/(2K)`, proportional to
ensemble width, so it rewards narrow sampling if used as a training target.

## Hard environment constraints (violating these has suspended the whole group before)

- Isambard-AI, Lustre. **Forbidden**: `find` from broad roots, `ls -R`, `ls -1td`, `du -sh`,
  `tree`, deep globs, `watch ls`. Use `lfs find` on a narrow path or read a breadcrumb file.
- **Never `rm`.** Rename with a timestamp suffix instead. This includes scratch files you
  created yourself and files you are about to recreate.
- **Never `scancel`** unless training is provably dead (nan loss, no step progress); report
  the job id to the user instead.
- Checkpoints and logs go to `$TMPDIR` / node-local, rsync back at the end. Per-job output
  directories keyed by `$SLURM_JOB_ID`. Resume reads `latest_checkpoint.json`, never `ls`.
- `gtop` before any `sbatch`; if any GPU in an allocation this account already holds is idle,
  attach with `srun --overlap --gres=gpu:4 --cpu-bind=none` and pick the free card inside the
  process with `CUDA_VISIBLE_DEVICES`. Judge idle by **per-card memory** (1-9 MiB), never by
  gtop's header count. Free cards cost nothing, so "this would use GPU time" is never a
  reason not to do something worth doing.
- `GRAD_ACCUM_STEPS` must be derived from a declared effective batch size, never defaulted.
- Log `step_loss` every ~1 min and checkpoint every ~15 min; these are different frequencies.
- Node-local scratch dies when the allocation expires; retrieve artefacts per run, not per batch.

## Failure modes this project has already committed, which the adversary will look for

- A knob that is set, printed, and recorded but never reaches the code (`TOKEN_MODE` was
  pinned to 26tok in five places, four silently).
- A default value that never applied because something downstream overwrote it.
- A metric whose name is not its semantics (`rollout/rewards` was normalised advantage).
- Dividing by a per-group constant shrinks noise but not bias, manufacturing significance.
- An effect that shrinks as n grows, because the power calculation used an effect size
  estimated from the same small sample.
- Claiming a whole line from one measured slice.
- A selection rule's consequence mistaken for a property of the data.
- A null control that shares its error with the treatment and therefore is not null.
- Estimating the noise floor on a different structure from the effect (unpaired vs paired).
- A verdict read at a quantile with 8 events.

## What to write

Write **one markdown file** at the path given in your task, in **English**, opening with a
short `## 中文速览` bullet list. Be concrete: name files, commands, parameters, sample
sizes, and the decision rule that each measurement feeds. Where you do not know something,
say so explicitly under a `## Open questions` heading rather than inventing it — the
adversarial pass rewards flagged gaps and punishes confident invention. Do not run GPU
jobs. Reading files and running short CPU checks is fine and encouraged.
