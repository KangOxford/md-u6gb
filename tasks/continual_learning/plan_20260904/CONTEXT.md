# Shared context brief for the 2026-09-04 planning round

You are one of ten agents. Five draft a plan; five adversarially review the drafts.
Nobody in this round runs a GPU job, submits sbatch, or attaches to an allocation.
Planning and cheap CPU-side verification only.

## 0. Hard environment rules (violating these has previously suspended the whole group)

- Isambard-AI / Lustre. FORBIDDEN, even once, even "just to look": recursive `ls`,
  `ls -R`, `ls -lR`, `ls -1td`, `find /` or `find /projects` or `find /home`,
  `du -sh` on a large path, `tree`, `locate`, deep recursive globs, `watch ls`.
  Safe: `cat`/`head`/`sed -n` on known exact paths, `grep` on a single file,
  `grep -r --include='*.py'` on a small tree, `lfs find <narrow path> -name ...`,
  and `ls <dir>` only when fewer than ~50 entries are expected.
- NEVER `rm`. This includes scratch, staging and temp directories **you created
  yourself**, and `rm -rf` on a directory you are about to recreate. If something
  must go, `mv <path> <path>_deprecated_$(date -u +%Y%m%dT%H%M%SZ)`. If you need a
  clean staging dir repeatedly, use a fresh unique name each time so nothing needs
  clearing: `BC=BC_$(date -u +%H%M%S); mkdir -p "$BC"`.
- NEVER `scancel`, in any form, with any flag.
- Write full absolute paths everywhere, including in section headings and tables.
  `/projects/...` is a symlink shorthand; expand to `/lus/lfs1aip2/projects/...`.
- All plan prose in English. Chinese is allowed only in an explicit `中文速览` block.
  Figures/tables/axis labels are always English.

## 1. Repositories and paths

| What | Full path |
|---|---|
| This repo (notes/tasks, "md-u6gb") | `/lus/lfs1aip2/projects/public/u6gb` |
| Current branch | `continual-learning-plan` |
| Continual-learning task dir | `/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning` |
| This planning round | `/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/plan_20260904` |
| sigma-0 main checkout | `/lus/lfs1aip2/projects/public/u6gb/sigma-0` |
| sigma-0 worktrees | `/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees` |
| m3 baselines (separate line) | `/lus/lfs1aip2/projects/public/u6gb/sigma-0-m3/baselines/m3` |
| Conda base for training | `/lus/lfs1aip2/projects/public/s5e/quant_team/quant/miniforge3` |

`sigma-0` is `github.com/KangOxford/sigma-0`, PRIVATE. Its PRs are stacked (one PR's
base is another PR's branch). 15 PRs are open right now (#49..#71).

## 2. The research line this plan is for

GitHub issue KangOxford/sigma-0 #73, "The continual learning system. keep training on
the cases it failed to learn". Proposal: mine rollouts that diverge from the true data
into a failure pool, and continue training on that pool.

`/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/PLAN.md` is the standing
plan for the wider plasticity / continual-pre-training line (Steps 0-5). Read it.
`/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/results/INVENTORY.md`
is the Step-0 inventory and lists four open items.
`/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/deep-reseach.md` is the
4971-line research archive behind PLAN.md (read selectively, do not cat it whole).

### What was measured yesterday (commit e8425cb1), on rollouts that already exist

Three prerequisites for the failure pool being "a set of contexts the model handles badly":

1. **Reliability.** Split-half rank correlation of a per-context failure score.
   One rollout per context: 0.36-0.48 raw, 0.15-0.25 corrected. A one-parameter fit
   puts k at roughly 20 rollouts per context for corrected reliability 0.80.
2. **Conditionality.** Raw squared error correlates 0.65 with the size of the realised
   move, and keeps 0.46 of its ranking after rollouts are detached from their contexts;
   on that score even a fully independent permutation of both halves still agrees at
   0.43. Ranking within realised-move bins drops both to 0.03 and 0.10 and keeps genuine
   signal. The two top-decile pools overlap by 40 percent.
3. **Decomposition.** Total squared error partitions exactly into a systematic term and a
   dispersion term; dispersion is 26-34 percent inside the top decile, and the model is
   under-dispersed, so that fraction is a floor on what training can remove.

Also established: split-half reliability alone certifies nothing — a score with a
consistently wrong pairing is as reliable as the correct one (0.49 vs 0.46). Only the
cross-pairing null separates them.

Code and artifacts:
- `/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/code/failure_pool_reliability.py`
- `/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/code/test_failure_pool_reliability.py`
- `/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/code/plasticity_probes.py`
- `/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/code/probe_weights_offline.py`
- `/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/code/attach_adaptation.sh`
- `/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/code/submit_adaptation_pair.sh`
- `/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/results/` (a1_step*.json, failure_pool_reliability.json)
- `/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/failure_pool_reliability.ipynb` + `.html`

## 3. Known-blocking open items (from INVENTORY.md §5)

1. Checkpoint roots for the R1-era 8M (wandb `zkrtl2ef`) and 78M (`pw8u0edj`) runs, and
   for the main-line mamba3 chain. Blocks PLAN Step 2.
2. Which long run retained an early/late checkpoint pair far enough apart. Blocks Step 2.
3. Tokens/step at the production setting — never recorded. Blocks all budget arithmetic.
4. 2025 shard completeness for the secondary stress slice.

Data window is **2022-01 .. 2025-12** only (SquashFS monthly shards at
`/lus/lfs1aip2/projects/public/s5e/quant_team/lob_preproc_sp500_squashfs`). The COVID
slice in the original research draft is out of window and unusable. Primary stress slice
is the 2024-08-05 volatility spike; secondary is 2025-04.

## 4. Documented failure modes of the person you are checking (recur; check for them)

- An effect that shrinks as n grows. Claims were made at n=1 and n=2 three times in one
  session and reversed each time. A standard-deviation from n=2 has one degree of freedom
  and is systematically small. Rule now in force: **no number enters a heading, gets
  bolded, or is called "best"/"cleanest" until n has stopped growing**; report the
  trajectory of the effect against n, not its value at one n.
- Measuring one slice and writing the whole line's conclusion. Fix: write the qualifier
  into the sentence before deciding it holds ("on test_ce, at tail>=0.5, ...").
- A selection rule's consequence mistaken for a property of the data.
- A verdict read at a quantile holding 8 events. Print event counts next to any
  quantile statistic.
- Dividing by a group-dependent constant: shrinks noise, not bias, so t-statistics
  inflate while the real difference stays flat.
- A knob that exists, is set, is logged, and never reaches the code.
- A default that may never have applied — read the last link of the assignment chain.
- A plausible mechanism narrated without reading what actually happened.
- Attributing an improvement in B to training when an already-measured improvement in A
  mechanically implies it.
- A metric's name is not its semantics.

## 5. What the plan must contain

The user's instruction, verbatim in substance:
- Spend most of the effort on the plan, not on execution.
- The plan file is the primary artifact. Keep it updated as work proceeds: mark progress
  in green, strike through anything found to be genuinely wrong.
- Run tasks against the plan, waiting for GPUs as needed. GPU availability is secondary.
- Always profile. Always maintain the plan file. Always run adversarial checks.
  Always look for files to delete (rename away) and code to refactor.
