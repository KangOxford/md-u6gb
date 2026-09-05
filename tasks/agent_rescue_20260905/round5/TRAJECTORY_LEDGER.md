# Read-only provenance and completeness ledger for the existing trajectory runs

Read-only. Nothing was started, attached to, or cancelled. Root:
`/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z/ckpt/`

**A directory is not a trajectory, and a `train_seed` field is not an independent
completion.** This ledger separates the two.

## 1. Inventory: 97 directories, 12 usable

| family | dirs | `max_step` values | `complete: true` | distinct seeds | dirs holding checkpoints |
|---|---|---|---|---|---|
| `wm_ft_traj_s*` | 45 | 900 ×7, 1500 ×18, **4800 ×13**, none ×7 | 33 | 37 of 38 | 38 |
| `wm_ft_traj1_r*` | 24 | 2400 ×24 | 24 | 24 of 24 | 24 (as `ckpt_latest`) |
| `wm_ft_traj3_s*` | 14 | 450 ×1, 1500 ×7, none ×6 | 2 | 8 of 8 | 8 |
| `wm_ft_r3rep_s*` | 8 | 1500 ×8 | 8 | 8 of 8 | 8 |
| `wm_ft_traj_r*` | 6 | 1050 ×3, 1500 ×2, none ×1 | **0** | 3 of 5 | 5 |

**`max_step` ranges over 450, 900, 1050, 1500, 2400, 4800.** A run stopped at 1500 is not a
replicate of a run to 4800, so the 97 are not 97 replicates of anything. **14 carry no
`ft_progress.json` at all**, so for those even the seed is unknown.

**Deduplicated, the only set matching the `wm_ft_multi3` / `wm_ft_multi4` budget (4800 steps,
complete) is 12 runs**: `wm_ft_traj_s30 … s41`, train seeds 30–41, **all distinct, none
repeated**, 32 checkpoints each, synced from node-local storage between 02:33:22Z and
02:48:23Z on 2026-09-05 across nodes nid010234/010288/010308/010488. One seed is duplicated
inside the wider `wm_ft_traj_s*` family (37 distinct of 38 recorded), which is why the dedup
step is not optional.

## 2. Two corrections to what this session said earlier

1. **`wm_ft_traj1_r*` are not missing their checkpoints.** Round 4 reported "0 checkpoints" for
   all 24. That was a defect in my own counter, which looked for `step_NNNN` or `*_stepN`
   directories; these runs keep a single `ckpt_latest/`. `wm_ft_traj1_r60/latest_checkpoint.json`
   records `{"step": 2400, "synced_at": "2026-09-05T07:06:43Z", "node": "nid010851"}` and the
   directory holds the orbax payload. **Withdrawn.**
2. **`s_trajectory` cannot be estimated today at zero GPU cost.** Round 4 asserted it could.
   It cannot: see §4.

## 3. Configuration provenance — recorded for 24, missing for the 12 that matter

`ft_progress.json` carries `argv`, `lr`, `anchor_lambda`, `clip`, `start_step` and `run_id` for
**24 dirs only** (the `wm_ft_traj1_r*` family). For the other 73, including all 12 at 4800, the
file carries only `arm`, `ckpt`, `train_seed`, `max_step`, `step`, `complete`, `hold_gen`,
`hold_real`. **The training configuration of the 12 is not recorded in any artefact under their
directory**, so it cannot be verified equal to `wm_ft_multi3`/`wm_ft_multi4` from the data.

What *is* recorded, for `wm_ft_traj1_r60`:

```
--ckpt   .../sigma-0/checkpoints_selftrain/j5705912_b30675li_5705912  --step 69378
--weights .../v5m_weights.npz   --prefix v5m
--train-seeds 96002 96003 ... 96075      (32 values)
--hold-seeds  96000 96001 ... 96071      (16 values)
--lr 1e-5  --anchor-lambda 1.0  --clip 1.0  --epochs 1  --micro 2  --max-step 2400
--out /local/user/1483804540/traj1_r60_6324130_279472/wm_ft_traj1_r60
```

That seed family (96xxx) and that budget (2400) are **not** those of the published line, whose
generation seeds are 97701–97710 and whose runs go to 4800. So `wm_ft_traj1_r*` is a different
experiment, not a replicate of round 3 or round 4.

## 4. Matching evaluation: none exists

| where a score would live | traj mentions |
|---|---|
| `/lus/lfs1aip2/projects/public/u6gb/nb_build_pr22/sweep_results.jsonl` | **0** — its `arm` values are only `multi4` (194), `unifw` (111), `multi3` (8) |
| `/home/u6gb/kangli.u6gb/crps_panel.jsonl` | **0** |
| `<task root>/data/` rollout directories | **0** |

**Twelve trained trajectories, zero scored.** Estimating `s_trajectory` on the published
estimand therefore requires generation and scoring — 12 trajectories × 8 tickers at ~17.2 min
per cell — which is GPU work. **This ledger does not start it and nothing here authorises it.**

## 5. The one free signal, and exactly what it is not

`ft_progress.json` records an in-training holdout readout at the last step. Across the 12:

| | value |
|---|---|
| mean `hold_gen / hold_real` at step 4800 | 1.0353 |
| **sd across the 12 trajectories** | **0.0006** |
| range | 1.0342 … 1.0363 |
| `wm_ft_multi3` | **1.0405** |
| `wm_ft_multi4` | **1.0381** |

**This sd is not `s_trajectory`.** It is the dispersion of a different quantity, measured on
the in-training holdout rather than on 8 tickers × 20 days, and reading 0.0006 as the
trajectory rung would repeat the error of comparing statistics computed on different domains.

`wm_ft_multi3` (1.0405) and `wm_ft_multi4` (1.0381) both fall **outside the range spanned by
all twelve** (1.0342 … 1.0363). **Corrected 2026-09-05 (round 6): that fact on its own does
not establish a different configuration**, and an earlier draft of this section leaned on it
too hard. Two runs lying outside the range of twelve others is a comparison of 2 against 12 on
a quantity whose between-run distribution is unknown, and at least three ordinary explanations
fit it without any configuration difference: the two published arms were trained weeks earlier
on a different code state; they are `reconstructed_from` entries rather than fresh writes; and
a range from twelve draws is not a tolerance interval. **Keep it as an anomaly lead, not as
evidence.**

What stands without it is §3: the configuration of the 12 is **not recorded in any artefact**,
so they cannot be *shown* to be replicates of the published arms either. Unrecorded is the
finding; different is a hypothesis.

## 6. Ownership

Everything runs under the shared account `kangli.u6gb`, so `sacct` cannot attribute a line to a
person. What can be said:

| job | name | state | start | elapsed | nodes |
|---|---|---|---|---|---|
| 6317365 | `u6gb-4-node-chain` | **CANCELLED** | 2026-09-04T21:40:32 | 07:59:49 | 4 |
| 6324130 | `u6gb-4-node-chain` | RUNNING | — | — | 4 |

The 12 at 4800 came from **6317365, which was itself cancelled**; their checkpoints synced at
02:33–02:48Z, inside that job's window. `wm_ft_traj1_r*` came from **6324130, still running**,
with four `run_topup.sh` processes live in the `tailfix-20260902` worktree.

**No trajectory training was started, attached to, or cancelled by this session, and none will
be.** The owner of the `wm_ft_traj*` line should be asked what it is measuring before anything
is planned against it.

## 7. Consequences for the 354 node-h plan

**Void, and not to be revived by these numbers.** The plan was priced on "the trajectory rung
has never been measured"; the correct statement is now "trajectories exist, their
configuration is unrecorded, none is scored, and a different line already owns them". Neither
sentence licenses launching 6 or 24 new trajectories. The next action is a question to the
owner, not an allocation.
