# Scripts Folder — 现状分析 & 重构设计

**Notion task**: "scripts folder" (page 37312c45) under "refactoring the code base" (36f12c45)
**Mode**: ANALYSIS / DESIGN ONLY — nothing moved or deleted. UTC 2026-06-05.
**Target repo**: `/projects/public/s5e/quant_team/quant/AlphaTrade/LOBS5/` (public mirror of kangli.s5e clone)

> Access caveat: current user `kangli.u6gb` (group `brics.u6gb`) can `ls` the repo root
> (parent dir is 777) but CANNOT read several key files: `train_full_autoreg.batch` and
> `node_wrapper.sh` are `-rw-------` owned by `aramis.s5e`; `scripts/`+`bin/` subdirs are
> `drwxrws---` group `brics.s5e`. The live call-chain below is sourced from the authoritative
> project `CLAUDE.md`, not from reading the locked files.

---

## 1. Current state (Q1)

53 shell/batch scripts are flat in the repo ROOT. The `scripts/` and `bin/` subdirs exist but
are empty placeholders (and group-locked). Taxonomy by filename:

| Category | Count | Files | Status |
|----------|-------|-------|--------|
| Training (entry) | 2 | train_full_autoreg.batch, train.sh | LIVE (canonical) |
| Launcher / lib | 1 | node_wrapper.sh (conda+CUDA+NCCL+XLA env) | LIVE (sourced by every job) |
| Inference | 3 | run_inference.batch, experiments_inference.sh, gdn_infer_wrapper.sh | LIVE (used by /bench) |
| Eval | 3 | eval_per_field.batch, eval_per_token.batch, validate_scaling_law.batch | LIVE |
| K3/KDA scale tests | 11 | test_k3_d512_{2,4,8,16,32}node.batch, test_k3_{tp2,tp2_2node,multigpu,smoke,s500_d128_2node,d512_L208k_2node}.batch | experiment leftover |
| K3/KDA wrappers | 8 | run_k3_d512_{2,4,8,16,32}node_wrapper.sh, run_k3_{2node,s500_d128_2node,d512_L208k_2node}_wrapper.sh | 1:1 dup of above |
| Profiling | 5 | profile_{k3_ncu,muon_ns,siso,tp2_ncu}.batch, measure_memory.batch, cuda_kernel_bench.sh | one-off |
| Infra/correctness tests | 6 | test_{cuda_ffi_gates,nccl_fix,node_health,python_cmd,srun,tp_equivalence}.batch | debug throwaway |
| Sweeps | 4 | phase_a_sweep.sh, phase_b_sweep.sh, phase_b_extended_sweep.sh, run_lobbench_sweep.sh | sweep |
| Data (SP500) | 4 | compress_sp500_{array,tickers}.batch, decompress_sp500_array.batch, zst_smoke_test.batch | data-prep one-off |
| Junk | 1+ | malformed filename `sbatch --nodes=256 --time=24:00:00` (botched redirect) | DELETE |

### Live path (authoritative, from project CLAUDE.md)

```
sbatch train_full_autoreg.batch        # env vars + model config + data paths + auto-resume logic
  └→ srun (1 task/node) → node_wrapper.sh   # conda activate, CUDA 12.6, custom NCCL 2.29.3,
       │                                       AWS OFI NCCL 1.18.0, XLA flags
       └→ python run_train.py → lob.train.train()

eval:      eval_post_training.batch (single node)
benchmark: lob_pipeline/_integrated.batch (separate repo, driven by /bench skill)
```

Env vars override model config: `D_MODEL, N_LAYERS, BLOCKS, SSM_SIZE_BASE, PER_GPU_BSZ,
CURTAIL_EPOCHS, NO_VALIDATION, HIERARCHICAL`. Multi-node auto-enables 2D mesh AllReduce.

### Three root causes (not "too many files")

1. **Per-scale duplication** — `test_k3_d512_{2,4,8,16,32}node.batch` + 5 matching wrappers =
   10 files that should be 1-2 scripts parameterized by `NODES=$N`. Node count belongs to
   `sbatch --nodes=`, not the filename. This is the main source of the explosion.
2. **Entry point and library mixed** — `node_wrapper.sh` (shared env setup, a library) sits in
   the same flat layer as dozens of `.batch` entry points; no `lib/` isolation.
3. **Permission / ownership chaos** — same dir mixes aramis `0600` + kangli `0777` + group-only
   `0770`. Collaborators cannot read key files (the canonical train script is unreadable here).

---

## 2. Redesign (Q2)

### Principle: classify by *role in the workflow*, isolate *library* from *entry points*,
### collapse *per-scale duplicates* into *parameterized* scripts.

```
LOBS5/
├── train_full_autoreg.batch        # OPTIONAL: keep one canonical entry symlink at root for muscle memory
└── scripts/
    ├── lib/                        # sourced, never sbatch'd directly
    │   ├── node_wrapper.sh         # conda + CUDA + NCCL + XLA (the one launcher)
    │   ├── env.sh                  # shared exports (CPATH, NCCL_BUFFSIZE, paths)
    │   └── slurm_common.sh         # shared #SBATCH defaults, job-name helper
    ├── train/
    │   ├── train_full_autoreg.batch    # the ONE canonical training entry (env-parameterized)
    │   ├── submit_scaling_law.sh       # batch-submit the 5 model sizes
    │   └── train.sh
    ├── eval/
    │   ├── eval_post_training.batch
    │   ├── eval_per_field.batch
    │   ├── eval_per_token.batch
    │   └── validate_scaling_law.batch
    ├── inference/
    │   ├── run_inference.batch
    │   ├── experiments_inference.sh
    │   └── gdn_infer_wrapper.sh
    ├── sweep/
    │   ├── phase_a_sweep.sh
    │   ├── phase_b_sweep.sh          # merge phase_b + phase_b_extended via a MODE flag
    │   └── run_lobbench_sweep.sh
    ├── data/
    │   ├── compress_sp500.batch      # merge array+tickers via a TARGET flag
    │   ├── decompress_sp500.batch
    │   └── zst_smoke_test.batch
    ├── profile/
    │   ├── profile_ncu.batch         # merge profile_k3_ncu + profile_tp2_ncu via a KERNEL flag
    │   ├── profile_muon_ns.batch
    │   ├── profile_siso.batch
    │   └── measure_memory.batch
    └── test/                         # smoke / correctness / scaling
        ├── test_k3.batch             # ONE script, NODES=$N TP=$T D_MODEL=$D — replaces 19 files
        ├── test_nccl_fix.batch
        ├── test_node_health.batch
        ├── test_srun.batch
        ├── test_tp_equivalence.batch
        └── test_cuda_ffi_gates.batch
```

### Concrete consolidations (the high-leverage wins)

| Today | After | Saving |
|-------|-------|--------|
| test_k3_d512_{2,4,8,16,32}node.batch + 5 wrappers + multigpu/smoke/tp2/s500/L208k | `test_k3.batch` driven by `NODES/TP/D_MODEL/SEQ_LEN` env | 19 → 1 |
| profile_k3_ncu + profile_tp2_ncu | `profile_ncu.batch KERNEL=k3|tp2` | 2 → 1 |
| phase_b_sweep + phase_b_extended_sweep | `phase_b_sweep.sh MODE=base|extended` | 2 → 1 |
| compress_sp500_array + compress_sp500_tickers | `compress_sp500.batch TARGET=array|tickers` | 2 → 1 |

Net: ~53 root scripts → ~22 organized files, with the live training path unchanged.

### Migration rules (Lustre-safe, reversible)

1. `git mv` (preserves history) into the new tree; do NOT copy+delete.
2. Keep `train_full_autoreg.batch` reachable at the old path via a symlink for one cycle so
   muscle-memory + the auto-resume `sbatch` self-reference don't break.
3. Fix permissions in one pass: single owner, `chmod -R g+rw` for `brics.s5e`, consistent `0775`.
4. Delete the junk artifact filename (requires explicit user confirmation per rm rule).
5. Archive (not delete) one-off experiment scripts into `scripts/archive/<exp>/` if uncertain.

### What I could NOT verify (needs access or owner action)

- Contents of `train_full_autoreg.batch` / `node_wrapper.sh` (aramis `0600`) — call chain above
  is from CLAUDE.md, not file read.
- Whether `scripts/` / `bin/` subdirs already hold anything (group-locked, `ls` returned 0).
- Exact dead-vs-live status of each test/profile script (needs `git log` recency or owner input).
