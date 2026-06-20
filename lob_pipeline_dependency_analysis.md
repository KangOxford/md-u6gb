# LOB Pipeline — Dependency Analysis & Consolidation Plan
**Notion task**: "build the dependencies lob pipeline" (page 36f12c45)
**Mode**: ANALYSIS / PLAN ONLY — nothing executed. UTC 2026-05-29.

---

## 0. The three pipelines (clears the A1 / A2 confusion)

The user named two paths (A1, A2). In fact there are **three** relevant `lob_pipeline` trees: two
*private* s5e clones, each with a *public mirror*. A1 and A2 are the public mirrors.

| Tag | Path | Owner | Role | u6gb access |
|-----|------|-------|------|-------------|
| **A1** | `/projects/public/s5e/quant_team/quant/AlphaTrade/lob_pipeline` | kangli.s5e | **Public mirror, in use since ~May 12** | root 777 ✓ |
| **A2** | `/projects/public/s5e/quant_team/lob_pipeline` | aramis.s5e | Public mirror, abandoned (R1) | root 777 ✓ |
| P1 (private) | `/projects/s5e/quant/AlphaTrade/lob_pipeline` | kangli.s5e | kangli's clone, `feat/split-bench` | **DENIED** |
| P2 (private) | `/projects/s5e/lob_pipeline` | aramis.s5e | aramis's automation clone | **DENIED** |

Mirror map: **A1 mirrors P1 (kangli)**, **A2 mirrors P2 (aramis)**.

---

## 1. Which skill does `/bench` use?

There are **3 copies** of `bench/SKILL.md`, and they are **not identical**:

| Copy | md5 | "Uses pipeline at …" |
|------|-----|----------------------|
| u6gb active `/bench` | `7119e93b` | `/projects/s5e/quant/AlphaTrade/lob_pipeline` (P1, kangli) → mirror = **A1** |
| inside A1 **and** A2 | `f1d3933b` | `/projects/s5e/lob_pipeline` (P2, aramis) → mirror = **A2** |

So: **the `/bench` you invoke (u6gb copy) resolves to the A1 lineage (kangli's clone).**
The copy sitting *inside* A1/A2 points at the A2 lineage and carries its own note:
*"NOT `/projects/s5e/lob_pipeline/` (aramis's, permission issues)"* — i.e. the skill itself documents R1.

### What `/bench` does (5 phases, 2 SLURM jobs per stock)
`run_lobbench_pipeline.sh` (orchestrator) → submits `pipeline/_integrated.batch` twice per stock:
- **Job 1 — inference** (GPU, `--infer_nodes`, default 1 SSM / 16 attention)
- **Job 2 — scoring** (1 node, CPU 256 threads, `--dependency=afterok:Job1`)

Inside `_integrated.batch`:
1. **Inference** — `run_inference.py` (JAX multi-GPU) or MarS (PyTorch 1-GPU) → `inference_results/<NAME>_<STOCK>_<JID>/{data_real,data_gen,data_cond}`
2. **Decode** `.npy`→CSV (+ MarS NPY→CSV)
3. **Stage + HF-match** → `bench_data/<MODEL>/<STOCK>/2023` symlink farm
4. **Score** — `lob_bench/run_bench.py` (21 unconditional metrics) → `results_<NAME>/scores/*.pkl`
5. **Merge + extended** (cond / time-lagged / context / divergence) + **plots** → `results_<NAME>/plots/*.png`

Modes: HF-matched (GOOG/INTC Jan-2023 only) vs Custom (`--no_hf_compare`, N random). Token modes 1/9/22/24/26/lobert/6tok_lossless (24 default).

---

## 2. R1 — the permission story (ground truth, not the guess)

The hypothesis was "bad chmod on A2 root". **That is not what's happening today.**

- **Both A1 and A2 *roots* are `drwxrwsrwx` (777+setgid)** — world rwx. The root chmod is fine on both.
- The concrete, still-true accessibility split is in the **dependency paths**, all of which sit under
  the **s5e-private** tree `/projects/s5e/…` and are **DENIED** to `kangli.u6gb` (group `brics.u6gb`, not `brics.s5e`):

| s5e-private dep (referenced by both configs) | u6gb |
|---|---|
| `/projects/s5e/lob_pipeline/pip_packages{,_extra}` (numba, statsmodels…) | DENIED |
| `/projects/s5e/lob_preproc_l100` (wide book L100) | DENIED |
| `/projects/s5e/lob_pipeline/data/<TICKER>_jan2026` (raw data default) | DENIED |

- The **public-mirror** deps A1 was rewired to use are all **OK**: public miniforge, public LOBS5,
  `lob_bench`, `lob_preproc_sp500_squashfs`, `recon_2026-05/output/squashfs`.

**So R1 = "A2 leans on s5e-private paths; A1 was re-pointed at world-readable public mirrors."**
Most likely the original May-12 issue was a recursive-chmod gap on A2's *file contents* (created by
aramis.s5e with restrictive umask) that has since largely converged to 777, but the *dependency wiring*
is the durable reason A1 is the safe choice for u6gb.

### Residual gaps even on A1 (honest caveats)
- `numba` is **absent** from the public env, but `metrics.py` guards the import (`try/except`,
  `_USE_NUMBA=False`) → scoring still runs, just without the numba fast-path. **Not fatal.**
- The dead `PYTHONPATH=/projects/s5e/.../pip_packages…` line in `_integrated.batch:490` points at a
  DENIED dir → silently ignored for u6gb. Harmless but should be cleaned up.
- Public `LOBS5/lob/` lacks `encoding_*tok.py`, and `LOBS5/decode_npy_to_csv.py` is missing →
  default 24tok works (pre-existing `encoding.py`), but **non-default token modes / `.npy` decode** fall
  back to experiment-worktree paths that may be s5e-private. **Verify before benching 22/26/lobert.**

---

## 3. T2 (CORE) — full dependency map

```
/bench  (which COPY decides which lineage)
 ├─ u6gb active SKILL.md (7119e93b) → P1 kangli  ── public mirror ⇒ A1   ← what you run
 └─ copy inside A1 & A2 (f1d3933b)  → P2 aramis  ── public mirror ⇒ A2

RUNTIME CHAIN (A1, the live one)
run_lobbench_pipeline.sh                                  [orchestrator]
  source pipeline/config.sh        → PYTHON, PARTITION, EXCLUDE_NODES, *_DATA, squashfs dirs
  read   lob_bench/hf_data_git/<S>/data_gen_lobs5         (HF-mode index extraction)
  read   pipeline/sample_indices/<S>_<N>.txt              (fixed eval indices)
  sbatch pipeline/_integrated.batch  × 2/stock            (P1 infer ▸ P2 score [afterok])
       │
       ▼ _integrated.batch  (5 phases, single SLURM job)
       env  PYTHON    = public miniforge                              [OK]
            LOBS5_DIR = public …/AlphaTrade/LOBS5                      [OK]
            PYTHONPATH+= /projects/s5e/…/pip_packages{,_extra}         [DENIED→ignored, numba lost]
            gymnax_exchange = LOBS5/Alphatrade/gymnax_exchange         [OK]
       P1  inference : cd LOBS5_DIR; python run_inference.py
                       encoding swap lob/encoding_<N>tok.py→encoding.py [⚠ *tok.py missing in mirror]
                       data: raw DATA_DIR | squashfs mount (_squashfs_helpers.sh)
                       → inference_results/<NAME>_<S>_<JID>/{data_real,data_gen,data_cond}
       P1.5 decode .npy→csv  (LOBS5/decode_npy_to_csv.py)             [⚠ missing in mirror→worktree fallback]
       P2  stage symlinks → bench_data/<MODEL>/<S>/2023  + HF matching
       P3  scoring : cd lob_bench; python run_bench.py  (21 uncond metrics)
                     run_bench → data_loading, scoring, eval, metrics
                     metrics.py: numba OPTIONAL (absent→pure-numpy fallback)
                     → results_<NAME>/scores/scores_uncond_*.pkl
       P3.5 downstream (opt): pipeline/downstream_metrics.py → results_<NAME>/downstream
       P4  merge (merge_shards.py) + extended (cond/time_lagged/context/div)
                     → results_<NAME>/scores/scores_{cond,time_lagged,context,div}_*.pkl
       P5  plots : python run_plotting.py → results_<NAME>/plots/*.png
```

### External code dependencies (the repos `/bench` needs)
| Dependency | Location (A1 lineage) | u6gb | Used by |
|---|---|---|---|
| **lob_pipeline** (orchestration) | A1 itself | OK | entry |
| **LOBS5** (inference + model) | public `…/AlphaTrade/LOBS5` (`run_inference.py` ✓) | OK | P1, P1.5 |
| **lob_bench** (scoring lib) | `A1/lob_bench` (`run_bench, scoring, metrics, eval, merge_shards, run_plotting`) | OK | P3-P5 |
| **gymnax_exchange** (JAX LOB engine) | `LOBS5/Alphatrade/gymnax_exchange` | OK | P1 sim |
| **public miniforge** (JAX 0.9.0.1) | `…/quant_team/quant/miniforge3` | OK | all |
| pip_packages / numba | `/projects/s5e/…` | DENIED | P3 (optional) |
| wide-book L100 / L500 | `/projects/s5e/lob_preproc_l100` / public recon squashfs | L100 DENIED / squashfs OK | P1 init |
| MarS (optional) | `exp_O4c_MarS_PyTorch`, converters | s5e-private | P1 MarS |

---

## 4. T1 — consolidate samples + scores into one clean folder (PLAN ONLY)

### Current sprawl (A1)
| Hierarchy | Count | Holds | Linkage |
|---|---|---|---|
| `inference_results/<NAME>_<STOCK>_<JID>/` | **758** | SAMPLES: data_real / data_gen / data_cond | name+stock+jid |
| `results_<NAME>/` | **676** | SCORES (`scores/*.pkl`), plots, scores_clean, downstream | name |
| `bench_data/<MODEL>/<STOCK>/2023/` | **666** | transient symlink farm → inference_results (ABSOLUTE links) | name |

~2,100 directories across three flat trees, joined only by naming + absolute symlinks.

### Proposed clean target
```
benchruns/
  <NAME>/
    <STOCK>_<JOBID>/
      samples/   {data_real, data_gen, data_cond}   ← inference_results/<NAME>_<STOCK>_<JID>
      scores/    *.pkl  (only the <STOCK>-matching pkls)  ← results_<NAME>/scores
      plots/     *.png                                    ← results_<NAME>/plots
      downstream/                                          ← results_<NAME>/downstream (if any)
      MANIFEST.json  {name, stock, jobid, ckpt, step, token_mode, n_seq}
```

### Migration approach (do NOT run yet)
1. **Build manifest in code, not by ls**: parse names — inference dirs `^(?P<name>.+)_(?P<stock>[A-Z.]+)_(?P<jid>\d+)$`; results dirs `^results_(?P<name>.+)$`; score pkls embed `<STOCK>` so split per stock.
2. **rsync (copy, not move)** into `benchruns/…`; keep originals until verified. Optionally hardlink (`--link-dest`) to avoid doubling disk.
3. `bench_data/` is a **disposable** staging farm (absolute symlinks) — exclude from migration; it can be regenerated.
4. **Run the reconciler on a COMPUTE node (sbatch), never login** — 2,100 dirs × many files is a Lustre metadata load. Stagger, write a `latest.json` breadcrumb, no recursive `ls`.
5. **Quiesce first**: confirm `squeue --me` has no in-flight bench jobs writing to these dirs.

### Risks
- Absolute symlinks in `bench_data/` break if `inference_results` later moves (acceptable — it's transient).
- A `results_<NAME>/` can hold **multiple stocks** → must split scores by `<STOCK>` token in the filename.
- Large metadata op → compute-node only, staggered.

### Long-term (separate, optional code change)
Re-point `INFER_OUTPUT` and `RESULTS_DIR` in `_integrated.batch` / `run_lobbench_pipeline.sh` to write the
consolidated `benchruns/<NAME>/<STOCK>_<JOBID>/` layout directly, so new runs need no reconciliation.
