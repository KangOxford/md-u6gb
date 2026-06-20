# Live SLURM Jobs

**Last updated**: 2026-03-17 00:30 UTC

| Job ID | Name | Experiment | Nodes | State | Elapsed | Est. Completion |
|--------|------|-----------|-------|-------|---------|-----------------|
| 2906836 | m2-8n-resume-msgpack | exp_K3_Mamba2 | 8N/32GPU | RUNNING | ~0:30 | ~2026-03-17 14:00 |
| 2895659 | lobs5_360m (auto-resume) | exp_J2_muon_optimizer | 16N/64GPU | RUNNING | — | — |

---

## Job Details

### j2878592 — K1-muon-16n-24h

**Experiment**: J2 Muon Optimizer (`exp/J2-muon-optimizer` branch)
**W&B**: https://wandb.ai/oxford-lob/k_scheduling/runs/35aavknm

**Rationale**: Testing whether K=1 (AllReduce every step) + Muon optimizer on 16N produces
cleaner convergence than the previous K=10 runs. K-value benchmarks showed K=1 is both
faster per step and produces less gradient noise than K=10 at 8N scale. 16N doubles the
effective batch size (128 vs 64 samples/step) for lower per-step variance.

**Key context**:
- Prior K=10 8N training plateaued at LOBbench KS~0.084 with high oscillation
- 32N K=10 AdamW hit KS=0.068 at one checkpoint but couldn't stay there
- Weighted soup of 32N checkpoints achieved KS=0.071 (free, no training)
- This run tests if K=1 + Muon can match or beat those results

**Config**:
- Model: 360M S5 (d_model=2048, 24 layers, blocks=32)
- Optimizer: Muon (muon_lr=0.02, wd=0.005) + Adam (SSM) + AdamW (rest)
- K=1 (AllReduce every step), no grad accumulation
- Cosine LR: period = epoch (423,942 steps), no COSINE_STEPS override
- MEM_FRACTION=0.80 (reduced from 0.90 to avoid NCCL OOM at 16N)
- FIRST_COLLECTIVE_TIMEOUT=1500 (Muon XLA compilation is slower)
- Contiguous node allocation: nid[010660-010675]
- Speed: ~1.71 s/step → ~50k steps in 24h

**Data**: 8 tickers × 4yr (2022-2025), test: Jan 2026
**Log**: `experiments/exp_J2_muon_optimizer/logs_lobs5/lobs5_2878592.out`

---

### j2906836 — m2-8n-resume-msgpack (Mamba2 production resume)

**Experiment**: K3 Mamba2 (`exp/K3-Mamba2` branch, `exp_K3_Mamba2` worktree)
**W&B**: https://wandb.ai/oxford-lob/lobs5-K3-Mamba2 (new run ID pending)

**Rationale**: Resume of Mamba2 production training from j2877655 (step 50,149).
Uses CXI-safe checkpoint system (flax msgpack) after discovering that Orbax
distributed save/restore corrupts CXI state on Mamba2 resumes.

**CXI Resume Bug (Mamba2-specific)**:
Orbax checkpoint restore creates per-array NCCL channels for Mamba2's many small
`self.param()` arrays (A_log, D, dt_bias × 10 layers × optimizer states = 250 extra
arrays vs S5). This corrupts CXI memory registrations, causing any subsequent new
NCCL operation (checkpoint save, device_get, JIT reshard) to crash with CXI RC:265
SIGABRT ~10 min after resume. Training AllReduce works fine (uses channels from
JIT compile). S5/GDN branches are not affected.

**Fix**: Custom checkpoint system bypassing Orbax entirely:
- **Save**: `addressable_data(0)` → numpy → `flax.serialization.to_bytes()` → `state.msgpack`
- **Load**: `flax.serialization.from_bytes()` — auto-detected by `load_checkpoint()`
- Zero NCCL/CXI at either end. Checkpoint format compatible with existing `load_metadata()`.
- **CAVEAT**: Resuming from msgpack checkpoints requires the K3 branch's `load_checkpoint()`
  which auto-detects `state.msgpack`. Other branches' `load_checkpoint()` won't find it.

**Training history**:
- j2877655 (fresh): 0→50,149 steps, 8h, completed normally (MAX_JOB_HOURS=8.5 default)
- 7 resume attempts crashed with CXI RC:265 (j2895947→j2906780)
- j2906200 (CHECKPOINT_EVERY=0): proved training AllReduce stable for 22+ min
- j2906836 (msgpack save): **first successful checkpoint save on resume** at step 51,100

**Model**: 77.7M Mamba2 (d=1024, L=6, d_inner=2048, n_heads=32, headdim=64, d_state=128)
**Config**: 8N/32GPU, K=0, BSZ=4/gpu, LR=1e-3 cosine→5e-5 over 150K steps
**Speed**: ~1.77 it/s (0.56 s/step)
**Data**: 8 tickers × 4yr (2022-2025), test: Jan 2026
**Log**: `experiments/exp_K3_Mamba2/logs_lobs5/training_2906836_node0.log`

**LOBbench (step 50K, partial training)**:
- Mean KS: 0.0902, Mean W1: 0.1436, Mean L1: 0.1155 (21 metrics, GOOG)
- Results: `/projects/s5e/lob_pipeline/results_mamba2-77M-step50k/`

### Job 2899185 — KDA 103M Triton Full Training (LR=1e-4)
| Field | Value |
|-------|-------|
| Job ID | 2899185 |
| Branch | exp/K4-KDA |
| Worktree | exp_K4_KDA |
| Config | KDA 103M, d=1024, L=6, nh=8, hd=128, ev=2 |
| Params | 103,623,325 |
| Infra | 16N/64GPU, BSZ=1/gpu, gBSZ=64, LOCAL_STEPS_K=10 |
| LR | 1e-4 (SSM LR=1e-4) — lowered from 5e-4 to fix NaN |
| Mode | Triton KDA (USE_FLA_KDA=1) |
| Time | 24h |
| Speed | ~2.03 it/s (480 ms/step) |
| Log | `/projects/s5e/quant/AlphaTrade/experiments/exp_K4_KDA/logs_lobs5/training_2899185_node0.log` |
| W&B | https://wandb.ai/oxford-lob/exp_K4_KDA/runs/7q9b3n2s |
| Submitted | 2026-03-16 ~16:00 |

```
Job:   2899185 (kda103m-lr1e4-v2)
Step:  1006/847885  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0.1%
Model: KDA 103M (d=1024, L=6, nh=8, hd=128) | 103,623,325 params
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 16N / 64 GPU | BSZ=1/gpu, gBSZ=64 | Local Steps K=10
LR:    1e-4 (was 5e-4, lowered to fix NaN)
Loss:  3.15 (at step 1006)
Speed: ~2.03 it/s (480 ms/step)
Time:  0:12 elapsed  |  ~116h remaining  |  24:00 limit
ETA:   ~174K steps in 24h → 20.5% of epoch
W&B:   https://wandb.ai/oxford-lob/exp_K4_KDA/runs/7q9b3n2s
Log:   /projects/s5e/quant/AlphaTrade/experiments/exp_K4_KDA/logs_lobs5/training_2899185_node0.log
```

Updated: 2026-03-16 16:37:22 UTC

Job:   2899322 (muon55m-16n-bench)
Step:  301/301  [██████████████████████████████]  100%
Model: 55M (d=1024, L=12, B=16, ssm=1024) | 55,498,861 params
Data:  8 tickers × 1yr (GOOG 2022) | test: Jan 2023
Infra: 16N / 64 GPU | BSZ=8/gpu, gBSZ=512 | Local Steps K=0
LR:    5e-4 (Muon: kernel_lr=0.02)
Loss:  ~1.34 (after 301 steps)
Speed: ~1.51 it/s (660 ms/step)
Time:  ~6min train + eval  |  30:00 limit
ETA:   done
W&B:   https://wandb.ai/oxford-lob/lobs5-55M-J2/runs/tpe1or58
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_J2_muon_optimizer/logs_j2/training_2899322_node0.log

Updated: 2026-03-16 16:50:00 UTC

```
Job:   2901104 (kda103m-lr1e4-v3)
Step:  0/847885  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: KDA 103M (d=1024, L=6, nh=8, hd=128) | 103,623,325 params
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 16N / 64 GPU | BSZ=1/gpu, gBSZ=64 | Local Steps K=10
LR:    1e-4 (SSM LR=1e-4)
Loss:  — (just submitted)
Speed: ~2.03 it/s expected
Time:  —  |  24:00 limit
Note:  v3, excluded nodes from v2 (NCCL deadlock at step 3478)
W&B:   pending
Log:   /projects/s5e/quant/AlphaTrade/experiments/exp_K4_KDA/logs_lobs5/training_2901104_node0.log
```

Updated: 2026-03-16 17:27 UTC
Job:   2905747 (kda-smoke-2n-30m)
User:  aramis.s5e
Infra: 2N / 8 GPU | BSZ=2/gpu, gBSZ=16 | Local Steps K=10
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_K5_KDA_recompute/logs_lobs5/lobs5_2905747.out

Updated: 2026-03-16 19:54:10 UTC

Job:   2906643 (kda-94m-k1-16n-24h)
User:  aramis.s5e
Model: KDA 94M (d=1024, L=6, B=16, 8h×128d) | 103,623,325 params
Infra: 16N / 64 GPU | BSZ=2/gpu, gBSZ=128 | Local Steps K=1
LR:    5e-4
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_K5_KDA_recompute/logs_lobs5/lobs5_2906643.out

Updated: 2026-03-16 22:15:06 UTC

Job:   2906749 (kda-94m-k1-16n-24h)
User:  aramis.s5e
Step:  1800/423942  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0.4%
Model: KDA 94M (d=1024, L=6, B=16, 8h×128d, expand_v=2) | 103,623,325 params
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 16N / 64 GPU | BSZ=2/gpu, gBSZ=128 | Local Steps K=1
LR:    5e-4
Loss:  1.68 (at step 1800)
Speed: ~1.01 s/step
Time:  0:34 elapsed  |  ~118h remaining  |  24:00 limit
ETA:   ~85K steps in 24h (~20% of epoch)
W&B:   https://wandb.ai/oxford-lob/lobs5-K5-KDA/runs/64jg1699
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_K5_KDA_recompute/logs_lobs5/lobs5_2906749.out

Updated: 2026-03-16 23:20:31 UTC


---

Job:   2906854 (lobert-55m-8n-24h)
User:  aramis.s5e
Step:  0/~42395×10  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: LOBERT ~55M (d=640, L=12, h=8, d_ff=2240)
Data:  GOOG 2022 (249 days) | test: Jan 2023
Infra: 8N / 32 GPU | BSZ=10/gpu, gBSZ=320
LR:    3e-4
Loss:  — (just started)
Speed: —
Time:  0:00 elapsed  |  ~24:00 remaining  |  24:00 limit
ETA:   —
W&B:   pending (project: K2_LOBERT)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_K2b_LOBERT/logs_lobs5/training_2906854_node0.log

Updated: 2026-03-16 22:48:00 UTC
Job:   2915937 (kda-94m-k1-16n-24h-resume)
User:  aramis.s5e
Step:  resuming from 28305/423942
Model: KDA 94M (d=1024, L=6, B=16, 8h×128d, expand_v=2) | 103,623,325 params
Infra: 16N / 64 GPU | BSZ=2/gpu, gBSZ=128 | Local Steps K=1
LR:    5e-4
Loss:  0.65 (at step 28305)
Ckpt:  resuming from j2906749_64jg1699_2906749 step 28305
W&B:   https://wandb.ai/oxford-lob/lobs5-K5-KDA/runs/64jg1699
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_K5_KDA_recompute/logs_lobs5/lobs5_2915937.out

Updated: 2026-03-17 10:06:22 UTC


---

Job:   2916169 (lobert-55m-4n-bsz40)
User:  aramis.s5e
Step:  0/~85K×10  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: LOBERT ~55M (d=640, L=12, h=8, d_ff=2240)
Data:  GOOG 2022 (249 days) | test: Jan 2023
Infra: 4N / 16 GPU | BSZ=40/gpu, gBSZ=640
LR:    4e-4
Loss:  — (just started)
Speed: —
Time:  0:00 elapsed  |  ~24:00 remaining  |  24:00 limit
ETA:   —
W&B:   pending (project: K2_LOBERT)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_K2b_LOBERT/logs_lobs5/training_2916169_node0.log

Updated: 2026-03-17 01:05:00 UTC

---

Job:   2916312 (lobert-55m-4n-bsz40)
User:  aramis.s5e
Step:  0/~85K×10  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: LOBERT ~55M (d=640, L=12, h=8, d_ff=2240)
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 4N / 16 GPU | BSZ=40/gpu, gBSZ=640
LR:    4e-4
Loss:  — (just started)
Speed: —
Time:  0:00 elapsed  |  ~24:00 remaining  |  24:00 limit
ETA:   —
W&B:   pending (project: K2_LOBERT)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_K2b_LOBERT/logs_lobs5/training_2916312_node0.log

Updated: 2026-03-17 01:30:00 UTC

---

Job:   2916567 (lobert-55m-4n-bsz30)
User:  aramis.s5e
Step:  0/~113K×10  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: LOBERT ~55M (d=640, L=12, h=8, d_ff=2240)
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 4N / 16 GPU | BSZ=30/gpu, gBSZ=480
LR:    3.7e-4
Loss:  — (just started)
Speed: —
Time:  0:00 elapsed  |  ~24:00 remaining  |  24:00 limit
ETA:   —
W&B:   pending (project: K2_LOBERT)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_K2b_LOBERT/logs_lobs5/training_2916567_node0.log

Updated: 2026-03-17 10:30:00 UTC

---

Job:   2916827 (lobert-55m-4n-bsz20)
User:  aramis.s5e
Step:  0/~170K×10  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: LOBERT ~55M (d=640, L=12, h=8, d_ff=2240)
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 4N / 16 GPU | BSZ=20/gpu, gBSZ=320
LR:    3e-4
Loss:  — (just started)
Speed: —
Time:  0:00 elapsed  |  ~24:00 remaining  |  24:00 limit
ETA:   ~6 epochs in 24h
W&B:   pending (project: K2_LOBERT)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_K2b_LOBERT/logs_lobs5/training_2916827_node0.log

Updated: 2026-03-17 11:00:00 UTC

---

Job:   2918784 (lobert-55m-4n-bsz20-hier)
User:  aramis.s5e
Step:  0/~170K×10  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: LOBERT ~55M (d=640, L=12, h=8, d_ff=2240)
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 4N / 16 GPU | BSZ=20/gpu, gBSZ=320 | hierarchical shard_map
LR:    3e-4
Loss:  — (just started)
Speed: —
Time:  0:00 elapsed  |  ~24:00 remaining  |  24:00 limit
ETA:   ~6 epochs in 24h
W&B:   pending (project: K2_LOBERT)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_K2b_LOBERT/logs_lobs5/training_2918784_node0.log

Updated: 2026-03-17 12:30:00 UTC

---

Job:   2928084 (lobert-55m-4n-bsz20-hier)
User:  aramis.s5e
Step:  0/~170K×10  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: LOBERT ~55M (d=640, L=12, h=8, d_ff=2240)
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 4N / 16 GPU | BSZ=20/gpu, gBSZ=320 | hierarchical shard_map
LR:    3e-4
Loss:  — (just started)
Speed: ~0.18 s/step (~7 it/s) expected
Time:  0:00 elapsed  |  ~24:00 remaining  |  24:00 limit
ETA:   ~2.8 epochs in 24h
W&B:   pending (project: K2_LOBERT)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_K2b_LOBERT/logs_lobs5/training_2928084_node0.log

Updated: 2026-03-17 17:15:00 UTC

---

Job:   2934255 (lobert-55m-4n-ckptfix)
User:  aramis.s5e
Step:  0/~170K×10  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: LOBERT ~55M (d=640, L=12, h=8, d_ff=2240)
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 4N / 16 GPU | BSZ=20/gpu, gBSZ=320 | hierarchical shard_map
LR:    3e-4
Fix:   device_get/device_put reshard + early first ckpt at ~10min
Loss:  — (just started)
Speed: ~0.17 s/step expected
W&B:   pending (project: K2_LOBERT)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_K2b_LOBERT/logs_lobs5/training_2934255_node0.log

Updated: 2026-03-17 18:50:00 UTC

---
## C10 NCCL_BUFFSIZE Sweep — Phase 1

Job:   2936940 (C10a-buff2M-16n)
User:  kangli.s5e
Model: 360M (d=2048, L=24, B=32) | CURTAIL_EPOCHS=300
Data:  GOOG 2022 + Jan2023 test
Infra: 16N / 64 GPU | BSZ=2/gpu, gBSZ=128
NCCL:  BUFFSIZE=2MB (control)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/logs_lobs5/lobs5_2936940.out

Job:   2936941 (C10b-buff8M-16n)
User:  kangli.s5e
NCCL:  BUFFSIZE=8MB
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/logs_lobs5/lobs5_2936941.out

Job:   2936942 (C10c-buff16M-16n)
User:  kangli.s5e
NCCL:  BUFFSIZE=16MB
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/logs_lobs5/lobs5_2936942.out

Job:   2936943 (C10d-buff32M-16n)
User:  kangli.s5e
NCCL:  BUFFSIZE=32MB
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/logs_lobs5/lobs5_2936943.out

Updated: 2026-03-17 (C10 Phase 1 sweep)

---

Job:   2944573 (lobert-55m-4n-fresh-msgpk)
User:  aramis.s5e
Step:  0/~170K×10  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: LOBERT ~55M (d=640, L=12, h=8, d_ff=2240)
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 4N / 16 GPU | BSZ=20/gpu, gBSZ=320 | hierarchical shard_map
LR:    3e-4
Fix:   msgpack checkpoints (zero Orbax/CXI), fresh start
Speed: ~0.17 s/step expected
W&B:   pending (project: K2_LOBERT)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_K2b_LOBERT/logs_lobs5/training_2944573_node0.log

Updated: 2026-03-17 23:20:00 UTC

Job:   3138292 (26tok-adamw-1n-24h-resume)
User:  aramis.s5e
Model: 55M (d=768, L=12, B=12, ssm=768) | ~32M params
Data:  GOOG × 2025 | test: 2026-01-02
Infra: 1N / 4 GPU | BSZ=10/gpu, gBSZ=40
LR:    5e-4 (cosine over 214K steps, warm restart at step 71521)
Opt:   AdamW WD=0.05
Resume: from j3108758 step 71521 (epoch 0, 99.8%)
EPOCHS: 3 (total ~214K steps)

Updated: 2026-03-18 16:30 UTC

Job:   3142671 (26tok-adamw-1n-24h-resume)
User:  aramis.s5e
Model: 55M (d=768, L=12, B=12, ssm=768) | ~32M params
Data:  GOOG × 2025 | test: 2026-01-02
Infra: 1N / 4 GPU | BSZ=10/gpu, gBSZ=40
Opt:   AdamW WD=0.05
Resume: from j3108758 step 71521
Fix:   eval watchdog 1200s→3600s (commit 3ac4713b)
Note:  j3138292+j3140187 crashed at val batch 6166/8003 due to watchdog timeout

Updated: 2026-03-18 17:30 UTC

Job:   3148081 (lobert-55m-4n-24h)
User:  aramis.s5e
Step:  25606/339154  [██░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  7.5%
Model: LOBERT 55M (d=640, L=12, h=8, d_ff=2240) | ~55M params
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 4N / 16 GPU | BSZ=20/gpu, gBSZ=320 | Local Steps K=10
LR:    5e-5
Loss:  1.693 (at step 25606)
Speed: ~0.19 s/step
Time:  resuming  |  ~16.5h remaining  |  24:00 limit
ETA:   should complete within time limit
W&B:   TBD (new run)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_K2b_LOBERT/logs_lobs5/lobs5_3148081.out

Updated: 2026-03-18 16:05:00 UTC

Job:   3149349 (N1b-split-muon-4n-24h)
User:  aramis.s5e
Model: 55M (d=768, L=12, B=12, ssm=768) | vocab=3212 (split price_high/low + size_high/low)
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 4N / 16 GPU | BSZ=10/gpu, gBSZ=160
Opt:   Muon (kernel_lr=0.02, WD=0.005)
Branch: exp/N1b-vocab-split
W&B:   N1b-vocab-split project

Job:   3149353 (N1-base-muon-4n-24h)
User:  aramis.s5e
Model: 55M (d=768, L=12, B=12, ssm=768) | vocab=2112 (shared price/size)
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 4N / 16 GPU | BSZ=10/gpu, gBSZ=160
Opt:   Muon (kernel_lr=0.02, WD=0.005)
Branch: exp/N1-26tok
W&B:   N1-26tok project
Note:  Baseline for vocab split A/B test — identical config except vocab

Updated: 2026-03-18 21:40 UTC

Job:   3149447 (N1b-split-muon-55m-4n)
User:  aramis.s5e
Model: 55M (d=1024, L=12, B=16, ssm=1024) | vocab=3212 (split price_high/low + size_high/low)
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 4N / 16 GPU | BSZ=10/gpu, gBSZ=160
Opt:   Muon (kernel_lr=0.02, WD=0.005), cosine_steps=150K
W&B:   N1b-vocab-split

Job:   3149448 (N1-base-muon-55m-4n)
User:  aramis.s5e
Model: 55M (d=1024, L=12, B=16, ssm=1024) | vocab=2112 (shared)
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 4N / 16 GPU | BSZ=10/gpu, gBSZ=160
Opt:   Muon (kernel_lr=0.02, WD=0.005), cosine_steps=150K
W&B:   N1-26tok
Note:  A/B test — only variable is vocab split

Updated: 2026-03-18 21:55 UTC

Job:   3149523 (bench_soup_gdn_last5_gen500_GOOG)
User:  aramis.s5e
Model: GDN soup_last5 (d=1024, L=6, B=16, ssm=1024, gdn_heads=8) | ~10M params
Data:  GOOG Jan 2026 | 3136 samples (fixed indices)
Infra: 1N / 4 GPU | BSZ=64, cond=500, gen=500
Log:   /lus/lfs1aip2/projects/s5e/lob_pipeline/logs/integrated_soup_gdn_last5_gen500_GOOG_3149523.out
Updated: 2026-03-18

Job:   3149525 (bench_soup_gdn_last5_gen1000_GOOG)
User:  aramis.s5e
Model: GDN soup_last5 (d=1024, L=6, B=16, ssm=1024, gdn_heads=8) | ~10M params
Data:  GOOG Jan 2026 | 3136 samples (fixed indices)
Infra: 1N / 4 GPU | BSZ=64, cond=500, gen=1000
Log:   /lus/lfs1aip2/projects/s5e/lob_pipeline/logs/integrated_soup_gdn_last5_gen1000_GOOG_3149525.out
Updated: 2026-03-18

Job:   3149526 (bench_soup_gdn_last5_gen2000_GOOG)
User:  aramis.s5e
Model: GDN soup_last5 (d=1024, L=6, B=16, ssm=1024, gdn_heads=8) | ~10M params
Data:  GOOG Jan 2026 | 3136 samples (fixed indices)
Infra: 1N / 4 GPU | BSZ=64, cond=500, gen=2000
Log:   /lus/lfs1aip2/projects/s5e/lob_pipeline/logs/integrated_soup_gdn_last5_gen2000_GOOG_3149526.out
Updated: 2026-03-18

Job:   3150039 (bench_soup_gdn_last5_gen500_GOOG)
User:  aramis.s5e
Model: GDN soup_last5 (d=1024, L=6, B=16, ssm=1024, gdn_heads=8)
Data:  GOOG × Jan 2026 | 3136 seqs | cond=500, gen=500
Infra: 1N / 4 GPU | BSZ=64 | skip_extended
Log:   /lus/lfs1aip2/projects/s5e/lob_pipeline/logs/integrated_soup_gdn_last5_gen500_GOOG_3150039.out
Updated: 2026-03-18 UTC

Job:   3150040 (bench_soup_gdn_last5_gen1000_GOOG)
User:  aramis.s5e
Model: GDN soup_last5 (d=1024, L=6, B=16, ssm=1024, gdn_heads=8)
Data:  GOOG × Jan 2026 | 3136 seqs | cond=500, gen=1000
Infra: 1N / 4 GPU | BSZ=64 | skip_extended
Log:   /lus/lfs1aip2/projects/s5e/lob_pipeline/logs/integrated_soup_gdn_last5_gen1000_GOOG_3150040.out
Updated: 2026-03-18 UTC

Job:   3150041 (bench_soup_gdn_last5_gen2000_GOOG)
User:  aramis.s5e
Model: GDN soup_last5 (d=1024, L=6, B=16, ssm=1024, gdn_heads=8)
Data:  GOOG × Jan 2026 | 3136 seqs | cond=500, gen=2000
Infra: 1N / 4 GPU | BSZ=64 | skip_extended
Log:   /lus/lfs1aip2/projects/s5e/lob_pipeline/logs/integrated_soup_gdn_last5_gen2000_GOOG_3150041.out
Updated: 2026-03-18 UTC

Job:   3150234 (bench_soup_gdn_last5_gen500_GOOG) [retry 3 — no --lobs5_dir]
User:  aramis.s5e
Model: GDN soup_last5 (d=1024, L=6, B=16, ssm=1024, gdn_heads=8)
Data:  GOOG × Jan 2026 | 3136 seqs | cond=500, gen=500
Infra: 1N / 4 GPU | BSZ=64 | skip_extended
Log:   /lus/lfs1aip2/projects/s5e/lob_pipeline/logs/integrated_soup_gdn_last5_gen500_GOOG_3150234.out
Updated: 2026-03-18 UTC

Job:   3150236 (bench_soup_gdn_last5_gen1000_GOOG) [retry 3]
User:  aramis.s5e
Log:   /lus/lfs1aip2/projects/s5e/lob_pipeline/logs/integrated_soup_gdn_last5_gen1000_GOOG_3150236.out
Updated: 2026-03-18 UTC

Job:   3150237 (bench_soup_gdn_last5_gen2000_GOOG) [retry 3]
User:  aramis.s5e
Log:   /lus/lfs1aip2/projects/s5e/lob_pipeline/logs/integrated_soup_gdn_last5_gen2000_GOOG_3150237.out
Updated: 2026-03-18 UTC

Job:   3151922 (nsa-1k-16n-24h)
User:  valentinm.s5e
Step:  0/233061  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: NSA 201M (d=1024, L=12, H=16, head_dim=64) | block_size=24, block_counts=16
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 16N / 64 GPU | BSZ=4/gpu, gBSZ=256
LR:    3e-4
Loss:  — (not started)
Speed: ~1.87 s/step (from benchmark)
Time:  0:00 elapsed  |  ~24:00 remaining  |  24:00 limit
ETA:   ~46K steps in 24h (~20% of epoch)
W&B:   pending
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O2_naive_sparse_attn/logs_lobs5/training_3151922_node0.log

Updated: 2026-03-19 01:15:00 UTC
Job:   3250593 (p1a-gdn94m-16n-24h)
User:  kangli.s5e
Model: GDN 94M (d=1024, L=6, B=16, nh=8, hd=128, expand_v=2, chunk=64)
Data:  8 tickers x 4yr | test: 2026-01
Infra: 16N / 64 GPU | BSZ=10/gpu, gBSZ=640
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_P1a_flat_9tok/logs_lobs5/lobs5_3250593.out

Updated: 2026-03-20 submitted

Job:   3251578 (550m-muon-16n-24h)
User:  aramis.s5e
Step:  0/?  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: 550M (d=2560, L=24, B=40, ssm=2560) | ~550M params (estimated)
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 16N / 64 GPU | BSZ=1/gpu, gBSZ=64 | Local Steps K=10
LR:    Muon=0.02, SSM=5e-4, WD=0.005
Loss:  — (not started)
Speed: — (not started)
Time:  0 elapsed  |  ~24h limit
ETA:   —
W&B:   lobs5-scaling-law (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/logs_lobs5/lobs5_3251578.out

Updated: 2026-03-21 UTC
## 550M BSZ=2 Smoke Test
Job:   3251739 (550m-bsz2-smoke)
User:  aramis.s5e
Model: 550M (d=2560, L=24, B=40, ssm=2560) | BSZ=2/gpu
Infra: 16N / 64 GPU | gBSZ=128 | Local Steps K=10
Opt:   Muon (LR=0.02, WD=0.005)
Time:  00:30:00 limit | CURTAIL=300
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/logs_lobs5/training_3251739_node0.log

Updated: 2026-03-21 03:28:21 UTC

## 559M Muon Full Training
Job:   3252028 (559m-muon-16n-24h)
User:  aramis.s5e
Model: 559M (d=2560, L=24, B=40, ssm=2560) | 559,271,533 params
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 16N / 64 GPU | BSZ=2/gpu, gBSZ=128 | Local Steps K=10
Opt:   Muon (LR=0.02, WD=0.005)
Speed: ~2.58 s/step (from smoke test 3251739)
Time:  24:00:00 limit
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/logs_lobs5/training_3252028_node0.log

Updated: 2026-03-21 04:04:22 UTC

---

Job:   3252054 (p1a-gdn94m-bsz2-16n-noval)
User:  kangli.s5e
Step:  0/227551  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: GDN 94M (d=1024, L=6, B=16, gdn_heads=8, gdn_dim=128) | ~94M params
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 16N / 64 GPU | BSZ=2/gpu, gBSZ=128 | MSG_SEQ_LEN=1024
LR:    default
Loss:  — (just started)
Speed: — (just started)
Time:  0:00 elapsed  |  ~24:00 remaining  |  24:00 limit
W&B:   pending
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_P1a_flat_9tok/logs_lobs5/lobs5_3252054.out

Updated: 2026-03-21 05:00:00 UTC

---

Job:   3252055 (p1b-gdn82m-bsz4-16n-noval)
User:  kangli.s5e
Step:  0/113775  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: GDN 82M (d=1024, L=6, B=16, gdn_heads=8, gdn_dim=128) | ~82M params
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 16N / 64 GPU | BSZ=4/gpu, gBSZ=256 | MSG_SEQ_LEN=1024
LR:    default
Loss:  — (just started)
Speed: — (just started)
Time:  0:00 elapsed  |  ~24:00 remaining  |  24:00 limit
W&B:   pending
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_P1_sigma_order/logs_lobs5/lobs5_3252055.out

Updated: 2026-03-21 05:00:00 UTC

## 559M Muon Full Training (BSZ=1, 15min ckpt)
Job:   3252954 (559m-muon-16n-24h-v2)
User:  aramis.s5e
Model: 559M (d=2560, L=24, B=40, ssm=2560) | 559,271,533 params
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 16N / 64 GPU | BSZ=1/gpu, gBSZ=64 | Local Steps K=10
Opt:   Muon (LR=0.02, WD=0.005)
Speed: ~2.11 s/step (from job 3251578)
Ckpt:  every 15min (first at ~5min) | ~2.8% overhead
Time:  24:00:00 limit
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/logs_lobs5/training_3252954_node0.log

Updated: 2026-03-21 11:58:39 UTC

Job:   3263365 (mars-goog-1n-8h)
User:  kangli.s5e
Step:  0/30133  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: MarS LLaMA 399M (d=1024, L=24, H=16) | 399,365,120 params
Data:  GOOG × 4yr | MarS order-level tokenization
Infra: 1N / 4 GPU | BSZ=64/gpu, gBSZ=256
LR:    3e-4
Loss:  (pending)
Speed: ~0.57 s/step (est from smoke test)
Time:  0:00 elapsed  |  ~4.8h remaining  |  8:00 limit
ETA:   ~5h total
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O4c_MarS_PyTorch/logs/torch_3263365.out

Updated: 2026-03-22 00:01:00 UTC

Job:   3263794 (mars-8tk-1n-8h)
User:  kangli.s5e
Step:  0/?  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: MarS LLaMA 399M (d=1024, L=24, H=16) | 399,365,120 params
Data:  8 tickers (GOOG,AAPL,NVDA,AMZN,META,TSLA,MSFT,AMD) × 4yr (2022-2025)
Infra: 1N / 4 GPU | BSZ=64/gpu, gBSZ=256
LR:    3e-4
Loss:  (pending)
Speed: ~0.57 s/step (est)
Time:  0:00 elapsed  |  est ~38h for full epoch  |  8:00 limit
ETA:   8h limit — will complete partial epoch
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O4c_MarS_PyTorch/logs/torch_3263794.out

Updated: 2026-03-22 00:15:00 UTC

Job:   3264182 (mars-8tk-16n-4h)
User:  kangli.s5e
Step:  0/14330  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: MarS LLaMA 399M (d=1024, L=24, H=16) | 399,365,120 params
Data:  8 tickers (GOOG,AAPL,NVDA,AMZN,META,TSLA,MSFT,AMD) × 4yr (2022-2025)
Infra: 16N / 64 GPU | BSZ=64/gpu, gBSZ=4096
LR:    3e-4
Loss:  (pending)
Speed: ~0.6 s/step (est)
Time:  0:00 elapsed  |  ~2.4h remaining  |  4:00 limit
ETA:   ~2.4h — first multi-node PyTorch MarS run
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O4c_MarS_PyTorch/logs/torch_3264182.out

Updated: 2026-03-22 00:40:00 UTC


Job:   3277505 (ticks-gdn-soup)
User:  aramis.s5e
Model: GDN 94M soup (5 ckpts) | 94,058,701 params
Data:  GOOG Jan 2026 | 3136 sequences
Infra: 1N / 4 GPU | scoring only (skip inference)
Notes: LOBbench rescore with new _ticks metrics (replacing _levels)
Log:   /lus/lfs1aip2/projects/s5e/lob_pipeline/logs/integrated_ticks-gdn-soup_GOOG_3277505.out

Job:   3277518 (ticks-mamba2-soup)
User:  aramis.s5e
Model: Mamba2 77M soup (10 ckpts) | ~77.7M params
Data:  GOOG Jan 2026 | 3136 sequences
Infra: 1N / 4 GPU | scoring only (skip inference)
Notes: LOBbench rescore with new _ticks metrics (replacing _levels)
Log:   /lus/lfs1aip2/projects/s5e/lob_pipeline/logs/integrated_ticks-mamba2-soup_GOOG_3277518.out

Updated: 2026-03-22 20:00:00 UTC

Job:   3277596 (ticks2-gdn-soup)
User:  aramis.s5e
Model: GDN 94M soup (5 ckpts) | 94,058,701 params
Data:  GOOG Jan 2026 | 3136 sequences
Infra: 1N / 4 GPU | FULL pipeline (inference + scoring)
Notes: Full re-run with new _ticks metrics — stability test vs job 3277505
Log:   /lus/lfs1aip2/projects/s5e/lob_pipeline/logs/integrated_ticks2-gdn-soup_GOOG_3277596.out

Job:   3277598 (ticks2-mamba2-soup)
User:  aramis.s5e
Model: Mamba2 77M soup (10 ckpts) | ~77.7M params
Data:  GOOG Jan 2026 | 3136 sequences
Infra: 1N / 4 GPU | FULL pipeline (inference + scoring)
Notes: Full re-run with new _ticks metrics — stability test vs job 3277518
Log:   /lus/lfs1aip2/projects/s5e/lob_pipeline/logs/integrated_ticks2-mamba2-soup_GOOG_3277598.out

Updated: 2026-03-22 20:25:00 UTC

Job:   3277622 (ticks2-mamba2-soup RESUBMIT)
User:  aramis.s5e
Model: Mamba2 77M soup (10 ckpts) | ~77.7M params
Data:  GOOG Jan 2026 | 3136 sequences
Infra: 1N / 4 GPU | FULL pipeline (inference + scoring)
Notes: Resubmit after adding gt_compare=False to K3 sample_new()
Log:   /lus/lfs1aip2/projects/s5e/lob_pipeline/logs/integrated_ticks2-mamba2-soup_GOOG_3277622.out

Updated: 2026-03-22 20:35:00 UTC

Job:   3277877 (ticks-wsoup-360m) | 3277878 (ticks-soup-120m) | 3277879 (ticks-soup-55m) | 3277880 (ticks-moe-118m) | 3277882 (ticks-kda-104m)
User:  aramis.s5e
Notes: LOBbench rescore batch — all 5 models with new _ticks metrics (scoring only, skip inference)
Updated: 2026-03-22 21:15:00 UTC

Job:   3280297 (hier-cl-5k)
User:  kangli.s5e
Desc:  MarS closed-loop bench, hierarchical sampling, 5K seq, GOOG
Infra: 1N / 4 GPU
Time:  ~2h estimated | 4:00:00 limit
Log:   /lus/lfs1aip2/projects/s5e/lob_pipeline/.worktrees/mars-bench/results_mars_hier_5k/logs/bench_3280297.err

Updated: 2026-03-23 06:26:20 UTC


Job:   3280437 (P1d-6tok-smoke)
User:  kangli.s5e
Model: 55M (d=1024, L=12, B=16, ssm=1024) | ENCODING=6tok_lossless
Data:  8 tickers × 4yr | vocab=24641 | 6tok/msg | seq=3000
Infra: 1N / 4 GPU | BSZ=10/gpu, gBSZ=40
LR:    default
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_P1d_lossless_bpe/logs_lobs5/lobs5_3280437.out

Updated: 2026-03-23 07:29:30 UTC


Job:   3280446 (P1d-6tok-bsz4)
User:  kangli.s5e
Model: 55M (d=1024, L=12, B=16, ssm=1024) | ENCODING=6tok_lossless
Data:  8 tickers × 4yr | vocab=24641 | 6tok/msg | seq=3000
Infra: 1N / 4 GPU | BSZ=4/gpu, gBSZ=16
Note:  BSZ=10 OOM (76.73 GiB request). Reduced to BSZ=4.
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_P1d_lossless_bpe/logs_lobs5/lobs5_3280446.out

Updated: 2026-03-23 07:38:01 UTC


Job:   3280478 (P1d-WT-bsz10)
User:  kangli.s5e
Model: 55M (d=1024, L=12, B=16, ssm=1024) | ENCODING=6tok_lossless | WEIGHT_TYING=True
Data:  8 tickers × 4yr | vocab=24641 | 6tok/msg | seq=3000
Infra: 1N / 4 GPU | BSZ=10/gpu, gBSZ=40
Note:  Weight tying saves 25.2M decoder params. Testing if BSZ=10 fits now.
Log:   experiments/exp_P1d_lossless_bpe_weight_tying/logs_lobs5/lobs5_3280478.out

Updated: 2026-03-23 08:03:58 UTC


Job:   3280632 (bench_mars-399m-jaxlob_GOOG)
User:  aramis.s5e
Model: MarS-399M (LLaMA, 399M params)
Data:  GOOG × Jan 2026 | 3136 sequences
Infra: 1N / skip inference (reuse NPY) | Phase 1.5 jaxlob + scoring
Note:  Testing jaxlob engine (same as LOBS5) vs previous mlib run (Job 3279473, WS-21=0.7409)
Log:   /lus/lfs1aip2/projects/s5e/lob_pipeline/logs/integrated_mars-399m-jaxlob_GOOG_3280632.out

Updated: 2026-03-23 09:13:00 UTC

Job:   3285086 (bench_mars-399m-inline_GOOG)
User:  aramis.s5e
Model: MarS-399M (LLaMA, 399M params)
Data:  GOOG × Jan 2026 | 3136 sequences
Infra: 1N / 1 GPU (inference) + CPU (passthrough conversion + scoring)
Note:  Inline capture — inference saves per-step messages + L10 books directly, zero reconstruction
       A/B test vs jaxlob (Job 3280649, WS-21=0.4386) and mlib (Job 3279473, WS-21=0.7409)
Log:   /lus/lfs1aip2/projects/s5e/lob_pipeline/logs/integrated_mars-399m-inline_GOOG_3285086.out

Updated: 2026-03-23 10:02:00 UTC

Job:   3287836 (P1d-WT-CCE-bsz10)
User:  kangli.s5e
Model: 55M (d=1024, L=12, B=16) | ENCODING=6tok_lossless | WT=True | ChunkedCE=True
Data:  8 tickers × 4yr | vocab=24641 | 6tok/msg | seq=3000
Infra: 1N / 4 GPU | BSZ=10/gpu, gBSZ=40
Note:  Weight tying + Chunked CE. Testing if BSZ=10 fits (was OOM without CCE).
Log:   experiments/exp_P1d_lossless_bpe_weight_tying/logs_lobs5/lobs5_3287836.out

Updated: 2026-03-23 13:04:50 UTC


Job:   3290489 (hier-cl-5k-ws21 GOOG)
User:  kangli.s5e
Desc:  Full LOBbench WS-21 pipeline: inference (closed-loop, hierarchical) → jaxlob convert → WS-21 scoring, 5K seq
Infra: 20N (1 for inference, up to 20 for scoring)
Time:  ~3h estimated | 4:00:00 limit
Log:   /lus/lfs1aip2/projects/s5e/lob_pipeline/results_hier-cl-5k-ws21/

Updated: 2026-03-23 15:41:45 UTC


Job:   3297057 (hier-cl-5k-ws21-v2 GOOG)
User:  kangli.s5e
Desc:  Retry WS-21: inference (jaxob, closed-loop, hierarchical) + scoring, 5K seq, 6h walltime
Infra: 20N
Time:  ~4.5h inference + ~1h scoring | 6:00:00 limit
Log:   /lus/lfs1aip2/projects/s5e/lob_pipeline/logs/integrated_hier-cl-5k-ws21-v2_GOOG_3297057.*

Updated: 2026-03-23 20:02:00 UTC


Job:   3300248 (P1d-WT-CCE-55M-16n-24h)
User:  kangli.s5e
Step:  0/?  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: 55M GDN (d=1024, L=12, B=16, ssm=1024) | ENCODING=6tok_lossless | WT+CCE
Data:  8 tickers × 4yr (2022-2025) | test: Jan 2026
Infra: 16N / 64 GPU | BSZ=10/gpu, gBSZ=640
LR:    default (3e-3)
Loss:  (pending)
Speed: ~4.25 it/s (from smoke test)
Time:  0:00 elapsed  |  ~?h remaining  |  24:00 limit
ETA:   pending
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_P1d_lossless_bpe_weight_tying/logs_lobs5/lobs5_3300248.out

Updated: 2026-03-23 23:45:00 UTC

Job:   3300736 (P1d-WT-55M-16n-bsz8)
User:  kangli.s5e
Step:  0/?  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: 55M GDN (d=1024, L=12, B=16, ssm=1024) | ENCODING=6tok_lossless | WT (no CCE)
Data:  8 tickers × 4yr (2022-2025) | test: Jan 2026
Infra: 16N / 64 GPU | BSZ=8/gpu, gBSZ=512
LR:    default (3e-3)
Loss:  (pending)
Speed: ~5.25 it/s (from 1N smoke test)
Log:   experiments/exp_P1d_lossless_bpe_weight_tying/logs_lobs5/lobs5_3300736.out
Note:  Chunked CE disabled for multi-node (shard_map doesn't support it). BSZ=8 fits.

Updated: 2026-03-24 00:30:00 UTC

Job:   3303994 (P1d-WT-55M-16n-24h)
User:  kangli.s5e
Step:  0/?  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: 55M GDN (d=1024, L=12, B=16, ssm=1024) | ENCODING=6tok_lossless | WT | P1C-base train code
Data:  8 tickers × 4yr (2022-2025) | test: Jan 2026
Infra: 16N / 64 GPU | BSZ=4/gpu, gBSZ=256
LR:    default (3e-3)
Loss:  (pending)
Speed: ~est from 2N smoke
Log:   experiments/exp_P1d_lossless_bpe_weight_tying/logs_lobs5/lobs5_3303994.out
Note:  Uses P1C-base train_helpers.py (no chunked CE). Eval warmup skipped (NO_VALIDATION).

Updated: 2026-03-24 02:30:00 UTC

Job:   3304312 (P1d-WT-55M-16n-v2)
User:  kangli.s5e
Step:  0/?  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: 55M GDN (d=1024, L=12, B=16, ssm=1024) | ENCODING=6tok_lossless | WT
Data:  8 tickers × 4yr (2022-2025) | test: Jan 2026
Infra: 16N / 64 GPU | BSZ=4/gpu, gBSZ=256
Note:  P1C-base code + eval warmup skip + Triton solve disable + FP divergence fix (feba490e)
Log:   experiments/exp_P1d_lossless_bpe_weight_tying/logs_lobs5/lobs5_3304312.out

Updated: 2026-03-24 03:00:00 UTC

Job:   3308483 (jaxob-cl-5k-v3 GOOG)
User:  kangli.s5e
Desc:  5K jaxob closed-loop + hierarchical sampling + return bench + sample_indices saving
Infra: 1N / 4 GPU
Engine: jaxob (correct, no mlib negative volume bug)
Time:  ~4.5h inference + scoring | 6:00:00 limit
Log:   mars-bench/results_mars_jaxob_5k_v3/logs/

Updated: 2026-03-24 07:11:00 UTC


Job:   3317975 (nsa-rope-swa-6L-16n-10h)
User:  valentinm.s5e
Step:  0/233061  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: NSA ~100M (d=1024, L=6, H=16, head_dim=64) | block_size=24, block_counts=16, window_size=512
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 16N / 64 GPU | BSZ=4/gpu, gBSZ=256
LR:    3e-4
Loss:  — (not started)
Speed: ~1.87 s/step (estimated from 12L benchmark)
Time:  0:00 elapsed  |  ~10:00 remaining  |  10:00 limit
ETA:   ~18K steps in 10h
W&B:   pending
Log:   logs_lobs5/training_3317975_node0.log

Updated: 2026-03-20 03:30:00 UTC

Job:   3326112 (llm-conv-v2-3seed)
User:  kangli.s5e
Step:  0/20  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: FinGAN (~1K params, LSTM Generator+Discriminator)
Data:  AMZN, BLK, APA (CRSP daily, 80/10/10 split)
Infra: 1N / 1 GPU | BSZ=N (full dataset)
LR:    1e-4 (RMSprop)
Loss:  (pending — meta-learning loss evolution)
Speed: ~1.5 min/round (LLM call + 3 tickers × 3 seeds × 100 epochs)
Time:  0:00 elapsed  |  ~45min remaining  |  2:00 limit
ETA:   ~45min total
W&B:   N/A (results in JSON)
Log:   /projects/s5e/quant/fingan/FlowFM_v2/meta_loss/results/llm_v2_3326112.out

Updated: 2026-03-24 19:30:00 UTC

Job:   3326113 (se-agent-v2-3seed)
User:  kangli.s5e
Step:  0/3  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: FinGAN (~1K params, LSTM Generator+Discriminator)
Data:  AMZN, BLK, APA (CRSP daily, 80/10/10 split)
Infra: 1N / 1 GPU | BSZ=N (full dataset)
LR:    1e-4 (RMSprop)
Loss:  (pending — SE-Agent loss evolution: 5 initial + 3 cycles)
Speed: ~2 min/evaluation (3 tickers × 3 seeds × 100 epochs)
Time:  0:00 elapsed  |  ~50min remaining  |  2:00 limit
ETA:   ~50min total (23 evaluations × 2 min)
W&B:   N/A (results in JSON)
Log:   /projects/s5e/quant/fingan/FlowFM_v2/meta_loss/results/se_agent_v2_3326113.out

Updated: 2026-03-24 19:30:00 UTC

Job:   3326125 (llm-conv-v2-3seed) [resubmit: conda fix]
User:  kangli.s5e
Log:   /projects/s5e/quant/fingan/FlowFM_v2/meta_loss/results/llm_v2_3326125.out
Updated: 2026-03-24 19:35:00 UTC

Job:   3326126 (se-agent-v2-3seed) [resubmit: conda fix]
User:  kangli.s5e
Log:   /projects/s5e/quant/fingan/FlowFM_v2/meta_loss/results/se_agent_v2_3326126.out
Updated: 2026-03-24 19:35:00 UTC

Job:   3326430 (se-agent-v2-10cyc)
User:  kangli.s5e
Step:  0/10 cycles  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: FinGAN (~1K params) | SE-Agent v2 (5 initial + 10 cycles × 6 ops = 65 evals)
Data:  AMZN, BLK, APA | 3 seeds/eval
Infra: 1N / 1 GPU
Speed: ~45s/eval → ~65 evals × 45s ≈ 49min
Time:  0:00 elapsed  |  ~49min remaining  |  2:00 limit
Log:   /projects/s5e/quant/fingan/FlowFM_v2/meta_loss/results/se_agent_v2_3326430.out

Updated: 2026-03-24 19:58:00 UTC

Job:   3326799 (llm-v3-seeded) [v3: baseline code in prompt]
User:  kangli.s5e
Log:   /projects/s5e/quant/fingan/FlowFM_v2/meta_loss/results/llm_v2_3326799.out

Job:   3326800 (se-agent-v3-10cyc) [v3: 9 baselines seed pool + 10 cycles]
User:  kangli.s5e
Log:   /projects/s5e/quant/fingan/FlowFM_v2/meta_loss/results/se_agent_v2_3326800.out

Updated: 2026-03-24 20:25:00 UTC

Job:   3327137 (llm-v4-10tk)
User:  kangli.s5e
Model: FinGAN LSTM ~1K params (meta-loss evolution)
Data:  10 META_TRAIN tickers × 2 seeds (v4 train/val split)
Infra: 1N / 1 GPU | BSZ=full (~8846)
Time:  0:00 elapsed  |  ~1.5h remaining  |  2:00 limit
Log:   /projects/s5e/quant/fingan/FlowFM_v2/meta_loss/results/llm_v2_3327137.out

Job:   3327138 (se-agent-v4-10cyc)
User:  kangli.s5e
Model: FinGAN LSTM ~1K params (meta-loss evolution, SE-Agent)
Data:  10 META_TRAIN tickers × 2 seeds (v4 train/val split)
Infra: 1N / 1 GPU | BSZ=full (~8846)
Time:  0:00 elapsed  |  ~3.5h remaining  |  4:00 limit
Log:   /projects/s5e/quant/fingan/FlowFM_v2/meta_loss/results/se_agent_v2_3327138.out

Updated: 2026-03-24 UTC

Job:   3327230 (llm-v3-resume-r21)
User:  kangli.s5e
Model: FinGAN LSTM ~1K params (meta-loss LLM+Conv, resume from v3 R20)
Data:  10 META_TRAIN tickers × 2 seeds (v4 eval, v3 conversation)
Infra: 1N / 1 GPU | BSZ=full (~8846)
Time:  0:00 elapsed  |  ~1.5h remaining  |  2:00 limit
Log:   /projects/s5e/quant/fingan/FlowFM_v2/meta_loss/results/llm_v2_3327230.out

Updated: 2026-03-24 UTC

Job:   3327390 (llm-v4-60r-resume)
User:  kangli.s5e
Model: FinGAN LSTM ~1K params (meta-loss LLM+Conv, resume from v4 R20 to R60)
Data:  10 META_TRAIN tickers × 3 seeds (v4 eval, v4 conversation context)
Infra: 1N / 1 GPU | BSZ=full (~8846)
Time:  0:00 elapsed  |  ~2h remaining  |  3:00 limit
Log:   /projects/s5e/quant/fingan/FlowFM_v2/meta_loss/results/llm_v2_3327390.out

Updated: 2026-03-24 UTC

Job:   3332664 (hier-cl-5k-ws21-v4 GOOG)
User:  kangli.s5e
Desc:  WS-21 full pipeline v4: inference (mlib, closed-loop, hierarchical, saves sample_pairs.txt) + passthrough convert + WS-21 scoring, 5K seq
Infra: 20N (1 for inference, up to 20 for scoring)
Time:  6:00:00 limit
Log:   /lus/lfs1aip2/projects/s5e/lob_pipeline/logs/integrated_hier-cl-5k-ws21-v4_GOOG_3332664.*

Updated: 2026-03-25 04:03:01 UTC


Job:   3332684 (sp4-ctx500-bsz64-bench)
User:  kangli.s5e
Step:  0/200  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: NSA PyTorch d=1024, L=12, H=16 | ~200M params
Data:  8 tickers (GOOG,AAPL,NVDA,AMZN,META,TSLA,MSFT,AMD) | 2022-2025
Infra: 1N / 4 GPU | BSZ=64/gpu, gBSZ=256 | SP=4
LR:    5e-4
Loss:  (pending)
Speed: ~? s/step (steady-state benchmark)
Time:  0:00 elapsed  |  ~5min remaining  |  30min limit
ETA:   ctx=500 steady-state speed measurement
W&B:   N/A (no_wandb)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O2b_rope_ctx4000_sp/logs/nsa_torch_3332684.out

Updated: 2026-03-25 00:00:00 UTC

Job:   3332685 (sp4-ctx500-bsz128-oom)
User:  kangli.s5e
Step:  0/3  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: NSA PyTorch d=1024, L=12, H=16 | ~200M params
Data:  8 tickers | 2022-2025
Infra: 1N / 4 GPU | BSZ=128/gpu | SP=4
LR:    5e-4
Loss:  (pending)
Speed: OOM probe
Time:  0:00 elapsed  |  ~2min remaining  |  15min limit
ETA:   ctx=500 max BSZ check (BSZ=128)
W&B:   N/A
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O2b_rope_ctx4000_sp/logs/nsa_torch_3332685.out

Updated: 2026-03-25 00:00:00 UTC

Job:   3332686 (sp4-ctx4k-bsz5-bench)
User:  kangli.s5e
Step:  0/30  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: NSA PyTorch d=1024, L=12, H=16 | ~200M params
Data:  8 tickers | 2022-2025
Infra: 1N / 4 GPU | BSZ=5/gpu, gBSZ=20 | SP=4
LR:    5e-4
Loss:  (pending)
Speed: ~? s/step (steady-state benchmark at ctx=4000)
Time:  0:00 elapsed  |  ~30min remaining  |  60min limit
ETA:   ctx=4000 true steady-state speed measurement
W&B:   N/A
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O2b_rope_ctx4000_sp/logs/nsa_torch_3332686.out

Updated: 2026-03-25 00:00:00 UTC

Job:   3332830 (O8-attn-72m-2n-30m)
User:  kangli.s5e
Step:  0/50  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: Transformer ~72M (d=768, L=6, heads=12, flash, bf16, remat) | ~72M params
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 2N / 8 GPU | BSZ=8/gpu, gBSZ=64
LR:    5e-4
Loss:  (pending)
Speed: (pending)
Time:  0:00 elapsed  |  ~30min limit
ETA:   smoke test (CURTAIL=50)
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O8_self_attention/logs_lobs5/lobs5_3332830.out

Updated: 2026-03-25 04:30:00 UTC

Job:   3332841 (bench-10tok-GOOG)
User:  kangli.s5e
Model: GDN 984M (d=2048, L=24, B=32, ssm=2048) | 10tok encoding | step 13045
Data:  GOOG Jan 2026
Infra: 1N / 4 GPU | N_SEQ=256, N_GEN=500
Task:  LOBbench scoring (inference + unconditional bench)
Ckpt:  experiments/exp_P1e_10tok/checkpoints/j3320846_obt15rs4_3320846/13045/
Log:   experiments/exp_P1e_10tok/logs_lobs5/bench_10tok_3332841.out

Updated: 2026-03-25 14:30:00 UTC

Job:   3333043 (bench-10tok-GOOG) — resubmit of 3332841 (data path fix)
User:  kangli.s5e
Model: GDN 984M (d=2048, L=24, B=32, ssm=2048) | 10tok encoding | step 13045
Data:  GOOG Jan 2026
Infra: 1N / 4 GPU | N_SEQ=256, N_GEN=500
Task:  LOBbench scoring (inference + unconditional bench)
Log:   experiments/exp_P1e_10tok/logs_lobs5/bench_10tok_3333043.out

Updated: 2026-03-25 15:45:00 UTC

Job:   3333045 (bench-10tok-GOOG) — resubmit #2 (cancel_t typo fix)
User:  kangli.s5e
Model: GDN 984M | 10tok | step 13045
Infra: 1N / 4 GPU | N_SEQ=256, N_GEN=500
Log:   experiments/exp_P1e_10tok/logs_lobs5/bench_10tok_3333045.out

Updated: 2026-03-25 16:00:00 UTC

Job:   3333058 (O8-attn-d1024-2n-30m)
User:  kangli.s5e
Step:  0/50  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: Transformer (d=1024, L=6, heads=16, flash, bf16, remat) | ~130M params
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 2N / 8 GPU | BSZ=4/gpu, gBSZ=32
LR:    5e-4
Loss:  (pending)
Speed: (pending) | Target: ≤8.0 s/step (10x GDN)
Time:  0:00 elapsed  |  ~30min limit
ETA:   smoke test (CURTAIL=50)
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O8_self_attention/logs_lobs5/lobs5_3333058.out

Updated: 2026-03-25 04:55:00 UTC

Job:   3333100 (O8-attn-d1024-2n-bench300)
User:  kangli.s5e
Step:  0/300  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: Transformer (d=1024, L=6, heads=16, flash, bf16, remat)
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 2N / 8 GPU | BSZ=4/gpu, gBSZ=32
LR:    5e-4
Loss:  (pending)
Speed: (pending) | Target: ≤8.0 s/step (10x GDN) | Prev: ~0.081 s/step @50 steps
Time:  0:00 elapsed  |  ~30min limit
ETA:   benchmark (CURTAIL=300, need step 250-300 for steady state)
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O8_self_attention/logs_lobs5/lobs5_3333100.out

Updated: 2026-03-25 05:10:00 UTC

Job:   3338873 (26tok-adamw-d1024-goog25)
User:  aramis.s5e
Model: 55M (d=1024, L=12, B=16, ssm=1024) | vocab=2112 (26tok)
Data:  GOOG × 2025 | test: 2026-01-02
Infra: 1N / 4 GPU | BSZ=7/gpu, gBSZ=28
Opt:   AdamW WD=0.05, cosine_steps=100K
Note:  22→24→26tok ablation — matches j2633975 (24tok) config exactly

Job:   3338885 (22tok-adamw-d1024-goog25)
User:  aramis.s5e
Model: 75M (d=1024, L=12, B=16, ssm=1024) | vocab=12012 (22tok)
Data:  GOOG × 2025 (lob_preproc/) | test: 2026-01-02
Infra: 1N / 4 GPU | BSZ=7/gpu, gBSZ=28
Opt:   AdamW WD=0.05, cosine_steps=100K
Branch: exp/N1c-22tok
Note:  22→24→26tok ablation — same core model, larger embedding due to vocab

Updated: 2026-03-20 05:30 UTC

Job:   3346100 (kda-hardclamp-resume)
User:  aramis.s5e
Step:  28305/? (resuming from j2906749 hard-clamp checkpoint)
Model: KDA 104M (d=1024, L=6, B=16, ssm=1024, hard-clamp alpha) | 103,623,325 params
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 16N / 64 GPU | BSZ=2/gpu, gBSZ=128 | Local Steps K=1
LR:    5e-4
Loss:  0.6532 (at step 28305, best in chain)
Speed: TBD
Time:  0:00 elapsed | 24:00 limit
ETA:   TBD
W&B:   TBD (project: oxford-lob/lobs5-K5-KDA)
Log:   exp_K5b_KDA_hardclamp/logs_lobs5/lobs5_3346100.out

Note: Resume from j2906749 step 28305 with ORIGINAL hard-clamp gate code (commit 578a7f6).
      Previous resume j3149352 accidentally swapped to sigmoid÷16 gate (commit 81f61f2).
      This run uses worktree exp_K5b_KDA_hardclamp (branch exp/K5b-KDA-hardclamp).

Updated: 2026-03-25 17:30:00 UTC

Job:   3346574 (kda-hc-from21k)
User:  aramis.s5e
Step:  21858/? (resuming from j2906749 hard-clamp checkpoint)
Model: KDA 104M (d=1024, L=6, B=16, ssm=1024, hard-clamp alpha) | 103,623,325 params
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 16N / 64 GPU | BSZ=2/gpu, gBSZ=128 | Local Steps K=1
LR:    5e-4
Loss:  ~0.69 (at step 21858)
Speed: TBD
Time:  0:00 elapsed | 24:00 limit
ETA:   TBD
W&B:   TBD (project: oxford-lob/lobs5-K5-KDA)
Log:   exp_K5b_KDA_hardclamp/logs_lobs5/lobs5_3346574.out

Note: Resume from j2906749 step 21858 (6.5k steps BEFORE the WS=0.1929 peak).
      Test: does KDA hard-clamp reproduce the s28305 dip, then degrade?
      If yes → overfitting. If no → original j3149352 resume was the cause.
      Cancelled 3346100 (was resuming from 28305, less clean experiment).

Updated: 2026-03-25 17:45:00 UTC

Job:   3349240 (sp4-ctx4k-compile)
User:  kangli.s5e
Step:  0/30  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: NSA PyTorch d=1024, L=12, H=16 | ~200M params | torch.compile max-autotune
Data:  8 tickers (GOOG,AAPL,NVDA,AMZN,META,TSLA,MSFT,AMD) | 2022-2025
Infra: 1N / 4 GPU | BSZ=5/gpu, gBSZ=20 | SP=4
LR:    5e-4
Loss:  (pending)
Speed: ~? s/step (torch.compile benchmark at ctx=4000)
Time:  0:00 elapsed  |  ~? remaining  |  60min limit
ETA:   torch.compile MFU test
W&B:   N/A (no_wandb)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O2b_rope_ctx4000_sp/logs/nsa_torch_3349240.out

Updated: 2026-03-25 01:00:00 UTC

Job:   3349241 (sp4-ctx500-compile)
User:  kangli.s5e
Step:  0/30  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: NSA PyTorch d=1024, L=12, H=16 | ~200M params | torch.compile max-autotune
Data:  8 tickers | 2022-2025
Infra: 1N / 4 GPU | BSZ=64/gpu, gBSZ=256 | SP=4
LR:    5e-4
Loss:  (pending)
Speed: ~? s/step (torch.compile benchmark at ctx=500)
Time:  0:00 elapsed  |  ~? remaining  |  30min limit
ETA:   torch.compile ctx=500 baseline
W&B:   N/A (no_wandb)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O2b_rope_ctx4000_sp/logs/nsa_torch_3349241.out

Updated: 2026-03-25 01:00:00 UTC

Job:   3349535 (O8-attn-125m-16n-1ep)
User:  kangli.s5e
Step:  0/?  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: Transformer 125M (d=1024, L=6, heads=16, flash, bf16, remat) | 125,449,011 params
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 16N / 64 GPU | BSZ=4/gpu, gBSZ=256
LR:    5e-4
Loss:  (pending)
Speed: (pending) | 2N baseline: 0.084 s/step
Time:  0:00 elapsed  |  ~1h limit
ETA:   ~10min estimated (full 1 epoch)
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O8_self_attention/logs_lobs5/lobs5_3349535.out

Updated: 2026-03-25 06:00:00 UTC

Job:   3349887 (O8-attn-125m-16n-1ep)
User:  kangli.s5e
Step:  0/?  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: Transformer 125M (d=1024, L=6, heads=16, flash, bf16, remat) | 125,449,011 params
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 16N / 64 GPU | BSZ=4/gpu, gBSZ=256
LR:    5e-4
Loss:  (pending)
Speed: (pending) | 2N baseline: 0.084 s/step
Time:  0:00 elapsed  |  ~1h limit
ETA:   ~10min estimated (full 1 epoch, NO_AUTO_RESUME=1)
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O8_self_attention/logs_lobs5/lobs5_3349887.out

Updated: 2026-03-25 18:25:00 UTC

Job:   3350352 (bench-10tok-qgate) — quality gate fix validation
User:  kangli.s5e
Model: GDN 984M | 10tok | step 13045 | WITH quality gate on ref_delta_s
Infra: 1N / 4 GPU | N_SEQ=256, N_GEN=500
Log:   experiments/exp_P1e_10tok/logs_lobs5/bench_10tok_3350352.out

Updated: 2026-03-25 22:00:00 UTC

Job:   3351487 (O8-attn-125m-16n-24h)
User:  kangli.s5e
Step:  0/233061  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: Transformer 125M (d=1024, L=6, heads=16, flash, bf16, remat) | 125,449,011 params
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 16N / 64 GPU | BSZ=4/gpu, gBSZ=256 | exclude=nid010307,nid010740
LR:    5e-4
Loss:  (pending)
Speed: ~1.56 it/s (0.641 s/step) expected | ~41.5h/epoch
Time:  0:00 elapsed  |  ~41.5h needed  |  24:00 limit
ETA:   Will complete ~57% of epoch in 24h. Auto-resume enabled.
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O8_self_attention/logs_lobs5/lobs5_3351487.out

Updated: 2026-03-25 19:00:00 UTC

Job:   3355487 (O8-attn-125m-4n-24h)
User:  kangli.s5e
Step:  0/?  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: Transformer 125M (d=1024, L=6, heads=16, flash, bf16, remat) | 125,449,011 params
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 4N / 16 GPU | BSZ=4/gpu, gBSZ=64 | --contiguous
LR:    5e-4
Loss:  (pending)
Speed: (pending)
Time:  0:00 elapsed  |  24:00 limit
ETA:   Strategy: 4N contiguous for stability (16N had 3 consecutive Slingshot crashes)
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O8_self_attention/logs_lobs5/lobs5_3355487.out

Updated: 2026-03-25 20:30:00 UTC
Job:   3355829 (grpo-smoke-2n-30m)
User:  kangli.s5e
Model: GDN 75M (d=1024, L=6, B=16, ssm=1024, gdn_nh=8, gdn_hd=128)
Infra: 2N / 8 GPU | BSZ=4/gpu, gBSZ=32
GRPO:  steps=10, G=4, lr=1e-5, clip=0.2, kl=0.01
Ckpt:  j2722414_mr8w0rjg_2722414 step=23356
Time:  00:30:00 limit
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O9_rl_ic/logs_lobs5/grpo_3355829.out

Updated: 2026-03-25 19:29:48 UTC

Job:   3356919 (grpo-smoke-2n-30m-v2)
User:  kangli.s5e
Model: GDN 75M (d=1024, L=6, B=16, ssm=1024, gdn_nh=8, gdn_hd=128)
Infra: 2N / 8 GPU | BSZ=4/gpu
GRPO:  steps=10, G=4, lr=1e-5
Ckpt:  j2722414_mr8w0rjg_2722414 step=23356
Fix:   GPUS_PER_NODE=4, NNODES=SLURM_NNODES
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O9_rl_ic/logs_lobs5/grpo_3356919.out

Updated: 2026-03-25 19:39:18 UTC

Job:   3357765 (O8-attn-K10-16n-bench300)
User:  kangli.s5e
Step:  0/300  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: Transformer 125M (d=1024, L=6, heads=16, flash, bf16, remat) | 125,449,011 params
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 16N / 64 GPU | BSZ=4/gpu, gBSZ=256 | LOCAL_STEPS_K=10 (fix for CXI overflow)
LR:    5e-4
Loss:  (pending)
Speed: (pending)
Time:  0:00 elapsed  |  30min limit
ETA:   Benchmark: testing if K=10 fixes Slingshot RC:265 crash
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O8_self_attention/logs_lobs5/lobs5_3357765.out

Updated: 2026-03-25 21:30:00 UTC

Job:   3358713 (sp4-ctx4k-cdefault)
User:  kangli.s5e
Step:  0/30  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: NSA PyTorch d=1024, L=12, H=16 | ~200M params | torch.compile mode=default
Data:  8 tickers | 2022-2025
Infra: 1N / 4 GPU | BSZ=5/gpu, gBSZ=20 | SP=4
LR:    5e-4
Loss:  (pending)
Speed: ~? s/step (compile=default benchmark, no CUDA graphs)
Time:  0:00 elapsed  |  ~? remaining  |  60min limit
ETA:   torch.compile mode=default vs 240 s/step baseline
W&B:   N/A
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O2b_rope_ctx4000_sp/logs/nsa_torch_3358713.out

Updated: 2026-03-25 06:00:00 UTC

Job:   3358714 (sp4-ctx500-cdefault)
User:  kangli.s5e
Step:  0/30  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: NSA PyTorch d=1024, L=12, H=16 | ~200M params | torch.compile mode=default
Data:  8 tickers | 2022-2025
Infra: 1N / 4 GPU | BSZ=64/gpu, gBSZ=256 | SP=4
LR:    5e-4
Loss:  (pending)
Speed: ~? s/step (compile=default benchmark, no CUDA graphs)
Time:  0:00 elapsed  |  ~? remaining  |  30min limit
ETA:   torch.compile mode=default vs 9.8 s/step baseline
W&B:   N/A
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O2b_rope_ctx4000_sp/logs/nsa_torch_3358714.out

Updated: 2026-03-25 06:00:00 UTC

Job:   3361652 (kda-hc-k64-16n)
User:  aramis.s5e
Step:  0/? (fresh start)
Model: KDA 104M (d=1024, L=6, B=16, ssm=1024, hard-clamp alpha) | 103,623,325 params
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 16N / 64 GPU | BSZ=2/gpu, gBSZ=128 | Local Steps K=64
LR:    5e-4
Loss:  TBD
Speed: ~2.2 it/s (from smoke test)
Time:  0:00 elapsed | 24:00 limit
ETA:   TBD
W&B:   TBD (project: oxford-lob/lobs5-K5-KDA)
Log:   exp_K5b_KDA_hardclamp/logs_lobs5/lobs5_3361652.out

Note: Fresh KDA hard-clamp training with K=64 local steps.
      K=64 averages gradients over 64×128=8192 samples per optimizer update.
      Hypothesis: high K stabilises noisy per-key-dim alpha gradients.
      Effective tokens/update = 64 × 128 × 12000 = 98.3M tokens.

Updated: 2026-03-25 22:45:00 UTC

Job:   3361783 (sp4-ctx4k-csel)
User:  kangli.s5e
Step:  0/30  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: NSA PyTorch d=1024, L=12, H=16 | ~200M params | compile=default + @compiler.disable on loop fns
Data:  8 tickers | 2022-2025
Infra: 1N / 4 GPU | BSZ=5/gpu, gBSZ=20 | SP=4
LR:    5e-4
Loss:  (pending)
Speed: ~? s/step (selective compile: skip sliding_window + compressed_chunked loops)
Time:  0:00 elapsed  |  ~? remaining  |  60min limit
ETA:   ctx=4000 selective compile benchmark (expect compile ~2-5min, then ~20% MFU)
W&B:   N/A
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O2b_rope_ctx4000_sp/logs/nsa_torch_3361783.out

Updated: 2026-03-25 12:00:00 UTC

Job:   3361784 (sp4-ctx500-csel)
User:  kangli.s5e
Step:  0/30  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: NSA PyTorch d=1024, L=12, H=16 | ~200M params | compile=default + @compiler.disable on loop fns
Data:  8 tickers | 2022-2025
Infra: 1N / 4 GPU | BSZ=64/gpu, gBSZ=256 | SP=4
LR:    5e-4
Loss:  (pending)
Speed: ~? s/step (selective compile: comparison vs full compile at ctx=500)
Time:  0:00 elapsed  |  ~? remaining  |  30min limit
ETA:   ctx=500 selective vs full compile comparison
W&B:   N/A
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O2b_rope_ctx4000_sp/logs/nsa_torch_3361784.out

Updated: 2026-03-25 12:00:00 UTC

Job:   3363792 (O8-attn-125m-16n-24h-bsz24)
User:  kangli.s5e
Step:  0/38844  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: Transformer 125M (d=1024, L=6, heads=16, flash, bf16, remat) | 125,449,011 params
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 16N / 64 GPU | BSZ=24/gpu, gBSZ=1536 | LOCAL_STEPS_K=10
LR:    5e-4
Loss:  (pending)
Speed: ~0.66 s/step est (from 2N sweep)
Time:  0:00 elapsed  |  ~7h est  |  24:00 limit
ETA:   ~7h (3.4x faster than BSZ=4 job)
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O8_self_attention/logs_lobs5/lobs5_3363792.out

Updated: 2026-03-26 02:00:00 UTC

Job:   3364018 (O8-attn-125m-16n-24h-bsz16)
User:  kangli.s5e
Step:  0/58265  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: Transformer 125M (d=1024, L=6, heads=16, flash, bf16, remat) | 125,449,011 params
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 16N / 64 GPU | BSZ=16/gpu, gBSZ=1024 | LOCAL_STEPS_K=10
LR:    5e-4
Loss:  (pending)
Speed: ~0.47 s/step est
Time:  0:00 elapsed  |  ~7.6h est  |  24:00 limit
ETA:   ~7.6h total (BSZ=24 OOM at 16N, reverted to BSZ=16)
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O8_self_attention/logs_lobs5/lobs5_3364018.out

Updated: 2026-03-26 02:10:00 UTC

Job:   3364038 (grpo-lr1e7-50s)
User:  kangli.s5e
Model: GDN 75M (d=1024, L=6, B=16)
GRPO:  steps=50, G=8, LR=1e-7 (down from 1e-5), clip=0.2
Ckpt:  j2722414 step=23356
Infra: 2N / 8 GPU
Time:  01:00:00 limit
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O9_rl_ic/logs_lobs5/training_3364038_node0.log

Updated: 2026-03-26 02:15:12 UTC

Job:   3364293 (O8-attn-125m-16n-24h-bsz8)
User:  kangli.s5e
Step:  0/116530  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: Transformer 125M (d=1024, L=6, heads=16, flash, bf16, remat) | 125,449,011 params
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 16N / 64 GPU | BSZ=8/gpu, gBSZ=512 | LOCAL_STEPS_K=10
LR:    5e-4
Loss:  (pending)
Speed: ~5.99 it/s (0.17 s/step) from 16N sweep
Time:  0:00 elapsed  |  ~5.4h est  |  24:00 limit
ETA:   ~5.4h (3.2x faster than BSZ=4, 16N topology-optimized BSZ)
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O8_self_attention/logs_lobs5/lobs5_3364293.out

Updated: 2026-03-26 02:45:00 UTC

Job:   3364666 (sp4-ctx4k-flashattn)
User:  kangli.s5e
Step:  0/30  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: NSA PyTorch d=1024, L=12, H=16 | ~200M params | Branch 3 = flash_attn sliding window
Data:  8 tickers | 2022-2025
Infra: 1N / 4 GPU | BSZ=5/gpu, gBSZ=20 | SP=4
LR:    5e-4
Speed: ~? s/step (flash_attn eager benchmark — expect huge speedup from 240s baseline)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O2b_rope_ctx4000_sp/logs/nsa_torch_3364666.out
Updated: 2026-03-26 12:00:00 UTC

Job:   3364667 (sp4-ctx4k-flash-compile)
User:  kangli.s5e
Step:  0/30  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: NSA PyTorch d=1024, L=12, H=16 | ~200M params | flash_attn + torch.compile=default
Data:  8 tickers | 2022-2025
Infra: 1N / 4 GPU | BSZ=5/gpu, gBSZ=20 | SP=4
LR:    5e-4
Speed: ~? s/step (flash_attn + compile — potential 20-40% MFU)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O2b_rope_ctx4000_sp/logs/nsa_torch_3364667.out
Updated: 2026-03-26 12:00:00 UTC

Job:   3364668 (sp4-ctx500-flash-compile)
User:  kangli.s5e
Step:  0/30  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: NSA PyTorch d=1024, L=12, H=16 | ~200M params | flash_attn + compile (regression check)
Data:  8 tickers | 2022-2025
Infra: 1N / 4 GPU | BSZ=64/gpu, gBSZ=256 | SP=4
LR:    5e-4
Speed: ~? s/step (expect ~4.8s like previous full compile)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O2b_rope_ctx4000_sp/logs/nsa_torch_3364668.out
Updated: 2026-03-26 12:00:00 UTC

Job:   3364790 (P1d-resume-fix-2n-30m)
User:  kangli.s5e
Model: GDN 55M (d=1024, L=12, B=16, ssm=1024) | 171,834,093 params
Data:  8 tickers x 4yr (2022-2025) | test: Jan 2026
Infra: 2N / 8 GPU | BSZ=4/gpu, gBSZ=32 | CURTAIL_EPOCHS=50
Purpose: Verify step-based checkpoint fix (resume deadlock at ~step 34028)
Log:   experiments/exp_P1d_lossless_bpe_weight_tying/logs_lobs5/

Updated: 2026-03-26 05:07:21 UTC


Job:   3364842 (10tok-82M-16n-1ep)
User:  kangli.s5e
Step:  0/113775  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: GDN 82M (d=1024, L=6, nh=8, hd=128, ev=2) | 10tok encoding | ~81,584,905 params
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 16N / 64 GPU | BSZ=4/gpu, gBSZ=256 | MSG_SEQ_LEN=1024
LR:    3e-3 (default cosine anneal)
Loss:  (pending)
Speed: (pending)
Time:  0:00 elapsed  |  ~8-12h remaining  |  24:00 limit
ETA:   ~8-12h (estimated from P1b 82M)
W&B:   (pending)
Log:   experiments/exp_P1e_10tok/logs_lobs5/lobs5_3364842.out

Updated: 2026-03-26 16:00:00 UTC

Job:   3364868 (P1d-6tok-55M-16n-24h-resume)
User:  kangli.s5e
Step:  33401/233061 (resuming)
Model: GDN 55M (d=1024, L=12, B=16, ssm=1024) | 171,834,093 params
Data:  8 tickers x 4yr (2022-2025) | test: Jan 2026
Infra: 16N / 64 GPU | BSZ=4/gpu, gBSZ=256
LR:    5e-4
Speed: ~1.22 it/s (0.82 s/step) expected
Time:  24:00:00 limit | ~45% of epoch expected
W&B:   pending
Log:   experiments/exp_P1d_lossless_bpe_weight_tying/logs_lobs5/

Updated: 2026-03-26 05:35:57 UTC


Job:   3364871 (nsa-ctx4k-2n-smoke)
User:  kangli.s5e
Step:  0/50  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: NSA PyTorch d=1024, L=12, H=16 | ~200M params | flash_attn sliding window
Data:  8 tickers (GOOG,AAPL,NVDA,AMZN,META,TSLA,MSFT,AMD) × 4yr (2022-2025)
Infra: 2N / 8 GPU | BSZ=5/gpu, gBSZ=40 | SP=4
LR:    5e-4
Loss:  (pending)
Speed: ~16 s/step (est from 1N benchmark)
Time:  0:00 elapsed  |  ~13min remaining  |  30min limit
ETA:   Multi-node SP+DDP smoke test
W&B:   N/A (no_wandb)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O2b_rope_ctx4000_sp/logs/nsa_torch_3364871.out

Updated: 2026-03-26 06:00:00 UTC

Job:   3364872 (nsa-ctx4k-16n-24h)
User:  kangli.s5e
Step:  0/11719  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: NSA PyTorch d=1024, L=12, H=16 | ~200M params | flash_attn sliding window
Data:  8 tickers (GOOG,AAPL,NVDA,AMZN,META,TSLA,MSFT,AMD) × 4yr (2022-2025) | test: 2026-01
Infra: 16N / 64 GPU | BSZ=5/gpu, gBSZ=320 | SP=4
LR:    5e-4
Loss:  (pending)
Speed: ~20 s/step (est, multi-node overhead)
Time:  0:00 elapsed  |  ~65h remaining  |  24:00 limit
ETA:   ~37% of epoch in 24h → will need resume
W&B:   (pending — lobs5-nsa-torch)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O2b_rope_ctx4000_sp/logs/nsa_torch_3364872.out

Updated: 2026-03-26 06:00:00 UTC

Job:   3364874 (dfm-94M-4n-24h)
User:  kangli.s5e
Step:  0/???  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: GDN 94M (d=1024, L=12, B=16, ssm=1024) | DFM post-training
Data:  8 tickers × 4yr (2022-2025) | cond=250, gen=250
Infra: 4N / 16 GPU | BSZ=10/gpu, gBSZ=160
LR:    backbone=5e-5, new(AdaLN)=1e-4
AR:    checkpoint j2722414 step 23356
Time:  0:00 elapsed  |  ~24:00 limit
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O12_GDN_DFM/logs_lobs5/dfm_3364874.out

Updated: 2026-03-26 05:41:19 UTC

Job:   3364896 (bench-P1a-9tok-GOOG)
User:  kangli.s5e
Model: GDN ~151M (d=1024, L=12, B=16) | 9tok/24tok encoding
Data:  GOOG Jan 2026 | 1024 sequences | 500 cond + 500 gen
Infra: 1N / 4 GPU | inference + LOBbench scoring
Log:   experiments/exp_P1a_flat_9tok/logs_lobs5/bench_9tok_3364896.out

Updated: 2026-03-26 06:05:02 UTC


Job:   3364917 (nsa-ctx4k-2n-nw0)
User:  kangli.s5e
Step:  0/50  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: NSA PyTorch d=1024, L=12, H=16 | ~200M params | flash_attn | N_WORKERS=0 (fix fork deadlock)
Data:  8 tickers × 4yr (2022-2025)
Infra: 2N / 8 GPU | BSZ=5/gpu, gBSZ=40 | SP=4
LR:    5e-4
Speed: ~16 s/step (est)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O2b_rope_ctx4000_sp/logs/nsa_torch_3364917.out
Updated: 2026-03-26 06:30:00 UTC

Job:   3364918 (nsa-ctx4k-16n-24h-v2)
User:  kangli.s5e
Step:  0/23273  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: NSA PyTorch d=1024, L=12, H=16 | ~200M params | flash_attn | N_WORKERS=0
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 16N / 64 GPU | BSZ=5/gpu, gBSZ=320 | SP=4
LR:    5e-4
W&B:   (pending — lobs5-nsa-torch)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O2b_rope_ctx4000_sp/logs/nsa_torch_3364918.out
Updated: 2026-03-26 06:30:00 UTC

Job:   3364922 (nsa-ctx1k-2n-smoke)
User:  kangli.s5e
Step:  0/50  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: NSA PyTorch d=1024, L=12, H=16 | ~200M params | flash_attn | ctx=1000 (24K tok)
Data:  8 tickers × 4yr (2022-2025)
Infra: 2N / 8 GPU | BSZ=20/gpu, gBSZ=160 | SP=4
LR:    5e-4
Speed: ~4 s/step (est from ctx=500 scaling)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O2b_rope_ctx4000_sp/logs/nsa_torch_3364922.out
Updated: 2026-03-26 07:00:00 UTC

Job:   3364923 (nsa-ctx1k-16n-24h)
User:  kangli.s5e
Step:  0/?  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: NSA PyTorch d=1024, L=12, H=16 | ~200M params | flash_attn | ctx=1000 (24K tok)
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 16N / 64 GPU | BSZ=20/gpu, gBSZ=1280 | SP=4
LR:    5e-4
W&B:   (pending — lobs5-nsa-torch)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O2b_rope_ctx4000_sp/logs/nsa_torch_3364923.out
Updated: 2026-03-26 07:00:00 UTC
Job:   3364954 (O14-swiglu-94M-16n-1ep)
User:  kangli.s5e
Model: GDN 94M + SwiGLU (d=1024, L=6, nh=8, hd=128, expand_v=2, mlp_ratio=8/3)
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 16N / 64 GPU | BSZ=4/gpu, gBSZ=256
LR:    5e-4
Log:   experiments/exp_O14_ffn_swiglu/logs_lobs5/training_3364954_node0.log

Updated: 2026-03-26 06:38:47 UTC


Job:   3366837 (kda-hc-k64-resume)
User:  aramis.s5e
Step:  20352/? (resuming from j3361652)
Model: KDA 104M (d=1024, L=6, B=16, ssm=1024, hard-clamp alpha) | 103,623,325 params
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 16N / 64 GPU | BSZ=2/gpu, gBSZ=128 | Local Steps K=64
LR:    5e-4
Loss:  TBD
Speed: ~1 it/s (16N)
Time:  0:00 elapsed | 24:00 limit
W&B:   TBD (project: oxford-lob/lobs5-K5-KDA)
Log:   exp_K5b_KDA_hardclamp/logs_lobs5/lobs5_3366837.out

Note: Resume of 3361652 (K=64 hard-clamp fresh run, reached step 20352 in 5h before SIGTERM).

Updated: 2026-03-26 05:00:00 UTC

Job:   3374290 (O14-swiglu-94M-16n-bsz2)
User:  kangli.s5e
Model: GDN 94M + SwiGLU (d=1024, L=6, ratio=8/3, d_ff=2816)
Infra: 16N / 64 GPU | BSZ=2/gpu, gBSZ=128
Note:  BSZ=4 OOM at 16N (MEM_FRACTION=0.80), reduced to BSZ=2
Updated: 2026-03-26 16:05:36 UTC


Job:   3378771 (O8-attn-125m-16n-24h-v4)
User:  kangli.s5e
Step:  0/77687 (BSZ=12 → fewer steps)
Model: Transformer 125M (d=1024, L=6, heads=16, flash, bf16, remat=False) | 125,449,011 params
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 16N / 64 GPU | BSZ=12/gpu, gBSZ=768 | LOCAL_STEPS_K=10
LR:    5e-4
Loss:  (pending)
Speed: ~0.43 s/step (estimated from benchmark)
Time:  0:00 elapsed | ~9.3h estimated | 24:00 limit
ETA:   ~9.3h (comfortable margin)
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O8_self_attention/logs_lobs5/training_3378771_node0.log

Updated: 2026-03-26 18:30:00 UTC

Job:   3378833 (nsa-ctx4k-16n-24h-v3)
User:  kangli.s5e
Step:  0/23273  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: NSA PyTorch d=1024, L=12, H=16 | ~200M params | flash_attn | CKPT_EVERY=500
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 16N / 64 GPU | BSZ=5/gpu, gBSZ=320 | SP=4
LR:    5e-4
W&B:   (pending — lobs5-nsa-torch)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O2b_rope_ctx4000_sp/logs/nsa_torch_3378833.out
Note:  Periodic checkpoint every 500 steps (fix for NCCL crash data loss)
Updated: 2026-03-26 18:00:00 UTC

Job:   3378834 (nsa-ctx1k-16n-24h-v2)
User:  kangli.s5e
Step:  0/23301  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: NSA PyTorch d=1024, L=12, H=16 | ~200M params | flash_attn | ctx=1000 (24K tok) | CKPT_EVERY=500
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 16N / 64 GPU | BSZ=20/gpu, gBSZ=1280 | SP=4
LR:    5e-4
W&B:   (pending — lobs5-nsa-torch)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O2b_rope_ctx4000_sp/logs/nsa_torch_3378834.out
Note:  Periodic checkpoint every 500 steps
Updated: 2026-03-26 18:00:00 UTC

Job:   3378836 (10tok-82M-16n-resume)
User:  kangli.s5e
Step:  87482/113775  [███████████████████████░░░░░░░]  76.9%
Model: GDN 84M (d=1024, L=6, nh=8, hd=128, ev=2) | 10tok encoding | 83,642,101 params
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 16N / 64 GPU | BSZ=4/gpu, gBSZ=256 | MSG_SEQ_LEN=1024
LR:    5e-4 (cosine anneal, resuming mid-schedule)
Loss:  0.4259 (at step 87,400, from parent job 3364842)
Speed: ~0.33 s/step (3.18 it/s)
Time:  0:00 elapsed  |  ~2.5h remaining  |  04:00 limit
ETA:   ~2.5h, well within 4h limit
W&B:   (will create new run or resume 4cbqfanv)
Log:   experiments/exp_P1e_10tok/logs_lobs5/lobs5_3378836.out
Ckpt:  resuming from step 87,482 | parent: j3364842_4cbqfanv_3364842

Updated: 2026-03-26 23:25:00 UTC

Job:   3379177 (O14-swiglu-164M-16n-bsz2)
User:  kangli.s5e
Model: GDN 164M + SwiGLU (d=1024, L=6, nh=8, hd=128, expand_v=2, mlp_ratio=8/3)
Infra: 16N / 64 GPU | BSZ=2/gpu, gBSZ=128
Note:  Resubmit of 3374290 (NCCL crash at step 150). Fixed duplicate --activation_fn.
Updated: 2026-03-26 18:06:49 UTC


Job:   3387137 (bench-10tok-82M-GOOG) — fully trained 82M 10tok, step 113775
User:  kangli.s5e
Model: GDN 82M | 10tok | step 113775 (1 full epoch)
Infra: 1N / 4 GPU | N_SEQ=256, N_GEN=500
Task:  LOBbench + return bench
Log:   experiments/exp_P1e_10tok/logs_lobs5/bench_10tok_3387137.out

Updated: 2026-03-27 01:30:00 UTC

Job:   3388351 (O14-remat-164M-16n-bsz4)
User:  kangli.s5e
Model: GDN 164M + SwiGLU + nn.remat(nothing_saveable)
Infra: 16N / 64 GPU | BSZ=4/gpu, gBSZ=256
Note:  Activation checkpointing enabled. BSZ=4 verified on 2N.
Updated: 2026-03-27 02:00:43 UTC


Job:   3388358 (O14-remat-shard-auto-16n)
User:  kangli.s5e
Model: GDN 164M + SwiGLU + remat(nothing_saveable) + shard_autotuning=true
Infra: 16N / 64 GPU | BSZ=4/gpu, gBSZ=256
Note:  Re-enabled shard_autotuning (was false, causing OFI crashes)
Updated: 2026-03-27 02:11:42 UTC


Job:   3388362 (bench-P1a-9tok-step35K)
User:  kangli.s5e
Model: GDN ~151M (d=1024, L=12, B=16) | 9tok encoding
Data:  GOOG Jan 2026 | 1024 sequences | 500 cond + 500 gen
Infra: 1N / 4 GPU | inference + LOBbench scoring
Log:   experiments/exp_P1a_flat_9tok/logs_lobs5/bench_9tok_3388362.out

Job:   3388364+3388365 (P1d-6tok-step33K bench)
User:  kangli.s5e
Model: GDN ~172M (d=1024, L=12, B=16) | 6tok lossless encoding
Data:  GOOG Jan 2026 | 1024 sequences | 500 cond + 500 gen
Infra: 1N / 4 GPU | inference + LOBbench scoring
Log:   lob_pipeline/results_P1d-6tok-step33K/

Job:   3388369 (P1c-5tok-55M-16n-10h)
User:  kangli.s5e
Model: GDN ~?M (d=1024, L=12, B=16) | 5tok MarS BPE encoding
Data:  8 tickers x 4yr (2022-2025)
Infra: 16N / 64 GPU | BSZ=4/gpu, gBSZ=256
Log:   experiments/exp_P1c_cartesian/logs_lobs5/

Job:   3388370 (P1e-10tok-55M-16n-10h)
User:  kangli.s5e
Model: GDN ~151M (d=1024, L=12, B=16) | 10tok encoding
Data:  8 tickers x 4yr (2022-2025)
Infra: 16N / 64 GPU | BSZ=4/gpu, gBSZ=256
Log:   experiments/exp_P1e_10tok/logs_lobs5/

Job:   3388448 (dfm-94M-16n-24h-1epoch)
User:  kangli.s5e
Step:  0/847885  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: GDN BiDFM 94M (d=1024, L=12, B=16, ssm=1024) | 248,719,437 params
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 16N / 64 GPU | BSZ=1/gpu, gBSZ=64
LR:    backbone=5e-5, new=1e-4
Loss:  (pending)
Speed: ~1.79 s/step
Time:  0:00 elapsed  |  ~421h remaining  |  24:00 limit
ETA:   ~24h (first segment, 5.7% of epoch)
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O12_GDN_DFM/logs_lobs5/dfm_3388448_node0.log

Updated: 2026-03-27 08:15:00 UTC

Job:   3388565 (dfm-94M-16n-24h-1epoch-r2)
User:  kangli.s5e
Step:  0/847885  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: GDN BiDFM 94M (d=1024, L=12, B=16, ssm=1024) | 248,719,437 params
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 16N / 64 GPU | BSZ=1/gpu, gBSZ=64 | --exclude=nid010507
LR:    backbone=5e-5, new=1e-4
Loss:  (pending)
Speed: ~1.79 s/step
Time:  0:00 elapsed  |  ~421h remaining  |  24:00 limit
ETA:   ~24h (first segment)
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O12_GDN_DFM/logs_lobs5/dfm_3388565_node0.log

Updated: 2026-03-27 09:00:00 UTC

Job:   3388574 (O14-dots-remat-16n-bsz4)
User:  kangli.s5e
Model: GDN 164M + SwiGLU + remat(dots_saveable) + shard_autotuning=true
Infra: 16N / 64 GPU | BSZ=4/gpu, gBSZ=256

Job:   3388575 (O14-noremat-16n-bsz4)
User:  kangli.s5e
Model: GDN 164M + SwiGLU + NO remat + shard_autotuning=true
Infra: 16N / 64 GPU | BSZ=4/gpu, gBSZ=256
Note:  A/B test: remat on (3388574) vs off (3388575). If no-remat OOMs, dots_saveable wins.
Updated: 2026-03-27 03:08:39 UTC


Job:   3388731 (nsa-ctx4k-16n-24h-resume)
User:  kangli.s5e
Step:  1500/23273  [██░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  6.4%
Model: NSA PyTorch d=1024, L=12, H=16 | ~200M params | flash_attn
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 16N / 64 GPU | BSZ=5/gpu, gBSZ=320 | SP=4
LR:    5e-4 (cosine from step 1500)
Loss:  ~1.12 (at step 1500 checkpoint)
Speed: ~16.0 s/step | 45.6% MFU
Time:  0:00 elapsed  |  ~97h remaining  |  24:00 limit
ETA:   Resume from step 1500 (prev job 3378833 crashed SIGTERM at step 1979)
W&B:   (pending — will create new run)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O2b_rope_ctx4000_sp/logs/nsa_torch_3388731.out
RESTORE: checkpoints_nsa/j3378833/step_1500.pt

Updated: 2026-03-27 06:00:00 UTC

Job:   3389864 (O14-swiglu-164M-16n-24h)
User:  kangli.s5e
Step:  0/233061  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: GDN 164M + SwiGLU (d=1024, L=6, nh=8, hd=128, d_ff=2816, fused gate_up)
       | 164,340,229 params | remat(dots_saveable) + shard_autotuning=true
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 16N / 64 GPU | BSZ=4/gpu, gBSZ=256
LR:    5e-4 (cosine, warmup 2330 steps)
Loss:  (pending)
Speed: ~1.57 s/step
Time:  0:00 elapsed  |  ~101h remaining  |  24:00 limit
ETA:   ~55K steps in 24h (24% epoch). Auto-resume chain needed.
W&B:   (pending)
Log:   experiments/exp_O14_ffn_swiglu/logs_lobs5/training_3389864_node0.log

Updated: 2026-03-27 08:00:00 UTC

Job:   3393701 (cl-55M-16n-12h)
User:  kangli.s5e
Desc:  Closed-loop autoregressive training, 55M (d=1024,L=12), 8 tickers x 4yr, 16N/64GPU, 1 epoch
Infra: 16N / 64 GPU | BSZ=1/gpu, gBSZ=64
LR:    3e-4
Time:  12:00:00 limit | ~9h estimated
W&B:   mars-cl project

Updated: 2026-03-27 11:11:46 UTC


Job:   3417496 (mamba3-muon-8n-26tok-k10)
User:  aramis.s5e
Step:  0/466138  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0.0%
Model: Mamba3 SISO 78.5M (d=1024, L=6, nh=32, hd=64, N=128) | 78,539,423 params
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 8N / 32 GPU | BSZ=4/gpu, gBSZ=128 | Local Steps K=10
LR:    muon_lr=0.01, ssm_lr=5e-4, muon_wd=0.005
Loss:  0.805 (at K10 verify, step 1230)
Speed: ~1.8 it/s (est)
Time:  0:00 elapsed  |  ~24h limit
W&B:   oxford-lob/mamba3
Log:   logs_lobs5/mamba3_3417496.out

Updated: 2026-03-28 01:38:36 UTC


Job:   3417571 (O4s-3type-399M-16n-6h)
User:  kangli.s5e
Desc:  MarS 3-type faithful teacher forcing, 399M (d=1024,L=24,H=16,FFN=4096,dropout=0.1), vocab=49152
Data:  8 tickers x 4yr, lob_preproc_mars_3type (C1+C2+I1+I2 fixes)
Infra: 16N / 64 GPU | BSZ=64/gpu, gBSZ=4096
LR:    3e-4, warmup=1000
Time:  6:00:00 limit
Branch: exp/O4s-MarS-3type-faithful (6b100bea)

Updated: 2026-03-28 01:53:21 UTC


Job:   3418924 (rank-G4-1n)
User:  kangli.s5e
Model: GDN 55M (d=512, L=6, B=8) | probit rank advantage
Data:  GOOG 2022-2025 | test: 2026
Infra: 1N / 4 GPU | GRPO G=4, steps=100, eval@25
LR:    1e-5
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O10c_ic_reward_rank/logs_lobs5/grpo_3418924.out
Updated: 2026-03-28 05:15 UTC

Job:   3418925 (rank-G32-1n)
User:  kangli.s5e
Model: GDN 55M (d=512, L=6, B=8) | probit rank advantage
Data:  GOOG 2022-2025 | test: 2026
Infra: 1N / 4 GPU | GRPO G=32, steps=100, eval@25
LR:    1e-5
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O10c_ic_reward_rank/logs_lobs5/grpo_3418925.out
Updated: 2026-03-28 05:15 UTC

Job:   3418926 (rank-G64-1n)
User:  kangli.s5e
Model: GDN 55M (d=512, L=6, B=8) | probit rank advantage
Data:  GOOG 2022-2025 | test: 2026
Infra: 1N / 4 GPU | GRPO G=64, steps=100, eval@25
LR:    1e-5
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O10c_ic_reward_rank/logs_lobs5/grpo_3418926.out
Updated: 2026-03-28 05:15 UTC

Job:   3418927 (rank-G128-1n)
User:  kangli.s5e
Model: GDN 55M (d=512, L=6, B=8) | probit rank advantage
Data:  GOOG 2022-2025 | test: 2026
Infra: 1N / 4 GPU | GRPO G=128, steps=100, eval@25
LR:    1e-5
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O10c_ic_reward_rank/logs_lobs5/grpo_3418927.out
Updated: 2026-03-28 05:15 UTC

Job:   3418935 (rank-G4-1n) [RETRY: fix d=1024]
User:  kangli.s5e
Model: GDN 55M (d=1024, L=6, B=16) | probit rank advantage
Infra: 1N / 4 GPU | GRPO G=4, steps=100
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O10c_ic_reward_rank/logs_lobs5/grpo_3418935.out
Updated: 2026-03-28 05:59 UTC

Job:   3418936 (rank-G32-1n) [RETRY: fix d=1024]
User:  kangli.s5e
Model: GDN 55M (d=1024, L=6, B=16) | probit rank advantage
Infra: 1N / 4 GPU | GRPO G=32, steps=100
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O10c_ic_reward_rank/logs_lobs5/grpo_3418936.out
Updated: 2026-03-28 05:59 UTC

Job:   3418937 (rank-G64-1n) [RETRY: fix d=1024]
User:  kangli.s5e
Model: GDN 55M (d=1024, L=6, B=16) | probit rank advantage
Infra: 1N / 4 GPU | GRPO G=64, steps=100
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O10c_ic_reward_rank/logs_lobs5/grpo_3418937.out
Updated: 2026-03-28 05:59 UTC

Job:   3418938 (rank-G128-1n) [RETRY: fix d=1024]
User:  kangli.s5e
Model: GDN 55M (d=1024, L=6, B=16) | probit rank advantage
Infra: 1N / 4 GPU | GRPO G=128, steps=100
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O10c_ic_reward_rank/logs_lobs5/grpo_3418938.out
Updated: 2026-03-28 05:59 UTC

Job:   3419060 (O4s-75M-3type-16n-4h)
User:  kangli.s5e
Desc:  MarS 3-type faithful teacher forcing, 75M (d=512,L=12,H=8,FFN=2048,dropout=0.1), vocab=49152
Data:  8 tickers x 4yr, lob_preproc_mars_3type (P0+P1 fixes: tick-snapped mid, raw time_to_open, pre-order book, trans_ratio=9 for C)
Infra: 16N / 64 GPU | BSZ=64/gpu, gBSZ=4096
LR:    3e-4, warmup=1000
Time:  4:00:00 limit
Branch: exp/O4s-MarS-3type-faithful (4d511cd0)

Updated: 2026-03-28 06:51:28 UTC


Job:   3420260 (mars-bench-goog-O4s-75M-3type)
User:  kangli.s5e
Step:  -/- (3-phase pipeline: inference → score → return_bench)
Model: MarS LLaMA 75M 3-type (d=512, L=12, H=8) | 79M params
Data:  GOOG Jan 2026 | 1024 sequences (500 cond + 500 gen)
Infra: 1N / 4 GPU | BSZ=64 | closed-loop (mlib matching engine)
Loss:  4.43 (training epoch 0, step 14330)
Speed: TBD (first bench run)
Time:  pending  |  est ~1.5h  |  4:00 limit
W&B:   https://wandb.ai/zheng-xiong-University%20of%20Oxford/mars-torch/runs/21znjfnb (training)
Log:   /lus/lfs1aip2/projects/s5e/lob_pipeline/.worktrees/mars-bench/results_mars_O4s-75M-3type/logs/bench_GOOG_3420260.out

Updated: 2026-03-28 12:55:00 UTC

Job:   3426963 (ws21-mars-3type-GOOG)
User:  kangli.s5e
Step:  -/- (2-phase: prepare_lobbench → LOBbench WS-21 scoring)
Model: MarS LLaMA 75M 3-type (d=512, L=12, H=8) | 79M params
Data:  GOOG | 1024 sequences from closed-loop inference (job 3420260)
Infra: 1N / 256 CPU (scoring only, no GPU)
Loss:  N/A (scoring job)
Speed: TBD
Time:  pending  |  est ~30min  |  2:00 limit
W&B:   N/A
Log:   /lus/lfs1aip2/projects/s5e/lob_pipeline/.worktrees/mars-bench/results_mars_O4s-75M-3type/logs/ws21_GOOG_3426963.out

Updated: 2026-03-28 13:15:00 UTC

Job:   3432208 (rank-G512-1n)
User:  kangli.s5e
Model: GDN 55M (d=1024, L=6, B=16) | Rank probit advantage | G=512
Infra: 1N / 4 GPU | GRPO G=512, steps=100, eval@25
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O10c_ic_reward_rank/logs_lobs5/grpo_3432208.out
Updated: 2026-03-28 19:10 UTC

Job:   3432209 (rank-G1024-1n)
User:  kangli.s5e
Model: GDN 55M (d=1024, L=6, B=16) | Rank probit advantage | G=1024
Infra: 1N / 4 GPU | GRPO G=1024, steps=100, eval@25
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O10c_ic_reward_rank/logs_lobs5/grpo_3432209.out
Updated: 2026-03-28 19:10 UTC

Job:   3432210 (rank-G2048-1n)
User:  kangli.s5e
Model: GDN 55M (d=1024, L=6, B=16) | Rank probit advantage | G=2048
Infra: 1N / 4 GPU | GRPO G=2048, steps=50, eval@25
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O10c_ic_reward_rank/logs_lobs5/grpo_3432210.out
Updated: 2026-03-28 19:10 UTC

Job:   3432211 (rank-G4096-1n)
User:  kangli.s5e
Model: GDN 55M (d=1024, L=6, B=16) | Rank probit advantage | G=4096
Infra: 1N / 4 GPU | GRPO G=4096, steps=25, eval@25
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_O10c_ic_reward_rank/logs_lobs5/grpo_3432211.out
Updated: 2026-03-28 19:10 UTC

Job:   3437321 (R2-dfm-16n-persample-t)
User:  kangli.s5e
Step:  0/466122  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: DFM-GDN 94M (d=1024, L=6, B=16, bidirectional) | ~94M params
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 16N / 64 GPU | BSZ=2/gpu, gBSZ=128
LR:    backbone 5e-5, new params 1e-4
Loss:  (starting)
Speed: ~1.3 s/step (estimated from previous run)
Time:  0:00 elapsed  |  ~168h remaining  |  24:00 limit
ETA:   24h covers ~66K steps (14% of epoch)
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R2_gdn_dfm/logs_lobs5/training_3437321_node0.log

Updated: 2026-03-29 03:30:00 UTC

Job:   3437906 (R2-dfm-16n-1ep-v3)
User:  kangli.s5e
Step:  0/466122  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: GDN BiDFM 94M (d=1024, L=6, B=16) | 248M DFM params
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 16N / 64 GPU | BSZ=2/gpu, gBSZ=128
LR:    backbone 5e-5, new 1e-4
Loss:  (pending)
Speed: ~1.5 s/step (estimated)
Time:  0:00 elapsed  |  ~194h remaining  |  24:00 limit
ETA:   24h covers ~58K steps (12.4%)
W&B:   (pending)
Log:   exp_R2_gdn_dfm/logs_lobs5/training_3437906_node0.log

Updated: 2026-03-29 06:30:00 UTC

Job:   3443051 (gate-jax-bsz1-2n)
User:  kangli.s5e
Step:  0/300  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: Mamba3 MIMO R=4 (d=1024, L=6, B=16, ssm=1024) | ~86M params
Data:  8 tickers x 4yr (2022-2025) | test: 2026-01
Infra: 2N / 8 GPU | BSZ=1/gpu, gBSZ=8
LR:    muon 0.01
Loss:  (pending)
Speed: (pending)
Time:  0:00 elapsed  |  ~0:30 limit
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1b_Mamba3_MIMO/logs_lobs5/lobs5_3443051.out
Note:  Decision gate #1: Pure JAX (use_triton=False), BSZ=1

Updated: 2026-03-29 20:00:00 UTC

Job:   3443052 (gate-jax-bsz2-2n)
User:  kangli.s5e
Step:  0/300  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: Mamba3 MIMO R=4 (d=1024, L=6, B=16, ssm=1024) | ~86M params
Data:  8 tickers x 4yr (2022-2025) | test: 2026-01
Infra: 2N / 8 GPU | BSZ=2/gpu, gBSZ=16
LR:    muon 0.01
Loss:  (pending)
Speed: (pending)
Time:  0:00 elapsed  |  ~0:30 limit
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1b_Mamba3_MIMO/logs_lobs5/lobs5_3443052.out
Note:  Decision gate #1: Pure JAX (use_triton=False), BSZ=2. KEY TEST: if OOM -> CUDA justified

Updated: 2026-03-29 20:00:00 UTC

Job:   3474477 (mimo-R1g-16n-bench)
User:  kangli.s5e
Model: MIMO d=1024, 6L, R=4, blocks=16 | ~22M params
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 16N / 64 GPU | BSZ=1/gpu, gBSZ=64 | Muon optimizer
Purpose: Speed benchmark (CURTAIL=300, 30min)
Branch: exp/R1g-MIMO-pure-jax (commit c5985d6e, NaN fixes ported)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1g_MIMO_pure_jax/logs_lobs5/lobs5_3474477.out

Updated: 2026-03-30 16:45:00 UTC

Job:   3475115 (mimo-r1g-16n-bench)
User:  kangli.s5e
Step:  0/300  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: MIMO R=4 (d=1024, 6L, B=16, ssm=1024) | ~22M params
Data:  GOOG 2022 (249 days) | test: Jan 2023
Infra: 16N / 64 GPU | BSZ=1/gpu, gBSZ=64 | Muon
LR:    muon_lr=0.01, ssm_lr=5e-4
Loss:  (pending)
Speed: (benchmark - measuring s/step)
Time:  0:00 elapsed  |  ~15min remaining  |  00:30 limit
ETA:   ~15min
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1g_MIMO_pure_jax/logs_lobs5/lobs5_3475115.out

Updated: 2026-03-30 15:45:00 UTC

Job:   3475991 (mimo-r1g-16n-bench-v2)
User:  kangli.s5e
Step:  0/300  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: MIMO R=4 (d=1024, 6L, B=16, ssm=1024) | ~22M params
Data:  GOOG 2022 (249 days) | test: Jan 2023
Infra: 16N / 64 GPU | BSZ=1/gpu, gBSZ=64 | Muon LR=1e-3
LR:    muon_lr=1e-3, ssm_lr=5e-4
Loss:  (pending)
Speed: (benchmark - measuring s/step)
Time:  0:00 elapsed  |  ~15min remaining  |  00:30 limit
ETA:   ~15min
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1g_MIMO_pure_jax/logs_lobs5/lobs5_3475991.out

Updated: 2026-03-30 15:55:00 UTC

Job:   3476164 (mimo-R1g-16n-adamw)
User:  kangli.s5e
Model: MIMO d=1024, 6L, R=4, blocks=16 | ~22M params
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 16N / 64 GPU | BSZ=1/gpu, gBSZ=64 | AdamW optimizer
Purpose: Speed benchmark (CURTAIL=300, 30min) — retry with AdamW after Muon NaN
Branch: exp/R1g-MIMO-pure-jax (commit c5985d6e)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1g_MIMO_pure_jax/logs_lobs5/lobs5_3476164.out

Updated: 2026-03-30 17:00:00 UTC

Job:   3476261 (mimo-R1g-16n-std)
User:  kangli.s5e
Model: MIMO d=1024, 6L, R=4, blocks=16 | ~22M params
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 16N / 64 GPU | BSZ=1/gpu, gBSZ=64 | standard (AdamW)
Purpose: Speed benchmark (CURTAIL=300, 30min)
Branch: exp/R1g-MIMO-pure-jax (commit c5985d6e)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1g_MIMO_pure_jax/logs_lobs5/lobs5_3476261.out

Updated: 2026-03-30 17:10:00 UTC

Job:   3476266 (mimo-r1g-16n-bench-v3)
User:  kangli.s5e
Step:  0/300  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: MIMO R=4 (d=1024, 6L, B=16, ssm=1024) | ~22M params
Data:  GOOG 2022 (249 days) | test: Jan 2023
Infra: 16N / 64 GPU | BSZ=1/gpu, gBSZ=64 | Muon LR=1e-3
LR:    muon_lr=1e-3, ssm_lr=5e-4
Loss:  (pending)
Speed: (benchmark - measuring s/step)
Time:  0:00 elapsed  |  ~15min remaining  |  00:30 limit
ETA:   ~15min
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1g_MIMO_pure_jax/logs_lobs5/lobs5_3476266.out

Updated: 2026-03-30 16:02:00 UTC

Job:   3477653 (mimo-729e-16n-bench)
User:  kangli.s5e
Step:  0/300  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: MIMO R=4 (d=1024, 6L, B=16, ssm=1024) @ commit 729e0f27 (NO NaN fixes)
Data:  GOOG 2022 (249 days) | test: Jan 2023
Infra: 16N / 64 GPU | BSZ=1/gpu, gBSZ=64 | Muon LR=1e-3
LR:    muon_lr=1e-3, ssm_lr=5e-4
Loss:  (pending)
Speed: (benchmark - baseline speed comparison)
Time:  0:00 elapsed  |  ~15min remaining  |  00:30 limit
ETA:   ~15min
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1g_baseline_bench/logs_lobs5/lobs5_3477653.out

Updated: 2026-03-30 16:25:00 UTC

Job:   3478235 (mimo-729e-16n-v2)
User:  kangli.s5e
Step:  0/300  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: MIMO R=4 (d=1024, 6L, B=16, ssm=1024) @ commit 729e0f27 (NO NaN fixes)
Data:  GOOG 2022 (249 days) | test: Jan 2023
Infra: 16N / 64 GPU | BSZ=1/gpu, gBSZ=64 | Muon LR=1e-3 | wandb=OFF
LR:    muon_lr=1e-3, ssm_lr=5e-4
Loss:  (pending)
Speed: (benchmark - baseline speed comparison)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1g_baseline_bench/logs_lobs5/lobs5_3478235.out

Updated: 2026-03-30 16:45:00 UTC

Job:   3478812 (mimo-R1g-16n-bench)
User:  kangli.s5e
Model: MIMO d=1024, 6L, R=4, blocks=16 | ~22M params
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 16N / 64 GPU | BSZ=1/gpu, gBSZ=64 | Muon optimizer
Purpose: Speed benchmark (CURTAIL=300, 30min) -- NaN fix verified on 2N
Branch: exp/R1g-MIMO-pure-jax (commit 49d0cd7e, RMSNorm(N) reverted)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1g_MIMO_pure_jax/logs_lobs5/lobs5_3478812.out

Updated: 2026-03-30 17:30:00 UTC

Job:   3485451 (mimo-729e-8n-bench)
User:  kangli.s5e
Step:  0/300  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: MIMO R=4 (d=1024, 6L, B=16, ssm=1024) @ commit 729e0f27 (NO fixes)
Infra: 8N / 32 GPU | BSZ=1/gpu, gBSZ=32 | Muon LR=1e-3 | wandb=OFF
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1g_baseline_bench/logs_lobs5/lobs5_3485451.out
Updated: 2026-03-30 18:30:00 UTC

Job:   3485452 (mimo-fixed-8n-bench)
User:  kangli.s5e
Step:  0/300  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: MIMO R=4 (d=1024, 6L, B=16, ssm=1024) @ commit c5985d6e (WITH fixes)
Infra: 8N / 32 GPU | BSZ=1/gpu, gBSZ=32 | Muon LR=1e-3 | wandb=OFF
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1g_MIMO_pure_jax/logs_lobs5/lobs5_3485452.out
Updated: 2026-03-30 18:30:00 UTC

Job:   3486415 (mimo-r1g-1n-smoke)
User:  kangli.s5e
Step:  0/50  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: MIMO R=4 (d=1024, 6L, B=16, ssm=1024) | ~94M params
Data:  8 tickers x 4yr (2022-2025) | test: 2026-01
Infra: 1N / 4 GPU | BSZ=1/gpu, gBSZ=4 | Muon LR=0.01 | SENTRY=ON
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1g_MIMO_pure_jax/logs_lobs5/lobs5_3486415.out
Updated: 2026-03-30 19:30:00 UTC

Job:   3486416 (mimo-r1g-8n-prod)
User:  kangli.s5e
Step:  0/1864553  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: MIMO R=4 (d=1024, 6L, B=16, ssm=1024) | ~94M params
Data:  8 tickers x 4yr (2022-2025) | test: 2026-01
Infra: 8N / 32 GPU | BSZ=1/gpu, gBSZ=32 | Muon LR=0.01 | SENTRY=ON
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1g_MIMO_pure_jax/logs_lobs5/lobs5_3486416.out
Updated: 2026-03-30 19:30:00 UTC

Job:   3489956 (mimo-r1g-bsz2-1n) | BSZ sweep
Job:   3489957 (mimo-r1g-bsz3-1n) | BSZ sweep
Job:   3489958 (mimo-r1g-bsz4-1n) | BSZ sweep
User:  kangli.s5e
Model: MIMO R=4 (d=1024, 6L) | SENTRY=ON | CURTAIL=50 | 1N/4GPU
Updated: 2026-03-30 21:30:00 UTC

Job:   3490772 (mimo-r1g-bsz2-k10-1n) | BSZ sweep v2 (K=10 Muon)
Job:   3490773 (mimo-r1g-bsz3-k10-1n) | BSZ sweep v2 (K=10 Muon)
User:  kangli.s5e
Model: MIMO R=4 (d=1024, 6L) | SENTRY=ON | CURTAIL=50 | 1N/4GPU | K=10
Updated: 2026-03-30 22:00:00 UTC

Job:   3491650 (mimo-r1g-8n-prod-v2)
User:  kangli.s5e
Step:  0/1864553  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: MIMO R=4 (d=1024, 6L, B=16, ssm=1024) | ~94M params
Data:  8 tickers x 4yr (2022-2025) | test: 2026-01
Infra: 8N / 32 GPU | BSZ=1/gpu, gBSZ=32 | K=10 | Muon LR=0.01 | SENTRY=ON
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1g_MIMO_pure_jax/logs_lobs5/lobs5_3491650.out
Updated: 2026-03-30 22:30:00 UTC

Job:   3492376 (mimo-r1g-8n-speed)
User:  kangli.s5e
Model: MIMO R=4 (d=1024, 6L, B=16, ssm=1024) | ~94M params
Infra: 8N / 32 GPU | BSZ=1/gpu, gBSZ=32 | K=10 | Muon LR=0.01 | SENTRY=ON
Goal:  Speed benchmark (CURTAIL=300, 30min)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1g_MIMO_pure_jax/logs_lobs5/lobs5_3492376.out
Updated: 2026-03-30 22:45:00 UTC

Job:   3493093 (mimo-r1g-8n-speed2)
User:  kangli.s5e
Model: MIMO R=4 (d=1024, 6L) | SENTRY=ON | CURTAIL=300
Infra: 8N / 32 GPU | BSZ=1/gpu, gBSZ=32 | K=10 | Muon LR=0.01
Fix:   /dev/shm cleanup + exclude nid011191
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1g_MIMO_pure_jax/logs_lobs5/lobs5_3493093.out
Updated: 2026-03-30 23:15:00 UTC

Job:   3494879 (mimo-r1g-8n-speed3)
User:  kangli.s5e
Model: MIMO R=4 (d=1024, 6L) | SENTRY=ON | CURTAIL=300
Infra: 8N / 32 GPU | BSZ=1/gpu, gBSZ=32 | K=10 | Muon LR=0.01
Fix:   _recent_step_time init (fe2e1359)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1g_MIMO_pure_jax/logs_lobs5/lobs5_3494879.out
Updated: 2026-03-31 00:00:00 UTC

Job:   3498485 (r1h-cuda-ffi-smoke-1n)
User:  kangli.s5e
Model: MIMO R=4 (d=1024, 6L) | CUDA FFI fwd+bwd | CURTAIL=50
Infra: 1N / 4 GPU | BSZ=1/gpu, gBSZ=4 | K=10 | Muon LR=0.01
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1h_MIMO_cuda_ffi/logs_lobs5/lobs5_3498485.out
Updated: 2026-03-31 01:25:00 UTC

Job:   3498740 (mimo-r1g-expguard-1n) | Stage 1: smoke test, NO sentry
Job:   3498741 (mimo-r1g-bsz2-nosentury) | Stage 2: BSZ sweep
Job:   3498742 (mimo-r1g-bsz3-nosentury) | Stage 2: BSZ sweep
Job:   3498743 (mimo-r1g-bsz4-nosentury) | Stage 2: BSZ sweep
User:  kangli.s5e
Model: MIMO R=4 (d=1024, 6L) | NO SENTRY | exp() guards (commit eeaf0ab2)
Infra: 1N/4GPU | CURTAIL=50 | Muon LR=0.01 | K=10
Updated: 2026-03-31 02:00:00 UTC

Job:   3498756 (r1h-cuda-ffi-smoke2-1n)
User:  kangli.s5e
Model: MIMO R=4 (d=1024, 6L) | CUDA FFI | CURTAIL=50 | padding fix 424edeb6
Infra: 1N / 4 GPU | BSZ=1/gpu, gBSZ=4 | K=10 | Muon LR=0.01
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1h_MIMO_cuda_ffi/logs_lobs5/lobs5_3498756.out
Updated: 2026-03-31 01:40:00 UTC

Job:   3499217 (mimo-r1g-2n-smoke) | Stage 1: 2N smoke, NO sentry, exp guards
Job:   3499218 (mimo-r1g-2n-bsz2) | Stage 2: BSZ sweep 2N
Job:   3499219 (mimo-r1g-2n-bsz3) | Stage 2: BSZ sweep 2N
Job:   3499220 (mimo-r1g-2n-bsz4) | Stage 2: BSZ sweep 2N
User:  kangli.s5e
Model: MIMO R=4 (d=1024, 6L) | NO SENTRY | exp guards (eeaf0ab2)
Infra: 2N/8GPU | CURTAIL=50 | Muon LR=0.01 | K=10 | hierarchical=True
Updated: 2026-03-31 02:10:00 UTC

Job:   3499384 (mimo-r1g-8n-nosentry)
User:  kangli.s5e
Model: MIMO R=4 (d=1024, 6L) | NO SENTRY | exp guards (eeaf0ab2)
Infra: 8N / 32 GPU | BSZ=1/gpu, gBSZ=32 | K=10 | Muon LR=0.01
Goal:  Speed benchmark (CURTAIL=300, 30min) — compare with sentry version j3494879
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1g_MIMO_pure_jax/logs_lobs5/lobs5_3499384.out
Updated: 2026-03-31 02:20:00 UTC

Job:   3499403 (mimo-r1g-8n-bsz2)
User:  kangli.s5e
Model: MIMO R=4 (d=1024, 6L) | NO SENTRY | exp guards (eeaf0ab2)
Infra: 8N / 32 GPU | BSZ=2/gpu, gBSZ=64 | K=10 | Muon LR=0.01
Goal:  Speed test with max BSZ (BSZ=2 ran 9 steps on 2N, may fit on 8N)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1g_MIMO_pure_jax/logs_lobs5/lobs5_3499403.out
Updated: 2026-03-31 02:30:00 UTC

Job:   3499461 (r1h-cuda-ffi-smoke3-2n)
User:  kangli.s5e
Model: MIMO R=4 (d=1024, 6L) | CUDA FFI | CURTAIL=50 | numerical fix 014eea4b
Infra: 2N / 8 GPU | BSZ=1/gpu, gBSZ=8 | K=10 | Muon LR=0.01
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1h_MIMO_cuda_ffi/logs_lobs5/lobs5_3499461.out
Updated: 2026-03-31 02:35:00 UTC

Job:   3499526 (r1h-hybrid-bwd-smoke-2n)
User:  kangli.s5e
Model: MIMO R=4 (d=1024, 6L) | CUDA fwd + JAX bwd (hybrid) | CURTAIL=50
Infra: 2N / 8 GPU | BSZ=1/gpu, gBSZ=8 | K=10 | Muon LR=0.01
Fix:   16d71687 — replace CUDA bwd with jax.vjp on pure JAX fwd
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1h_MIMO_cuda_ffi/logs_lobs5/lobs5_3499526.out
Updated: 2026-03-31 03:00:00 UTC

Job:   3499998 (r1h-hybrid-smoke5-2n)
User:  kangli.s5e
Model: MIMO R=4 (d=1024, 6L) | CUDA fwd CS=16 + JAX bwd | CURTAIL=50
Infra: 2N / 8 GPU | BSZ=1/gpu, gBSZ=8 | K=10 | Muon LR=0.01
Fix:   17666292 — force CS=16 in fwd (was 64 from mamba3.py)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1h_MIMO_cuda_ffi/logs_lobs5/lobs5_3499998.out
Updated: 2026-03-31 03:15:00 UTC

Job:   3500230 (r2j-triton-2n-smoke) | R2J Triton MIMO smoke test
Job:   3500231 (r2j-triton-2n-bsz2) | R2J BSZ sweep
Job:   3500232 (r2j-triton-2n-bsz3) | R2J BSZ sweep
Job:   3500233 (r2j-triton-2n-bsz4) | R2J BSZ sweep
User:  kangli.s5e
Model: MIMO R=4 (d=1024, 6L) | TRITON fwd + JAX bwd | exp2 guards + base fix
Infra: 2N/8GPU | CURTAIL=50 | Muon LR=0.01 | K=10
Commit: f34bbde4
Updated: 2026-03-31 03:00:00 UTC

Job:   3500269 (r1h-bsz2-sweep-2n)
User:  kangli.s5e
Model: MIMO R=4 (d=1024, 6L) | CUDA fwd + JAX bwd | BSZ sweep
Infra: 2N / 8 GPU | BSZ=2/gpu, gBSZ=16 | K=10 | Muon LR=0.01
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1h_MIMO_cuda_ffi/logs_lobs5/lobs5_3500269.out
Updated: 2026-03-31 03:30:00 UTC

Job:   3500285 (r1h-speed-8n-bsz1)
User:  kangli.s5e
Model: MIMO R=4 (d=1024, 6L) | CUDA fwd + JAX bwd | CURTAIL=300
Infra: 8N / 32 GPU | BSZ=1/gpu, gBSZ=32 | K=10 | Muon LR=0.01
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1h_MIMO_cuda_ffi/logs_lobs5/lobs5_3500285.out
Updated: 2026-03-31 03:45:00 UTC

Job:   3500346 (r1h-purejax-speed-8n)
User:  kangli.s5e
Model: MIMO R=4 (d=1024, 6L) | Pure JAX (no CUDA FFI) | CURTAIL=300
Infra: 8N / 32 GPU | BSZ=1/gpu, gBSZ=32 | K=10 | Muon LR=0.01
Fix:   ae053fea — sentry removal, CUDA FFI disabled, exp guards
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1h_MIMO_cuda_ffi/logs_lobs5/lobs5_3500346.out
Updated: 2026-03-31 04:10:00 UTC

Job:   3500419 (r1h-fullcuda-smoke-2n)
User:  kangli.s5e
Model: MIMO R=4 (d=1024, 6L) | FULL CUDA fwd+bwd | CURTAIL=50
Infra: 2N / 8 GPU | BSZ=1/gpu, gBSZ=8 | K=10 | Muon LR=0.01
Fix:   c6f2bfd0 — dADT + intrachunk dQ/dK + dGamma_diag in bwd_pass2
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1h_MIMO_cuda_ffi/logs_lobs5/lobs5_3500419.out
Updated: 2026-03-31 05:10:00 UTC

Job:   3500451 (r1k-tilelang-2n-smoke) | R1k tilelang DLPack bridge, BSZ=1
Job:   3500452 (r1k-tilelang-2n-bsz2) | R1k BSZ sweep
Job:   3500453 (r1k-tilelang-2n-bsz3) | R1k BSZ sweep
Job:   3500454 (r1k-tilelang-2n-bsz4) | R1k BSZ sweep
User:  kangli.s5e
Model: MIMO R=4 (d=1024, 6L) | tilelang fwd+bwd via DLPack | K=0
Infra: 2N/8GPU | CURTAIL=50
Commit: 6cd5e5cb (exp/R1k-MIMO-tilelang)
Updated: 2026-03-31 04:30:00 UTC

Job:   3500457 (r2j-fulltriton-2n-smoke) | Full Triton fwd+bwd smoke test
User:  kangli.s5e
Model: MIMO R=4 (d=1024, 6L) | FULL TRITON (fwd+bwd) | no sentry
Infra: 2N/8GPU | CURTAIL=50 | Muon LR=0.01 | K=10
Commit: 5306fc12 (Triton bwd wired into custom_vjp)
Updated: 2026-03-31 05:00:00 UTC

Job:   3500513 (rank-G128-300s)
User:  kangli.s5e
Model: GDN 84M (d=1024, L=6, B=16, ssm=1024)
Infra: 1N / 4 GPU | G=128, GRPO_STEPS=300, EVAL_EVERY=25
LR:    1e-5
Time:  ~21h estimated | 24:00 limit
Purpose: Post-training scaling law - extend DA_h249 monotonic trend for G=128
Updated: 2026-03-31 12:00:00 UTC

Job:   3500514 (rank-G32-300s)
User:  kangli.s5e
Model: GDN 84M (d=1024, L=6, B=16, ssm=1024)
Infra: 1N / 4 GPU | G=32, GRPO_STEPS=300, EVAL_EVERY=25
LR:    1e-5
Time:  ~16h estimated | 24:00 limit
Purpose: Post-training scaling law - extend DA_h249 monotonic trend for G=32
Updated: 2026-03-31 12:00:00 UTC

Job:   3500515 (rank-G64-300s)
User:  kangli.s5e
Model: GDN 84M (d=1024, L=6, B=16, ssm=1024)
Infra: 1N / 4 GPU | G=64, GRPO_STEPS=300, EVAL_EVERY=25
LR:    1e-5
Time:  ~17h estimated | 24:00 limit
Purpose: Post-training scaling law - extend DA_h249 monotonic trend for G=64
Updated: 2026-03-31 12:00:00 UTC

---
Job:   3500633 (scaling-G32-300s)
User:  kangli.s5e
Model: GDN 84M (d=1024, L=6, B=16, ssm=1024)
Data:  GOOG 2022 + Jan2023 test | GRPO rollouts
Infra: 1N / 4 GPU | GROUP_SIZE=32, GRPO_STEPS=300, EVAL_EVERY=25
LR:    1e-5
Purpose: Post-training scaling law — extended G=32 run
Log:   /projects/s5e/quant/AlphaTrade/experiments/exp_O10c_ic_reward_rank/logs_lobs5/grpo_3500633_node0.log

Job:   3500634 (scaling-G64-300s)
User:  kangli.s5e
Model: GDN 84M (d=1024, L=6, B=16, ssm=1024)
Infra: 1N / 4 GPU | GROUP_SIZE=64, GRPO_STEPS=300, EVAL_EVERY=25
LR:    1e-5
Purpose: Post-training scaling law — extended G=64 run
Log:   /projects/s5e/quant/AlphaTrade/experiments/exp_O10c_ic_reward_rank/logs_lobs5/grpo_3500634_node0.log

Job:   3500635 (scaling-G128-300s)
User:  kangli.s5e
Model: GDN 84M (d=1024, L=6, B=16, ssm=1024)
Infra: 1N / 4 GPU | GROUP_SIZE=128, GRPO_STEPS=300, EVAL_EVERY=25
LR:    1e-5
Purpose: Post-training scaling law — extended G=128 run
Log:   /projects/s5e/quant/AlphaTrade/experiments/exp_O10c_ic_reward_rank/logs_lobs5/grpo_3500635_node0.log

Job:   3500636 (scaling-G256-200s)
User:  kangli.s5e
Model: GDN 84M (d=1024, L=6, B=16, ssm=1024)
Infra: 1N / 4 GPU | GROUP_SIZE=256, GRPO_STEPS=200, EVAL_EVERY=25
LR:    1e-5
Purpose: Post-training scaling law — new G=256 data point
Log:   /projects/s5e/quant/AlphaTrade/experiments/exp_O10c_ic_reward_rank/logs_lobs5/grpo_3500636_node0.log

Updated: 2026-03-31 06:30:00 UTC

Job:   3544400 (def-G32-all-lr5e7) [DEFINITIVE]
User:  kangli.s5e
Model: GDN 84M (d=1024, L=6, B=16, ssm=1024)
Infra: 1N / 4 GPU | G=32, 500 steps, eval_every=50, n_eval=400
LR:    5e-7 | EMA=0.1, median baseline, grad accumulation
Fix:   P0 gradient accumulation + EMA + median + reduced eval
Updated: 2026-04-01 04:30:00 UTC

Job:   3544401 (def-G32-all-lr3e6) [DEFINITIVE]
User:  kangli.s5e
Same as 3544400 but LR=3e-6 (higher for comparison)
Updated: 2026-04-01 04:30:00 UTC

Job:   3544402 (def-G64-all-lr5e7) [DEFINITIVE]
User:  kangli.s5e
Same as 3544400 but G=64 (more samples per step)
Updated: 2026-04-01 04:30:00 UTC

Job:   3594166 (rsweep-R1-endpoint)
User:  kangli.s5e
Model: GDN 84M | ES G=64, sigma=0.01, rank=4, Adam
Infra: 1N / 4 GPU | 10 steps, eval_every=5, n_eval=256
Reward: ic_endpoint (baseline, h249)
Updated: 2026-04-02 22:30:00 UTC

Job:   3594167 (rsweep-R2-multi)
User:  kangli.s5e
Reward: ic_multi_horizon (weighted h9+h99+h199+h249)
Updated: 2026-04-02 22:30:00 UTC

Job:   3594168 (rsweep-R3-dense)
User:  kangli.s5e
Reward: ic_dense (6 sample ticks per rollout)
Updated: 2026-04-02 22:30:00 UTC

Job:   3594169 (rsweep-R4-spread)
User:  kangli.s5e
Reward: ic_spread (IC + spread divergence penalty)
Updated: 2026-04-02 22:30:00 UTC

Job:   3594170 (rsweep-R5-sharpe)
User:  kangli.s5e
Reward: ic_sharpe (Sharpe-like IC contribution ratio)
Updated: 2026-04-02 22:30:00 UTC

Job:   3603584 (n4096-G32-h99-8N) [HIGH PRIORITY]
User:  kangli.s5e
Model: GDN 84M | ES G=32, sigma=0.01, rank=4, Adam
Infra: 8N / 32 GPU | n_rollouts=4096/dir, direction-sharded (4 dirs/node)
Reward: ic_endpoint @ h99
Key:   First test of n_rollouts=4096 — IC SE ≈ 0.016, SNR > 6.0
       Direction sharding: 8x speedup via parallel dir evaluation
       eval_every=1 to see every step's effect
Updated: 2026-04-03 07:00:00 UTC

Job:   3604374 (n4096-G4-1N) [HIGH PRIORITY]
User:  kangli.s5e
Model: GDN 84M | ES G=4, sigma=0.01, rank=4, Adam
Infra: 1N / 4 GPU | n_rollouts=4096/dir, single node (no multi-node device issues)
Reward: ic_endpoint @ h99 | eval_every=1
Key:   IC SE ≈ 0.016, fitness SNR > 6.0
       G=4 → 4 dirs × 94 min = 94 min/step, 10 steps in ~16h
Updated: 2026-04-03 12:00:00 UTC

Job:   3604377 (n4096-G8-1N)
User:  kangli.s5e
Same but G=8. Expected ~7 steps in 24h.
Updated: 2026-04-03 12:00:00 UTC

Job:   3615812 (n4096-G4-denseSlide)
User:  kangli.s5e
Model: GDN 84M | ES G=4, n_rollouts=4096, chunked
Reward: ic_dense_sliding (ALL (t,t+h) pairs, ~31K points/rollout)
Key:   Combined n4096 noise reduction + dense reward. IC SE ≈ 0.001
Updated: 2026-04-04 02:30:00 UTC

Job:   3615825 (n256-G32-denseSlide)
User:  kangli.s5e
Model: GDN 84M | ES G=32, n_rollouts=256
Reward: ic_dense_sliding | 20 steps, faster (~4h for completion)
Key:   Tests if dense_sliding helps at lower n_rollouts
Updated: 2026-04-04 02:30:00 UTC

Job:   3634979 (eggroll-4n-P256)
User:  aramis.s5e
Config: P=256, E=16, sigma=0.001, rank=1, LR=1e-4
Infra: 4N / 16 GPU | 500 ES steps
Time:  04:00:00 limit
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_S1_rollout/logs_lobs5/eggroll_3634979.out

Updated: 2026-04-05 15:20:28 UTC


Job:   3642158 (eggroll-4n-P256-v3)
User:  aramis.s5e
Config: P=256, E=16, sigma=0.001, rank=1, LR=1e-4
Infra: 4N / 16 GPU | 500 ES steps
Fix:   pickle checkpoint (params.pkl) with error handling + resume support
Time:  04:00:00 limit
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_S1_rollout/logs_lobs5/eggroll_3642158.out

Updated: 2026-04-05 21:01:40 UTC


Job:   3660957 (mamba3-34m-eggroll-4n-P256-s300 LOBbench)
User:  aramis.s5e
Task:  LOBbench inference + scoring on EGGROLL j3642158 step 300
Infra: 1N / 4 GPU
Time:  02:00:00 limit
Results: /lus/lfs1aip2/projects/s5e/lob_pipeline/results_mamba3-34m-eggroll-4n-P256-s300/

Updated: 2026-04-07 20:29:36 UTC


Job:   3675783 (triton-fix2b-2n)
User:  aramis.s5e
Step:  smoke test (CURTAIL_EPOCHS=300)
Model: 75m (d_model=1024, L=12, B=16, ssm=1024)
Data:  GOOG 2022 (default)
Infra: 2N / 8 GPU | BSZ=4/gpu, gBSZ=32 | hierarchical=True
Loss:  TBD (just submitted)
Speed: TBD (just submitted)
Time:  0:00 elapsed | ~30min limit
ETA:   testing Fix 2B autotune prune (commit e84c9a77)
W&B:   TBD
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3/logs_lobs5/training_3675783_node0.log
Notes: Verifies fix for Bug #7 (kernel launch OOM). Drops 10/90 autotune
       configs (num_stages=3 + num_warps=8). If this works, Triton smoke
       on 2N hierarchical is unblocked.

Updated: 2026-04-08 14:30:00 UTC

Job:   3677064 (triton-fix2b-v2)
User:  aramis.s5e
Step:  smoke test (CURTAIL_EPOCHS=300)
Model: 75m (d_model=1024, L=12, B=16, ssm=1024)
Data:  GOOG 2022 (default)
Infra: 2N / 8 GPU | BSZ=4/gpu, gBSZ=32 | hierarchical=True
Loss:  TBD (just submitted)
Speed: TBD (just submitted)
Time:  0:00 elapsed | ~30min limit
ETA:   testing tightened Fix 2B v2 (commit b8c420f7)
W&B:   TBD
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3/logs_lobs5/training_3677064_node0.log
Notes: Fix 2B v2 drops 17/90 configs (chunk-size-aware shmem rules).
       TRITON_PRINT_AUTOTUNING=1 set for diagnostic if it still crashes.
       Job 3675783 (Fix 2B v1) only dropped (3,8) and still OOMed.

Updated: 2026-04-08 14:55:00 UTC

Job:   3679147 (triton-fix2b-v3)
User:  aramis.s5e
Step:  smoke test (CURTAIL_EPOCHS=300)
Model: 75m (d_model=1024, L=12, B=16, ssm=1024)
Data:  GOOG 2022 (default)
Infra: 2N / 8 GPU | BSZ=4/gpu, gBSZ=32 | hierarchical=True
Loss:  TBD (just submitted)
Speed: TBD (just submitted)
Time:  0:00 elapsed | ~30min limit
ETA:   testing Fix 2B v3 (commit 7b7ef8f9): drop all (*,8) warp configs
W&B:   TBD
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3/logs_lobs5/training_3679147_node0.log
Notes: Job 3677064 v2 still OOMed at cs=64. v3 drops all 8-warp configs
       (register pressure hypothesis: 8 warps fully consume GH200 65KB
       register file). Per-kernel: 9->6 configs. Also TRITON_PRINT_AUTOTUNING=1
       forwarded via --export= for diagnostic if v3 still fails.

Updated: 2026-04-08 16:05:00 UTC

Job:   3679147 (triton-fix2b-v3) [DONE — CRASHED]
User:  aramis.s5e
Step:  step 0->2 (autotune trial)
Model: 75m
Infra: 2N / 8 GPU | BSZ=4/gpu | hierarchical=True
Result: gpuLaunchKernel CUDA_ERROR_OUT_OF_MEMORY at 4-6 min, exit 15:0
        Even with 6 most-conservative configs (1,2)..(3,4), launch overflows.
        Hypothesis (8-warp register pressure) at best partial.
        DECISION: stop iterating Python prune. Revert Fix 2B. Plan C++ patch.
Log:   logs_lobs5/training_3679147_node0.log

Updated: 2026-04-08 16:30:00 UTC

Job:   3680665 (triton-debug-2n)
User:  aramis.s5e
Step:  smoke test (CURTAIL_EPOCHS=300)
Model: 75m | Triton ON, no prune (just diagnostic logger)
Infra: 2N / 8 GPU | BSZ=4/gpu, gBSZ=32 | hierarchical=True
Notes: Diagnostic-only run. Goal: identify which kernel + constexpr combo
       OOMs at gpuLaunchKernel. Uses log_kernel_autotune helper from
       commit 1a3451e6. Requires jax_triton_kda kwargs-forwarding patch
       in /home/aramis.s5e/local_packages (local-only).
       Also tests Bug #4 workaround from commit a0b404e0 (eval_step
       apply_fn signature now matches train_step).
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3/logs_lobs5/training_3680665_node0.log

Updated: 2026-04-08 16:55:00 UTC

Job:   3680734 (triton-debug-v2)
User:  aramis.s5e
Step:  smoke test (CURTAIL_EPOCHS=300)
Model: 75m | Triton ON, no prune (fixed diagnostic logger)
Infra: 2N / 8 GPU | BSZ=4/gpu, gBSZ=32 | hierarchical=True
Notes: Resubmit of 3680665 with fixed log_kernel_autotune (commit 60d766c5).
       Previous version read constexprs from kwargs (empty); now reads
       from named_args where they actually live. Also call counter +
       dump-everything pattern. Goal: identify exact kernel + constexprs
       at the LAST log line before gpuLaunchKernel OOM.

Updated: 2026-04-08 17:25:00 UTC

Job:   3682893 (triton-8n-bug4-verify)
User:  aramis.s5e
Step:  smoke test (CURTAIL_EPOCHS=5000 to trigger validate at epoch boundary)
Model: 75m | Triton ON, no prune, diagnostic logger still wired
Infra: 8N / 32 GPU | BSZ=4/gpu, gBSZ=128 | hierarchical=True | NO --contiguous
LR:    default
Loss:  TBD
Speed: expecting ~2.32 it/s (matching j3432283)
Time:  0:00 elapsed | 1:30 limit
ETA:   ~45 min (36 min for 5000 steps + ~5 min compile + init)
W&B:   TBD
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3/logs_lobs5/training_3682893_node0.log
Notes: Production-config smoke test. Goals:
       (1) Confirm Triton still works at 8N with full 90-config autotune
           (post-Fix-2B revert, commit 59764868)
       (2) Verify Bug #4 workaround (commit a0b404e0, eval_step apply_fn
           signature matched to train_step) prevents validate() crash
       (3) Diagnostic logger (commit 60d766c5) runs but is no-prune; should
           log all kernel signatures at autotune time without affecting
           training
       CURTAIL_EPOCHS=5000 chosen to approximate j3432283's crash step
       (original crash was at step 5009 during validate).

Updated: 2026-04-08 17:55:00 UTC

Job:   3682893 (triton-8n-bug4-verify) [DONE — FAILED at allocation]
User:  aramis.s5e
State: FAILED, exit 0:53, elapsed 00:00:02, NO log files produced
Nodes: nid[011116-011118,011130,011161,011167,011169,011174]
Cause: Cluster-level failure (bad node or prologue issue, not our code)
Notes: Resubmit with --exclude=nid[011116-011118,011130,011161,011167,
       011169,011174] in the next session. See tasks/triton_v2_plan.md
       "Step 1" for the retry command.
       8N Triton + Bug #4 fix verification still PENDING.

Updated: 2026-04-08 18:25:00 UTC

────────────────────────────────────────────────────────────
Phase A Mamba3 Pre-Scale Ablation (rope_fraction sweep + baseline)
Submitted: 2026-04-08 22:26:57 UTC
User: aramis.s5e
Branch/worktree: exp/R1-Mamba3 @ experiments/exp_R1_Mamba3
Commit: ae304547 (exp(mamba3): defer LR/WD sweeps to after Phase B, dedupe baseline)

Jobs:  3696344 m3-8m-base-a  (baseline: LR=0.01, WD=0.005, rope=0.5)
       3696345 m3-8m-r000-a  (rope_fraction=0.0, no RoPE)
       3696346 m3-8m-r025-a  (rope_fraction=0.25)
       3696347 m3-8m-r075-a  (rope_fraction=0.75)
       3696348 m3-8m-r100-a  (rope_fraction=1.0, full)
User:  aramis.s5e
Step:  0/pending  (not yet started, queue: Priority)
Model: Mamba3 SISO 8M (d=256, L=6, d_state=128, headdim=64, expand=2) | ~8.1M params
Data:  8 tickers × 2022-2025 (GOOG,AAPL,NVDA,AMZN,META,TSLA,MSFT,AMD) | test: 2026-01
Infra: 4N / 16 GPU | BSZ=10/gpu, gBSZ=160 | Local Steps K=10
LR:    Muon kernel LR=0.01, µP SSM-LR=2.0e-3
Loss:  pending
Speed: pending
Time:  0:00 elapsed  |  ~3:30h limit per job
ETA:   ~3-5h total (depends on queue wait for 5 parallel 4N allocations)
W&B:   https://wandb.ai/oxford-lob/mamba3-phase-a-ablation
Log:   /projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3/logs_lobs5/lobs5_{3696344..3696348}.out

Post-train: LOBbench c250g250 on final checkpoints, primary ranking = WS-21 (GOOG Jan 2026)

Updated: 2026-04-08 22:26:57 UTC

────────────────────────────────────────────────────────────
Phase B Mamba3 Memory Cluster 2D Grid (seq_len × d_state + headdim)
Submitted: 2026-04-09 09:53:23 UTC
User: aramis.s5e
Branch: exp/R1-Mamba3 @ 82bc4b07

Grid (12 jobs): seq_len ∈ {500,1000,2000,4000} × d_state ∈ {64,128,256}
  3703790-3703801 (m3-14m-s{seq}-d{ds}-b)
Headdim (2 jobs): headdim ∈ {32,128} at baseline
  3703802 m3-14m-h032-b, 3703803 m3-14m-h128-b
Model: Mamba3 SISO 14M (d=384, L=6, headdim=64, rope=1.0) | ~14.4M params
Data:  8 tickers × 2022-2025 | test: 2026-01
Infra: 4N / 16 GPU per job | BSZ scales with seq_len | 1.66M tok/step
LR:    Muon kernel LR=0.01, µP SSM-LR=1.33e-3
Time:  5:00:00 limit per job
W&B:   https://wandb.ai/oxford-lob/mamba3-phase-b-memory

Updated: 2026-04-09 09:53:23 UTC

Job:   3714958 (book-abl-nobook-13L)
User:  aramis.s5e
Model: ~8.1M nobook (d=256, L=13, blocks=8, ssm=256) — capacity-matched to book model
Data:  8 tickers × 2022-2025 | test: 2026-01
Infra: 4N / 16 GPU | BSZ=10/gpu, gBSZ=160 | Local Steps K=10
LR:    Muon kernel_lr=0.01, SSM_LR=2e-3, wd=0.005
Purpose: R11 Experiment 4 — capacity-matched no-book ablation (13 layers, ~8.1M vs 4.3M original nobook)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R11_book-ablation/logs_lobs5/

Updated: 2026-04-09 UTC

Job:   3727722 (m3-14m-s4k-d256-remat)
User:  aramis.s5e
Model: 14M (d=384, L=6) | Mamba3 s4000 d_state=256 headdim=64
Infra: 4N / 16 GPU | BSZ=1/gpu, gBSZ=16 | REMAT=1
Time:  smoke test (CURTAIL_EPOCHS=300) | 00:30:00 limit
W&B:   mamba3-phase-b-extended
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3/logs_lobs5/training_3727722_node0.log

Job:   3727723 (m3-14m-s4k-d512-remat)
User:  aramis.s5e
Model: 14M (d=384, L=6) | Mamba3 s4000 d_state=512 headdim=64
Infra: 4N / 16 GPU | BSZ=1/gpu, gBSZ=16 | REMAT=1 REMAT_POLICY=dots
Time:  smoke test | 00:30:00 limit
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3/logs_lobs5/training_3727723_node0.log

Job:   3727724 (m3-14m-s8k-d256-remat)
User:  aramis.s5e
Model: 14M (d=384, L=6) | Mamba3 s8000 d_state=256 headdim=64
Infra: 4N / 16 GPU | BSZ=1/gpu, gBSZ=16 | REMAT=1 REMAT_POLICY=dots
Time:  smoke test | 00:30:00 limit
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3/logs_lobs5/training_3727724_node0.log

Job:   3727725 (m3-14m-s8k-d512-remat)
User:  aramis.s5e
Model: 14M (d=384, L=6) | Mamba3 s8000 d_state=512 headdim=64
Infra: 4N / 16 GPU | BSZ=1/gpu, gBSZ=16 | REMAT=1 REMAT_POLICY=dots
Time:  smoke test | 00:30:00 limit
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3/logs_lobs5/training_3727725_node0.log

Updated: 2026-04-10 09:20:00 UTC

Job:   3728133 (m3-14m-s4k-d256-full)
User:  aramis.s5e
Model: 14M (d=384, L=6) | Mamba3 s4000 d_state=256 headdim=64 rope=1.0
Infra: 4N / 16 GPU | BSZ=1/gpu, gBSZ=16 | REMAT=1 | K=10
Time:  full training | 12:00:00 limit
W&B:   mamba3-phase-b-extended
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3/logs_lobs5/training_3728133_node0.log

Updated: 2026-04-10 09:50:00 UTC

Job:   3735540 (tp4-s4k-d256-smoke)
User:  aramis.s5e
Model: 14M (d=384, L=6) | Mamba3 s4000 d_state=256 headdim=64 rope=1.0
Infra: 1N / 4 GPU | BSZ=1/gpu, gBSZ=1 | REMAT=1 | TP_SIZE=4
Time:  smoke test (CURTAIL_EPOCHS=50) | 00:30:00 limit
W&B:   mamba3-tp-smoke
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3/logs_lobs5/training_3735540_node0.log

Updated: 2026-04-10 13:30:00 UTC

Job:   3745874 (m3-14m-s4k-d512-tp4-16n-6h)
User:  aramis.s5e
Step:  0/~?      [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%  (PENDING)
Model: Mamba3 SISO 14M (d=384, L=6, headdim=64, expand=2, d_state=512)
Data:  8 tickers × 4yr (2022-2025, 26tok) | test: 2026-01
Infra: 16N / 64 GPU | TP=4 intra-node, DP=16 inter-node | BSZ=2/node, gBSZ=32 | REMAT=1
LR:    Muon 0.01, WD 0.005, SSM 1.33e-3
Loss:  (pending)
Speed: (pending — est ~1.2-1.5 s/step based on s4k-d256 Phase 2)
Time:  0m elapsed  |  6h limit  |  NO_AUTO_RESUME=1
ETA:   ~4-5h for 40k steps expected
W&B:   (pending — mamba3-phase-b-extended project)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3/logs_lobs5/lobs5_3745874.out
Note:  First full s4k-d512 training — TP=4+REMAT validated at BSZ=1, BSZ=2 extrapolated to ~52 GiB/GPU (76% of 0.80 budget). No_auto_resume to prevent resume-drift (see exp_R1 s4k-d256 regression).

Updated: 2026-04-11 09:55:00 UTC

Job:   3746367 (book-abl-hier-nobook-8M)
User:  aramis.s5e
Step:  0/~30000 (target for fair comparison with other ablation arms)
Model: Hierarchical NoBook 8.1M (d=256, n_message_layers=2, Dense(256->256), n_fused_layers=11, blocks=8) | capacity matches flat 13L nobook
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 4N / 16 GPU | BSZ=10/gpu, gBSZ=160 | Local Steps K=10
LR:    muon_lr=0.01, ssm_lr=2e-3, wd=0.005
Loss:  (pending)
Speed: (pending)
Time:  0:00 elapsed  |  5:00:00 limit
ETA:   ~2.5h for 30K steps (extrapolating from 5.5M hier-nobook 67K/5h)
W&B:   (pending)
Log:   /projects/s5e/quant/AlphaTrade/experiments/exp_R11_book-ablation/logs_lobs5/lobs5_3746367.out
Note:  Tests whether matching capacity (8.1M) + hierarchical arch (Dense bottleneck) can recover bws=0 performance.
       Previous 5.5M hier nobook gave WS=0.152 at step 29060, matching flat 13L nobook (0.149).
       Both fail to match bws=0 (0.094), suggesting book encoder presence during training is what matters,
       not architecture or capacity alone. This run doubles down on capacity to rule out under-parameterization.

Updated: 2026-04-11 14:00:00 UTC

Job:   3746993 (m3-78m-triton-smoke-2n)
User:  aramis.s5e
Step:  ?/300  (smoke test, curtail)
Model: 75M preset (d=1024, L=6, BSZ=4) | ~78M params
Data:  8 tickers × 4yr (24tok_preproc)
Infra: 2N / 8 GPU | BSZ=4/gpu, gBSZ=32 | Local Steps K=10
LR:    5e-4 SSM, 0.01 Muon
Purpose: Verify Triton kernel works at 78M production scale (R1 bug audit showed all 6 critical bugs fixed in R12)
Expected: If training path is clean, should complete 300 curtail steps without crash
Triton: MAMBA3_USE_TRITON=True, rope_fraction=0.5 (default)
Validation: NO_VALIDATION=True (eval path tested separately later)
Log:   (to be filled)

Updated: 2026-04-11 12:25:00 UTC

Job:   3748136 (m3-merge-t2-sisobwd)
User:  aramis.s5e
Step:  N/A (kernel test suite, not training)
Model: Mamba3 SISO kernel — CUDA FFI backward 11-gradient test
Infra: 1N / 1 GPU | siso_bwd.py test suite
Purpose: Tier 2 gate for R1→R12 merge (pre-commit regression check)
Gate:  must pass all 3 test groups (fwd save, dZ, end-to-end gradients) before merge commit
Time:  queued  |  30min limit
W&B:   N/A (test suite, no training)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R12_m3_cuda/logs/siso_bwd_3748136.out

Updated: 2026-04-11 15:55:00 UTC

Job:   3748277 (m3-merge-t4-base-jax)
User:  aramis.s5e
Purpose: Tier 4 - R12 merge post-commit baseline parity (1N JAX, TP=1, REMAT=0)
Model: Mamba3 75m preset (d=1024, L=6, blocks=16, ssm=1024) | d_state=128
Infra: 1N / 4 GPU | BSZ=4/gpu, gBSZ=16 | Muon LR=0.01
Config: MAMBA3_USE_TRITON=False TP_SIZE=1 REMAT=0 CURTAIL_EPOCHS=300 NO_VALIDATION=True
Time:  queued  |  30min limit
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R12_m3_cuda/logs_lobs5/lobs5_3748277.out
Updated: 2026-04-11 16:15:00 UTC

Job:   3748278 (m3-merge-t5-tp4r1-jax)
User:  aramis.s5e
Purpose: Tier 5 - R12 merge validates TP=4 + REMAT=1 on JAX path (1N)
Model: Mamba3 75m preset (d=1024, L=6, blocks=16, ssm=1024) | d_state=128
Infra: 1N / 4 GPU | BSZ=4/gpu | TP=4 head-parallel intra-node
Config: MAMBA3_USE_TRITON=False TP_SIZE=4 REMAT=1 REMAT_POLICY=dots CURTAIL_EPOCHS=300
Time:  queued  |  30min limit
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R12_m3_cuda/logs_lobs5/lobs5_3748278.out
Updated: 2026-04-11 16:15:00 UTC

Job:   3748279 (m3-merge-t6-tp4r1-tri)
User:  aramis.s5e
Purpose: Tier 6 - R12 merge TARGET combination: Triton + TP=4 + REMAT=1 at d_state=128 (1N)
Model: Mamba3 75m preset (d=1024, L=6, blocks=16, ssm=1024) | d_state=128
Infra: 1N / 4 GPU | BSZ=4/gpu | TP=4 head-parallel intra-node
Config: MAMBA3_USE_TRITON=True TP_SIZE=4 REMAT=1 REMAT_POLICY=dots CURTAIL_EPOCHS=300
Time:  queued  |  30min limit
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R12_m3_cuda/logs_lobs5/lobs5_3748279.out
Updated: 2026-04-11 16:15:00 UTC

Job:   3751338 (m3-14m-s4k-d512-tp4-16n-6h-bsz1)
User:  aramis.s5e
Step:  0/?      [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%  (PENDING)
Model: Mamba3 SISO 14M (d=384, L=6, headdim=64, expand=2, d_state=512)
Data:  8 tickers × 4yr (2022-2025, 26tok) | test: 2026-01
Infra: 16N / 64 GPU | TP=4, DP=16 | BSZ=1/node, gBSZ=16 | REMAT=1
LR:    Muon 0.01, WD 0.005, SSM 1.33e-3
Loss:  (pending)
Speed: (pending)
Time:  0m elapsed  |  6h limit  |  NO_AUTO_RESUME=1
ETA:   ~4-5h for 40k steps expected
W&B:   (pending — mamba3-phase-b-extended project)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3/logs_lobs5/lobs5_3751338.out
Note:  Resubmit of j3745874 which OOM'd at BSZ=2 (152 GiB peak vs 85.5 GiB budget). BSZ=1 validated at ~31 GiB/GPU in Phase 2.

Updated: 2026-04-12 00:05:00 UTC

Job:   3759978 (m3-14m-s4k-d512-tp4-16n-6h-k20)
User:  aramis.s5e
Step:  24530+/?  (resuming from j3751338 step 24530)
Model: Mamba3 SISO 14M (d=384, L=6, d_state=512, headdim=64) + TP=4 + REMAT
Data:  8 tickers × 4yr (2022-2025, 26tok) | test: 2026-01
Infra: 16N / 64 GPU | TP=4, DP=16 | BSZ=1/node, gBSZ=16 | LOCAL_STEPS_K=20 | REMAT=1
LR:    Muon 0.01, WD 0.005, SSM 1.33e-3
Speed: ~1.17 it/s (from j3751338)
Time:  6h limit | NO_AUTO_RESUME=1
Note:  Resume of j3751338. K bumped 10→20 for less noisy convergence. LOBbench on step 24530 pending TP ckpt conversion fix.

Updated: 2026-04-12 09:30:00 UTC

Job:   3760036 (m3-14m-s4k-d512-tp4-32n-6h-fresh)
User:  aramis.s5e
Step:  0/?      [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: Mamba3 SISO 14M (d=384, L=6, d_state=512, headdim=64) + TP=4 + REMAT
Data:  8 tickers × 4yr (2022-2025, 26tok) | test: 2026-01
Infra: 32N / 128 GPU | TP=4, DP=32 | BSZ=1/node, gBSZ=32 | LOCAL_STEPS_K=10 | REMAT=1
LR:    Muon 0.01, WD 0.005, SSM 1.33e-3
Note:  Fresh start (no resume). 2x gradient quality vs j3751338 (gBSZ=32 vs 16). K=10 (K=20 OOM'd at 333 GiB).

Updated: 2026-04-12 10:45:00 UTC

Job:   3837025 (ctx-46m-s2k-bsz32-k10)
User:  aramis.s5e
Model: 46M (d=768, L=6, B=12) | Mamba3 s2k-d256
Infra: 4N / 16 GPU | BSZ=2/gpu, gBSZ=32 | Local Steps K=10
LR:    0.01 (Muon)
Purpose: BSZ/K ablation — reduced per-step noise via bsz=2
W&B:   mamba3-phase-b-extended (pending)
Updated: 2026-04-15 19:30:00 UTC

Job:   3837026 (ctx-46m-s2k-bsz16-k20)
User:  aramis.s5e
Model: 46M (d=768, L=6, B=12) | Mamba3 s2k-d256
Infra: 4N / 16 GPU | BSZ=1/gpu, gBSZ=16 | Local Steps K=20
LR:    0.01 (Muon)
Purpose: BSZ/K ablation — more tokens/sync via higher K
W&B:   mamba3-phase-b-extended (pending)
Updated: 2026-04-15 19:30:00 UTC

Job:   3837027 (ctx-46m-s2k-bsz16-k5)
User:  aramis.s5e
Model: 46M (d=768, L=6, B=12) | Mamba3 s2k-d256
Infra: 4N / 16 GPU | BSZ=1/gpu, gBSZ=16 | Local Steps K=5
LR:    0.01 (Muon)
Purpose: BSZ/K ablation — more frequent syncs, less staleness
W&B:   mamba3-phase-b-extended (pending)
Updated: 2026-04-15 19:30:00 UTC

Job:   3837518 (slaw-s2k-d256-4m)
User:  aramis.s5e
Model: ~4M (d=256, L=6, B=12) | Mamba3 s2k-d256
Infra: 2N / 8 GPU | BSZ=1/gpu, gBSZ=8 | K=10
Purpose: s2k scaling law sweep
Updated: 2026-04-15 20:15:00 UTC

Job:   3837519 (slaw-s2k-d320-8m)
User:  aramis.s5e
Model: ~8M (d=320, L=6, B=12) | Mamba3 s2k-d256
Infra: 2N / 8 GPU | BSZ=1/gpu, gBSZ=8 | K=10
Purpose: s2k scaling law sweep
Updated: 2026-04-15 20:15:00 UTC

Job:   3837520 (slaw-s2k-d512-26m)
User:  aramis.s5e
Model: ~26M (d=512, L=6, B=12) | Mamba3 s2k-d256
Infra: 2N / 8 GPU | BSZ=1/gpu, gBSZ=8 | K=10
Purpose: s2k scaling law sweep
Updated: 2026-04-15 20:15:00 UTC

Job:   3837521 (slaw-s2k-d1024-78m)
User:  aramis.s5e
Model: ~78M (d=1024, L=6, B=12) | Mamba3 s2k-d256
Infra: 4N / 16 GPU | BSZ=1/gpu, gBSZ=16 | K=10
Purpose: s2k scaling law sweep (may need TP=2 if OOM)
Updated: 2026-04-15 20:15:00 UTC

Job:   3848782 (ctx-46m-s2k-cooldown-lr003)
User:  aramis.s5e
Model: 46M (d=768, L=6, B=12) | Mamba3 s2k-d256
Infra: 4N / 16 GPU | BSZ=1/gpu, gBSZ=16 | K=10
LR:    0.01 cosine, CURTAIL_EPOCHS=120000 (LR≈0.003 at step 79740, decaying to ~0.0007)
Resume: from j3817596 step 79740
Purpose: LR cooldown/annealing to establish s2k WS-21 floor
Updated: 2026-04-16 07:15:00 UTC

Job:   3849167 (ctx-46m-s4k-cooldown-lr005)
User:  aramis.s5e
Model: 46M (d=768, L=6, B=12) | Mamba3 s4k-d256 TP=2
Infra: 4N / 16 GPU | BSZ=1/gpu, gBSZ=16 | K=10
LR:    0.01 cosine, CURTAIL_EPOCHS=100000 (LR≈0.005 at step 49890, decaying to ~0.002)
Resume: from j3825699 step 49890
Purpose: LR cooldown for s4k, parallel comparison with s2k cooldown
Updated: 2026-04-16 07:45:00 UTC

Job:   3852068 (ctx-46m-s4k-cooldown-lr005-8n)
User:  aramis.s5e
Model: 46M (d=768, L=6, B=12) | Mamba3 s4k-d256 TP=2
Infra: 8N / 32 GPU | BSZ=1/gpu, gBSZ=16 | K=10
LR:    0.01 cosine, CURTAIL_EPOCHS=100000 (LR≈0.005 at step 49890, decaying to ~0.002)
Resume: from j3825699 step 49890 (8N to avoid resume OOM)
Purpose: LR cooldown for s4k, parallel comparison with s2k cooldown (j3848782)
Updated: 2026-04-16 08:30:00 UTC
Job:   3863502 (nobook-hier-14m-smoke)
User:  aramis.s5e
Model: ~14M (d=384, L=8, B=12) hierarchical no-book | d_state=256
Data:  8 tickers x 2022-2025 | test: 2026-01
Infra: 2N / 8 GPU | BSZ=4/gpu, gBSZ=64 | Local Steps K=10
LR:    Muon 0.01
Exp:   smoke test for hierarchical no-book port
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3/logs_lobs5/lobs5_3863502.out

Updated: 2026-04-16 16:23:38 UTC
Job:   3864098 (curriculum-s4k-d256-from-s2k-s83310)
User:  aramis.s5e
Step:  83310/??? (resume from best s2k cooldown checkpoint)
Model: 46M (d=768, L=6, B=24) | d_state=256, rope=1.0
Data:  8 tickers x 2022-2025, MSG_SEQ_LEN=4000 | test: 2026-01
Infra: 2N / 8 GPU | BSZ=1/gpu, gBSZ=8 | Local Steps K=10
LR:    Muon 0.007, CURTAIL_EPOCHS=25000, cosine decay
Exp:   curriculum s4k from converged s2k (s83310, WS=0.082)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3/logs_lobs5/lobs5_3864098.out

Updated: 2026-04-16 17:12:04 UTC

Job:   3875999 (78m-s500to2k-curr-8n)
User:  aramis.s5e
Model: 78M Mamba3 (d=1024, L=6, B=16, d_state=128, headdim=64, rope=0.5, tok26) | ~78M params
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01 | msg_seq_len=2000
Infra: 8N / 32 GPU | BSZ=4/gpu, gBSZ=128 | Local Steps K=10 | REMAT=1
LR:    Muon 0.005 (fresh cosine after RESTORE_RESET_SCHEDULE), wd=0.005, warmup off
Init:  restored from j3417629_pw8u0edj_3417629 step 46050 (best 78M s500, WS-21=0.044 @ c250g250)
Goal:  test whether curriculum s500→s2k beats 46M s2k floor (0.082) and/or 78M s500 base (0.044)
CURTAIL_EPOCHS=20000 (~20k new steps on s2k)
W&B:   mamba3-curriculum (URL TBD)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3/logs_lobs5/lobs5_3875999.out

Updated: 2026-04-17 09:34:33 UTC


Job:   3876185 (46m-s2k-bookzero-4N)
User:  aramis.s5e
Model: 46M Mamba3 (d=768, L=6, B=12, d_state=256, headdim=64, rope=1.0, tok26) | ~46M params
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01 | msg_seq_len=2000 | BOOK_ABLATION=zero
Infra: 4N / 16 GPU | BSZ=1/gpu, gBSZ=160 | Local Steps K=10 | REMAT=1
LR:    Muon 0.01, wd=0.005, warmup_end=0.01
Init:  from scratch (no restore) | CURTAIL_EPOCHS=50000
Goal:  test whether training with zero-book input (keep architecture, remove info) beats
       46M s2k with-book floor (0.082) at c=250. Cleaner ablation than hierarchical_nobook
       (0.110). Per-feature analysis suggests book-over-dependence is the failure mode —
       if so, this should improve on most non-bid-depth features.
W&B:   mamba3-book-ablation (URL TBD)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3/logs_lobs5/lobs5_3876185.out

Updated: 2026-04-17 09:58:12 UTC


=== Mamba3 SISO Phase A smoke tests (2026-04-18) ===

Job:   3968029 (m3-60m-a-2n-smoke)
User:  aramis.s5e
Model: Mamba3 60M (d=896, L=6) | ~61.3M params
Data:  8 tickers × 2022-2025 | smoke (CURTAIL_EPOCHS=300)
Infra: 2N / 8 GPU | BSZ=4/gpu, gBSZ=32 | Local Steps K=10
LR:    muon=0.01, ssm=5.71e-4
Purpose: verify d=896 config + bsz=4 fits at Phase A scale

Job:   3968030 (m3-90m-a-2n-smoke)
User:  aramis.s5e
Model: Mamba3 90M (d=1088, L=6) | ~88.5M params
Data:  8 tickers × 2022-2025 | smoke (CURTAIL_EPOCHS=300)
Infra: 2N / 8 GPU | BSZ=4/gpu, gBSZ=32 | Local Steps K=10
LR:    muon=0.01, ssm=4.71e-4
Purpose: verify d=1088 config + bsz=4 fits

Job:   3968031 (m3-120m-a-2n-smoke)
User:  aramis.s5e
Model: Mamba3 120M (d=1280, L=6) | ~121.0M params
Data:  8 tickers × 2022-2025 | smoke (CURTAIL_EPOCHS=300)
Infra: 2N / 8 GPU | BSZ=4/gpu, gBSZ=32 | Local Steps K=10
LR:    muon=0.01, ssm=4.00e-4
Purpose: verify d=1280 config + bsz=4 fits

Job:   3968032 (m3-200m-a-2n-smoke)
User:  aramis.s5e
Model: Mamba3 200M (d=1664, L=6) | ~201.8M params
Data:  8 tickers × 2022-2025 | smoke (CURTAIL_EPOCHS=300)
Infra: 2N / 8 GPU | BSZ=4/gpu, gBSZ=32 | Local Steps K=10
LR:    muon=0.01, ssm=3.08e-4
Purpose: verify d=1664 config + bsz=4 fits (largest, highest OOM risk)

Updated: 2026-04-18 (Phase A smoke-test submission)

=== Mamba3 SISO Phase A production runs (2026-04-18) ===

Job:   3971246 (m3-60m-a-8n)
User:  aramis.s5e
Model: Mamba3 60M (d=896, L=6) | 61.3M params
Data:  8 tickers × 2022-2025 | test: Jan 2026
Infra: 8N / 32 GPU | BSZ=4/gpu, gBSZ=128 | Local Steps K=10
LR:    muon=0.01, ssm=5.71e-4
Target: 1.5e19 FLOPs isoflop slice (~20k steps)
Time:  5h limit

Job:   3971247 (m3-90m-a-8n)
User:  aramis.s5e
Model: Mamba3 90M (d=1088, L=6) | 88.5M params
Data:  8 tickers × 2022-2025 | test: Jan 2026
Infra: 8N / 32 GPU | BSZ=4/gpu, gBSZ=128 | Local Steps K=10
LR:    muon=0.01, ssm=4.71e-4
Target: 1.5e19 FLOPs isoflop slice (~14k steps)
Time:  5h limit

Job:   3971248 (m3-120m-a-16n)
User:  aramis.s5e
Model: Mamba3 120M (d=1280, L=6) | 121.0M params
Data:  8 tickers × 2022-2025 | test: Jan 2026
Infra: 16N / 64 GPU | BSZ=2/gpu, gBSZ=128 | Local Steps K=10
LR:    muon=0.01, ssm=4.00e-4
Target: 1.5e19 FLOPs isoflop slice (~10k steps)
Time:  5h limit | NOTE: bsz=2 to fit SSM-state memory at d=1280

Job:   3971249 (m3-200m-a-32n)
User:  aramis.s5e
Model: Mamba3 200M (d=1664, L=6) | 201.8M params
Data:  8 tickers × 2022-2025 | test: Jan 2026
Infra: 32N / 128 GPU | BSZ=1/gpu, gBSZ=128 | Local Steps K=10
LR:    muon=0.01, ssm=3.08e-4
Target: 1.5e19 FLOPs isoflop slice (~6k steps)
Time:  5h limit | NOTE: bsz=1 to fit SSM-state memory at d=1664

Updated: 2026-04-18 (Phase A production submission)

Job:   3976919 (m3-60m-a-8n-retry)
User:  aramis.s5e
Model: Mamba3 SISO 60m-a (d_model=896, n_layers=6, 61.3M params)
Data:  8 tickers × 2022-2025 | test: 2026-01
Infra: 8N / 32 GPU | BSZ=4/gpu, gBSZ=128 | Local Steps K=10
LR:    SSM=5.71e-4, Muon=0.01
Notes: Retry after 3971246 crashed at step 199 (NCCL watchdog timeout).
       Excluded: nid010034, nid010040, nid010100, nid010524, nid010531, nid010628, nid010654, nid010828

Submitted: 2026-04-18 (post-compaction)

Job:   3978268 (m3-200m-a-32n-resume1)
User:  aramis.s5e
Model: Mamba3 SISO 200m-a (d_model=1664, n_layers=6, 201.8M params)
Data:  8 tickers × 2022-2025 | test: 2026-01
Infra: 32N / 128 GPU | BSZ=1/gpu, gBSZ=128 | Local Steps K=10
Notes: Auto-resume from checkpoint after 3971249 (FAILED 2h13m, suspected NCCL deadlock).
       Auto-resume logic in train_full_autoreg.batch resubmitted from latest mid-epoch ckpt.

Updated: 2026-04-19

---
Job:   3999988 (diloco-smoke-2N-14M-s500-K10)
User:  aramis.s5e
Step:  0/~300  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: 14M (d=384, L=6, B=12, ssm=384, dstate=128) | ~14M params
Data:  GOOG,AAPL,NVDA,AMZN,META,TSLA,MSFT,AMD × 2022-2025 | test: 2026-01
Infra: 2N / 8 GPU | BSZ=8/gpu, gBSZ=64 | Local Steps K=10, DiLoCo Nesterov (lr=0.7, β=0.9)
LR:    SSM 1.33e-3, Muon 0.01
Loss:  (pending)
Speed: (pending)
Time:  0:00 elapsed  |  ~30 min limit
ETA:   smoke — CURTAIL_EPOCHS=300 (~300 scan-groups if K=10 → 3000 micro-steps)
W&B:   (pending, project=mamba3-diloco-smoke)
Log:   logs_lobs5/lobs5_3999988.out

Purpose: validate DiLoCo Nesterov code path compiles and runs (2N inter-node pseudo-grad pmean).

Updated: 2026-04-19 10:50:00 UTC

---
Job:   4000032 (diloco-46m-s2k-d128-K10)
User:  aramis.s5e
Step:  0/?  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: 46M (d=768, L=6, B=24, ssm=768, dstate=128, headdim=64, rope=0.5) | paper recipe
Data:  8 tickers × 2022-2025 | test: 2026-01
Infra: 8N / 32 GPU | BSZ=1/gpu, gBSZ=32 | Local Steps K=10, DiLoCo Nesterov (lr=0.7, β=0.9)
LR:    SSM 6.67e-4, Muon 0.01, WD 0.005
Loss:  (pending)
Speed: (pending)
Time:  0:00 elapsed  |  9:00 limit
ETA:   first DiLoCo production run at paper scale
W&B:   (pending, project=mamba3-diloco-46m-s2k)
Log:   logs_lobs5/lobs5_4000032.out

Purpose: test DiLoCo Nesterov at 46M paper scale with s=2000 to see if pseudo-grad + Nesterov rescues context scaling under the "4 seq/node, noisy trajectories" regime. Matched to paper recipe except seq_len=500→2000 and BSZ=4→1 per GPU. Effective outer batch K×gBSZ=320.

Updated: 2026-04-19 10:58:00 UTC

---
Job:   4002044 (diloco-46m-s2k-d128-K10-lr05-b07)
User:  aramis.s5e
Step:  0/?  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: 46M (d=768, L=6, B=24, ssm=768, dstate=128, headdim=64, rope=0.5) | paper recipe
Data:  8 tickers × 2022-2025 | test: 2026-01
Infra: 8N / 32 GPU | BSZ=1/gpu, gBSZ=32 | Local Steps K=10, DiLoCo Nesterov (lr=0.5, β=0.7)
LR:    SSM 6.67e-4, Muon 0.01, WD 0.005
Loss:  (pending)
Speed: (pending)
Time:  0:00 elapsed  |  9:00 limit
ETA:   restart of 4000032 with safer outer hyperparams + outer-step telemetry logged
W&B:   (pending, project=mamba3-diloco-46m-s2k)
Log:   logs_lobs5/lobs5_4002044.out

Purpose: 4000032 had one 4.2 loss spike at step 17790 under paper-default (lr=0.7, β=0.9). Muon inner may amplify per-node direction noise; lr=0.5 + β=0.7 caps worst-case outer step by 3x while still meaningful Nesterov. Telemetry now logs ‖Δ‖, ‖v‖, ‖step‖, ‖update‖ per outer group (commit 97f1a537).

Updated: 2026-04-19 14:20:00 UTC

Job:   4065770 (78m-curr-s2k-cont-19470)
User:  aramis.s5e
Step:  19470 → ~39470 (20k more at s=2k)
Model: 78M Mamba3 (d=1024, L=6, B=16, d_state=128, headdim=64, rope=0.5, tok26)
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01 | msg_seq_len=2000
Infra: 8N / 32 GPU | BSZ=1/gpu, gBSZ=32 | Local Steps K=10 | REMAT=1
LR:    Muon 0.005 (continue schedule, RESET=False), wd=0.005, warmup off
Init:  resume from j3877493_bnvzuy2a_3877493 step 19470 (78M curriculum, prior 19.5k steps at s=2k)
Goal:  Test undertraining hypothesis — do cross-feature conditionals (spread|time, spread|vol) improve
       with 2x more data exposure at s=2k? Next step: bump to s=4k after another 20k steps.
CURTAIL_EPOCHS=20000
W&B:   oxford-lob/mamba3-curriculum/runs/bnvzuy2a (resume)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3/logs_lobs5/lobs5_4065770.out

Updated: 2026-04-20 18:55:00 UTC

Job:   4082152 (78m-curr-s2k-longrun-ch1)
User:  aramis.s5e
Model: 78M Mamba3 (d=1024, L=6, B=16, d_state=128, headdim=64, rope=0.5, tok26)
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01 | msg_seq_len=2000
Infra: 8N / 32 GPU | BSZ=1/gpu, gBSZ=32 | Local Steps K=10 | REMAT=1
LR:    Muon 0.001 (5× below original), SSM 1e-4, wd=0.005, warmup=0.02 (1.6k steps)
       Cosine over 80k steps, RESET_SCHEDULE=True (state.step → 0 fresh)
Init:  resume from j3877493_bnvzuy2a_3877493 step 19470 (curriculum end-of-first-cycle)
Goal:  Test late-training hypothesis — does LOBbench recover past the early peak
       if we give the model another meaningful training cycle at lower LR?
       Chunk 1 of 3-4 needed to complete 80k steps.
CURTAIL_EPOCHS=80000
W&B:   oxford-lob/mamba3-curriculum/runs/bnvzuy2a (continue)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3/logs_lobs5/lobs5_4082152.out

Updated: 2026-04-20 15:15:00 UTC

Job:   4078893 (bench_ext-78m-curr-s19470-c250g250)
User:  aramis.s5e
Type:  LOBbench extended scoring (uncond + cond + div + context + time_lagged)
Model: 78M Mamba3 curriculum @ step 19470 (ckpt j3877493_bnvzuy2a_3877493)
Infra: 1N | c=250, g=250, n=3136 fixed-index GOOG Jan 2026
Goal:  Baseline for longrun continuation — measure curriculum's cross-feature conditionals
       at its current endpoint to see if extended training moves them.
Results (partial, uncond+cond done; div+context pending):
  Uncond WS-21:          0.0595  [.059, .069]
  spread|time cond:      0.2586  [.252, .272]   ← 31% worse than 78M paper (0.198)
  spread|volatility:     0.1966  [.192, .215]   ← 17% worse than 78M paper (0.168)
  ask_volume|spread:     0.0926  [.088, .102]   ← 19% worse than 78M paper (0.078)
Interpretation: curriculum's +19.5k steps at s=2k (2.7× total tokens) made the model
  WORSE at c=250 g=250 eval, not better. Kills pure undertraining hypothesis.
  Consistent with s=2k specialization trading short-horizon quality for long-horizon.
Log:   /lus/lfs1aip2/projects/s5e/lob_pipeline/logs/integrated_ext-78m-curr-s19470-c250g250-n3136_GOOG_4078893.out

Updated: 2026-04-20 15:53:00 UTC

Job:   4092111 (book-depth-profile)
User:  kangli.s5e
Step:  N/A (one-shot profiling)
Model: N/A (data profiling, not training)
Data:  8 tickers (GOOG,AAPL,NVDA,AMZN,META,TSLA,MSFT,AMD) × 2022-2025, L10 orderbook_10_proc.npy
Infra: 1N / 1 GPU (requested for scheduling) | 64 CPU | 30min
LR:    N/A
Loss:  N/A
Speed: N/A
Time:  0:00 elapsed  |  ~30min remaining  |  30:00 limit
ETA:   ~10-15 min (I/O bound: ~384 files × 30MB)
W&B:   N/A (no wandb)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/tasks/book_depth_profiling/logs/profile_4092111.out
Task:  book_depth_profiling (price-rank + tick-index cumsum CDF)

Updated: 2026-04-20 14:35:00 UTC


Job:   4093982 (dugast-q1-8tk-1n-30m)
User:  kangli.s5e
Task:  Dugast 2026 paper replication — Option A same-day orders
Model: OLS regression log(Delay_µs) ~ log(total_depth) + controls + Stock FE + Day × 30min Block FE
Data:  8 tickers (AAPL GOOG MSFT META AMZN NVDA AMD TSLA) × Q1 2025 (2025-01-02 → 2025-03-31, ~62 days)
Infra: 1N / 1 GPU / 64 CPU
Time:  0:00 elapsed  |  ~15 min expected  |  00:30 limit
Target: β ∈ [0.1, 0.3] to match paper's [0.133, 0.169]
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/tasks/verify_market_depth_execution_delays/empirical/logs/optionA_4093982.out

Updated: 2026-04-20 $(date -u +%H:%M:%S) UTC

Job:   4094065 (mamba3-79M-mimo-2n-smoke)
User:  kangli.s5e
Step:  0/300  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: Mamba3 MIMO R=4 (d=1024, L=6, blocks=16, ssm=1024) | ~79-82M params
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 2N / 8 GPU | BSZ=1/gpu, gBSZ=8 | Local Steps K=10
LR:    muon_lr=0.01, ssm_lr=5e-4, wd=0.005
Loss:  (pending)
Speed: ~unknown (smoke test)
Time:  0:00 elapsed  |  ~30min limit
ETA:   <15min expected (based on j3604382)
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1g_MIMO_pure_jax/logs_lobs5/lobs5_4094065.out
Purpose: Verify MIMO NaN-free at current HEAD (e1f27cb1) — baseline for full 16N training

Updated: 2026-04-20 $(date -u +%H:%M:%S) UTC

Job:   4094297 (book-depth-profile-v2)
User:  kangli.s5e
Step:  N/A (one-shot profiling, resubmit after j4092111 timeout)
Fix:   v1 hung in mmap+random-access on Lustre + buffered stdout (no progress visible)
       v2: full np.load (no mmap), STRIDE=20 sampling, python3 -u, per-file log
Data:  8 tickers × 2022-2025, ~384 files × ~90K snaps/file = ~34.5M snaps total
Infra: 1N / 1 GPU / 64 CPU | 60min limit
Time:  0:00 elapsed  |  ~60min remaining  |  60:00 limit
ETA:   ~10-20 min (I/O bound, ~30MB/file × 384 = ~11GB)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/tasks/book_depth_profiling/logs/profile_4094297.out

Updated: 2026-04-20 15:10:00 UTC

Job:   4094323 (book-depth-profile-v3)
User:  kangli.s5e
Step:  N/A (one-shot profiling)
Fix:   v2 (j4094297) crashed at 3s: NameError ROWS_PER_FILE (rename-miss)
       v3: fixed line 280 and 302 to use STRIDE instead
Data:  8 tickers × 2022-2025, ~384 files × ~90K snaps/file
Infra: 1N / 1 GPU / 64 CPU | 60min limit
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/tasks/book_depth_profiling/logs/profile_4094323.out
Monitor: bg ID b6km1pxkd

Updated: 2026-04-20 15:32:00 UTC

Job:   4102752 (78m-curr-s2to4k-ch1)
User:  aramis.s5e
Type:  Curriculum extension s=2k → s=4k
Model: 78M Mamba3 (d=1024, L=6, B=16, d_state=128, headdim=64, rope=0.5, tok26)
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01 | msg_seq_len=4000 (NEW)
Infra: 8N / 32 GPU | BSZ=1/gpu, gBSZ=32 | Local Steps K=10 | REMAT=1
LR:    Muon 0.001 (5× below original), SSM 1e-4, wd=0.005, warmup=0.02 (1.6k steps)
       Cosine over 80k steps, RESET_SCHEDULE=True (state.step → 0 fresh)
Init:  resume from j4082152 final ckpt (longrun chunk 1, expected ~step 38k at end)
Deps:  afterany:4082152
Goal:  Test whether longer-context training (s=4k) extracts structural information
       that s=2k saturates. Natural extension of curriculum (s500→s2k→s4k).
       Wandb starts a fresh run since this is a new phase (no WANDB_RUN_ID set).
CURTAIL_EPOCHS=80000
W&B:   oxford-lob/mamba3-curriculum (new run, name TBD = "j4102752")
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3/logs_lobs5/lobs5_4102752.out

Updated: 2026-04-20 18:00:00 UTC

Job:   4105625 (mamba3-79M-mimo-16n-24h)
User:  kangli.s5e
Step:  0/~172000  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: Mamba3 MIMO R=4 (d=1024, L=6, blocks=16, ssm=1024) | ~79-82M params
Data:  8 tickers (GOOG,AAPL,NVDA,AMZN,META,TSLA,MSFT,AMD) × 4yr (2022-2025) | test: 2026-01
       data_root: /lus/lfs1aip2/projects/s5e/lob_preproc_26tok
Infra: 16N / 64 GPU | BSZ=1/gpu, gBSZ=64 | Local Steps K=10 (effective 640)
LR:    muon_lr=0.01, ssm_lr=5e-4, wd=0.005
Loss:  (pending)
Speed: ~2.7 it/s @ 2N smoke (16N 大致同量级)
Time:  0:00 elapsed  |  24h limit
ETA:   ~1-2 epochs (116k steps/epoch @ gBSZ=64)
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1g_MIMO_pure_jax/logs_lobs5/lobs5_4105625.out
Purpose: MIMO 79M production training, compare vs SISO 78M anchor pw8u0edj->ra5i8nkd
Worktree: exp_R1g_MIMO_pure_jax (HEAD e1f27cb1)

Updated: 2026-04-20 UTC

Job:   4105645 (78m-adamw-probe-s2k-3h)
User:  aramis.s5e
Model: 78M Mamba3 (d=1024, L=6, B=16, d_state=128, headdim=64, rope=0.5, tok26)
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01 | msg_seq_len=2000
Infra: 8N / 32 GPU | BSZ=1/gpu, gBSZ=32 | Local Steps K=10 | REMAT=1
LR:    AdamW 1e-4 (5× reduced, matches Muon longrun), wd=0.005 (match Muon regime)
       Cosine over 80k steps (matches longrun schedule), warmup=0.02, RESET_SCHEDULE=True
Init:  resume from j4082152 step 24880 (mid-longrun Muon ckpt)
Goal:  Test optimizer-only variable — does AdamW's diagonal preconditioning beat
       Muon's spectral at s=2k continuation? Head-to-head vs Muon longrun endpoint (~step 38k).
CURTAIL_EPOCHS=12000 (~3h wallclock at 8N)
W&B:   (fresh run, TBD)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3/logs_lobs5/lobs5_4105645.out

Updated: 2026-04-20 19:15:00 UTC

Job:   4105646 (78m-adamw-curr-s2k-20k)
User:  aramis.s5e
Model: 78M Mamba3 (d=1024, L=6, B=16, d_state=128, headdim=64, rope=0.5, tok26)
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01 | msg_seq_len=2000
Infra: 8N / 32 GPU | BSZ=1/gpu, gBSZ=32 | Local Steps K=10 | REMAT=1
LR:    AdamW 5e-4 (fresh curriculum, matches Muon Curriculum 1), wd=0.005
       Cosine over 20k steps, warmup=0.02, RESET_SCHEDULE=True
Init:  resume from j3417629 step 46050 (78M Muon-best s=500 pretrain — same seed as Curriculum 1)
Goal:  Parallel AdamW curriculum track. Direct comparison to j3877493 Muon Curriculum 1 
       (same pretrain ckpt, same step count, same LR, swap optimizer only).
CURTAIL_EPOCHS=20000 (matches Curriculum 1 duration, ~5h wallclock)
W&B:   (fresh run, TBD)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3/logs_lobs5/lobs5_4105646.out

Updated: 2026-04-20 19:15:00 UTC


Job:   4105682 (dugast-q1-8tk-spec1234-1n-30m)
User:  kangli.s5e
Task:  Dugast 2026 Table 2 full 4-column replication (augmented controls)
Model: OLS 4 specs; adds bid_ask_spread, is_buy, log_size, log_ex_size, log_vol_30; Day×Block FE
Data:  8 tickers (AAPL GOOG MSFT META AMZN NVDA AMD TSLA) × Q1 2025 (60 days)
Infra: 1N / 1 GPU / 64 CPU
Time:  0:00 elapsed  |  ~20 min expected  |  00:30 limit
Target: col(2) ~0.169, col(3) ~0.153, col(4) ~0.156 (paper Table 2)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/tasks/verify_market_depth_execution_delays/empirical/logs/optionA_4105682.out

Updated: 2026-04-20 (auto)

Job:   4105686 (78m-adamw-curr-s2k-20k-wd05) [resubmit of 4105646]
User:  aramis.s5e
Model: 78M Mamba3 (d=1024, L=6, B=16, d_state=128, headdim=64, rope=0.5, tok26)
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01 | msg_seq_len=2000
Infra: 8N / 32 GPU | BSZ=1/gpu, gBSZ=32 | Local Steps K=10 | REMAT=1
LR:    AdamW 5e-4 (matches Muon Curriculum 1's SSM LR), wd=0.05 (AdamW native, matches 46M signal regime)
       Cosine over 20k steps, warmup=0.02, RESET_SCHEDULE=True
Init:  resume from j3417629 step 46050 (78M Muon-best s=500 pretrain — same seed as Curriculum 1)
Goal:  Pure AdamW curriculum reproducing the regime where 46M AdamW won.
       Direct comparison to j3877493 Muon Curriculum 1 (same pretrain, steps, LR; AdamW + WD=0.05).
       Note: WD=0.05 is 10× higher than Muon's 0.005 — risk of fast weight decay on Muon-trained ckpt;
       this matches AdamW's native regime (where the signal came from), not a controlled WD comparison.
CURTAIL_EPOCHS=20000 (matches Curriculum 1 duration, ~5h wallclock)
W&B:   (fresh run, TBD)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3/logs_lobs5/lobs5_4105686.out

Updated: 2026-04-20 19:30:00 UTC

Job:   4107419 (78m-adamw-curr-s2k-20k-wd05) [resubmit of 4105686 — TOKEN_MODE fix]
User:  aramis.s5e
Model: 78M Mamba3 (d=1024, L=6, B=16, d_state=128, headdim=64, rope=0.5, tok26)
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01 | msg_seq_len=2000
Infra: 8N / 32 GPU | BSZ=1/gpu, gBSZ=32 | Local Steps K=10 | REMAT=1
LR:    AdamW 5e-4 (matches Muon Curriculum 1's SSM LR), wd=0.05 (AdamW native)
       Cosine over 20k steps, warmup=0.02, RESET_SCHEDULE=True
Init:  resume from j3417629 step 46050 (78M Muon-best s=500 pretrain)
Bug:   prev attempt 4105686 crashed at 15s — TOKEN_MODE=26 (no "tok" suffix) selected
       wrong DATA_ROOT (/lob_preproc instead of /lob_preproc_26tok), dataloader EOF.
       Fixed by setting TOKEN_MODE=26tok explicitly.
CURTAIL_EPOCHS=20000 (~5h wallclock)
W&B:   (fresh run, TBD)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3/logs_lobs5/lobs5_4107419.out

Updated: 2026-04-20 21:55:00 UTC

Job:   4108541 (tf-8m-2n-smoke)
User:  aramis.s5e
Step:  — / ~300 (smoke, CURTAIL=300)
Model: Transformer 8M (d=256, L=6, heads=4, d_ff=1024) | ~6M params
Data:  8 tickers × 2022-2025 | test: 2026-01 | 26tok
Infra: 2N / 8 GPU | BSZ=10/gpu, gBSZ=80 | Local Steps K=10
LR:    AdamW 2e-3, Muon 0.01, WD 0.005
Loss:  pending
Speed: pending
Time:  pending  |  45min limit
ETA:   smoke; verifies 26tok + Muon + flash on Transformer
W&B:   https://wandb.ai/oxford-lob/transformer-scaling-law (project created on first submit)
Log:   /projects/s5e/quant/AlphaTrade/experiments/exp_O8_self_attention/scaling_logs/lobs5_4108541.out

Updated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")

Job:   4108704 (tf-78m-2n-smoke)
User:  aramis.s5e
Step:  — / ~300 (smoke, CURTAIL=300)
Model: Transformer 78M (d=1024, L=6, heads=16, d_ff=4096)
Data:  8 tickers × 2022-2025 | test: 2026-01 | 26tok
Infra: 2N / 8 GPU | BSZ=8/gpu, gBSZ=64 (smoke clamp) | Local Steps K=10
LR:    AdamW 5e-4, Muon 0.01, WD 0.005
Loss:  pending
Speed: pending
Time:  pending  |  45min limit
ETA:   smoke; validates BSZ=8 memory at d=1024 before full sweep
W&B:   pending
Log:   /projects/s5e/quant/AlphaTrade/experiments/exp_O8_self_attention/scaling_logs/lobs5_4108704.out

Updated: 2026-04-20 23:19:57 UTC

## Transformer Scaling-Law Production Sweep (cross-arch Priority A)
Submitted: 2026-04-20 23:43:08 UTC by aramis.s5e

Job:   4108841 (tf-8m-2n) | 4108842 (tf-14m-2n) | 4108843 (tf-23m-4n)
Job:   4108844 (tf-34m-4n) | 4108845 (tf-46m-4n) | 4108846 (tf-78m-4n)
User:  aramis.s5e
Model: Transformer SISO (d=256..1024, L=6, heads=d/64, d_ff=4d, flash bf16)
Data:  8 tickers × 2022-2025 | test: 2026-01 | 26tok | ctx=500 msgs = 13000 tokens
Infra: 2N (8m,14m) / 4N (23m+) × 4 GPU | BSZ=16(8m,14m)/BSZ=8(23m+) per GPU | gBSZ=128 | K=10 DiLoCo
LR:    AdamW µP 5e-4 × (1024/d), Muon kernel LR=0.01, WD=0.005
ETA:   5h walltime budget each
Total: ~100 node-hr (half of Mamba3 sweep due to Transformer's lower per-GPU memory)
W&B:   https://wandb.ai/oxford-lob/transformer-scaling-law
Logs:  /projects/s5e/quant/AlphaTrade/experiments/exp_O8_self_attention/scaling_logs/lobs5_{4108841..4108846}.out

Updated: 2026-04-20 23:43:08 UTC

## Transformer Sweep COMPLETED (all 6 sizes)
Finished: 2026-04-21 07:21:38 UTC by aramis.s5e

| JobID | Size | FinalStep | Ckpts | Speed | W&B |
|---|---|---|---|---|---|
| 4108841 | 8m  | 53098 | 20 | 4.16 it/s | https://wandb.ai/oxford-lob/transformer-scaling-law/runs/y6qya8s3 |
| 4108842 | 14m | 40312 | 20 | 2.11 it/s | https://wandb.ai/oxford-lob/transformer-scaling-law/runs/7mqajiyb |
| 4108843 | 23m | 60309 | 20 | 4.49 it/s | https://wandb.ai/oxford-lob/transformer-scaling-law/runs/ypo60zdm |
| 4108844 | 34m | 49789 | 20 | 3.72 it/s | https://wandb.ai/oxford-lob/transformer-scaling-law/runs/u8na1hup |
| 4108845 | 46m | 42079 | 20 | 3.17 it/s | https://wandb.ai/oxford-lob/transformer-scaling-law/runs/izainmab |
| 4108846 | 78m | 31769 | 20 | 2.43 it/s | https://wandb.ai/oxford-lob/transformer-scaling-law/runs/97o4ua84 |

Tokens seen: 8m ≈ 88B, 78m ≈ 53B (gBSZ=128 × 13000 tokens/seq)
Next: submit test CE evaluation on Jan 2026 × 8 tickers for each final ckpt + multi-ckpt D-curve

Updated: 2026-04-21 07:21:38 UTC

Job:   4128785 (lobbench-tf78m-c250g250)
User:  aramis.s5e
Model: Transformer 78M final ckpt (step 31320, d=1024, L=6, heads=16) | 26tok
Task:  LOBbench inference + scoring, GOOG 3136 HF-matched sequences
Config: cond=250, gen=250, --skip_extended (uncond + div only)
Infra: 1N / 4 GPU, walltime 1h
Ckpt:  /projects/s5e/quant/AlphaTrade/experiments/exp_O8_self_attention/checkpoints/j4108846_97o4ua84_4108846
Results: /lus/lfs1aip2/projects/s5e/lob_pipeline/results_tf-78m-s31320/

Updated: 2026-04-21 09:04:14 UTC

Job:   4128806 (lobbench-tf78m-v2) — 4128785 failed, model_code_dir was main LOBS5 which doesn't support transformer
User:  aramis.s5e
Model: Transformer 78M step 31320
Change: --lobs5_dir /projects/s5e/quant/AlphaTrade/experiments/exp_O8_self_attention (uses O8 run_inference.py + transformer-aware model code)
Config: cond=250, gen=250, token_mode=26, --skip_extended
Infra: 1N / 4 GPU, walltime 1h
Results: /lus/lfs1aip2/projects/s5e/lob_pipeline/results_tf-78m-s31320-v2/

Updated: 2026-04-21 09:11:23 UTC

Job:   4128839 (lobbench-tf78m-v3) — 4128806 OOM at batch=64 (62GB KV cache allocation)
User:  aramis.s5e
Root cause: TransformerBlock.max_cache_len=25000 × L=6 × heads=16 × head_dim=64 × batch=64 × 2(KV) × 2B = 157 GB. SSMs don't have KV cache so default batch=64 was SSM-calibrated.
Fix:   --batch_size 4 (targets ~10 GB KV cache)
Config: cond=250, gen=250, token_mode=26, batch=4, --skip_extended, --lobs5_dir O8
Infra: 1N / 4 GPU, walltime 1.5h
Results: /lus/lfs1aip2/projects/s5e/lob_pipeline/results_tf-78m-s31320-v3/

Updated: 2026-04-21 09:17:28 UTC

## Transformer Test-CE Evaluation Sweep (Priority A finish)
Submitted: 2026-04-21 09:19:34 UTC by aramis.s5e

| Job | Size | Ckpt dir | Output CSV |
|---|---|---|---|
| 4128850 | tf-78m | j4108846_97o4ua84_4108846 | test_ce_tf78m.csv |
| 4128852 | tf-8m  | j4108841_y6qya8s3_4108841 | test_ce_tf-8m.csv |
| 4128853 | tf-14m | j4108842_7mqajiyb_4108842 | test_ce_tf-14m.csv |
| 4128854 | tf-23m | j4108843_ypo60zdm_4108843 | test_ce_tf-23m.csv |
| 4128855 | tf-34m | j4108844_u8na1hup_4108844 | test_ce_tf-34m.csv |
| 4128856 | tf-46m | j4108845_izainmab_4108845 | test_ce_tf-46m.csv |

Config: CURTAIL_EPOCH=100, MICRO_BSZ=16, 8h walltime, 1N × 4 GPU each
Eval: 20 ckpts × 8 tickers = 160 pairs per job
Output: /projects/s5e/quant/AlphaTrade/experiments/exp_O8_self_attention/test_ce_tf-*.csv

Updated: 2026-04-21 09:19:34 UTC

## Transformer FLOP measurement via dmon (direct HMMA util)
Submitted: 2026-04-21 10:07:49 UTC by aramis.s5e

| Job | Size | Output |
|---|---|---|
| 4130131 | 8M  | gpm_results/gpm_tf-8M_j4130131.* |
| 4130132 | 14M | gpm_results/gpm_tf-14M_j4130132.* |
| 4130133 | 23M | gpm_results/gpm_tf-23M_j4130133.* |
| 4130134 | 34M | gpm_results/gpm_tf-34M_j4130134.* |
| 4130135 | 46M | gpm_results/gpm_tf-46M_j4130135.* |
| 4130137 | 78M | gpm_results/gpm_tf-78M_j4130137.* |

Config: 300 profile steps, 3 warmup, BSZ=1, 1 GPU, 15min walltime
Measures HMMA tensor core utilization + step time. Computes:
  total_tflops_per_s = hmma_pct × 989 + fp32_pct × 67 + fp16_pct × 989 (GH200 peaks)
  flops_per_step = total_tflops_per_s × step_time_s
  This is the DIRECT ground-truth for the scaling-law FLOPs axis (Chinchilla/Kaplan conventions use the theoretical count; dmon captures achieved including kernel efficiency).

Updated: 2026-04-21 10:07:49 UTC

## Transformer 78M + RoPE retrain (LOBbench fairness)
Submitted: 2026-04-21 10:18:45 UTC by aramis.s5e

Job:   4130205 (tf-78m-rope-4n)
Motivation: Mamba3 uses rope_fraction=0.5; prior Transformer 78M used sinusoidal PE.
           Not fair cross-arch comparison. User flag: "RoPE won't affect CE in our
           case 90% sure", so retrain mainly to unlock LOBbench re-eval.
Model: Transformer 78M + RoPE (d=1024, L=6, heads=16, d_ff=4096, rope_base=10000)
Data:  8 tickers × 2022-2025, 26tok | test: 2026-01
Infra: 4N / 16 GPU | BSZ=8/gpu, gBSZ=128 | Local Steps K=10, walltime 5h
LR:    AdamW µP 5e-4, Muon kernel LR=0.01, WD=0.005
W&B:   https://wandb.ai/oxford-lob/transformer-scaling-law (pending)

Code changes:
  s5/transformer.py: + TransformerBlockRoPE (new class), + _rope_freqs/_rope_rotate helpers
  lob/init_train.py: + use_rope, rope_base plumbing to init_TransformerBlock
  run_train.py: + --use_rope, --rope_base CLI flags
  scaling_node_wrapper.sh: + USE_ROPE, ROPE_BASE env var emission

Updated: 2026-04-21 10:18:45 UTC

## Transformer 78M + RoPE retrain (resubmit after odd-head_dim crash)
Submitted: 2026-04-21 12:48 UTC by aramis.s5e

Job:   4132064 (tf-78m-rope-4n)
Prior: 4130205 FAILED at 1:42 — book pre-layer H=d_book=503, n_heads=1 → head_dim=503 hit
       assert in _rope_freqs. Fixed s5/transformer.py: odd head_dim falls back to
       identity (cos=1, sin=0); main attention head_dim=64 rotates normally.
Also:  NO_AUTO_RESUME=1 to prevent restoring non-RoPE 78M checkpoint into RoPE model.
Model: Transformer 78M + RoPE (d=1024, L=6, heads=16, d_ff=4096, rope_base=10000, full-RoPE)
Data:  8 tickers × 2022-2025, 26tok | test: 2026-01
Infra: 4N / 16 GPU | BSZ=8/gpu, gBSZ=128 | Local Steps K=10, walltime 5h
LR:    AdamW µP 5e-4, Muon kernel LR=0.01, WD=0.005
W&B:   https://wandb.ai/oxford-lob/transformer-scaling-law (pending)

Updated: 2026-04-21 12:48 UTC

Job:   4135469 (hybrid-78m-500ctx-2n-smoke)
User:  kangli.s5e
Step:  0/~300  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: Hybrid 5L: 4×Mamba3 + 1×Attn@L3 (d=1280, ssm_state=128, head_dim=128, 10 heads) | ~78-85M params (实测待定)
Data:  GOOG 2022 × 1yr | test: Jan 2023 | msg_seq_len=500
Infra: 2N / 8 GPU | BSZ=2/gpu, gBSZ=16 | Local Steps K=1
LR:    default (ssm_lr_base standard)
Loss:  (pending smoke test)
Speed: (pending, first run with Hybrid arch)
Time:  0:00 elapsed  |  ~0:30h remaining  |  00:30 limit
ETA:   ~30min total (CURTAIL_EPOCHS=300)
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/logs_lobs5/lobs5_4135469.out

Phase:  Phase 3+4 combined (d_model calibration + smoke test)
Commit: 049c1a23 (plumbing) on top of e0f7f1c8 (hybrid code)
Goal:   (1) verify code compiles + trains without NaN; (2) read exact param count from init logs; (3) measure steady-state s/it to decide if 78M or 46M for production.

Updated: 2026-04-20 15:10:00 UTC

Job:   4135538 (ctx500-78m-wce-2n-smoke)
User:  kangli.s5e
Step:  0/~300  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: Mamba3 78M SISO (d=1024, L=6, blocks=16, ssm=1024, d_state=128, headdim=64, rope=0.5)
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01 | 26tok encoding
Infra: 2N / 8 GPU (--contiguous) | BSZ=4/gpu, gBSZ=32 | Hierarchical 2D mesh
LR:    ssm=5e-4, muon=0.01, WD=0.005
Loss:  (pending)
Speed: (pending, ~1 s/step est at 2N)
Time:  0:00 elapsed  |  ~30min smoke  |  00:30:00 limit
ETA:   Smoke test — expect ~300 curtail steps + compile time
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_weighted_ce_ns_us/logs_lobs5/lobs5_4135538.out
Note:  TOKEN_WEIGHTS_PRESET=downweight_time. Key validation: [WeightedCE] log line should show weights vector (len=21), loss should be finite and decreasing.

Updated: 2026-04-21 13:22:00 UTC

Job:   4136282 (hybrid-78m-500ctx-2n-smoke2)
User:  kangli.s5e
Step:  0/~300  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: Hybrid 5L: 4×Mamba3 + 1×Attn@L3 (d=1280, 10 heads) | ~78-85M params (实测待定)
Data:  GOOG (26tok default) | msg_seq_len=500 orders → ~13000 tokens (500×26)
Infra: 2N / 8 GPU | BSZ=2/gpu, gBSZ=16 | Local Steps K=1
LR:    default
Loss:  (pending)
Speed: (pending)
Time:  0:00 elapsed  |  ~0:30h remaining  |  00:30 limit
ETA:   ~30min (CURTAIL=300)
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/logs_lobs5/lobs5_4136282.out

Resubmit note: 4135469 crashed at step 0 — defensive check rejected TOKEN_MODE=26tok (now fixed, commit 521d70f1)
Phase:  Phase 3+4 (param calibration + smoke)
Commit: 521d70f1 on top of 049c1a23 + e0f7f1c8

Updated: 2026-04-20 15:45:00 UTC

Job:   4136610 (ctx500-78m-wce-8n-8h)
User:  kangli.s5e
Step:  0/~46050  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: Mamba3 78M SISO (d=1024, L=6, blocks=16, ssm=1024, d_state=128, headdim=64, rope=0.5)
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01 | 26tok encoding
Infra: 8N / 32 GPU | BSZ=4/gpu, gBSZ=128 | Hierarchical 2D mesh (auto ≥2N)
LR:    ssm=5e-4, muon=0.01, WD=0.005
Loss:  (pending; smoke test at 2N reached Train=2.29 Val=2.02 Test=1.79 at step 301)
Speed: ~0.6 s/step at 2N (smoke); expect slower first-run, hope 0.8-1.0 s/step at 8N after JIT
Time:  0:00 elapsed | 8:00 limit (auto-resume chain up to 3x = 24h total)
ETA:   ~7-8h for 46050 steps (~78M baseline total)
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_weighted_ce_ns_us/logs_lobs5/lobs5_4136610.out
Note:  TOKEN_WEIGHTS_PRESET=downweight_time (delta_t_s/time_s_ref=0.1, delta_t_ns/time_ns_ref=0.01).
       Apples-to-apples vs 78M baseline (job 3417629, wandb pw8u0edj in lobs5-360M-G30 or mamba3 project).
       Smoke test (2N × 30min, wandb y2yopptp): PASS, loss finite, weights installed correctly.

Updated: 2026-04-21 14:36:00 UTC

Job:   4137867 (hybrid-78m-d1152-2n-smoke)
User:  kangli.s5e
Step:  0/~300  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: Hybrid 5L: 4×Mamba3 + 1×Attn@L3 (d=1152, 9 heads × head_dim=128) | ~84M params (预估)
Data:  GOOG 26tok | msg_seq_len=500 → ~13000 tokens
Infra: 2N / 8 GPU | BSZ=2/gpu, gBSZ=16 | Local Steps K=1
LR:    default
Loss:  (pending)
Speed: (pending)
Time:  0:00 elapsed  |  ~0:30h remaining  |  00:30 limit
ETA:   ~20min (CURTAIL=300)
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/logs_lobs5/lobs5_4137867.out

Phase:  Phase 3+4 calibration re-try (first smoke 4136282 d=1280 gave 103M, this aims 78M)
Commit: same as 521d70f1 (no code change, just env var)

Updated: 2026-04-20 16:10:00 UTC

Job:   4138134 (ctx500-78m-wce-8n-8h-retry)
User:  kangli.s5e
Step:  0/~46050  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: Mamba3 78M SISO (d=1024, L=6, blocks=16, ssm=1024)
Data:  8 tickers × 4yr (2022-2025) | 26tok | TOKEN_WEIGHTS_PRESET=downweight_time
Infra: 8N / 32 GPU | BSZ=4/gpu gBSZ=128
LR:    ssm=5e-4, muon=0.01, WD=0.005
Loss:  (pending)
Speed: ~0.7 s/step expected (smoke 2N was 0.6 s/step; 8N slightly slower due to AllReduce)
Time:  0:00 elapsed | 8:00 limit
ETA:   ~46050 × 0.7s ≈ 9h > 8h limit → auto-resume chain expected
W&B:   (pending; previous retry was armu2juu, crashed nid010188)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_weighted_ce_ns_us/logs_lobs5/lobs5_4138134.out
Note:  Retry after j4136610 crashed at step ~113 with NCCL OFI error on nid010188. Bad node now excluded.

Updated: 2026-04-21 14:52:00 UTC

Job:   4138166 (siso-360m-2n-bsz2-baseline)
User:  kangli.s5e
Step:  0/?  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: Mamba3 SISO 360M (d=2048, L=24, B=32, ssm=2048) | pure JAX default
Data:  default MSG_SEQ_LEN=500 (26 tok/msg, context=13000 tokens)
Infra: 2N / 8 GPU | BSZ=2/gpu, gBSZ=16 | CURTAIL_EPOCHS=300 benchmark
Loss:  (pending)
Speed: ~TBD
Time:  0:03 elapsed  |  ~27min remaining  |  30min limit
ETA:   ~30min total (fixed by CURTAIL_EPOCHS)
W&B:   (pending — NO_VALIDATION=1, wandb probably enabled)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R100_Mamba3_cudaffi/logs_lobs5/siso-360m-2n-bsz2-baseline_4138166.out
Purpose: v2 Phase 1 baseline — measure pure JAX Mamba3 SISO speed + memory ceiling

Updated: 2026-04-20 14:58:00 UTC

Job:   4138676 (siso-360m-2n-bsz2-baseline-v2)
User:  kangli.s5e
Step:  0/?  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: Mamba3 SISO 360M | resubmit after 4138166 CUDA noise (real cause: OOM 225GB @ step 2, sharding issue)
Data:  MSG_SEQ_LEN=500 default
Infra: 2N / 8 GPU | BSZ=2/gpu, gBSZ=16 | CURTAIL=300
LR:    default
Loss:  (pending)
Speed: (pending — expect same OOM)
Time:  PENDING
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R100_Mamba3_cudaffi/logs_lobs5/siso-360m-2n-bsz2-baseline-v2_4138676.out
Purpose: Verify env fix removes CUDA init noise (but 225GB OOM likely unchanged)

Updated: 2026-04-20 15:05:00 UTC

Job:   4138756 (siso-78m-2n-bsz10-baseline)
User:  kangli.s5e
Step:  0/?  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: Mamba3 SISO 78M (d=1024, L=6, B=16, ssm=1024) | known-good config for 2N
Data:  MSG_SEQ_LEN=500 default
Infra: 2N / 8 GPU | BSZ=10/gpu, gBSZ=80 | CURTAIL=300
LR:    default 5e-4
Loss:  (pending)
Speed: (pending)
Time:  PENDING
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R100_Mamba3_cudaffi/logs_lobs5/siso-78m-2n-bsz10-baseline_4138756.out
Purpose: v2 Phase 1 primary baseline — real pure JAX speed + memory reference

Updated: 2026-04-20 15:05:00 UTC

Job:   4138959 (siso-78m-2n-bsz10-baseline-v3)
User:  kangli.s5e
Step:  0/?  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: Mamba3 SISO 78M (d=1024, L=6, B=16, ssm=1024) | pure JAX, USE_TRITON=False
Data:  MSG_SEQ_LEN=500 default, 26 tok/msg
Infra: 2N / 8 GPU | BSZ=10/gpu, gBSZ=80 | CURTAIL=300
LR:    5e-4 default
Loss:  (pending)
Speed: (pending)
Time:  PENDING
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R100_Mamba3_cudaffi/logs_lobs5/siso-78m-2n-bsz10-baseline-v3_4138959.out
Purpose: v2 Phase 1 baseline v3 — reverted batch to strip unsupported --diloco_outer args

Updated: 2026-04-20 15:08:00 UTC

Job:   4140238 (siso-78m-2n-bsz10-baseline-r1)
User:  kangli.s5e
Step:  0/?  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: Mamba3 SISO 78M (d=1024, L=6, B=16, ssm=1024) | from R1 mainline (commit 9f6db8be)
Data:  MSG_SEQ_LEN=500 default, 26 tok/msg
Infra: 2N / 8 GPU | BSZ=10/gpu, gBSZ=80 | CURTAIL=300 | USE_TRITON=False (pure JAX)
LR:    5e-4 default
Loss:  (pending)
Speed: (pending)
Time:  PENDING
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3/logs_lobs5/siso-78m-2n-bsz10-baseline-r1_4140238.out
Purpose: v2 Phase 1 CANONICAL baseline — R1 mainline, pure JAX, reference for kernel fusion

Updated: 2026-04-20 15:15:00 UTC

Job:   4140645 (mamba3-75m-2n-bsz10-purejax)
User:  kangli.s5e
Step:  0/?  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: Mamba3 SISO 75M (preset) | SSM_TYPE=mamba3, MAMBA3_USE_TRITON=False (PURE JAX)
Data:  MSG_SEQ_LEN=500 default, 26 tok/msg
Infra: 2N / 8 GPU | BSZ=10/gpu, gBSZ=80 | CURTAIL=300 | NO_VALIDATION=1
LR:    5e-4 default
Loss:  (pending)
Speed: (pending)
Time:  PENDING
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3/logs_lobs5/mamba3-75m-2n-bsz10-purejax_4140645.out
Purpose: Real v2 baseline — Mamba3 SISO pure JAX (prior 4 fails were running GDN Triton by default, OOM 197GB)

Updated: 2026-04-20 17:30:00 UTC

## Transformer 78M + RoPE resubmit (NCCL stall @ step 190 on 4132064)
Submitted: 2026-04-21 (post-compact) by aramis.s5e

Job:   4140881 (tf-78m-rope-4n)
Prior: 4132064 FAILED at 22:32 (step watchdog 900s). Training ran 190 steps @ ~1.5 it/s,
       then collective hung — no NCCL WARN in logs, looks like transient Slingshot issue.
       RoPE itself was working (odd-head_dim fallback confirmed).
Model: Transformer 78M + RoPE (d=1024, L=6, heads=16, d_ff=4096, rope_base=10000, full-RoPE)
Data:  8 tickers × 2022-2025, 26tok | test: 2026-01
Infra: 4N / 16 GPU | BSZ=8/gpu, gBSZ=128 | Local Steps K=10, walltime 5h
Extra: NCCL_DEBUG=WARN (to catch collective if it hangs again)
Log:   /lus/.../exp_O8_self_attention/scaling_logs/training_4140881_node*.log

Job:   4140882 (mamba3-75m-2n-bsz2-purejax)
User:  kangli.s5e
Step:  0/?  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: Mamba3 SISO 75M | SSM_TYPE=mamba3, MAMBA3_USE_TRITON=False (pure JAX)
Data:  MSG_SEQ_LEN=500 default, 26 tok/msg, ctx=13000 tokens
Infra: 2N / 8 GPU | BSZ=2/gpu, gBSZ=16 | CURTAIL=300 | NO_VALIDATION=1
LR:    5e-4 default
Loss:  (pending)
Speed: (pending)
Time:  PENDING
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3/logs_lobs5/mamba3-75m-2n-bsz2-purejax_4140882.out
Purpose: v2 Phase 1 baseline - BSZ=2 conservative (BSZ=10 OOM 267GB). Lower-bound establish, then sweep up.

Updated: 2026-04-20 17:40:00 UTC

Job:   4140911 (ctx500-78m-wce-4n-ga2-8h) [PLAN B]
User:  kangli.s5e
Step:  0/~46050  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: Mamba3 78M (d=1024, L=6, blocks=16, ssm=1024)
Data:  8 tickers × 4yr (2022-2025) | 26tok | TOKEN_WEIGHTS_PRESET=downweight_time
Infra: 4N / 16 GPU (--contiguous) | BSZ=4/gpu gBSZ=128 via GRAD_ACCUM_STEPS=2
LR:    ssm=5e-4, muon=0.01, WD=0.005
Loss:  (pending)
Speed: (pending; estimated ~1.5 s/step with grad_accum=2 micro-steps)
Time:  0:00 elapsed | 8:00 limit
ETA:   ~20h total for 46050 steps → ~3x auto-resume chain
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_weighted_ce_ns_us/logs_lobs5/lobs5_4140911.out
Note:  PLAN B after 4/4 consecutive 8N crashes with OFI RC 265. 4N reduces Slingshot
       pressure; grad_accum=2 maintains global BSZ 128 for apples-to-apples vs 78M baseline.
       Hypothesis: 8N scale + 2D hierarchical mesh triggers fabric congestion that
       doesn't occur at 4N.

Updated: 2026-04-21 16:56:00 UTC

Job:   4140921 (book-extra-profile)
User:  kangli.s5e
Step:  N/A (one-shot extra profiling)
Data:  Same 8 tickers × 2022-2025, STRIDE=20
Scope: 4 new profiling dimensions -- spread hist, rank-tick joint, per-year CDF, ask/bid asym
Infra: 1N / 1 GPU / 64 CPU | 60min limit
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/tasks/book_depth_profiling/logs/extra_4140921.out

Updated: 2026-04-21 (time tbd) UTC

Bench:  mamba3_mimo79m_j4114758_s73390_26tok (16 jobs)
User:   kangli.s5e
Type:   LOBbench c250g250, n_sequences=3136, 26tok
Target: MIMO 79M checkpoint j4114758 step 73390
Tickers: GOOG AAPL NVDA AMZN META TSLA MSFT AMD
Jobs:   GOOG=4141157/4141158 AAPL=4141159/4141160 NVDA=4141161/4141162
        AMZN=4141163/4141164 META=4141165/4141166 TSLA=4141167/4141168
        MSFT=4141169/4141170 AMD=4141171/4141172
        (infer / score pairs, 1N each, 4h walltime)
Results: /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/lob_pipeline/results_mamba3_mimo79m_j4114758_s73390_26tok/
Compare: Mamba3 SISO 78M j3417629 at results_ext-78m-curr-c1750g250-n3136 (paper anchor)

Updated: 2026-04-21 UTC

## Transformer test-CE RE-SUBMIT (TOKEN_MODE bug fix)
Submitted: 2026-04-21 (post-compact) by aramis.s5e

Bug found in first pass (4128850-4128856): test_ce CSVs show seq_len=12000 = 500×24,
meaning eval ran with 24tok encoding instead of the 26tok the models trained on.
Root cause: `lob/encoding.py` reads TOKEN_MODE env var AT IMPORT TIME. CLI arg
--token_mode 26tok gets parsed too late. Fix: export TOKEN_MODE=26tok in batch
before python launches.

| Job      | Label   | Ckpt                                         | Output                 |
|----------|---------|----------------------------------------------|------------------------|
| 4141196  | tf-78m  | j4108846_97o4ua84_4108846                    | test_ce_tf78m.csv      |
| 4141197  | tf-8m   | j4108841_y6qya8s3_4108841                    | test_ce_tf-8m.csv      |
| 4141198  | tf-14m  | j4108842_7mqajiyb_4108842                    | test_ce_tf-14m.csv     |
| 4141199  | tf-23m  | j4108843_ypo60zdm_4108843                    | test_ce_tf-23m.csv     |
| 4141200  | tf-34m  | j4108844_u8na1hup_4108844                    | test_ce_tf-34m.csv     |
| 4141201  | tf-46m  | j4108845_izainmab_4108845                    | test_ce_tf-46m.csv     |

Prior (buggy) CSVs renamed to *_24tok_buggy.csv (preserved, not used).

Job:   4141314 (mamba3-75m-2n-bsz3-purejax)
User:  kangli.s5e
Step:  0/?  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: Mamba3 SISO 75M (pure JAX) | SSM_TYPE=mamba3, MAMBA3_USE_TRITON=False
Infra: 2N / 8 GPU | BSZ=3/gpu, gBSZ=24 | CURTAIL=300
Purpose: v2 Phase 1 BSZ sweep — find OOM boundary (BSZ=2 ok, BSZ=10 OOM 267GB)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3/logs_lobs5/mamba3-75m-2n-bsz3-purejax_4141314.out

Updated: 2026-04-20 17:50:00 UTC

Job:   4141315 (mamba3-75m-2n-bsz4-purejax)
User:  kangli.s5e
Step:  0/?  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: Mamba3 SISO 75M (pure JAX) | SSM_TYPE=mamba3, MAMBA3_USE_TRITON=False
Infra: 2N / 8 GPU | BSZ=4/gpu, gBSZ=32 | CURTAIL=300
Purpose: v2 Phase 1 BSZ sweep — find OOM boundary (BSZ=2 ok, BSZ=10 OOM 267GB)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3/logs_lobs5/mamba3-75m-2n-bsz4-purejax_4141315.out

Updated: 2026-04-20 17:50:00 UTC

Job:   4141316 (mamba3-75m-2n-bsz6-purejax)
User:  kangli.s5e
Step:  0/?  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: Mamba3 SISO 75M (pure JAX) | SSM_TYPE=mamba3, MAMBA3_USE_TRITON=False
Infra: 2N / 8 GPU | BSZ=6/gpu, gBSZ=48 | CURTAIL=300
Purpose: v2 Phase 1 BSZ sweep — find OOM boundary (BSZ=2 ok, BSZ=10 OOM 267GB)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3/logs_lobs5/mamba3-75m-2n-bsz6-purejax_4141316.out

Updated: 2026-04-20 17:50:00 UTC

Job:   4141825 (78m-curr-s2to4k-ch1)
User:  aramis.s5e
Model: 78M Mamba3 (d=1024, L=6, B=16, d_state=128, headdim=64, rope=0.5, tok26)
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01 | msg_seq_len=4000 ← EXTENDED
Infra: 8N / 32 GPU | BSZ=1/gpu, gBSZ=32 | Local Steps K=10 | REMAT=1
LR:    Muon 0.001 / SSM 1e-4 (5× reduced, continuation regime), wd=0.005
       Cosine over 80k steps, warmup=0.02, RESET_SCHEDULE=True (fresh schedule)
Init:  resume from j4082152 step 35280 (longrun ch1 FINAL)
Goal:  Context extension s=2k → s=4k. Longrun already beats paper at g≥1000 + div_sub2
       at c≥1000; s=4k exposure should push long-horizon metrics further.
       Muon choice validated by 46M direct optimizer A/B (Muon +58-71% at g≥1000).
CURTAIL_EPOCHS=80000 (~9h walltime at 8N, s=4k roughly 2× slower than s=2k per step)
W&B:   (fresh run, TBD)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3/logs_lobs5/lobs5_4141825.out

Updated: 2026-04-21 UTC

Job:   4147195 (mamba3-75m-d512-2n-bsz2-purejax)
User:  kangli.s5e
Step:  0/?  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: Mamba3 SISO 75M preset + MAMBA3_D_STATE=512 | pure JAX (USE_TRITON=False)
Data:  MSG_SEQ_LEN=500 default, 26 tok/msg
Infra: 2N / 8 GPU | BSZ=2/gpu, gBSZ=16 | CURTAIL=300 | NO_VALIDATION=1
LR:    5e-4 default
Purpose: v2 Phase 1 CANONICAL baseline v2 — match aramis d_state=512 for K3 comparison
  Prior d_state=128 baseline: 4140882 (0.354 s/step @ BSZ=2)
  Next: K3 forward enable (M1), then K3 backward (M2)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3/logs_lobs5/mamba3-75m-d512-2n-bsz2-purejax_4147195.out

Updated: 2026-04-20 19:20:00 UTC

Job:   4149626 (mamba3-siso-verify-v2)
User:  kangli.s5e
Step:  N/A (SISO correctness + BCNorm probe)
Model: N/A (pure algorithm test, no training)
Data:  random synthetic (T=32, nheads=4, d_state=16)
Infra: 1N / 1 GPU | CPU-oriented test, GPU for JAX init only
LR:    N/A
Loss:  N/A
Speed: < 2min est
Time:  00:00 elapsed  |  00:30 limit
ETA:   ~2 min compute, depends on queue
W&B:   N/A
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/tasks/mamba3_siso_official_check/logs/verify_4149626.out

Updated: 2026-04-21 21:02 UTC


Job:   4149653 (R11verify-GOOG-real-4yr-4n)
User:  kangli.s5e
Step:  0/30,430  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: 8M Mamba3 SISO (d=256, L=6, d_state=128, Muon) | ~8.10M params
Data:  GOOG-only | 2022-01-01 to 2025-12-31 (4yr) | real book
Infra: 4N / 16 GPU | BSZ=10/gpu, gBSZ=160 | Local Steps K=10
LR:    muon_lr=0.01, ssm_lr=2e-3, wd=0.005
Loss:  (pending)
Speed: ~0.4 s/step (est)
Time:  0:00 elapsed  |  ~3.5h remaining  |  5:00 limit
ETA:   ~3.5h total
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R11_book-ablation/logs_lobs5/lobs5_4149653.out

Purpose: 2x2 controlled ablation (minimum viable) -- Cell 1 Single-Real.
         Matches Aramis R11 multi-ticker config except TICKERS=GOOG only.
         Pairs with 4149654 (zero) vs existing j3443014/j3534821 (multi).

Updated: 2026-04-21 UTC

Job:   4149654 (R11verify-GOOG-zero-4yr-4n)
User:  kangli.s5e
Step:  0/30,430  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: 8M Mamba3 SISO (d=256, L=6, d_state=128, Muon) | ~8.10M params
Data:  GOOG-only | 2022-01-01 to 2025-12-31 (4yr) | BOOK_ABLATION=zero
Infra: 4N / 16 GPU | BSZ=10/gpu, gBSZ=160 | Local Steps K=10
LR:    muon_lr=0.01, ssm_lr=2e-3, wd=0.005
Loss:  (pending)
Speed: ~0.4 s/step (est)
Time:  0:00 elapsed  |  ~3.5h remaining  |  5:00 limit
ETA:   ~3.5h total
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R11_book-ablation/logs_lobs5/lobs5_4149654.out

Purpose: 2x2 controlled ablation (minimum viable) -- Cell 2 Single-Zero.
         Pairs with 4149653 (real). Completes the single-ticker side of
         the Single/Multi x Real/Zero factorial.

Updated: 2026-04-21 UTC

Bench:  mamba3_mimo79m_j4114758_s73390_26tok (RESUBMIT with --lobs5_dir fix)
User:   kangli.s5e
Type:   LOBbench c250g250, n_sequences=3136, 26tok
Fix:    --lobs5_dir /lus/lfs1aip2/.../exp_R1g_MIMO_pure_jax (prev failed: main LOBS5 has mimo_rank=1, checkpoint is MIMO R=4)
Jobs:   GOOG=4149735/4149736 AAPL=4149737/4149738 NVDA=4149739/4149740
        AMZN=4149741/4149742 META=4149743/4149744 TSLA=4149745/4149746
        MSFT=4149747/4149748 AMD=4149749/4149750
Cancelled: prev 4141158,4141160,4141162,4141164,4141166,4141168,4141170,4141172 (DependencyNeverSatisfied)

Updated: 2026-04-21 UTC

Job:   4141337 (ctx500-78m-wce-8n-8h-retry-resume10) [WINNING RUN]
User:  kangli.s5e
Step:  20101/46050  [█████████░░░░░░░░░░░░░░░░░░░░░]  43.6%
Model: Mamba3 78M SISO (d=1024, L=6, blocks=16, ssm=1024)
Data:  8 tickers × 4yr (2022-2025) | 26tok | TOKEN_WEIGHTS_PRESET=downweight_time
Infra: 8N / 32 GPU | BSZ=4/gpu gBSZ=128 | Hierarchical 2D mesh
LR:    ssm=5e-4, muon=0.01, WD=0.005
Speed: 0.73 s/step (MFU 4.3%, 1347 TFLOPS) stable
Time:  4:05 elapsed | 3:55 remaining in this sbatch
ETA:   ~5.3h remaining for 46050 steps → will auto-resume 1x after 8h sbatch ends
W&B:   https://wandb.ai/oxford-lob/lobs5-360M-G30/runs/ocnmuea9
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_weighted_ce_ns_us/logs_lobs5/lobs5_4141337.out
Note:  10th in auto-resume chain after 9 cluster-fabric OFI crashes. Fabric self-recovered.
       Plan B (4140911 4N+ga2) scancel'd per user choice A after both running stable.
       TOKEN_WEIGHTS_PRESET=downweight_time: delta_t_s/time_s_ref=0.1, delta_t_ns/time_ns_ref=0.01

Updated: 2026-04-21 19:00:00 UTC


Job:   4150094 (R11verify-GOOG-real-4yr-4n)  [RETRY post-fix d118378d]
User:  kangli.s5e
Step:  0/30,430  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: 8M Mamba3 SISO (d=256, L=6, d_state=128, Muon) | ~8.10M params
Data:  GOOG-only | 2022-01-01 to 2025-12-31 (4yr) | real book
Infra: 4N / 16 GPU | BSZ=10/gpu, gBSZ=160 | Local Steps K=10
LR:    muon_lr=0.01, ssm_lr=2e-3, wd=0.005
Loss:  (pending)
Speed: ~0.4 s/step (est)
Time:  0:00 elapsed  |  ~3.5h remaining  |  5:00 limit
ETA:   ~3.5h total
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R11_book-ablation/logs_lobs5/lobs5_4150094.out

Purpose: 2x2 controlled ablation Cell 1. Prior attempt 4149653 crashed
         in 3min on repeat_book vmap kwarg bug, fixed in commit d118378d
         (5 call sites reverted from shift_start=True kwarg to positional).

Updated: 2026-04-21 UTC

Job:   4150095 (R11verify-GOOG-zero-4yr-4n)  [RETRY post-fix d118378d]
User:  kangli.s5e
Step:  0/30,430  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: 8M Mamba3 SISO (d=256, L=6, d_state=128, Muon) | ~8.10M params
Data:  GOOG-only | 2022-01-01 to 2025-12-31 (4yr) | BOOK_ABLATION=zero
Infra: 4N / 16 GPU | BSZ=10/gpu, gBSZ=160 | Local Steps K=10
LR:    muon_lr=0.01, ssm_lr=2e-3, wd=0.005
Loss:  (pending)
Speed: ~0.4 s/step (est)
Time:  0:00 elapsed  |  ~3.5h remaining  |  5:00 limit
ETA:   ~3.5h total
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R11_book-ablation/logs_lobs5/lobs5_4150095.out

Purpose: 2x2 controlled ablation Cell 2. Prior attempt 4149654 crashed
         same bug. Fixed and resubmitted.

Updated: 2026-04-21 UTC

Bench:  mamba3_mimo79m_j4114758_s73390_26tok (v3 — --save_format fix)
User:   kangli.s5e
Fix:    commit a47ff7e9 adds --save_format arg to MIMO worktree run_inference.py
Prev:   v1 ScopeParamShapeError (mimo_rank mismatch), v2 argparse error (--save_format)
Jobs:   GOOG=4150186/4150187 AAPL=4150188/4150189 NVDA=4150190/4150191
        AMZN=4150192/4150193 META=4150194/4150195 TSLA=4150196/4150197
        MSFT=4150198/4150199 AMD=4150200/4150201
Cancelled: v2 4149736,4149738,...,4149750 (8 stuck score jobs)

Updated: 2026-04-21 UTC

Job:   4150312 (hybrid-85m-d1152-16n-24h)
User:  kangli.s5e
Step:  0/~466K (40 epochs, 11.6K steps/epoch at gBSZ=128)  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: Hybrid 5L: 4×Mamba3 + 1×Attn@L3 (d=1152, 9 heads × head_dim=128) | 84.95M params (smoke verified)
Data:  GOOG 26tok default | msg_seq_len=500 orders → ~13000 tokens/sample
Infra: 16N / 64 GPU | BSZ=2/gpu, gBSZ=128 | Local Steps K=1
LR:    default (ssm_lr_base standard)
Loss:  (pending, smoke @ 2N reached 2.27 train / 1.55 val at step 300)
Speed: (pending; smoke @ 2N had 0.11 s/step steady)
Time:  0:00 elapsed  |  ~24:00h remaining  |  24:00:00 limit
ETA:   multi-session (24h ≈ 3 epochs @ 9 it/s; auto-resume chain covers full 40 epochs)
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/logs_lobs5/lobs5_4150312.out

Phase:  Phase 5 production training (persistent)
Commit: 521d70f1 on mamba3 branch
Lineage: smoke1 4135469 (crashed 26tok) → fix 521d70f1 → smoke2 4136282 d=1280 (103M over) → smoke3 4137867 d=1152 (84.9M ✓) → PRODUCTION 4150312

Updated: 2026-04-21 17:30:00 UTC

Bench:  mamba3_mimo79m_j4114758_s73390_26tok (v4 — mimo_rank fix)
User:   kangli.s5e
Fix:    commit 92d5f9ca: pass mimo_rank through initialize_carry in inference_no_errcorr.py
Chain:  v1 shape(mimo_rank=1) → v2 argparse(--save_format) → v3 scan carry shape → v4 ?
Jobs:   GOOG=4150473/4150474 AAPL=4150475/4150476 NVDA=4150477/4150478
        AMZN=4150479/4150480 META=4150481/4150482 TSLA=4150483/4150484
        MSFT=4150485/4150486 AMD=4150487/4150488
Cancelled: v3 4150187,...,4150201 (8 stuck score)

Updated: 2026-04-21 UTC

========================================================================
Job:   4150495 (variance-C0-bootstrap)
User:  kangli.s5e
Step:  bootstrap/bootstrap  [variance measurement, not training]
Model: GDN 84M soup (j2722414_mr8w0rjg, 5-ckpt soup)
Data:  GOOG Jan 2026 (6272 pre-existing rollouts from soup-gdn-h250_GOOG_3294924)
Infra: 1N / 1 GPU (placeholder) | 64 CPU | CPU-only bench
LR:    N/A (eval)
Loss:  N/A
Speed: N/A
Time:  0m elapsed | ~30m budget | 30m limit
ETA:   ~30m
W&B:   N/A
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/tasks/dense-reward-grpo-posttraining/agent_outputs/variance_bootstrap_4150495.log

Purpose: PRAP §7 variance re-measurement (C0 baseline σ_empirical). 
         20 × bootstrap of 5000 rollouts → σ(IC), σ(DirAcc) per horizon.
         Result JSON updates PRAP §3 decision thresholds.

Updated: 2026-04-21 ~22:50 UTC
========================================================================

========================================================================
Job:   4150629 (variance-C0-bootstrap-v2, resubmit after 4150495 fatal)
User:  kangli.s5e
Step:  bootstrap/bootstrap
Model: GDN 84M soup (j2722414_mr8w0rjg, 5-ckpt soup)
Data:  GOOG Jan 2026 (3136 paired rollouts, sample_size=2000 per bootstrap)
Infra: 1N / 1 GPU (placeholder) | 64 CPU | CPU-only bench
LR:    N/A
Loss:  N/A
Speed: N/A
Time:  0m elapsed | 30m limit
W&B:   N/A
Log:   /lus/.../dense-reward-grpo-posttraining/agent_outputs/variance_bootstrap_4150629.log

Crash note (4150495): sample_size=5000 exceeded total pairs 3136. Fixed in PRAP v1.4.

Updated: 2026-04-21 ~23:20 UTC
========================================================================

========================================================================
Job:   4151124 (dense-smoke-1n-5step) [PRAP Stage 1 smoke]
User:  kangli.s5e
Step:  0/5 [dense reward GRPO smoke]
Model: GDN 84M (j2722414 step 23356, single step)
Data:  GOOG 2022 train (DATA_DIR) / Jan 2023 test (TEST_DIR)
Infra: 1N / 4 GPU | G=8, PER_GPU_BSZ=4, GRPO_STEPS=5 
Config: DENSE_REWARD=True, REWARD_DELTA=10, KL_COEFF=0.1
LR:    GRPO_LR (default)
Loss:  pending
Speed: pending
Time:  0m elapsed | 30m limit
ETA:   ~15-25m (5 GRPO steps + setup)
W&B:   pending (grep wandb.ai in node0 log)
Log:   /lus/.../exp_O11_dense_grpo/logs_lobs5/grpo_4151124.out
NodeLog: /lus/.../exp_O11_dense_grpo/logs_lobs5/training_4151124_node0.log

Purpose: PRAP Stage 1 pass gate validation. Check:
  (1) R_std > 0.3 (dense reward dispersion alive)
  (2) adv_std > 0.1 (PRAP §8 Stage 1 gate)
  (3) DirAcc_local_h10 > 52% (per-step direction signal > random)
  (4) No NaN, gnorm < 10

Updated: 2026-04-21 ~23:35 UTC
========================================================================

========================================================================
Crash note (4151124):  DATA_DIR `/lus/.../kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2022` 
                        不存在 (batch 默认 stale). data 已移到 /projects/s5e/quant/GOOG2016TO2021/2022/.
Fix: DATA_DIR env override, resubmitted as new job.
========================================================================

========================================================================
Crash note (4151299):  FileNotFoundError: Orbax CheckpointManager 期望 root
                        (parent of step dirs). RESTORE=.../j2722414_mr8w0rjg_2722414/23356
                        错在多了 /23356. Fix: RESTORE=parent, RESTORE_STEP=23356.
========================================================================

========================================================================
Job:   4151645 (dense-smoke-v5) [PRAP Stage 1 ✅ PASSED]
User:  kangli.s5e
Result: 5/5 steps, exit 0:0, 5:32 elapsed.
  R_std ∈ [0.47, 0.56]  (gate: >0.3 ✅)
  adv_std ∈ [0.91, 0.97]  (gate: >0.1 ✅)
  adv_frac_nz ∈ [0.70, 0.82]  (gate: >0.5 ✅)
  batch_IC trajectory: -0.06 → -0.06 → +0.05 → 0.00 → +0.02
  gnorm: 283 → 245 → 47 → 85 → 25 (converging)
W&B:   https://wandb.ai/oxford-lob/lobs5-O9-grpo/runs/8640ybqo

Stage 1 passed all gates. Dense reward implementation verified functional.

Crash history (for reference):
  v1 4151124  DATA_DIR stale path                            (env fix)
  v2 4151299  RESTORE 多 /23356 (Orbax root vs step semantics) (env fix)
  v3 4151390  d_model=512 vs ckpt d=1024                     (env fix)
  v4 4151493  gdn_num_heads=4 vs 8 (+GDN_* env override)     (code + env, commit 2ff3a313)
  v5 4151645  41-col vs 43-col data format                    (symlink + env fix) → SUCCESS

Updated: 2026-04-22 ~00:00 UTC
========================================================================

========================================================================
Job:   4151922 (dense-stage2-1n-100step) [PRAP Stage 2 ✅ PASSED]
User:  kangli.s5e
Result: exit 0:0, 40/100 steps, 4h limit reached.
  Per-step: 360s (6 min/step) — slower than smoke (60s) due to dense backward 2x + ckpt save
  R_mean trajectory: -0.061 (smoke baseline) → +0.027 (step 39), multiple peaks at +0.13
  Average R_mean over 40 steps: ~+0.05 (consistent improvement, no regress)
  ckpt saved at grpo_step=25 (optimizer_step=50)
  PRAP Stage 2 Pass gate "不 regress" ✅ satisfied
W&B:   https://wandb.ai/oxford-lob/lobs5-O9-grpo/runs/ao21bo39

Lessons: 
  (1) Dense path doubles optimizer steps per grpo step → 50h for 500 step on 1N
  (2) PRAP v1.7 amendment: Stage 3 → 240 step, eval ckpts (100/200/240)
Updated: 2026-04-22 ~04:00 UTC
========================================================================

Job:   4153896 (hybrid-85m-d1152-8n-24h)
User:  kangli.s5e
Step:  0/~930K (40 epochs, 23K steps/epoch at gBSZ=64)  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: Hybrid 5L: 4×Mamba3 + 1×Attn@L3 (d=1152, 9 heads × head_dim=128) | 84.95M params
Data:  GOOG 26tok default | msg_seq_len=500 orders → ~13000 tokens/sample
Infra: 8N / 32 GPU | BSZ=2/gpu, gBSZ=64 | Local Steps K=1
LR:    default (ssm_lr_base standard; note: gBSZ=64 half of 16N target, LR sqrt-scale may apply)
Loss:  (pending)
Speed: (pending; smoke @ 2N had 0.11 s/step steady, 8N projected similar)
Time:  0:00 elapsed  |  ~24:00h remaining  |  24:00:00 limit
ETA:   multi-session (auto-resume chain for full 40 epochs)
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/logs_lobs5/lobs5_4153896.out

Phase:  Phase 5 production (downscaled from 16N due to FairShare queue wait)
Lineage: 4150312 16N (scancelled, 17h wait) → 4153896 8N
Commit: 521d70f1 on mamba3 branch

Updated: 2026-04-22 01:00:00 UTC

===== LOBbench for Weighted CE (j4141337 step 39742) =====
Submitted: 2026-04-22
Pipeline: /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/lob_pipeline
Name:     j4141337_step39742_wce_26tok
Config:   8 tickers × 5000 sequences, no HF compare, 1 node/ticker, 8h walltime
Results:  /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/lob_pipeline/results_j4141337_step39742_wce_26tok/

Job IDs:
  GOOG: 4161650 (infer) + 4161651 (score)
  AAPL: 4161652 + 4161653
  NVDA: 4161654 + 4161655
  AMZN: 4161656 + 4161657
  META: 4161658 + 4161659
  TSLA: 4161660 + 4161661
  MSFT: 4161662 + 4161663
  AMD:  4161664 + 4161665

Comparison target: 78M baseline j3417629 (wandb pw8u0edj), same 8 tickers, 26tok.
Training status: weighted CE reached step 39742 of 46050 target (86%).

Updated: 2026-04-22 UTC

Job:   4161782 (s8k-tp4-pathB-smoke)
User:  aramis.s5e
Step:  resume from j4082152 step 35280
Model: 78M Mamba3 (d=1024, L=6, ssm=128) — Path B (TP=N==TP=1 fix)
Data:  GOOG 2022 (will pick up curriculum from longrun)
Infra: 8N / 32 GPU | TP=4, BSZ=1/gpu, gBSZ=8 | REMAT=1
LR:    inherited from longrun
Loss:  longrun final (TBD step 1 should match)
Speed: TBD
Time:  ~30min smoke (CURTAIL_EPOCHS=300)
ETA:   30min
W&B:   TBD
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3/logs_lobs5/lobs5_4161782.out
Note:  First test of Path B fix in production (committed 13f04e52). Goal:
       confirm longrun TP=1 ckpt loads into TP=4 model with no shape mismatch
       and step-1 loss continues smoothly.

Updated: 2026-04-22 (smoke submitted)

Job:   4184212 (m3-78m-rmsfix-8n)
User:  aramis.s5e
Step:  0/466138  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0.0%
Model: 78M Mamba3 (d=1024, L=6, headdim=64, nh=32, ssm=128) | ~78.5M params
       FIX: RMSNorm bug corrected (commit ffe45d66 + Path B 13f04e52)
       BCNorm now active (was constant scaler ~1.0 due to epsilon trap)
       out_norm now active (was epsilon=2048, now eps=1e-6)
Data:  8 tickers (GOOG/AAPL/NVDA/AMZN/META/TSLA/MSFT/AMD) × 4yr (2022-2025) | test: 2026-01
Infra: 8N / 32 GPU | BSZ=4/gpu, gBSZ=128 | Local Steps K=10
LR:    Muon 0.01 (kernel) / SSM 5e-4 / WD 0.005
Loss:  pending (job submitted)
Speed: pending
Time:  0:00 elapsed | ~6:00 limit
ETA:   ~6h to ~46k steps (matches original step 46050 reference)
W&B:   pending
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3/logs_lobs5/lobs5_4184212.out
Note:  Re-train of original j3417629 with BCNorm/out_norm fixed. 1 epoch budget.

Updated: 2026-04-22 17:30:00 UTC

[Update] Job 4184212 — 50min in, healthy
W&B:   https://wandb.ai/oxford-lob/mamba3/runs/yhmpnuk6
Step:  ~3890/466138  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0.83%
Speed: ~1.55 it/s steady state (~0.65 s/step)
MFU:   10.4% (3280 TFLOPS)
Saves: mid-epoch checkpoints at step 1469, 2919 (working correctly)
Status: no NaN, no OOM, no crashes — RMSNorm fix training cleanly

Updated: 2026-04-22 18:20:00 UTC

---
Job:   4187637 (acf-val-msft)
User:  kangli.s5e
Task:  LOBbench ACF-only validation (Phase 1 metrics)
Model: j4141337_step39742_wce_26tok (26tok pretrained)
Data:  MSFT × 2023 | 5000 real/gen/cond pairs (reused inference)
Infra: 1N / 1 GPU / 64 CPU | 30min walltime
Metrics: returns_autocorr_lag1, returns_autocorr_lag10, ofi_autocorr_lag1
Repo:  lob_pipeline/lob_bench commit 267cfb4
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/tasks/lobbench_time_step_dependency/logs/acf_val_4187637.out
Purpose: verify new Phase 1 ACF metrics (returns ≈ 0, OFI ≈ 0.2-0.4) on real data

Updated: 2026-04-22

Job:   4173063 (mamba3-siso-rmsnorm-fix-smoke)
User:  kangli.s5e
Step:  N/A (smoke test, 30 min, CURTAIL=300)
Model: 10M (d_model=512, L=6, ssm=512, BSZ=20/GPU, bgBSZ=160)
Data:  LOBS5 default (GOOG 2022)
Infra: 2N / 8 GPU | contiguous
LR:    default sqrt(k) scaled from 5e-4 baseline
Loss:  N/A
Speed: N/A
Time:  00:00 elapsed  |  00:30 limit
ETA:   ~30 min
W&B:   TBD
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3_siso_official_check/logs_lobs5/lobs5_4173063.out
Purpose: verify RMSNorm()+exp guard fix doesn't NaN (testing 2026-03-30 NaN hypothesis)

Updated: 2026-04-22 UTC

Job:   4187975 (mamba3-siso-fix-smoke-v2)
User:  kangli.s5e
Step:  N/A (smoke test v2 with SSM_TYPE=mamba3 MODEL_PRESET=75m)
Model: 75M Mamba3 SISO (per batch preset)
Data:  LOBS5 default
Infra: 2N / 8 GPU | contiguous
LR:    default sqrt(k) scaled
Loss:  N/A
Speed: N/A
Time:  00:00 elapsed  |  00:30 limit
ETA:   ~30 min
W&B:   TBD
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3_siso_official_check/logs_lobs5/lobs5_4187975.out
Purpose: verify RMSNorm fix + exp guard don't NaN on real Mamba3 (v1 accidentally ran GDN default)

Updated: 2026-04-22 UTC

Job:   4188163 (mamba3-siso-fix-smoke-v3)
User:  kangli.s5e
Step:  N/A (smoke v3 with BSZ=2/GPU to avoid 75m OOM)
Model: 75M Mamba3 SISO (d=1024, L=6, ssm=1024, BSZ=2/GPU global=16)
Data:  LOBS5 default
Infra: 2N / 8 GPU | contiguous
Loss:  N/A
Speed: N/A
Time:  00:00 elapsed  |  00:30 limit
ETA:   ~30 min
W&B:   TBD
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3_siso_official_check/logs_lobs5/lobs5_4188163.out
Purpose: 3rd attempt - v2 OOM'd at BSZ=10, try BSZ=2 to fit 2-node memory

Updated: 2026-04-22 UTC

Job:   4236845 (m3-78m-rmsfix-resume)
User:  aramis.s5e
Step:  resuming from 33480 → target ≥46050  (paper-step-match)
Model: 78M Mamba3 (RMSNorm fix active)
Data:  same as 4184212 (8 tickers × 4yr, test 2026-01)
Infra: 8N / 32 GPU | BSZ=4/gpu, gBSZ=128 | K=10
LR:    Muon 0.01 / SSM 5e-4 / WD 0.005
Loss:  resuming from s33480 (was healthy, no NaN)
Speed: ~5580 steps/h expected (matches initial run)
Time:  0:00 elapsed | 3:30 limit
ETA:   ~12570 steps in ~2:15h, walltime margin for re-init
W&B:   pending (will append to original yhmpnuk6 run via auto-resume?)
Log:   .../logs_lobs5/lobs5_4236845.out
Note:  Bug-fix BCNorm/out_norm at 33480 yielded WS-21 0.077 vs paper 0.044
       at 46050. Suspected step undertraining. Resuming to step-match.

Updated: 2026-04-23 01:30:00 UTC

Job:   4241687-4241702 (kbench-78m × {jax,cuda} × BSZ {1,2,4,8,12,16,24,32})
User:  aramis.s5e
Step:  - (infra bench, CURTAIL_EPOCHS=300)
Model: Mamba3 78M (d=1024, L=6, ssm=1024)
Data:  GOOG 2022
Infra: 2N / 8 GPU per job | 16 jobs total
LR:    5e-4 SSM, 0.01 Muon (irrelevant, no convergence check)
Purpose: CUDA-FFI kernel vs pure-JAX, sweep max BSZ + steady-state step time
Time:  30min wall-clock limit per job
Log:   logs_lobs5/training_<JOBID>_node0.log

Job:   4241703-4241716 (kbench-120m × {jax,cuda} × BSZ {1,2,4,8,12,16})
User:  aramis.s5e
Model: Mamba3 120M (d=1280, L=6)
Infra: 2N / 8 GPU per job | 12 jobs total
Purpose: same as above, 120M size

Job:   4241728-4241735 (kbench-200m × {jax,cuda} × BSZ {1,2,4,8})
User:  aramis.s5e
Model: Mamba3 200M (d=1664, L=6)
Infra: 4N / 16 GPU per job | 8 jobs total
Purpose: same as above, 200M size (tight memory)

Updated: 2026-04-23 (36 total infra-bench jobs)

========================================================================
Job:   4153400 (dense-stage3-1n-240step) [PRAP Stage 3 ✅ DONE]
User:  kangli.s5e
Result: exit 0:0, 24:00:15 elapsed (TIMEOUT with graceful save), 209/240 steps
  Training R_mean trajectory (every 30 steps):
    0-29:    +0.034 (early learning)
    30-59:   +0.007 (dip)
    60-89:   +0.016
    90-119:  +0.036 (recovery)
    120-149: +0.033
    150-179: +0.022
    180-209: +0.034 (strong end)
  Overall mean over 210 steps: +0.026
  Max R_mean ever: +0.180 (step 120-149)
  
  Checkpoints on disk: opt 300, 350, 400 (grpo 150/175/200)
  Backed up pre-training: opt200_grpo100 (1.0GB)
W&B:   https://wandb.ai/oxford-lob/lobs5-O9-grpo/runs/6asy28tp

========================================================================
Jobs:  4247690-4247693 (eval-O11-*) [PRAP §2 Final Eval]
User:  kangli.s5e
Status: 4 parallel eval jobs, each 1N × 3.5h, N_EVAL=5000, seed=42

  4247690 eval-O11-grpo100      opt200 backup   (PRAP pre-reg ckpt 1)
  4247691 eval-O11-grpo150      opt300 disk     (mid)
  4247692 eval-O11-grpo200      opt400 disk     (PRAP pre-reg ckpt 2)
  4247693 eval-O11-baseline     step 23356      (single-step baseline for fair comparison)

Test:   GOOG Jan 2026 28 days (/projects/s5e/quant/GOOG_2026_24tok_preproc, 43-col)
Protocol: eval_high_precision.batch, single seed eval per ckpt
Next: when all 4 done, compare IC@h=10 per ckpt vs baseline

Updated: 2026-04-23 06:00 UTC
========================================================================

Job:   4253441 (tar-kang-archive)
User:  kangli.s5e
Task:  Tar 14 dirs in /projects/s5e/public/quant_team/kang_archive/ into .tar.zst tarballs
Infra: 1N / 256 CPU | parallel=4, zstd-T64 per worker
Src:   /projects/s5e/public/quant_team/kang_archive/ (22M+ inodes, ~5TB bytes)
Out:   /projects/s5e/public/quant_team/kang_tarballs/
Time:  0:00 elapsed | ~2h est | 8:00:00 limit
Notes: Non-destructive (originals preserved). Deletes require user confirmation.
Log:   /projects/s5e/quant/logs_tar/tar_4253441.out

Updated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")

Job:   4255547 (tar-retry-4dirs)
User:  kangli.s5e
Task:  Retry tar for 4 Lustre-shutdown dirs: exp_O10c_ic_reward_rank, exp_stable_ema, exp_stable_lowlr, OLD_HOME_preMigration_snapshot_2025
Infra: 1N / 256 CPU | parallel=4, zstd-T64 per worker, --ignore-failed-read, --exclude=nid010235
Src:   /projects/s5e/public/quant_team/kang_archive/ (4 dirs, ~20M inodes total)
Out:   /projects/s5e/public/quant_team/kang_tarballs/
Time:  ~2h est | 4:00:00 limit
Log:   /projects/s5e/quant/logs_tar/tar_retry_4255547.out

Updated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")

Job:   4256515 (tar-O10c-rank)
User:  kangli.s5e
Task:  Tar exp_O10c_ic_reward_rank (7.36M inodes, ~1-2 TB est)
Infra: 1N / 256 CPU | zstd-T256 single tar, --ignore-failed-read
Time:  ~60-90 min est | 2:00:00 limit
Log:   /projects/s5e/quant/logs_tar/tar_tar-O10c-rank_4256515.out
Updated: 2026-04-24 12:44:45 UTC

Job:   4256516 (tar-stable-ema)
User:  kangli.s5e
Task:  Tar exp_stable_ema (3.17M inodes, ~500GB est)
Infra: 1N / 256 CPU | zstd-T256, --ignore-failed-read
Time:  ~30-50 min est | 1:00:00 limit
Log:   /projects/s5e/quant/logs_tar/tar_tar-stable-ema_4256516.out
Updated: 2026-04-24 12:44:45 UTC

Job:   4256517 (tar-stable-lowlr)
User:  kangli.s5e
Task:  Tar exp_stable_lowlr (212K inodes, ~50GB est)
Infra: 1N / 256 CPU | zstd-T256, --ignore-failed-read
Time:  ~10-20 min est | 0:30:00 limit
Log:   /projects/s5e/quant/logs_tar/tar_tar-stable-lowlr_4256517.out
Updated: 2026-04-24 12:44:45 UTC

Job:   4256518 (tar-OLD-HOME)
User:  kangli.s5e
Task:  Tar OLD_HOME_preMigration_snapshot_2025 (8.24M inodes)
Infra: 1N / 256 CPU | zstd-T256, --ignore-failed-read
Time:  ~30-60 min est | 1:00:00 limit
Log:   /projects/s5e/quant/logs_tar/tar_tar-OLD-HOME_4256518.out
Updated: 2026-04-24 12:44:45 UTC

Job:   4259782 (sqfs-O10c-rank)
User:  kangli.s5e
Task:  Squashfs archive exp_O10c_ic_reward_rank (7.36M inodes, ~988 GB)
Infra: 1N / 256 CPU | mksquashfs.static, zstd-3, -processors 256, -b 1M
Out:   /projects/s5e/public/quant_team/kang_tarballs/exp_O10c_ic_reward_rank.sqfs
Time:  ~30-90 min est | 2:00:00 limit
Notes: Switched from tar (stuck at 20 MB/s) to mksquashfs for parallel reads on 7.36M tiny files
Log:   /projects/s5e/quant/logs_tar/sqfs_O10c_rank_4259782.out
Updated: 2026-04-24 14:22:56 UTC

Job:   4265611 (tarP-O10c-rank)
User:  kangli.s5e
Task:  Parallel subdir tar of exp_O10c_ic_reward_rank (7.36M inodes, 988GB, 74 subdirs)
Infra: 1N / 256 CPU | xargs -P16 concurrent tars, each with zstd -T16
Out:   /projects/s5e/public/quant_team/kang_tarballs/exp_O10c_ic_reward_rank__*.tar.zst (~74 files)
Time:  ~30-90min est | 3:00:00 limit
Log:   /projects/s5e/quant/logs_tar/tarP_O10c_rank_4265611.out
Updated: 2026-04-24 15:27:05 UTC

Job:   4298768 (lz-baseline-8tk)
User:  aramis.s5e
Step:  N/A (one-shot encode+compress)
Model: N/A — LZ compression baseline only
Data:  8 tickers × Jan 2026 (v1 test corpus) | 26-tok encoding
Infra: 1N / 0 GPU | CPU-only (16 cores)
Purpose: Validate LZ baseline methodology against v1 fitted E ≈ 0.47 nats/token
Time:  ~1.5h budget
W&B:   N/A
Log:   /projects/s5e/quant/AlphaTrade/experiments/scaling_law_plots/logs/lz_baseline_4298768.out

Updated: 2026-04-25 (UTC)

Job:   4302149 (lz-baseline-8tk, resubmit preset=6)
User:  aramis.s5e
Step:  N/A (one-shot encode+compress)
Model: N/A — LZ compression baseline
Data:  8 tickers × Jan 2026 | 26-tok encoding
Infra: 1N / 0 GPU | 256 CPUs (parallel one-process-per-ticker)
LZMA:  preset=6 (no EXTREME) — ~10x faster than 9-EXTREME
Time:  ~30min budget
W&B:   N/A
Log:   /projects/s5e/quant/AlphaTrade/experiments/scaling_law_plots/logs/lz_baseline_4302149.out
Note:  Supersedes 4301750 (cancelled — preset=9 EXTREME too slow for NVDA)

Updated: 2026-04-25 (UTC)

Job:   4302877 (lz-baseline-8tk, numpy encoder)
User:  aramis.s5e
Step:  N/A
Model: N/A — LZ compression baseline
Data:  8 tickers × Jan 2026 | 26-tok encoding
Infra: 1N / 0 GPU | 256 CPUs (parallel one-process-per-ticker)
LZMA:  preset=6 streaming
Encoder: pure-numpy port (no JAX) — verified vs JAX in pre-flight
Time:  30min limit (expect ~10-15min wall)
W&B:   N/A
Log:   /projects/s5e/quant/AlphaTrade/experiments/scaling_law_plots/logs/lz_baseline_4302877.out
Note:  Supersedes 4302149 (cancelled — JAX dispatch overhead too slow for big tickers)

Updated: 2026-04-25 (UTC)

Job:   4306119 (L1-realdt-smoke)
User:  kangli.s5e
Step:  0/300  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: 10M Mamba3 (d=512, L=6, B=8, ssm=512) | ~10M params
Data:  GOOG 2022 + Jan 2023 test | TOKEN_MODE default
Infra: 2N / 8 GPU | BSZ=20/gpu, gBSZ=160 | Local Steps K=1
SSM:   mamba3 (REQUIRED — exp/L1-real-dt has plumbing only for Mamba3SSM)
LR:    default
Loss:  (pending)
Speed: (pending — XLA compile ~2-3min)
Time:  0:00 elapsed  |  ~25:00 remaining  |  30:00 limit
ETA:   ~30:00 total (smoke test, CURTAIL=300)
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_L1_real_dt/logs_lobs5/lobs5_4306119.out
Branch: exp/L1-real-dt @ 4ae50bb4 (real-Δt plumbing through Mamba3SSM)
Goal:  verify SSM dynamics don't NaN with real inter-arrival Δt clipped to [1e-6, 1.0]

Updated: 2026-04-25 13:46:01 UTC

Job:   4307139 (L1-realdt-smoke-v2)
User:  kangli.s5e
Step:  0/300  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0%
Model: 22.9M Mamba3 (d=512, L=6, B=8, ssm=512, n_heads=16, d_state=128)
Data:  GOOG 2022 + Jan 2023 test
Infra: 2N / 8 GPU | BSZ=4/gpu, gBSZ=32 (BSZ=20 OOM in v1 due to Mamba3 selective SSM heavier than S5)
SSM:   mamba3 (REQUIRED)
LR:    default
Loss:  (pending)
Speed: (pending)
Time:  0:00 elapsed  |  ~30:00 limit
W&B:   (pending)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_L1_real_dt/logs_lobs5/lobs5_4307139.out
Branch: exp/L1-real-dt @ 4ae50bb4
Goal:  v1 OOM 232 GiB at BSZ=20 → v2 BSZ=4 smaller mem to isolate whether Δt path is correct
Prev:  4306119 OOM 283.52 GiB single tensor allocation (RESOURCE_EXHAUSTED step 2)

Updated: 2026-04-25 14:18:13 UTC

Job:   4332890 (m3-78m-nobcnorm-8n)
User:  aramis.s5e
Step:  0/?  [waiting]  0%
Model: Mamba3 78M NoBCNorm (d=1024, L=6, ssm=1024, blocks=32) | ~78M params
Data:  8 tickers × 4yr (2022-2025), TOKEN_MODE=26tok, DATA_ROOT=lob_preproc | test: 2026-01
Infra: 8N / 32 GPU | BSZ=4/gpu, gBSZ=128 | OPT=muon, SSM_LR=5e-4
LR:    5e-4
Loss:  pending
Speed: pending
Time:  0:00 elapsed | 6:00 limit
ETA:   pending
W&B:   pending
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3_NoBCNorm/logs_lobs5/lobs5_4332890.out
Notes: BCNorm removed (test no-BCNorm hypothesis). Branch: exp/R1-Mamba3-NoBCNorm @ 258c1fb0
       DATA_ROOT fixed (lob_preproc_26tok dir was deleted, consolidated into lob_preproc)

Updated: 2026-04-25 17:42:45 UTC

Job:   4332890 CANCELLED
User:  aramis.s5e
Reason: Catastrophic loss divergence ~step 4k (loss 0.7 → 14+, never recovered).
Cause: Removing B_norm/C_norm entirely also removed their learnable scale params.
       The "buggy" run effectively had a learnable per-feature scale on B/C
       (since the inert RMSNorm reduced to scale * (B/~11.3)). Removing the
       layer outright gives the SSM unconstrained B/C magnitudes, which
       diverge with the standard recipe.
Decision: Leaving the NoBCNorm direction. Real RMSNorm (rmsfix) trains
          stably and has a category-by-category tradeoff vs buggy
          (uncond marginals: buggy wins; cond/context: rmsfix wins or ties).
          That's defensible as a paper baseline.

Cancelled: 2026-04-25 18:30 UTC

================================================================
Phase 0 SMOKE TESTS — Scaling Law v2 (NeurIPS 2026)
Submitted: 2026-04-25, by aramis.s5e
Plan: scaling_law_plots/SCALING_LAW_PLAN_V2.md
Mode: CURTAIL_EPOCHS=300, EPOCHS=1, NO_VALIDATION=True, --time=00:45:00
Optimizer: OPT_CONFIG=muon MUON_LR=0.01 SSM_LR_BASE=5e-4 WD=0.005
Tickers: default 8-ticker (smoke); Phase 1 will switch to S&P 500
Pass criterion: reach step 300 without NaN/OOM/divergence

Mamba3 (13 sizes, head_dim=64, L=6):
  4339417 smoke_m3_0p2M  (d=64,   2N, BSZ=16)
  4339418 smoke_m3_1M    (d=128,  2N, BSZ=16)
  4339419 smoke_m3_4M    (d=192,  2N, BSZ=16)
  4339420 smoke_m3_6M    (d=256,  2N, BSZ=16)
  4339421 smoke_m3_10M   (d=320,  2N, BSZ=16)
  4339422 smoke_m3_14M   (d=384,  2N, BSZ=16)
  4339423 smoke_m3_23M   (d=512,  4N, BSZ=8)
  4339424 smoke_m3_46M   (d=768,  4N, BSZ=8)
  4339425 smoke_m3_78M   (d=1024, 8N, BSZ=4)
  4339426 smoke_m3_120M  (d=1280, 16N, BSZ=2)
  4339427 smoke_m3_200M  (d=1664, 16N, BSZ=2)
  4339428 smoke_m3_350M  (d=2048, 32N, BSZ=1)
  4339429 smoke_m3_500M  (d=2304, 32N, BSZ=1)

Transformer (11 sizes, head_dim=64, L=6):
  4339430 smoke_tf_0p2M  (d=64,   2N, BSZ=16)
  4339431 smoke_tf_1M    (d=128,  2N, BSZ=16)
  4339432 smoke_tf_4M    (d=192,  2N, BSZ=16)
  4339433 smoke_tf_6M    (d=256,  2N, BSZ=16)
  4339434 smoke_tf_10M   (d=320,  2N, BSZ=16)
  4339435 smoke_tf_14M   (d=384,  2N, BSZ=16)
  4339436 smoke_tf_23M   (d=512,  4N, BSZ=8)
  4339437 smoke_tf_46M   (d=768,  4N, BSZ=8)
  4339438 smoke_tf_78M   (d=1024, 8N, BSZ=4)
  4339439 smoke_tf_120M  (d=1280, 16N, BSZ=2)
  4339440 smoke_tf_200M  (d=1664, 16N, BSZ=2)

Updated: 2026-04-25 (Phase 0 smoke launched, 24 jobs in queue)
================================================================

================================================================
Phase 0 SMOKE RETRY (2026-04-25)
Submitted: 2026-04-25, by aramis.s5e
Reason: 6 m3 mid-range OOM (BSZ too high), 11 TF perm error on log dir

Mamba3 retries at lower BSZ (gBSZ relaxed for smoke):
  4341076 smoke2_m3_4M    (d=192,  2N, BSZ=4)
  4341077 smoke2_m3_6M    (d=256,  2N, BSZ=4)
  4341078 smoke2_m3_10M   (d=320,  2N, BSZ=4)
  4341079 smoke2_m3_14M   (d=384,  2N, BSZ=4)
  4341080 smoke2_m3_23M   (d=512,  4N, BSZ=2)
  4341081 smoke2_m3_46M   (d=768,  4N, BSZ=2)

Transformer retries (logs → aramis_smoke_logs/, perm fix):
  4341082 smoke2_tf_0p2M  (d=64,   2N, BSZ=16)
  4341083 smoke2_tf_1M    (d=128,  2N, BSZ=16)
  4341084 smoke2_tf_4M    (d=192,  2N, BSZ=16)
  4341085 smoke2_tf_6M    (d=256,  2N, BSZ=16)
  4341086 smoke2_tf_10M   (d=320,  2N, BSZ=16)
  4341087 smoke2_tf_14M   (d=384,  2N, BSZ=16)
  4341088 smoke2_tf_23M   (d=512,  4N, BSZ=8)
  4341089 smoke2_tf_46M   (d=768,  4N, BSZ=8)
  4341090 smoke2_tf_78M   (d=1024, 8N, BSZ=4)
  4341091 smoke2_tf_120M  (d=1280, 16N, BSZ=2)
  4341092 smoke2_tf_200M  (d=1664, 16N, BSZ=2)

Updated: 2026-04-25
================================================================

================================================================
Phase 0 TF SMOKE — Round 3 (2026-04-25)
Submitted: 2026-04-25, by aramis.s5e
Reason: Round 2 TF jobs hit perm error on node_wrapper.sh (mode 600, owned kangli.s5e)
Fix: copy of R1 wrapper into scaling_law_plots/tf_smoke_workdir/ + sbatch from there with WORKDIR=$O8_dir override

  4342202 smoke3_tf_0p2M  (d=64,   2N, BSZ=16)
  4342203 smoke3_tf_1M    (d=128,  2N, BSZ=16)
  4342204 smoke3_tf_4M    (d=192,  2N, BSZ=16)
  4342205 smoke3_tf_6M    (d=256,  2N, BSZ=16)
  4342206 smoke3_tf_10M   (d=320,  2N, BSZ=16)
  4342207 smoke3_tf_14M   (d=384,  2N, BSZ=16)
  4342208 smoke3_tf_23M   (d=512,  4N, BSZ=8)
  4342209 smoke3_tf_46M   (d=768,  4N, BSZ=8)
  4342210 smoke3_tf_78M   (d=1024, 8N, BSZ=4)
  4342211 smoke3_tf_120M  (d=1280, 16N, BSZ=2)
  4342212 smoke3_tf_200M  (d=1664, 16N, BSZ=2)

Updated: 2026-04-25
================================================================

================================================================
Phase 0 TF SMOKE — Round 4 (2026-04-26)
Submitted: 2026-04-26, by aramis.s5e
Reason: Round 3 hit cuDNN flash-attn dtype mismatch (default fp32 vs required bf16)
Fix: --dtype passthrough added to my wrapper; mirrors O8 production defaults
       (DTYPE=bfloat16, USE_FLASH=True, REMAT=False, D_FF=0)

  4344807 smoke4_tf_0p2M  (d=64,   2N, BSZ=16)
  4344808 smoke4_tf_1M    (d=128,  2N, BSZ=16)
  4344809 smoke4_tf_4M    (d=192,  8N, BSZ=4)   ← BSZ-recalibrated
  4344810 smoke4_tf_6M    (d=256,  8N, BSZ=4)
  4344811 smoke4_tf_10M   (d=320,  8N, BSZ=4)
  4344812 smoke4_tf_14M   (d=384,  8N, BSZ=4)
  4344813 smoke4_tf_23M   (d=512,  16N, BSZ=2)
  4344814 smoke4_tf_46M   (d=768,  16N, BSZ=2)
  4344815 smoke4_tf_78M   (d=1024, 8N, BSZ=4)
  4344816 smoke4_tf_120M  (d=1280, 16N, BSZ=2)
  4344817 smoke4_tf_200M  (d=1664, 16N, BSZ=2)

Cancelled (round 3): 4342210, 4342211, 4342212 (would have failed same dtype bug)

Updated: 2026-04-26
================================================================

================================================================
Phase 0 TF SMOKE — Round 5 (2026-04-26)
Submitted: 2026-04-26, by aramis.s5e
Reason: Round 4 over-conservative; re-submit at v1-extrapolated BSZ to find true ceiling
v1 reference (transformer-scaling-law wandb): d=256-384 BSZ=16, d=512-1024 BSZ=8

  4345057 smoke5_tf_4M    (d=192,  2N, BSZ=16) — extrap from v1 d=256 BSZ=16
  4345058 smoke5_tf_120M  (d=1280, 8N, BSZ=4)  — extrap from v1 d=1024 BSZ=8
  4345059 smoke5_tf_200M  (d=1664, 8N, BSZ=4)  — push from BSZ=2 (round 4); validates ceiling

Cancelled (round 4 conservative): 4344809, 4344816, 4344817

Updated: 2026-04-26
================================================================

================================================================
Phase 0 TF SMOKE — Round 6 (2026-04-26)  [BSZ=32 ceiling test]
Submitted: 2026-04-26, by aramis.s5e
Reason: Push 0.2M-4M to BSZ=32 (1N×4GPU = gBSZ=128). If OOM, BSZ=16 is known-good fallback.

  4345077 smoke6_tf_0p2M  (d=64,   1N, BSZ=32)
  4345079 smoke6_tf_1M    (d=128,  1N, BSZ=32)
  4345080 smoke6_tf_4M    (d=192,  1N, BSZ=32)

Cancelled (round 4/5 BSZ=16 versions): 4344807, 4344808, 4345057

Updated: 2026-04-26
================================================================

Job:   4358918 (fl-m3-0p2M-2n)
User:  aramis.s5e
Step:  TBD
Model: Mamba3 0.2M (d=64, L=6, BSZ=16, ssm=64) | profile-only
Data:  8 tickers × 4yr (2022-2025) | test: 2026-01
Infra: 2N / 8 GPU | BSZ=16/gpu, gBSZ=128 | DiLoCo K=10
LR:    Muon kernel=0.01, AdamW aux=8.0e-3
Loss:  N/A (profile run, CURTAIL=300)
Speed: TBD
Time:  TBD elapsed | 25min limit
ETA:   ~10-15 min wall (dmon-instrumented smoke for §4.5 FLOPs/tok recalibration at production topology)
W&B:   https://wandb.ai/oxford-lob/mamba3-flops-profile-v2
Log:   /projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3/logs_lobs5/training_4358918_node0.log

Updated: 2026-04-26 12:09:00 UTC

Job:   4359693 (fl-m3-0p2M-2n-validate)
User:  aramis.s5e
Step:  TBD
Model: Mamba3 0.2M (d=64, L=6, BSZ=16, ssm=64) | profile-only with SKIP_TEST_EVAL=True
Data:  8 tickers × 4yr | 26tok | gBSZ=128 / 8 GPU
Infra: 2N / 8 GPU | DiLoCo K=10
LR:    Muon kernel=0.01, AdamW aux=8.0e-3
Loss:  N/A
Speed: TBD (4359025: 12 it/s steady)
Time:  TBD elapsed | 10 min limit
ETA:   ~3-5 min wall (no test eval phase)
W&B:   https://wandb.ai/oxford-lob/mamba3-flops-profile-v2
Log:   /projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3/logs_lobs5/training_4359693_node0.log

Updated: 2026-04-26 12:30:00 UTC

Job:   4384545 (zst-smoke-meta)
User:  kangli.s5e
Step:  smoke test (CURTAIL_EPOCHS=300)
Model: 10M (d=512, L=6, B=8, ssm=512)
Data:  META × ~30 days (compressed to /local/$USER/zst_smoke_4384545/META/*.npy.zst on the fly)
Infra: 1N / 4 GPU | BSZ=2/gpu, gBSZ=8 | 64 cpu
Speed: TBD
Time:  ~30 min compress + 30 min train, --time=01:30:00
ETA:   verify zst-only reader runs end-to-end on real training pipeline
W&B:   TBD (will fill from log)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/logs_lobs5/zst_smoke_4384545.out

Updated: 2026-04-27 00:35:00 UTC

Job:   4384594 (zst-smoke-meta retry, fix: writable scratch path)
User:  kangli.s5e
Step:  smoke test (CURTAIL_EPOCHS=300)
Model: 10M (d=512, L=6, B=8, ssm=512)
Data:  META × ~30 days (compress to first writable: $TMPDIR or /local/user/<uid> or /scratch)
Infra: 1N / 4 GPU | BSZ=2/gpu, gBSZ=8 | 64 cpu
Time:  --time=01:30:00
W&B:   TBD
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/logs_lobs5/zst_smoke_4384594.out
Prev:  4384545 (failed: /local/$USER/ Permission denied at 0:43s)

Updated: 2026-04-27 01:38:00 UTC

Job:   4393167 (zst-compress-MO-HAL)
User:  kangli.s5e
Step:  Production rollout pair 1/238 (sp500 native tickers)
Model: N/A (CPU compress)
Data:  MO (137G, 2060 npy) + HAL (195G, 2048 npy) = 332 GB / 4108 files
Infra: 1N / 64 cpu / 1 gpu (unused) | xargs -P 32 zstd -3 per-file pipeline
Pipeline: zstd --keep -> verify -> rm .npy
Time:  --time=00:30:00, est 5-15min wall
ETA:   ~15min total (Lustre I/O bound at ~5 GB/s aggregate)
W&B:   N/A
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/logs_lobs5/zst_compress_4393167.out
Goal:  Free ~312 GB Lustre quota (332 GB raw -> ~20 GB zst)

Updated: 2026-04-27 02:58:00 UTC

Job:   4393180 (zst-compress-MO-HAL-retry)
User:  kangli.s5e
Step:  Production rollout pair 1/238 retry (267 files remaining from 4393167)
Model: N/A (CPU compress)
Data:  MO (9 npy left) + HAL (258 npy left) ≈ 22 GB raw remaining
Infra: 1N / 64 cpu / 1 gpu (unused) | xargs -P 32 zstd -3 per-file pipeline
Pipeline: zstd --keep -> verify -> rm .npy (same script auto-skips .zst)
Time:  --time=00:30:00, est 2-3 min wall
ETA:   should fully complete with ~316 GB headroom freed by 4393167
W&B:   N/A
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/logs_lobs5/zst_compress_4393180.out
Goal:  Finish MO+HAL compression, verify 0 .npy remaining, total ~316+ GB freed
Prev:  4393167 (FAILED exit 1, but 3841/4108 files compressed, 317 GB freed; safety pipeline preserved 267 .npy on verify_fail)

Updated: 2026-04-27 03:13:00 UTC

Job:   4393238 (zst-compress-DVN-DAL → DVN only due to --export comma split)
User:  kangli.s5e
Step:  Production rollout pair 2 split: DVN compressing now, DAL queued separately
Model: N/A (CPU compress)
Data:  DVN (247G, 2044 npy) — DAL (226G, 2058 npy) submitting separately
Infra: 1N / 64 cpu / 1 gpu (unused) | xargs -P 32 zstd -3 per-file pipeline
Pipeline: zstd --keep -> verify -> rm .npy
Time:  --time=00:30:00, est 3-5 min wall
W&B:   N/A
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/logs_lobs5/zst_compress_4393238.out
Note:  --export=ALL,TICKERS=DVN,DAL silently dropped DAL (SLURM treats comma as var-pair separator). Submitting DAL as separate job 4393NNN.

Updated: 2026-04-27 03:22:30 UTC

Job:   4393240 (zst-compress-DAL)
User:  kangli.s5e
Step:  Production rollout pair 2 part 2: DAL standalone (--export TICKERS=DAL, no comma)
Model: N/A (CPU compress)
Data:  DAL (226G, 2058 npy)
Infra: 1N / 64 cpu / 1 gpu (unused)
Time:  --time=00:30:00, est 3 min wall
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/logs_lobs5/zst_compress_4393240.out

Updated: 2026-04-27 03:23:00 UTC

Job:   4393242 (zst-compress-WDC)
User:  kangli.s5e
Step:  Production rollout pair 3 part 1
Data:  WDC (231G, 2064 npy) — sp500 native
Infra: 1N / 64 cpu / 1 gpu (unused)
Time:  --time=00:30:00, est <1 min wall (per pair-2 baseline 19s)
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/logs_lobs5/zst_compress_4393242.out

Job:   4393243 (zst-compress-SYF)
User:  kangli.s5e
Step:  Production rollout pair 3 part 2
Data:  SYF (140G, 2046 npy) — sp500 native
Infra: 1N / 64 cpu / 1 gpu (unused)
Time:  --time=00:30:00, est <1 min wall
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/logs_lobs5/zst_compress_4393243.out

Updated: 2026-04-27 03:27:00 UTC

Job:   4393248 (zst-smoke-val)
User:  kangli.s5e
Step:  smoke + val: verify .zst reader works in full train+val flow
Model: 10M (d=512, L=6, B=8, ssm=512)
Data:  MO + DAL .zst (compressed earlier this session) | TRAIN_DATE_RANGE=2022-2025, TEST_DATE_RANGE=2026-01
Infra: 1N / 4 GPU | PER_GPU_BSZ=2, CURTAIL_EPOCHS=500
DataRoot: /lus/lfs1aip2/projects/s5e/lob_preproc_sp500 (override from default lob_preproc/)
Time:  --time=00:30:00, est ~10 min wall
W&B:   TBD
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/logs_lobs5/lobs5_4393248.out
Goal:  Confirm .zst reader handles val set (test_dir glob + decompression at val time)
Prev:  4384619 (smoke without val, succeeded; this adds val)

Updated: 2026-04-27 03:32:00 UTC

Job:   4393280 (zst-rollout job array [1-477%15])
User:  kangli.s5e
Step:  Production rollout: ALL remaining 477 sp500 native tickers
Model: N/A (CPU compress)
Data:  477 tickers, est ~50 TB raw -> ~3 TB zst
Infra: 477 × (1N / 64 cpu / 1 gpu) | xargs -P 32 per-task | --array=1-477%15 (15 concurrent)
Pipeline: zstd --keep -> verify -> rm .npy
Time:  --time=00:30:00 per task, est ~1 hour total wall
ETA:   ~30-90 min wall (depends on FairShare throttling)
W&B:   N/A
Log:   /lus/.../logs_lobs5/zst_rollout_4393280_<task>.out (one per task)
Goal:  Free ~47 TB Lustre (~50 TB raw -> ~2.7 TB zst at avg ratio 19x)

Updated: 2026-04-27 04:08:00 UTC

Job:   4396372 (zst-smoke-1m)
User:  aramis.s5e
Step:  -/300 [smoke test, will run to curtail]
Model: 1M Mamba3-SISO (d=128, L=4, B=4, ssm=128) | ~1M params
Data:  META 30 days (compressed to /local/$USER/zst_smoke_4396372/META)
Infra: 1N / 4 GPU | BSZ=2/gpu, gBSZ=8 | Local Steps K=N/A (single-node)
LR:    default
Loss:  -
Speed: -
Time:  0:00 elapsed | 0:30 limit
ETA:   ~10-15 min total (compress 1-2 min + XLA compile 3-5 min + 300 steps 1-2 min)
W&B:   pending
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/logs_lobs5/zst_smoke_4396372.out
Purpose: throughput smoke for zstd .npy.zst dataloader path (cherry-picked compression chain into exp/R1-Mamba3)

Updated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")

Job:   4396474 (m3-0p2M-s5-2n)
User:  aramis.s5e
Step:  -/300 [smoke for Phase 1+2 launch readiness]
Model: 0.2M Mamba3-SISO (d=64, L=6, blocks=1, ssm=64) | seed=5
Data:  488 SP500 tickers via /lob_preproc_sp500/ | TICKERS_FILE=tickers.txt
Infra: 2N / 8 GPU | BSZ=16/gpu, gBSZ=128 | LOCAL_STEPS_K=10
LR:    Muon kernel 0.01, SSM 8.0e-3 (µP-scaled), AdamW aux µP-scaled
Loss:  -
Speed: -
Time:  0:00 elapsed | 0:45 limit (CURTAIL=300)
ETA:   ~10-15 min (XLA compile + 300 steps)
W&B:   pending — neurips-mamba3-scaling-runs
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3/logs_lobs5/lobs5_4396474.out
Purpose: validate Phase 1+2 launch path (488 tickers, .npy.zst dataloader, pooled Jan+Feb test split, Muon-AdamW optimizer)

Updated: 2026-04-27 07:48:43 UTC

---
Job:   4401642 (zst-goog22to26-raw)
User:  aramis.s5e
Task:  Compress /projects/s5e/GOOG_22_26/raw_csv (187 GB, 1055 CSVs) at zstd --ultra -22
Infra: 1N CPU-only (1 GPU min by cluster) | 64 cpus, parallel=24
Time:  4h limit
Log:   /projects/s5e/GOOG_22_26/compress_logs/zst_4401642.out
Script: /projects/s5e/GOOG_22_26/compress_raw_csv.batch

Updated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")

Job:   4402408 (r1g-smoke-1n-cuda)
User:  kangli.s5e
Step:  smoke (CURTAIL_EPOCHS=300, ~50 steps actual)
Model: 75M Mamba3 (D=1024 L=12 SSM=1024) | Mamba3 d_state=128
Data:  default (LOBSTER 8 tickers x 4yr)
Infra: 1N / 4 GPU | BSZ=4/gpu | tp_size=1 (default)
LR:    default 5e-4
Loss:  TBD
Speed: TBD
Time:  TBD
ETA:   30min limit
W&B:   TBD
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/logs_lobs5/lobs5_4402408.out

Updated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")

Job:   4402417 (r1g-smoke-1n-jax)
User:  kangli.s5e
Step:  smoke (CURTAIL_EPOCHS=300, ~50 steps actual)
Model: 75M Mamba3 (D=1024 L=12 SSM=1024) | Mamba3 d_state=128, MAMBA3_USE_CUDA=False
Data:  default (LOBSTER 8 tickers x 4yr)
Infra: 1N / 4 GPU | BSZ=4/gpu | tp_size=1 (default)
LR:    default 5e-4
Loss:  TBD
Speed: TBD
Time:  TBD
ETA:   30min limit
W&B:   TBD
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/logs_lobs5/lobs5_4402417.out

Updated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")

Job:   4402421 (r1g-smoke-2n-cuda)
User:  kangli.s5e
Step:  smoke (CURTAIL_EPOCHS=300, ~50 steps actual)
Model: 75M Mamba3 (D=1024 L=12 SSM=1024) | Mamba3 d_state=128
Data:  default (LOBSTER 8 tickers x 4yr)
Infra: 2N / 8 GPU | BSZ=4/gpu | tp_size=2 (TP across 2 nodes)
LR:    default 5e-4
Loss:  TBD
Speed: TBD
Time:  TBD
ETA:   30min limit
W&B:   TBD
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/logs_lobs5/lobs5_4402421.out

Updated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")


Job:   4411188 (m3-0p2M-s5-2n)
User:  aramis.s5e
Step:  PENDING (Priority)
Model: M3 0p2M (d=64, L=6, blocks=1, ssm=64)
Data:  488 tickers × 2022-2025 | test: 2026-01-01 to 2026-01-31
Infra: 2N / 8 GPU | BSZ=16/gpu, gBSZ=128 | N_WORKERS=4
LR:    Muon kernel=0.01, AdamW aux=8.0e-3
Notes: First production-config run after OOM fix (commit 03d5d543).
       Validates n_cache_files=250→8 + per-ticker n_cache=0 + per-ticker num_workers=0.
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_R1_Mamba3/logs_lobs5/training_4411188_node0.log

Updated: 2026-04-28 03:24:38 UTC

Job:   4422301 (r1g-smoke-1n-cuda)
User:  kangli.s5e
Step:  smoke (CURTAIL_EPOCHS=300, ~50 steps)
Model: 75M Mamba3 (D=1024 L=6 SSM=1024) | d_state=128, FFI default path
Data:  TOKEN_MODE=24tok, lob_preproc/ (8 tickers)
Infra: 1N / 4 GPU | BSZ=4/gpu | tp_size=1
LR:    default 5e-4
Loss:  TBD
Speed: TBD
Time:  PENDING -> ?
ETA:   30min limit
W&B:   TBD
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/logs_lobs5/lobs5_4422301.out

Updated: 2026-04-27 19:39 UTC

Job:   4422302 (r1g-smoke-1n-jax)
User:  kangli.s5e
Step:  smoke override --mamba3_use_cuda=False (JAX path)
Model: 75M Mamba3 | same as above but MAMBA3_USE_CUDA=False
Data:  TOKEN_MODE=24tok
Infra: 1N / 4 GPU | BSZ=4/gpu
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/logs_lobs5/lobs5_4422302.out

Updated: 2026-04-27 19:39 UTC

Job:   4422303 (r1g-smoke-2n-cuda)
User:  kangli.s5e
Step:  smoke multinode (TP_SIZE=2)
Model: 75M Mamba3 | FFI multinode dispatch
Data:  TOKEN_MODE=24tok
Infra: 2N / 8 GPU | BSZ=4/gpu | DP=4 x TP=2
Log:   /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/logs_lobs5/lobs5_4422303.out

Updated: 2026-04-27 19:39 UTC

Job:   4485950 (sqfs-2023-07)
User:  aramis.s5e
Type:  squashfs-build (no GPU, 1N/72-core)
Month: 2023-07 (902 GB source → ~900 GB shard expected)
Index: in-shard index.json baked in at build time
Time:  90min limit
Log:   /lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/scaling_law_plots/squashfs/logs/sqfs_build_4485950.out

Updated: 2026-05-08 16:55:00 UTC

Job:   4488641 (sqfs-2023-01-zstd9)
User:  aramis.s5e
Type:  squashfs-build (no GPU, 1N/72-core)
Month: 2023-01 (1.1 TB source → expect ~550-700 GB compressed shard)
Index: in-shard index.json baked in
Comp:  zstd-9 (A/B test vs uncompressed 2023-07/2026-01)
Time:  90min limit (zstd build will be slower than no-compression)
Log:   /lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/scaling_law_plots/squashfs/logs/sqfs_build_4488641.out

Updated: 2026-05-08 17:55:00 UTC

Job:   4490542 (sqfs-2023-01-lz4hc)
User:  aramis.s5e
Type:  squashfs-build (no GPU, 1N/72-core)
Month: 2023-01 (1.1 TB source → expect ~700-900 GB lz4hc shard)
Index: in-shard index.json baked in
Comp:  lz4hc (zstd unavailable in mksquashfs.static 4.6.1)
Note:  Replaces failed 4488641 which tried zstd9 and got
       "Compressor zstd is not supported"
Time:  90min limit
Log:   /lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/scaling_law_plots/squashfs/logs/sqfs_build_4490542.out

Updated: 2026-05-08 18:05:00 UTC

Job:   4494753 (sqfs-2023-09-lz4)
User:  aramis.s5e
Type:  squashfs-build (lz4 default, 1N/72-core)
Month: 2023-09 (809 GB source → expect ~100 GB shard)
Index: in-shard index.json
Time:  90min limit
Log:   /lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/scaling_law_plots/squashfs/logs/sqfs_build_4494753.out

Updated: 2026-05-09 00:30:00 UTC

Job:   4495189 (sqfs-array, 46-task array, %4 concurrent)
User:  aramis.s5e
Type:  squashfs build array (lz4, 1N/72-core per task)
Tasks: 46 months × ~17 min each = ~3.5h total wall at %4
Pipeline: build_month_shard → verify_shard → delete-source
Order: smallest source first (2023-04 → 2024-01)
Time:  60min limit per task
Logs:  /lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/scaling_law_plots/squashfs/logs/array_4495189_*.out

Updated: 2026-05-09 01:25:00 UTC

Job:   4495189 (sqfs-array) — COMPLETED
End:   2026-05-09
Result: 46/46 array tasks succeeded, 0 failures
Total: 50 shards built (incl. 4 from before), 9.7 TB on Lustre
Quota: started 197.9 TB, ended 141.7 TB, freed ~56 TB
Pipeline: build → verify → delete-source per task

Updated: 2026-05-09 (array end)

Job:   4498658 (sqfs-multi-N48) — COMPLETED
End:   2026-05-09
Result: exit 0:0, 17:15 wall, full pipeline (train+val+per-ticker test) green
Coverage: 48 shards mounted, 466-ticker × 4-year corpus = 4.97M training sequences
MFU:   10.6% / 420 TFLOPS @ step 300 (Mamba3 0.2M model, BSZ 64, 4 GPUs)
Tag:   first multi-shard, multi-month production smoke
W&B:   https://wandb.ai/oxford-lob/mamba3-squashfs-multi/runs/8plu95a3

Updated: 2026-05-09 (multi-shard pipeline validated)

Job:   4498917 + 4498918 (parallel rebuild)
User:  aramis.s5e
Type:  rebuild_from_shard (FUSE source → lz4 + in-shard index)
Months: 2026-01 (1.6 TB → ~200 GB), 2023-07 (968 GB → ~120 GB)
Recovers: ~2.3 TB after old inode released by unmount
Time:  90min limit

Updated: 2026-05-09 08:30 UTC
