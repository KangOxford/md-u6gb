# Step 2 adaptation launcher — three defects cleared, one open, no run yet completed

Session of 2026-09-05/06. Four attached steps: three failed, and **the fourth is training**.
The launcher had **never produced a live step before this session** (R1-F4), so what follows
is a working launcher plus the record of what each failure established.

```
6324130.2071  cl-adapt-early-s42   FAILED   1:0   00:00:51   ticker coverage
6324130.2073  cl-adapt-late-s42    FAILED   1:0   00:00:51   ticker coverage
6324130.2115  cl-adapt-late2-s42   FAILED   1:0   00:00:53   CUDA_ERROR_NO_DEVICE
6324130.2113  cl-adapt-early2-s42  RUNNING        00:03:05   Epoch 0, steps advancing
```

## Durable paths

| what | path |
|---|---|
| launcher | `/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/code/attach_adaptation.sh` |
| pinned ticker set | `/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/results/tickers_2024-08.txt` |
| coverage check | `/lus/lfs1aip2/projects/public/u6gb/tasks/continual_learning/results/ticker_coverage_2024-08.json` |
| node-local run logs | `nid010851:/tmp/kangli.u6gb/sigma0/cl_probe_logs/cl-adapt-{early2,late2}-s42/` — **node-local, lost when the allocation ends** |
| W&B | `oxford-lob/sigma0-continual` runs `bxk0lokm` (early), `ttvvroco` (late2) |

## Defect 1 — the mount root, CLEARED with evidence

`node_wrapper.sh:342` blanks `SQUASHFS_MULTI_MOUNT_ROOT` unconditionally, so the unique root
exported by the launcher never reached the code and paired members collided. It cannot be
fixed by editing that file: 45 named steps belonging to other sessions were running against
it, and editing an executing script is how 14 workers once exited 127.

Routed around instead: `node_wrapper.sh:13` honours an inherited `SIGMA0_JOB_TMPDIR` and
line 19 exports it as `TMPDIR`, which is what line 370's mount-root default is built from.

**Evidence it worked** — from the node-local log of the first attempt:

```
[*] Loaded data index from /tmp/kangli.u6gb/sigma0/cl_probe_early/sp500_squashfs_6324130_0/2024-08/index.json: 20069 files cached
```

`cl_probe_early` is the per-member root. R1-F4 is cleared by a run, not by an argument.

## Defect 2 — ticker coverage, CLEARED; and the first coverage check I ran was vacuous

The first two members died at 51 s on
`AssertionError: No paired message+book files in index for BAC`.

**A correction.** My first coverage check printed `shard has 482; request had 0; missing from
shard: []` — the glob for the config matched nothing, so the request set was **empty and the
check was trivially true**. It established nothing. Redone against the file the launcher
actually reads:

| | size | provenance |
|---|---:|---|
| request universe | **488** | `/lus/lfs1aip2/projects/public/u6gb/sigma-0/configs/train/dfm_smoke_1gpu.yaml`, key `env_TICKERS` |
| shard universe | **482** | `.../lob_preproc_sp500_squashfs/shard_2024-08.squashfs::index.json`, 20,069 files |
| requested, absent from shard | **6** | `BAC, EXE, PSKY, Q, SNDK, XYZ` |
| in shard, not requested | 0 | the shard is a strict subset |
| trainable intersection | **482** | |

The launcher now reads the pinned 482 from one file both members share. A matched pair must
train on identical tickers, so deriving the list per member would be a second defect even if
each derivation succeeded.

## Defect 3 — a false fix of mine, reverted

I added a `SQUASHFS_DIR` override with the rationale "SQUASHFS_DIR was never set here". **That
was wrong**: line 82 already set it, to a path that exists and holds 50 shards. The edit and
its rationale were reverted before relaunch. Had it stayed, it would have been a duplicate
definition carrying a false explanation for a failure it did not cause.

## Defect 4 — OPEN

Both relaunched members were placed on **one node**, each requesting `--gres=gpu:1` with
`--gpu-bind off`. **One of the two got the GPU and is training**; the other died with

```
RuntimeError: cuInit(0) failed: CUDA_ERROR_NO_DEVICE
```

That asymmetry is the diagnosis: two `--exact` steps contend for per-step GPU allocation on
one node and the second gets nothing, which is why the first attempt (one member per node)
reached the ticker assertion instead. The untested fix is one member per node, or a single
step holding `--gres=gpu:4` with `CUDA_VISIBLE_DEVICES` selecting per member.
**Not attempted: no new GPU run is authorised.**

## What is not established

**Nothing is known about early-versus-late plasticity**, because that question needs a *pair*
and only the early member is running. A single member at a curtailed 200-step budget answers
no comparison; what it establishes is that the launcher works, which is what four attempts
were spent on.

The running member reached `Epoch 0, Batch 0` with `Device cuda:0 Used: 702.39 MB` and step
times falling from 117.8 s to 26.9 s as compilation amortised. Its checkpoints go to
`$SIGMA0/checkpoints_cl_probe`; **its logs are node-local on nid010851 and die with the
allocation**, which had 1:54 left at launch.
