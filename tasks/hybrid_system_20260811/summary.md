# Hybrid inference supervisor summary

State: **complete with K3 capacity gate**  
Updated: `2026-08-11T17:59:07Z`

The exact Notion target has been fetched and converted into four isolated model tasks. Current primary-source revisions and checkpoint byte sizes are pinned. A live read-only probe found eight empty GPUs inside allocation `5980502`, alongside unrelated GPU-resident processes that must remain untouched.

The Jamba2 3B smoke runtime and safety launcher are committed. A task-local 4.6 GB environment was successfully created without changing the shared Miniforge installation; its exact local package set is recorded in `01_jamba2_3b_inference/requirements.lock`.

Recommended next action: download and hash the pinned Jamba2 snapshot on a compute node, then repeat the ownership/state/process probe immediately before launching on one explicitly selected physical GPU.

The first compute-side download command exited before network or model work because Slurm resolved bare `env` to a non-executable user-local file. The committed launcher now invokes `/usr/bin/env` explicitly; this was a wrapper-path failure, not a model or GPU failure.

The retry succeeded on `nid011179`: 13 files at the pinned revision were downloaded and locally SHA256-hashed, with exactly 6,394,271,296 safetensor bytes. The next gate is a real one-GPU generation after immediate device revalidation.

The first guarded GPU smoke selected an actually empty GH200 (3 MiB used and no compute PID) but stopped before model load because Torch `2.13.0+cu130` is newer than the node's driver compatibility level `12070`. Raw evidence is preserved in `01_jamba2_3b_inference/runs/smoke_20260811T165758Z/`. The next action is a driver-compatible ARM Torch pin, not a model-code change.

Torch has now been repinned locally to the official AArch64 `2.9.1+cu126` wheel and the CUDA 13 packages were removed. The login node does not expose the CUDA 12 shared libraries that this wheel dynamically links, so the next authoritative check is inside the compute allocation. The runtime remains unpromoted.

The dependency-resolving retry installed all CUDA 12.6 packages. The audit is down to one failure: `nvidia-cusparselt-cu12==0.7.1` carries an incompatible platform tag. This must be resolved before the compute import and model generation are retried.

The remaining package warning is now narrowly audited: NVIDIA's legacy `manylinux2014_sbsa` tag is inconsistent with UV's accepted tags, but the shipped shared object is an AArch64 ELF. The exact exception verifier and login-node Torch import both pass. A user-supplied `gtop` sample at `17:14:29Z` shows all 16 GPUs in allocation `5980502` idle; the next gate is compute-side CUDA initialization and generation on one revalidated device.

The first post-snapshot launch correctly aborted before CUDA/model work because `nid010053` GPU 0 had acquired PID `164189`. The point-in-time snapshot was not treated as a reservation; the next attempt will use a fresh process-level probe.

A guarded attempt on `5975573` then proved the CUDA 12.6 runtime can initialize on a GH200. It stopped before weight load because peak-memory statistics were reset before allocator initialization. The required code change is narrow: create the first CUDA tensor before the reset and use the renamed allocator configuration variable.

Jamba2-3B is now functionally deployed. The corrected guarded run on `nid010197` physical GPU 3 loaded the pinned 6.39 GB checkpoint and produced 32 new tokens in 15.31 seconds with 6.40 GB peak GPU memory. Raw preflight, log, output, and structured provenance are committed under `01_jamba2_3b_inference/runs/smoke_20260811T172020Z/`.

Nemotron Nano 9B v2 is also functionally deployed. Its repository-provided naive path loaded the checkpoint but generated degenerate text, so it was not promoted. After `causal-conv1d 1.6.2.post1` and `mamba-ssm 2.3.2.post1` were built from source against CUDA 12.6, the byte-identical upstream model implementation generated a relevant 24-token answer in 45.61 seconds with 18.17 GB peak GPU memory. Evidence is in `02_nemotron_nano_9b_v2_inference/runs/smoke_20260811T173651Z/`.

Kimi Linear 48B-A3B is functionally deployed across two guarded GH200 GPUs. The environment required the checkpoint-compatible `fla-core 0.4.0` API, an explicit `tiktoken` dependency, and a documentation/backend-selection compatibility view. The successful run loaded 98.25 GB of pinned weights without CPU/disk offload and generated a relevant 24-token answer in 44.35 seconds; peak GPU allocations were 45.58 GB and 52.79 GB. Evidence is in `03_kimi_linear_48b_inference/runs/smoke_20260811T175406Z/`.

Kimi K3 is correctly capacity-gated. The raw checkpoint leaves only 71.15 GB total headroom across 16 GH200 GPUs, while the current supported Hopper SGLang shape is 32 GPUs and the vLLM path requires a CUDA 13/r580+ environment. Its pinned GitHub source, non-weight model repository, license, technical report, vLLM recipe, and SGLang recipe are locally captured and hashed. Jamba2, Nemotron Nano 9B v2, and Kimi Linear 48B-A3B are all promoted functional deployments.

The exact `Hybrid` Notion page was updated in place and re-fetched at `2026-08-11T18:01:07Z`; the rendered result contains the success table, compatibility findings, K3 capacity gate, official citations, and local artifact paths while preserving the original page content.
