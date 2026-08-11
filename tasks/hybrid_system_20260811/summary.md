# Hybrid inference supervisor summary

State: **running**  
Updated: `2026-08-11T16:55:37Z`

The exact Notion target has been fetched and converted into four isolated model tasks. Current primary-source revisions and checkpoint byte sizes are pinned. A live read-only probe found eight empty GPUs inside allocation `5980502`, alongside unrelated GPU-resident processes that must remain untouched.

The Jamba2 3B smoke runtime and safety launcher are committed. A task-local 4.6 GB environment was successfully created without changing the shared Miniforge installation; its exact local package set is recorded in `01_jamba2_3b_inference/requirements.lock`.

Recommended next action: download and hash the pinned Jamba2 snapshot on a compute node, then repeat the ownership/state/process probe immediately before launching on one explicitly selected physical GPU.

The first compute-side download command exited before network or model work because Slurm resolved bare `env` to a non-executable user-local file. The committed launcher now invokes `/usr/bin/env` explicitly; this was a wrapper-path failure, not a model or GPU failure.

The retry succeeded on `nid011179`: 13 files at the pinned revision were downloaded and locally SHA256-hashed, with exactly 6,394,271,296 safetensor bytes. The next gate is a real one-GPU generation after immediate device revalidation.

The first guarded GPU smoke selected an actually empty GH200 (3 MiB used and no compute PID) but stopped before model load because Torch `2.13.0+cu130` is newer than the node's driver compatibility level `12070`. Raw evidence is preserved in `01_jamba2_3b_inference/runs/smoke_20260811T165758Z/`. The next action is a driver-compatible ARM Torch pin, not a model-code change.

Torch has now been repinned locally to the official AArch64 `2.9.1+cu126` wheel and the CUDA 13 packages were removed. The login node does not expose the CUDA 12 shared libraries that this wheel dynamically links, so the next authoritative check is inside the compute allocation. The runtime remains unpromoted.

Kimi K3 is correctly capacity-gated: its 1.56 TB checkpoint cannot fit in the currently empty device pool. No result has yet been promoted to `completed`.
