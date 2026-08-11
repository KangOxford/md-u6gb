# Task 01: Jamba2 3B inference

- Model: `ai21labs/AI21-Jamba2-3B`
- Revision: `525c6c8e1d9f5bddedfbdc1dbb0ade2df84230c9`
- Official weight bytes: `6,394,271,296`
- Architecture: 28-layer Jamba; Mamba layers with attention at offset 7 and period 14
- Minimum planned device set: one revalidated empty H100/GH200 GPU

Success requires a task-local checkpoint snapshot, a pinned environment, exit code zero from a real GPU generation, non-empty generated text, and captured GPU/model provenance. Merely importing the configuration is not success.

## Reproducible workflow

All commands are non-destructive and refuse to use a GPU with a compute process or more than 64 MiB allocated at launch time.

```bash
./prepare_runtime.sh
./prepare_model_on_allocation.sh JOB_ID NODE
./run_on_allocation.sh JOB_ID NODE PHYSICAL_GPU
```

The runtime deliberately sets `use_mamba_kernels=False` for the first compatibility smoke. This exercises the actual Jamba model and generation path through Transformers' reference implementation without requiring an unverified CUDA extension build against the site's Python 3.13 / Torch 2.11 / CUDA 13 stack. Kernel-optimized serving is a separate performance gate after functional inference succeeds.

Runtime preparation completed at `2026-08-11T16:55:37Z`. The isolated environment resolved Torch `2.13.0+cu130`, Transformers `4.57.1`, and Accelerate `1.11.0`; see `requirements.lock` and `runtime_manifest.json`. The generated 4.6 GB `.venv` is task-local and intentionally not committed.

The pinned model snapshot completed at `2026-08-11T16:57:14Z`. All 13 files were hashed locally, both safetensor shards total exactly `6,394,271,296` bytes, and the resolved revision equals the requested revision. The committed audit record is `model/download_manifest.json`; the 6.0 GiB snapshot itself remains task-local.

The first GPU smoke (`runs/smoke_20260811T165758Z`) passed the empty-device gate on `nid011179` GPU 1 (3 MiB used, no compute PID), then failed before model load. Torch `2.13.0+cu130` rejected the node's driver compatibility level `12070`. This run is evidence of an environment ABI mismatch only; it is not evidence against Jamba2 inference. The environment must be repinned to a CUDA 12.7-compatible ARM build before retry.

The corrective runtime pin is official PyTorch `2.9.1+cu126` from `https://download.pytorch.org/whl/cu126`. That index publishes a CPython 3.13 AArch64 wheel and CUDA 12.6 is within the node driver's reported compatibility. `prepare_runtime.sh` now resolves this exact build before any retry.

The repin installed successfully and reduced the task-local environment from 4.6 GB to 863 MB, but its login-node import check cannot find dynamically linked CUDA 12 libraries (`libcudart.so.12` and `libcublas.so`). The next check is the authoritative compute-node library surface; this is not yet a usable runtime claim.

The dependency-resolving retry installed the complete CUDA 12.6 wheel set and expanded the isolated environment to 4.8 GB. `uv pip check` now reports one remaining problem: `nvidia-cusparselt-cu12==0.7.1` has an incompatible platform tag. The package and compute import gates remain open until that wheel is corrected or proven unnecessary and safely removed.

The cuSPARSELt audit found that NVIDIA's installed wheel declares the legacy `manylinux2014_sbsa` tag, which UV does not recognize as AArch64, while the shipped `libcusparseLt.so.0` is independently verified as ELF64 machine `AArch64` (`e_machine=183`). `verify_runtime.py` accepts only this exact one-package/tag/binary combination; all other `uv pip check` failures remain fatal.
