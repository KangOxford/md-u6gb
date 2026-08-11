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
