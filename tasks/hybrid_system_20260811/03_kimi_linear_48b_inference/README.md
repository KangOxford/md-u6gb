# Task 03: Kimi Linear 48B-A3B inference

- Model: `moonshotai/Kimi-Linear-48B-A3B-Instruct`
- Revision: `e1df551a447157d4658b573f9a695d57658590e9`
- Official weight bytes: `98,248,224,120`
- Architecture: 27 layers with 20 KDA and seven global MLA layers; 48B total / 3B activated MoE
- Device gate: two simultaneously empty GH200 GPUs on one node

The runtime pins the official `fla-core==0.5.2` KDA implementation. The two-GPU launcher refuses either device if it has a compute PID or more than 64 MiB allocated, exposes only those physical devices, forbids CPU/disk offload after model dispatch, and records the complete device-map distribution.

```bash
./prepare_runtime.sh
./prepare_model_on_allocation.sh JOB_ID NODE
./run_on_allocation.sh JOB_ID NODE GPU_A GPU_B
```

The independent runtime completed at `2026-08-11T17:41:29Z`: 45 pinned packages, Torch `2.9.1+cu126`, Transformers `4.57.1`, and `fla-core 0.5.2`, occupying 5,307,878,989 bytes. Torch and FLA imports pass; the only package-check exception is the already audited NVIDIA SBSA/AArch64 wheel tag.

The first guarded launch passed both GPU gates and then stopped before tokenizer/model load because the remote tokenizer code imports an undeclared `tiktoken` dependency. The runtime now pins the current CPython 3.13 AArch64 wheel, `tiktoken==0.13.0`, before retrying.

The next import exposed a documentation-only compatibility defect: Transformers `auto_docstring` assumes annotations have `__name__`, while the pinned remote code uses Python 3.13 `types.UnionType`. `prepare_compatibility_model.py` preserves the full original snapshot and creates a symlink view whose only code change is removing the two runtime documentation decorators; both source hashes are recorded.

After full two-GPU model loading, `fla-core 0.5.2` failed at the first KDA gate because its API renamed `g_bias` to `dt_bias`. A direct audit of the published wheels shows `0.4.0` has the exact five-argument signature used by the pinned model, while every release from `0.4.1` onward has the renamed API. The runtime is therefore pinned to `fla-core==0.4.0`, not the model card's unsafe open-ended `>=0.4.0` range.

- Model: `moonshotai/Kimi-Linear-48B-A3B-Instruct`
- Revision: `e1df551a447157d4658b573f9a695d57658590e9`
- Official weight bytes: `98,248,224,120`
- Architecture: 27 layers, including 20 KDA layers and 7 global MLA layers, with 256 routed experts and 3B active parameters
- Planned device set: at least two revalidated empty GPUs; four GPUs matches the official vLLM example

Success requires a task-local checkpoint snapshot, the official KDA-capable runtime, exit code zero from a real multi-GPU generation, non-empty generated text, and captured device placement. A single 96 GB-class GPU leaves too little headroom after raw weights and is not the safe launch plan.
