# Task 02: Nemotron Nano 9B v2 inference

- Model: `nvidia/NVIDIA-Nemotron-Nano-9B-v2`
- Revision: `6533e8de2c68e4536bf7c411d7a3ce5734111476`
- Official weight bytes: `17,776,492,512`
- Architecture: 56-layer Nemotron-H; Mamba-2/MLP hybrid with four attention layers
- Minimum planned device set: one revalidated empty H100/GH200 GPU

Success requires a task-local checkpoint snapshot, a pinned environment, exit code zero from a real GPU generation, non-empty generated text, and captured GPU/model provenance. The NVIDIA model license must remain next to the local snapshot.

## Reproducible workflow

```bash
./prepare_runtime.sh
./prepare_model_on_allocation.sh JOB_ID NODE
./.venv/bin/python prepare_compatibility_model.py --source model --destination compatibility_model
./run_on_allocation.sh JOB_ID NODE PHYSICAL_GPU
```

The pinned upstream snapshot remains byte-for-byte unchanged and includes NVIDIA's configuration, implementation, tokenizer, model card, safety/privacy/explainability documents, and four checkpoint shards. Its remote implementation contains a complete pure-PyTorch Mamba-2 fallback but unconditionally imports the gated RMSNorm function from `mamba-ssm`. Because the site's AArch64/Python 3.13 runtime does not provide that CUDA extension, `prepare_compatibility_model.py` creates a non-overwriting symlink view and replaces only that import with the algebraically equivalent upstream reference formula. Both implementation hashes are recorded. This is a functional compatibility deployment, not an optimized throughput claim.

Runtime preparation completed at `2026-08-11T17:27:40Z`. The independent 5,299,391,581-byte environment contains 43 pinned packages, imports Torch `2.9.1+cu126`, and passes the same narrowly scoped NVIDIA SBSA/AArch64 library audit as the Jamba deployment. See `requirements.lock` and `runtime_manifest.json`.

The pure-PyTorch path loads and executes the checkpoint but currently fails the semantic quality gate with a repeated `-1` output, both with and without generation caching. `build_fast_kernels_on_allocation.sh` therefore provides an isolated, compute-node-only attempt to build the official `causal-conv1d==1.6.2.post1` and `mamba-ssm==2.3.2.post1` paths against CUDA 12.6. A zero exit and meaningful generation are required before this deployment is promoted.

That source build completed successfully on `nid010197` at `2026-08-11T17:35:13Z`; both packages import from the task-local environment. The post-build lock contains 65 packages and the environment occupies 6,604,213,291 bytes. The next run must use the byte-identical upstream `modeling_nemotron_h.py` and pass the semantic quality gate.
