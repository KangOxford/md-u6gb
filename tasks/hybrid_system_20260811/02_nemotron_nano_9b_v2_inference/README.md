# Task 02: Nemotron Nano 9B v2 inference

- Model: `nvidia/NVIDIA-Nemotron-Nano-9B-v2`
- Revision: `6533e8de2c68e4536bf7c411d7a3ce5734111476`
- Official weight bytes: `17,776,492,512`
- Architecture: 56-layer Nemotron-H; Mamba-2/MLP hybrid with four attention layers
- Minimum planned device set: one revalidated empty H100/GH200 GPU

Success requires a task-local checkpoint snapshot, a pinned environment, exit code zero from a real GPU generation, non-empty generated text, and captured GPU/model provenance. The NVIDIA model license must remain next to the local snapshot.
