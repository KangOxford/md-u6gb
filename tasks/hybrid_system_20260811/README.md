# Hybrid inference system

This directory implements the request on the Notion page [Hybrid](https://app.notion.com/p/3b912c4568fd806b9200ecead77d7bf8): study current open hybrid architectures, bring their official code and checkpoints into isolated local task folders, and prove inference on unused Isambard-AI GPUs without disturbing existing work.

## Folder contract

Each research or deployment target has its own folder:

- `00_portfolio_research/`: cross-model architecture and deployment selection.
- `01_jamba2_3b_inference/`: AI21 Jamba2 3B deployment.
- `02_nemotron_nano_9b_v2_inference/`: NVIDIA Nemotron Nano 9B v2 deployment.
- `03_kimi_linear_48b_inference/`: Moonshot Kimi Linear 48B-A3B deployment.
- `04_kimi_k3_feasibility/`: Kimi K3 source capture and hardware feasibility gate.

Model weights, virtual environments, cloned upstream repositories, and generated logs stay inside the relevant task folder but are not committed to the root Markdown repository. Their pinned revisions and verification evidence are recorded in committed manifests and reports.

## Current status

The user-supplied `gtop` sample at `2026-08-11T17:14:29Z` showed all 16 GPUs in allocation `5980502` physically idle, but a launch-time probe less than two minutes later found that training processes had returned to all four nodes. The guard aborted before CUDA initialization. Allocation `5980745` remains fully active and excluded. A separate launch on an actually empty GPU in the nearly expired allocation `5975573` completed the Jamba2-3B smoke without stopping, moving, or overwriting any process. Every deployment repeats the same inside-allocation GPU process/memory gate at launch time.

The task-wide state machine is recorded in `manifest.json`, `events.jsonl`, `state.json`, and `summary.md`.
