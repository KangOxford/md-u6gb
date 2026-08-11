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

As of `2026-08-11T16:41:37Z`, research selection and live-allocation preflight are complete. Eight GPUs in allocation `5980502` were physically empty at the probe instant, but other GPUs on the same nodes hold unrelated Python processes. No process has been stopped, moved, or overwritten. Deployment code and GPU smoke runs are the next gates.

The task-wide state machine is recorded in `manifest.json`, `events.jsonl`, `state.json`, and `summary.md`.
