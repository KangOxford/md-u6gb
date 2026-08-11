# Hybrid inference supervisor summary

State: **running**  
Updated: `2026-08-11T16:55:37Z`

The exact Notion target has been fetched and converted into four isolated model tasks. Current primary-source revisions and checkpoint byte sizes are pinned. A live read-only probe found eight empty GPUs inside allocation `5980502`, alongside unrelated GPU-resident processes that must remain untouched.

The Jamba2 3B smoke runtime and safety launcher are committed. A task-local 4.6 GB environment was successfully created without changing the shared Miniforge installation; its exact local package set is recorded in `01_jamba2_3b_inference/requirements.lock`.

Recommended next action: download and hash the pinned Jamba2 snapshot on a compute node, then repeat the ownership/state/process probe immediately before launching on one explicitly selected physical GPU.

Kimi K3 is correctly capacity-gated: its 1.56 TB checkpoint cannot fit in the currently empty device pool. No result has yet been promoted to `completed`.
