# Hybrid inference supervisor summary

State: **planned**  
Updated: `2026-08-11T16:41:37Z`

The exact Notion target has been fetched and converted into four isolated model tasks. Current primary-source revisions and checkpoint byte sizes are pinned. A live read-only probe found eight empty GPUs inside allocation `5980502`, alongside unrelated GPU-resident processes that must remain untouched.

Recommended next action: implement and commit the Jamba2 3B smoke runtime, then repeat the ownership/state/process probe immediately before launching on one explicitly selected physical GPU.

Kimi K3 is correctly capacity-gated: its 1.56 TB checkpoint cannot fit in the currently empty device pool. No result has yet been promoted to `completed`.
