# Task 04: Kimi K3 feasibility and source capture

- Model: `moonshotai/Kimi-K3`
- Model revision: `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`
- Official GitHub revision observed: `3cb39dfd32e51c3328e2e4b4af21341247d06c43`
- Official weight bytes: `1,560,936,091,448` across 96 safetensor shards
- Architecture: 93 text layers, 69 KDA plus 24 gated MLA, AttnRes blocks, 896 routed experts with 16 selected, native vision, MXFP4-compressed weights

The current point-in-time pool of eight empty 96 GB GPUs cannot hold the checkpoint weights, even before runtime state, activations, communication buffers, or KV/recurrent state. Therefore this task's first deliverable is a pinned source/recipe capture and an explicit capacity plan. It must not be called locally deployed until a supported distributed engine produces real output from the local weights.

No K3 weights will be downloaded speculatively into a launch configuration that cannot run them. This prevents a 1.56 TB download from being mistaken for deployment progress.

`capture_source.py` non-destructively downloads the exact GitHub archive and every non-safetensor file from the pinned Hugging Face revision into this task folder. It records a SHA256 and byte size for every file in `source_manifest.json`, while explicitly recording that all 96 weight shards were skipped by the capacity gate.

```bash
./capture_source.py
```
