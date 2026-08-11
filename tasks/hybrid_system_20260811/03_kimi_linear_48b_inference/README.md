# Task 03: Kimi Linear 48B-A3B inference

- Model: `moonshotai/Kimi-Linear-48B-A3B-Instruct`
- Revision: `e1df551a447157d4658b573f9a695d57658590e9`
- Official weight bytes: `98,248,224,120`
- Architecture: 27 layers, including 20 KDA layers and 7 global MLA layers, with 256 routed experts and 3B active parameters
- Planned device set: at least two revalidated empty GPUs; four GPUs matches the official vLLM example

Success requires a task-local checkpoint snapshot, the official KDA-capable runtime, exit code zero from a real multi-GPU generation, non-empty generated text, and captured device placement. A single 96 GB-class GPU leaves too little headroom after raw weights and is not the safe launch plan.
