# Task 01: Jamba2 3B inference

- Model: `ai21labs/AI21-Jamba2-3B`
- Revision: `525c6c8e1d9f5bddedfbdc1dbb0ade2df84230c9`
- Official weight bytes: `6,394,271,296`
- Architecture: 28-layer Jamba; Mamba layers with attention at offset 7 and period 14
- Minimum planned device set: one revalidated empty H100/GH200 GPU

Success requires a task-local checkpoint snapshot, a pinned environment, exit code zero from a real GPU generation, non-empty generated text, and captured GPU/model provenance. Merely importing the configuration is not success.
