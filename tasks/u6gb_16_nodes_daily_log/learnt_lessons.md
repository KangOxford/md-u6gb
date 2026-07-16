# Learnt Lessons

- Coverage must be reconstructed from allocated RUNNING intervals, not submission counts or PENDING jobs.
- Update and verify Notion before persisting the local daily record.
- The launcher is one 16-node job, not sixteen single-node jobs; exact-name-only accounting would miss its auto-resume jobs.
- A complete UTC-day report needs a one-walltime accounting lookback so jobs that started before 00:00 are still clipped into the target window.
- Second-level precision must not obscure the operational question: were 16 nodes materially RUNNING throughout the day?
- `sbatch` acceptance proves only that the request is valid; `AllocTRES` and `NodeList` must be populated before claiming the 16 nodes are reserved.
- Queue disappearance needs `sacct` attribution: `CANCELLED` with zero elapsed time is not a runtime failure.
- Composition keeps the allocation script simple while allowing monitoring and cancellation policy to evolve independently.
