# Learnt Lessons

- Coverage must be reconstructed from allocated RUNNING intervals, not submission counts or PENDING jobs.
- Update and verify Notion before persisting the local daily record.
- The launcher is one 16-node job, not sixteen single-node jobs; exact-name-only accounting would miss its auto-resume jobs.
- A complete UTC-day report needs a one-walltime accounting lookback so jobs that started before 00:00 are still clipped into the target window.
- Second-level precision must not obscure the operational question: were 16 nodes materially RUNNING throughout the day?
- `sbatch` acceptance proves only that the request is valid; `AllocTRES` and `NodeList` must be populated before claiming the 16 nodes are reserved.
- Queue disappearance needs `sacct` attribution: `CANCELLED` with zero elapsed time is not a runtime failure.
- Composition keeps the allocation script simple while allowing monitoring and cancellation policy to evolve independently.
- Slurm `PrivateData` means a user-visible queue cannot explain all priority ordering or reservations.
- The monitor should terminate after winner selection rather than become a general-purpose polling daemon.
- Require both explicit candidate membership and a name-prefix match before issuing `scancel`.
- A documented Slurm blocking command still needs live validation on the cluster's PENDING-state implementation.
- Avoid heartbeat log spam: unchanged PENDING state should remain implicit between transition events.
