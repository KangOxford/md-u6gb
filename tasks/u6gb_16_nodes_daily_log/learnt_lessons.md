# Learnt Lessons

- Coverage must be reconstructed from allocated RUNNING intervals, not submission counts or PENDING jobs.
- Update and verify Notion before persisting the local daily record.
- The launcher is one 16-node job, not sixteen single-node jobs; exact-name-only accounting would miss its auto-resume jobs.
