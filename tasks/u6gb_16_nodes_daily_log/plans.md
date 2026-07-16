# Plans

- At UTC 00:15, analyze the previous complete UTC day and append one evidence-only section to the Notion child page.
- The logger has no authority to submit, cancel, or modify experiment jobs.
- Read the real batch as command provenance each day and report its hash and resource directives.
- Order every daily report as coverage/gaps, Slurm state/results, then command provenance.
- Monitor Job `5678750` under the exact fleet name `u6gb-16-nodes-18-jluy-001`; do not count it as coverage until it is RUNNING.
- Keep exactly one active 16-node allocation in steady state.
- Implement any future redundant-candidate convergence as a separate event-driven monitor, without modifying the allocation payload.
- Do not interpret partition-wide idle nodes as guaranteed schedulable capacity for the 16-node gang job.
- Implement the independent monitor with one blocking `scontrol wait_job` per candidate and a single reconciliation after the first waiter returns.
- Test pure winner-selection and cancellation-scope logic before starting the live monitor.
- Poll no faster than once per 60 seconds, write only state changes, and exit as soon as one winner is selected.
