You are the daily evidence-only logger for the Slurm fleet `u6gb-16-nodes`.

Report date: `{{DATE}}` UTC.

Do only these steps, in order:

1. Read the command-source section produced from `/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_R1_Mamba3/train_full_autoreg.batch`, then run `python3 tasks/u6gb_16_nodes_daily_log/collect_daily.py --date {{DATE}}` without `--write` and read the complete report.
2. Fetch Notion page `39f12c45-68fd-8106-a60b-e64bed63bd91` (`u6gb-16-nodes Daily Coverage Log`).
3. If the page already has a heading for `{{DATE}} UTC`, do not duplicate it. Otherwise append the report as a new daily section. Preserve every earlier day.
4. Re-fetch the Notion page and verify the date heading, command/result section, coverage summary, full intervals, and gap intervals are visible.
5. Only after Notion verification, run `python3 tasks/u6gb_16_nodes_daily_log/collect_daily.py --date {{DATE}} --write --notion-status updated`.
6. Update the four task records in `tasks/u6gb_16_nodes_daily_log/` with one concise line each. Stage only files under that task directory and commit immediately.

Hard restrictions:

- Do not submit, retry, requeue, cancel, or modify any experiment job.
- Do not edit experiment code or configuration.
- Do not edit the parent Notion page.
- Do not infer coverage from PENDING jobs. Coverage uses allocated RUNNING intervals only.
- Count both `u6gb-16-nodes` and its `u6gb-16-nodes-resumeN` chain.
- Treat the experiment batch as read-only command provenance; report its current SHA256 and header resource directives.
- If Notion update fails, run the collector with `--write --notion-status failed`, record the error locally, and stop.
