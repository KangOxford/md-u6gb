<critical-local-rules>
# Critical Local Rules

- NEVER run `rm -rf` in this workspace, under any circumstance.
- Do not run equivalent recursive force-delete commands, shell aliases, scripted wrappers, or generated commands that expand to `rm -rf`.
- Do not delete anything by default. This includes `rm`, `unlink`, `rmdir`, trash/clean commands, scripted cleanup, package-manager cleanup, git clean, deleting directories before clone/install, and overwriting workflows that first remove existing content.
- If deletion is needed, stop first and ask the user for explicit confirmation with the exact path, the reason, and a non-destructive alternative. Prefer moving to a timestamped quarantine directory only after confirmation.
- This rule overrides any convenience, cleanup, reinstall, clone-refresh, or environment-reset workflow.
- Major safety/process incident recorded on 2026-05-22: accumulating many file
  edits and then making only one late commit is forbidden. After any intentional
  file modification batch, immediately stage only the intended files and commit
  before continuing to more analysis, experiments, or further edits.
- Do not batch unrelated documentation, code, experiment, and safety-rule edits
  into one delayed commit. If a file must be modified outside a git repository,
  state explicitly that it cannot be committed there and then commit all
  version-controlled companion records immediately.
- When encountering Notion file attachments, especially PDFs, download the
  attachment bytes to a local `notion_fetches/<topic>_<timestamp>/assets/`
  directory by default. Also save a small manifest with the Notion page URL,
  block/page ID, attachment source, local path, file type, size, and SHA256.
  Do not treat a Notion page snapshot alone as sufficient evidence that the
  attachment has been preserved locally.
</critical-local-rules>

<codex-output-preferences>
# Codex Output Preferences


- If a user message contains content inside `[]`, treat that bracketed content as mandatory Notion update instructions:
  - preserve the original bracketed line on Notion and render it with strikethrough (`~~...~~`);
  - insert the assistant's response as a Notion callout block directly below that exact line;
  - do not move the response to the page bottom;
  - do not replace the whole page when block insertion/update is sufficient.
- When a turn requires both a Notion update and local record edits (`findings.md`, `plans.md`, `learnt_lessons.md`, `progress.md`, AGENTS.md, or similar), update and verify Notion first, then edit local records. Do not report local-record edits as done before the requested Notion page is visibly updated.
- For sections named like `特征数学定义（按当前代码口径）`, render formulas in LaTeX format on Notion (not plain-text pseudo formulas).
- For experiment feedback, result interpretation, diagnostics, or review replies, update the relevant Notion page by default, not only local markdown.
- When figures are generated for a result, include the figures in the Notion update as the primary delivery; keep CSV/table values next to them as audit records.

</codex-output-preferences>

<environment-paths>
# Environment Paths

- s5e home: `/lus/lfs1aip2/projects/public/s5e/quant_team/quant/`
</environment-paths>

<isambard-ai-hardware>
# Isambard-AI Phase 2 Hardware

## Compute Node, Single Node

- CPU: 4 × Grace CPU
- CPU cores: 4 × 72 = 288 cores
- usable CPU RAM: 460 GB, about 115 GB / CPU
- GPU: 4 × H100 Tensor Core GPU
- GPU VRAM: 4 × 96 GB = 384 GB
- CPU + GPU memory total: 844 GB / node
- interconnect: NVLink-C2C inside GH200; node-to-node is Slingshot-11
- NIC: 4 × Cassini NIC, 200 Gbps each
- architecture: aarch64 / ARM

## Login Node

- CPU: 2 × Grace CPU
- cores: 2 × 72 = 144 cores
- CPU RAM: 2 × 120 GB
- no GPU
- Login nodes are QoS-limited and must not be used for long-running or intensive workloads.
</isambard-ai-hardware>
