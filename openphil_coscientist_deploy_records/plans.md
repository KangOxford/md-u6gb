# Plans

1. Keep API/UI running in tmux session `openphil_coscientist`.
2. If agent execution is required, provide `ANTHROPIC_API_KEY` through a protected local `.env` or shell environment outside chat/tool-command logs.
3. Verify any future agent run via `state/sessions/<session>/runtime.log`.

## 2026-06-22 plan

1. Add a small host-mode override so Streamlit can use `COSCIENTIST_API_URL=http://127.0.0.1:8765` instead of the Docker-only hostname.
2. Commit that code change immediately in the `OpenPhil_coscientist` git repo.
3. Restart API and UI in tmux without writing pasted API keys to files or command lines.
4. Verify `/health` and the UI HTTP response.
5. Update the Notion checklist only for the parts that are actually verified.
