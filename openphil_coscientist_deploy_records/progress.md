# Progress

- Status: API and UI deployed in tmux on `login42`; agent API-key configuration remains intentionally out-of-band.
- Completed:
  - Checked public HTTPS clone.
  - Checked local GitHub CLI availability.
  - Checked SSH repository access.
  - Checked for an existing local clone at shallow workspace depth.
  - Created these local task records outside the future clone path.
  - Located the authenticated clone.
  - Installed host-mode dependencies in `.venv/`.
  - Installed dev extras and ran tests: `8 passed`.
  - Started API and UI in tmux session `openphil_coscientist`.
  - Verified API `/health` and UI root endpoint.
- Access:
  - API: `http://127.0.0.1:8765`
  - UI: `http://127.0.0.1:8501`
- Tunnel clarification:
  - On 2026-06-15 12:50 UTC, remote API/UI were still healthy on `login42`.
  - Running `ssh -L ... kangli.u6gb@login42` from another remote login node exposes the forwarded ports on that remote login node, not directly on the user's local Mac.
  - For Mac browser access, the port-forwarding command must originate on the Mac or be chained through the login node that the Mac can reach.
- Next step: set Anthropic credentials securely before running real research/evolution tasks.

## 2026-06-22 refresh

- Status: prior API/UI deployment is down; host-mode restart in progress.
- Completed:
  - Fetched the Notion page and identified the `OpenPhil_coscientist` install/API checklist.
  - Reconfirmed the local clone exists at `/lus/lfs1aip2/projects/public/u6gb/OpenPhil_coscientist`.
  - Reconfirmed `.env` is missing and pasted API keys were not copied into files or commands.
  - Reconfirmed Docker is unavailable on this host.
  - Reconfirmed `openphil_coscientist` tmux session is absent and ports `8765`/`8501` are down.
  - Reconfirmed host-mode Python dependencies are installed in `.venv`.
- Next step: patch host-mode UI API URL handling, commit the change, restart services, and verify.
