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

## 2026-06-22 restart complete

- Status: API and UI are live in tmux session `openphil_coscientist`; API-key configuration remains intentionally incomplete.
- Completed:
  - Added environment overrides to `ui/app.py` for `COSCIENTIST_API_URL` and `COSCIENTIST_STATE_DIR`.
  - Committed the nested repo code change: `8d0478f Allow host-mode UI API configuration`.
  - Ran tests after the code change: `8 passed in 2.80s`.
  - Started API in tmux window `openphil_coscientist:api`.
  - Started UI in tmux window `openphil_coscientist:ui` with `COSCIENTIST_API_URL=http://127.0.0.1:8765`.
  - Verified API `/health` and UI root endpoint.
- Access:
  - API: `http://127.0.0.1:8765`
  - UI: `http://127.0.0.1:8501`
- Remaining:
  - `.env` is still absent, and no pasted API keys were copied into local files or command lines.
  - Real Claude agent execution still needs secure credential configuration.

## 2026-06-22 Notion writeback

- Status: Notion page is updated and verified.
- Completed:
  - Checked the install task on `openphil-quant`.
  - Added a deployment callout with the local path, commit `8d0478f`, test result, tmux session, API URL, and UI URL.
  - Left `APIs` unchecked and added a credential-status callout.
  - Re-fetched the Notion page and verified the updated layout.
- Remaining:
  - Exposed keys are still present on Notion pending explicit user confirmation to delete or rotate them.

## 2026-06-22 Mac access page

- Status: Notion subpage for Mac access is created and verified.
- Completed:
  - Verified the live service is on `login40`.
  - Created child page `Mac 访问 server 上的 OpenPhil Coscientist UI`.
  - Included the Mac Terminal SSH tunnel command and browser URL.
  - Included fallback local ports `18501` and `18765`.
  - Re-fetched the child page and parent page to verify visibility.
  - Downloaded the parent page `image.png` attachment to `notion_fetches/openphil_mac_tunnel_20260622T1129/assets/image.png`.
  - Wrote attachment manifest at `notion_fetches/openphil_mac_tunnel_20260622T1129/manifest.md`.
- Remaining:
  - The user should run the SSH tunnel from Mac Terminal and keep that Terminal window open while using the UI.
