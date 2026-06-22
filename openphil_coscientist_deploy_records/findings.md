# Findings

- Target repository: `https://github.com/yhg01/OpenPhil_coscientist`.
- Public HTTPS access failed: GitHub requested credentials during `git clone`.
- Public web access returned unavailable/404 behavior.
- `gh` CLI is not installed on this machine.
- SSH access failed with `Permission denied (publickey)`.
- No local `OpenPhil_coscientist` directory was found at the workspace root or within a max-depth-3 search.
- The failed HTTPS clone did not leave a partial `OpenPhil_coscientist` directory.
- The provided Anthropic and OpenAI keys were not echoed into commands, files, or commits during this blocked turn.
- User confirmed a successful authenticated clone under their home directory using GitHub username `KangOxford`.
- The clone is available at `/projects/public/u6gb/OpenPhil_coscientist`.
- Docker is not installed on `login42`, so the Docker Compose path cannot run on this host.
- Host-mode prerequisites are available: Python 3.13.12, Node v25.7.0, npm 11.10.1, and Claude Code 2.1.175.
- The repository currently references Anthropic/Claude only; no OpenAI API-key usage was found in the code search.
- Python dependencies were installed into ignored `.venv/`.
- Import smoke test passed for `api.server`, `ui.app`, `research.runtime`, and `evolution.runtime`.
- Test suite passed: `8 passed in 1.10s`.
- Deployment runs in tmux session `openphil_coscientist` on `login42`.
- API health check passed at `http://127.0.0.1:8765/health`.
- Streamlit UI responded `HTTP/1.1 200 OK` at `http://127.0.0.1:8501/`.
- Source tree remains clean on `main`; only ignored install/runtime artifacts exist.
- The pasted API keys were still not written to files or command lines.

## 2026-06-22 refresh

- Notion page `openphil-quant` was fetched and contains the same install/API checklist target.
- The page currently contains pasted API credentials; values were not echoed into commands or written to local files.
- Local clone exists at `/lus/lfs1aip2/projects/public/u6gb/OpenPhil_coscientist` and points to `https://github.com/yhg01/OpenPhil_coscientist.git`.
- `git fetch origin` currently fails non-interactively because GitHub HTTPS authentication is unavailable in this shell.
- The local source tree is clean and reports `main...origin/main` against the last-known remote ref.
- `.env` is missing, so runtime Claude credentials are not configured on disk.
- Docker is still unavailable on this host: `docker` command not found.
- The previous `openphil_coscientist` tmux session is not running.
- API health check failed on `127.0.0.1:8765` and Streamlit failed on `127.0.0.1:8501`; the old deployment is down.
- Host-mode dependencies remain installed in `.venv` with Python 3.13.12, `coscientist` editable install, FastAPI, Streamlit, and `claude-agent-sdk`.
- `ui/app.py` hardcodes Docker hostname `http://coscientist-api:8765`; host-mode UI needs an API URL override to make UI actions reach `127.0.0.1:8765`.
- Host-mode UI override was added and committed in `OpenPhil_coscientist` as `8d0478f`.
- Test suite after the code change passed: `8 passed in 2.80s`.
- API and UI were restarted in tmux session `openphil_coscientist` with two windows: `api` and `ui`.
- API health check passed at `http://127.0.0.1:8765/health`.
- Streamlit UI responded `HTTP/1.1 200 OK` at `http://127.0.0.1:8501/`.
- tmux pane logs showed clean Uvicorn and Streamlit startup.
- Notion page `openphil-quant` was updated and re-fetched: install checkbox is checked, deployment callout is under the install item, and API credential status callout is under `APIs`.
- The credential block remains on Notion because deletion was not explicitly requested or confirmed.

## 2026-06-22 Mac access page

- Current service host was verified as `login40` with user `kangli.u6gb`.
- `openphil_coscientist` tmux session was still running with two windows.
- API health check and UI HTTP check passed immediately before writing the Mac-access instructions.
- Created Notion child page: https://app.notion.com/p/38712c4568fd81a58cd8e7e56138e879
- The child page documents the Mac-side SSH tunnel command, browser URL, fallback local ports, and troubleshooting checklist.
- Re-fetched the child page and verified the content under parent `openphil-quant`.
- Re-fetched the parent page and confirmed the child page block is visible.
- Parent page contained a Notion-hosted `image.png` attachment; archived it locally under `notion_fetches/openphil_mac_tunnel_20260622T1129/assets/` with a manifest.
- User corrected that Mac SSH target is `u6gb.aip2.isambard` and each login is randomly assigned to a login node.
- The original direct `kangli.u6gb@login40` tunnel instruction was wrong for this access pattern unless `login40` is directly SSH-reachable from the Mac.
- Updated the Notion child page to use `ssh -J kangli.u6gb@u6gb.aip2.isambard ... kangli.u6gb@login40`, preserving the user's bracketed correction as struck-through text with a callout.
- Re-fetched the child page and verified the corrected ProxyJump command and explanation rendered in place.
