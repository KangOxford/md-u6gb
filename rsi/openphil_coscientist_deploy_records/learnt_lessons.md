# Learnt Lessons

- This repository appears to be private or otherwise inaccessible from the current machine.
- Do not create files inside `OpenPhil_coscientist` until the clone is available, because doing so could make the clone destination non-empty.
- Keep deployment keys out of shell command lines and version-controlled files.
- The user's interactive shell can access the private repository after username/password authentication.
- The repo's Docker path is documented, but this login host lacks Docker; host-mode deployment is the workable path here.
- The app surfaces can start without the API key, but actual Claude agent sessions require Anthropic authentication.
- A 2026-06-22 refresh showed the old tmux deployment can become stale; always recheck live ports before marking Notion checkboxes complete.
- The Streamlit app's default API hostname is valid inside Docker Compose but not in host-mode on a login node; prefer an environment override for local deployment.
- The minimal host-mode override is enough to keep Docker Compose behavior unchanged while letting login-node Streamlit reach the local API.
- When Notion contains exposed keys, update adjacent status without repeating values and do not delete the original secret block unless the user explicitly confirms deletion.
- For local-browser access to a server-bound Streamlit app, the SSH `-L` tunnel must originate on the Mac; running the tunnel from another server only exposes the forwarded port on that server.
- When a Notion fetch exposes an attachment during a writeback task, archive the attachment bytes and a manifest even if the attachment is not the main requested deliverable.
- On Isambard-style load-balanced login hosts, a tunnel to the public login alias reaches whichever login node the alias assigns; to reach a service bound to `127.0.0.1` on a specific login node, use a final SSH target for that node via `ProxyJump`.
- For this OpenPhil repo, the most useful explanation frame is not model-training code; it is a file-state-driven research/evolution orchestration system with research agents, HITL, event logs, and git-worktree-based self-modification.
