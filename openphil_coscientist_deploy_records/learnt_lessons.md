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
