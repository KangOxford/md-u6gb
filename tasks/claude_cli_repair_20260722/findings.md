# Findings

- 2026-07-22T11:16:22Z: On `login42`, `PATH` already includes `/home/u6gb/kangli.u6gb/miniforge3/bin`, so this is not a shell initialization problem.
- `/home/u6gb/kangli.u6gb/miniforge3/bin/claude` is a broken symlink to `../lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe`; the target is absent and the package directory contains only an empty `bin/` directory.
- No user-owned `claude` or `node` process and no `.nfs*` placeholder currently blocks repair.
- npm reports `stable=2.1.206` and `latest/next=2.1.217`.
- 2026-07-22T11:17:08Z: `npm install -g @anthropic-ai/claude-code@2.1.206` completed successfully (`added 1 package, and changed 1 package`).
- 2026-07-22T11:17:56Z: A fresh login shell resolves `claude` to the Miniforge path and reports `2.1.206 (Claude Code)`; npm lists the same version and `claude.exe` is restored as an executable 255,376,112-byte file.
- 2026-07-22T11:33:20Z: The user requested a delete-and-reinstall cycle after the verified repair. This requires explicit confirmation of the exact package and launcher paths before deletion under workspace policy.
