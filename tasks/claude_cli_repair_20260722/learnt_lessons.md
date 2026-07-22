# Learnt Lessons

- A `command not found` result can occur even with the correct `PATH` when the command is a broken symlink; verify both the link and its final target before changing shell startup files.
- npm dist-tags currently distinguish the stable release from the newer latest/next track, so automated repair should pin `stable` explicitly.
- Reinstalling the pinned package is sufficient when no live process or NFS placeholder holds the missing executable; no package-tree deletion is needed.
- Verification should cover shell resolution, CLI self-reported version, npm package state, and the final symlink target rather than relying on install exit status alone.
- Even when the user asks for a clean reinstall, deletion must wait for explicit path-level confirmation; an in-place pinned reinstall is the available non-deleting alternative and is already verified healthy.
- Before an npm uninstall on NFS-backed storage, confirm no old Claude process holds the package executable open.
