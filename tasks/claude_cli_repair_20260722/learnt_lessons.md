# Learnt Lessons

- A `command not found` result can occur even with the correct `PATH` when the command is a broken symlink; verify both the link and its final target before changing shell startup files.
- npm dist-tags currently distinguish the stable release from the newer latest/next track, so automated repair should pin `stable` explicitly.
