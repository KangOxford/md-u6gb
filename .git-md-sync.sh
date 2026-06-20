#!/usr/bin/env bash
# Event-driven markdown -> git autosync for the u6gb workspace.
#
# WHY event-driven (not a per-minute cron):
#   - This login node has NO crontab command.
#   - HPC rules forbid resident polling daemons on login nodes (anti-pattern #9).
#   - The workspace root is a ~300-entry Lustre tree (jax_cache, coredumps,
#     logs_*, 19 nested git repos). A per-minute `git add -A` would be a
#     metadata storm. So we only ever touch the ONE md file that just changed.
#
# Modes:
#   .git-md-sync.sh <path>     sync a single file (used by the PostToolUse hook)
#   .git-md-sync.sh --all      (re)sync the whole allowlist (manual / initial)
#   (stdin JSON)               hook passes {"tool_input":{"file_path": "..."}}
#
# Allowlist (u6gb's own authored md only; nested repos are out of scope):
#   <repo>/*.md                            depth-1 notes (plans/findings/...)
#   <repo>/.claude/projects/*/memory/*.md  auto-memory
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
cd "$REPO" || exit 0
LOCK=/tmp/.md-u6gb-push.lock

in_scope() {  # $1 = path relative to REPO ; return 0 if in allowlist
  case "$1" in
    .claude/projects/*/memory/*.md) return 0 ;;
    */*) return 1 ;;     # any other subdir md is out of scope
    *.md) return 0 ;;    # depth-1 md
    *) return 1 ;;
  esac
}

push_bg() {  # non-blocking, serialized, short-lived (exits in ~1s, NOT a daemon)
  ( flock 9; git push -q origin main 2>/dev/null || true ) 9>"$LOCK" &
}

sync_one() {  # $1 = absolute or relative path
  local rel
  rel=$(realpath --relative-to="$REPO" "$1" 2>/dev/null) || return 0
  case "$rel" in ../*|/*) return 0 ;; esac   # resolved outside the repo
  in_scope "$rel" || return 0
  [ -f "$REPO/$rel" ] || return 0
  git add -f -- "$rel" 2>/dev/null || return 0
  git diff --cached --quiet -- "$rel" && return 0   # nothing changed
  git commit -q -m "autosync: $rel @ $(date -u +%FT%TZ)" -- "$rel" 2>/dev/null || return 0
  push_bg
}

# --- mode: full allowlist resync -------------------------------------------
if [ "${1:-}" = "--all" ]; then
  shopt -s nullglob
  for f in "$REPO"/*.md "$REPO"/.claude/projects/*/memory/*.md; do
    git add -f -- "$(realpath --relative-to="$REPO" "$f")" 2>/dev/null
  done
  if ! git diff --cached --quiet; then
    git commit -q -m "autosync --all @ $(date -u +%FT%TZ)" 2>/dev/null || true
  fi
  push_bg
  exit 0
fi

# --- mode: single file (arg or hook stdin) ---------------------------------
FILE="${1:-}"
if [ -z "$FILE" ] && [ ! -t 0 ]; then
  FILE=$(python3 -c "import sys,json;print(json.load(sys.stdin).get('tool_input',{}).get('file_path',''))" 2>/dev/null)
fi
[ -n "$FILE" ] && sync_one "$FILE"
exit 0
