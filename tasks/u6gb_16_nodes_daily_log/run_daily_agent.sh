#!/bin/bash
set -euo pipefail

ROOT="/lus/lfs1aip2/projects/public/u6gb"
TASK_DIR="$ROOT/tasks/u6gb_16_nodes_daily_log"
REPORT_DATE="${1:-$(date -u -d yesterday +%F)}"
PROMPT_FILE="$TASK_DIR/daily_agent_prompt.md"
AGENT_LOG="$TASK_DIR/agent_logs/${REPORT_DATE}.md"

mkdir -p "$TASK_DIR/agent_logs"

if ! sed "s/{{DATE}}/$REPORT_DATE/g" "$PROMPT_FILE" | \
  codex exec --ephemeral --sandbox workspace-write --cd "$ROOT" \
    --output-last-message "$AGENT_LOG" -; then
  python3 "$TASK_DIR/collect_daily.py" \
    --date "$REPORT_DATE" --write --notion-status failed
  exit 1
fi
