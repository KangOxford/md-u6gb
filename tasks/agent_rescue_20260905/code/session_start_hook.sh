#!/bin/bash
R=/lus/lfs1aip2/projects/public/u6gb/.claude/agent_registry
"$R/agent_reg.sh" rescue 2>/dev/null | tail -1
out=$("$R/agent_reg.sh" pending 2>/dev/null)
case "$out" in
  "no unfinished agents"*|"(registry empty)"*) ;;
  *) echo "$out" ;;
esac
