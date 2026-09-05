#!/bin/bash
# Durable per-agent journal. Usage: source journal.sh <slug>
# Then: charter "..."  jrnl "..."  finding "claim" "number"  nxt "..."
# Everything appends to ONE file so it costs one inode (Lustre inode quota ~0 free).
_JR=/lus/lfs1aip2/projects/public/u6gb/.claude/agent_registry/work/${1:-unnamed}
mkdir -p "$_JR" 2>/dev/null
export _JPROG="$_JR/progress.md"
[ -f "$_JPROG" ] || : > "$_JPROG"
_jts(){ date -u +%Y-%m-%dT%H:%M:%SZ; }
charter(){ printf '## CHARTER %s\n%s\n\n' "$(_jts)" "$*" >> "$_JPROG"; echo "charter -> $_JPROG"; }
jrnl(){    printf -- '- %s  %s\n' "$(_jts)" "$*" >> "$_JPROG"; }
finding(){ printf -- '- **FINDING** %s  %s  || EVIDENCE: %s\n' "$(_jts)" "$1" "$2" >> "$_JPROG"; }
nxt(){     printf -- '- NEXT %s  %s\n' "$(_jts)" "$*" >> "$_JPROG"; }
echo "journal: $_JPROG"
