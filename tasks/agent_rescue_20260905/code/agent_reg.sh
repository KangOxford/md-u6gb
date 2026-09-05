#!/bin/bash
# Durable agent registry.  The registry itself lives under
# /lus/lfs1aip2/projects/public/u6gb/.claude/agent_registry, which is a symlink to
# /home/u6gb/kangli.u6gb/.claude/agent_registry on VAST /home -- NOT on Lustre.  That
# placement is load-bearing: on 2026-09-04T18:30Z the Lustre project inode quota was at
# its hard cap and every prompt.txt write failed, which is why nine registry rows carry
# their prompt inline instead of in a sibling file.
#
#   agent_reg.sh add <agent_id> <slug> <one-line description>   [prompt on stdin]
#   agent_reg.sh done <agent_id>
#   agent_reg.sh fail <agent_id> <reason...>
#   agent_reg.sh pending
#   agent_reg.sh rescue
#   agent_reg.sh recover <agent_id|slug> [--force]
#   agent_reg.sh stage <agent_id|slug> <prepared|submitted|processed|artifact> [note] [path]
#   agent_reg.sh stages
#   agent_reg.sh verify
R=/lus/lfs1aip2/projects/public/u6gb/.claude/agent_registry
REG="$R/registry.jsonl"
case "${1:-}" in
add)
  ID="$2"; SLUG="$3"; shift 3; DESC="$*"
  mkdir -p "$R/work/$SLUG"
  # The prompt is the thing that actually lets a lost agent be re-created.  Write it to a
  # sibling file AND carry it inline: the sibling write is the one that failed under a full
  # filesystem, and a prompt that exists in only one place is a prompt with a single point
  # of failure.
  PROMPT=""
  [ -t 0 ] || PROMPT=$(cat)
  [ -n "$PROMPT" ] && printf '%s' "$PROMPT" > "$R/work/$SLUG/prompt.txt"
  ID="$ID" SLUG="$SLUG" DESC="$DESC" WORK="$R/work/$SLUG" PROMPT="$PROMPT" python3 -c '
import json,os,datetime
print(json.dumps({
 "ts": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
 "agent_id": os.environ["ID"], "slug": os.environ["SLUG"], "desc": os.environ["DESC"],
 "status": "running",
 "session_dir": os.environ.get("CLAUDE_SESSION_DIR",""),
 "work": os.environ["WORK"],
 "prompt_inline": os.environ.get("PROMPT",""),
 "node": os.uname().nodename}))' >> "$REG"
  echo "registered $SLUG ($ID) -> $R/work/$SLUG" ;;
done)
  printf '{"ts":"%s","agent_id":"%s","status":"done"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$2" >> "$REG" ;;
fail)
  # An agent that died is not the same as an agent that finished.  Recording both as
  # "not running" would make the two indistinguishable at exactly the moment the
  # difference decides whether to re-create it.
  ID="$2"; shift 2
  ID="$ID" WHY="$*" python3 -c '
import json,os,datetime
print(json.dumps({
 "ts": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
 "agent_id": os.environ["ID"], "status": "failed", "reason": os.environ["WHY"]}))' >> "$REG"
  echo "recorded FAILED $ID: $*" ;;
pending)
  [ -f "$REG" ] || { echo "(registry empty)"; exit 0; }
  python3 "$R/reg_report.py" "$REG" ;;
rescue)
  # Best-effort: copy whatever transcripts are still in tmpfs onto Lustre.
  # No cap: a silent truncation here drops whole sessions, and the session that
  # matters is usually the newest one, which sorts last.
  # NOTE this is layer 3 and it is NOT the primary durable copy -- see reg_report.py.
  D=$(ls -d /run/user/$(id -u)/claude-$(id -u)/*/*/tasks 2>/dev/null)
  echo "scanning $(echo "$D" | grep -c .) session dir(s)"
  n=0
  for d in $D; do
    for f in "$d"/*.output; do
      [ -f "$f" ] || continue
      b=$(basename "$f"); cp -n "$f" "$R/transcripts/$b" 2>/dev/null && n=$((n+1))
    done
  done
  echo "rescued $n transcript(s) to $R/transcripts/" ;;
recover)
  shift; python3 "$R/reg_recover.py" "$@" ;;
verify)
  python3 "$R/reg_verify.py" ;;
stage)
  shift; python3 "$R/reg_stage.py" "$@" ;;
stages)
  python3 "$R/reg_stage.py" table ;;
*) sed -n '2,20p' "$0" ;;
esac
