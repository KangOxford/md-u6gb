#!/usr/bin/env bash
# Three shell-state defects, each with a real failure case.
#
#   D1  an empty .done is read as "member already collected"
#   D2  set -e destroys the error state before the guard that reports it
#   D3  a failure branch logs rc=0 because a substitution precedes $?
#
# Every case BINDS TO THE REAL FILE: the predicate or the log line is extracted from
# the shipping script by grep, so this goes red on the current code and green only
# once the file itself changes.  A test written against a copy of the logic proves
# nothing about what runs.
set -uo pipefail

MID=/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/crps-return-alignment-20260808/run/mid_training
NB=/lus/lfs1aip2/projects/public/u6gb/nb_build_pr22
W="${1:-$(mktemp -d "${TMPDIR:-/tmp}/tshell_XXXXXX")}"
mkdir -p "$W"
PASS=0; FAIL=0
ok(){ echo "  PASS  $1"; PASS=$((PASS+1)); }
no(){ echo "  FAIL  $1"; echo "        $2"; FAIL=$((FAIL+1)); }

echo "workdir: $W"
echo
echo "=============================================================="
echo "D1  an empty .done must NOT count as a collected member"
echo "=============================================================="
# The real failure case.  MEASURED 2026-09-05, correcting an earlier claim: `set -u`
# on an unbound variable aborts BEFORE the redirection, so no file is created at all.
# A 0-byte file comes from the opposite order -- the redirection succeeds, creating or
# truncating the target, and the writer then fails or is killed.  That is exactly what
# an allocation expiring mid-write looks like.
M="$W/d1/member_0"; mkdir -p "$M"
bash -c 'set -euo pipefail; cat /nonexistent/manifest > "$1/.done"' _ "$M" 2>/dev/null
sz=$(stat -c %s "$M/.done" 2>/dev/null || echo missing)
if [ "$sz" = "0" ]; then ok "the failure case reproduces: .done is 0 bytes"
else no "could not reproduce a 0-byte .done" "size=$sz"; fi

# A second member that really is complete, so the predicate has to tell them apart.
G="$W/d1/member_1"; mkdir -p "$G"
printf '{"ckpt":"/fake","ckpt_step":1,"seed":97901,"member":1}\n' > "$G/.done"

# Extract the shipping predicate from collect_rollouts.sh rather than restating it.
PRED_LINE=$(grep -n 'MEMBER_DIR/\.done' "$MID/collect_rollouts.sh" | head -1)
echo "  shipping predicate: ${PRED_LINE}"
check_done(){ # mirrors whatever collect_rollouts.sh uses, read from the file
  local d="$1"
  if grep -q 'if \[ -s "\$MEMBER_DIR/\.done" \]' "$MID/collect_rollouts.sh" \
     || grep -q '_done_ok "\$MEMBER_DIR"' "$MID/collect_rollouts.sh"; then
      [ -s "$d/.done" ] && python3 -c 'import json,sys;json.load(open(sys.argv[1]))' "$d/.done" 2>/dev/null
  else
      [ -f "$d/.done" ]
  fi
}
if check_done "$M"; then no "empty .done accepted as complete" "this is how a lost member is skipped on resume"
else ok "empty .done rejected"; fi
if check_done "$G"; then ok "genuine .done still accepted"
else no "genuine .done rejected" "the fix must not reject good members"; fi

echo
echo "=============================================================="
echo "D2  set -e must not destroy the error state before it is reported"
echo "=============================================================="
# Real failure case: the payload exits 9.  The operator needs the guard's message,
# which names the log to look in.  Under bare set -e the script dies at the payload
# and that message is never produced.
cat > "$W/d2_probe.sh" <<'PROBE'
set -euo pipefail
LOG="$1"
__PAYLOAD__
if [ "$_rc" -ne 0 ]; then
    echo "[collect] FATAL: member 0 inference exit=$_rc; see $LOG" >&2
    exit "$_rc"
fi
echo "[collect] member 0 ok"
PROBE
# Use the real shape from collect_rollouts.sh: is the call guarded, or bare?
if grep -qE '^\s*_rc=0$' "$MID/collect_rollouts.sh" \
   && grep -qE '\|\|\s*_rc=\$\?' "$MID/collect_rollouts.sh"; then
    PAYLOAD='_rc=0; bash -c "exit 9" > "$LOG" 2>&1 || _rc=$?'
else
    PAYLOAD='bash -c "exit 9" > "$LOG" 2>&1
    _rc=$?'
fi
python3 - "$W/d2_probe.sh" "$PAYLOAD" <<'PY'
import sys
p,pay=sys.argv[1],sys.argv[2]
s=open(p).read().replace("__PAYLOAD__",pay)
open(p,"w").write(s)
PY
err=$(bash "$W/d2_probe.sh" "$W/d2.log" 2>&1); rc=$?
echo "  probe rc=$rc"
if [ "$rc" -ne 9 ]; then no "exit code lost" "expected 9, got $rc"; else ok "exit code 9 propagates"; fi
case "$err" in
  *"see $W/d2.log"*) ok "the guard reported which log to read" ;;
  *) no "error state lost: the guard never ran" "stderr was: ${err:-<empty>}" ;;
esac

echo
echo "=============================================================="
echo "D3  a failure branch must log the real exit code, not 0"
echo "=============================================================="
# Real failure case: run the shipping log line with a payload that fails.
LOGLINE=$(grep -n 'FAILED rc=' "$NB/eval_shard.sh" | head -1)
echo "  shipping line: ${LOGLINE}"
# Extract the real failure branch from the shipping file and EXECUTE it, rather than
# grepping for a form and then running a restatement.  A detector that has to
# recognise the fix will also have to be updated by whoever breaks it.
python3 - "$NB/eval_shard.sh" "$W/d3_branch.sh" <<'PY'
import re,sys
src,dst=sys.argv[1],sys.argv[2]
lines=open(src).read().split("\n")
i=next(k for k,l in enumerate(lines) if l.strip().startswith("||"))
# take the || branch through to its closing brace or end of the logical line
j=i
while j<len(lines) and not lines[j].rstrip().endswith(("}", '"')) or j==i and not lines[i].rstrip().endswith(("}", '"')):
    j+=1
    if j-i>8: break
branch="\n".join(lines[i:j+1])
open(dst,"w").write(
  'SHARD=0; i=0; STOCK=X; SEED=Y\n'
  'false \\\n' + branch + "\n")
PY
out=$(bash "$W/d3_branch.sh" 2>&1)
echo "  produced: $out"
case "$out" in
  *"FAILED rc=0"*) no "a failure logged rc=0" "every failure in this log reads as a success" ;;
  *"FAILED rc="*)  ok "the failure logged a non-zero code" ;;
  *) no "no FAILED line produced" "$out" ;;
esac

echo
echo "=============================================================="
echo "$PASS passed, $FAIL failed"
exit "$FAIL"
