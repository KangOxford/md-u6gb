#!/bin/bash
# Idle-GPU watch, every 15 minutes. Read-only: it reports, it never submits.
#
# Replaces a watcher that ran for hours and produced zero lines because it was
# broken three separate ways. Each is fixed here, and each fix is a rule from
# the project's own list that the previous script's comments quoted and its code
# contradicted:
#
#   1. It judged from gtop's HEADER idle count. That count includes "held" cards
#      -- ones with resident memory that nobody is computing on -- so it reports
#      free capacity that cannot be used. The only trustworthy signal is per-GPU
#      memory, and 1-9 MiB is what a genuinely free card reads. This script
#      counts cards, not headers.
#
#   2. It gated on `MINE -lt 2`: report only when I have fewer than two runs of
#      my own going. That is gating on the task, and the rule is to gate on the
#      resource. Idle cards matter because somebody is waiting, not because I
#      happen to be idle. Condition here is `free >= 4 AND pending >= 1`,
#      independent of what I am doing.
#
#   3. Its log was a file that did not exist yet, and the project is at its
#      inode limit (51,200,000/51,200,000), so every append failed silently.
#      Writing under $HOME, which is a different quota.
#
# It does not submit anything. Jobs from other work lines cannot simply be
# attached -- a second copy would write the same output directory as the queued
# one -- so the correct action is to surface the job id and let a human decide.
set -u

# ── SINGLETON ────────────────────────────────────────────────────────────
# The rule that produced this script says "confirm the monitor is running at
# the start of every session, and after every resume". Nothing was responsible
# for the other half, so each session started another copy: on 2026-09-04 there
# were five of this script plus two older watchers, seven processes polling the
# same thing and appending to the same log. A watcher that duplicates itself is
# not more reliable, it is louder.
#
# flock makes a second start a no-op instead of a duplicate, so the rule stays
# "always (re)start it" and idempotence lives here rather than in my memory.
LOCK=/lus/lfs1aip2/projects/public/u6gb/logs/.gpu_watch_15min.lock
exec 9>"$LOCK" || exit 1
if ! flock -n 9; then
    echo "[gpu_watch] another instance already holds $LOCK -- exiting, not duplicating."
    exit 0
fi


LOG=/lus/lfs1aip2/projects/public/u6gb/logs/gpu_watch_15min.log
INTERVAL=${INTERVAL:-1800}   # 30 minutes（2026-09-04 用户令：巡查是为计划服务的，
                      # 不是为了找活干。看到空卡不许即兴起任务——只能起计划里
                      # 已经排好、且当前依赖已满足的那个作业；不合适就等。
FREE_MIB=10           # a card under this is genuinely free; "held" cards sit far above

while true; do
    TS=$(date -u +%FT%TZ)
    # Strip NULs: gtop emits them, and a log containing NUL makes grep
    # report "Binary file matches" and print no lines -- indistinguishable
    # from "nothing matched" for anyone reading this log later.
    RAW=$(timeout 240 gtop --once 2>/dev/null); GTOP_RC=$?
    SNAP=$(printf '%s' "$RAW" | tr -d "\000")
    # A probe that did not answer is not a reading of zero. gtop times out
    # (rc 124, seen 2026-09-09) and has also been seen returning a plain squeue
    # listing with no per-card rows at all; both leave the awk below with
    # nothing to match, so NFREE would come out 0 -- indistinguishable from
    # "the cluster is full", which is the failure this watcher exists to avoid.
    NGH=$(printf '%s\n' "$SNAP" | grep -c "GH200" || true)

    # Per-GPU truth: job, node, index, MiB used. gtop prints memory in GB, so a
    # free card shows "mem  0.0/95.6G" and a held one shows e.g. "mem 88.7/95.6G".
    FREELIST=$(printf '%s\n' "$SNAP" | awk '
        /^ ▸ job/   { job = $3 }
        /^   nid/   { node = $1; gpu = -1 }
        /GH200/     { gpu++
                      if (match($0, /mem +[0-9.]+\//)) {
                          m = substr($0, RSTART, RLENGTH)
                          gsub(/[^0-9.]/, "", m)
                          if (m + 0 < 0.05) printf "%s %s gpu%d\n", job, node, gpu
                      } }')
    NFREE=$(printf '%s' "$FREELIST" | grep -c . )
    # gtop has also been seen returning successfully with a well-formed table
    # that silently omits cards: on 2026-09-09 at 19:47Z it parsed cleanly and
    # yielded 0 free across every allocation, while nvidia-smi on nid011318
    # ninety seconds later read cards 0/1/2 at 1, 1, 3 MiB. A confident zero off
    # a truncated table is worse than a timeout, because the timeout announces
    # itself. So check the table's SIZE against what Slurm says we hold: four
    # cards per allocated node. Well under that means the table is partial and
    # its zero means nothing.
    NNODE=$(squeue -u "$USER" -h -t RUNNING -o "%D" 2>/dev/null | awk '{s+=$1} END{print s+0}')
    EXPECT=$(( NNODE * 4 ))
    if [ "${GTOP_RC:-1}" -ne 0 ] || [ "${NGH:-0}" -eq 0 ]; then
        NFREE=-1
    elif [ "$EXPECT" -gt 0 ] && [ "${NGH:-0}" -lt $(( EXPECT / 2 )) ]; then
        NFREE=-1
    fi

    PEND=$(squeue -u "$USER" -h -t PENDING -o "%i %j %D %R" 2>/dev/null)
    NPEND=$(printf '%s' "$PEND" | grep -c . )
    DEAD=$(printf '%s\n' "$PEND" | grep -i "DependencyNeverSatisfied" || true)

    # Inode headroom: currently the binding constraint on everything, and it
    # changes without warning, so it belongs in the same snapshot.
    # Pull the files column by matching the filesystem row rather than by line
    # number: the header wraps differently on compute and login nodes, and a
    # line-number parse returned "-/0" on compute -- a broken field that reads as
    # a real measurement, which is the exact failure class this watcher exists
    # to avoid.
    INODES=$(lfs quota -p 1483804535 /lus/lfs1aip2 2>/dev/null \
             | awk '/lfs1aip2/ {gsub(/\*/,"",$6); print $6"/"$8; exit}')
    [ -z "${INODES:-}" ] && INODES="unreadable"

    # Holder-shaped allocations, minus the ones whose owner is already
    # computing on them. On 2026-09-09 a line attached to 6438897 after
    # seeing empty cards and no owner steps; the owner started its own
    # steps five minutes later and the two workloads shared cards for 23
    # minutes, which costs 2.29x wall-clock and 0.106 bpb silently. So an
    # allocation carrying any non-.batch step is not listed here at all.
    BUSY=$(squeue -u "$USER" -h -s -o "%i %j" 2>/dev/null \
           | awk '$2 != "batch" { split($1, a, "."); print a[1] }' | sort -u)
    HOLDS=$(squeue -u "$USER" -h -t RUNNING -o "%i %j %D %N %L" 2>/dev/null \
            | grep -iE "hold|place" \
            | awk -v busy="$BUSY" '
                BEGIN { n = split(busy, b, "\n"); for (i = 1; i <= n; i++) skip[b[i]] = 1 }
                !($1 in skip)' || true)
    NHOLD=$(printf '%s' "$HOLDS" | grep -c . )

    if { [ "$NFREE" -ge 4 ] && [ "$NPEND" -ge 1 ]; } || [ -n "$DEAD" ]; then
        {
            echo "$TS  ACTIONABLE  free_gpus=$NFREE pending=$NPEND inodes=$INODES"
            printf '%s\n' "$FREELIST" | sed 's/^/    free: /'
            printf '%s\n' "$PEND"     | sed 's/^/    pending: /'
            [ -n "$DEAD" ] && printf '%s\n' "$DEAD" | sed 's/^/    !! DEAD CHAIN (never runs, needs a human): /'
            echo "    -> own line: attach. other lines: report the job id, do not"
            echo "       attach a second copy onto the same output directory."
        } >> "$LOG"
    elif [ "$NFREE" -lt 0 ]; then
        # Probe failed. Say so, and fall back to the one signal that does not
        # go through gtop: placeholder allocations in the RUNNING list, which
        # exist to be attached to. Verify the cards inside the step, never here
        # -- a census is stale within a minute.
        # And say it plainly: no owner step now does NOT mean the owner is done.
        # Only an owner saying so out loud does.
        {
            echo "$TS  PROBE FAILED  gtop rc=${GTOP_RC} gh200_rows=${NGH} of ~${EXPECT} expected  pending=$NPEND inodes=$INODES"
            [ -n "$HOLDS" ] && printf '%s\n' "$HOLDS" | sed 's/^/    holder, no owner step right now (NOT a release -- ask the owner, verify cards inside the step): /'
            [ -n "$DEAD" ] && printf '%s\n' "$DEAD" | sed 's/^/    !! DEAD CHAIN (never runs, needs a human): /'
        } >> "$LOG"
    else
        echo "$TS  quiet  free_gpus=$NFREE (gtop rows $NGH of ~$EXPECT) pending=$NPEND holders_no_step=$NHOLD inodes=$INODES" >> "$LOG"
    fi

    sleep "$INTERVAL"
done
