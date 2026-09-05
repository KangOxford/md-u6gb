# Drain the cell queue onto whatever cards are genuinely free, re-reading gtop
# each round. A cell whose output directory already holds score.json is skipped,
# so the drainer is idempotent and can be restarted.
set -uo pipefail
# gtop lays out to the terminal width and emits NO GH200 lines below ~200 columns.
# In an 80-column tmux window the probe returned nothing and was logged as
# "free cards: 0" -- indistinguishable from a full cluster. Force the width, and
# make a failed probe say so instead of reporting zero.
export COLUMNS=200 LINES=200
O=/lus/lfs1aip2/projects/public/u6gb/tasks/crps_analysis_20260905
LOG=$O/logs/drain.log
MAXPAR=${MAXPAR:-12}
for round in $(seq 1 200); do
  todo=0
  while read -r LABEL CKPT TICKER; do
    [ -f "$O/cells/${LABEL}_${TICKER}/score.json" ] || todo=$((todo+1))
  done < $O/cell_queue.txt
  running=$(squeue -u "$USER" -s -h -o '%j' 2>/dev/null | grep -c '^cell-' || true)
  echo "[$(date -u +%H:%M:%SZ)] round $round  todo=$todo  running=$running" >> $LOG
  [ "$todo" -eq 0 ] && { echo "[$(date -u +%H:%M:%SZ)] queue drained" >> $LOG; break; }
  if [ "$running" -lt "$MAXPAR" ]; then
    timeout 200 gtop --once 2>/dev/null | tr -d '\000' > $O/gtop_raw.txt
    if ! grep -q GH200 $O/gtop_raw.txt; then
      echo "  PROBE FAILED: gtop emitted no GH200 lines ($(wc -c < $O/gtop_raw.txt) bytes)" >> $LOG
      sleep 240; continue
    fi
    awk '
      /^ ▸ job/{j=$3} /^   nid/{n=$1}
      /GH200/{ if ($0 ~ /idle/ && $0 ~ /mem +0\.0\//) { match($0,/\[[0-9]\]/); print j, n, substr($0,RSTART+1,1) } }' \
      $O/gtop_raw.txt > $O/free_now.txt
    nfree=$(wc -l < $O/free_now.txt)
    echo "  free cards: $nfree" >> $LOG
    i=0; declare -A PER=()
    while read -r LABEL CKPT TICKER; do
      [ -f "$O/cells/${LABEL}_${TICKER}/score.json" ] && continue
      squeue -u "$USER" -s -h -o '%j' 2>/dev/null | grep -qx "cell-${LABEL}-${TICKER}" && continue
      [ $(( running + i )) -ge "$MAXPAR" ] && break
      read -r J N C < <(sed -n "$((i+1))p" $O/free_now.txt)
      [ -z "${J:-}" ] && break
      [ "${PER[$N]:-0}" -ge 3 ] && { i=$((i+1)); continue; }
      LABEL=$LABEL CKPT=$CKPT TICKER=$TICKER WANT=$C \
      srun --overlap --jobid=$J --nodelist=$N --nodes=1 --ntasks=1 --gres=gpu:4 \
           --cpu-bind=none --job-name=cell-${LABEL}-${TICKER} \
           bash $O/score_cell.sh > "$O/logs/cell_${LABEL}_${TICKER}.log" 2>&1 < /dev/null &
      echo "  launched ${LABEL}/${TICKER} -> $N card$C" >> $LOG
      PER[$N]=$(( ${PER[$N]:-0} + 1 )); i=$((i+1)); sleep 4
    done < $O/cell_queue.txt
  fi
  sleep 240
done
