set -uo pipefail
O=/lus/lfs1aip2/projects/public/u6gb/tasks/crps_analysis_20260905
i=0; declare -A PER
while read -r LABEL CKPT TICKER; do
  read -r J N C < <(sed -n "$((i+1))p" $O/free_cards.txt)
  [ -z "${J:-}" ] && { echo "[launch] out of cards at $LABEL/$TICKER"; break; }
  [ "${PER[$N]:-0}" -ge 3 ] && { echo "[launch] $N already has 3; skipping"; i=$((i+1)); continue; }
  LOG=$O/logs/cell_${LABEL}_${TICKER}.log
  LABEL=$LABEL CKPT=$CKPT TICKER=$TICKER WANT=$C \
  srun --overlap --jobid=$J --nodelist=$N --nodes=1 --ntasks=1 --gres=gpu:4 \
       --cpu-bind=none --job-name=cell-${LABEL}-${TICKER} \
       bash $O/score_cell.sh > "$LOG" 2>&1 < /dev/null &
  echo "[launch] $LABEL/$TICKER -> $N card$C"
  PER[$N]=$(( ${PER[$N]:-0} + 1 )); i=$((i+1)); sleep 4
done < $O/cell_queue.txt
echo "[launch] started $i"; wait
