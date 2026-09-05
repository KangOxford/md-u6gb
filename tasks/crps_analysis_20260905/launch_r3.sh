set -uo pipefail
O=/lus/lfs1aip2/projects/public/u6gb/tasks/crps_analysis_20260905
i=0
declare -A PER
while read -r J N C; do
  [ "${PER[$N]:-0}" -ge 3 ] && { echo "[launch] skip $N card$C (already 3 runs; host RAM is the binding constraint)"; continue; }
  [ $i -ge 8 ] && break
  TS=$((40+i)); LOG=$O/logs/r3rep_s${TS}.log
  TSEED=$TS WANT=$C MAXSTEP=1500 \
  srun --overlap --jobid=$J --nodelist=$N --nodes=1 --ntasks=1 \
       --gres=gpu:4 --cpu-bind=none --job-name=r3rep-s${TS} \
       bash $O/r3_replicate.sh > "$LOG" 2>&1 < /dev/null &
  echo "[launch] tseed=$TS -> $N card$C (alloc $J)"
  PER[$N]=$(( ${PER[$N]:-0} + 1 )); i=$((i+1)); sleep 5
done < $O/free_cards.txt
echo "[launch] started $i runs"
wait
