# Retry the harvest until every live run has step 1200 on Lustre, then keep going
# for 1350/1500 while the runs last. Exits when nothing is left running.
set -uo pipefail
O=/lus/lfs1aip2/projects/public/u6gb/tasks/crps_analysis_20260905
T=/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z
LOG=$O/logs/harvest_loop.log
for i in $(seq 1 60); do
  live=$(squeue -u "$USER" -s -h -o '%j' 2>/dev/null | grep -c r3rep)
  have=$(ls -d $T/ckpt/wm_ft_r3rep_s4*/*_step1200 2>/dev/null | wc -l)
  echo "[$(date -u +%H:%M:%SZ)] round $i  live=$live  step1200_on_lustre=$have/8" >> $LOG
  STEPS="1050 1200 1350 1500" bash $O/harvest_r3.sh >> $LOG 2>&1
  have=$(ls -d $T/ckpt/wm_ft_r3rep_s4*/*_step1200 2>/dev/null | wc -l)
  echo "[$(date -u +%H:%M:%SZ)] after harvest: step1200_on_lustre=$have/8" >> $LOG
  if [ "$live" -eq 0 ]; then echo "[$(date -u +%H:%M:%SZ)] no runs left; final harvest done" >> $LOG; break; fi
  sleep 300
done
