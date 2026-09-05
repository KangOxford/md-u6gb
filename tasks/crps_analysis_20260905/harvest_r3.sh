# Pull node-local checkpoints for the given steps onto Lustre. Read-only on the
# trainer's side; never removes anything.
set -uo pipefail
O=/lus/lfs1aip2/projects/public/u6gb/tasks/crps_analysis_20260905
T=/lus/lfs1aip2/projects/public/u6gb/tasks/crps_return_alignment_20260808T025024Z
STEPS=${STEPS:?e.g. STEPS="1050 1200"}
declare -A NODE JOB PREFIX
while read -r s n j p; do NODE[$s]=$n; JOB[$s]=$j; PREFIX[$s]=$p; done < $O/run_map.txt
for n in $(printf '%s\n' "${NODE[@]}" | sort -u); do
  seeds=""; jid=""
  for s in "${!NODE[@]}"; do [ "${NODE[$s]}" = "$n" ] && { seeds="$seeds $s"; jid=${JOB[$s]}; }; done
  args=""
  for s in $seeds; do args="$args ${s}:${PREFIX[$s]}"; done
  echo "[harvest] $n (alloc $jid): seeds$seeds"
  srun --overlap --jobid=$jid --nodelist=$n --nodes=1 --ntasks=1 --gres=gpu:4 \
       --cpu-bind=none --job-name=harvest-r3 bash -c "
    for pair in $args; do
      s=\${pair%%:*}; pre=\${pair#*:}
      dest=$T/ckpt/wm_ft_r3rep_s\$s
      mkdir -p \"\$dest\"
      for st in $STEPS; do
        src=\"\${pre}_step\${st}\"
        if [ -d \"\$src\" ]; then
          cp -a --update \"\$src\" \"\$dest\"/ && echo \"  s\$s step\$st -> ok (\$(du -sm \"\$src\" 2>/dev/null | cut -f1) MB)\"
        else
          echo \"  s\$s step\$st -> not yet on node\"
        fi
      done
    done" < /dev/null 2>&1 | grep -v '^srun:'
done
