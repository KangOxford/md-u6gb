#!/bin/bash
# 等冷环境搬完腾出 /home 空间,然后把训练栈安装挂回计算节点。
set -u
T=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/ldm_rl
MOVELOG=$T/logs/move_cold_envs2.log
for i in $(seq 1 240); do
    grep -q "全部处理完" "$MOVELOG" 2>/dev/null && break
    sleep 30
done
echo "[$(date -u +%H:%M:%S)] 搬迁结束(或等超时),验证 /home 可写 1GB"
PROBE=/home/u6gb/kangli.u6gb/.space_probe
if ! dd if=/dev/zero of="$PROBE" bs=1M count=1024 status=none 2>/dev/null; then
    echo "FATAL: /home 仍然写不下 1GB,不起安装(空跑无意义)"; : > "$PROBE" 2>/dev/null; exit 3
fi
: > "$PROBE"     # 截断而非删除
echo "[$(date -u +%H:%M:%S)] 空间够,起安装"

# 找一个还活着、CPU 空的自有分配来挂
JOB=$(squeue -u "$USER" -h -t RUNNING -o "%i %j %L" | awk '$2=="u6gb-4-node-chain"{print $1; exit}')
[ -n "$JOB" ] || JOB=$(squeue -u "$USER" -h -t RUNNING -o "%i %L" | sort -k2 -r | head -1 | awk '{print $1}')
NODE=$(squeue -h -j "$JOB" -o "%N" | sed 's/[][]//g' | cut -d, -f1 | sed 's/-.*//')
echo "挂到 job=$JOB node=$NODE"
LOG=$T/logs/install_train_$(date -u +%Y%m%dT%H%M%SZ).log
echo "$LOG" > $T/logs/LATEST_INSTALL_LOG
exec srun --overlap --jobid="$JOB" --nodes=1 --ntasks=1 -w "$NODE" \
  --gres=gpu:1 --cpus-per-task=64 --cpu-bind=none --job-name=ldmrl-install \
  bash -c "MAX_JOBS=48 bash $T/code/install_train_env.sh" > "$LOG" 2>&1
