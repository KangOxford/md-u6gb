#!/bin/bash
# 一条链:打包冷环境腾 /home -> 验可写 -> 装训练栈。全部在计算节点上跑。
# 挂到自有分配 6217606 的 nid010402(4/4 全空,剩 16h30m),不排队。
set -u
T=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/ldm_rl
JOB=${JOB:-6217606}
NODE=${NODE:-nid010402}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$T/logs/packinstall_${STAMP}.log
echo "$LOG" > $T/logs/LATEST_INSTALL_LOG

srun --overlap --jobid="$JOB" --nodes=1 --ntasks=1 -w "$NODE" \
     --gres=gpu:1 --cpus-per-task=64 --cpu-bind=none --job-name=ldmrl-install \
     bash -c '
set -u
T='"$T"'
echo "===== 起于 $(date -u +%FT%TZ) 于 $(hostname) ====="

echo "----- 段 1/3: 打包冷环境 -----"
bash $T/code/pack_cold_envs.sh

echo "----- 段 2/3: 验证 /home 能写下 12GB -----"
PROBE=$HOME/.space_probe
if ! dd if=/dev/zero of="$PROBE" bs=1M count=12288 status=none 2>/dev/null; then
    echo "FATAL: /home 写不下 12GB,不起安装(空跑无意义)"
    : > "$PROBE"; quota -s 2>/dev/null | tail -2; exit 3
fi
: > "$PROBE"      # 截断而非删除
echo "OK: 12GB 可写"
quota -s 2>/dev/null | tail -2

echo "----- 段 3/3: 装训练栈 -----"
MAX_JOBS=48 bash $T/code/install_train_env.sh
echo "===== 全链结束 $(date -u +%FT%TZ) rc=$? ====="
' > "$LOG" 2>&1
