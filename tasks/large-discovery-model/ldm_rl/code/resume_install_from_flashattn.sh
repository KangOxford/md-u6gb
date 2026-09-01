#!/bin/bash
# 从 flashattn 段起续跑剩下的安装段。
# 前面的 env / sglang 两段已经成功(torch 2.11.0+cu129,cuda avail True),不重跑。
#
# 用 bash -lc 起 srun:非登录 shell 里 `module` 函数没定义,gcc 会停在 7.5.0,
# 而 torch 的头文件直接 #error 要求 >= 9。脚本内部也有 lmod 初始化兜底,
# 这里再走一层登录 shell 是双保险。
set -u
T=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/ldm_rl
JOB=${JOB:-6217606}
NODE=${NODE:-nid010402}
LOG=$T/logs/install_resume_$(date -u +%Y%m%dT%H%M%SZ).log
echo "$LOG" > $T/logs/LATEST_INSTALL_LOG
echo "日志 $LOG"

setsid nohup srun --overlap --jobid="$JOB" --nodes=1 --ntasks=1 -w "$NODE" \
  --gres=gpu:1 --cpus-per-task=96 --cpu-bind=none --job-name=ldmrl-install \
  bash -lc "
    module load gcc-native/12.3 2>/dev/null || true
    echo \"[resume] 节点 \$(hostname)  gcc=\$(gcc -dumpversion)\"
    for s in flashattn te tms router megatron slime patch verify; do
        echo \"########## 段 \$s ##########\"
        STAGE=\$s MAX_JOBS=32 bash $T/code/install_train_env.sh || {
            echo \"!! 段 \$s 失败,停在这里\"; exit 1; }
    done
    echo '########## 全部段完成 ##########'
  " > "$LOG" 2>&1 &
echo "launched pid=$!"
