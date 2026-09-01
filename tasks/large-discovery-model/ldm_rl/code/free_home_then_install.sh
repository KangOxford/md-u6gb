#!/bin/bash
# 一条链:腾 /home 空间 -> 验证可写 -> 把训练栈安装挂回计算节点。
# 必须在 tmux 里跑(前面两次用 setsid nohup 都被会话结束时收掉了)。
set -u
T=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/ldm_rl
echo "===== 起于 $(date -u +%FT%TZ) ====="
bash $T/code/move_cold_envs.sh 2>&1
echo "===== 搬迁段结束,进入安装段 ====="
bash $T/code/resume_install_after_space.sh 2>&1
echo "===== 全链结束 $(date -u +%FT%TZ) ====="
