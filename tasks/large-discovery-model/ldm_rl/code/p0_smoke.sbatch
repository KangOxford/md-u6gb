#!/bin/bash
#SBATCH --job-name=ldmrl-p0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --partition=workq
#SBATCH --output=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/ldm_rl/logs/p0_%j.out
#
# P0 关卡:1.5B GRPO 冒烟(真 reward,真 docking)。
#
# 为什么是 sbatch 而不是 attach:提交时两个自有分配的 16+16 张卡里只剩 1 张空的,
# 其余都在跑别的战线(利用率 88-100%)。attach-first 的前提「有空卡」此刻不成立。
#
# 为什么 --exclusive 且要 4 卡:run_train_real_slime.sh 内部按
# CUDA_VISIBLE_DEVICES=1,2,3 分卡(actor 1 张 + sglang rollout 2 张),
# 那是 slime 自己的分配方式。申请 4 张时 step 内看到 0,1,2,3(已实测),
# 1,2,3 有效;只申请 3 张的话设备 3 不存在,而那类错误不报错,
# 只会让那一路指向不存在的卡。
set -u
T=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/ldm_rl
source "$T/code/site_env.sh"

export EPISODES=$REPO_ROOT/rl_episodes_sm_acqmax.jsonl
export SAVE=$REPO_ROOT/rl/qwen2.5-1.5B_slime_p0_smoke

echo "[p0] job=$SLURM_JOB_ID 节点=$(hostname)"
echo "[p0] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<未设>}"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "[p0] EPISODES=$EPISODES"
echo "[p0] SAVE=$SAVE"

# 看门狗:训练卡住时自动 dump Python 栈。
# 2026-09-01 P0 在 "Creating placement group with 3 GPUs..." 之后停了 16 分钟,
# CPU 负载 0.02、GPU 全空、Ray 集群健康且无 pending demand、驱动的 core-worker
# 日志在 ray.init 之后一行都没有 —— 一个没有任何输出的死锁。而**别的 srun step
# 看不到 batch step 的进程**(PID 命名空间隔离),事后没法 attach 上去看栈。
# 所以把 py-spy 放进同一个 step 里,由它在同一个命名空间中定期取栈。
watchdog() {
    local logdir="$1" last_size=0 stall=0
    sleep 240
    while :; do
        local sz; sz=$(stat -c %s "$logdir" 2>/dev/null || echo 0)
        if [ "$sz" = "$last_size" ]; then
            stall=$((stall+1))
            if [ "$stall" -ge 2 ]; then     # 日志 4 分钟没长
                echo "=== [watchdog] 日志 $((stall*120))s 没有增长,dump 所有 python 栈 ==="
                for pid in $(pgrep -u "$USER" -f "train\.py" 2>/dev/null); do
                    echo "--- py-spy dump pid $pid ---"
                    "$CONDA_PREFIX/bin/py-spy" dump --pid "$pid" --nonblocking 2>&1 | head -60
                done
                stall=0
            fi
        else
            stall=0
        fi
        last_size=$sz
        sleep 120
    done
}
watchdog "/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/ldm_rl/logs/p0_${SLURM_JOB_ID}.out" &
_WD=$!
trap 'kill $_WD 2>/dev/null' EXIT

srun --ntasks=1 --gres=gpu:4 --cpus-per-task=128 --cpu-bind=none \
     --job-name=ldmrl-p0-train \
     bash -lc "module load gcc-native/12.3 2>/dev/null || true
               source $T/code/site_env.sh >/dev/null 2>&1
               export EPISODES=$REPO_ROOT/rl_episodes_sm_acqmax.jsonl
               export SAVE=$REPO_ROOT/rl/qwen2.5-1.5B_slime_p0_smoke
               bash \$LAUNCH/run_train_real_slime.sh"
echo "[p0] 结束 rc=$?"
