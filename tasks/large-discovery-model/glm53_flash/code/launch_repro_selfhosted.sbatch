#!/bin/bash
#SBATCH --job-name=ldm-repro-selfhosted
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --output=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/glm53_flash/logs/selfhosted_%j.out
# LDM v0.1 C1 复现 —— 服务与实验同作业。
#
# 为什么自带服务:上一次(6204550)把 GLM 留在另一个分配里,那个分配到期后服务
# 随之消失,6 条 campaign 全部在 glm_health=000 下空跑。服务与实验必须同寿命。
#
# 为什么独占:评测是 300 秒定时训练,与人共卡 ⇒ 同样时间跑更少 token ⇒ bpb 变差,
# 争抢会把「哪组邻居更闲」伪装成「哪组方法更好」。
#
# 布局:节点0 = GLM 服务(TP4 占满 4 卡);节点1 = A1/B1/A2/B2;节点2 = A3/B3。
# 同种子的 A/B 落同一节点,节点差异在成对比较里抵消。
set -u
T=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/glm53_flash
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
GLM_NODE=${NODES[0]}
echo "[repro] job=$SLURM_JOB_ID nodes=${NODES[*]} glm_node=$GLM_NODE"

# ---- 1-2. 起 GLM 服务并等健康;起不来就降参重试 ----
# KDA 可用状态块随分配波动(实测 512 / 136 两种),max-num-seqs 太高会直接
# 启动失败。6218011 就是这样让一个 3 节点作业在 7 分钟内报废的,所以这里
# 不一次定死参数,而是逐档降。
READY=0
for MAXSEQS in 32 16 8; do
    export MAXSEQS
    echo "[repro] 起服尝试 max-num-seqs=$MAXSEQS"
    srun --exclusive --nodes=1 --ntasks=1 -w "$GLM_NODE" --gres=gpu:4 --cpus-per-task=64 \
        --job-name=glm53-vllm --cpu-bind=none bash "$T/code/serve_vllm_sif.sh" \
        > "$T/logs/selfhosted_glm_${SLURM_JOB_ID}_ms${MAXSEQS}.log" 2>&1 &
    GLM_PID=$!
    for i in $(seq 1 180); do
        if curl -s --max-time 5 -o /dev/null "http://${GLM_NODE}:8383/health"; then READY=1; break; fi
        kill -0 $GLM_PID 2>/dev/null || { echo "[repro] 服务进程退出(max-num-seqs=$MAXSEQS),降档重试"; break; }
        sleep 10
    done
    [ "$READY" = "1" ] && { echo "[repro] GLM 就绪于 ${GLM_NODE}:8383,max-num-seqs=$MAXSEQS,用时 $((i*10))s"; break; }
done
[ "$READY" = "1" ] || { echo "[repro] FATAL: 三档都起不来,不起跑(空跑无意义)"; exit 4; }

# ---- 3. 起 6 条 campaign,每条独占一张 GPU ----
# 不要自己设 CUDA_VISIBLE_DEVICES:`srun --gres=gpu:1 --exclusive` 已经把一张卡
# 隔离给这个 step,step 内部看到的永远是设备 0。6216995 那次给 GPU 1/2/3 的四条
# 因此指向不存在的设备,每次评测 11 秒返回哨兵值 1e+09。
launch() {  # $1=node $2=arm $3=seed
    srun --exclusive --nodes=1 --ntasks=1 -w "$1" --gres=gpu:1 \
        --cpus-per-task=64 --job-name="repro-$2$3" --cpu-bind=none \
        bash -c "ARM=$2 SEED=$3 GLM_HOST=$GLM_NODE bash $T/code/run_repro_arm.sh" \
        > "$T/logs/sh_arm$2_s$3_${SLURM_JOB_ID}.log" 2>&1 &
    echo "[repro] launched arm$2 seed$3 -> $1"
}
launch "${NODES[1]}" A 1
launch "${NODES[1]}" B 1
launch "${NODES[1]}" A 2
launch "${NODES[1]}" B 2
launch "${NODES[2]}" A 3
launch "${NODES[2]}" B 3

# ---- 4. 等全部 campaign 结束(不等服务:它会一直挂着) ----
wait -n $(jobs -p | grep -v "^${GLM_PID}$") 2>/dev/null
for job in $(jobs -p); do [ "$job" != "$GLM_PID" ] && wait "$job" 2>/dev/null; done
echo "[repro] ALL_ARMS_DONE"
