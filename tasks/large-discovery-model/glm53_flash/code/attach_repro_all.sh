#!/bin/bash
# attach 版复现启动器:把服务 + 6 条 campaign 放进一个已有分配。
#
# 与 sbatch 版(launch_repro_selfhosted.sbatch)的关键差异 —— CUDA_VISIBLE_DEVICES:
#   sbatch + `--exclusive --gres=gpu:1`:Slurm 已隔离一张卡,step 内 index 恒为 0,
#     再设 CVD 会指向不存在的设备(6216995 的死因)。
#   attach + `--overlap`:重叠 step 可能拿到同一张物理卡,所以要 `--gres=gpu:4`
#     让 step 看见全部四张,再用 CVD 钉住其中一张。
# 同一个变量,两种启动方式下一个是毒药一个是解药。
set -u
T=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/glm53_flash
JOB=${JOB:?需要 JOB=<分配号>}
GLM_NODE=${GLM_NODE:?需要 GLM_NODE}
NODE_P=${NODE_P:?需要 NODE_P(放 4 条的节点)}
NODE_Q=${NODE_Q:?需要 NODE_Q(放 2 条的节点)}
Q_GPUS=${Q_GPUS:-1 2}      # NODE_Q 上可用的卡号(0 号可能有邻居)
STAMP=$(date -u +%Y%m%dT%H%MZ)

echo "[attach] job=$JOB glm=$GLM_NODE p=$NODE_P q=$NODE_Q stamp=$STAMP"

# ---- 1. 服务(占满 4 卡,TP4) ----
setsid nohup srun --overlap --jobid=$JOB -w "$GLM_NODE" --ntasks=1 --gres=gpu:4 \
    --cpus-per-task=64 --job-name=glm53-vllm --cpu-bind=none \
    bash "$T/code/serve_vllm_sif.sh" \
    > "$T/logs/at_glm_${STAMP}.log" 2>&1 &
echo "[attach] 服务已起,等健康(最多 30 分钟)"

# ---- 2. 等健康:不通过就不起跑,绝不空跑 ----
READY=0
for i in $(seq 1 180); do
    curl -s --max-time 5 -o /dev/null "http://${GLM_NODE}:8383/health" && { READY=1; break; }
    sleep 10
done
[ "$READY" = "1" ] || { echo "[attach] FATAL: 服务 30 分钟未就绪,不起跑"; exit 4; }
echo "[attach] GLM 就绪,用时 $((i*10))s"

# ---- 3. 6 条 campaign,每条钉一张卡 ----
launch() {  # $1=node $2=gpu $3=arm $4=seed
    setsid nohup srun --overlap --jobid=$JOB -w "$1" --ntasks=1 --gres=gpu:4 \
        --cpus-per-task=48 --job-name="repro-$3$4" --cpu-bind=none \
        bash -c "export CUDA_VISIBLE_DEVICES=$2; ARM=$3 SEED=$4 GLM_HOST=$GLM_NODE bash $T/code/run_repro_arm.sh" \
        > "$T/logs/at_arm$3_s$4_${STAMP}.log" 2>&1 &
    echo "[attach] arm$3 seed$4 -> $1 cvd=$2"
    sleep 2
}
# 同种子的 A/B 落同一节点,让节点差异在成对比较里抵消
launch "$NODE_P" 0 A 1
launch "$NODE_P" 1 B 1
launch "$NODE_P" 2 A 2
launch "$NODE_P" 3 B 2
set -- $Q_GPUS
launch "$NODE_Q" "$1" A 3
launch "$NODE_Q" "$2" B 3

echo "[attach] 全部起完,stamp=$STAMP"
echo "$STAMP" > "$T/results/last_attach_stamp"
