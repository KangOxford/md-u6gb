#!/bin/bash
# (7.1) message-level perplexity —— attach 到已有 allocation 的空闲卡上跑。
#
# 走 node_wrapper.sh 的 MODEL_ZOO_CE_EVAL=1 分支，因为它会先挂 squashfs 再
# exec evaluate_model_zoo_ce.py，CUDA / NCCL / 库路径与训练逐字一致。
#
# 留出集必须是 **2026-01**：baseline 的 TRAIN_DATE_RANGE 到 2025-12-31，
# 2025-12 是训练数据，拿它算 CE 会得到一个偏低且无意义的数。
#
# 口径说明：26tok 的 26 个位全部计入损失（mask 为全 1），所以
#     nats/message = nats/token x 26
# 严格成立。**跨编码时这条不成立**，因为 26tok 每条消息 26 位里有 10 位是
# 近乎确定的 ref 字段，等于在分母里塞白送的位；本任务两臂同为 26tok，可比。
#
# 必需 env: ATTACH_JOBID NODE ARCHITECTURE CE_CHECKPOINT_ROOT ARM_ID
set -uo pipefail
: "${ATTACH_JOBID:?}" "${NODE:?}" "${ARCHITECTURE:?}" "${CE_CHECKPOINT_ROOT:?}" "${ARM_ID:?}"
TASKDIR=/lus/lfs1aip2/projects/public/u6gb/tasks/hybrid_system_20260811/05_hybrid_mamba3_lob
export WORKDIR=/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/hybrid-mamba3-nemotron-20260811
GPU=${CE_GPU:-3}                       # 邻居的单卡作业占 GPU0，默认用 GPU3
MAX_RESIDUAL_MIB=${MAX_RESIDUAL_MIB:-4096}

echo "[gate] $NODE GPU$GPU，阈值 ${MAX_RESIDUAL_MIB} MiB"
GATE=$(timeout 120 srun --jobid="$ATTACH_JOBID" --overlap --nodes=1 --ntasks=1 \
  -w "$NODE" --cpu-bind=none bash -c '
    m=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F", *" -v g="'"$GPU"'" "\$1==g{print \$2}")
    echo "GATE $(hostname) gpu'"$GPU"'_used_mib=${m:-999999}"' 2>&1 | grep "^GATE")
echo "$GATE"
USED=$(echo "$GATE" | sed -E 's/.*_used_mib=([0-9]+).*/\1/')
[ -n "$USED" ] && [ "$USED" -le "$MAX_RESIDUAL_MIB" ] || {
    echo "FATAL: $NODE GPU$GPU 上残留 ${USED:-?} MiB，中止" >&2; exit 3; }

for v in $(env | grep -oE '^SLURM[A-Z_]*'); do unset "$v"; done
mkdir -p "$TASKDIR/results/ce"
export MODEL_ZOO_CE_EVAL=1
export CE_CHECKPOINT_ROOT
export CE_OUTPUT_JSON="$TASKDIR/results/ce/ce_${ARM_ID}_$(date -u +%Y%m%dT%H%M%SZ).json"
export ARCHITECTURE
export JAX_SEED=${JAX_SEED:-42}
export CE_TICKERS=${CE_TICKERS:-GOOG}
export CE_DATE_RANGE=${CE_DATE_RANGE:-2026-01-01,2026-01-31}
export CE_BATCH_SIZE=${CE_BATCH_SIZE:-8}
export CE_MAX_BATCHES=${CE_MAX_BATCHES:-32}
export CE_TARGETS=${CE_TARGETS:-99}          # 目标只用于打印，设成够不着的值
export CE_STEPS=${CE_STEPS:-all}
export QUANT_ROOT=/lus/lfs1aip2/projects/public/s5e/quant_team/quant
export CONDA_ENV=base
export DATA_ROOT=/lus/lfs1aip2/projects/public/s5e/quant_team/lob_preproc_sp500_squashfs
export SQUASHFS_MULTI_MODE=1
export SQUASHFS_DIR="$DATA_ROOT"
export SQUASHFS_MONTHS=2026-01
export SQUASHFS_MULTI_MOUNT_ROOT="/tmp/kangli.u6gb/ce_${ARM_ID}_$(date -u +%Y%m%dT%H%M%SZ)"
export FORBID_RAW_NPYZST=1
export GPUS_PER_NODE=1
export TOKEN_MODE=26tok
export USE_WANDB=False
export SLURM_JOB_ID="$ATTACH_JOBID"

echo "[ce] arm=$ARM_ID arch=$ARCHITECTURE ckpt=$CE_CHECKPOINT_ROOT"
echo "[ce] held-out=$CE_DATE_RANGE  batches=$CE_MAX_BATCHES x bsz=$CE_BATCH_SIZE"
echo "[ce] out=$CE_OUTPUT_JSON"
exec srun --jobid="$ATTACH_JOBID" --overlap --exact --nodes=1 --ntasks=1 \
     -w "$NODE" --cpus-per-task=32 --cpu-bind=none \
     bash -c "CUDA_VISIBLE_DEVICES=$GPU exec bash '$WORKDIR/run/base_model/node_wrapper.sh'"
