#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Hybrid Mamba3 x Nemotron — 正式训练，独立 sbatch（不 attach）
#
# 为什么不继续 attach：本任务在 2026-08-12 连续三次冒烟失败，两次的根因都来自
# 「分配是共享的」——5980502 上有 CRPS 的推理占着 71.5 GB，5980745 属于 BPE
# R6.2。attach 省下的排队时间，被残留挂载与显存争用的排查成本吃光了。正式训练
# 要跑 5 小时，值得用一个干净的独立分配。
#
# 用法： bash submit_hybrid_prod.sh          # 提交
#        DRY_RUN=1 bash submit_hybrid_prod.sh  # 只打印配置与命令
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

WORKDIR=/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/hybrid-mamba3-nemotron-20260811
TASKDIR=/lus/lfs1aip2/projects/public/u6gb/tasks/hybrid_system_20260811/05_hybrid_mamba3_lob
NNODES=${NNODES:-4}
WALLTIME=${WALLTIME:-06:00:00}

# ── 模型：与 baseline 逐项相同，只换 ARCHITECTURE 与 attention 配置 ──────────
export ARCHITECTURE=hybrid_mamba3
export D_MODEL=640
export N_LAYERS=6
export BLOCKS=20
export SSM_SIZE_BASE=640
export PER_GPU_BSZ=4
export SSM_LR_BASE=8.0e-4
export OPT_CONFIG=muon
export MUON_LR=0.01
export WEIGHT_DECAY=0.005
export MAMBA3_D_STATE=128
export MAMBA3_EXPAND=2
export MAMBA3_HEADDIM=64
export MAMBA3_CHUNK_SIZE=64
export MAMBA3_ROPE_FRACTION=0.5
export MAMBA3_USE_TRITON=False
export TOKEN_MODE=26tok
export MSG_SEQ_LEN=500

# Hybrid：位置由 registry 按 Nemotron 规则算（L=6 -> layer 3）。
# heads=10 使 head_dim=640/10=64，满足 Pallas 因果核的 <=256 且 %8==0；
# 否则回退到物化 LxL，13k token 下必然 OOM。
export HYBRID_ATTN_HEADS=10
export HYBRID_ATTN_FLASH=True
export HYBRID_ATTN_PE=False

# ── 并行与批量（与 baseline 相同）────────────────────────────────────────────
export GPUS_PER_NODE=4
export TP_SIZE=1
export GRAD_ACCUM_STEPS=1
export HIERARCHICAL=True
export REMAT=0
# 实测 hybrid 峰值 67.1 GB；0.85 ≈ 83 GB，比峰值高 24%，且给 cuBLAS/Pallas
# 留出初始化空间（0.90 在共享卡上正是冒烟三号 cublas 失败的直接原因）。
export MEM_FRACTION=0.85

# ── 数据：与 baseline 逐字相同的 48 月 x 8 票 ────────────────────────────────
MONTHS=""
for y in 2022 2023 2024 2025; do
  for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
    MONTHS="${MONTHS:+$MONTHS,}${y}-${m}"
  done
done
export SQUASHFS_MULTI_MODE=1
export SQUASHFS_DIR=/lus/lfs1aip2/projects/public/s5e/quant_team/lob_preproc_sp500_squashfs
export SQUASHFS_MONTHS="$MONTHS"
export FORBID_RAW_NPYZST=1
export TICKERS="GOOG,AAPL,NVDA,AMZN,META,TSLA,MSFT,AMD"
export TRAIN_DATE_RANGE="2022-01-01,2025-12-31"
export NO_VALIDATION=True
export N_DATA_WORKERS=12
export QUANT_ROOT=/lus/lfs1aip2/projects/public/s5e/quant_team/quant
export CONDA_ENV=base
export DATA_ROOT=/lus/lfs1aip2/projects/public/s5e/quant_team/lob_preproc_sp500_squashfs

# ── 时长、LR schedule 与 checkpoint ──────────────────────────────────────────
# COSINE_STEPS 覆盖 total_steps（src/lob/train.py:431-433）。一个 epoch 是
# 939,147 步，按它铺 schedule 会让 warmup 吃掉 30% 训练且全程不退火，而退火段
# 恰恰是 loss 降得最多、生成质量真正成型的一段。32,000 与 baseline 完全一致。
export EPOCHS=1
export MINI_EPOCHS=1
export COSINE_STEPS=32000
export CURTAIL_EPOCHS=32000     # 单位是步不是 epoch
export MAX_JOB_HOURS=5.5        # hybrid 每步慢 10.5%，比 baseline 的 5.0 放宽
export CHECKPOINT_EVERY=3000
export DISABLE_STEP_WATCHDOG=0
export WATCHDOG_TIMEOUT=1800
export STEP_TIMEOUT=600

# ── W&B：新设置起新 project（A9），一律 online ───────────────────────────────
export USE_WANDB=True
export WANDB_MODE=online
export WANDB_ENTITY=oxford-lob
export WANDB_PROJECT=sp500-hybrid-mamba3-35m
export WANDB_DIR=/local/user/1483804540

echo "════════════════════════════════════════════════════════════════"
echo " Hybrid Mamba3 x Nemotron — 正式训练（独立 sbatch）"
echo "════════════════════════════════════════════════════════════════"
printf " %-22s %s\n" \
  "architecture" "$ARCHITECTURE (attention: nemotron rule, L=6 -> layer 3)" \
  "attention" "heads=$HYBRID_ATTN_HEADS head_dim=$((D_MODEL / HYBRID_ATTN_HEADS)) flash=$HYBRID_ATTN_FLASH pe=$HYBRID_ATTN_PE" \
  "d_model / n_layers" "$D_MODEL / $N_LAYERS" \
  "params (预期)" "35,435,423  (baseline 33,610,439, +5.43%)" \
  "global batch" "$((PER_GPU_BSZ * GPUS_PER_NODE * NNODES))  ($PER_GPU_BSZ/GPU x $GPUS_PER_NODE x $NNODES 节点)" \
  "optimizer" "$OPT_CONFIG muon_lr=$MUON_LR ssm_lr=$SSM_LR_BASE wd=$WEIGHT_DECAY" \
  "schedule" "warmup 1% -> cosine, COSINE_STEPS=$COSINE_STEPS" \
  "data" "48 月 2022-01..2025-12 x 8 票, 26tok, msg_seq_len=$MSG_SEQ_LEN (13000 tok)" \
  "steps / walltime" "$CURTAIL_EPOCHS 步 / 上限 ${MAX_JOB_HOURS}h (sbatch $WALLTIME)" \
  "checkpoint" "每 $CHECKPOINT_EVERY 步" \
  "mem_fraction" "$MEM_FRACTION (实测峰值 67.1 GB)" \
  "wandb" "$WANDB_PROJECT @ $WANDB_ENTITY ($WANDB_MODE)"
echo "════════════════════════════════════════════════════════════════"

CMD=(sbatch --nodes="$NNODES" --time="$WALLTIME"
     --job-name=hybrid-m3-nemotron-train
     --output="$TASKDIR/logs/prod_%j.out"
     "$WORKDIR/run/base_model/train_full_autoreg.batch")

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[dry-run] ${CMD[*]}"
    exit 0
fi

cd "$WORKDIR"
"${CMD[@]}"
