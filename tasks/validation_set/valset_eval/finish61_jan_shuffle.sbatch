#!/bin/bash
#SBATCH --job-name=jance-finish61
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=02:30:00
#SBATCH --partition=workq
#SBATCH --output=/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/logs/jance_finish61_%j.out
# Jan-2026 shuffle 132-ckpt CE 评测收尾：剩余 61 个小档（10M 尾部+6M/4M/1M/0p2M, bsz32）。
# 独立 job 而非 attach 5827830：该节点被 backfill124 占满 8-12h（handoff §9 禁止并行挂，
# gate 在其 per-ckpt 间隙会假开）。OUT_DIR 复用断点目录，已有 71 json 自动跳过。
set -uo pipefail
VE=/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/valset_eval
cd "$VE"
OUT_DIR=$(cat "$VE/latest_jan_results_dir.txt")
echo "[finish61] job=$SLURM_JOB_ID node=$SLURMD_NODENAME out=$OUT_DIR start=$(date -u +%FT%TZ)"
MODE=jan MANIFEST="$VE/manifest_132ckpt.json" TOTAL=132 \
  bash "$VE/parallel_valset.sh" "$OUT_DIR"
echo "[finish61] done jsons=$(ls "$OUT_DIR"/valce_*.json 2>/dev/null | wc -l)/132 end=$(date -u +%FT%TZ)"
