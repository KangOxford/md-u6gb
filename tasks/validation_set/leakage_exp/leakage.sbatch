#!/bin/bash
#SBATCH --job-name=valset-leak-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --time=04:30:00
#SBATCH --partition=workq
#SBATCH --output=/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/logs/leakage_sbatch_%j.out
# 独立 allocation 跑泄漏实验（预留 job 的 GPU 被 model-zoo 推理评测合法占用，
# 遵守 CLAUDE.md "Physical GPU gate before overlap"——不与其争抢，排队干净跑）。
exec bash /lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/leakage_exp/run_leakage.sh
