#!/bin/bash
# 训 G12C 活性模型(E9 的前置)。与训练栈无关,可与安装并行。
#
# 关键是**与现成的 G12D 模型用同一套流程**,否则 E9 的
# 「G12C(迁移) vs G12D(同分布)」比较会把「迁移差异」和「评测器不同」混在一起。
# 下面的参数逐项抄自 best_g12d_model_metadata.json:
#   endpoint IC50 · exact_relation_only(默认只留 '=') · 直接测定
#   random_seed 714 · test_size 0.2 · 按 scaffold split 的 RMSE 选模型
#   选中的是 ensemble_nn_ridge_rf(cpu_sklearn),GPU 基准只作诊断
#
# HANDOFF §6 提到的 g12c_docking_benchmark.csv 在代码里不存在(全仓库无引用),
# 那句提法与脚本对不上,这里不用它。
set -u
T=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/ldm_rl
REPO=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/LDM-rl
PY=/home/u6gb/kangli.u6gb/envs/ldm-rl/bin/python
OUT=$T/results/g12c_qsar_$(date -u +%Y%m%dT%H%M%SZ)
CACHE=$T/results/chembl_cache

mkdir -p "$OUT" "$CACHE"
cd "$REPO"
export PYTHONPATH=$REPO/rl:$REPO:${PYTHONPATH:-}

echo "[g12c] 输出 -> $OUT"
echo "[g12c] ChEMBL 缓存 -> $CACHE"
exec "$PY" tasks/small_molecule/core/activity_modeling/train_g12c_qsar.py \
    --output-dir "$OUT" \
    --cache-dir  "$CACHE" \
    --mutation G12C \
    --endpoint IC50 \
    --random-seed 714 \
    --test-size 0.2 \
    --workers 16 \
    --optuna-trials 0
