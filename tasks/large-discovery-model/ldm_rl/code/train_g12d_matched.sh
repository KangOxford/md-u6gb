#!/bin/bash
# 用**当前这份** train_g12c_qsar.py 重训 G12D 活性模型,得到与 G12C 流程匹配的一对评测器。
#
# 为什么需要:仓库里现成的 best_g12d_model.joblib 是**旧版流程**的产物 ——
#   现成 G12D 的 splits = random / scaffold / source_assay / assay_family(没有 document)
#   当前脚本的 splits  = random / scaffold / assay / document
#   现成 G12D 选中 ensemble_nn_ridge_rf;当前脚本的名册里是 ensemble_nn_ridge_lgbm
# 选模型的规则是固定优先级 (document, scaffold, assay, random) 取第一个可用的,
# 所以现成的那个落在 scaffold,我们跑出来的落在 document —— 不是参数没对齐,是流程不同。
#
# 后果落在 E9:设计要「评 G12C(迁移) + G12D(同分布)」。两侧若用两个不同流程做出来的
# 活性模型,性能差里就同时含**真实的迁移难度**与**两个评测器自身的精度差**,分不开。
# 所以再花几分钟把 G12D 也用当前脚本跑一遍,E9 用这一对做主结果;
# 仓库自带的那个保留,用于与作者的数对账。
set -u
T=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/ldm_rl
REPO=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/LDM-rl
PY=/home/u6gb/kangli.u6gb/envs/ldm-rl/bin/python
OUT=$T/results/g12d_qsar_matched_$(date -u +%Y%m%dT%H%M%SZ)
CACHE=$T/results/chembl_cache

mkdir -p "$OUT" "$CACHE"
cd "$REPO"
export PYTHONPATH=$REPO/rl:$REPO:${PYTHONPATH:-}

echo "[g12d-matched] 输出 -> $OUT"
echo "[g12d-matched] ChEMBL 缓存 -> $CACHE"
exec "$PY" tasks/small_molecule/core/activity_modeling/train_g12c_qsar.py \
    --output-dir "$OUT" \
    --cache-dir  "$CACHE" \
    --mutation G12D \
    --endpoint IC50 \
    --random-seed 714 \
    --test-size 0.2 \
    --workers 16 \
    --optuna-trials 0
