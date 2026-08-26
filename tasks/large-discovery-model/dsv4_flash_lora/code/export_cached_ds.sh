#!/bin/bash
# 离线构建 cached dataset(ms-swift 官方路径,docs Command-line-parameters cached_dataset 节)。
#
# 为什么必须离线:datasets 的 map 缓存名 = 内容+函数闭包哈希。self-cognition 的
# 预处理闭包里有随运行变化的输入,三次运行(mini/full16/64g)在同一缓存目录留下
# 三个不同名的 cache-*.arrow —— 「同 cap 预热」对它不成立,64 rank 冷竞写必崩
# (6144379 死于 alpaca-zh 冷缓存,6146475 死于 self-cognition 冷缓存,同族)。
# cached_dataset 把 tokenize 全部挪到这里单进程做完,训练时 64 rank 只 mmap 只读
# arrow,运行时不再有 map、不再有 fingerprint、不再有竞态。
#
# 用法: bash export_cached_ds.sh   (登录节点直接跑,CPU-only,~2 分钟)
set -euo pipefail
source /lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/dsv4_flash_lora/code/env_dsv4.sh

CAP=${DATASET_CAP:-600}
OUT=$DSV4_TASK/data/cached_ds_cap${CAP}

if [ -d "$OUT/train" ]; then
    echo "[export] $OUT/train 已存在,跳过(要重建先 mv 走旧目录)"; exit 0
fi

# 预处理相关参数与 train_lora.sh 逐项镜像(模板/裁剪/切分必须一致,训练侧才可比)
swift export \
    --model $MODEL_DIR \
    --dataset "AI-ModelScope/alpaca-gpt4-data-zh#${CAP}" \
              "AI-ModelScope/alpaca-gpt4-data-en#${CAP}" \
              "swift/self-cognition#${CAP}" \
    --model_author swift \
    --model_name swift-robot \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --split_dataset_ratio 0.01 \
    --max_length 4096 \
    --dataset_num_proc 4 \
    --to_cached_dataset true \
    --output_dir $OUT

echo "[export] DONE -> $OUT"
