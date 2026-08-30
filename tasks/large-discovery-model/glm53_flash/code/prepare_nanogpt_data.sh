#!/bin/bash
# nanoGPT(climbmix)数据准备:下载分片 + 训 8192 词表 BPE + tokenize
# 重活(下载 + 多进程 BPE),必须在计算节点跑,不在登录节点。
# 用法: srun --overlap --jobid=<alloc> -w <node> -n1 --cpus-per-task=64 bash prepare_nanogpt_data.sh
set -u -o pipefail
L=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/Large-Discovery-Models
PY=/home/u6gb/kangli.u6gb/envs/ldm-nanogpt/bin/python
# 数据放 Lustre 项目盘而不是 HOME 默认:分片是几十 GB 级,且计算节点要共享读
export AUTORESEARCH_CACHE_DIR=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/data/autoresearch_cache
SHARDS=${SHARDS:-8}

echo "[prep] host=$(hostname) shards=$SHARDS cache=$AUTORESEARCH_CACHE_DIR"
mkdir -p "$AUTORESEARCH_CACHE_DIR"
cd $L/tasks/nanogpt/scripts
$PY prepare.py --num-shards "$SHARDS" --download-workers 8 || { echo "PREP_FAILED rc=$?"; exit 5; }
echo "[prep] DONE"
du -sh --apparent-size "$AUTORESEARCH_CACHE_DIR" 2>/dev/null
echo "NANOGPT_DATA_READY"
