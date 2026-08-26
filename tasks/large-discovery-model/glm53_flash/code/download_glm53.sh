#!/bin/bash
# GLM-5.3-Flash 权重下载(ModelScope ZhipuAI/GLM-5.3-Flash,FP8 ~306GiB,62 分片)
# login 节点纯网络顺序 IO。ModelScope CLI 不走 hf_xet,没有 dsv4 当时的
# rust 线程池打穿 nproc 配额问题;仍留重试循环兜网络抖动。
set -u
DEST=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/models/GLM-5.3-Flash
MS=/home/u6gb/kangli.u6gb/envs/dsv4-venv/bin/modelscope
export MODELSCOPE_CACHE=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/models/ms_cache
mkdir -p "$DEST"

for i in $(seq 1 30); do
    "$MS" download --model ZhipuAI/GLM-5.3-Flash --local_dir "$DEST" && { echo "[dl] DONE round=$i"; break; }
    echo "[dl] retry $i rc=$?"; sleep 15
done

n=$(ls -1 "$DEST"/*.safetensors 2>/dev/null | wc -l)
sz=$(du -sh --apparent-size "$DEST" 2>/dev/null | cut -f1)
echo "[dl] safetensors_shards=$n (expect 62)  total=$sz"
[ "$n" -eq 62 ] && echo "[dl] COMPLETE" || echo "[dl] INCOMPLETE"
