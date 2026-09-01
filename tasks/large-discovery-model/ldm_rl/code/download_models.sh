#!/bin/bash
# 下三个模型到 Lustre。放 Lustre 而不是 /home:
#   /home 是 101G 硬配额(空间紧),Lustre 还有 83T 空间;
#   而 HF 仓库每个只有 9-16 个文件,对 Lustre 只剩 2.4% 的 inode 几乎没有压力。
# 这正好是两个文件系统各自的强项。
set -u
DEST=${DEST:-/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/models}
PY=/home/u6gb/kangli.u6gb/envs/ldm-rl/bin/python
mkdir -p "$DEST"

# 下载缓存也放 Lustre —— 默认的 ~/.cache/huggingface 会吃掉 /home 配额,
# 而 HF 是先下到缓存再硬链到 local-dir,缓存与目标在同一文件系统才不会翻倍占用。
export HF_HOME="$DEST/.hf_home"
export HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p "$HF_HOME"

say() { echo "[$(date -u +%H:%M:%S)] $*"; }

dl() {   # $1=repo_id  $2=本地目录名
    local repo=$1 name=$2
    if [ -f "$DEST/$name/.download_ok" ]; then say "$name 已完成,跳过"; return; fi
    say "下 $repo -> $DEST/$name"
    $PY - "$repo" "$DEST/$name" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
p = snapshot_download(repo_id=repo, local_dir=dest, max_workers=8)
print("  ->", p)
PY
    if [ $? -eq 0 ]; then
        touch "$DEST/$name/.download_ok"
        say "  $name 完成,$(du -sh "$DEST/$name" 2>/dev/null | cut -f1)"
    else
        say "  !! $name 失败"
    fi
}

dl "Qwen/Qwen2.5-1.5B-Instruct"                        "Qwen2.5-1.5B-Instruct"
dl "Qwen/Qwen3.5-9B"                                   "Qwen3.5-9B"
dl "Yangtze-ailab/LDM-CoT-SFT-Qwen3.5-9B-MixedScience" "LDM-CoT-SFT"

say "=== 全部结束 ==="
for d in Qwen2.5-1.5B-Instruct Qwen3.5-9B LDM-CoT-SFT; do
    printf "  %-26s %s\n" "$d" "$(du -sh "$DEST/$d" 2>/dev/null | cut -f1)"
done
