#!/bin/bash
# 把冷环境从 /home(101G 硬配额,已撞顶)搬到 Lustre。
# 用 rsync --remove-source-files:每个文件**传完并校验后**才解除源端链接,
# 中断可续传;收尾只用 rmdir —— rmdir 对非空目录会失败,所以丢不了数据。
set -u
COLD=/lus/lfs1aip2/projects/public/u6gb/envs_cold
SRC=/home/u6gb/kangli.u6gb/envs

# 等已经在跑的裸 mv 结束(上一版脚本起的)
while pgrep -f "^mv /home/u6gb/kangli.u6gb/envs/" >/dev/null; do sleep 20; done
echo "[$(date -u +%H:%M:%S)] 没有在跑的 mv 了"

move_one() {
  local e=$1
  [ -d "$SRC/$e" ] || { echo "  $e: 源不在,跳过"; return; }
  local n0; n0=$(find "$SRC/$e" -type f 2>/dev/null | wc -l)
  echo "[$(date -u +%H:%M:%S)] $e: 源端 $n0 个文件 -> 开始"
  mkdir -p "$COLD/$e"
  rsync -a --remove-source-files "$SRC/$e/" "$COLD/$e/" 2>&1 | tail -3
  # 只收空目录
  find "$SRC/$e" -depth -type d -exec rmdir {} \; 2>/dev/null
  local left; left=$(find "$SRC/$e" 2>/dev/null | wc -l)
  local n1; n1=$(find "$COLD/$e" -type f 2>/dev/null | wc -l)
  echo "  目标 $n1 个文件;源端残留 $left 项 $([ "$left" = 0 ] && echo '(已清空,目录已收)' || echo '(还有东西,没动)')"
}

for e in glm53-vllm_cu130_deprecated_20260827T0000Z glm53-vllm cu13-test; do move_one "$e"; done
echo "[$(date -u +%H:%M:%S)] 全部处理完"
quota -s 2>/dev/null | tail -2
