#!/bin/bash
# alpha 扫描监控。
#
# 上一版的退出条件是「squeue 里没有 alpha step」，它在**还没启动**和**跑完了**
# 两种情况下取同一个值，于是在第一次启动死掉、第二次启动之前的空档里判定完成，
# 对着零个产物打分。这一版要求两件事同时成立才算结束：
#   (a) 连续 3 次看不到 step（而不是 1 次），且
#   (b) 至少见过一次 step 存在（`seen=1`），或者已经有产物
# 并且中途在产物数达到 20 时先打一次分 —— 部分结果就能定方向。
cd "$(dirname "$0")/.."
seen=0; gone=0; scored20=0
while :; do
    n=$(squeue -s -u kangli.u6gb -o "%j" 2>/dev/null | grep -c '^dfm-alpha')
    p=$(ls rollouts_alpha/alpha_a100_*_2026-01_learned.npz 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] && { seen=1; gone=0; } || gone=$((gone+1))
    if [ "$p" -ge 20 ] && [ "$scored20" -eq 0 ]; then
        scored20=1
        echo "=== 产物达 20 ticker，先打一次分 ($(date -u +%H:%M:%SZ)) ==="
        python3 code/score_alpha.py 2>&1 | tail -30
    fi
    if [ "$seen" -eq 1 ] && [ "$gone" -ge 3 ]; then
        echo "=== alpha step 连续 3 次缺席且此前存在过 -> 真的结束了 ==="
        break
    fi
    [ "$p" -ge 55 ] && { echo "=== 55/60 ticker 已出，提前收 ==="; break; }
    sleep 120
done
echo "=== 最终打分 ($(date -u +%H:%M:%SZ)), 产物 $(ls rollouts_alpha/*.npz 2>/dev/null|wc -l) 个 ==="
python3 code/score_alpha.py 2>&1 | tail -40
