#!/usr/bin/env bash
# 依次跑 A/B/C 三种存点路径，同一批节点、同一份配置，只差 reshard 的实现。
#
#   A  device_get -> broadcast_one_to_all -> device_put     现状
#   B  device_get -> device_put                             去掉那次全设备 AllReduce
#   C  完全不 reshard                                        直接交给 Orbax
#
# 要回答的不止「会不会死锁」。即使一次都不死，三者的**耗时**也直接决定该怎么修：
# broadcast_one_to_all 把整棵树 jit 成一次融合 psum，实测 8 叶子 180ms、
# 300 叶子 8s —— 开销随叶子数强烈增长，而真实 train state（params + Adam m,v）
# 正好有数百个叶子。
set -uo pipefail
T=/lus/lfs1aip2/projects/public/u6gb/tasks/hybrid_system_20260811/05_hybrid_mamba3_lob
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$T/logs/repro_sweep_${STAMP}"
mkdir -p "$OUT"
ALLOC=${ALLOC:?}; NODELIST=${NODELIST:?}
ITERS=${ITERS:-100}; MB=${MB:-1500}; LEAVES=${LEAVES:-300}
CHURN=${CHURN:-6000}; MEMFRAC=${MEMFRAC:-0.55}; TIMEOUT=${TIMEOUT:-150}
P0=${P0:-29520}
i=0
for M in A B C; do
    i=$((i+1))
    echo "=== mode $M  $(date -u +%H:%M:%S)"
    ALLOC="$ALLOC" NODELIST="$NODELIST" MODE="$M" ITERS="$ITERS" MB="$MB" \
      LEAVES="$LEAVES" CHURN="$CHURN" MEMFRAC="$MEMFRAC" TIMEOUT="$TIMEOUT" \
      PORT=$((P0 + i)) bash "$T/code/run_repro.sh" > "$OUT/mode_${M}.log" 2>&1
    rc=$?
    # 统计：是否挂死、慢次数、平均耗时
    hung=$(grep -ac "WATCHDOG\|判为挂死\|判为死锁" "$OUT/mode_${M}.log" || true)
    slown=$(grep -ac "^\[repro r0.*慢：" "$OUT/mode_${M}.log" || true)
    last=$(grep -a "完成 .* 次" "$OUT/mode_${M}.log" | head -1)
    avg=$(grep -aoE "reshard=[0-9]+ms" "$OUT/mode_${M}.log" | sed 's/[^0-9]//g' \
          | awk '{s+=$1;n++} END{if(n)printf "%.0f", s/n; else print "n/a"}')
    echo "    rc=$rc 挂死行=$hung 慢=$slown 平均 reshard=${avg}ms  ${last}"
done
echo "=== 全部完成，日志在 $OUT"
