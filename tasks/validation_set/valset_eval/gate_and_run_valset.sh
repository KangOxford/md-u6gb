#!/bin/bash
# 预声明 GPU gate + 全量 valset eval：在节点上本地轮询 GPU1-3 显存，
# 三卡均 < GATE_MB 且连续 2 次采样通过后，启动 run_valset_ce.sh（33 ckpt 全量）。
# 绝不 kill 任何现有进程；gate 超时则报错退出。
# 用法: gate_and_run_valset.sh <OUT_DIR>
set -euo pipefail
OUT_DIR="$1"
GATE_MB=2000
GATE_TIMEOUT_S=$((11 * 3600))
POLL_S=60
VE_DIR=/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/valset_eval

echo "[gate] waiting for GPU 1,2,3 used_mem < ${GATE_MB} MiB (poll ${POLL_S}s, timeout ${GATE_TIMEOUT_S}s)"
t0=$(date +%s); pass=0
while :; do
    now=$(date +%s)
    if (( now - t0 > GATE_TIMEOUT_S )); then
        echo "[gate] TIMEOUT after $((now-t0))s — giving up"; exit 42
    fi
    # GPU 1,2,3 的 used MiB（不含 GPU0，那是 LOB-Bench 的地盘）
    mapfile -t used < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
                        | awk -F', ' '$1>=1 {print $2}')
    ok=1
    for u in "${used[@]}"; do (( u < GATE_MB )) || ok=0; done
    if (( ok == 1 && ${#used[@]} == 3 )); then
        pass=$((pass+1))
        echo "[gate] pass ${pass}/2 (used: ${used[*]}) t+$((now-t0))s"
        if (( pass >= 2 )); then break; fi
    else
        pass=0
        echo "[gate] busy (used: ${used[*]}) t+$((now-t0))s"
    fi
    sleep "$POLL_S"
done
echo "[gate] OPEN — launching full 33-checkpoint valset eval"
exec "$VE_DIR/run_valset_ce.sh" "$OUT_DIR" "" "1,2,3" 3
