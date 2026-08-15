#!/bin/bash
# 数据量阶梯快照器。
#
# 为什么可以这样做，而不必等三条独立的臂各自跑完。`dfm_train_worker.py:421-426`
# 的 `_sched` 只有 warmup + `constant_schedule` —— **没有 cosine，也不依赖总步数**。
# 于是同 seed、同数据序、同 LR 的三个「数据量档」是**同一条轨迹的三个停点**，
# 而不是三条独立轨迹。实测证据：`dv5_lr1e4_s0u` 与 `dv25_lr1e4_s0u` 在 step 2550
# 的 loss 完全相同（3.7606，四位小数）。
#
# 而 checkpoint 每 100 步覆写同一个文件（`ckpt_every=100`），所以只要在正确的
# 时刻复制，就能从**一条**臂上取到整条阶梯 —— 而且它会在 step 3500 经过当前
# 基线 `lg488b_g2` 的训练长度，给出一个同 seed 的对照点，比跨臂比较干净。
#
# 阶梯（相对预训练 138.8M 条消息的数据量比）：
#     3500  -> 1:79   当前基线 lg488b_g2 的长度
#     6940  -> 1:20
#    13880  -> 1:10
#    27760  -> 1:5
#    55520  -> 1:2.5
set -u
cd "$(dirname "$0")/.."
ARM=${ARM:-dv25_lr1e4_s0u}
SRC=artifacts_dv/${ARM}_state.msgpack
DST=artifacts_dv/ladder
mkdir -p "$DST"
# ckpt_every=2000，所以可取的步数是 2000 的倍数；目标向上取到最近的可取点
TARGETS="4000 8000 14000 28000 56000"

step_of() { python3 -c "
import json,sys
try: print(json.load(open('$1'))['step'])
except Exception: print(-1)
"; }

echo "=== 阶梯快照 $ARM  目标 $TARGETS  $(date -u +%H:%M:%SZ) ==="
declare -A DONE
NDONE=0
last=x
stall=0
while :; do
    s=$(step_of "${SRC}.meta")
    for t in $TARGETS; do
        [ -n "${DONE[$t]:-}" ] && continue
        if [ "$s" -ge "$t" ] 2>/dev/null; then
            out="$DST/${ARM}_s${t}_state.msgpack"
            cp "$SRC" "$out" && cp "${SRC}.meta" "${out}.meta"
            got=$(step_of "${out}.meta")
            # 复制到的实际步数可能略超目标（每 100 步存一次），记下真值
            echo "  [snap] 目标 $t -> 实际 $got  $out  $(date -u +%H:%M:%SZ)"
            DONE[$t]=$got
            NDONE=$((NDONE+1))
        fi
    done
    [ "$NDONE" -ge 5 ] && { echo "=== 五个阶梯全部取到 ==="; break; }
    # 臂死了才停。阈值必须大于 checkpoint 间隔：ckpt_every=2000 步 x ~0.6 s
    # = 20 分钟才写一次，10 分钟的阈值会把「正常等待」误判成「臂死了」——
    # 第一次就是这么停的。80 x 30 s = 40 分钟。
    if [ "${s:-0}" = "$last" ]; then
        stall=$((stall+1))
    else
        stall=0
    fi
    last=$s
    [ "$stall" -ge 80 ] && { echo "=== step 停在 $s 超过 40 分钟，停止等待 ==="; break; }
    sleep 30
done
echo "取到的阶梯:"
for t in $TARGETS; do echo "  $t -> ${DONE[$t]:-未取到}"; done
