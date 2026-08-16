#!/usr/bin/env bash
# claim_gate.sh —— 起训之前的物理闸门：这些卡此刻是不是真的空的
#
# 用法：
#   ALLOC=6000409 NODELIST=nid011212 OWNER=claude-ctx2k TIER=T0 \
#     NOTE="2k baseline arm" bash claim_gate.sh
#   退出 0 = 可以起飞；非 0 = 别起，原因已打印。
#
# ── 这里曾经有一个锁层，2026-08-14 用户令删除 ───────────────────────────────
#
# 用户原话：「不许不许 lock 人家 GPU，以后不许带这种功能」。
#
# 原设计是「先声明、后体检」四步：①读锁表 ②写自己的锁 ③物理体检 ④不过则回滚。
# 论证是：快照式检查只能证明 t=0 干净，证明不了整个 8 分钟启动窗口干净
# （2026-08-13 10:19 baseline 被邻居 vLLM 挤死就发生在窗口里），而锁在整个
# 窗口里都摆在表上。
#
# 这个论证成立，但它没算另一侧的代价：**锁会比使用活得久。**
# 锁是一次性写下的意图，使用是会中途死掉的事实——attach 的 step 一死，锁还挂着。
# 2026-08-14 实测：nid010499 挂 claude-h3 的锁、nid010549 挂 claude-dyna2 的锁，
# 两节点 8 张卡全程 0% / 0.0 GB，锁的唯一效果是让别人不敢用。
#
# 保护启动窗口省下的是「偶尔被挤死一次」，锁失效造成的是「持续占着不放」。
# 后者更贵，而且它不报错，没人会发现。
#
# 于是判据回到唯一不会比事实活得久的那个：**nvidia-smi 此刻显示它是空的**
# （外加用户明示）。进程死了，显存立刻就还回来。
set -uo pipefail

ALLOC=${ALLOC:?需要 ALLOC=<jobid>}
NODELIST=${NODELIST:?需要 NODELIST=<slurm 节点表达式>}
OWNER=${OWNER:-claude-ctx2k}
TIER=${TIER:-T0}
NOTE=${NOTE:-"${TIER} exclusive"}
TTL=${TTL:-3600}
MAX_RESIDUAL_MIB=${MAX_RESIDUAL_MIB:-4096}

NODES=$(scontrol show hostnames "$NODELIST" 2>/dev/null | tr '\n' ' ')
[ -z "$NODES" ] && { echo "FATAL: 无法展开 NODELIST=$NODELIST" >&2; exit 2; }
NNODES=$(echo "$NODES" | wc -w)
echo "[claim] 目标 ${NNODES} 个节点：${NODES}"
echo "[claim] 身份 who=${OWNER} tier=${TIER} alloc=${ALLOC}"

gate_failed() { exit "$1"; }

# ── 物理体检 ────────────────────────────────────────────────────────────────
# nvidia-smi 是唯一的判据。它有一个已知的盲区：它只能证明 t=0 干净，证明不了
# 整个启动窗口干净（邻居可能在窗口里长大）。这个盲区被接受，因为唯一能覆盖它
# 的办法是声明，而声明的代价更大——见文件头。
#
# **严格程度随判错代价缩放，而这恰好就是 tier 的定义：**
#
#   T0  启动窗口 8 分钟，邻居可能在窗口里长大（实测 nid010556 从 658 MiB 涨到
#       10,108 MiB）。判错 = 整条臂死掉 + 两条臂步数岔开。→ 判据「零 compute PID」，
#       因为一个持有 CUDA context 却只占 658 MiB 的进程，是「即将分配」不是「空闲」。
#
#   T1/T2 判错只是重跑几分钟。→ 判据「剩余显存 ≥ 需求」，进程数只记录不否决。
#       用零 PID 卡 T1 会为了邻居的 0.7 GB 白等几小时（另一个会话 2026-08-13
#       正是这样把 OOM 归因成「有别人的进程」，造出过严判据）。
#
# 一个判据套所有场景，必然在一头过严、另一头过松。
GATE_NEED_MIB=${GATE_NEED_MIB:-85000}      # 训练需要的空闲显存（MEM_FRACTION 0.85 × 95.6G）
case "$TIER" in
    T0) GATE_REQUIRE_ZERO_PIDS=${GATE_REQUIRE_ZERO_PIDS:-1} ;;
    *)  GATE_REQUIRE_ZERO_PIDS=${GATE_REQUIRE_ZERO_PIDS:-0} ;;
esac

_CLEAN_ENV=$(env | grep -oE '^SLURM[A-Z_]*' | sed 's/^/-u /' | tr '\n' ' ')
GATE_OUT=$(timeout 200 env $_CLEAN_ENV srun --jobid="$ALLOC" --overlap \
    --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 -w "$NODELIST" --cpu-bind=none \
    bash -c '
      pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr "\n" " ")
      worst=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
      # 最小可用 = 该节点所有卡里剩得最少的那张
      minfree=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits \
                | awk -F", *" "{f=\$1-\$2; if(m==\"\"||f<m) m=f} END{print m+0}")
      echo "GATE $(hostname) worst_used_mib=${worst:-0} min_free_mib=${minfree:-0} pids=[${pids}]"
    ' 2>&1 | grep "^GATE")
echo "$GATE_OUT"

GOT=$(echo "$GATE_OUT" | grep -c "^GATE")
if [ "$GOT" -ne "$NNODES" ]; then
    echo "FATAL: 只收到 ${GOT}/${NNODES} 个节点的体检回报，视同不通过。" >&2
    gate_failed 3
fi
if [ "$GATE_REQUIRE_ZERO_PIDS" = "1" ] && echo "$GATE_OUT" | grep -q "pids=\[[0-9]"; then
    echo "FATAL: [${TIER}] 有节点存在 compute PID —— 持有 context 的进程随时会长大。中止，不动它。" >&2
    gate_failed 3
fi
MINFREE=$(echo "$GATE_OUT" | sed -E 's/.*min_free_mib=([0-9]+).*/\1/' | sort -n | head -1)
if [ -n "$MINFREE" ] && [ "$MINFREE" -lt "$GATE_NEED_MIB" ]; then
    echo "FATAL: 最紧的一张卡只剩 ${MINFREE} MiB < 需求 ${GATE_NEED_MIB} MiB。中止。" >&2
    gate_failed 3
fi
WORST=$(echo "$GATE_OUT" | sed -E 's/.*worst_used_mib=([0-9]+).*/\1/' | sort -n | tail -1)

echo "[claim] 物理体检通过（tier=${TIER} 零PID要求=${GATE_REQUIRE_ZERO_PIDS}，最坏残留 ${WORST} MiB，最小可用 ${MINFREE} MiB ≥ ${GATE_NEED_MIB}）"
echo "[claim] ✅ 可以起飞（无锁：这是 t=0 的快照，启动窗口内不受保护）"
exit 0
