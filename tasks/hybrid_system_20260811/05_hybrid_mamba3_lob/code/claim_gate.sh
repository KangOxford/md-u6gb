#!/usr/bin/env bash
# claim_gate.sh —— 起训之前的「双查 + 声明」闸门（规则 R1/R2 的机械实现）
#
# 用法：
#   ALLOC=6000409 NODELIST=nid011212 OWNER=claude-ctx2k TIER=T0 \
#     NOTE="2k baseline arm" bash claim_gate.sh
#   退出 0 = 可以起飞且锁已上；非 0 = 别起，原因已打印。
#
# ── 为什么需要这个东西 ───────────────────────────────────────────────────────
#
# 2026-08-13 10:19，baseline 在 allocation 6000412 上被邻居的 vLLM（92 GB）挤死，
# rank2 的 CUDA context 建不起来，JAX 分布式 init 在 2 分钟超时后四个 rank 全灭。
#
# 事后查锁表，那四个节点 claude-dfm 占三个、crps-wm_ft 占一个，**都按规矩声明过**。
# 违规的是我这一侧：老闸门只查 nvidia-smi，不查锁表。
#
# 而且光把物理检查做得更细也救不了，这是原理问题：
#
#     10:11  物理闸门：0 个 compute PID  →  通过
#     10:11–10:19  启动窗口 8 分钟（挂 48 个 squashfs 分片 + JAX init）
#                  ↑ 邻居在这个窗口里起来
#     10:19  死
#
# **快照式检查保护不了一个比它长两个数量级的窗口。** 它只能证明 t=0 干净，
# 证明不了 [0, +8min] 干净。能覆盖窗口的只有声明：锁在整个窗口里都摆在表上，
# 邻居查自己的闸门时会看见。
#
# 所以本脚本的顺序是**先声明、后体检**，不是反过来：
#
#     ① 读锁表 —— 有别人的 live 锁就立刻退出，一根手指都不动
#     ② 写自己的锁 —— 竞态窗口从 8 分钟缩到 ①②之间的几秒
#     ③ 物理体检 —— 这一步要 ~30s，此时锁已经在保护我们了
#     ④ 体检不过就把 ② 的锁撤掉，绝不留下没有依据的声明
#
# ③ 放在 ② 后面是有代价的：体检不过时要回滚。宁可付这个代价，
# 因为反过来（体检→声明）会把整个 ~30s 的体检时间重新暴露成竞态窗口。
set -uo pipefail

ALLOC=${ALLOC:?需要 ALLOC=<jobid>}
NODELIST=${NODELIST:?需要 NODELIST=<slurm 节点表达式>}
OWNER=${OWNER:-claude-ctx2k}
TIER=${TIER:-T0}
NOTE=${NOTE:-"${TIER} exclusive"}
TTL=${TTL:-3600}
MAX_RESIDUAL_MIB=${MAX_RESIDUAL_MIB:-4096}
NODELOCK=${NODELOCK:-/lus/lfs1aip2/projects/public/u6gb/tasks/node_status/nodelock}

NODES=$(scontrol show hostnames "$NODELIST" 2>/dev/null | tr '\n' ' ')
[ -z "$NODES" ] && { echo "FATAL: 无法展开 NODELIST=$NODELIST" >&2; exit 2; }
NNODES=$(echo "$NODES" | wc -w)
echo "[claim] 目标 ${NNODES} 个节点：${NODES}"
echo "[claim] 身份 who=${OWNER} tier=${TIER} alloc=${ALLOC}"

# ── ① 读锁表：别人的 live 锁一票否决 ────────────────────────────────────────
# --no-probe 省掉那次 ~8s 的 srun 探测。这里只需要「谁声明了」，不需要活动状态；
# 而且探测会推进 last_probe，可能把别人的锁推过期——查一下就让别人的锁老化，
# 是把只读操作做成了副作用。
if [ -x "$NODELOCK" ]; then
    FOREIGN=$(timeout 120 "$NODELOCK" ls --json --no-probe 2>/dev/null | \
      NODES="$NODES" OWNER="$OWNER" ALLOW_SHARED="${GATE_ALLOW_SHARED:-0}" python3 -c '
import json, os, sys
try:
    reg = json.load(sys.stdin)
except Exception:
    sys.exit(0)                       # 读不到注册表 → 不阻塞，交给物理闸门兜底
me = os.environ["OWNER"]
for n in os.environ["NODES"].split():
    e = reg.get(n)
    if not e or not e.get("who") or e["who"] == me:
        continue
    # 认注册表自己的过期规则，别把死锁当活锁。
    # 判据是 last_probe - last_activity > ttl，**不是 now - last_activity**：
    # 「没观测到活动」不等于「观测到空闲」，用 now 会在没人跑 refresh 的时候
    # 把所有锁一起放出去 —— 恰好在最需要锁的时刻（谁都没盯着）。
    la  = e.get("last_activity") or 0
    lp  = e.get("last_probe") or 0
    ttl = e.get("ttl") or 600
    if lp and la and (lp - la) > ttl:
        sys.stderr.write("  [claim] %s 的 %s 锁已过期（闲置 %ds > ttl %ds），按无锁处理\n"
                         % (n, e["who"], lp - la, ttl))
        continue
    # 对方自己声明 shared 时允许共租（需 GATE_ALLOW_SHARED=1 显式打开）。
    # 这不是绕过 R4：R4 禁止的是**抢占**，而 shared 是占用者主动让出的邀请。
    # 判据必须是**对方写的**，不是我自己认为「应该能挤一挤」——
    # 后者就是 R4 要防的那种单方面判断。共租仍然要过显存闸门。
    note = (e.get("note") or "")
    if os.environ.get("ALLOW_SHARED") == "1" and "shared" in note.lower():
        sys.stderr.write("  [claim] %s 的 %s 声明为 shared，允许共租：%s\n"
                         % (n, e["who"], note[:60]))
        continue
    print("%s\t%s\t%s" % (n, e["who"], note[:70]))
')
    if [ -n "$FOREIGN" ]; then
        echo "FATAL: 这些节点已被别人声明，不碰（规则 R4：抢占只向下且永不由 agent 执行）" >&2
        printf '%s\n' "$FOREIGN" | while IFS=$'\t' read -r n w note; do
            printf '  %-10s ← %-14s %s\n' "$n" "$w" "$note" >&2
        done
        exit 4
    fi
    echo "[claim] ① 锁表干净，无他人声明"
else
    echo "[claim] ① 警告：找不到 $NODELOCK，跳过声明层检查（只剩物理层）" >&2
fi

# ── ② 先把锁写上，再去体检 ──────────────────────────────────────────────────
#
# PREEXISTING 这个判断不能省。回滚只该撤掉**本次新建**的锁，绝不能撤掉本来就在
# 那儿的。2026-08-13 首版没有它，第一次测试就把正在跑的 hybrid（nid010230）
# 自己的 T0 锁给 unlock 了：体检发现「有 compute PID」→ 回滚 → 保护没了。
# 而那个 compute PID 正是被保护的训练本身。**同一条路径既是保护者又是回滚者时，
# 必须记住进来之前的状态。**
PREEXISTING=0
if [ -x "$NODELOCK" ]; then
    if timeout 120 "$NODELOCK" ls --json --no-probe 2>/dev/null | \
       NODES="$NODES" OWNER="$OWNER" python3 -c '
import json, os, sys
try: reg = json.load(sys.stdin)
except Exception: sys.exit(1)
ns = os.environ["NODES"].split()
me = os.environ["OWNER"]
sys.exit(0 if all((reg.get(n) or {}).get("who") == me for n in ns) else 1)
'; then
        PREEXISTING=1
        echo "[claim] ② 这些节点本来就是我的声明，保持不动"
    fi
fi

# 共租模式下不写锁：注册表是每节点单主的，写进去等于抹掉对方的声明 ——
# 那正是 R4 禁止的单方面动作，只不过换成了在注册表里做而不是 kill 进程。
LOCKED=0
if [ "${GATE_ALLOW_SHARED:-0}" = "1" ]; then
    echo "[claim] ② 共租模式：不写锁，不覆盖对方的声明"
elif [ -x "$NODELOCK" ]; then
    if timeout 120 "$NODELOCK" lock "$NODELIST" --who "$OWNER" -j "$ALLOC" \
         --ttl "$TTL" --note "${TIER}: ${NOTE}" >/dev/null 2>&1; then
        LOCKED=1
        [ "$PREEXISTING" = "1" ] && echo "[claim] ② 声明已续期（ttl=${TTL}s）" \
                                 || echo "[claim] ② 已声明（ttl=${TTL}s）"
    else
        echo "[claim] ② 警告：上锁失败，继续但窗口无保护" >&2
    fi
fi
unlock_and_die() {
    if [ "$LOCKED" = "1" ] && [ "$PREEXISTING" = "0" ]; then
        timeout 120 "$NODELOCK" unlock "$NODELIST" >/dev/null 2>&1 \
          && echo "[claim] ④ 体检未过，已撤回 ② 的声明"
    elif [ "$PREEXISTING" = "1" ]; then
        echo "[claim] ④ 体检未过，但声明是进来之前就有的，保留不撤"
    fi
    exit "$1"
}

# ── ③ 物理体检 ──────────────────────────────────────────────────────────────
# 锁表是意图，nvidia-smi 是事实。两者都会单独骗人：
#   只信锁表 → 没声明就闯进来的邻居看不见（vLLM 那次）
#   只信 nvidia-smi → 声明了但进程还没起来的邻居看不见（本次）
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
    unlock_and_die 3
fi
if [ "$GATE_REQUIRE_ZERO_PIDS" = "1" ] && echo "$GATE_OUT" | grep -q "pids=\[[0-9]"; then
    echo "FATAL: [${TIER}] 有节点存在 compute PID —— 持有 context 的进程随时会长大。中止，不动它。" >&2
    unlock_and_die 3
fi
MINFREE=$(echo "$GATE_OUT" | sed -E 's/.*min_free_mib=([0-9]+).*/\1/' | sort -n | head -1)
if [ -n "$MINFREE" ] && [ "$MINFREE" -lt "$GATE_NEED_MIB" ]; then
    echo "FATAL: 最紧的一张卡只剩 ${MINFREE} MiB < 需求 ${GATE_NEED_MIB} MiB。中止。" >&2
    unlock_and_die 3
fi
WORST=$(echo "$GATE_OUT" | sed -E 's/.*worst_used_mib=([0-9]+).*/\1/' | sort -n | tail -1)

echo "[claim] ③ 物理体检通过（tier=${TIER} 零PID要求=${GATE_REQUIRE_ZERO_PIDS}，最坏残留 ${WORST} MiB，最小可用 ${MINFREE} MiB ≥ ${GATE_NEED_MIB}）"
echo "[claim] ✅ 可以起飞，锁已持有"
exit 0
