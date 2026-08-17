#!/bin/bash
# 一波并发 bench：把 N 个 (arm, step, seed) 分配到一个 allocation 的 N 个节点上。
#
# 为什么要这个包装：run_bench_attached.sh 最后是 exec srun，会阻塞；
# 四个 bench 需要四个独立进程。放在一个 tmux window 里 & 起，比开四个 window 好管。
#
# 用法：
#   ALLOC=6019793 NODES="nid010680 nid010681 nid010682 nid010683" \
#   JOBS="hybpm:4479:2027 hybpm:4479:2028 hyb:4555:2028 base:5062:2026" \
#   bash code/wave_launch.sh
set -uo pipefail

T=/lus/lfs1aip2/projects/public/u6gb/tasks/hybrid_system_20260811/05_hybrid_mamba3_lob
CK=/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/hybrid-mamba3-nemotron-20260811/checkpoints

: "${ALLOC:?需要 ALLOC}" "${NODES:?需要 NODES}" "${JOBS:?需要 JOBS}"

# arm -> (checkpoint 目录, architecture)
# 目录来自实测的 checkpoint 清单；**不要**凭目录名猜臂，臂身份以 metadata 为准，
# 这里只是选权重，bench 内部会从 metadata 读全部配置（inference.py:161）。
ckdir_of() {
    case "$1" in
        base)  echo "$CK/j6000409_je7cvor0_6000409" ;;   # 4462 4515 4789 5062 5330
        base2) echo "$CK/j6000409_gjnf0e03_6000409" ;;   # 5330 5408 5644 5844 6059 6265 6509
        hyb)   echo "$CK/j5998835_44awyydg_5998835" ;;   # 4555 ... 5103 5297 ... 6258 6466
        hybpm) echo "$CK/j6011444_dbjkbfh5_6011444" ;;   # 3849 4059 4269 4479
        hybpm2)echo "$CK/j6011842_2hedgxa5_6011842" ;;   # 4854 5063 5271
        hybpm3)echo "$CK/j6022465_wmarov18_6022465" ;;   # 5535 5673 5846 5994 —— 训练停在 5994
        *)     return 1 ;;
    esac
}
arch_of() {
    case "$1" in
        base|base2) echo "mamba3" ;;
        hyb|hybpm|hybpm2|hybpm3) echo "hybrid_mamba3" ;;
        *) return 1 ;;
    esac
}
armid_of() {   # 产物标签用的短名，保持与既有产物一致
    case "$1" in
        base|base2) echo "base" ;;
        hyb) echo "hyb" ;;
        hybpm|hybpm2|hybpm3) echo "hybpm" ;;
    esac
}

# JOBS 的元素格式：arm:step:seed[:world[:gpu_offset]]
#
# world 默认 4（整节点）。**为什么 world 可以不是 4**：唯一的硬约束是
# N_SEQUENCES % (world × per_gpu_batch) == 0，即 3136 % (world × 8) == 0。
# 3136 = 2⁶ × 7²  ⇒  world ∈ {1, 2, 4} 都整除，**world = 3 不行**（3136 无因子 3）。
# 所以一个只剩 3 张空卡的节点仍然能跑一个 world=2 的 bench，序列数、索引集、
# 口径完全不变，只是慢一倍。之前把「4 卡」当成硬条件，白白空转了 20 张卡。
set -- $NODES
i=0
for spec in $JOBS; do
    IFS=':' read -r arm step seed jworld joff <<< "$spec"
    jworld="${jworld:-4}"; joff="${joff:-0}"
    if [ $(( 3136 % (jworld * 8) )) -ne 0 ]; then
        echo "FATAL: world=$jworld 除不尽 3136（3136 % $((jworld*8)) != 0），换 1/2/4" >&2; exit 6
    fi
    i=$((i+1)); eval "node=\${$i:-}"
    if [ -z "$node" ]; then echo "FATAL: 第 $i 个任务没有节点可用（NODES 里只有 $# 个）" >&2; exit 2; fi

    ckdir=$(ckdir_of "$arm") || { echo "FATAL: 未知 arm=$arm" >&2; exit 3; }
    arch=$(arch_of "$arm")
    armid=$(armid_of "$arm")

    # 起飞前自证：这个 step 真的在这个目录里
    if [ ! -d "$ckdir/$step/state" ]; then
        echo "FATAL: $ckdir/$step/state 不存在（arm=$arm step=$step）" >&2; exit 4
    fi

    TS=$(date -u +%Y%m%dT%H%M%SZ)
    LOG="$T/logs/bench_${armid}${step}_s${seed}_${node}_w${jworld}g${joff}_${TS}.log"
    echo "[wave] $node GPU[$joff..$((joff+jworld-1))] <- $armid@$step s$seed (world=$jworld)   log=$LOG"
    ( cd "$T" && \
      ATTACH_JOBID="$ALLOC" NODE="$node" ARCHITECTURE="$arch" \
      ARM_ID="$armid" ARM_NAME="$armid" \
      CHECKPOINT_PATH="$ckdir" CHECKPOINT_STEP="$step" \
      GENERATION_SEED="$seed" BENCH_WORLD_SIZE="$jworld" BENCH_GPU_OFFSET="$joff" \
      BENCH_BATCH="$T/bench_scripts/bench_2k.batch" \
      bash code/run_bench_attached.sh > "$LOG" 2>&1 ) &
    sleep 8      # 错开启动，别让四个同时挂 squashfs 分片
done
# 这两行的措辞必须互不包含：2026-08-15 后台巡检 grep "全部结束" 时，
# 等待那句里也有这四个字，于是巡检在 3 秒后就报「完成」。
# 标准在成功和未成功时取值相同 = 零区分力。现在用 WAVE_RUNNING / WAVE_COMPLETE 两个互不包含的标记词。
echo "[wave] WAVE_RUNNING 已起 $i 个，等它们跑完…"
wait
echo "[wave] WAVE_COMPLETE $i 个全部退出 $(date -u +%H:%M:%SZ)"
