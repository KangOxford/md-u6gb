#!/bin/bash
# 把 bench 脚本挂到已有 allocation 的一个空闲节点上跑。
# 默认 bench_hybrid.batch（500 条上下文，cond250/gen250）；
# BENCH_BATCH 可换成 bench_2k.batch（2,000 条，cond1000/gen1000）。
#
# 用途是「用掉本来空转的卡」，不是省排队：评测约 1-3 小时，仍属可 attach 的
# 量级，但必须 setsid 起，否则会随会话一起死（2026-08-12 丢过 12,735 步）。
#
# 必需 env: ATTACH_JOBID NODE ARCHITECTURE ARM_ID ARM_NAME CHECKPOINT_PATH CHECKPOINT_STEP
set -uo pipefail
: "${ATTACH_JOBID:?}" "${NODE:?}" "${ARCHITECTURE:?}" "${CHECKPOINT_PATH:?}" "${CHECKPOINT_STEP:?}"
TASKDIR=/lus/lfs1aip2/projects/public/u6gb/tasks/hybrid_system_20260811/05_hybrid_mamba3_lob

# 时间关卡：attach 的 step 是宿主 allocation 的孩子，**宿主到点，step 一起死**。
#
# 2026-08-12 12:59:45：三个 bench 同时被 CANCELLED，同一秒。查下来是宿主
# 5980745 跑满 24h 自然 COMPLETED（Elapsed 23:58:14），而我在它只剩约 18 分钟
# 时挂了三个各需约 40 分钟的任务。三份生成全部作废。
# 物理关卡只回答「卡是不是空的」，回答不了「这个分配还能活多久」。
MIN_WALLTIME_MIN=${MIN_WALLTIME_MIN:-60}
LEFT=$(squeue -j "$ATTACH_JOBID" -h -o "%L" 2>/dev/null | tr -d ' ')
if [ -n "$LEFT" ]; then
  # %L 形如 D-HH:MM:SS / HH:MM:SS / MM:SS
  LEFT_MIN=$(python3 -c "
import sys
s=sys.argv[1]; d=0
if '-' in s: d,s=s.split('-',1); d=int(d)
p=[int(x) for x in s.split(':')]
while len(p)<3: p.insert(0,0)
print(d*1440 + p[0]*60 + p[1] + (1 if p[2] else 0))" "$LEFT" 2>/dev/null)
  echo "[gate] 宿主 $ATTACH_JOBID 剩余 walltime: $LEFT (${LEFT_MIN:-?} min)"
  if [ -n "$LEFT_MIN" ] && [ "$LEFT_MIN" -lt "$MIN_WALLTIME_MIN" ]; then
    echo "FATAL 宿主只剩 ${LEFT_MIN} min，低于所需 ${MIN_WALLTIME_MIN} min。换一个分配，别挂在这。"
    exit 2
  fi
else
  echo "[gate] 警告：读不到 $ATTACH_JOBID 的剩余时间，跳过时间关卡"
fi

# 物理关卡：**只看我要用的那几张卡**，不看整个节点。
#
# 节点级判定在共享 allocation 上过严。这里的邻居跑的是 --world_size=1 的单卡
# 推理：它在 GPU0 预分配 86 GB，在 GPU1-3 只留 context（各 0.6 GB）。节点级
# 判定看到 PID 就整机拒绝，于是 12 张真正空闲的卡被白白锁住。按卡判定既不碰
# 别人的进程，又能把那部分算力用起来。
# 标准：目标卡上显存 < 阈值即视为可用（context 约 0.6 GB，阈值取 4 GB）。
BENCH_WORLD_SIZE=${BENCH_WORLD_SIZE:-4}
BENCH_GPU_OFFSET=${BENCH_GPU_OFFSET:-0}
MAX_RESIDUAL_MIB=${MAX_RESIDUAL_MIB:-4096}
TARGET_GPUS=$(seq $BENCH_GPU_OFFSET $((BENCH_GPU_OFFSET + BENCH_WORLD_SIZE - 1)) | tr '\n' ',' | sed 's/,$//')
echo "[gate] $NODE 目标卡 [$TARGET_GPUS]，阈值 ${MAX_RESIDUAL_MIB} MiB"
GATE=$(timeout 120 srun --jobid="$ATTACH_JOBID" --overlap --nodes=1 --ntasks=1 \
  -w "$NODE" --cpu-bind=none bash -c '
    used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
    worst=0
    for g in $(echo "'"$TARGET_GPUS"'" | tr "," " "); do
      m=$(echo "$used" | awk -F", *" -v g="$g" "\$1==g{print \$2}")
      [ -n "$m" ] && [ "$m" -gt "$worst" ] && worst=$m
    done
    echo "GATE $(hostname) target_worst_mib=$worst"' 2>&1 | grep "^GATE")
echo "$GATE"
WORST=$(echo "$GATE" | sed -E 's/.*target_worst_mib=([0-9]+).*/\1/')
if [ -z "$WORST" ] || [ "$WORST" -gt "$MAX_RESIDUAL_MIB" ]; then
    echo "FATAL: $NODE 目标卡 [$TARGET_GPUS] 上残留 ${WORST:-?} MiB > ${MAX_RESIDUAL_MIB}，中止，不动别人的进程。" >&2
    exit 3
fi

for v in $(env | grep -oE '^SLURM[A-Z_]*'); do unset "$v"; done
export ATTACH_JOBID NODE ARCHITECTURE CHECKPOINT_PATH CHECKPOINT_STEP
export BENCH_WORLD_SIZE BENCH_GPU_OFFSET
export ARM_ID="${ARM_ID:-arm}" ARM_NAME="${ARM_NAME:-arm}"
export SLURM_JOB_ID="$ATTACH_JOBID"

# --job-name 必须给真名字。node_budget_monitor.py 按正则 ^(bash|sh|zsh|...)$ 判 step 是否
# idle，没名字的 step 会让整个作业被记成 IDLE-HELD 并随时可能被预算关卡 scancel。
# 2026-08-15 12:53 有 9 个 step 在两秒内 CANCELLED(0:9)，主因是父进程随会话死（已用 setsid 修），
# 但这些 step 当时全叫 `bash` —— 那是另一条独立的、还敞开着的失败通道。起名字不是绕过关卡，
# 是把度量修对。出处 [[reference_u6gb_node_budget_guardrail]]。
STEP_JOB_NAME="${STEP_JOB_NAME:-bench-${ARM_ID:-arm}${CHECKPOINT_STEP}-s${GENERATION_SEED:-2026}}"
echo "[bench] srun step name = $STEP_JOB_NAME"
exec srun --jobid="$ATTACH_JOBID" --overlap --exact --nodes=1 --ntasks=1 \
     --job-name="$STEP_JOB_NAME" \
     -w "$NODE" --cpus-per-task=72 --cpu-bind=none \
     bash "${BENCH_BATCH:-$TASKDIR/bench_scripts/bench_hybrid.batch}"
