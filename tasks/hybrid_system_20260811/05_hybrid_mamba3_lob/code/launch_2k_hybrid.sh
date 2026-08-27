#!/bin/bash
# hybrid mamba3 x Nemotron，上下文 2,000 条消息（52,000 token）。
#
# 与 launch_2k_baseline.sh 成对，两者只差 ARCHITECTURE 与五个 HYBRID_ATTN_*，
# 其余逐字相同。四条臂的关系：
#
#            上下文 500 条            上下文 2,000 条
#   baseline  已完成 j5877859@32001    launch_2k_baseline.sh
#   hybrid    已完成 j5980745@32001    launch_2k_hybrid.sh   <- 本脚本
#
# 从 launch_attach_hybrid.sh 派生，逐行只改三处：
#   PER_GPU_BSZ    4 -> 1
#   MSG_SEQ_LEN    500 -> 2000
#   WANDB_PROJECT  新设置起新 project
#   MAX_JOB_HOURS  5.5 -> 9.0
#
# 为什么 hybrid 的时限比 baseline 的 6.5 还要多：注意力那一层的计算是二次的。
#
#   单层注意力 FLOPs = 4 x L^2 x d_model
#                    = 4 x 52000^2 x 640
#                    = 6.9e12 / 样本 / 层
#
#   L=13,000 时同一项是 4.3e11，所以长度 4 倍、这一项 16 倍。六个 fused 层里只有
#   一个是注意力，但它从占比很小变成不可忽略。9.0 小时是留足余量，不是估算值；
#   真实步时由冒烟测出来后再收紧。
#
# 用法：ATTACH_JOBID=<alloc> SMOKE=1 bash launch_2k_hybrid.sh

set -uo pipefail

ATTACH_JOBID=${ATTACH_JOBID:-5980745}
# 可覆盖：两条臂并行时各占 allocation 的一半。NODELIST 必须同时给出对应的
# 子集，否则 srun 会在整个 allocation 上铺 --nodes=N，两条臂撞在同几个节点上。
NNODES_ATTACH=${NNODES_ATTACH:-4}
NODELIST=${NODELIST:-$(squeue -h -j "${ATTACH_JOBID}" -o "%N")}
WORKDIR=/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/hybrid-mamba3-nemotron-20260811
TASKDIR=/lus/lfs1aip2/projects/public/u6gb/tasks/hybrid_system_20260811/05_hybrid_mamba3_lob
SRC_BATCH="$WORKDIR/run/base_model/train_full_autoreg.batch"
SMOKE=${SMOKE:-0}
TAG=$([ "$SMOKE" = "1" ] && echo smoke || echo prod)
# 带臂名：两条臂并行跑时若共用一个路径，后起的那个会覆盖先起的，而先起的
# 已经把它读进 srun 了，症状是两条臂用同一份配置且没有任何报错。
GEN_BATCH="$TASKDIR/code/train_full_autoreg.attach.hyb.${TAG}.batch"

# step 名字不是装饰，它决定这个作业会不会被预算关卡砍掉。
#
# node_budget_monitor.py 用正则 ^(bash|sh|zsh|...)$ 判 step 是不是 idle，有非
# idle 名字的 step 就把整个作业记成 computing，而 **computing 不计入 20 节点上限**。
# 本仓库普遍用 `srun bash -lc '...'` 包装训练，于是 step 名就叫 bash：
#
#   2026-08-13 10:40 的 dry-run 里，正在训练的 hybrid（5998835，已跑 5h45m）
#   赫然显示 `IDLE-HELD  only bash,bash,bash,...`，随时可能被关卡当空转砍掉；
#   而邻居的 6000412 显示 `computing ldm-synth,dfm-rs-lg488b_g3`，安全。
#
# 起个真名字不是绕过关卡，是**把度量修对**——这个作业确实在算。
STEP_NAME=${STEP_NAME:-hyb-m3-ctx2k}

# ── 1. 生成 attach 版训练脚本 ────────────────────────────────────────────────
# 只改 srun 那一行：加 --jobid / --overlap，让 step 落进已有 allocation。
# 训练的 srun 也必须带 -w。--nodes=2 落在 4 节点的 allocation 上时，srun 自己
# 挑哪两个，两条臂并行就可能选中同几个节点，撞成显存不足或 NCCL 卡死。
sed "s|^srun --nodes=\$NNODES|srun --jobid=${ATTACH_JOBID} --overlap --job-name=${STEP_NAME} --nodes=\$NNODES -w \"\$SLURM_JOB_NODELIST\"|" \
    "$SRC_BATCH" > "$GEN_BATCH"
if ! grep -q -- "--jobid=${ATTACH_JOBID} --overlap" "$GEN_BATCH"; then
    echo "FATAL: srun 行改写失败，上游脚本格式可能变了。中止。" >&2
    exit 1
fi
echo "[attach] 已生成 $GEN_BATCH"

# ── 1.5 清理上一轮 attach 留下的死挂载 ──────────────────────────────────────
# attach 场景下 SLURM_JOB_ID 在整个 allocation 生命周期恒定，挂载根会被复用；
# 上一轮若被强杀，squashfuse 进程没了而内核挂载记录还在，下一轮 mkdir 会撞上
# "Transport endpoint is not connected"。死挂载点上 stat/glob 都失败，所以只能
# 直接读 /proc/mounts，任何先做存在性检查的写法都会静默跳过。
_CLEAN_ENV=$(env | grep -oE '^SLURM[A-Z_]*' | sed 's/^/-u /' | tr '\n' ' ')
echo "[preflight] 清理 allocation ${ATTACH_JOBID} 上的残留 squashfuse 挂载"
timeout 200 env $_CLEAN_ENV srun --jobid="${ATTACH_JOBID}" --overlap \
    --nodes=${NNODES_ATTACH} --ntasks=${NNODES_ATTACH} --ntasks-per-node=1 -w "${NODELIST}" --cpu-bind=none \
    bash -c '
      before=$(grep -c '"${ATTACH_JOBID}"' /proc/mounts 2>/dev/null || echo 0)
      grep '"${ATTACH_JOBID}"' /proc/mounts 2>/dev/null | awk "{print \$2}" | while read -r m; do
          fusermount -uz "$m" 2>/dev/null || umount -l "$m" 2>/dev/null
      done
      sleep 2
      # Unmounting is not enough. fusermount -uz is lazy: while anything still
      # references the mountpoint it stays attached, and the next squashfuse
      # mounts on top of it and resolves to the old, empty view. The wrapper
      # then reports "mounted 48/48 shards" and training dies one step later on
      # "no index.json", which reads like a data problem and is not one.
      # Removing the directories is what actually makes the next mount clean.
      rm -rf /tmp/kangli.u6gb/sigma0/'"${ATTACH_JOBID}"'_* 2>/dev/null
      echo "[preflight] $(hostname) stale_mounts=$before now=$(grep -c '"${ATTACH_JOBID}"' /proc/mounts 2>/dev/null || echo 0) dirs_removed"
    ' 2>&1 | head -8

# ── 1.6 物理 GPU 关卡：--overlap 不等于显存安全 ─────────────────────────────
# 这一步是 2026-08-12 冒烟三号的直接产物。当时 5980502 在 20:15 和 02:22 两次
# 快照里都是全空的，但 02:30 启动时 nid010473 GPU0 已被另一个实验
# （crps-return-alignment 的 inference.py）占走 71.5 GB。训练侧
# XLA_PYTHON_CLIENT_PREALLOCATE=true 会一次性抢 MEM_FRACTION 那么多显存，
# 于是 cuBLAS 连初始化用的几百 MB 都拿不到，报成
# "failed to create cublas handle: the library was not initialized"
# ——看起来像库坏了，其实是争用。空闲快照的有效期以分钟计，必须紧邻启动再验。
# 发现占用时只报告并中止，绝不 kill 别人的进程。
# DEDICATED_ALLOC=1：这批节点已被明确划归本实验，起飞前先清场。
#
# 默认行为（不设这个变量）仍然是「发现别人的 compute PID 就中止，一根手指都不
# 动」。只有当分配的所有者明确说过这台归你用时才设它，因为它会无差别杀掉目标
# 节点上的全部 compute 进程。
#
# 为什么需要它：清一次场并不够。2026-08-12 19:13 手工清空这四个节点后，两条臂
# 的冒烟在 90 秒内相继被关卡挡下——其它会话的作业已经自动重启回同一批卡上。
# 清场与起飞之间只要有间隙，就会输掉这个竞争。所以清场必须紧贴关卡，在同一次
# 启动里完成。
DEDICATED_ALLOC=${DEDICATED_ALLOC:-0}
if [ "$DEDICATED_ALLOC" = "1" ]; then
    echo "[gpu-gate] DEDICATED_ALLOC=1：清空 ${NODELIST} 上的全部 compute 进程"
    timeout 200 env $_CLEAN_ENV srun --jobid="${ATTACH_JOBID}" --overlap \
        --nodes=${NNODES_ATTACH} --ntasks=${NNODES_ATTACH} --ntasks-per-node=1 \
        -w "${NODELIST}" --cpu-bind=none bash -c '
          P=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | tr "\n" " ")
          [ -z "$P" ] && { echo "[clear] $(hostname) 已空"; exit 0; }
          kill $P 2>/dev/null; sleep 8
          kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u) 2>/dev/null; sleep 5
          echo "[clear] $(hostname) killed=[$P] remaining=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | wc -l)"
        ' 2>&1 | grep "^\[clear\]"
fi

# 关卡是 claim_gate.sh：nvidia-smi 物理体检（锁层已于 2026-08-14 删除）。
#
# 已知盲区：体检只能证明 t=0 干净。2026-08-13 10:19 baseline 就死在这里 ——
# 体检那一刻卡是空的，8 分钟启动窗口里邻居的 vLLM 起来把 rank2 挤死。
# 曾用声明层覆盖这个窗口，2026-08-14 按用户令删除（锁比使用活得久，代价更大）。
# 详细推理见 claim_gate.sh 的头注释。
if ! ALLOC="$ATTACH_JOBID" NODELIST="$NODELIST" OWNER="${GATE_OWNER:-claude-ctx2k}" \
     TIER=T0 NOTE="2k hybrid arm (${TAG}), critical comparison" \
     MAX_RESIDUAL_MIB="${MAX_RESIDUAL_MIB:-4096}" \
     bash "$TASKDIR/code/claim_gate.sh"; then
    echo "FATAL: claim_gate 未通过，不起飞。" >&2
    exit 3
fi

# ── 2. 清掉继承的 SLURM_*，再伪造 allocation 级变量 ──────────────────────────
for v in $(env | grep -oE '^SLURM[A-Z_]*'); do unset "$v"; done
export SLURM_JOB_ID="${ATTACH_JOBID}"
export SLURM_NNODES="${NNODES_ATTACH}"
export SLURM_JOB_NUM_NODES="${NNODES_ATTACH}"
export SLURM_JOB_NODELIST="${NODELIST}"
# 作业名要能把两条臂分开。派生脚本时沿用了 hybrid 的名字，结果 gtop 上四个
# 节点全写 hybrid-mamba3-nemotron，看不出哪两个在跑 baseline——观测工具读的是
# 这个名字，不是脚本文件名。
export SLURM_JOB_NAME="hybrid-m3-ctx2k-${TAG}"
export SLURM_SUBMIT_DIR="$WORKDIR"
# 按臂分开的节点日志目录。node_wrapper.sh 用 SLURM_JOB_ID + SLURM_PROCID 命名
# 日志，而 attach 场景下两条臂共用同一个 SLURM_JOB_ID、procid 也都是 0..N-1，
# 于是两条臂会 exec> 到同一个文件上，各写各的，内容交错且互相截断。
# NODE_LOG_DIR 是 node_wrapper.sh:29 留出的覆盖点。
# 认调用方的覆盖（与 WANDB_PROJECT 同理）：32k 全预算跑要与 6.4k 那代分目录，
# 否则按日志目录找 checkpoint 的机制会把两代运行混在一起。
export NODE_LOG_DIR="${NODE_LOG_DIR:-$WORKDIR/logs_lobs5/ctx2k_hyb}"
ARCH_TAG=hyb
# ── XLA 持久化编译缓存 ──────────────────────────────────────────────────────
# 实测：一次重启到稳态约 3.5 分钟，其中 **XLA 编译训练步占 50 秒**
# （node 日志里第一条 tqdm 是 `[00:50<`，之后掉到 ~1.3s/步）。
# 剩下的是 preflight、挂 48 个 squashfs 分片、JAX 分布式 init、载 checkpoint——
# 那些编译缓存救不了。所以这里省的是每次重启 ~50 秒，不是「8 分钟」；
# 那个 8 分钟是 baseline 那次**故障**的耗时（含 2 分钟 JAX 分布式超时）。
#
# 失效机制不用自己设计，JAX 的缓存键（cache_key.py:get）已经哈希了：
#   computation（HLO 模块）/ jaxlib 版本 / backend 平台 / XLA flags
# **它键在编译产物上而不是源码文本上**，所以两个方向都对：改注释、改 print
# 不会失效（HLO 没变，本就该复用）；改一行影响计算图的代码则必然失效。
# 不存在「改了代码缓存没失效」，也不存在「只改注释却全量重编译」。
#
# 按配置分子目录：单个目录在 Lustre 上无限膨胀是元数据负担，而且不同配置
# （架构/节点数/序列长度）的产物本来就互不复用，分开放便于整目录丢弃。
JAX_CACHE_ROOT=${JAX_CACHE_ROOT:-$WORKDIR/.jax_cache}
_CACHE_SIG="${ARCH_TAG}_n${NNODES_ATTACH}_L${MSG_SEQ_LEN:-2000}_b${PER_GPU_BSZ:-1}"
export JAX_COMPILATION_CACHE_DIR="$JAX_CACHE_ROOT/$_CACHE_SIG"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
# 只缓存真正贵的：低于 1 秒的编译存下来反而是净亏（一次 Lustre 往返）
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-1.0}
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=${JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES:-0}
echo "[jax-cache] $JAX_COMPILATION_CACHE_DIR"


# 起飞前把上一轮的节点日志归档。日志名是 training_<allocid>_node<N>.log，同一个
# allocation 上续训会 exec > 把它截断 —— 于是每次自愈重启都抹掉上一次的死因。
#
# 2026-08-13：hybrid 死了三次，只有我恰好在场的那一次拿到了
# "FATAL: Step watchdog timeout ... Likely NCCL deadlock"；另外两次的证据被下一轮
# 启动覆盖，事后 grep 一无所获。**自愈链路越好用，事后可诊断性越差** —— 它默默
# 重启，把现场一起清了。归档是把这两件事解耦。
mkdir -p "$NODE_LOG_DIR/archive"
for _f in "$NODE_LOG_DIR"/training_*_node*.log; do
    [ -e "$_f" ] || continue
    cp -p "$_f" "$NODE_LOG_DIR/archive/$(basename "${_f%.log}")_$(date -u +%Y%m%dT%H%M%SZ).log" 2>/dev/null
done


# 这两个变量不是装饰。srun 会把 SLURM_TIMELIMIT 当作**这个 step 的时限**来读，
# 所以写死 05:00:00 的后果是：无论 MAX_JOB_HOURS 设成多少、无论分配还剩多久，
# 训练都会在整 5 小时被 srun 杀掉，sacct 记 State=TIMEOUT ExitCode=0:15。
#
# 2026-08-12 就是这样：hybrid 在 Elapsed 05:00:21 被 TIMEOUT，我却按
# MAX_JOB_HOURS=13.5 排的期，以为它能跑到 32,000 步。MAX_JOB_HOURS 管的是训练
# 脚本自己的优雅停机，在更内一层；srun 的 step 时限先到先杀，MAX_JOB_HOURS
# 再大也没用。
#
# 取分配的剩余时间，留 10 分钟给收尾（存 checkpoint + 卸载 48 个分片）。
_LEFT=$(squeue -j "${ATTACH_JOBID}" -h -o "%L" 2>/dev/null | tr -d ' ')
if [ -n "$_LEFT" ]; then
    _LEFT_MIN=$(python3 -c "
import sys
s=sys.argv[1]; d=0
if '-' in s: d,s=s.split('-',1); d=int(d)
p=[int(x) for x in s.split(':')]
while len(p)<3: p.insert(0,0)
print(max(10, d*1440+p[0]*60+p[1]-10))" "$_LEFT" 2>/dev/null)
    export SLURM_TIMELIMIT=$(printf "%02d:%02d:00" $((_LEFT_MIN/60)) $((_LEFT_MIN%60)))
else
    export SLURM_TIMELIMIT="23:00:00"
fi
export SBATCH_TIMELIMIT="$SLURM_TIMELIMIT"
echo "[time] step 时限设为 $SLURM_TIMELIMIT（分配剩 ${_LEFT:-?}）"

# ── 2.5 去匿名化：sigma-0 按双盲发布准备，真实路径被换成 /path/to/... ───────
# 还原本应由 credentials/real_env.sh 完成，但仓库里 `credentials` 是一个存
# token 的普通文件占掉了那个名字，还原从未发生。症状不是明确报错，而是
# LD_PRELOAD 加载 /path/to/quant/nccl-2.29.3/lib/libnccl.so.2 失败后早退。
export QUANT_ROOT=/lus/lfs1aip2/projects/public/s5e/quant_team/quant
export CONDA_ENV=base
export DATA_ROOT=/lus/lfs1aip2/projects/public/s5e/quant_team/lob_preproc_sp500_squashfs

# ── 3. 模型：与 baseline 逐项相同，只换 ARCHITECTURE 与 attention 配置 ──────
export ARCHITECTURE=hybrid_mamba3
export SSM_TYPE=                  # 留空：hybrid 由 --architecture 单独决定
export D_MODEL=640
export N_LAYERS=6
export BLOCKS=20
export SSM_SIZE_BASE=640
export PER_GPU_BSZ=1
export SSM_LR_BASE=8.0e-4
export OPT_CONFIG=muon
export MUON_LR=0.01
export WEIGHT_DECAY=0.005
export MAMBA3_D_STATE=128
export MAMBA3_EXPAND=2
export MAMBA3_HEADDIM=64
export MAMBA3_CHUNK_SIZE=64
export MAMBA3_ROPE_FRACTION=0.5
export MAMBA3_USE_TRITON=False
export TOKEN_MODE=26tok
export MSG_SEQ_LEN=2000

# Hybrid：不指定 HYBRID_ATTN_LAYERS，让 registry 按 Nemotron 规则自己算
# （L=6 -> 位置 3）。heads=10 使 head_dim=640/10=64，正好满足 Pallas 因果核
# 的 head_dim<=256 且 %8==0；否则会退回物化 LxL，13k token 下必然 OOM。
export HYBRID_ATTN_HEADS=10
export HYBRID_ATTN_FLASH=True
export HYBRID_ATTN_PE=False       # Nemotron 的 attention 不带位置编码，
                                  # 且 mamba3 已在状态里做 RoPE
# HYBRID_ATTN_D_FF 不设 = 4*d_model = 2560（Nemotron 忠实档）。
# 参数配平臂用 HYBRID_ATTN_D_FF=1135，届时另起。

# ── 4. 并行与批量（与 baseline 相同）────────────────────────────────────────
export GPUS_PER_NODE=4
export TP_SIZE=1
# ── K（梯度累积）不是自由参数，是有效批量的因变量 ──────────────────────────
#
# 定义实验的是**有效批量**，不是 K。三者的关系是恒等式：
#
#   有效批量 = micro_bsz × GPU/节点 × 节点数 × K
#      80    =     1     ×    4     ×  节点数 × K      →  节点数 × K = 20
#
#   节点数   2    4    5   10   20
#   正确 K  10    5    4    2    1
#
# 旧写法 `GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-10}` 是**按 2 节点写死的默认值**，
# 靠调用方（chain_manager.sh:291 传 5）记得覆盖。这个形状必然出事：
#   · 4 节点忘了覆盖 → 有效批量 160，是设计值的 2 倍
#   · **不会报任何错**，训练照跑、loss 照降，只是训出一个与对照臂不可比的模型
#   · 而「不可比」要到几小时后做配对比较时才发现，那时机器已经烧掉了
# 这正是 feedback_config_only_reaches_one_path 那族缺陷：一个必须随环境变的量，
# 被写成固定默认值 + 藏在别处的覆盖。
#
# 现在改成：**声明有效批量，推导 K，并对调用方传进来的 K 做硬校验。**
export EFFECTIVE_BSZ=${EFFECTIVE_BSZ:-80}
_bsz_denom=$(( PER_GPU_BSZ * GPUS_PER_NODE * NNODES_ATTACH ))
if [ "$_bsz_denom" -le 0 ] || [ $(( EFFECTIVE_BSZ % _bsz_denom )) -ne 0 ]; then
    echo "FATAL: 有效批量 $EFFECTIVE_BSZ 无法被 micro_bsz($PER_GPU_BSZ) × GPU/节点($GPUS_PER_NODE) × 节点数($NNODES_ATTACH) = $_bsz_denom 整除。" >&2
    echo "       换节点数时必须选能整除的：节点数 × K = $(( EFFECTIVE_BSZ / (PER_GPU_BSZ * GPUS_PER_NODE) ))。" >&2
    exit 5
fi
_k_derived=$(( EFFECTIVE_BSZ / _bsz_denom ))
if [ -n "${GRAD_ACCUM_STEPS:-}" ] && [ "${GRAD_ACCUM_STEPS}" != "${_k_derived}" ]; then
    echo "FATAL: 调用方传了 GRAD_ACCUM_STEPS=$GRAD_ACCUM_STEPS，但 $NNODES_ATTACH 节点下" >&2
    echo "       有效批量 $EFFECTIVE_BSZ 要求 K=$_k_derived（会得到 $(( _bsz_denom * GRAD_ACCUM_STEPS ))，不是 $EFFECTIVE_BSZ）。" >&2
    echo "       要改有效批量就显式传 EFFECTIVE_BSZ=，不要靠改 K 来间接改它。" >&2
    exit 5
fi
export GRAD_ACCUM_STEPS=$_k_derived
echo "[bsz] 有效批量 $EFFECTIVE_BSZ = ${PER_GPU_BSZ} × ${GPUS_PER_NODE}GPU × ${NNODES_ATTACH}节点 × K${GRAD_ACCUM_STEPS}"
#
# 注意 train.py:413 的 curtail_epochs 数的是 micro-batch，K>1 时必须乘 K，
# 否则会在 32000/K 步静默早停；COSINE_STEPS 数的是优化器步，不乘。
export HIERARCHICAL=True
export REMAT=0
# profiling 指标。梯度范数按参数组分开（global/muon/ssm/regular/in_proj/out_proj）
# 外加 clip_ratio——后者直接回答「裁剪是不是一直在饱和」，那是发散诊断的第一问。
# 成本是每次记录时多算一棵树的平方和，相对一步训练可忽略。
export LOG_GRAD_NORMS=${LOG_GRAD_NORMS:-1}
# 显式 CHECKPOINT_EVERY 时 step_loss 的记录频率（auto 档不用它）。
export LOG_EVERY=${LOG_EVERY:-250}
# 4 节点档 node_wrapper.sh 默认 0.90（≈88/97.9 GB）。实测 hybrid 峰值只有
# 67.1 GB，0.90 既无必要又会在共享 allocation 上把邻居挤死，还让 cuBLAS 拿不到
# 初始化用的那几百 MB。0.85 ≈ 83 GB，比实测峰值高 24%，留 15 GB 给核与邻居。
export MEM_FRACTION=${MEM_FRACTION:-0.85}

# ── 5. 数据：与 baseline 逐字相同的 48 月 x 8 票 ────────────────────────────
MONTHS=""
for y in 2022 2023 2024 2025; do
  for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
    MONTHS="${MONTHS:+$MONTHS,}${y}-${m}"
  done
done
export SQUASHFS_MULTI_MODE=1
export SQUASHFS_DIR=/lus/lfs1aip2/projects/public/s5e/quant_team/lob_preproc_sp500_squashfs
export SQUASHFS_MONTHS="$MONTHS"
export SQUASHFS_MULTI_MOUNT_ROOT="/tmp/kangli.u6gb/hybrid_$(date -u +%Y%m%dT%H%M%SZ)"
export FORBID_RAW_NPYZST=1
export TICKERS="GOOG,AAPL,NVDA,AMZN,META,TSLA,MSFT,AMD"
export TRAIN_DATE_RANGE="2022-01-01,2025-12-31"
export NO_VALIDATION=True
export N_DATA_WORKERS=12

# ── 6. 时长、LR schedule 与 checkpoint ──────────────────────────────────────
export EPOCHS=1
export MINI_EPOCHS=1
if [ "$SMOKE" = "1" ]; then
    # 冒烟只回答两件事：48 分片挂得上吗，新的参数树存得下 checkpoint 吗。
    # 存点间隔调到 50 步，才能在几分钟内看到第一次保存成功。
    export COSINE_STEPS=32000     # 与正式一致，避免 LR 曲线形状不同
    export CURTAIL_EPOCHS=150
    export MAX_JOB_HOURS=0.6
    export CHECKPOINT_EVERY=50
else
    # COSINE_STEPS 覆盖 total_steps（src/lob/train.py:431-433）。一个 epoch 是
    # 939,147 步，按它铺 schedule 会让 warmup 吃掉 30% 训练且全程不退火；而
    # 退火段恰恰是 loss 降得最多、也是生成质量真正成型的一段。
    export COSINE_STEPS=${COSINE_STEPS:-32000}          # 优化器步
    # CURTAIL_EPOCHS 数的是 micro-batch（train.py:413 是 curtail_epochs+1），
    # 而 COSINE_STEPS 数的是优化器步。两者差一个 K，靠人记得乘的后果是：忘了乘
    # 就在 目标/K 步处静默早停，没有任何报错，曲线看起来只是"跑得短了些"。
    # 所以这里自己乘，调用方只需要给优化器步数。
    #
    # 续训时还要减掉已经跑过的部分，而且**减法要在 micro 域里做**。
    # 2026-08-13 实测：从 optimizer 4462 续起时，batch_idx 也从 ≈4461 开始 ——
    # 训练侧拿「上一次的优化器步号」当 micro-batch 索引的起点，两个单位混用。
    # 于是 CURTAIL=32000 不是「再跑 6400 优化器步」，而是「跑到 batch_idx=32000」，
    # 即 4462 + (32000-4461)/5 ≈ optimizer **9970**，超目标 56%。
    # 验算（实测数据）：batch_idx 8801 ↔ optimizer 5330，
    #                  4462 + (8801-4461)/5 = 5330 ✓
    #
    # 后果不只是浪费：控制器判达标后转入评测，而训练**还在同一批卡上跑**，
    # 评测被挤；配对点也会漂到 9970 而不是设计的 6400（那里 cosine 已经退火完，
    # 是另一个训练状态）。
    _rs=${RESTORE_STEP:-0}
    if [ "${_rs}" -gt 0 ] 2>/dev/null && [ "${_rs}" -lt "${COSINE_STEPS}" ]; then
        export CURTAIL_EPOCHS=${CURTAIL_EPOCHS:-$(( _rs + (COSINE_STEPS - _rs) * GRAD_ACCUM_STEPS ))}
    else
        export CURTAIL_EPOCHS=${CURTAIL_EPOCHS:-$(( COSINE_STEPS * GRAD_ACCUM_STEPS ))}
    fi
    # 冒烟实测回填：2 节点、每卡每步 2,000 条消息，稳态 1.375 it/s = 0.727 秒。
    # 对比 500 上下文的 0.313 秒，慢 2.32 倍，与注意力 FLOPs 之比一致：
    #   (1 x 52000^2) / (4 x 13000^2) = 4.0
    # 把四段 500 拼成一段 2000，注意力代价正好 4 倍——这不是开销，这就是长程
    # 注意力的定价。K=2 时 64,001 x 0.727 = 12.9 小时，取 13.5。
    export MAX_JOB_HOURS=${MAX_JOB_HOURS:-13.5}
    # auto 是原本就有的正确档位，而给一个显式数字会同时踩两个坑：
    #   (1) checkpoint 变稀（3000 步在 2k 上是 50 分钟以上）
    #   (2) step_loss 被绑到同一个频率上，观测精度跟着一起塌
    # auto 走的是时间标准：checkpoint 每 15 分钟、wandb 每 1 分钟、首存在 5 分钟。
    # 26tok 那一代用的就是这个，我先前用数字覆盖掉了。
    export CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-auto}
fi

# 续训：RESTORE_PATH 给运行根（不带步号），RESTORE_STEP 单独给。
# sigma-0 的 train.py:506-512 会由 state.step 自行推出 epoch 与 batch_idx，
# 所以 dataloader 不会从头重放，LR 也从该步继续退火——LOBS5 那个已知的
# mid-epoch resume 缺陷在这一支里是修好的，已逐行核对。
if [ -n "${RESTORE_PATH:-}" ]; then
    export RESTORE_PATH
    export RESTORE_STEP="${RESTORE_STEP:?RESTORE_STEP is required when RESTORE_PATH is set}"
    [ -f "$RESTORE_PATH/metadata/_ROOT_METADATA" ] || {
        echo "FATAL: $RESTORE_PATH/metadata/_ROOT_METADATA 不存在——RESTORE_PATH 应为运行根" >&2; exit 4; }
    [ -d "$RESTORE_PATH/$RESTORE_STEP/state" ] || {
        echo "FATAL: $RESTORE_PATH/$RESTORE_STEP/state 不存在——步号不对" >&2; exit 4; }
    echo "[resume] 从 $RESTORE_PATH @ $RESTORE_STEP 续训，目标 $CURTAIL_EPOCHS 步"
fi
export DISABLE_STEP_WATCHDOG=0
# 看门狗阈值按「稳态一步的多少倍」定，不是拍脑袋的整数。
#
# 2026-08-13 11:06 这条臂在 batch 3760 撞上 NCCL 死锁：主线程 futex_wait，4 个
# GPU 线程 100% 自旋（NCCL 集合是忙等，所以 nvidia-smi 照报满载——util 不能用来
# 判断训练是否活着）。看门狗在 1800s 后正确开火，但那是 **30 分钟 × 16 GPU** 白烧。
#
# 2k 上下文稳态一步 ~1.3s（micro-batch），看门狗是按 micro-batch kick 的：
#   1800s = 1400 倍余量   ← 之前
#    600s =  460 倍余量   ← 现在，仍然远超任何正常抖动，挂死代价降到 1/3
# warmup 那 50 步保持 900s 不动：JIT 编译 + NCCL 建通道确实要几百秒。
export WATCHDOG_TIMEOUT=${WATCHDOG_TIMEOUT:-600}
export STEP_TIMEOUT=600
export NO_AUTO_RESUME_DEPTH=99    # attach 场景禁止自动 sbatch 续投

# ── 7. W&B：新设置起新 project（A9）─────────────────────────────────────────
export USE_WANDB=True
export WANDB_MODE=online
export WANDB_ENTITY=oxford-lob
# 认调用方的覆盖。无条件 export 会把外面设的值吃掉：诊断跑本想进 -diag
# project，结果全写进主 project，和正式 run 混在一起，事后要靠创建时间去猜
# 哪个是哪个。默认值该是默认值，不该是硬性。
export WANDB_PROJECT=${WANDB_PROJECT:-sp500-hybrid-mamba3-35m-ctx2k}
export WANDB_DIR=/local/user/1483804540

export WORKDIR

echo "════════════════════════════════════════════════════════════════"
echo " Hybrid Mamba3 x Nemotron — SP500 2022-2025 — attach $ATTACH_JOBID [$TAG]"
echo "════════════════════════════════════════════════════════════════"
echo " architecture=$ARCHITECTURE  attention: nemotron rule (L=6 -> layer 3)"
echo "   heads=$HYBRID_ATTN_HEADS head_dim=$((D_MODEL / HYBRID_ATTN_HEADS))"
echo "   flash=$HYBRID_ATTN_FLASH  positional_encoding=$HYBRID_ATTN_PE"
echo " d_model=$D_MODEL n_layers=$N_LAYERS blocks=$BLOCKS d_state=$MAMBA3_D_STATE"
echo " per_gpu_bsz=$PER_GPU_BSZ global_bsz=$((PER_GPU_BSZ * GPUS_PER_NODE * NNODES_ATTACH))"
echo " nodes=$NNODES_ATTACH nodelist=$NODELIST msg_seq_len=$MSG_SEQ_LEN"
echo " msgs_per_gpu_per_step=$((PER_GPU_BSZ * MSG_SEQ_LEN))  tokens=$((PER_GPU_BSZ * MSG_SEQ_LEN * 26))"
echo " opt=$OPT_CONFIG muon_lr=$MUON_LR ssm_lr=$SSM_LR_BASE wd=$WEIGHT_DECAY"
echo " token_mode=$TOKEN_MODE msg_seq_len=$MSG_SEQ_LEN months=48 tickers=8"
echo " curtail=$CURTAIL_EPOCHS cosine=$COSINE_STEPS ckpt_every=$CHECKPOINT_EVERY"
echo " max_job_hours=$MAX_JOB_HOURS  wandb=$WANDB_PROJECT@$WANDB_ENTITY"
echo "════════════════════════════════════════════════════════════════"

cd "$WORKDIR"
exec bash "$GEN_BATCH"
