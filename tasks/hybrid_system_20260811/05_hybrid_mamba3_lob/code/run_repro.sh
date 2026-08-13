#!/usr/bin/env bash
# 在两个节点上跑 checkpoint 死锁复现器。
#
# 环境要与训练一致，否则复现不出来的话分不清是「修好了」还是「环境不同」：
#   同一个 conda（JAX 0.9.0.1）
#   同一份自编 NCCL 2.29.3（ARM CAS 修复），走 LD_PRELOAD
#   多节点同款 NCCL_BUFFSIZE=2MB（Slingshot 自适应路由的 latency jitter 补偿）
# 不设 FI_CXI_* 那一组：node_wrapper.sh 只在 >=8 节点时开，训练跑的是 4 节点，
# 保持一致。
set -uo pipefail
ALLOC=${ALLOC:?}
NODELIST=${NODELIST:?}
MODE=${MODE:-A}
ITERS=${ITERS:-200}
MB=${MB:-64}
TIMEOUT=${TIMEOUT:-120}
LEAVES=${LEAVES:-8}
CHURN=${CHURN:-0}
MEMFRAC=${MEMFRAC:-0.12}
PORT=${PORT:-29511}
T=/lus/lfs1aip2/projects/public/u6gb/tasks/hybrid_system_20260811/05_hybrid_mamba3_lob
HEAD=$(scontrol show hostnames "$NODELIST" | head -1)
NN=$(scontrol show hostnames "$NODELIST" | wc -l)

_CLEAN=$(env | grep -oE '^SLURM[A-Z_]*' | sed 's/^/-u /' | tr '\n' ' ')
echo "[repro] alloc=$ALLOC nodes=$NODELIST(${NN}N) mode=$MODE head=$HEAD:$PORT"

timeout 1800 env $_CLEAN srun --jobid="$ALLOC" --overlap \
  --nodes="$NN" --ntasks="$NN" --ntasks-per-node=1 -w "$NODELIST" --cpu-bind=none \
  bash -c '
    # 路径以训练日志里的为准（[Wrapper] Python: ... 那一行），不是 CLAUDE.md 里
    # 那条 /projects/s5e/quant/miniforge3 —— 后者对 s5e 账号成立，本会话是 u6gb，
    # 读它会 Permission denied 然后**静默退回系统 python 跑 CPU**，测出一个漂亮
    # 但毫无意义的「无挂死」。
    CONDA=/lus/lfs1aip2/projects/public/s5e/quant_team/quant/miniforge3
    export PATH=$CONDA/bin:$PATH
    NCCL_LIB=/lus/lfs1aip2/projects/public/s5e/quant_team/quant/nccl-2.29.3/lib
    export LD_PRELOAD=$NCCL_LIB/libnccl.so.2
    export LD_LIBRARY_PATH=$NCCL_LIB:/lus/lfs1aip2/projects/public/s5e/quant_team/quant/aws-ofi-nccl-1.18.0/lib:${LD_LIBRARY_PATH:-}
    export NCCL_BUFFSIZE=2097152
    # NCCL_DEBUG=INFO 会在每次 commInitRank 时打日志 —— 这是唯一能直接回答
    # 「broadcast_one_to_all 到底有没有新建 communicator」的证据。
    # 只在需要时打开：INFO 很吵，且会拖慢启动。
    if [ "'"${NCCLDBG:-0}"'" = "1" ]; then
      export NCCL_DEBUG=INFO
      export NCCL_DEBUG_SUBSYS=INIT,ENV
    fi
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    # 共租：只拿一小块显存，别影响邻居
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_MEM_FRACTION='"$MEMFRAC"'
    export JAX_COORDINATOR_ADDRESS='"$HEAD:$PORT"'
    export SLURM_PROCID=${SLURM_PROCID}
    export SLURM_NNODES='"$NN"'
    python -u '"$T"'/code/repro_ckpt_deadlock.py --mode '"$MODE"' --iters '"$ITERS"' --mb '"$MB"' --timeout '"$TIMEOUT"' --leaves '"$LEAVES"' --churn-mb '"$CHURN"'
  ' 2>&1
echo "[repro] srun 退出码 $?"
