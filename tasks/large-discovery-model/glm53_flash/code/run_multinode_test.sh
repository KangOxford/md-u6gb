#!/bin/bash
# 2 节点 8 卡 NCCL 实测(attach 到已有分配)
set -u
T=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/glm53_flash
JOB=${JOB:?}; NODES=${NODES:?}   # 逗号分隔两个节点
MASTER=$(echo "$NODES" | cut -d, -f1)
srun --overlap --jobid=$JOB -w "$NODES" --nodes=2 --ntasks=2 --ntasks-per-node=1 \
     --gres=gpu:4 --cpus-per-task=64 --job-name=mn-nccl --cpu-bind=none bash -c "
  # Slingshot-11:必须指名高速网卡 hsn*,否则 NCCL 会在选网卡时挂死
  # (裸跑第一次就是卡在 init_process_group,10 分钟无输出)
  # 关键:torch 轮子自带的 NCCL 没有 Slingshot/libfabric 传输层,多机 init 直接挂死。
  # 必须挂 AWS OFI NCCL 插件(裸跑两次都卡在 init_process_group,10 分钟无输出)。
  Q=/projects/public/s5e/quant_team/quant
  export LD_LIBRARY_PATH=$Q/nccl-2.29.3/lib:$Q/aws-ofi-nccl-1.18.0/lib:/opt/cray/libfabric/1.22.0/lib64:\${LD_LIBRARY_PATH:-}
  export LD_PRELOAD=$Q/nccl-2.29.3/lib/libnccl.so.2\${LD_PRELOAD:+:\$LD_PRELOAD}
  export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
  export NCCL_IB_DISABLE=1          # Slingshot 是以太网底座,不是 IB
  export NCCL_BUFFSIZE=2097152      # 2MB:补自适应路由的 latency jitter
  export NCCL_DEBUG=WARN
  export MASTER_ADDR=$MASTER MASTER_PORT=29577
  export NNODES=2 NODE_RANK=\$SLURM_NODEID NPROC_PER_NODE=4
  exec /home/u6gb/kangli.u6gb/envs/ldm-nanogpt/bin/torchrun \
     --nnodes=2 --nproc_per_node=4 --node_rank=\$SLURM_NODEID \
     --master_addr=$MASTER --master_port=29577 \
     $T/code/test_multinode.py
"
