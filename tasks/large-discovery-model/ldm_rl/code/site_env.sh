#!/bin/bash
# Isambard-AI 上的站点配置。跑 slime_launch 里任何脚本之前先 source 它。
#
#   source site_env.sh && bash $LAUNCH/run_train_real_9b.sh
#
# 这些变量在上游脚本里已改成 ${X:-作者原值} 形态,所以这里 export 之后才会生效;
# 作者机器上不 source 这个文件,行为与原来一模一样。
set -a

# ---- 代码 ----
export REPO_ROOT=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/LDM-rl
export SLIME_ROOT=$REPO_ROOT/rl/slime
export MEGATRON_ROOT=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/Megatron-LM
export LAUNCH=$REPO_ROOT/rl/slime_launch

# ---- 环境 ----
# 训练栈与评测栈是两个环境,这是 remote_env.py 写死的边界(它用 task_python 起子进程
# 跑评测,并显式剥掉 LD_LIBRARY_PATH/CUDA_HOME/CUDA_VISIBLE_DEVICES)。
export CONDA_PREFIX=/home/u6gb/kangli.u6gb/envs/ldm-rl-train    # 训练栈
export TASK_PYTHON=/home/u6gb/kangli.u6gb/envs/ldm-rl/bin/python # 评测栈(reward)

# ---- 模型与产物(放 Lustre:空间足;HF 仓库文件少,不吃 inode) ----
export HF_MODELS=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/models
export MODEL_HF=$HF_MODELS/Qwen3.5-9B
export WORKDIR=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/ldm_rl

# 站点版 config:只改了三条路径(gp_history_file / vina_bin / nn_model_path),
# 其余数值与上游 config_real.json 一致。上游那份不动,便于对账与回馈。
export CONFIG=$WORKDIR/code/config_real.isambard.json

# ---- cudart 13 拦截 ----
# 作者用 /root/cudart_block 放一个空的 libcudart.so.13 挡住系统里的 CUDA 13。
# 本机没有系统级 CUDA 13,而且 /root 不可写。指到可写位置,内容留空即可;
# 若 TE 的 cudnn-frontend 检查真的报 cu13 再往里放空文件。
export CUDART_BLOCK=${TMPDIR:-/tmp}/cudart_block_$USER
mkdir -p "$CUDART_BLOCK"

# ---- 评测栈资产 ----
export VINA_BIN=/home/u6gb/kangli.u6gb/envs/ldm-rl/bin/vina
export NN_MODEL=$REPO_ROOT/tasks/small_molecule/resources/models/best_g12d_model.joblib

# 本机走 direct:GH200 上 ray job submit 两次都以 504 结束(创建作业的 POST 在
# dashboard 的 JobHead 子进程模块那里超时)。单节点本地集群不需要 job server,
# 直接 ray.init 连已起的集群即可。上游默认仍是 jobsubmit,不受影响。
export SLIME_LAUNCH_MODE=direct

# flashinfer 在**运行时**用 ninja+nvcc JIT 编内核(RMSNorm 等)。那个子进程不继承
# 安装脚本里做过的两件事,于是:
#   1. 退回系统的 gcc 7(/usr/lib64/gcc/aarch64-suse-linux/7/.../ld)
#   2. 报 `ld: cannot find -lcudart`
# 第 2 条不是缺文件 —— $CUDA_HOME/lib 下裸名 libcudart.so 就在那里(实测)。
# 是链接器不知道去那里找:**LD_LIBRARY_PATH 管运行时加载,LIBRARY_PATH 才管链接**,
# 而启动脚本只设了前者。两个变量名只差三个字母,极易混。
export CUDA_HOME=${CUDA_HOME:-/home/u6gb/kangli.u6gb/envs/ldm-rl-train}
export LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/targets/sbsa-linux/lib:${LIBRARY_PATH:-}"
export CPATH="$CUDA_HOME/include:${CPATH:-}"
# 让 JIT 用 gcc 12:系统默认 7,而 torch 的头文件直接 #error 要求 >= 9
if [ -x /opt/cray/pe/gcc-native/12/bin/gcc ]; then
    export CC=${CC:-/opt/cray/pe/gcc-native/12/bin/gcc}
    export CXX=${CXX:-/opt/cray/pe/gcc-native/12/bin/g++}
    # 不要把 gcc 目录放到 PATH 首位:那会顺带改变同名命令的解析
    # (实测把 `python` 解析到了别的解释器)。CC/CXX 指对就够了,
    # 编译器是靠这两个变量被调用的,不是靠 PATH 顺序。
    export PATH="$PATH:/opt/cray/pe/gcc-native/12/bin"
fi

export PYTHONPATH=$MEGATRON_ROOT:$REPO_ROOT/rl:$REPO_ROOT:${PYTHONPATH:-}
set +a

echo "[site] REPO_ROOT=$REPO_ROOT"
echo "[site] 训练栈=$CONDA_PREFIX"
echo "[site] 评测栈=$TASK_PYTHON"
echo "[site] 模型=$HF_MODELS"
