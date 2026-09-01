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

export PYTHONPATH=$MEGATRON_ROOT:$REPO_ROOT/rl:$REPO_ROOT:${PYTHONPATH:-}
set +a

echo "[site] REPO_ROOT=$REPO_ROOT"
echo "[site] 训练栈=$CONDA_PREFIX"
echo "[site] 评测栈=$TASK_PYTHON"
echo "[site] 模型=$HF_MODELS"
