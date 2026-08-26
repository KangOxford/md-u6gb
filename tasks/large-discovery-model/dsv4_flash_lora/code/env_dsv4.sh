#!/bin/bash
# DeepSeek-V4-Flash LoRA 公共环境 —— 被所有训练/测试脚本 source
export DSV4_TASK=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/dsv4_flash_lora
export DSV4_VENV=/home/u6gb/kangli.u6gb/envs/dsv4-venv
export MODEL_DIR=/lus/lfs1aip2/projects/public/u6gb/models/DeepSeek-V4-Flash

source $DSV4_VENV/bin/activate

# venv 之下的 slime-train 才有 nvidia pip 库(cudnn/cublas/nccl...);TE 按 sysconfig
# 路径找 cudnn 会看到 venv 的空 site-packages,必须显式把底层库目录挂上
_BASE_SP=/lus/lfs1aip2/projects/public/u6gb/tasks/openmle_rsi_20260812/envs/slime-train/lib/python3.12/site-packages
_VENV_SP=$DSV4_VENV/lib/python3.12/site-packages
_NVLIBS=""
for _d in $_VENV_SP/nvidia/*/lib $_BASE_SP/nvidia/*/lib; do   # venv 优先(装了新 cudnn 时压过底层旧版)
    [ -d "$_d" ] && _NVLIBS="${_NVLIBS:+$_NVLIBS:}$_d"
done
export LD_LIBRARY_PATH="${_NVLIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [ -d "$_VENV_SP/nvidia/cudnn" ]; then
    export CUDNN_PATH=$_VENV_SP/nvidia/cudnn
else
    export CUDNN_PATH=$_BASE_SP/nvidia/cudnn
fi
export CUDNN_HOME=$CUDNN_PATH
# TE 兜底 dlopen 找的是不带版本号的 libnvrtc.so,而 pip 轮子只有 .so.12;
# TE 的 CUDA_PATH 递归 glob 能匹配版本号 → 指到 nvidia 轮子根目录一次通杀
export CUDA_PATH=$_BASE_SP/nvidia

# megatron 是命名空间包:底层 slime-train 的 megatron-core 0.18.2 wheel 会在
# sys.path 顺序上压过 editable 的 dev 版(legacy .pth 排末尾),旧版没有 dsv4_hybrid。
# PYTHONPATH 置顶 vendor,强制 dev 版胜出:
export PYTHONPATH=$DSV4_TASK/vendor/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}

# hub 缓存统一放共享盘(数据集在 16 节点间只下一份)
export MODELSCOPE_CACHE=/lus/lfs1aip2/projects/public/u6gb/models/ms_cache
export HF_HOME=/lus/lfs1aip2/projects/public/u6gb/models/hf_home
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_TELEMETRY=1

# 编译类缓存放节点本地盘,不打 Lustre 元数据
export TRITON_CACHE_DIR=${TMPDIR:-/tmp}/triton_cache_${USER}
export CUDA_MODULE_LOADING=LAZY

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export CUDA_DEVICE_MAX_CONNECTIONS=1     # megatron sequence_parallel 要求
export TOKENIZERS_PARALLELISM=false

# NCCL 不做手工调优:X-VLA 4 节点 PyTorch 训练零 NCCL 配置跑得正常,照抄极简主义
