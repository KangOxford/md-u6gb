#!/bin/bash
# 用 pip 的 CUDA 12.8 组件拼一棵给 torch cpp_extension 用的工具链树。
# 背景:系统模块最高 CUDA 12.6,而 FlashMLA nv_dev 的源码用了 12.8 才有的
# __nv_fp8_e8m0;torch 本身是 cu12.8 构建。轮子有 aarch64 版。
# 前提:venv 已激活(pip 指向 venv)。
set -u -o pipefail

VENV=/home/u6gb/kangli.u6gb/envs/dsv4-venv
SP=$VENV/lib/python3.12/site-packages
TREE=/home/u6gb/kangli.u6gb/envs/dsv4-cuda128

pip install -q "nvidia-cuda-nvcc-cu12==12.8.*" "nvidia-cuda-runtime-cu12==12.8.*" \
               "nvidia-cuda-cccl-cu12==12.8.*" || exit 1
# crt 头(host_config.h 等)在新打包里可能单列;装得上就装,装不上再看
pip install -q "nvidia-cuda-crt-cu12==12.8.*" 2>/dev/null || true

mkdir -p $TREE/include $TREE/lib64
ln -sfn $SP/nvidia/cuda_nvcc/bin  $TREE/bin
ln -sfn $SP/nvidia/cuda_nvcc/nvvm $TREE/nvvm
for pkg in cuda_runtime cuda_cccl cuda_nvcc cuda_crt; do
    if [ -d $SP/nvidia/$pkg/include ]; then
        for f in $SP/nvidia/$pkg/include/*; do
            ln -sfn "$f" $TREE/include/$(basename "$f")
        done
    fi
done
for f in $SP/nvidia/cuda_runtime/lib/*; do
    ln -sfn "$f" $TREE/lib64/$(basename "$f")
done
# 关键自检:nvcc 可跑、e8m0 头在、crt 在
$TREE/bin/nvcc --version | tail -1 || exit 2
grep -ql "__nv_fp8_e8m0" $TREE/include/cuda_fp8.h* 2>/dev/null || grep -rql "e8m0" $TREE/include/cuda_fp8.hpp 2>/dev/null || echo "WARN: cuda_fp8 头里没搜到 e8m0(继续,编译见真章)"
[ -e $TREE/include/crt/host_config.h ] || echo "WARN: 缺 crt/host_config.h"
echo "CUDA128_TREE_OK $TREE"
