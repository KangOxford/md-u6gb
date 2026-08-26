#!/bin/bash
# DeepSeek-V4-Flash LoRA 环境构建 —— 必须在计算节点上跑(登录节点禁止编译)
# 用法: srun --jobid=<alloc> --overlap -N1 -n1 -w <node> bash build_env.sh
set -u -o pipefail

TASK=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/dsv4_flash_lora
VENDOR=$TASK/vendor
VENV=/home/u6gb/kangli.u6gb/envs/dsv4-venv
BASE_PY=/lus/lfs1aip2/projects/public/u6gb/tasks/openmle_rsi_20260812/envs/slime-train/bin/python

step() { echo; echo "===== [$(date +%H:%M:%S)] $* ====="; }

step "0 主机与网络"
hostname; echo "cores: $(nproc)"
curl -sI -m 15 https://pypi.org/simple/ | head -1 || { echo "FATAL: 计算节点无出网"; exit 2; }

step "1 宿主编译器"
# hpc_sdk 把 nvc/nvc++ 排上 PATH,torch cpp_extension 会误选它们当宿主编译器
# (nvc++ 不认 GCC 旗标,nvcc 拒收 nvc 作 ccbin)。强制 GNU 工具链:
export CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12
$CXX --version | head -1 || { echo "FATAL: 没有 g++-12"; exit 3; }

step "2 venv (--system-site-packages, 继承 slime-train 的 torch/TE)"
[ -e $VENV/bin/python ] || $BASE_PY -m venv --system-site-packages $VENV
source $TASK/code/env_dsv4.sh     # 激活 venv + 挂 nvidia 库路径(与训练同款)
python -V; which python

step "2.5 CUDA 12.8.1 conda 工具链"
# 系统最高 12.6,FlashMLA nv_dev 要 12.8 头(__nv_fp8_e8m0);torch 是 cu12.8 构建。
# pip 的 aarch64 nvcc 轮子是阉割版(只有 ptxas),conda nvidia 渠道才有完整 nvcc。
export CUDA_HOME=/home/u6gb/kangli.u6gb/envs/cuda1281
[ -x $CUDA_HOME/bin/nvcc ] || { echo "FATAL: conda cuda1281 无 nvcc"; exit 3; }
export PATH=$CUDA_HOME/bin:$PATH
# conda cuda 包用 targets/sbsa-linux 布局:头文件给宿主编译器要走 CPATH,lib64 要补链接
export CPATH="$CUDA_HOME/targets/sbsa-linux/include${CPATH:+:$CPATH}"
[ -e $CUDA_HOME/lib64 ] || ln -sfn lib $CUDA_HOME/lib64
nvcc --version | tail -1; echo "CUDA_HOME=$CUDA_HOME"

step "3 构建工具 + 预编译轮子 (cudnn-frontend / cutlass-dsl)"
pip install -q -U pip wheel setuptools ninja packaging || exit 4
pip install -q nvidia-cudnn-frontend nvidia-cutlass-dsl || exit 4
# 注意:torch 2.9.1 精确钉 nvidia-cudnn-cu12==9.10.2.21,步骤 6 的 ms-swift 安装会把任何
# 提前升级降回去。若 DSA 反向真报 cudnn 版本不足,再手动:
#   pip install -U --no-deps nvidia-cudnn-cu12   (env_dsv4 已让 venv 库优先加载)

step "4 Megatron-LM dev@fd1121b8 (editable, --no-deps 防 torch 被动升级)"
pip install -q -e $VENDOR/Megatron-LM --no-deps || exit 5

step "5 mcore-bridge (editable)"
pip install -q -e $VENDOR/mcore-bridge || exit 6

step "6 ms-swift main (editable, 带依赖)"
pip install -q -e $VENDOR/ms-swift || exit 7

step "7 fast_hadamard_transform (源码编译, SM90)"
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=48
python -c "import fast_hadamard_transform" 2>/dev/null && echo "已存在,跳过" || \
  pip install -v --no-build-isolation git+https://github.com/Dao-AILab/fast-hadamard-transform.git 2>&1 | tail -5 || exit 8

step "8 FlashMLA nv_dev (源码编译, 只编 sm_90a)"
python -c "from flash_mla import flash_mla_sparse_fwd" 2>/dev/null && echo "已存在,跳过" || {
  cd $VENDOR/FlashMLA
  FLASH_MLA_DISABLE_SM100=1 FLASH_MLA_DISABLE_FP16=1 MAX_JOBS=48 \
    pip install -v --no-build-isolation . 2>&1 | tail -8 || exit 9
}

step "9 sanity imports"
cd $TASK   # 不能留在 FlashMLA 源码目录:heredoc 的 python 会把 cwd 加进 sys.path,
           # import 到源码树而不是装好的包(build6 的假阴性)
python - <<'EOF'
import torch, transformer_engine
print("torch", torch.__version__, "| TE", transformer_engine.__version__)
from cudnn import DSA; print("cudnn DSA ns OK")
import fast_hadamard_transform; print("fast_hadamard OK")
from flash_mla import flash_mla_sparse_fwd; print("flash_mla OK")
import megatron.core; print("megatron.core OK")
import mcore_bridge; print("mcore_bridge OK")
import swift; print("swift", swift.__version__)
import importlib.metadata as im
for p in ("transformers", "datasets", "modelscope", "peft", "trl"):
    try: print(p, im.version(p))
    except Exception as e: print(p, "MISSING")
EOF
[ $? -eq 0 ] && echo "===== BUILD_ENV_DONE =====" || { echo "===== SANITY FAILED ====="; exit 10; }
