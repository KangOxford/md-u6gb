#!/bin/bash
# vLLM glm-release 分支源码构建(PR#53906,GLM-5.3-Flash 原生支持;0.28.0 发行版没有,
# 静默退回 transformers 兜底后端后在 KDA 参数 k_conv1d 上崩——见 serve_6150683.out)。
# 必须在计算节点跑(登录节点禁止编译),预计 40-90 分钟。
# 用法: srun --overlap --jobid=<alloc> -w <node> -n1 --gres=gpu:1 bash build_vllm_glm.sh
set -u -o pipefail
SRC=/home/u6gb/kangli.u6gb/src/vllm-glm
VENV=/home/u6gb/kangli.u6gb/envs/glm53-vllm
step(){ echo; echo "===== [$(date +%H:%M:%S)] $* ====="; }

step "0 主机与出网"
hostname; echo "cores: $(nproc)"; uptime
curl -sI -m 15 https://pypi.org/simple/ | head -1 || { echo "FATAL: 计算节点无出网"; exit 2; }

step "1 工具链(dsv4 build_env.sh 验证过的组合)"
# hpc_sdk 的 nvc++ 会被误选为宿主编译器,强制 GNU;conda cuda1281 是完整 12.8 工具链
export CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12
$CXX --version | head -1 || { echo "FATAL: 没有 g++-12"; exit 3; }
# 第 5 轮教训:分支钉的 CUTLASS 需要 ≥12.9 的 nvcc 前端(12.8 在 cute/type_traits
# 折叠表达式上报错),默认用 cuda129;conda 布局的三处补齐做成幂等步骤
export CUDA_HOME=${CUDA_HOME_OVERRIDE:-/home/u6gb/kangli.u6gb/envs/cuda129}
[ -x $CUDA_HOME/bin/nvcc ] || { echo "FATAL: $CUDA_HOME 无 nvcc"; exit 3; }
export PATH=$CUDA_HOME/bin:$PATH
export CPATH="$CUDA_HOME/targets/sbsa-linux/include${CPATH:+:$CPATH}"
[ -e $CUDA_HOME/lib64 ] || ln -sfn lib $CUDA_HOME/lib64
(cd $CUDA_HOME/include && for f in ../targets/sbsa-linux/include/*; do ln -sfn "$f" .; done)
[ -e $CUDA_HOME/lib/libnvrtc.so ] || ln -sfn libnvrtc.so.12 $CUDA_HOME/lib/libnvrtc.so
nvcc --version | tail -1

step "2 venv 与构建工具"
source $VENV/bin/activate
# --no-build-isolation ⇒ build-system.requires 得全数预装(pyproject 清单照抄,
# setuptools 有上界 <81,setuptools-rust 是第一轮缺的那个)
pip install -q -U pip wheel jinja2 'cmake>=3.26.1' ninja 'packaging>=24.2' \
    'setuptools>=77.0.3,<81.0.0' 'setuptools-scm>=8.0' 'setuptools-rust>=1.9.0' || exit 4
python -c "import torch; print('torch', torch.__version__, '| cuda_avail:', torch.cuda.is_available())"

step "3 剥 torch 钉版 + 假版本号"
cd $SRC
python use_existing_torch.py || true
# depth-1 克隆没有 tag,setuptools_scm 拿不到版本
export SETUPTOOLS_SCM_PRETEND_VERSION=0.28.1.dev0
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.28.1.dev0

step "4 编译安装(sm90;vllm cmake 会自己给 cutlass/fp8 内核升 9.0a)"
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=48
export VLLM_TARGET_DEVICE=cuda
pip install -e . --no-build-isolation || { echo "FATAL: 编译失败"; exit 5; }

step "5 sanity(必须换出源码目录防 cwd 遮蔽)"
cd /tmp
python - <<'EOF'
import vllm, torch
print("vllm", vllm.__version__, "| torch", torch.__version__, "| cuda_avail:", torch.cuda.is_available())
from vllm.model_executor.models.registry import ModelRegistry
archs = [a for a in ModelRegistry.get_supported_archs() if 'glm5' in a.lower()]
print("Glm5 archs:", archs)
assert archs, "registry 里没有 Glm5*,分支不对或安装未生效"
import importlib.metadata as im
for p in ("flashinfer-python","transformers","xgrammar"):
    try: print(p, im.version(p))
    except Exception: print(p, "ABSENT")
EOF
[ $? -eq 0 ] && echo "===== VLLM_GLM_BUILD_DONE =====" || { echo "===== BUILD_SANITY_FAILED ====="; exit 10; }
