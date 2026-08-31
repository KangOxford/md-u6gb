#!/bin/bash
# 训练栈(slime + Megatron + TE + sglang)在 aarch64/GH200 上的安装。
# 移植自 rl/slime/build_conda.sh,替换项见 ldm_rl/PLAN.md §2。
#
# 必须在计算节点上跑(有编译步骤;登录节点禁止编译)。
# 分段执行: STAGE=<名字> bash install_train_env.sh   不给 STAGE 则全跑。
# 每段幂等,失败后可单独重跑那一段。
set -u -o pipefail

ENV_PREFIX=${ENV_PREFIX:-/home/u6gb/kangli.u6gb/envs/ldm-rl-train}
WORK=${WORK:-/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model}
LDM=$WORK/LDM-rl
SLIME_DIR=$LDM/rl/slime
MEGATRON_DIR=$WORK/Megatron-LM
LOGDIR=$WORK/ldm_rl/logs
mkdir -p "$LOGDIR"

# 作者钉的版本(build_conda.sh 头部),原样保留以便对账
SGLANG_VERSION="0.5.15.post1"
MEGATRON_COMMIT="1dcf0dafa884ad52ffb243625717a3471643e087"
TMS_COMMIT="8d30c59ca12a68d9deccbc9c6599076a1218cbc5"
SGLROUTER_TAG="v0.3.2-9daabcd"

export CONDA_PKGS_DIRS=/home/u6gb/kangli.u6gb/.conda/pkgs
export PIP_CACHE_DIR=/home/u6gb/kangli.u6gb/.cache/pip
CONDA=/home/u6gb/kangli.u6gb/miniforge3/bin/conda
PY=$ENV_PREFIX/bin/python
PIP=$ENV_PREFIX/bin/pip

say() { echo "[$(date -u +%H:%M:%S)] $*"; }
run_stage() { [ -z "${STAGE:-}" ] || [ "${STAGE}" = "$1" ]; }

# ---------------------------------------------------------------- 1. env + cuda
if run_stage env; then
  say "=== [env] 建环境 + CUDA 12.9.1 工具链 ==="
  [ -x "$PY" ] || $CONDA create -p "$ENV_PREFIX" python=3.12 pip -c conda-forge -y || exit 1
  # TE / flash-attn / torch_memory_saver 编译要 CUDA_HOME 下的头文件与 nvcc
  $CONDA install -p "$ENV_PREFIX" -y \
      cuda=12.9.1 cuda-nvtx=12.9.79 cuda-nvtx-dev=12.9.79 nccl \
      -c nvidia/label/cuda-12.9.1 -c nvidia -c conda-forge || exit 1
  # sgl-router 要 Rust
  $CONDA install -p "$ENV_PREFIX" -y -c conda-forge rust cmake ninja || exit 1
fi

export CUDA_HOME="$ENV_PREFIX"
# GH200 是 Hopper sm90。显式给出,让 flash-attn/TMS 编译不必探测在用的 GPU。
export TORCH_CUDA_ARCH_LIST="9.0"
export PATH="$ENV_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# ---------------------------------------------------------------- 2. torch + sglang
if run_stage sglang; then
  say "=== [sglang] sglang 轮子 + 强制回退到 cu129 ==="
  # 与 build_conda 的差别:作者用源码 `pip install -e sglang python[all]`,那要 Rust
  # 编 sglang-grpc;aarch64 上 PyPI 已有该版本轮子,直接用轮子。
  $PIP install --no-cache-dir "sglang[all]==${SGLANG_VERSION}" \
      --extra-index-url https://download.pytorch.org/whl/cu129 || exit 1
  # sglang 元数据要 torch==2.11.0(PyPI 默认 cu13) + flashinfer[cu13];驱动 565=CUDA12.7
  # 装上 cu13 会让 torch.cuda.is_available()=False。按作者的办法强制换回 cu129。
  $PIP install --no-cache-dir --force-reinstall --no-deps \
      torch==2.11.0+cu129 torchvision torchaudio==2.11.0+cu129 \
      --index-url https://download.pytorch.org/whl/cu129 || exit 1
  # sgl-deep-gemm 作者钉 0.1.4,但 cu129 索引里 aarch64 最低是 0.1.5rc3
  $PIP install --no-cache-dir --force-reinstall --no-deps \
      sglang-kernel==0.4.4 sgl-deep-gemm==0.1.5rc3 \
      --index-url https://docs.sglang.ai/whl/cu129/ || exit 1
  say "--- 卸掉 sglang 拖进来的 cu13 运行库,换 cu12 ---"
  $PIP uninstall -y nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc \
      nvidia-cuda-runtime nvidia-cudnn-cu13 nvidia-cufft nvidia-cufile \
      nvidia-curand nvidia-cusolver nvidia-cusparse nvidia-cusparselt-cu13 \
      nvidia-nccl-cu13 nvidia-nvjitlink nvidia-nvshmem-cu13 nvidia-nvtx \
      nvidia-cutlass-dsl-libs-cu13 || true
  $PIP install --no-cache-dir --force-reinstall --no-deps \
      nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
      nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12==9.16.0.29 nvidia-cufft-cu12 \
      nvidia-cufile-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 \
      nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12 \
      nvidia-nvjitlink-cu12 nvidia-nvshmem-cu12 nvidia-nvtx-cu12 \
      --index-url https://download.pytorch.org/whl/cu129 \
      --extra-index-url https://pypi.org/simple || exit 1
  say "--- 验证 torch 认得卡(这一步在登录节点必然 False,只在计算节点有意义) ---"
  $PY -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available())"
fi

# ---------------------------------------------------------------- 3. flash-attn(源码编)
if run_stage flashattn; then
  say "=== [flashattn] flash-attn 2.8.3 源码编(社区轮子只有 x86) ==="
  MAX_JOBS=${MAX_JOBS:-64} NVCC_APPEND_FLAGS="--threads 4" \
    $PIP install -v --no-cache-dir --no-build-isolation flash-attn==2.8.3 \
    2>&1 | tail -30
  $PY -c "import flash_attn;print('flash_attn', flash_attn.__version__)" || exit 1
fi

# ---------------------------------------------------------------- 4. TE + FLA
if run_stage te; then
  say "=== [te] transformer_engine 2.16.0(aarch64 预编译;作者钉的 2.16.1 只有 x86) ==="
  $PIP install --no-cache-dir --no-build-isolation \
      "transformer_engine[pytorch]==2.16.0" || exit 1
  say "=== [te] flash-linear-attention 0.4.2(Qwen3.5 GDN 的默认后端) ==="
  # FlashQLA 跳过:它要 tilelang,而 tilelang 只发 x86 轮子。FLA 是 triton 实现,与架构无关。
  $PIP install --no-cache-dir flash-linear-attention==0.4.2 || exit 1
fi

# ---------------------------------------------------------------- 5. torch_memory_saver
if run_stage tms; then
  say "=== [tms] torch_memory_saver(必须 --no-build-isolation,否则出纯 python 轮子) ==="
  TMS_CUDA_MAJOR=$($PY -c 'import torch;print(torch.version.cuda.split(".")[0])')
  export TMS_CUDA_MAJOR
  $PIP install -v --no-cache-dir --force-reinstall --no-build-isolation \
      "git+https://github.com/zhuzilin/torch_memory_saver.git@${TMS_COMMIT}" 2>&1 | tail -20
  # 判据:装出来的包里必须有编译好的 .so,纯 python(~46KB)说明编译被跳过了
  $PY - <<'PYEOF'
import pathlib, torch_memory_saver as t
d = pathlib.Path(t.__file__).parent
so = list(d.rglob('*.so'))
print('torch_memory_saver at', d, '  .so 数量 =', len(so))
assert so, 'FATAL: 没有编译产物,sglang 会报 Only hook_mode=preload supports pauseable CUDA Graph'
PYEOF
fi

# ---------------------------------------------------------------- 6. sglang_router(Rust 编)
if run_stage router; then
  say "=== [router] slime 的 sgl-router fork(release 只有 x86,源码编) ==="
  $PIP install --no-cache-dir --force-reinstall \
      "git+https://github.com/zhuzilin/sgl-router.git@${SGLROUTER_TAG}" 2>&1 | tail -20
  $PY -c "import sglang_router;print('router',sglang_router.__version__)" || exit 1
fi

# ---------------------------------------------------------------- 7. Megatron
if run_stage megatron; then
  say "=== [megatron] Megatron-LM @ ${MEGATRON_COMMIT} ==="
  if [ ! -d "$MEGATRON_DIR/.git" ]; then
      git clone https://github.com/NVIDIA/Megatron-LM.git "$MEGATRON_DIR" || exit 1
  fi
  git -C "$MEGATRON_DIR" checkout "$MEGATRON_COMMIT" || exit 1
  # --no-build-isolation:setup.py 会 shell 出去调 `python3 -m pybind11` 编
  # megatron.core.datasets.helpers_cpp;隔离环境里找不到 pybind11 就"静默跳过",
  # 之后 GPT 数据集加载会坏 —— 这是个不报错的失败,必须先装好构建依赖。
  $PIP install --no-cache-dir "setuptools<80.0.0" pybind11 "packaging>=24.2" || exit 1
  (cd "$MEGATRON_DIR" && $PIP install -e . --no-build-isolation) || exit 1
  $PY -c "import megatron.core.datasets.helpers_cpp as h;print('helpers_cpp OK', h.__file__)" \
      || echo "!! helpers_cpp 未编出来,GPT 数据集路径会坏(本任务用自定义 rollout,可能不致命)"
fi

# ---------------------------------------------------------------- 8. slime
if run_stage slime; then
  say "=== [slime] 依赖 + 本体(--no-deps,别让 pip 重解析踩掉钉住的原生库) ==="
  $PIP install --no-cache-dir -r "$SLIME_DIR/requirements.txt" || exit 1
  (cd "$SLIME_DIR" && $PIP install -e . --no-deps) || exit 1
  say "--- 收尾钉版(照 build_conda) ---"
  $PIP install --no-cache-dir nvidia-cudnn-cu12==9.16.0.29 || true
  $PIP install --no-cache-dir "numpy==1.26.4" "scipy==1.17.1" || true
  # kernels 0.15.x 会让 `import sglang` 抛 ValueError
  $PIP install --no-cache-dir "kernels<0.15.0" || true
fi

# ---------------------------------------------------------------- 9. 补丁
if run_stage patch; then
  say "=== [patch] 应用 slime 的 sglang 补丁 ==="
  PATCH_DIR="$SLIME_DIR/docker/patch/v${SGLANG_VERSION}"
  SGLANG_PKG=$($PY -c "import sglang,pathlib;print(pathlib.Path(sglang.__file__).parents[1])" 2>/dev/null)
  say "补丁目录 $PATCH_DIR ; sglang 安装在 $SGLANG_PKG"
  ls "$PATCH_DIR" 2>/dev/null || echo "!! 补丁目录不存在,跳过(需人工核对)"
fi

# ---------------------------------------------------------------- 10. 自检
if run_stage verify; then
  say "=== [verify] 逐项 import ==="
  $PY - <<'PYEOF'
import importlib, sys
mods = ["torch","transformer_engine","transformer_engine.pytorch","sglang",
        "sglang_router","megatron.core","slime","ray","fla","flash_attn",
        "torch_memory_saver"]
bad = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"  OK   {m:32s} {getattr(mod,'__version__','')}")
    except Exception as e:
        print(f"  FAIL {m:32s} {type(e).__name__}: {e}")
        bad.append(m)
import torch
print(f"\n  torch {torch.__version__}  cuda={torch.version.cuda}  available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print("  device:", torch.cuda.get_device_name(0))
sys.exit(1 if bad else 0)
PYEOF
fi
say "=== 结束 ==="
