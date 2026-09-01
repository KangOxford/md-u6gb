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

# 包缓存放节点本地,不占 /home 配额(101G 硬顶)。
# conda 的 pkgs_dirs 是一条**回退链而不是强制指定**:首选目录建不出来时它不报错,
# 静默滑到下一个可写目录 ~/.conda/pkgs —— 那正好在配额里面。2026-08-31 那次安装
# 就是这样撞的顶(报错路径是 /home/.../.conda/pkgs/cmake-4.4.3...),配置一次都没生效。
# 所以这里必须先建目录并验证可写,建不出来就停,不让它悄悄回退。
export CONDA_PKGS_DIRS=${TMPDIR:-/tmp}/conda_pkgs_$USER
mkdir -p "$CONDA_PKGS_DIRS" || { echo "FATAL: 建不出 $CONDA_PKGS_DIRS"; exit 2; }
[ -w "$CONDA_PKGS_DIRS" ] || { echo "FATAL: $CONDA_PKGS_DIRS 不可写"; exit 2; }
echo "[pkgs] conda 包缓存 -> $CONDA_PKGS_DIRS ($(df -h "$CONDA_PKGS_DIRS" | tail -1 | awk '{print $4}') 可用)"
export PIP_NO_CACHE_DIR=1
export PIP_CACHE_DIR="$CONDA_PKGS_DIRS/pip"      # 同理:pip 的缓存也别落 /home
export TMPDIR=${TMPDIR:-/tmp}                    # 源码编译的中间产物同样不落 /home
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

# 节点默认是 gcc 7.5.0(SUSE),而 torch 的 c10/util/C++17.h 直接 #error
# "We need GCC 9 or later" —— flash-attn / TE / torch_memory_saver 三段全会挂。
# 集群有 gcc-native/{12.3,13.2,14.2};取 12.3:CUDA 12.9 的 nvcc 对宿主编译器
# 支持到 GCC 13,12.3 在安全区内。
if [ -z "${LDMRL_SKIP_GCC_MODULE:-}" ]; then
  _gccver=$(gcc -dumpversion 2>/dev/null | cut -d. -f1)
  if [ -z "$_gccver" ] || [ "$_gccver" -lt 9 ] 2>/dev/null; then
    # 非交互 shell(srun bash -c)里 `module` 这个函数根本没定义,module load 会静默
    # 什么都不做 —— 实测加载"成功"后 gcc 仍是 7。所以先显式把 lmod 初始化进来。
    # 本机路径是 $MODULESHOME/init/bash = /opt/cray/pe/lmod/lmod/init/bash。
    if ! command -v module >/dev/null 2>&1; then
      for _init in "${MODULESHOME:-/opt/cray/pe/lmod/lmod}/init/bash" \
                   /opt/cray/pe/lmod/lmod/init/bash /usr/share/lmod/lmod/init/bash; do
        [ -f "$_init" ] && { . "$_init"; break; }
      done
    fi
    module load gcc-native/12.3 2>/dev/null || module load gcc/12.3 2>/dev/null || true
  fi
fi
export CC=${CC:-$(command -v gcc)}
export CXX=${CXX:-$(command -v g++)}
echo "[gcc] CC=$CC ($($CC -dumpversion 2>/dev/null))  CXX=$CXX"
_gccmajor=$($CC -dumpversion 2>/dev/null | cut -d. -f1)
if [ -z "$_gccmajor" ] || [ "$_gccmajor" -lt 9 ] 2>/dev/null; then
    echo "FATAL: gcc 版本 $($CC -dumpversion 2>/dev/null) < 9,torch 扩展编不了。"
    echo "       在计算节点上先 'module load gcc-native/12.3' 再跑本脚本。"
    exit 2
fi

# aarch64 特有:conda 的 CUDA 包把头文件放在 $PREFIX/targets/sbsa-linux/include
# (sbsa = NVIDIA 给 ARM64 服务器架构的目标名),**但只把 lib 链回了 $PREFIX/lib,
# include 那一侧没链**。而 flash-attn / torch_memory_saver / TE 的 setup.py 都按
# $CUDA_HOME/include 找 cuda_runtime.h,于是编译报 "No such file or directory"。
# 这种不对称最容易漏:lib 在,看起来工具链是完整的。
_TGT=$(ls -d "$ENV_PREFIX"/targets/*-linux 2>/dev/null | head -1)
if [ -n "$_TGT" ] && [ ! -e "$ENV_PREFIX/include/cuda_runtime.h" ]; then
  _n=0
  for _h in "$_TGT"/include/*; do
      _b=$(basename "$_h"); [ -e "$ENV_PREFIX/include/$_b" ] && continue
      ln -s "$_h" "$ENV_PREFIX/include/$_b" && _n=$((_n+1))
  done
  echo "[cuda] 从 $_TGT/include 补了 $_n 个头文件软链到 \$CUDA_HOME/include"
fi
[ -e "$ENV_PREFIX/include/cuda_runtime.h" ] || { echo "FATAL: \$CUDA_HOME/include 下没有 cuda_runtime.h,后面的源码编译必挂"; exit 2; }

# 同一个毛病的第二处:cuDNN。它不来自 conda 的 CUDA 包,而是 pip 包
# nvidia-cudnn-cu12,头文件落在 site-packages/nvidia/cudnn/include/,
# 库落在 .../cudnn/lib/。而 transformer_engine_torch 的编译按 $CUDA_HOME 找,
# 于是报 "fatal error: cudnn.h: No such file or directory"(2026-09-01 te 段挂在这)。
#
# 还有一层:pip 的 cuDNN **只发版本化的 libcudnn.so.9,没有裸的 .so** ——
# 那个包是给运行时 dlopen 用的,而链接器要的是 -lcudnn 能解析到的 libcudnn.so。
# 所以除了软链,还要补出裸名。
#
# cudnn_frontend 是另一个 pip 包(nvidia-cudnn-frontend),头文件直接在
# site-packages/include/ 下,TE 也要它。
_SP=$($PY -c "import site;print(site.getsitepackages()[0])" 2>/dev/null)
if [ -n "$_SP" ] && [ -d "$_SP/nvidia/cudnn/include" ]; then
  _n=0
  for _h in "$_SP"/nvidia/cudnn/include/*; do
      _b=$(basename "$_h"); [ -e "$ENV_PREFIX/include/$_b" ] && continue
      ln -s "$_h" "$ENV_PREFIX/include/$_b" && _n=$((_n+1))
  done
  _m=0
  for _l in "$_SP"/nvidia/cudnn/lib/*.so.*; do
      _b=$(basename "$_l")
      [ -e "$ENV_PREFIX/lib/$_b" ] || { ln -s "$_l" "$ENV_PREFIX/lib/$_b" && _m=$((_m+1)); }
      # libcudnn_graph.so.9 -> libcudnn_graph.so  (链接器要的裸名)
      _bare=${_b%%.so.*}.so
      [ -e "$ENV_PREFIX/lib/$_bare" ] || { ln -s "$_l" "$ENV_PREFIX/lib/$_bare" && _m=$((_m+1)); }
  done
  echo "[cudnn] 头文件软链 $_n 个,库软链 $_m 个(含裸 .so) -> \$CUDA_HOME"
fi
# cudnn_frontend(header-only)的头直接在 site-packages/include/ 下
if [ -n "$_SP" ] && [ -e "$_SP/include/cudnn_frontend.h" ]; then
  _n=0
  for _h in "$_SP"/include/cudnn_frontend*; do
      _b=$(basename "$_h"); [ -e "$ENV_PREFIX/include/$_b" ] && continue
      ln -s "$_h" "$ENV_PREFIX/include/$_b" && _n=$((_n+1))
  done
  echo "[cudnn] cudnn_frontend 头文件软链 $_n 个"
fi
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
  # 不要 `| tail -30`:编译失败时真正的 nvcc/g++ 错误在前面几百行,tail 只留下
  # Python 的包装 traceback("Error compiling objects for extension"),等于把病因扔了。
  # 2026-09-01 第一次编挂就是这样,只能重跑一遍才看到原因。全量写进单独的文件。
  FA_LOG="$LOGDIR/flashattn_build_$(date -u +%Y%m%dT%H%M%SZ).log"
  echo "[flashattn] 完整编译日志 -> $FA_LOG"
  # 架构名册:flash-attn 2.8.3 的 setup.py 读的是 **FLASH_ATTN_CUDA_ARCHS**,默认
  # "80;90;100;120" —— 四个架构,每个编译单元编四遍。它既不读 TORCH_CUDA_ARCH_LIST,
  # 也没有 FLASH_ATTENTION_DISABLE_SM80 这个开关(我第一次凭印象设的,没有生效)。
  # GH200 实测 compute_cap = 9.0,所以只留 90,编译量与峰值内存都降到 1/4。
  #   setup.py:69-70  return os.getenv("FLASH_ATTN_CUDA_ARCHS", "80;90;100;120").split(";")
  #
  # 并发:2026-09-01 用 MAX_JOBS=32 + --threads 4 编到 MaxRSS 302GB 被 OOM 杀掉
  # (sacct 记 OUT_OF_MEMORY)。四个架构那一版每个 TU 都很贵;现在只剩一个架构,
  # 但仍把默认并发压到 16 × 2 线程,并让调用方能覆盖。
  MAX_JOBS=${MAX_JOBS:-16} \
    NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:---threads 2}" \
    FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-90}" \
    $PIP install -v --no-cache-dir --no-build-isolation flash-attn==2.8.3 \
    > "$FA_LOG" 2>&1
  _fa_rc=$?
  if [ $_fa_rc -ne 0 ]; then
      echo "[flashattn] 失败(rc=$_fa_rc)。日志里第一条真实错误:"
      grep -m5 -nE "error:|fatal error|Killed|cannot find|undefined reference|No such file" "$FA_LOG" | sed 's/^/    /'
      echo "[flashattn] 全文见 $FA_LOG"
      exit 1
  fi
  # import 成功不等于装好:namespace package 残留也能 import。要求拿得到 __version__。
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
