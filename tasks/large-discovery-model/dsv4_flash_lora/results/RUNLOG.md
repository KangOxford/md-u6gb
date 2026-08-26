# DeepSeek-V4-Flash LoRA 跑通记录(2026-08-26)

## 环境构建时间线

| 时刻 | 事件 | 结果 |
|---|---|---|
| 11:50 | 权重下载第一次启动(8 并发,hf_xet 在场) | ❌ 25/73 处无声 SIGKILL:login 节点 nproc 上限 1900,xet 的 rust 线程池打穿配额 |
| 12:01 | 构建第 1 次(nvcc 12.6 via hpc_sdk) | ❌ sanity 失败:TE 找不到 cudnn——venv 的 sysconfig 路径遮住了底层 slime-train 的 nvidia 库 |
| 12:05 | 下载重启(HF_HUB_DISABLE_XET=1 + 重试循环) | ✅ 稳定 ~130G/160G @12:30 |
| 12:10 | 构建第 2 次(pipefail 修复暴露真错) | ❌ fast_hadamard 编译失败:hpc_sdk 把 nvc++ 排上 PATH,被 torch 误当宿主编译器 |
| 12:15 | 构建第 3 次(CC=gcc-12 CXX=g++-12) | ❌ CUDA_HOME 指向 hpc_sdk `compilers/` 子树,其 include 的 NVHPC 版 `arm_neon.h` 毒害 g++(`__builtin_shufflevector` 报错) |
| 12:22 | 权重下载完成:159.6GB,17.5 分钟 | ✅ |
| 12:23 | 任务目录迁至 `tasks/large-discovery-model/dsv4_flash_lora`(用户指定工作目录);editable 断链由构建重跑修复 | ✅ |
| 12:23 | 构建第 4 次(CUDA_HOME=hpc_sdk `cuda/12.6` 干净子树 + venv 内升级 cudnn) | 🔄 编译中 |

### CUDA 版本问答存档(用户问「没有 cuda 13 吗?」)

模块表最高 CUDA 12.6(hpc_sdk 24.11)。就算有 13 也不能用:torch 是 cu12.8 构建,
扩展编译要求 CUDA **大版本**与 torch 一致(12 对 12);12.6 对 12.8 的小版本差只是警告。
需要 12.8 时走 pip 的 `nvidia-cuda-nvcc-cu12`,不动系统模块。

## 已知暗雷(未爆,已备预案)

1. **cudnn 9.10.2 偏老**:DSA 反向走 cudnn-frontend 1.27 的 DSA 图 API,底层 libcudnn
   可能需要 ≥9.15。预案:venv 内 `pip install -U nvidia-cudnn-cu12`,env_dsv4.sh 已改为
   venv 库优先加载。等 smoke 出信号再动。
2. **transformers 版本**:ms-swift 依赖解析可能在 venv 里装了不同于底层 5.15.0 的版本,
   sanity 会打印,若 deepseek_v4 模板报错再对齐。
3. **数据集竞态**:16 节点同时向 ModelScope 下载 → 已备 prefetch_datasets.py,单点预取。

## 关键决策(偏离官方文档处)

| 项 | 文档 | 本任务 | 理由 |
|---|---|---|---|
| merge_lora | true | **false** | merge 会在结尾写 ~500GB BF16 全量权重上 Lustre;跑通只需 LoRA 增量。要 merge 时单独跑一次 export |
| EP | 8(单机 8 卡) | **16** | GH200 单卡 96G:EP8 每卡 ~76G 权重太紧,EP16 每卡 ~44G 稳 |
| GBS | 32 | 64(64 卡)/16(16 卡 smoke) | Megatron 要求 GBS 整除 DP×MBS;TP=PP=1 时 DP=世界大小 |
| 宿主编译器 | (x86 无此问题) | CC/CXX=gcc-12 | hpc_sdk 的 nvc++ 会被 torch cpp_extension 误选 |
