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

| 12:33 | 构建第 5 次(pip 拼 CUDA 12.8 树) | ❌ aarch64 的 nvcc-cu12 轮子是阉割版:只有 ptxas/crt 头/libdevice,无 nvcc 本体 |
| 12:36 | conda nvidia 渠道建 `~/envs/cuda1281`(nvcc 12.8 + cudart-dev + cccl + 数学库头) | ✅ 完整工具链 |
| 12:38 | 构建第 6 次(conda CUDA 12.8.1) | ✅ **FlashMLA 编译通过**;sanity 报 flash_mla 假阴性(cwd 遮蔽,已修) |
| 12:42 | 中立目录九项 sanity | ✅ **全绿**:torch 2.9.1+cu128 / TE 2.15 / cudnn DSA ns / fast_hadamard / flash_mla / megatron.core 0.18.2+0c08601f / mcore_bridge / swift 4.6.0.dev0 / transformers 5.14.1 |
| 12:44 | TE 加载 nvrtc 失败(prefetch 中发现):pip 轮子只有 `.so.12`,TE 兜底 dlopen 找不带版本号的名字 | ✅ `CUDA_PATH` 指到 nvidia 轮子根目录,TE 的递归 glob 能配版本号 |
| 12:45 | 数据集预取完成:train 2979 / val 21,入共享缓存 | ✅ |
| 12:46 | 冒烟 `dsv4-smoke-mini` 提交(job 6143685,1N 4 卡,59 分钟时限) | 🔄 排队 |

补充:cudnn 升 9.24 被 ms-swift 安装步骤降回 9.10(torch 2.9.1 精确钉 9.10.2.21)。
先按 9.10 冒烟;若 DSA 反向报 cudnn 版本不足,`pip install -U --no-deps nvidia-cudnn-cu12`。

| 13:35 | 排队 1h 未调度;发现 nid010329 4 卡全空 → attach 冒烟(TAG=smoke_mini_att) | ✅ 规则「空卡+排队两侧同查」兑现 |
| 13:50 | mini 构建 v1 崩:Fp8Dequantizer 不认 FP4 融合专家张量(numel 2^32 reshape 失败) | ✅ 改为**量化原样**拷贝(保留 scale + quantization_config),让 bridge 走与全量模型同一条加载路径 |
| 13:57 | 训练崩:`Invalid experimental attention variant: dsv4_hybrid` | ✅ 根因:megatron 是命名空间包,底层 megatron-core 0.18.2 wheel 压过 editable dev 版(legacy .pth 排 sys.path 末尾);PYTHONPATH 置顶 vendor/Megatron-LM |
| 14:01 | 冒烟 v3 重跑(全部修复叠加) | ✅ **mini 冒烟通过**:74/74 步,loss 16.2→6.4,eval 9.04,LoRA checkpoint-74 落盘,exit 0;稳态 1.2s/it,显存 21G;**cudnn 9.10 足够 DSA 反向,无需升级** |
| 14:05 | 16 节点正式作业提交:**6144379**(dsv4-lora-64g,EP16×DP4,GBS64,6h 时限);冗余排队冒烟 6143685 已 `scontrol hold` | 🔄 排队 |
| 14:10 | 新链分配 6141106(4N16 卡)整片空转 → attach **full16 冒烟**(全量模型,EP16,TRAIN_ITERS=12) | 🔄 |
| 14:25 | full16 装载:43 层 × ~2.5s ≈ 2 分钟(FP4→BF16 bridge 转换 + Lustre 读) | ✅ 远快于预估 |
| 14:28 | **full16 首步迭代成功:loss 0.894 / grad_norm 0.27 / 每卡显存 52.3 GiB(EP16)**,余量 44G;64 卡同构无显存风险;继续跑满 111 步验证多节点保存 | ✅ 决定性 |

注:TRAIN_ITERS=12 未生效(swift 按 epoch 推导 111 步)——`--train_iters` 疑被
num_train_epochs 覆盖,对冒烟无害,正式跑用整 epoch 本来就是意图。

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
