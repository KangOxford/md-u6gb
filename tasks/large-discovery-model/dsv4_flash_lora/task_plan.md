# DeepSeek-V4-Flash LoRA 64 卡跑通(2026-08-26)

## 目标

按 ms-swift 官方最佳实践(docs/source/BestPractices/deepseek-v4.md),用 Megatron-SWIFT
在 **64 张 GH200(16 节点 × 4 卡)** 上把 DeepSeek-V4-Flash 的 **LoRA SFT 跑通**。
用户指令:「Flash先64卡跑通吧,LoRA就行」。

## 模型事实(来自 HF config.json)

| 项 | 值 |
|---|---|
| 总参数 | ~250-270B MoE(43 层,每层 256 路由专家 + 1 共享,每专家 3×4096×2048≈25M) |
| 激活参数 | ~10B 量级(topk=6) |
| 注意力 | V4 混合:DSA(稀疏,lightning indexer)+ CSA(压缩,compress_ratios)+ 滑窗 128 |
| 权重存储 | 专家 FP4、其余 FP8,共 159.6 GB,46 个 safetensors 分片 |
| 加载行为 | ms-swift 自动把 FP4 转成 BF16(FP4 训练不支持) |
| MTP | num_nextn_predict_layers=1(训练开 `--mtp_num_layers 1`) |

显存账:BF16 专家 ≈ 258B×2B。EP8 → 每卡 ~65G 专家 + ~11G 非专家,96G 卡太紧;
**EP16 → 每卡 ~33G 专家 + 11G 非专家,余量充足**。64 卡 = EP16 × DP4。

## 技术栈

| 组件 | 来源 | 状态 |
|---|---|---|
| torch 2.9.1+cu128 / TE 2.15.0 (ARM) | 现成环境 `slime-train`(openmle_rsi 任务,不改动) | ✅ 已有 |
| venv(--system-site-packages) | `/home/u6gb/kangli.u6gb/envs/dsv4-venv`(VAST,省 Lustre inode) | 待建 |
| Megatron-LM dev @ fd1121b8(文档测试 commit) | vendor/ 克隆,editable 安装 | ✅ 已克隆钉版 |
| ms-swift main @ 2ca4b90 | vendor/ | ✅ 已克隆 |
| mcore-bridge main @ 3197f74 | vendor/ | ✅ 已克隆 |
| nvidia-cudnn-frontend 1.27.0(DSA 反向) | pip aarch64 轮子 | ✅ 有轮子 |
| nvidia-cutlass-dsl 4.7.1(CuTe DSL) | pip 轮子 | ✅ 有轮子 |
| FlashMLA @ nv_dev(DSA 稀疏前向,SM90) | 源码编译(计算节点) | 待编 |
| fast_hadamard_transform(indexer 旋转) | 源码编译(计算节点) | 待编 |
| flash-attn | **不需要**(V4 路径零引用;常规注意力走 TE cuDNN fused) | 跳过 |

## 阶段

| # | 阶段 | 资源 | 状态 |
|---|---|---|---|
| 0 | 文档/模型/环境侦察 | login CPU | ✅ 完成 |
| 1 | 权重下载(159.6GB,8 并发) | login tmux `dsv4-dl` 窗口 | 🔄 进行中 |
| 2 | venv + 纯 Python 安装 + 两个 CUDA 内核编译 | attach 到 6136391 空节点(CPU 编译) | 待做 |
| 3 | 数据集下载(alpaca-gpt4 zh/en + self-cognition) | login CPU | 待做 |
| 4 | 单节点 4 卡加载冒烟(mini 4 层模型,验证 import/内核) | 6136391 空卡 attach | 待做 |
| 5 | 8-16 卡 LoRA 冒烟(EP8/EP16,2-4 节点 attach,几十步) | 6136391 的 4 节点空卡 | 待做 |
| 6 | 64 卡(16 节点)LoRA 正式跑通 | sbatch(名下最多 14 节点,必须排队) | 待做 |

## 关键决策记录

- **不动 slime-train 环境本体**,venv 继承 site-packages;新包全进 venv。
- venv 放 VAST home:u6gb 项目区 inode 只剩 ~58 万(50.62M/51.2M),不容一个新环境。
- Megatron 钉在文档测试过的 `fd1121b8`;ms-swift/mcore-bridge 取 main 并记录 commit。
- 注意力后端按文档 `--attention_backend flash`;DSA 前向=FlashMLA、反向=cudnn-frontend DSA,
  两者都装。CP/packing 不开,CuTe CP 布局内核 import 失败可容忍。
- 编译一律在计算节点(登录节点禁编译);TORCH_CUDA_ARCH_LIST=9.0 只编 SM90。
- 64 卡并行:LoRA 下 TP 不支持 → EP16 × DP4,`micro_batch_size` 视显存 1-4。

## 产出物

- 权重: `/lus/lfs1aip2/projects/public/u6gb/models/DeepSeek-V4-Flash`
- 环境: `/home/u6gb/kangli.u6gb/envs/dsv4-venv`
- 脚本: `code/`(download_model.py, build_env.sh, smoke_*.sh, launch_64gpu_lora.sbatch)
- 日志: `logs/`;结果与结论: `results/`
