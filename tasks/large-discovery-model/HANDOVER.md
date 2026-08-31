# HANDOVER — LDM 复现实验交接

**写于** 2026-08-31 · **交接人** junming · **工作目录** `tasks/large-discovery-model/`

---

## 1. 现在跑到哪了

**目标**:用 GLM-5.3-Flash 作提案模型,复现 LDM v0.1（[arXiv:2608.15669](https://arxiv.org/abs/2608.15669)）
的 C1 结论 —— *nanoGPT 训练代码搜索里,LDM 的验证 BPB 绝对下降幅度是 LLM-only 反思的 **2.4 倍***。

| 项 | 状态 |
|---|---|
| 单次真评测跑通 | ✅ `val_bpb 1.019308`,300s 训练,MFU 33.3%,50.3M 参数 |
| GLM-5.3-Flash 服务 | ✅ TP4 单机四卡,冒烟对话通过 |
| A/B 对照框架 | ✅ 两组配置 + 契约 profile 就绪 |
| **有效 A/B 数据** | ❌ **还没有。这是你要接着做的事** |

**当前作业:`6220124`**（3 节点独占,12h,自带服务)。跑完 6 条 campaign
（3 seed × 2 组),产物落 `Large-Discovery-Models/tasks/nanogpt/runs/repro_arm*/`。

### 前面几轮为什么没出数据

| 作业 | 死因 | 已修 |
|---|---|---|
| 6204550 | 服务在另一个分配,那分配到期后服务消失,6 条 campaign 在 `glm_health=000` 下空跑 1h | 服务与实验同作业 + 健康不通过 `exit 4` |
| 6216995 | ① launcher 自设 `CUDA_VISIBLE_DEVICES=1/2/3`,而 `--gres=gpu:1` 已隔离好卡 → 指向不存在的设备,每次评测 11s 返回哨位值 `1e+09`<br>② 起服漏了 `--tool-call-parser`,`operation_tool` 走 tool calling → 30/30 候选 `400 Bad Request` | 删掉 CVD;加回标志;起跑前**真发一次带 tools 的请求**,失败 `exit 10` |

**共性**:前置检查查的是「进程在不在」,真正会坏的是「这条具体路径通不通」。

---

## 2. 你接手要做的三步

### 第一步 —— 看作业跑完没

```bash
squeue -j 6220124
sacct -j 6220124 --format=JobID,State,ExitCode -nP     # 不在 squeue ≠ 成功
tail -c 2000 tasks/large-discovery-model/glm53_flash/logs/sh_armA_s1_6220124.log | tr '\r' '\n' | tail -5
```

**先看这一条**:`best_score` 是不是 `1e+09`。是的话说明所有候选评测都失败,
数据无效,别往下分析。

```bash
python3 -c "
import json,glob
for d in sorted(glob.glob('tasks/large-discovery-model/Large-Discovery-Models/tasks/nanogpt/runs/repro_arm*/summary.json')):
    s=json.load(open(d)); print(d.split('/')[-2], s['best_score'], s['evaluation_count'])"
```

### 第二步 —— 算论文那个比值

每组从**共同起点**算 val_bpb 的绝对下降 Δ,然后比 `Δ_A / Δ_B` 与论文的 2.4×。

- 起点 = warmup 里那份原始 `real_train.py` 的 val_bpb
- Δ = 起点 − 该组最好的 val_bpb
- 三个 seed 成对比（同 seed 的 A/B 落在同一节点,节点差异已抵消）

**单条轨迹不构成因果估计**（论文自己也这么写）。三个 seed 至少要能给出方向一致性。

### 第三步 —— 重跑（如果数据还是不行）

```bash
cd tasks/large-discovery-model/glm53_flash
gtop                                      # 铁律:sbatch 前先看有没有空卡
sbatch code/launch_repro_selfhosted.sbatch
```

改预算就编辑 `code/ng_A_ldm.yaml` / `ng_B_llmonly.yaml` 的 `iterations`,
**同时改 `Large-Discovery-Models/tasks/nanogpt/experiment.json` 里对应 profile 的
`locked_args`**,否则契约校验会拦下来。

---

## 3. 对照实验的设计要点（改配置前必读）

**两组只差一个参数**:

| | A 组（LDM） | B 组（LLM-only） |
|---|---|---|
| `acquisition-feedback` | **brief** | **none** |
| 其余全部 | 完全相同 | 完全相同 |

两个容易搞错的地方:

1. **`breadth × depth` 是真评测次数,不是「候选池大小」**。每一个都要跑一次 300 秒
   真训练。`breadth=4 depth=4` 20 轮 = 345 次评测（35 小时）;`breadth=1 depth=1`
   = 45 次（4.5 小时）。两组设不同 ⇒ 预算差 7.7 倍,比出来的差距会被
   「多跑了 7 倍」解释掉。论文提交的配置就是 `breadth=1 depth=1`。

2. **LDM 的机制开关是 `--acquisition-feedback`,默认 `none`**。它才是把 GP 的
   预测与不确定性喂回提案 prompt 的那个参数。不是 breadth/depth,也不是
   `surrogate-mode`。

3. **定时评测必须独占 GPU**。评测是跑满 300 秒报 bpb —— 被抢卡就是同样时间跑更少
   token,bpb 变差。**GPU 争抢会把「哪组邻居更闲」伪装成「哪组方法更好」。**
   `sbatch` 必须带 `--exclusive`。

---

## 4. 服务器配置（Isambard-AI Phase 2）

官方规格页:<https://docs.isambard.ac.uk/specs/>

### 集群

| 项 | 值 |
|---|---|
| 节点数 | **1,320**（aarch64 / ARM） |
| GPU 总数 | 5,280 × NVIDIA GH200 Grace Hopper Superchip |
| 互联 | Slingshot 11,每节点 **4 × Cassini NIC,每张 200 Gbps** |
| 调度 | Slurm,分区 `workq`,默认时限 4h,`MaxTime=UNLIMITED` |

### 单个计算节点

| 项 | 值 |
|---|---|
| Superchip | 4 × GH200（每个 = 1 Grace CPU + 1 Hopper H100） |
| CPU | 4 × Grace，**4 × 72 = 288 核**，Neoverse-V2 |
| CPU 内存 | 4 × 120 GB（**可用 460 GB/节点**，115 GB/CPU） |
| GPU 内存 | 4 × 96 GB（合计 384 GB） |
| 内存合计 | 844 GB（CPU + GPU） |
| 节点内互联 | NVLink-C2C（CPU↔GPU）；GPU 间 **NV6**（每对 6 条 NVLink） |
| 功耗 | 每个 GH200 封顶 **660 W**，CPU/GPU 动态分配 |
| 驱动 / CUDA | **565.57.01 / CUDA 12.7**，compute capability 9.0 |

> `free -g` 会报 856 GB —— 那是 Grace-Hopper 统一内存的读数，
> 与官方「460 GB 可用 CPU 内存」口径不同，按官方数规划。

### 登录节点

**没有 GPU**。1 × Grace CPU Superchip，2 × 72 核，2 × 120 GB。
只能改文件、提作业、跑轻量 CPU 脚本（<30 分钟、<16 GB、顺序读写）。

### 存储

| 路径 | 用途 | 配额 |
|---|---|---|
| `/lus/lfs1aip2/projects/public/u6gb/` | 项目盘（Lustre） | 200 T / **5120 万 inode** |
| `$HOME` | venv、小文件 | 另一个文件系统，15 P |

⚠️ **venv 不要建在项目盘**：一个带 torch 的 venv 几万个文件，几个就把 inode 吃掉。

⚠️ **ARM 架构**：x86_64 编译的代码和 conda 环境**都不能用**，容器镜像必须是 arm64。

⚠️ **Lustre 禁令**：递归 `ls`／大范围 `find`／`du -sh` 大目录／`tree`／用 `ls` 找最新
checkpoint —— 这些会打爆元数据服务。2026-05-08 全组作业因此被挂起过一次。

---

## 5. 多卡实测结果

### 单机多卡 ✅ 已验证

**GLM-5.3-Flash 张量并行 TP4**（一个节点四张 GH200）：

```
306 GB FP8 权重 → 每卡约 77 GB
装载 1–4 分钟 + DeepGEMM 预热约 1000 内核 + CUDA graph ≈ 首次 10 分钟
起来后 OpenAI 兼容接口，冒烟对话（含思维链 + 算术）通过
```

起法：`bash glm53_flash/code/attach_serve_sif.sh <jobid> <node>`

三个必需参数（少一个就起不来）：
- `--max-num-seqs 256` —— KDA 层状态块预算上限 512，默认 1024 会拒绝启动
- **不要** `--kv-cache-dtype fp8` —— `fp8_ds_mla` 内核要求 `pe_dim=64`，与 NoPE-MLA 不容
- 容器内 `TMPDIR=/tmp` —— 宿主 TMPDIR 路径容器里不存在，DeepGEMM JIT 的 nvcc 写不出临时文件

### 多机多卡 ⚠️ 未跑通，但根因已定位

跑 2 节点 × 4 卡的 NCCL all-reduce，**三次都挂死在 `init_process_group`**，无任何输出。

**根因**：torch 轮子自带的 NCCL **没有 Slingshot/libfabric 传输层**，多机通信初始化直接挂。
仅设 `NCCL_SOCKET_IFNAME=hsn0-3` 不够。

**已知可用的配方**（另一条线 16 节点跑通过，见
`/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_R1_Mamba3/node_wrapper.sh`）：

```bash
Q=/projects/public/s5e/quant_team/quant
export LD_LIBRARY_PATH=$Q/nccl-2.29.3/lib:$Q/aws-ofi-nccl-1.18.0/lib:/opt/cray/libfabric/1.22.0/lib64:$LD_LIBRARY_PATH
export LD_PRELOAD=$Q/nccl-2.29.3/lib/libnccl.so.2${LD_PRELOAD:+:$LD_PRELOAD}
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_IB_DISABLE=1        # Slingshot 是以太网底座，不是 IB
export NCCL_BUFFSIZE=2097152    # 2 MB，补自适应路由的 latency jitter
export NCCL_P2P_DISABLE=0       # 绝不设 1，会废掉 NVLink
```

那份 wrapper 的注释里写着：AWS OFI 插件用 1.8.1 版会导致
「512N（2048 GPU）NCCL comm init hang」——**和我撞的是同一个症状**（我这边压根没挂插件）。

我的测试脚本在 `glm53_flash/code/{test_multinode.py,run_multinode_test.sh}`，
已经把上面这些配置写进去了，但**没有在自建 NCCL 环境下验证过**——
`ldm-nanogpt` 那个 venv 装的是 pip 版 torch 2.9.1+cu128，
要跑多机得先确认 `LD_PRELOAD` 的自建 NCCL 与它 ABI 兼容。**这是留给你的一个待办。**

LDM 目前的实验**不需要多机**：每条 campaign 只用 1 张卡，服务用 4 张，
横向扩展靠多起几条 campaign 而不是把一条铺开。多机是 LDM-RL 训练才会用到。

---

## 6. 环境与关键路径

```bash
~/envs/ldm-venv       # LDM 共享层 + openai 客户端
~/envs/ldm-causal     # causal_discovery 任务
~/envs/ldm-nanogpt    # nanoGPT 任务（含 torch 2.9.1+cu128，要 GPU）
```

**torch 版本别乱升**：驱动 565 = CUDA 12.7，cu128/cu129 能用，
**cu130 装上去 `torch.cuda.is_available()` 直接是 False**——PyPI 的默认 torch 轮子
已经切到 CUDA 13 了，装之前先看 `+cuXXX` 后缀。

```
tasks/large-discovery-model/
├── Large-Discovery-Models/       # 上游仓库 b5dab16a
├── models/GLM-5.3-Flash/         # 306 G FP8 权重
├── data/autoresearch_cache/      # nanoGPT 数据 + BPE 词表（788 M）
├── vendor/MLS-Bench/             # 钉版 cfd57a7e
├── glm53_flash/
│   ├── code/                     # 起服、A/B 配置、启动器、多机测试
│   ├── images/*.sif              # vLLM 官方 arm64-cu129 镜像 9.5 G
│   ├── logs/                     # 全部运行日志
│   └── results/{RUNLOG,REPRO_PLAN}.md
├── docs/RUNBOOK_LDM_ON_ISAMBARD.md   # 完整实操手册（比本文详细）
└── HANDOVER.md                       # 本文
```

**遇事先看 `glm53_flash/results/RUNLOG.md`** —— 逐条时间线，
每个坑什么时候踩的、怎么定性的、怎么修的都在里面。
