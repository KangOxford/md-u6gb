# Hybrid 序列模型架构与本地部署报告

**报告日期**：2026-08-11  
**报告范围**：AI21 Jamba2 3B、NVIDIA Nemotron Nano 9B v2、Moonshot Kimi Linear 48B-A3B、Moonshot Kimi K3  
**证据口径**：固定模型 revision、官方模型配置与技术报告、本地源代码，以及 Isambard-AI GH200 上的真实推理记录。本文区分“架构可分析”“权重已下载”“真实推理成功”三种状态。

## 执行摘要

Hybrid 不是把两种模型简单拼接，也不是一个单独的新算子。它的核心是：**让不同的信息通路承担不同工作**。

- Mamba/SSM 或 KDA 负责绝大多数长序列扫描，把历史压进固定大小的递归状态，降低长上下文成本。
- 少量全局 Attention 或 MLA 层保留按内容直接访问任意历史 token 的能力，弥补有限状态压缩造成的信息损失。
- MoE 在通道维度只激活少数专家，用较小的每-token 计算量承载更大的总参数容量。
- Kimi K3 再加入 AttnRes，让层在“深度方向”选择前面哪些块值得读取，并加入原生视觉通路。

因此，这四个系统实际上覆盖了两条主要路线：

1. **SSM + Attention**：Jamba2、Nemotron，用状态空间递归承担主干计算，用少量注意力恢复全局检索能力。
2. **Linear Attention + MLA + MoE**：Kimi Linear、Kimi K3，用 KDA 状态矩阵承担主干计算，用周期性 MLA 做全局内容寻址，再用稀疏专家扩展模型宽度。

本地验证结论是：Jamba2 3B、Nemotron Nano 9B v2 和 Kimi Linear 48B-A3B 已完成真实 GPU 生成；Kimi K3 已完成固定版本的源码、配置、技术报告和部署配方归档，但由于容量与受支持的并行拓扑不足，未被错误标记为已部署。

## 1. “Hybrid”到底混合了什么

| 混合维度 | 主要组件 | 组件分工 | 在本报告中的实例 |
|---|---|---|---|
| 序列 / token 混合 | Mamba、KDA、Attention、MLA | 递归层负责低成本长序列扫描；全局层负责精确内容寻址 | 四个模型全部涉及 |
| 通道 / width 混合 | Dense FFN、Sparse MoE、LatentMoE | FFN 做每个 token 的非线性变换；MoE 只路由到少数专家 | Kimi Linear、Kimi K3 |
| 深度 / layer 混合 | 普通 residual、AttnRes | 普通残差逐层累积；AttnRes 按权重读取较早层或块 | Kimi K3 |
| 模态混合 | 文本 embedding、视觉编码器、projector | 把图像/视频特征映射到语言主干的共享空间 | Kimi K3 |
| 系统 / 部署混合 | 单卡、张量切分、专家并行、量化 | 架构节省的计算不等于权重能装进单卡；需要匹配内存与通信拓扑 | Kimi Linear、Kimi K3 |

一个典型 Hybrid 主干可以抽象为：

```text
token / visual embeddings
          │
          ▼
┌─────────────────────────────────────────────┐
│ 高比例高效 mixer：Mamba 或 KDA              │
│ - 递归更新固定大小状态                      │
│ - 主要承担长上下文扫描                      │
├─────────────────────────────────────────────┤
│ 低比例全局 mixer：Attention 或 MLA          │
│ - 直接读取历史 token                        │
│ - 周期性恢复全局内容寻址                    │
├─────────────────────────────────────────────┤
│ Dense FFN 或 Sparse MoE                     │
│ - 做通道变换                                │
│ - MoE 只激活 Top-k 专家                     │
└─────────────────────────────────────────────┘
          │
          ▼
普通 residual，或跨层/跨块选择性的 AttnRes
```

关键点是：**Hybrid 的目标不是让每个 token 同时执行所有昂贵路径，而是让不同路径以不同频率、不同稀疏度工作。**

## 2. 五个核心组件如何工作

### 2.1 全局 Attention：能力强，但上下文成本随长度增长

标准因果自注意力可以写为：

$$
A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt d}+M_{\text{causal}}\right),\qquad O=AV.
$$

它的优势是每个 query 都能按内容直接选择历史 token；弱点是长序列 prefill 的注意力矩阵成本随序列长度近似二次增长，解码时 KV cache 也随上下文长度增长。

MLA（Multi-head Latent Attention）仍然是全局注意力，但先把每个 token 的 K/V 压缩为低维 latent，再在计算时上投影重建各头的内容 K/V。它没有取消全局寻址，而是主要降低 KV cache。Kimi 系列把 MLA 放在少数周期性层中，因此全模型仍能进行全局检索，但不必让所有层都承担完整 KV 成本。

### 2.2 Mamba / Mamba-2：用选择性状态递归压缩历史

状态空间层的核心可抽象为：

$$
h_t=\bar A_t h_{t-1}+\bar B_t x_t,\qquad y_t=C_t h_t+D x_t.
$$

与固定线性系统不同，Mamba 让状态更新参数依赖当前输入，因此可以选择性地保留、遗忘或写入信息。推理时，历史被压缩进固定形状的状态；每增加一个 token，不需要重新读取完整历史 KV。代价是这种状态是有损摘要，不具备全注意力那种“回到第 73,421 个 token 精确取值”的天然能力，所以 Jamba2 和 Nemotron 都保留少量 Attention 层作为全局校正通路。

### 2.3 KDA：把线性注意力变成可改写、可遗忘的关联记忆

Kimi Delta Attention（KDA）维护每个头的矩阵状态 $S_t\in\mathbb{R}^{d_k\times d_v}$。其核心更新是：

$$
S_t=\left(I-\beta_t k_tk_t^\top\right)\operatorname{Diag}(\alpha_t)S_{t-1}+\beta_tk_tv_t^\top,
\qquad o_t=S_t^\top q_t.
$$

这里有三个动作：

1. $\operatorname{Diag}(\alpha_t)$ 对不同 key 通道施加不同的遗忘率；这比单一标量门更细粒度。
2. $I-\beta_tk_tk_t^\top$ 用 delta rule 擦除旧状态中与当前 key 冲突的预测。
3. $\beta_tk_tv_t^\top$ 写入新的 key-value 关联。

所以 KDA 不是简单做“历史向量求和”，而是在一个有限状态关联记忆中持续执行**忘记—纠错—写入**。训练和 prefill 时，它采用 chunk-wise 算法：chunk 之间递归，chunk 内转换为并行矩阵乘；解码时保留固定形状状态。Kimi K3 又把 log-decay 下界固定到 $g_{\min}=-5$，让 16-token tile 的缩放保持在 BF16 动态范围内，从而让对角 tile 也能走 Tensor Core 矩阵乘。

### 2.4 Sparse MoE：扩大总容量，但只计算少数专家

典型 MoE 层可写为：

$$
y=E_{\text{shared}}(x)+\sum_{i\in\operatorname{TopK}(r(x))}p_iE_i(x).
$$

Router $r(x)$ 为每个 token 选择少数专家。它降低的是**每-token 激活计算**，不是模型总权重；全部专家参数仍要存储并在多卡之间布置。Kimi Linear 为每个 token 从 256 个 routed experts 中选 8 个，并始终经过 1 个 shared expert；Kimi K3 扩展到 896 选 16，并保留 2 个 shared experts。

这解释了为什么“48B total / 3B active”仍然需要装载约 98.25 GB 权重，也解释了为什么 K3 虽然每 token 激活约 104B，完整 checkpoint 仍达到约 1.56 TB。

### 2.5 AttnRes：把注意力从 token 轴扩展到网络深度轴

普通 residual 把所有前层信息累加进单个 $h_l$；随着深度增加，较早信息必须反复穿过这个瓶颈。AttnRes 为第 $l$ 层设置一个可学习 pseudo-query $w_l$，对 embedding 和此前层/块的输出计算权重：

$$
\alpha_{i\rightarrow l}=\operatorname{softmax}_i\left(w_l^\top\operatorname{RMSNorm}(k_i)\right),
\qquad h_l=\sum_{i<l}\alpha_{i\rightarrow l}v_i.
$$

Kimi K3 没有保存全部 93 层输出，而是使用 **12 层一个块的 Block AttnRes**：块内累积，块间选择性注意。这样把需要常驻的深度状态从与层数 $L$ 成正比，降到与块数 $N$ 成正比，同时让后层可以绕过纯顺序残差路径读取较早表示。

## 3. 四套架构逐一拆解

### 3.1 Jamba2 3B：最小型 SSM-Transformer Hybrid

**配置事实**：28 层，hidden size 2560，20 个 query heads、1 个 KV head，256K context。Attention 的 offset 为 7、period 为 14，因此按 1-based 层号为第 8、22 层；其余 26 层是 Mamba。Mamba 状态维度为 16，短卷积核为 4，expand 为 2。

```text
Mamba ×7 → Attention → Mamba ×13 → Attention → Mamba ×6
```

它的设计逻辑是：

- 26 个 Mamba 层承担绝大多数序列压缩与传递，控制长上下文内存。
- 两个 Attention 层像周期性“全局检索检查点”，让表示重新接触未压缩的 token 历史。
- 20Q/1KV 是 multi-query attention，进一步减少这两个全局层的 KV cache。
- 配置虽然保留 Jamba 的 expert 调度字段，但 `num_experts=1`、`num_experts_per_tok=1`，所以这个 3B 版本实际上是 dense FFN，不应把它描述成真正的稀疏 MoE。

**适合场景**：单卡、边缘或资源受限部署；需要比纯小 Transformer 更友好的长上下文内存，但不要求大规模专家容量。

**本地结果**：固定 revision `525c6c8e1d9f5bddedfbdc1dbb0ade2df84230c9`；6,394,271,296 字节权重；1 张 GH200；生成 32 tokens 用时 15.31 秒；峰值 GPU 分配 6,396,346,368 字节。该功能性 smoke 使用 Transformers reference Mamba path（`use_mamba_kernels=false`），证明模型可运行，不代表优化后的 serving 吞吐。

### 3.2 Nemotron Nano 9B v2：把序列混合与通道混合拆成不同层

Nemotron-H 的 56 字符 layer pattern 直接规定每层类型：`M` 为 Mamba-2，`*` 为 Attention，`-` 为 MLP-only。固定配置实际包含：

- 27 个 Mamba-2 层；
- 25 个 MLP-only 层；
- 4 个 Attention 层，按 1-based 层号为 15、22、31、40。

这与“每个 Transformer block 都有 Attention + FFN”不同：Nemotron 把 token mixing 和 channel mixing 分散到独立 residual 层。Mamba 层负责递归序列建模，MLP-only 层增加非线性通道变换，极少数 Attention 层提供全局内容寻址。

关键配置为 hidden size 4480；Mamba-2 使用 128 个 state heads、每头维度 80、state size 128、8 groups、卷积核 4；Attention 使用 40 个 query heads 和 8 个 KV heads；context 为 128K。该 9B checkpoint 是 dense 模型，不使用稀疏专家。

**与 Jamba2 的差异**：Jamba2 是“多数 Mamba block + 两个 Attention block”；Nemotron 则显式把 Mamba、MLP、Attention 三种 layer role 交错排列。后者能单独调节“做多少序列混合”和“做多少通道变换”。

**本地结果**：固定 revision `6533e8de2c68e4536bf7c411d7a3ce5734111476`；17,776,492,512 字节权重；1 张 GH200；生成 24 tokens 用时 45.61 秒；峰值 18,168,224,768 字节。纯 PyTorch fallback 虽然退出码为零，但生成退化的重复 `-1`，因此未被提升为成功结果；最终结果使用针对 CUDA 12.6 源码构建的 `causal-conv1d 1.6.2.post1` 和 `mamba-ssm 2.3.2.post1`，并执行 byte-identical upstream implementation。`use_cache=false` 是当前兼容性选择，不是最终性能配置。

### 3.3 Kimi Linear 48B-A3B：KDA、全局 MLA 与 MoE 的三重 Hybrid

Kimi Linear 有 27 个 attention/mixer 层：20 个 KDA、7 个全局 MLA。层序列基本是 `KDA ×3 → MLA`，最后再以一个 MLA 收尾：

```text
(KDA ×3 → global MLA) ×6 → KDA ×2 → global MLA
```

配置层面的数据流是：

1. **KDA 主干**：32 heads、每头状态 key/value 维度 128、short-conv kernel 4。20/27 的层不需要随 context 增长的完整 KV cache。
2. **周期性 MLA**：第 4、8、12、16、20、24、27 层做全局注意力；KV latent rank 为 512。它保留精确的跨 token 内容寻址，同时压缩全局层缓存。
3. **Sparse MoE**：第一个 block 是 dense，之后每层使用 256 个 routed experts 中的 Top-8，并加入 1 个 shared expert；每个 routed expert 的中间维度为 1024。

其总参数为 48B，但每 token 激活约 3B，context 为 1M。这里的两个“稀疏”概念不能混淆：KDA/MLA 的 3:1 是**层类型稀疏**；256 选 8 是**专家激活稀疏**。前者减少长上下文 token mixing 成本，后者减少通道计算成本。

**本地结果**：固定 revision `e1df551a447157d4658b573f9a695d57658590e9`；98,248,224,120 字节权重；2 张同节点 GH200，无 CPU/disk offload；生成 24 tokens 用时 44.35 秒；两卡峰值分别为 45,583,835,648 和 52,793,666,048 字节。成功运行要求 checkpoint 对应的 `fla-core==0.4.0`（后续版本已把 `g_bias` API 改为 `dt_bias`）、显式补充 `tiktoken==0.13.0`，以及不改变模型数学的 Python 3.13 文档装饰器/backend-selection 兼容视图。

### 3.4 Kimi K3：把 Hybrid 扩展到 token、width、depth 和 vision 四个轴

Kimi K3 的 text backbone 有 93 层、hidden size 7168，序列混合组成是 69 KDA + 24 Gated MLA：

```text
(KDA ×3 → Gated MLA) ×23 → final Gated MLA
```

最后额外放一个 Gated MLA，保证主干输出前必定经过全局注意力。K3 的改动不是单纯“把 Kimi Linear 放大”：

#### KDA 路径

- 96 heads，状态 head dimension 128。
- 把 Kimi Linear 无下界的 negative-Softplus log-decay 改为 $g=-5\cdot\operatorname{sigmoid}(\cdot)$；$\alpha=\exp(g)$ 因而不会小于 $e^{-5}$。
- 使用 input-dependent full-rank output gate；token 可以控制从 KDA 状态读出的每个通道。
- 下界门控让 16-token tile 的缩放留在 BF16 动态范围，减少特殊 position-pair 路径，提升 Tensor Core 可执行性。

#### Gated MLA 路径

- MLA 继续缓存 512 维 KV latent，并通过全局注意力读取任意历史内容。
- K3 的 MLA 使用 NoPE；顺序和近因信息主要由中间 KDA 层提供，全局 MLA 专注内容交互。
- MLA 输出同样增加 full-rank channel gate，让当前 token 决定接受哪些全局通道。

#### Block AttnRes 深度路径

配置的 `attn_res_block_size=12`。K3 将主干划分为约 8 个 12-layer block 和一个部分尾块，并始终把 embedding 作为可读取源。层不再只能接收“前一层累加后的单个状态”，而能对 embedding、前块输出和当前块部分和做选择性组合。它处理的是**跨层信息流**，不是替代 token attention。

#### Stable LatentMoE 宽度路径

- 总参数 2.8T，每 token 激活约 104B。
- 除首个 dense 层外，使用 896 个 routed experts 中的 16 个，另加 2 个 full-width shared experts；路由稀疏度为 56。
- routed experts 在较窄 latent 空间中工作，聚合后经过 RMSNorm 再上投影回 7168 维，降低大专家池的通信与激活尺度压力。
- SiTU-GLU 用有界 tanh soft-cap 抑制低精度下的激活爆炸；Quantile Balancing 用于极大专家池的负载平衡。

#### 原生视觉路径

图像/视频先经过 401M 参数的 MoonViT-V2，再由轻量 projector 映射到 7168 维共享 embedding 空间，与文本 token 一起进入同一主干。固定配置中的视觉编码器有 27 层、12 heads、patch size 14。

#### 量化与部署现实

K3 从后训练阶段使用量化感知训练，发布形态为 MXFP4 weights / MXFP8 activations。但“每 token 只激活 104B”和“MXFP4”都不意味着它能装入少量 GH200：固定 checkpoint 仍为 1,560,936,091,448 字节、96 个 shards。16 张 GH200 的聚合显存仅比原始权重多 71,151,481,032 字节，平均每卡约 4.45 GB 余量，不足以容纳 engine state、激活、通信 buffer、KDA state、MLA KV 与视觉张量。当前 GH200/Hopper 的公开 SGLang 参考形状是 32 GPU，因此本次只完成 source-complete / capacity-gated 交付，没有进行虚假的“已部署”声明。

## 4. 横向比较：四个模型怎样分配计算

| 模型 | 高效序列路径 | 全局路径 | 通道路径 | 深度/模态增强 | 总/激活参数 | 本地状态 |
|---|---|---|---|---|---:|---|
| Jamba2 3B | 26/28 Mamba | 2 Attention，20Q/1KV | Dense FFN | 普通 residual | 3B / 3B | 1 GPU 成功 |
| Nemotron Nano 9B v2 | 27/56 Mamba-2 | 4 Attention；另有 25 MLP-only | Dense MLP | 分离 mixer role | 9B / 9B | 1 GPU 成功 |
| Kimi Linear 48B-A3B | 20/27 KDA | 7 MLA | 256 选 8 + 1 shared | 普通 residual | 48B / 3B | 2 GPU 成功 |
| Kimi K3 | 69/93 KDA | 24 Gated MLA | 896 选 16 + 2 shared | Block AttnRes + MoonViT-V2 | 2.8T / 104B | source-complete；容量门控 |

四者的共同规律是：模型越往右发展，Hybrid 的轴越多。Jamba2 只在 token mixer 上混合；Nemotron 再把 MLP role 独立出来；Kimi Linear 同时在 token mixer 和 experts 上做稀疏；K3 则继续加入 depth routing、latent expert space、原生视觉和低精度训练。

## 5. 为什么这种设计比“全用一种算子”更合理

### 5.1 全用 Attention

优点是精确全局读取；缺点是长上下文的计算和缓存昂贵。对于 256K—1M context，让每层都保留完整 KV 往往是系统瓶颈。

### 5.2 全用递归 / 线性层

优点是状态大小不随历史长度线性增长；缺点是有限状态必然压缩信息，精确检索、复制和远距离绑定更困难。

### 5.3 Hybrid 的折中

高比例递归层维持低成本“工作记忆”，低比例全局层定期访问“原始档案”。MoE 再把参数容量和每-token 计算解耦；AttnRes 把顺序深度传递改成选择性深度读取。换句话说：

> Mamba/KDA 解决“历史太长，不能每层都重读”；Attention/MLA 解决“状态压缩后仍要精确查找”；MoE 解决“想增加知识容量，但不想每 token 计算全部参数”；AttnRes 解决“网络太深，早期表示不应只能逐层传递”。

## 6. 本地部署证据与工程含义

| Target | 固定 revision | 权重字节 | GPU | 生成 | 用时 | 峰值 GPU 分配 |
|---|---|---:|---:|---:|---:|---:|
| Jamba2 3B | `525c6c8e1d9f5bddedfbdc1dbb0ade2df84230c9` | 6,394,271,296 | 1 | 32 tokens | 15.31 s | 6,396,346,368 B |
| Nemotron Nano 9B v2 | `6533e8de2c68e4536bf7c411d7a3ce5734111476` | 17,776,492,512 | 1 | 24 tokens | 45.61 s | 18,168,224,768 B |
| Kimi Linear 48B-A3B | `e1df551a447157d4658b573f9a695d57658590e9` | 98,248,224,120 | 2 | 24 tokens | 44.35 s | 45,583,835,648 B + 52,793,666,048 B |
| Kimi K3 | `9f62e4e9fffbd0a83ddd60e1c209d828994b3569` | 1,560,936,091,448 | 至少需要受支持的多节点形状 | 未启动 | — | 容量门控 |

这些是**功能性 smoke**，证明固定版本的权重、模型实现、依赖和 GPU 路径能够产生非空且语义相关的输出。它们不是共同数据集、共同输入长度、共同 batch size 下的性能 benchmark，所以不能据此把 15.31 秒与 44.35 秒直接当成架构快慢排名。

调度方面，`gtop` 是时间点快照，不是 GPU 锁。一次快照显示空闲的 GPU 在不到两分钟内重新出现其他 PID；启动 guard 因而在创建 CUDA context 前退出。最终三次成功运行都在计算节点上重新检查 owner、allocation state、物理 GPU PID 和显存占用后才限制 `CUDA_VISIBLE_DEVICES` 启动，没有停止或覆盖其他任务。

## 7. 选型建议

### 7.1 首先要低风险单卡落地

选择 **Jamba2 3B**。它最容易验证端到端链路，权重和峰值分配都约 6.4 GB，适合作为 Hybrid runtime、量化、服务接口和长上下文测试的第一基线。

### 7.2 想研究 dense Mamba-2 / Attention 的层角色设计

选择 **Nemotron Nano 9B v2**。它的价值不只是参数更大，而是 27 Mamba + 25 MLP-only + 4 Attention 的显式角色分离。不过部署必须使用正确的 Mamba kernel 与 cache 配置；“程序退出零”不足以证明语义正确。

### 7.3 目标是 1M context、MoE 与可运行的大型 Hybrid

选择 **Kimi Linear 48B-A3B**。它同时覆盖 KDA、周期性 MLA 和稀疏 MoE，而且已在两张 GH200 上无 offload 跑通，是当前最有代表性的可实验架构。后续应优先测长 context 下的 prefill、TPOT、状态/KV 占用和检索质量。

### 7.4 目标是 frontier multimodal / agentic 架构研究

研究 **Kimi K3**，但部署前先取得受支持的 32-GPU Hopper 或等价高显存拓扑，并满足 engine、driver 和 CUDA 版本要求。当前最有价值的工作是用已归档源码做结构审计、并行规划和容量模型，而不是先下载 1.56 TB 权重再等待无法满足的运行条件。

## 8. 下一轮公平评测应该怎么做

若目的是“选哪套架构”，下一轮必须使用共同协议：

1. 共同 tokenizer-aware 输入长度：4K、32K、128K，Kimi 路线再加 1M 档。
2. 共同任务：needle retrieval、multi-hop retrieval、长文摘要、代码仓库问答和短上下文质量集。
3. 共同生成长度、batch size、dtype、warm-up 次数和计时边界。
4. 分开记录 prefill tokens/s、decode tokens/s、TPOT、峰值显存、KV/递归状态大小与跨卡通信。
5. 质量和性能分别判定；退化文本、重复 token 或 silent fallback 必须判失败，即使进程退出码为零。

当前结果回答的是“能否在本地真实执行”，尚未回答“在相同质量约束下谁最快、谁最好”。

## 9. 局限与开放问题

- Jamba2 当前使用 reference Mamba 路径，尚未进行 optimized kernel benchmark。
- Nemotron 的成功 smoke 使用 `use_cache=false`，需单独修复/验证 serving cache 后再测吞吐。
- Kimi Linear 只完成短生成；1M context 声明尚未在本地用长输入验证。
- Kimi K3 尚无本地推理结果，任何质量或吞吐结论都只能来自官方资料，不能与前三个本地 smoke 混为一谈。
- 三个成功结果的模型大小和生成 token 数不同，当前延迟不可直接横比。

## 10. 可审计证据

### 本地结构与结果

- `tasks/hybrid_system_20260811/00_portfolio_research/REPORT.md`
- `tasks/hybrid_system_20260811/01_jamba2_3b_inference/model/config.json`
- `tasks/hybrid_system_20260811/01_jamba2_3b_inference/runs/smoke_20260811T172020Z/result.json`
- `tasks/hybrid_system_20260811/02_nemotron_nano_9b_v2_inference/model/config.json`
- `tasks/hybrid_system_20260811/02_nemotron_nano_9b_v2_inference/runs/smoke_20260811T173651Z/result.json`
- `tasks/hybrid_system_20260811/03_kimi_linear_48b_inference/model/config.json`
- `tasks/hybrid_system_20260811/03_kimi_linear_48b_inference/runs/smoke_20260811T175406Z/result.json`
- `tasks/hybrid_system_20260811/04_kimi_k3_feasibility/source_model_metadata/config.json`
- `tasks/hybrid_system_20260811/04_kimi_k3_feasibility/source_github/k3_tech_report.pdf`
- `tasks/hybrid_system_20260811/04_kimi_k3_feasibility/source_manifest.json`
- `tasks/hybrid_system_20260811/04_kimi_k3_feasibility/recipe_manifest.json`

### 官方来源

1. [AI21 Jamba2 3B model card](https://huggingface.co/ai21labs/AI21-Jamba2-3B)
2. [AI21 Jamba2 release](https://www.ai21.com/blog/introducing-jamba2/)
3. [NVIDIA Nemotron Nano 9B v2 model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2)
4. [NVIDIA Nemotron Nano 2 technical report](https://arxiv.org/abs/2508.14444)
5. [Kimi Linear paper](https://huggingface.co/papers/2510.26692)
6. [Kimi Linear repository](https://github.com/MoonshotAI/Kimi-Linear)
7. [Kimi Linear 48B-A3B Instruct model card](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct)
8. [Kimi K3 repository and technical report](https://github.com/MoonshotAI/Kimi-K3)
9. [Kimi K3 model card](https://huggingface.co/moonshotai/Kimi-K3)

