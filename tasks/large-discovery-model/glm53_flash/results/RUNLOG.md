# GLM-5.3-Flash 下载与部署记录(2026-08-26)

目标:下载 GLM-5.3-Flash 并在本集群部署成可调用的推理服务(用户令「下载和部署」)。

## 模型速览(发布当天,2026-08-26)

| 项 | 值 |
|---|---|
| 仓库 | HF `zai-org/GLM-5.3-Flash` / ModelScope `ZhipuAI/GLM-5.3-Flash`(MIT) |
| 规模 | 320B 总参 / 18B 激活,45 层,288 专家选 8 |
| 结构 | KDA 线性注意力 + NoPE 稀疏 MLA(kv_lora_rank=512),MTP 1 层,原生多模态(文/图/视频) |
| 权重 | FP8,~306 GiB,62 个 safetensors 分片(共 73 个文件) |
| 上下文 | 1,048,576 token(部署先限 64k) |
| 官方部署 | vLLM ≥0.27,TP4 + `--kv-cache-dtype fp8` + MTP 投机解码;flashinfer ≥0.6.17 |

## 部署方案

- **栈**:vLLM 0.28.0 —— PyPI 有 `cp38-abi3-manylinux_aarch64` 官方轮子,GH200 上零编译
  (与 dsv4 训练栈当时手工编 FlashMLA/TE 成对照,推理栈 ARM 生态成熟)。
- **venv**:`/home/u6gb/kangli.u6gb/envs/glm53-vllm`(HOME 侧;u6gb 项目 inode 仅剩 ~44 万,
  venv 几万文件不能落项目配额)。
- **显存账**:306/4 = 76.5G 权重/卡,util 0.92 ⇒ ~11G/卡 给 KV。KDA 层状态常数大小、
  MLA 层 512 维压缩,64k 上下文可行;不够再降 MAXLEN。
- **落点**:attach 到占位分配 6141106 的空节点 nid010815(4 卡整占,起名 glm53-vllm),
  不排队。第一轮不带 MTP 投机解码,点亮后再加。

## 时间线

| 时刻 | 事件 | 结果 |
|---|---|---|
| 23:42 | 调研:vLLM 官方部署页 + HF config;PyPI 0.28.0 aarch64 轮子在;ModelScope 已上架 | ✅ |
| 23:45 | 下载启动(tmux glm-dl,ModelScope 73 文件 → `models/GLM-5.3-Flash`);venv 构建启动(tmux glm-venv) | 🔄 |
| 23:55 | venv v1 完成(vllm 0.28.0/torch 2.13.0/transformers 5.16.1/flashinfer 0.6.16.post3),但节点 sanity 失败:**PyPI 默认轮子是 cu130 构建,节点驱动 565.57 只到 CUDA 12.7**,major 跨代 ⇒ `cuda_available=False` | ❌ |
| 00:0x | 修正:GitHub release 有 **`vllm-0.28.0+cu129` aarch64 变体轮子**,pytorch cu129 索引有配套 torch 2.13.0+cu129(cp312 aarch64)。CUDA 12.x 内小版本前向兼容,dsv4 的 cu128 栈同驱动已验证。旧 venv 挪 `_cu130_deprecated_*`,v2 重建(tmux glm-venv2) | 🔄 |

| 00:2x | venv v2 完成(**vllm 0.28.0+cu129 / torch 2.13.0+cu129**,nvidia 库全线 cu12 家族);nid010413 上 GPU sanity 全绿:cuda_avail=True、sm90、matmul 通、vllm 导入 | ✅ |
| 00:2x | 起服双路预案入库:attach 优先(逐卡验空),`launch_serve.sbatch` 后备(启动先探 `results/server_host`,已有健康服务即退,防双跑);当刻 8 节点全被另一条线评测波占满,等下载完成时再探 | ✅ |

| 00:38 | **下载完成:62/62 分片,306G,一轮无重试**(全程 ~50 分钟) | ✅ |
| 00:40 | 8 节点仍被评测波占满 ⇒ 双路启动:后备 sbatch **6150683**(1N,12h,防双跑检查)已入队;监视「空节点或 sbatch 起跑」先到先得 | 🔄 |

| 01:0x | nid010252 空出(heavy 任务退场),attach 起服;但 serve_vllm.sh 的编译缓存目录 `/local/scratch` 计算节点**不可写**,改 `${TMPDIR:-/tmp}`(沿 dsv4)重启 | ✅ 修正 |
| 01:1x | **重启操作失误丢掉节点**:打断第一次 srun 的 C-c 两下间隔 >1s(srun 要求 1s 内连按才放弃),后续连发把**已带修正的第二次启动**的 srun 客户端也打掉;vLLM 还在 CPU 侧启动、未占显存,另一条线的 bench 调度器看到"空节点"整占 4 卡(XLA 预占 88G/卡)。**两次误判**:88G≈util 预算线让我把对面进程当成自己的权重装载;查 cmdline 才定性。根因是我 C-c 处置毛糙,非对面抢跑 | ❌ 教训 |
| 01:2x | 重挂「空节点 or sbatch 6150683 起跑」监视,等下一个窗口;下轮 attach 起服后前 3 分钟盯 compute-apps,外来进程落卡即早退重试 | 🔄 |

| 08-27 10:32 | 后备 sbatch 6150683 抢到节点起服,**5 分 25 秒失败**——日志给出决定性死因:vllm 0.28.0 **没有 Glm5Next 原生支持**,静默退回 transformers 兜底后端,在 KDA 参数 `k_conv1d` 上 ValueError | ❌ 定性 |
| (间歇) | 用户令转向:**用 GLM-5.3-Flash 做推理给 LDM(Large-Discovery-Models)准备训练数据**;dsv4 线随权重删除终结 | 🎯 新目标 |
| 16:2x | 支持矩阵查清:vLLM main/SGLang 均无 glm5_next;唯一 day-0 是 transformers 5.16.1(本地 AutoConfig 解析✅);实现在**未合并 PR#53906**(93 文件,+14k 行,含 CUDA 内核改动 ⇒ 纯 Python 叠加不可行) | ✅ |
| 16:39 | 决策:**源码构建 PR 分支** `ZJY0516/vllm@glm-release`(36bb3795)于 nid011131(6197253,23h 窗口);工具链沿 dsv4(gcc-12 + conda cuda1281 12.8) | 🔄 |
| 16:4x | 构建三连败三连修:①缺 setuptools-rust(--no-build-isolation 需预装 build-system.requires 全量)②tail -30 吃掉报错 ③conda CUDA 顶层 include/ 无头文件(FindCUDA 不认 targets/ 布局)→ 符号链接补齐,第 4 轮在跑 | 🔄 |
| 16:5x | **LDM 侧并行打通**:仓库 clone(b5dab16a)、uv 环境(ldm-venv)、mock campaign 跑通、**采集发射 4 行 ldm-2.0 IR + ldm_sft.jsonl**✅;机制结论:发射在任务侧,runner 会 pushd 进任务目录 ⇒ `LDM_DATA_COLLECTION_DIR` 必须绝对路径 | ✅ |
| 16:5x | causal_discovery 任务预算 `llm_requests: 0` ⇒ **它的 campaign 不调 LLM**;GLM 在数据管线中的角色:(a) LLM-proposal 类任务(nanogpt 等)的 campaign 本体 (b) 全部 IR 的 reasoning 增强(augment);real 模式需 `--upstream-root`(MLS-Bench 钉版) | ✅ 定性 |

注:flashinfer 解析为 0.6.16.post3(vllm 0.28.0 钉版),低于部署页写的 0.6.17;
第一轮不带 MTP 先点亮,若 KDA/稀疏 MLA 核函数报错再单独升 flashinfer。
另:nid010815 已被另一条线的评测重新占用(每卡 20.5G)——起服时 attach_serve.sh
会逐卡验空,到时挑当刻真空的节点(6141106 或 6136391 皆可,后者剩 9h+ 也够)。
