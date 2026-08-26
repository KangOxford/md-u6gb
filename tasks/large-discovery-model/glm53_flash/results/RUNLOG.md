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
