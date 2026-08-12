# MiniMax-H3 复刻计划

任务根目录：`/lus/lfs1aip2/projects/public/u6gb/tasks/minimax_h3_20260812T/`
环境：`venv/`（继承宿主 torch 2.8.0+cu129，装 diffusers 0.40.0.dev0 + transformers 4.57.6）

---

## 总体思路

**架构不重写，训练配方才是要造的东西。**

diffusers 里的 `MiniMaxH3Transformer3DModel` 就是参考实现本身，且它的每一个维度都是构造参数。所以 H3-nano = **同一个类，小 config**，架构保真是构造性的，不存在「实现得不太一样」的风险。同理，packed 序列布局用参考实现的 `build_packed_sequence`（纯 `@staticmethod`），训练与推理不可能漂移。

真正要造的是 MiniMax 从未发布的部分：训练时的序列组装、双 shift 流匹配损失、三阶段配方、以及数据管线。

---

## H3-nano 缩放表：保持全部比例，只改宽度和深度

| 量 | H3 (33B) | H3-nano | H3-micro | H3-small | 保持的比例 |
|---|---|---|---|---|---|
| `hidden_size` | 5376 | **576** | 384 | 768 | — |
| attn 内维 / hidden | 7168/5376 | 768/576 | 512/384 | 1024/768 | **4/3** |
| `ffn_dim` / hidden | 14336/5376 | 1536/576 | 1024/384 | 2048/768 | **8/3** |
| `time_embed_dim` / hidden | 2688/5376 | 288/576 | 192/384 | 384/768 | **1/2** |
| 旋转通道 / head_dim | 96/128 | 48/64 | 48/64 | 48/64 | **3/4** |
| `num_layers` | 50 | 12 | 8 | 16 | 深度是唯一自由旋钮 |
| `in_channels` / `audio_in_channels` | 24 / 32 | 24 / 32 | 同 | 同 | **完全相同**（真 VAE） |
| `text_dim` | 5120 | 5120 | 同 | 同 | **完全相同**（真 Qwen3-VL-32B） |
| 参数量 | 32.5 B | **101.7 M** | 32.7 M | ~250 M | — |

**为什么 `in_channels` / `text_dim` 不缩**：这样 H3-nano 与 33B 模型在**同一个 latent 空间**去噪、被**同一份文本特征**条件化。行为差异就只能归因于 transformer 本身，而不是 tokenizer 或 conditioner。这是让比较有意义的前提。

nano 的参数分布（实测）：AdaLN 35.4%、FFN 36.6%、注意力 24.4%，与 33B 的 39.4% / 35.0% / 23.3% 同构。

---

## 六个阶段

```
[1] 取资产 + 跑通官方 checkpoint          job 5998320  ← 已提交，排队中
     ├─ 下载 143 GB（transformer 66 + text_encoder 66 + vae 10.4 + audio_vae 0.6）
     ├─ 下载 VGGSound 分片 00,01（约 34 GB，不解包）
     ├─ 参数普查：验证「13B 在 AdaLN」
     └─ T2VA 生成 960×544 / 124 帧，导出带音轨 mp4 + packed 布局实测

[2] 数据管线（用 H3 自己的冻结编码器）
     ├─ 文本库：309 类 → Context-IR 模板 → Qwen3-VL L50 → (310, 96, 5120)
     │           第 310 行是 null prompt（CFG 用）
     └─ 媒体：流式读 tarball → 解码 73 帧@256px + 32 kHz 立体声
                → VisualVAE.sample() + AudioVAE.mode() → 归一化 → latent 分片

[3] 预训练  T2VA，从随机初始化
     ├─ 流匹配 v_target = x0 − ε
     ├─ 视频 shift=12 / 音频 shift=3，两个时间步独立采样
     └─ CFG dropout 10%（为阶段 5 的蒸馏准备一个有引导的教师）

[4] 后训练 A：SFT → FL2VA
     └─ 首帧（可选尾帧）锚点作为前置视频行，t 钉在 0.999

[5] 后训练 B：CFG 蒸馏
     └─ 学生一次前向匹配教师的 v_unc + w(v_cond − v_unc)
        目标：复刻「guidance baked into the weights，每步一次前向」

[6] 评估：把可验证的官方陈述逐条测掉
```

---

## 阶段 6 的评估项：每一项都对应一条官方陈述

| # | 要验证的陈述 | 度量 | 判据 |
|---|---|---|---|
| E1 | 音视频**联合**生成且同步 | 音频包络 vs 帧间差分能量的互相关峰值滞后 | 峰值应在 0 附近；与打乱配对的对照相比显著 |
| E2 | 「guidance baked in，每步一次前向」 | 蒸馏前后的前向次数与墙钟 | 前向数 2(N−1) → (N−1)，墙钟约减半 |
| E3 | 蒸馏不该毁掉质量 | 留出集流匹配损失 + 样本对比 | 蒸馏后损失与教师 CFG 输出的差距 |
| E4 | 「锚点在 0.999 训练，t=1.0 是分布外」 | FL2VA 模型在 t∈{1.0, 0.999, 0.99, 0.9} 条件下的重建误差 | 0.999 应优于 1.0——这是对官方训练事实的直接复现 |
| E5 | 双 shift（12/3）是必要的 | 消融：(12,3) vs (1,1) vs (3,12) 三组同预算训练 | (12,3) 应最优；若不然说明该选择另有依据 |
| E6 | VAE 往返保真（管线正确性闸门） | encode→decode 的 PSNR / 音频 SI-SDR | 低于阈值说明预处理写错了，训练前必须拦下 |
| E7 | 缩放行为 | micro / nano / small 三点的留出损失 | 应单调改善；这是「调小一点」这个动作本身的代价量化 |

---

## 计算预算

节点闸门上限 20，现有 16 节点在跑其他实验（hybrid/BPE），H3 这条线按 **1–2 节点**规划。

| 阶段 | 节点×时长 | 说明 |
|---|---|---|
| [1] 取资产+冒烟 | 1 × ~3 h | 143 GB 下载是主要耗时 |
| [2] 预处理 | 1 × ~3 h | 文本 309 次前向（分钟级）+ 约 2 万 clip 的 VAE 编码 |
| [3] 预训练 | 1 × ~4 h | 序列 1748，nano 101.7 M，4×GH200 DDP |
| [4][5] 后训练 | 1 × ~3 h | 从预训练 checkpoint 继续 |
| [6] 评估 | 1 × ~1 h | |

序列长度推算（256px / 73 帧 / 96 文本 token）：

```
video_rows = latent_frames × (16/2) × (16/2) = 22 × 64 = 1408
audio_rows = 2 声道 × 122 latent            =        244
text_rows  =                                          96
                                            ────────────
sequence_length                                     1748
```

---

## 硬性工程约束（本集群）

- checkpoint 写 `$TMPDIR`，作业结束 `rsync` 回 Lustre
- resume 只读原子写入的 `latest_checkpoint.json` 面包屑，**绝不 ls checkpoint 目录**
- VGGSound tarball **不解包**（每片约 1 万个 mp4，摊到 Lustre 就是每轮 1 万次 MDT open）
- 每次 sbatch 前跑 squeue 去重检查
- W&B `online`，`WANDB_DIR=$TMPDIR`

---

## 当前状态

- [x] 侦察：架构、调度器、packing、VAE、文本条件全部反推完毕 → `findings.md`
- [x] 环境：venv + diffusers main + transformers 4.57.6 + PyAV
- [x] `code/h3nano.py`：模型 config、几何、流匹配、packed 布局 —— CPU 自检通过（101.652 M）
- [x] `code/train.py`：三阶段训练 —— CPU 干跑通过；checkpoint 保存/剪枝/面包屑/resume 单独验证通过
- [x] `code/preprocess_vggsound.py`、`code/sample.py` —— 语法与接口就绪
- [ ] job 5998320 排队中（Priority=1，账户 fair-share 被 16 个在跑节点摊薄）
- [ ] 阶段 2–6
