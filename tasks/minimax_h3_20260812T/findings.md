# MiniMax-H3 架构与训练配方：侦察结果

日期：2026-08-12　　任务目录：`/lus/lfs1aip2/projects/public/u6gb/tasks/minimax_h3_20260812T/`

---

## 0. 三条必须先说清楚的前提修正

用户的原始要求里有两条前提与事实不符，先列出来，因为它们改变了「复刻」的定义。

| 用户前提 | 事实 | 影响 |
|---|---|---|
| 「看看他文章里怎么做的」 | **技术报告至今未发布**。官方博客原文：*"We'll be sharing the full H3 Technical Report soon."*（2026-07-31 发布，至 2026-08-12 仍无 arXiv/PDF） | 没有「文章」可照抄。改为从**已发布的权重、config、以及 diffusers 官方移植代码**反推——这比技术报告更硬，因为它是可执行的 |
| 「他数据集应该也开源了」 | **训练数据未开源**，训练代码、微调配方、LoRA trainer 也都没有。许可证非 OSI 认证，商用带署名与分成条款 | 用 **VGGSound**（199,467 条 10 秒真实带音轨视频 + 309 类标签）作为公开替身，并明确标注这不是 MiniMax 的语料 |
| 「H3 是个大语言模型」（隐含） | H3 是**全模态视频+音频生成模型**，不是 LLM。33B dense DiT + 流匹配 | 复刻对象是扩散 Transformer，不是自回归语言模型 |

还有一条边界：H3 是**三模块系统**，只开源了中间那个。

```
用户输入 ──► H3-Context-IR ──► H3-Base ──► H3-Regenerate-2K ──► 2K 成片
             （闭源托管）      （开源）      （闭源托管）
             官方原话："critical to the quality of the final output"
```

所以即使拿到全部开源权重，也复刻不出官方演示的质量。本任务复刻 **H3-Base**，并自建一个小型 Context-IR 替身。

---

## 1. H3-Omni-Transformer 架构（来自 `transformer/config.json` + diffusers 移植）

```
                          ┌─────────── 一条 packed 序列 ───────────┐
  text (Qwen3-VL L50)  ──►│ tag=1                                  │
  首/尾关键帧 latent    ──►│ tag=0  (t = max(t_video, 0.999))       │──► 50 × Block ──┬─► proj_out      → 视频速度
  音频 latent (立体声)  ──►│ tag=2  (t = t_audio,  shift=3)         │                 └─► audio_proj_out → 音频速度
  目标视频 latent      ──►│ tag=0  (t = t_video,  shift=12)        │
                          └────────────────────────────────────────┘
                            full self-attention，**无 cross-attention**
```

| 参数 | 值 | 备注 |
|---|---|---|
| `hidden_size` | 5376 | 残差流宽度 |
| `num_layers` | 50 | |
| `num_attention_heads` × `attention_head_dim` | 56 × 128 = **7168** | **比 hidden 宽 4/3 倍**，刻意的容量分配 |
| `ffn_dim` | 14336 | = 8/3 × hidden，SwiGLU，无 bias |
| `time_embed_dim` | 2688 | = hidden/2，所有 AdaLN 的输入 |
| `adaln_out_features` | **96768** | = 6 × 5376 × **3 模态** |
| `in_channels` / `audio_in_channels` | 24 / 32 | 两个 VAE 的 latent 通道 |
| `patch_size` | (1,2,2) | patchify 后有效空间下采样 32× |
| `text_dim` | 5120 | Qwen3-VL-32B 宽度 |
| `rope_freq_dim` | 16 | 3 轴共用，旋转 2×3×16=96 / 128 通道（75%） |
| `num_refiner_layers` | 2 | 只作用于文本流，无 AdaLN 无 RoPE |

### 1.1 「13B 在 AdaLN」的算术验证

官方称 33B 中约 13B 在 AdaLN 分支。这可以纯算术验证，不必相信：

```
AdaLN_params = num_layers × (time_embed_dim × adaln_out_features)
             = 50 × (2688 × 96768)
             = 50 × 260,112,384
             = 13.01 B                              ← 官方「~13B」✓

Backbone_per_layer = attn + ffn
                   = [3×(5376×7168) + 7168×5376] + [5376×28672 + 14336×5376]
                   = 154.1M + 231.2M
                   = 385.3M
Total = 50 × (385.3M + 260.1M) + embed/head/refiner
      = 32.27 B + 0.2 B
      ≈ 32.5 B                                      ← 官方「33B」✓
```

分片大小也对得上：transformer 13 片 × 5.165 GB ≈ 66 GB，bf16 每参 2 字节 → 33 B。

### 1.2 设计上最关键的一行代码

```python
adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags   # 3 = video/text/audio
```

这一行是整个架构的枢纽。它让 **同一次 forward 里不同的行处于不同的噪声水平**：

- 目标视频行 → `t_video`（shift=12 的调度）
- 音频行 → `t_audio`（shift=3 的调度，**独立**）
- 首尾关键帧条件行 → `max(t_video, 0.999)`
- 文本行 → **继承 `t_video`**（先 `torch.full(seq, video_t)` 再覆盖其余）

因此文本表示**不是静态条件**，它的 AdaLN 调制随去噪进度变化。这点极易在复刻时写错。

### 1.3 模态差异只存在于三处

官方 docstring 原文：*"Modality-specific behaviour comes only from the two input patch projections, the per-row AdaLN modality tag, and the two output heads."*

注意力和 FFN 的权重**三个模态完全共享**。这比 SD3 的 MMDiT（文本/图像各一套完整权重）省得多，代价是靠 AdaLN 的 3 倍参数量来做模态区分——13B 就花在这里。

---

## 2. 音视频同步机制：共用一个 1/40 秒的旋转时钟

这是全篇最精妙的设计，且**不需要任何显式对齐损失**。

```
_ROPE_FRAME_RESCALE   = 5/3
_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)      # 和为 17
```

推导：

```
音频 latent 速率 = 40 Hz  →  1 个位置单位 = 1/40 秒
视频 24 fps，因果 VAE：
    第 1 个 latent 帧覆盖 1 原始帧 = 1/24 秒 = 40/24 = 5/3 单位   = 1 × 5/3 ✓
    其余 latent 帧覆盖 4 原始帧   = 4/24 秒 = 40/6  = 6.667 单位 = 4 × 5/3 ✓
所以 5/3 = 40/24，正是「音频 latent 速率 ÷ 视频帧率」
```

两个模态因此落在**同一根时间轴**上，RoPE 自己就把口型和声音对齐了。

其余布局细节：

| 行类型 | t 坐标 | h 坐标 | w 坐标 |
|---|---|---|---|
| 文本 | `0 .. T-1` | 0 | 0 |
| 关键帧条件 | 锚点时刻 | frame_grid | frame_grid |
| 音频（声道优先） | `T + arange(L)` 重复 2 次 | **0（无高度）** | 左声道钉 `width_grid[0]`，右声道钉 `width_grid[-1]` |
| 目标视频 | 非均匀 `5/3×(1,4,4,4,4)` 累积 | frame_grid | frame_grid |

- 空间网格按 `sqrt(latent_area)` 做**长宽比归一化**后 ×32，右端点排除。
- **文本长度会平移整个媒体时钟**（媒体行从 `t=T` 开始），这是真实副作用不是 bug。
- 立体声用「空间位置的两个端点」编码，省掉了额外的 channel embedding。

---

## 3. 流匹配：符号是反的

`MiniMaxH3Scheduler` 的三处与 diffusers 标准流匹配不兼容（所以它是独立类）：

1. **速度朝向数据**：`x0 = x_t + σ·v`（标准是 `x0 = x_t − σ·v`）
2. **`t = 1 − σ`，且 `t = 1` 表示干净**（标准调度器暴露 `σ×1000`，方向相反）
3. σ 网格从 `linspace(1, 0, N)` 起，终点 0 计入步数，移位后用 `unique_consecutive` 去重

训练目标由此**唯一确定**：

```
前向    x_t = t·x_0 + (1−t)·ε
定义    x_0 = x_t + (1−t)·v
解 v    v  = (x_0 − x_t)/(1−t)
          = (x_0 − t·x_0 − (1−t)·ε)/(1−t)
          = ((1−t)·x_0 − (1−t)·ε)/(1−t)
          = x_0 − ε                      ← 与 t 无关
```

时间步移位：`σ' = s·σ / (1 + (s−1)·σ)`，**视频 s=12，音频 s=3**。两个模态用两个 scheduler 实例，在一次 transformer 调用内各走各的。

---

## 4. 两个 VAE

### H3-VisualVAE（`f16t4d24`，10.4 GB）
```
spatial_downsample_factors  [2,2,2,2,1,1]  →  16×
temporal_downsample_factors [1,2,2,1,1,1]  →  4×
latent_channels 24 ;  clip_length 17 ;  token_drop 3
编码器 = CNN，解码器 = 36 层 ViT（dim 2048，4 个 register token）  ← 非对称，所以体积大
```

**帧数几何**（源码证实）：`n_chunks = ceil(F/17)`，每块 `ceil(17/4)=5` 个 latent，末尾丢 `token_drop=3` 个：

```
F = 17n + 5  →  n_chunks = n+1  →  latents = 5(n+1) − 3 = 5n + 2
例：F=73（=17×4+5）→ 22 个 latent 帧。反查：4×17 + (1+4) = 73 ✓
```
这与旋转时钟的 `(1,4,4,4,4)` 分组是同一事实的两个视图。

**像素约定**：ImageNet 归一化，基准区间 `[0,1]`，**不是常见的 `[-1,1]`**。

### H3-AudioVAE（0.6 GB）
```
encoder_rates [2,4,4,5,5] → 乘积 800 ;  32000/800 = 40 Hz ✓
latent_channels 32 ;  立体声两声道作为 batch 独立编码
```

### latent 归一化（由 pipeline 做，不在 VAE 内）
`(z − latents_mean) / latents_std`，逐通道。视频 24 个常数，音频 32 个。

### 采样约定（易错）
| | 视频 | 音频 |
|---|---|---|
| 后验 | `posterior.sample()` | `posterior.mode()`，**从不采样** |
| 条件路径种子 | 固定 42 | — |
| 精度 | 采样后**舍入到 fp16**（约 11 bit）再归一化 | — |

### 归一化后音频 latent 的尺度是 0.48，不是 1（实测，且与参考一致）

在 2000 个真实 clip 上实测归一化后的 latent：

```
video  mean +0.0594  std 1.0508   per-channel mean [-0.397,+0.652]  std [0.854,1.302]   ≈ 0/1 ✓
audio  mean +0.0133  std 0.4827   per-channel std  [0.436, 0.517]                       ≈ 0.48
```

音频差了约一半，且**各通道高度一致**（0.436–0.517），是固定因子而非散乱。

推断：音频取 `posterior.mode()`（分布均值），而 config 的 `latents_std` 应是在
**采样**的 latent 上统计的。`z ~ N(μ,σ)` 时 `Var(z) = Var(μ) + E[σ²]`，只取 μ
丢掉了 `E[σ²]`，除以基于采样的 std 就得到 <1。

**参考实现做的完全一样**（`posterior.mode()` → 除以 `latents_std`），所以这是 H3
自己的约定，不是复刻误差。

**对训练的影响**：流匹配加的是单位尺度的 `randn`，而音频信号只有 0.48 尺度，
所以同一个 `t` 下音频的信噪比约为视频的一半。这可能是**音频 shift=3 而视频
shift=12** 的原因之一（音频调度把更多步数放在低噪声端）——这是观察，不是定论。

**这个量只有专门查才看得到**：E6 往返闸门 encode→decode 中间不做归一化，
对归一化代码完全无感；归一化只出现在语料构建里。

---

## 5. 文本条件：不是 caption，是结构化文档

官方示例 prompt 长 **5,650 – 33,323 tokens**，固定分节：

```
integrated_multimodal_description: [Shot 1] ... [Shot 2] 在 00:04.500 ...
overall_soundscape:  （画内音，逐事件描述）
non_diegetic_music:  （画外配乐）
```
Ref2VA 另有 `subject_definitions:` / `retention_analysis:` / `summary:`。

编码方式：
- **verbatim tokenize，无 chat template，无特殊 token**
- 取 Qwen3-VL-32B 的 `hidden_states[50]`（**不是最后一层**，最后一层是 post-norm，不是训练时用的条件）
- 输出 `(1, N, 5120)`

---

## 6. 已文档化的训练事实（可作为复刻靶子）

这些不是猜测，是官方 model card / diffusers 文档的直接陈述：

| 事实 | 出处原文 |
|---|---|
| **发布权重是 guidance-distilled** | *"guidance is baked into the weights, so there is no guider, no `negative_prompt` and no `guidance_scale`, and every step runs exactly one forward pass"* |
| **训练时锚点被轻微加噪** | *"The released model was trained with its anchors very slightly noised, so conditioning on exactly `t = 1.0` is off-distribution"*（`keyframe_noise_aug = 0.999`） |
| 训练画布 | `1344 × 768`（`canvas_short_edge=768`, `canvas_max_pixels=1032192`） |
| 生成约束 | 24 fps，5–15 秒，`num_frames = 17n+5`，长宽为 32 的倍数 |
| FL2VA / Ref2VA 是两个后训练分支 | 共享除 transformer 外的全部组件 |
| 音频 VAE 受 VA-VAE 启发优化 latent 空间 | *"Inspired by VA-VAE, we optimize the latent space..."* |

---

## 6.5 「2K 不是超分做的」：一个被描述了机制但没开源模块的部分

官方博客对 H3-Regenerate-2K 的说法值得单独记一笔，因为它描述的是**机制**而不只是产品：

> *"Instead of using a conventional dedicated super-resolution module, the H3 base
> model regenerates its own low-resolution output in-context, which lets it draw on
> the original multimodal context again to produce high-resolution output,
> recovering details that traditional super-resolution can only 'guess' at."*

翻译成架构语言：**把自己刚生成的低分辨率视频，当作一条 reference 塞回同一条 packed
序列，连同原始文本条件一起，在高分辨率画布上重新生成一遍。** 这正是 `ref2va`
工作流的能力（diffusers 文档里有「把一次 t2va 的生成结果直接喂回去当
`MiniMaxH3VideoReference`」的完整示例），所以它不需要任何新结构——超分被表达成了
「同一个模型的第二次条件生成」。

配套的 `H3-VAE` 那条 *"4x gain in effective sequence length"* 是让这件事在经济上
成立的前提：没有那 4 倍压缩，2K 的序列长度会让第二次生成不可承受。

**本复刻不做这一步**，理由不是机制不清楚，而是：(a) `H3-Regenerate-2K` 模块本身
闭源，无从对照；(b) 在 256px 上「重生成到 512px」会让序列从 1748 涨到约 5000，
注意力成本约 8 倍；(c) 一个 101.7 M 模型在 256px 上的输出质量还不足以让「重生成
恢复细节」这个论断有意义——那是对已经很好的低分辨率结果才成立的。这属于
**理解了但按预算不做**，不是没看懂。

---

## 7. 复刻边界（诚实清单）

| 能复刻 | 不能复刻 | 原因 |
|---|---|---|
| 架构（逐参数一致，只是变小） | 33B 规模 | 算力 |
| packed 序列布局 | H3-Context-IR | 闭源托管 |
| 双 shift 流匹配训练 | 原始训练语料 | 未开源 |
| 关键帧锚点 + 0.999 加噪 | H3-Regenerate-2K | 闭源托管 |
| CFG 蒸馏（单次前向） | 官方超参（LR/batch/步数） | 无技术报告 |
| 用**同一套冻结 VAE + 同一个 Qwen3-VL** 编码 | 原生稀疏注意力 | 未包含在开源推理实现中 |
