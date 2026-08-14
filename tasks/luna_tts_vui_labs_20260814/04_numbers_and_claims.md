# 04 · 数字核对与话术拆解

> 这是本目录的核心文件。如果只读一个，读这个。
>
> 做法：把公众号的每一句强断言，拿去和**论文自己的表格**对。
> 论文（arXiv:2608.11593）在方法论上相当克制，很多 caveat 是论文自己写的；
> 失真主要发生在**论文 → 公众号**这一段，而不是论文本身。

---

## 一句话结论

**Luna-TTS 是一份扎实的工作**（0.6B 打赢 1.5B–2B 的对手，masked diffusion 路线在
生产级 TTS 上第一次跑通），**但公众号的"六项全球第一 / 碾压谷歌"是两处独立的拼接**：

1. 六项第一里，**质量四项和效率两项来自两个不同的模型**，且两者的取舍是论文明写的
2. "碾压谷歌"对应的那次评测，**论文自己的原话是"不能确立统计显著差异"**

---

## 一、"六项第一"是怎么拼出来的

公众号原文（第 70 行）：

> 总结来说，就是语音质量最好、生成效率最高、首包延迟最低。

Luna-TTS Family 有**两个变体**，共享 tokenizer / 数据 / 0.6B 骨干，只在生成范式上不同。
把论文 Table 5 和 Table 8 的数字并到一张表上，问题立刻可见：

| | 生成方式 | zh CER↓ | zh SIM↑ | en WER↓ | en SIM↑ | 首块延迟 | 端到端 RTF↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| **Luna-TTS** | 全网格并行，32 步 | **0.73** 🥇 | **79.7** 🥇 | **1.49** 🥇 | **76.8** 🥇 | 419.6 ms | 0.0410 |
| **Luna-TTS** | 全网格并行，16 步 | （同上，评测用 32 步） | | | | 216.0 ms | **0.0211** 🥇 |
| **Luna-TTS Realtime** | 块因果流式，8 步 / 2×H20 | 1.08 | 76.9 | 1.81 | 73.4 | **41.6 ms** 🥇 | 0.0240 |
| MiniMax-Speech（对照） | AR | 0.83 | 78.3 | 1.65 | 69.2 | — | — |
| Qwen-Audio-3.0-TTS（对照） | AR | 0.84 | 79.2 | 1.54 | 76.2 | — | — |
| OmniVoice 0.6B（对照） | masked diffusion | 0.84 | 77.7 | 1.60 | 74.1 | — | 0.0319 |

三件事同时成立，而它们互相矛盾：

```
质量第一   → Luna-TTS（非流式）      它的"首包"= 整段生成完，216–420 ms，没有流式概念
延迟第一   → Luna-TTS Realtime       它的 zh CER 1.08，输给 MiniMax(0.83)/Qwen(0.84)/OmniVoice(0.84)
RTF 第一   → Luna-TTS 16 步 0.0211   比 Realtime 的 0.0240 还低，但同样不能流式
```

**没有任何单个模型同时拿到这六项。** 这不是论文在骗人，论文 Table 1 专门开了一节
"Design Trade-offs at a Glance"，第一句就是 "We regard neither variant as dominant"
（我们不认为哪个变体占优）。是公众号把两列成绩并成了一行。

论文对这个取舍还给了量化：Realtime 相对 Luna-TTS "trading roughly 0.3 CER/WER and
3 SIM points"（换来 41.6 ms 首块延迟）。这是个诚实且合理的工程折衷，只是不该被表述成
"又快又好都是第一"。

### 为什么全并行版本反而"首包"更慢

这点反直觉，值得单独说。Luna-TTS 一次性把整个 $T\times Q$ 网格去噪出来，
**在最后一步完成之前，一个字节的音频都拿不到**，所以它的"首块延迟"恒等于"完整响应"
（表里 419.6 = 419.6，216.0 = 216.0，两列数字一模一样，这就是标志）。

Realtime 把时间轴切成 1.28 s 的块，第一块一算完就能送出去播，后面的块边播边算。
所以它首包 41.6 ms，但完整 10.6 秒音频要 254.0 ms（比 Luna-TTS 16 步的 216.0 ms 慢）。

```
Luna-TTS        [██████████████████████] ──► 全部音频       首包 = 216 ms（= 全部）
                 ↑ 中途没有任何输出

Realtime        [███]►[███]►[███]►[███]►     首包 41.6 ms，随后每块 ~53 ms 续上
                  ↑ 第一块就能播
```

---

## 二、"碾压谷歌"

公众号标题：**《全球第一，碾压谷歌！……》**，正文第 14 行：

> 在独立AI评测平台Artificial Analysis的Speech Arena榜单上，它直接超越了谷歌，获得全球第三！

对照三处证据：

**(a) Seed-TTS-Eval 表里根本没有谷歌模型。** 13 个对照系统是 Seed-TTS、MaskGCT、
F5-TTS、CosyVoice 3、MiniMax-Speech、GLM-TTS、Qwen3-TTS、Qwen-Audio-3.0-TTS、
MOSS-TTS、VoxCPM2、OmniVoice。那"四项第一"和谷歌无关。

**(b) 论文里唯一和谷歌正面比的地方，结论是"没有统计显著差异"。**
论文 §6.3 的内部 English TTS Arena（Table 14）：

| 排名 | 系统 | Elo | 90% bootstrap 区间 | 胜–负 | 胜率 |
|---:|---|---:|---|---:|---:|
| 1 | **Luna-TTS** | 1548.47 | 1531.98 – 1566.59 | 495–368 | 57.36% |
| 2 | Gemini 3.1 Flash TTS | 1546.13 | 1517.81 – 1572.46 | 221–196 | 53.00% |
| 3 | Inworld TTS 2 | 1495.67 | 1474.79 – 1519.24 | 283–317 | 47.17% |
| 4 | ElevenLabs Eleven v3 | 1484.37 | 1436.10 – 1529.14 | 50–69 | 42.02% |
| 5 | StepAudio 2.5 TTS | 1480.39 | 1458.27 – 1506.09 | 229–298 | 43.45% |
| 6 | MiniMax Speech 2.8 HD | 1445.53 | 1391.82 – 1495.23 | 41–71 | 36.61% |

Luna 领先 Gemini **2.34 分**，而两者的 bootstrap 区间几乎完全重叠。
论文自己的原话：

> "Luna-TTS has the highest nominal ranking in this Arena evaluation, **although the
> available evidence does not establish a statistically significant difference between
> Luna-TTS and Gemini 3.1 Flash TTS**."

头对头（Table 15）更直白：**vs Gemini 117–113，胜率 50.87%**，就是掷硬币。

**(c) 这个 Arena 是公司自建的，而且对照设置不对等。** 论文写明是
"**Our** internal English TTS Arena"，且 Luna 用的三个音色是
"obtained by **further fine-tuning** Luna-TTS on internal recordings"（针对性微调过的专属音色），
对手用的是"three voices **sampled at random** from the preset voices officially
recommended on its website"（官网预设里随机抽的）。论文解释这是为了控制音色库规模，
理由成立，但性质上仍然是"我方特训 vs 对方随机抽签"。

**判定：论文说的是"名义第一但不显著"，公众号说的是"碾压"。**

---

## 三、CV3-Eval：公众号跳过了不利的那一列

公众号第 61 行：

> 更是在「野外」复杂环境下的CV3-Eval评测中，展现了惊人的纠错与抗噪能力。

这句本身没撒谎（没说第一），但它遮住了论文表 9 的实际形状：

| 系统 | zh↓ | en↓ | ja↓ | ko↓ | **四语平均↓** | hard-zh↓ | hard-en↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen-Audio-3.0-TTS | 3.35 | 4.25 | **4.78** 🥇 | **4.30** 🥇 | **4.17** 🥇 | 7.44 | 6.71 |
| Qwen3-TTS-1.7B-Base † | 3.19 | 3.92 | 5.00 | 4.63 | 4.18 | 9.36 | 7.47 |
| CosyVoice 3 | 3.91 | 4.99 | 7.57 | 5.69 | 5.54 | 9.77 | 10.55 |
| VoxCPM2 2B | 3.55 | 6.21 | 5.88 | 9.95 | 6.40 | 8.10 | 7.48 |
| OmniVoice 0.6B † | 3.89 | 4.57 | 7.24 | 11.80 | 6.88 | 11.98 | 19.69 |
| **Luna-TTS** | **3.17** 🥇 | **3.18** 🥇 | 5.00 | 5.93 | 4.32 | **6.90** 🥇 | **6.18** 🥇 |
| **Luna-TTS Realtime** | 3.62 | 4.06 | 6.36 | 5.76 | 4.95 | 12.56 | 13.98 |

拆开看：

| 子集 | Luna-TTS 的位置 |
|---|---|
| 中文 3.17 / 英文 3.18 | 🥇 第一（中文对 Qwen3 的 3.19 是平手级差距） |
| 两个 hard 子集 6.90 / 6.18 | 🥇 第一，且领先明显（这才是"抗噪纠错"的真凭据） |
| 日语 5.00 / 韩语 5.93 | ❌ 输给 Qwen-Audio-3.0 的 4.78 / 4.30 |
| **四语平均 4.32** | ❌ **输给 Qwen-Audio-3.0 的 4.17** |

论文摘要的措辞非常精确：**"the lowest Mandarin and English error rates"**（只说中英），
论文正文也直接承认 "Qwen-Audio-3.0-TTS retains the best Japanese, Korean, and
four-language-average results"。**这处 caveat 是论文自己写的，被公众号省略了。**

Realtime 在 hard 子集上的崩塌值得单独注意：hard-zh **12.56 vs 6.90**（1.8×），
hard-en **13.98 vs 6.18**（2.3×）。论文给了两个机理解释，都是块因果的直接代价：

1. **提交不可逆**：Luna-TTS 可以在全局精炼里回头改早期帧，Realtime 一个块 commit 就冻结，
   早期错误会持续污染后面每一块
2. **长度控制**：Luna-TTS 由 duration predictor 给出帧数，Realtime 靠学出来的 EOS 决策，
   不规则文本 + 低质量提示音频容易把它误导成截断或跑飞

---

## 四、逐条对照表

| # | 公众号说法 | 可核查证据 | 判定 |
|---:|---|---|---|
| 1 | 「语音质量最好 + 生成效率最高 + 首包延迟最低」六项第一 | 质量 4 项属 Luna-TTS，延迟属 Realtime，RTF 最低的又是 Luna-TTS 16 步；论文 Table 1 明写 "neither variant is dominant" | ⚠️ **三个配置拼成一个模型** |
| 2 | 「碾压谷歌」 | 论文原话"不能确立统计显著差异"，头对头 117–113 | ❌ **与论文结论相反** |
| 3 | 「Artificial Analysis Speech Arena 全球第三，超越谷歌」 | 论文里没有这个榜；这是公司在外部榜的某时点快照。Elo 榜周周变，同期检索到的 AA 榜前五含 Gemini 3.1 Flash TTS | ⚠️ **快照，且需注明时间** |
| 4 | 「HF TTS Arena 登顶全球第一，击败 ElevenLabs」 | 属实（外部报道可查），但那是**人类盲测 Elo**，与 Seed-TTS-Eval 的客观 CER/SIM 不是一回事，不能互相印证 | ⚠️ **两类指标被混说成一件事** |
| 5 | 「Seed-TTS-Eval 四项全第一」 | Table 8 支持 | ✅ 属实 |
| 6 | 0.6B 参数打赢 1.5B / 1.7B / 2B | Table 8 支持（CosyVoice3 1.5B、Qwen3-TTS 1.7B、MOSS 1.7B、VoxCPM2 2B 全部落后） | ✅ **这才是真亮点** |
| 7 | 「显著优势超越了字节 Seed、MiniMax、智谱和 Qwen 系列」 | zh SIM 79.7 vs Seed-TTS 79.6 差 **0.1**；en SIM 76.8 vs Qwen-Audio-3.0 的 76.2 差 0.6。论文自己用的词是 "marginally above" | ⚠️ **"显著"用错了** |
| 8 | 「击败 Qwen 系列」 | 表里 Qwen3-TTS 那行是 **`-Base`（未后训练）**，而 Luna-TTS 是过了 GRPO 的；论文另有 Qwen-Audio-3.0-TTS 是引用值 | ⚠️ **对照阶段不对等** |
| 9 | 100 万小时四语料库（中 43.4 / 英 43.1 / 日韩 13.6） | 论文 Table 2 完全一致 | ✅ 属实 |
| 10 | 「2 张 H20 上 TTFT 41.6 ms，RTF 0.024，超 40 倍实时」 | 论文 Table 5 完全一致（12 次预热运行的中位数，不含网络） | ✅ 属实，但要带上"预热 + batch 1 + 不含网络"的前提 |
| 11 | 「情感表现力超越 ElevenLabs」 | 混合：Expression Quality **4.54 🥇**、E-MOS **3.90 🥇**；但 Emotion Match **2.63 是第三**（MiniMax 2.70 / ElevenLabs 2.67），N-MOS 4.14 第二（ElevenLabs 4.18） | ⚠️ **四项里两项第一** |
| 12 | NVV（笑声/叹息等）能力 | PCER **39.95 🥇**、Recall **66.76 🥇**、F1 **72.52 🥇**；但 NVV Accuracy 3.60 是**第三**（MiniMax 3.95），Precision 79.38 第二（ElevenLabs 87.45） | ⚠️ 同上，有强有弱 |
| 13 | 「中国版 Thinking Machines」 | 纯类比修辞。相似处只有"顶尖研究者创业 + 从一个模态切入"；TML 做的是通用模型与可复现推理，VUI 做语音 | 🏷️ 修辞，非事实断言 |

论文自己还写了一节 **Limitations**，这些内容公众号一句没提：

> 韩语是四种语言里最弱的（CER 5.93），可能与韩语只占语料 6.9% 有关；语言覆盖只有 4 种，
> 远窄于大规模多语言 TTS；Luna-TTS 依赖**外部** duration predictor；
> Realtime 的 1.28 s 块大小是**写死**的、不随内容自适应，在 hard 子集上落后最明显。

---

## 五、一个容易被忽略的方法学细节：⋄ 与 †

论文 §6.1 明确区分两类数字：

| 标记 | 含义 | 影响 |
|---|---|---|
| **⋄** | 从对方的论文/技术报告/公开对比表里**直接引用** | 论文原话："may differ in evaluation details"（评测细节可能不同） |
| **†** | 用官方 checkpoint + 官方评测工具**自己复现** | 同协议，可比 |

Table 8 里**只有 Qwen3-TTS-12Hz-1.7B-Base 是 †**，其余对手全是 ⋄。也就是说，
"打赢 MiniMax / Seed-TTS / Qwen-Audio-3.0" 这些比较，是**跨报告的数字对齐**，
不是同一次实验里跑出来的。论文把这个 caveat 标得很清楚，公众号把它整个丢掉了。

这跟本仓库反复吃过的亏是同一件事：**不同来源的数字放进同一张表，
表格的形状会暗示一种它并不具备的可比性。**

---

## 指标名词速查

| 指标 | 全称 / 算法 | 方向 | 说明 |
|---|---|---|---|
| **CER / WER** | Character / Word Error Rate | ↓ | 拿 ASR 去"听"合成音频再转成文字，与原文比错误率。中文用 Paraformer-zh 算字错率，英文用 Whisper-large-v3 算词错率。**这是内容正确性，不是音质** |
| **SIM** | Speaker Similarity | ↑ | 合成音频与参考音频的说话人嵌入余弦相似度。Seed-TTS-Eval 用 WavLM-large，CV3-Eval 用 ERes2Net，**两者不可互相比较**（论文明确警告） |
| **RTF** | Real-Time Factor | ↓ | 生成 1 秒音频要花多少秒。0.024 = 40 倍实时 |
| **TTFT / 首块延迟** | Time To First Token/Block | ↓ | 从调用到第一块音频可播。**流式才有意义** |
| **NFE** | Number of Function Evaluations | ↓ | 串行前向次数。AR 是 O(T)~O(TQ)，Luna-TTS 是常数 S，Realtime 是 S_b·⌈T/B⌉ |
| **Elo** | Bradley-Terry 拟合的配对胜负强度 | ↑ | 盲测投票分。**与 CER/SIM 不可换算**，也不能说"Elo 高所以 CER 低" |
| **E-Sim** | Emotion Similarity | ↑ | emotion2vec_plus_large 嵌入相似度 |
| **PCER** | 在抽取出的非语言符号上算的 CER | ↓ | 衡量笑声/叹息这类事件有没有出现在该出现的位置 |
| **MOS / N-MOS / E-MOS** | Mean Opinion Score（真人 1–5 分） | ↑ | N = 自然度，E = 情感传达准确性 |
| **LALM 评分** | Large Audio Language Model as judge | ↑ | 这里用的是 Gemini 3.1 Pro Preview 当裁判 |

---

## 数字出处索引

本文件所有数字均可追溯：

| 数字 | 出处 |
|---|---|
| Seed-TTS-Eval 全表 | 论文 Table 8（demo 页同表可交叉验证） |
| CV3-Eval 全表 | 论文 Table 9 |
| 延迟 / RTF 六行配置 | 论文 Table 5 |
| 与开源系统的 RTF 对比 | 论文 Table 6 |
| 首包延迟横向对比 | 论文 Table 7（Cartesia 50 ms、ElevenLabs Flash v2.5 ~75 ms 等） |
| NVV 客观 / 主观 | 论文 Table 10 / Table 11 |
| 情感控制 | 论文 Table 12（E-Sim）/ Table 13 |
| 内部 Arena | 论文 Table 14 / Table 15 |
| 语料构成 | 论文 Table 2 |
| 训练阶段 | 论文 Table 4 |
