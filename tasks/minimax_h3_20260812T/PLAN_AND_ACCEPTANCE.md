# H3 复刻：完成计划与验收判据

任务根：`/lus/lfs1aip2/projects/public/u6gb/tasks/minimax_h3_20260812T/`
写于 2026-08-14。配套文档：`findings.md`（架构反推）、`progress.md`（滚动记录）、`README.md`（怎么跑）。

---

## 0. 先说清楚这个任务在验收什么

用户的原始要求是「照文章复刻」，但**技术报告至今未发布、训练数据未开源**。
所以本任务复刻的是 **H3-Base 的机制**，靶子是官方**已文档化并可检验的陈述**，
不是官方 demo 的画质。

这条区分决定了全部判据的形状：

| 不拿来当判据 | 拿来当判据 |
|---|---|
| 画面好不好看 | 参数分布是否与 33B 同构 |
| FVD / 人评 | 训练目标的符号是否正确（可用调度器精确验证） |
| 与官方 demo 比较 | 双 shift、模态 AdaLN、锚点 0.999、CFG 蒸馏是否真的生效 |

一个 101.7 M 模型在 VGGSound 上训几小时，画面必然模糊。**拿画质当判据等于设一个
注定失败的目标**，而机制是否被正确复刻是可以精确回答的。

---

## 1. 阶段、产物、判据

每一行的「判据」都是一个数字或一个文件，「检查」列是得到它的确切命令。

| # | 阶段 | 产物 | 验收判据（PASS 条件） | 检查命令 |
|---|---|---|---|---|
| **S0** | 架构反推 | `findings.md` | AdaLN 参数量算术 = 13.01 B（官方「~13B」）；总量 32.27 B（官方「33B」）；`adaln_out/hidden = 18 = 6×3` | 已完成 ✅ 见 §1.1 |
| **S1** | 代码正确性 | `code/test_h3nano.py` | **13/13 通过**，其中「完美速度走 64 步落回 x₀」误差 < 1e-3 | `$PY code/test_h3nano.py` |
| **S2** | 资产落地 | `ckpt/h3/`、`data/vggsound/` | checkpoint **≥ 100 GB**（脚本内已硬断言）；`vggsound.csv` 存在；`transformer/config.json` 存在 | `du -sh ckpt/h3` |
| **S3** | 官方 33B 跑通 | `runs/*/h3_reference/` | `param_census.json` 里 `adaln_branches / TOTAL` ∈ [0.35, 0.45]；`h3_t2va_smoke.mp4` 存在且 > 1 MB 且含音轨 | `01_fetch_and_smoke.batch` 阶段 3 |
| **S4** | latent 语料 | `data/latents/` | `manifest.json` 的 `kept ≥ 8000`；`latent_frames == 22`（= `5n+2`，n=4）；`audio_latents == 122` | `cat data/latents/manifest.json` |
| **S5** | VAE 往返闸门 | `E6_roundtrip.json` | 视频 **PSNR ≥ 28 dB**、音频 **SI-SDR ≥ 10 dB** | `evaluate.py roundtrip` |
| **S6** | 预训练 | `runs/pretrain-*/checkpoints/` | 留出 loss 相对随机初始化**下降 ≥ 50%**；`latest_checkpoint.json` 可 resume | `evaluate.py heldout` |
| **S7** | FL2VA 后训练 | `runs/sft_fl2va-*/` | 锚点扫描里 **t=0.999 的 video loss < t=1.0 的** | `evaluate.py anchors` |
| **S8** | CFG 蒸馏 | `runs/distill_cfg-*/` | 采样前向次数 **2(N−1) → (N−1)**；墙钟降幅 ≥ 40% | `sample.py --guidance-scale 0` |
| **S9** | 报告 | `REPORT.md` | E1–E7 每项有数字或明确「未做+原因」 | 人读 |

### 1.1 S0 已完成的证据（不下载权重，靠 safetensors 头部 range 请求）

```
adaln_proj.linear = Linear(2688 → 96768)   96768/5376 = 18 = 6 调制参数 × 3 模态 ✓
AdaLN 总量 = 50 × 96768 × 2688            = 13.01 B   ← 官方 "~13B"  ✓
模型总量   = 50 × (154.1+231.2+260.1) M   = 32.27 B   ← 官方 "33B"   ✓
attn 内维 7168 / hidden 5376              = 1.3333    = 4/3          ✓
```

---

## 2. E1–E7：把官方陈述逐条变成数字

这七项是本任务的**核心交付**。每项对应一条官方说过、且可独立检验的话。

| # | 官方陈述（原文） | 度量 | PASS 判据 | 状态 |
|---|---|---|---|---|
| **E1** | *"video and audio come out of the same denoising loop"*（联合生成、同步） | 音频包络 × 帧间差分能量的互相关，**带错配对照** | `matched_peak − control_peak > 0`，且中位滞后 \|lag\| ≤ 3 帧 | 待跑 |
| **E2** | *"guidance is baked into the weights ... every step runs exactly one forward pass"* | 采样前向次数与墙钟 | 前向 `2(N−1) → (N−1)`；墙钟降 ≥ 40% | 待跑 |
| **E3** | 蒸馏不该毁掉能力 | 留出流匹配 loss（**固定时间步网格**，两个 checkpoint 同噪声水平） | 蒸馏后 loss 不高于教师 **+20%** | 待跑 |
| **E4** | *"trained with its anchors very slightly noised, so conditioning on exactly t = 1.0 is off-distribution"* | 锚点 t ∈ {1.0, 0.999, 0.99, 0.95, 0.9} 扫描 | `loss(0.999) < loss(1.0)` | 待跑 |
| **E5** | 双 shift（视频 12 / 音频 3） | 三组同预算训练：(12,3) / (1,1) / (3,12) | (12,3) 留出 loss 最低 | 视预算，可缺 |
| **E6** | 管线正确性闸门 | VAE encode→decode | 视频 PSNR ≥ 28 dB，音频 SI-SDR ≥ 10 dB | 待跑 |
| **E7** | 「调小一点」的代价 | micro / nano / small 三点留出 loss | 随参数量**单调下降** | 视预算，可缺 |

**E5 / E7 允许缺席**，但必须在 `REPORT.md` 里写明「未做 + 原因（预算）」，
不允许静默略过——这是本仓库 `feedback_verify_before_done` 的要求。

### 判据阈值是怎么定的（不是拍脑袋）

- **PSNR ≥ 28 dB**：H3-VisualVAE 是 f16t4d24 的高压缩 VAE，官方称「across-the-board
  gains in reconstruction quality」。28 dB 是「压缩 VAE 重建可用」的常规下限；
  低于它说明**像素约定写错了**（例如用了 `[-1,1]` 而非 ImageNet over `[0,1]`），
  这正是这道闸门要抓的东西。
- **loss 下降 ≥ 50%**：随机初始化的流匹配 loss ≈ 2×latent 方差。降一半说明模型
  确实学到了条件结构，而不是只学会输出均值。
- **锚点 0.999 < 1.0**：这是**对官方训练事实的直接复现**。若不成立，说明我的
  `keyframe_noise_aug` 没真正生效，或者训练步数不足以形成这个差异——两种都要如实写。
- **E1 必须有对照**：只报「匹配对相关性为正」没有意义，声音和画面共享响度包络就能
  做到。判据是**匹配对显著高于错配对**。

---

## 3. 执行路线（受算力现实约束）

### 3.1 硬约束

- **不许 lock GPU**（2026-08-14 用户令，机制已物理删除）
- **每次 sbatch 前先跑 `gtop`**；有空卡就 attach，绝不排队
- `held`（0% util 但显存 > 64 MiB）**不是空卡**
- attach 的 srun 必须 `setsid nohup`，否则随会话死
- **不许 `scancel`**（本仓库 P0）

### 3.2 当前资源现实（2026-08-14T12:05Z）

```
gtop:  唯一分配 6007121（4N/16GPU）全部 held，每张只剩约 11 GB → 无空卡
       但 CPU 0.4–0.7%、RAM 余约 370 GB  → 零 GPU 的下载可以 attach
排队:  6011373（fetch+smoke, 1N）、6011374（pipeline, 1N, 依赖前者）
```

### 3.3 分工

| 路径 | 承担 | 为什么 |
|---|---|---|
| **attach 零 GPU step**（进行中，`6007121.85`） | S2 下载 143 GB | 长杆，且完全不需要 GPU，不与显存持有者冲突 |
| **排队的 6011373 / 6011374** | S3–S8 | 需要 GPU；资产已落地后 `fetch_assets.py` 秒过 |
| **`03_pipeline_small.batch`**（待命） | 全流程单卡兜底 | 若只有单卡空出来 |

**顺序原则：产物耐久度优先。** 资产和 latent 语料落在 Lustre 上能活过任何 step 死亡；
训练每 1000 步 checkpoint，可在任意节点 resume。所以先做耐久的，再做可重来的。

---

## 4. 什么算「这个任务完成了」

**必须全部满足：**

1. S1 = 13/13 ✅（已达成）
2. S2–S6 全部 PASS —— 即：资产落地、官方 33B 跑通并出片、语料建成、VAE 闸门过、
   H3-nano 从随机初始化训到留出 loss 降一半以上
3. E1、E2、E4、E6 四项有**数字**（这四项不依赖额外训练预算）
4. `REPORT.md` 写完，其中 E3/E5/E7 要么有数字，要么写明「未做 + 原因」

**明确不在验收范围内**（已在 `findings.md` §7 记录）：

- H3-Context-IR、H3-Regenerate-2K（闭源托管，无从对照）
- 原生稀疏注意力（未包含在开源推理实现中）
- 与官方 demo 的画质比较（三模块只开源中间一个，比较无意义）
- 时长在 H3 的 5–15 秒窗口之外（用 73 帧 = 3.04 秒，理由与代价见 `task_plan.md`）

---

## 5. 当前进度对照

| 阶段 | 状态 | 实测值 |
|---|---|---|
| S0 架构反推 | ✅ | AdaLN 13.01 B / 总量 32.27 B，与官方「~13B / 33B」吻合 |
| S1 代码正确性 | ✅ | 13/13；完美速度走 64 步落回 x₀ 误差 **8.34e-07** |
| S2 资产落地 | ✅ | **144.1 GB，完整自洽**（见下）+ CSV 199,467 行 + tarball 16.9 GB |
| S3 官方 33B | ⏸ 等 GPU | 排队中 `6011373` |
| S4 latent 语料 | ⏸ 等 GPU | **解码路径已用真实 mp4 验证：6/6 通过**（见下） |
| S5–S8 | ⏸ 等 GPU | |
| S9 报告 | ⏸ | |

### S2 验收证据（`runs/S2_checkpoint_verify.json`）

```
transformer       14 shards    66.28 GB (index says 66.28)   ✓ 逐位吻合
text_encoder      14 shards    66.71 GB (index says 66.71)   ✓
vae                3 shards    10.42 GB (index says 10.42)   ✓
audio_vae          2 files      0.61 GB
total 144.1 GB  →  checkpoint complete and self-consistent
```

三道闸门各抓各的，缺一不可：

| 闸门 | 抓的失败 | 本次是否抓到 |
|---|---|---|
| 目录体积 ≥100 GB | 「只下了 1 个文件」 | 抓到过（job 5998320，0.0 GB） |
| 分片 vs 索引 `total_size` | 「文件都在但某片截断」 | 本次全部吻合 |
| 入口文件存在 | 「144 GB 权重但没有 `model_index.json`」 | **本次抓到** |

第三道最反直觉：144.1 GB 的体积断言完全通过，缺的只是 3 KB 的管线入口——
而 `ModularPipeline.from_pretrained` 第一件事就是读它。

### S4 前置：解码路径已用真实 mp4 验证（`code/_decode_smoke.py`，零 GPU）

```
decoded 6 clips in 2.5s        →  0.42 s/clip
  v(73, 256, 256, 3) uint8        std 37.8–68.3   非空白
  a(2, 97333) stereo @32kHz       peak 0.22–1.11  非静音
6/6 clips decoded correctly
```

`decode_clip` 此前只在合成张量上被间接测过，**从没解码过一个真的 mp4**——
它涉及 PyAV 容器解析、可变帧率最近邻重采样、音频重采样器 flush、中心窗口对齐，
每一处都可能在真实文件上翻车。登录节点跑不了（4 GiB cgroup 会杀掉流式解压 16.9 GB
gzip 的进程），所以 attach 一个**零 GPU** 的 step 到活着的分配上跑。

**两个对后续有用的读数**：

1. **0.42 s/clip → 解码不是瓶颈**。16 worker 并行下 12,000 clip 约 5 分钟，
   S4 的时间几乎全花在 GPU 上的 VAE 编码。预算要按 GPU 分，不是按 CPU。
2. **有一个 clip 的音频峰值 1.1109 > 1.0**。不是 bug：参考实现的
   `MiniMaxH3AudioReference` 直接传原始 float32 波形、不归一化，所以照做是对的。
   但记下来——若后续音频重建质量异常，这是第一个该查的地方。
