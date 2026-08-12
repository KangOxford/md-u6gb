# MiniMax-H3 复刻

把 MiniMax H3（2026-07-31 发布的 33B 全模态视频+音频生成模型）缩小到 **101.7 M**，
从随机初始化训到出片，并把官方**声明过但没给配方**的机制逐条测出来。

先读 `findings.md`（架构反推结果与前提修正），再读 `task_plan.md`（六阶段计划与显式偏离）。
`progress.md` 是滚动记录。

---

## 一句话说清这是在复刻什么

MiniMax **没有发布技术报告，也没有发布训练数据或训练代码**。发布的是权重、
config，以及 diffusers 官方移植。所以：

- **架构不重写**：`MiniMaxH3Transformer3DModel` 就是参考实现，它的每个维度都是构造参数，
  H3-nano 只是同一个类的小 config。packed 序列布局直接调用参考实现的
  `build_packed_sequence`，帧数几何直接调用它的 `align_num_frames` /
  `video_latent_num_frames` / `audio_latent_num_frames`。这些都不可能与推理漂移。
- **训练配方才是本仓库造的东西**：训练时的序列组装、双 shift 流匹配损失、三阶段配方、
  以及数据管线。每条规则都能追溯到已发布产物里可检验的东西（调度器的代数、推理时的
  逐行时间步分配、文档化的 `keyframe_noise_aug`）。

---

## 目录

```
code/
  h3nano.py                模型 config、几何、流匹配、packed 布局      ← 核心库
  preprocess_vggsound.py   文本库 + 流式 tarball → VAE latent
  train.py                 三阶段：pretrain / sft_fl2va / distill_cfg
  sample.py                双调度器去噪 + 用 H3 自己的 VAE 解码
  evaluate.py              E1/E3/E4/E6：把官方陈述变成数字
  h3_smoke_infer.py        跑通官方 33B checkpoint，记录参数普查与真实布局
  fetch_assets.py          下载 checkpoint 与 VGGSound（不解包）
  01_fetch_and_smoke.batch 阶段 1
  02_pipeline.batch        阶段 2–6（单次分配跑完）
ckpt/h3_meta/              官方 config 与 VAE 源码（已下载，40 MB）
ckpt/h3/                   143 GB 权重（阶段 1 产出）
data/vggsound/             tarball，**保持打包状态**
data/latents/              latent 分片 + text_bank.pt（阶段 2 产出）
runs/                      checkpoint 与评估结果
venv/                      diffusers 0.40.0.dev0 + transformers 4.57.6 + PyAV
```

---

## 跑法

```bash
cd /lus/lfs1aip2/projects/public/u6gb/tasks/minimax_h3_20260812T

# 阶段 1：取资产 + 跑通官方 33B checkpoint（1 节点 4 GPU，6 h）
sbatch --job-name=h3-fetch-smoke code/01_fetch_and_smoke.batch

# 阶段 2-6：语料 → 闸门 → 预训练 → FL2VA → CFG 蒸馏 → 评估（1 节点 4 GPU，12 h）
sbatch --dependency=afterok:<阶段1 job id> --job-name=h3-nano-pipeline code/02_pipeline.batch
```

阶段 2–6 可用环境变量覆盖，无需改代码：

```bash
MODEL=small PRETRAIN_STEPS=40000 BATCH=6 CLIP_LIMIT=40000 GUIDANCE=7.0 \
  sbatch code/02_pipeline.batch
```

单独跑某一步：

```bash
PY=./venv/bin/python

# 只建文本库（309 类 Context-IR prompt + 1 条 null，走真 Qwen3-VL-32B 第 50 层）
$PY code/preprocess_vggsound.py --root . --phase text

# 只编码媒体（流式读 tarball，绝不解包）
$PY code/preprocess_vggsound.py --root . --phase media --shards 00,01 --limit 20000

# 预训练
torchrun --nproc_per_node=4 code/train.py --root . --stage pretrain --model nano --steps 24000

# 双 shift 消融（E5）：只改两个数字
torchrun --nproc_per_node=4 code/train.py --root . --stage pretrain --run-name ablate-shift-1-1 \
    --video-shift 1.0 --audio-shift 1.0 --steps 24000

# 采样：教师要 guidance（每步 2 次前向），蒸馏后的学生不要（1 次）
$PY code/sample.py --root . --checkpoint runs/<run>/checkpoints --out /tmp/s \
    --guidance-scale 5.0 --save-decoded
$PY code/sample.py --root . --checkpoint runs/<distilled>/checkpoints --out /tmp/s0 \
    --guidance-scale 0 --save-decoded

# 评估
$PY code/evaluate.py --root . --out E6.json roundtrip --n 16          # VAE 往返闸门
$PY code/evaluate.py --root . --out E3.json heldout --checkpoint <ckpt>
$PY code/evaluate.py --root . --out E1.json avsync  --samples /tmp/s/samples.pt
$PY code/evaluate.py --root . --out E4.json anchors --checkpoint <fl2va ckpt>
```

---

## 复刻时最容易写错的五件事

按踩坑代价排序。前四条都不会报错，只会安静地产出错误结果。

| # | 陷阱 | 正确做法 |
|---|---|---|
| 1 | 给 batch 里每个样本抽各自的时间步 | **不行**。`timestep_indices` 形状是 `(seq_len,)` 不是 `(batch, seq_len)`，batch 轴是纯复制轴。整个 batch 共用一对 `(t_video, t_audio)`，两个模态之间才是独立的。时间步多样性靠梯度累积补。 |
| 2 | 以为文本行没有时间步 | 文本行**继承视频的 t**（参考实现先 `torch.full(seq, video_t)` 再覆盖其余）。所以文本条件不是静态的。 |
| 3 | 速度目标写成 `ε − x₀` | H3 的速度**朝向数据**（`x₀ = x_t + σ·v`），目标是 **`v = x₀ − ε`**，且 `t = 1` 表示干净。符号与常规流匹配相反。 |
| 4 | 像素归一化用 `[-1, 1]` | H3 的视频 VAE 用 **ImageNet 均值/方差，基准区间 `[0, 1]`**。视频后验要 `.sample()` 且舍入到 fp16，音频后验用 `.mode()` **从不采样**。 |
| 5 | 把 VGGSound tarball 解包 | 每片约 1 万个 mp4，摊到 Lustre 就是每轮 1 万次 MDT open。流式读，一次大顺序读进、一个大 latent 分片出。 |

---

## 已知边界

- **H3-Context-IR 与 H3-Regenerate-2K 是闭源托管服务**，官方明说 Context-IR
  「对最终输出质量至关重要」。本仓库自建了一个模板化的小型 Context-IR
  （同样的三段式 `integrated_multimodal_description` / `overall_soundscape` /
  `non_diegetic_music`），但它不可能等价。
- **训练数据不是 MiniMax 的**。VGGSound 是公开替身。
- **时长 3.04 秒落在 H3 的 5–15 秒窗口之外**，理由与代价见 `task_plan.md`。
- 原生稀疏注意力未包含在开源推理实现中，因此也不在复刻范围内。
