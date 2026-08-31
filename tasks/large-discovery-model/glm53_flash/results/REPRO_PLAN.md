# 用 GLM-5.3-Flash 复现 LDM v0.1 主要结论

**论文**:Large Discovery Models (LDM v0.1): Empirically-grounded Model-Based
Open-Ended Search,[arXiv:2608.15669](https://arxiv.org/abs/2608.15669),
仓库 [yzailab/Large-Discovery-Models](https://github.com/yzailab/Large-Discovery-Models) @ b5dab16a。

**目标**(用户令):用 GLM-5.3-Flash 作提案模型复现主要实验结论,争取更好。

## 论文的三个主结论

| # | 场景 | 论文结论 | 基准 | 评测机制 |
|---|---|---|---|---|
| C1 | nanoGPT 训练代码搜索 | 验证 BPB 的**绝对下降幅度是 LLM-only 反思的 2.4 倍** | Karpathy 原始 LLM-only 循环 | 每候选真训练 300 秒 GPU,报 `val_bpb` |
| C2 | 抗体 CDRH3 设计 | 200 次评测后**结合能均值低 18.2%** | LLM-only 反思、AntBO | Absolut! 结构模拟器,五个 PDB 靶点 |
| C3 | 小分子多目标 | Pareto 前沿**超体积 +62.4%(vs LLM-only)/ +63.1%(vs 经典 BO)** | LLM-only、经典 BO | Vina 对接 + KRAS G12D 活性模型,80 次迭代 |

## 本机可行性(先算再动)

| # | 阻塞 | 判定 |
|---|---|---|
| **C1** | 无。任务环境纯 pip(torch),评测在本地 GPU 跑,数据 `karpathy/climbmix-400b-shuffle` 可下 | ✅ **主攻** |
| C2 | Absolut! 二进制需外部安装;且 `ldm-tts-antibody` 钉 transformers 4.13 → tokenizers 0.10.3 **aarch64 无轮子、需 Rust 编译**(TEST_REPORT E3 已实测失败) | ⚠️ 次选,先解依赖 |
| C3 | 需 Vina 二进制 + `best_g12d_model.joblib`(**不公开分发**,须向维护者索取) | ❌ 素材缺失,不可自足复现 |

**结论:以 C1 为复现主体。** 它是三者中唯一素材自足、评测确定、且能在本机
GPU 上真跑的场景;也是论文里唯一给出可核对轨迹(`assets/examples/real_100_20260809/`,
val_bpb 0.986220 → 0.981844)的场景。

## C1 的复现设计

**要复现的量不是"某个 BPB 值",而是一个比值**:LDM 相对 LLM-only 反思的
绝对改善倍数(论文 2.4×)。所以必须成对跑两组,同起点、同预算、同评测:

| 组 | `acquisition-feedback` | 说明 |
|---|---|---|
| **A. LDM** | `brief` | GP 把预测与不确定性喂回提案,即论文说的"代理把外部观测变成预测与认知不确定性,再由采集函数决定下一步" |
| **B. LLM-only** | `none` | 同模型同预算,只反思实际评测结果,没有代理信号 |

**其余逐项相同**:提案模型 = GLM-5.3-Flash(本机 TP4 服务)、起点 = 同一份
`real_train.py`、`breadth=1 depth=1`(每轮恰好 1 次真评测)、`warmup=5`、
`iterations=40`、`seed-policy=best`、`surrogate-mode=lcb`、评测 = 300 秒真训练。

**为什么落在这一项**:读代码确认 `breadth x depth` 决定的是**真评测次数**,
不是 GP 打分的候选数 —— 用它做区分会让 A 组多跑 7.7 倍评测。真正把代理接进
回路的开关是 `--acquisition-feedback`(默认 `none`),它决定 GP 的后验要不要
进提案提示。这才是"有没有经验代理"这一条区别的机械实现。

**判定**:`Δ_A / Δ_B` 与论文 2.4× 比较;`Δ` 取从共同起点算起的 val_bpb 绝对下降。
单条轨迹不构成因果估计(论文自己也这么说),所以多种子是必需项而非可选项。

## 落地细节(实测确定,非假设)

**单次评测实测**(nid011132,GH200 单卡):
```
val_bpb 1.019308 | training_seconds 300.0 | total_seconds 364.1
mfu 33.31% | 50.3M params | 799 steps | 418.9M tokens | peak 45.1 GiB
```
起点 bpb 与论文的 0.986220 不同(数据分片数、卡型不同),**这不影响复现**:
要比的是同起点下两组的改善幅度之比,不是绝对值。

**每组必须独占一张 GPU**:评测是 300 秒定时训练,被抢卡 ⇒ 同样 300 秒跑更少
token ⇒ bpb 变差。GPU 争抢会直接伪造出「这组更差」的结论。A 组落 nid011131、
B 组落 nid011132,各一张卡 —— 靠 `srun --gres=gpu:1 --exclusive` 分卡,
**不自设 `CUDA_VISIBLE_DEVICES`**(见下方缺陷表第三行)。

**契约扩展**:仓库契约把 `iterations=100 / warmup=20 / method=best_of_n` 锁死,
只覆盖 LDM 一组的满预算。没有绕过校验,而是按它的 schema **正式登记两个新
profile**:`repro_ldm_arm` 与 `repro_llmonly_arm`,两者的 `locked_args` **完全相同**
(`best_of_n` / `iterations=40` / `warmup=5`)—— 契约锁的是"花了多少真评测",
而那正是必须对齐的量;区别项 `acquisition-feedback` 不是预算项,不进 locked_args。
每组 85 次真评测(5 预热 + 40 轮 x 2),按 12 小时作业窗口反推。
契约摘要随每次运行快照落盘,改动可追溯。

## 共卡会伪造结论(实测,已据此改方案)

第一对试跑起在 6197253 的 nid011131/011132 上,起跑后逐进程核对发现:
另一条线的 dfm 评测**铺满了这三个节点的 12 张卡**(每卡 29.2 GB),
我的训练(49.6 GB)与它挤在同一张 GPU 0 上、利用率 100%。

对定时评测这是致命的:**同样 300 秒,被抢卡就跑更少 token,bpb 直接变差**。
两组若共卡程度不同,「哪组更好」就成了「哪组邻居更闲」。所以:

- 第一对(6197253 上那两条)**降级为试跑**,只用来验证 campaign 机制跑得通
  (LLM 生成、GP 拟合、产物落盘),**不产出结论数据**。
- 正式跑另开 **`--exclusive` 独占作业**(6204550,2 节点,3 种子 × 2 组 = 6 条,
  每条独占一卡)。同种子的 A/B 落同一节点,让节点差异在成对比较里抵消。

多种子是相对论文的增量:论文 nanoGPT 只报了单条轨迹,自己也写明
"not a controlled causal estimate";3 种子成对能给出方向一致性。

## 三轮起跑暴露的缺陷(全部已修,记录以免重犯)

| 轮次 | 缺陷 | 表现 | 修法 |
|---|---|---|---|
| 6204550 | 服务与实验分在两个作业 | 托管 GLM 的分配先到期,6 条 campaign 在 `glm_health=000` 下空跑 1 小时 | 服务改为**与实验同作业**,健康不通过就不起跑(exit 9) |
| 6216995 | `breadth x depth` 是**真评测次数**,不是 GP 打分的候选数 | A 组 345 次 vs B 组 45 次,预算差 7.7 倍 —— 这样比是把"多跑评测"算到代理头上 | 两组同为 `breadth=1 depth=1`,唯一差别改为 `acquisition-feedback`(GP 引导开/关) |
| 6216995 | 自设 `CUDA_VISIBLE_DEVICES=1/2/3` | `srun --gres=gpu:1` 已隔离好卡,step 内部永远是设备 0;那四条指向不存在的卡,每次评测 11 秒返回 1e+09 | 删掉,靠 `--gres=gpu:1 --exclusive` 分卡 |
| 6216995 | 起服省掉了 `--tool-call-parser/--enable-auto-tool-choice` | `/health` 返回 200,但 `operation_tool` 生成器走 tool calling,每个候选吃 **400 Bad Request**,30/30 外层候选 `generation_error` | 加回官方标志;起跑前**真发一次带 tools 的请求**,失败 exit 10 |

**这四条的共性**:前置检查查的是"进程在不在",而真正会坏的是"这条具体路径通不通"。
现在 `run_repro_arm.sh` 的两道检查(健康 + 工具调用探针)都是**发真实形状的请求**。

## 执行状态

见 `RUNLOG.md`。GLM 服务形态与坑见记忆 `glm53-flash-serving-on-gh200`。
