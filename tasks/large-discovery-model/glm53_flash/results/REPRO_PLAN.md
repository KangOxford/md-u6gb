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

| 组 | 配置 | 说明 |
|---|---|---|
| **A. LDM** | `real_operation_tool_best_of_n.yaml` + GP 代理 + LCB 采集 | 完整 generate→select→evaluate→update 回路 |
| **B. LLM-only** | 同一提案模型、同预算,去掉代理与采集(反思式迭代) | 论文的对比基准 |

共同参数:提案模型 = **GLM-5.3-Flash**(本机 TP4 服务),起点 = 同一份
`real_train.py`,每候选评测 = 300 秒真训练,迭代数按预算定(论文用 100)。

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
B 组落 nid011132,各一张卡。

**契约扩展**:仓库契约把 `iterations=100 / warmup=20 / method=best_of_n` 锁死,
只覆盖 LDM 一组的满预算。没有绕过校验,而是按它的 schema **正式登记两个新
profile**:`repro_ldm_arm`(best_of_n 20+5)与 `repro_llmonly_arm`(single_turn 20+5)。
预算 25 次真评测 ≈ 2.5 小时/组,按 18 小时窗口反推,留出多种子余量。
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

## 执行状态

见 `RUNLOG.md`。GLM 服务形态与坑见记忆 `glm53-flash-serving-on-gh200`。
