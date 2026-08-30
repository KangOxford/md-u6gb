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

## 执行状态

见 `RUNLOG.md`。GLM 服务形态与坑见记忆 `glm53-flash-serving-on-gh200`。
