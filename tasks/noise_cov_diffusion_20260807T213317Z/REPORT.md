# 噪声协方差 Diffusion 在 LOB 数据上的实验报告

**滚动记录文件** — 每拿到一个结果就在此追加，不等全部做完再回写。

- 工作目录：`/lus/lfs1aip2/projects/public/u6gb/tasks/noise_cov_diffusion_20260807T213317Z/`
- 论文 codebase：`https://github.com/OpenReviewAnonymous/Diffusion`（NeurIPS 2025），本地克隆于
  `/run/user/1483804540/claude-1483804540/-lus-lfs1aip2-projects-public-u6gb/06d15d16-f48d-4ec8-9438-fcd4d669738f/scratchpad/ncd_paper/`
- 数据：`/lus/lfs1aip2/projects/public/u6gb/datasets/lob_flat43_example_20260807T135612Z/`
  （8 只 SP500 大市值科技股 × 2025-12-01 × 每票前 20 万事件，43 列宽表）
- 起始时间：UTC 2026-08-07T21:33Z

## 任务状态

| # | 任务 | 状态 |
|---|---|---|
| 0 | 数据属性分析 | **已完成**，见第一节 |
| 0.1 | 使用论文 codebase | **已完成**，方法与损失已对齐，见第二节 |
| 1 | iid 噪声 baseline | 进行中，首轮结果见第四节（发现主干容量瓶颈） |
| 2 | 高维高斯噪声（HDGN） | 进行中，同上 |
| 3 | 「iid 最终能追上但更贵」是否被实验支持 | 待完成 |
| 4 | LOB-Bench 打分与报告 | 评估层已实现并验证，见第三节 |
| 5 | 详细 md 文件 | **本文件**，滚动更新 |

---

## 一、任务 0：数据属性分析

脚本 `code/analyze_data.py`，输出 `data_analysis.json`，图 `figs/fig0_structure.png`、`figs/fig0b_concentration.png`。

样本构造：43 列宽表 → 可逆特征化（`log1p(Δt)`、价格转相对 BBO 中间价的 tick 偏移、量取 `log1p`）→
按 43 个通道 z-score → 沿事件轴切成 T=16 的窗口 → 展平成 D = 16×43 = **688** 维。
共 100,000 个窗口（train 90,000 / test 10,000）。

用相对中间价而非绝对价，是因为绝对价的协方差会被中间价的日内漂移这一个主成分吃掉
（所有价格列相关系数 > 0.9999），那种「强相关」只反映非平稳性，不反映盘口结构。

### 1.1 截面结构（单行 43 通道）

| 量 | 实测值 |
|---|---|
| 平均 \|非对角相关\| | 0.330 |
| \|r\| > 0.9 的通道对占比 | 6.6% |
| \|r\| > 0.5 的通道对占比 | 见 `data_analysis.json` |
| 相关矩阵条件数 | **1.90 × 10¹³** |
| PC1 解释方差 | 44.2% |
| 95% 方差所需主成分 | 24 / 43 |

条件数 1.9×10¹³ 已逼近 float64 精度极限（~1e16），说明 43 个通道里存在**近乎精确的线性关系**：
10 档价格彼此相差一个近似恒定的 tick 间距，因此矩阵在数值上接近奇异。

### 1.2 窗口协方差（D = 688）

| 量 | LOB 数据 | iid 噪声（Σ=I）应为 |
|---|---|---|
| 条件数 | **3.03 × 10⁵** | 1 |
| PC1 解释方差 | **44.1%** | 0.15% |
| 50% 方差所需主成分 | **2** | 344 |
| 90% 方差所需主成分 | 30 | 619 |
| 99% 方差所需主成分 | 187 | 681 |
| 有效秩（participation ratio） | **4.8** | 688 |
| 有效秩（entropy） | 17.7 | 688 |

**这是整份分析里最关键的数字**：688 维数据的方差参与比只有约 **4.8 个自由度**。
iid diffusion 在 688 个方向等量注入噪声，其中约 99.3% 的噪声能量打在数据几乎不去的方向上。

### 1.3 边际分布与时间结构

| 量 | 实测值 |
|---|---|
| 拒绝正态的通道数（Jarque-Bera, 5%） | **43 / 43** |
| 中位 \|偏度\| | 1.00 |
| 中位超额峰度 | 6.59 |

| 通道 | lag-1 自相关 |
|---|---|
| `ask_p1_tick` | **+0.992** |
| `ask_v1_log` | +0.964 |
| `trade_p_tick` | +0.525 |
| `log_dt` | +0.488 |

盘口价格在事件尺度上几乎是持续的（0.992），这既是截面相关的来源，也是窗口内时间相关的来源。
两者共同构成 D=688 上那个高度非各向同性的 Σ。

### 1.4 小结

这份数据在三个层面同时违反 iid 高斯假设：截面强相关（条件数 1e13）、时间强持续（lag-1 0.99）、
边际重尾非正态（43/43 拒绝正态，中位超额峰度 6.6）。噪声协方差在这里不是可选的调优项，
而是数据的第一性结构。

---

## 二、任务 0.1：论文 codebase 的方法

论文 repo 关键文件：`ddpm.py`（iid baseline）、`adaptive_ddpm.py`（论文方法）、
`diffwave.py` / `adaptive_diffwave.py`、`adaptiv_ddpm_regularity.py`（正则项消融）。

### 2.1 可学习的噪声协方差

`AdaptiveDDPM` 把噪声协方差参数化为 Cholesky 因子并**作为模型参数学习**：

```python
self.L_params = nn.Parameter(torch.randn(input_dim, input_dim) * 0.01)
def get_L(self):
    L = self.L_params * self.tril_mask          # 下三角
    L[diag, diag] = F.softplus(L[diag, diag])   # 对角为正 -> 保证正定
def get_noise(self, B):
    xi = torch.randn(B, input_dim)
    return xi @ L.T                              # ε ~ N(0, LLᵀ)
```

这比「固定 Σ = 样本协方差」更强：噪声协方差可以适应去噪任务本身，而不只是匹配数据的二阶矩。

### 2.2 损失是 Mahalanobis，不是 MSE

```python
r = eps_theta - eps
z = solve_triangular(L, r.T, upper=False).T     # z = L⁻¹ r
recon_loss = mean(sum(z**2, dim=1))             # = (ε_θ−ε)ᵀ Σ⁻¹ (ε_θ−ε)
loss = recon_loss + λ1*‖L‖_F² + λ2*‖diag L‖²    # λ1 默认 1e-2, λ2 默认 0
```

两点必须记住：

1. **噪声协方差必须与损失度量绑定**。用 N(0,Σ) 加噪，就要用 Σ⁻¹ 加权度量误差，这对应正确的高斯似然。
   本实验初版曾误用普通 MSE，导致 HDGN 的 loss（0.63）看似远低于 iid（0.98）——那不是「学得更快」，
   而是低秩噪声让 MSE 天然偏低。换成 Mahalanobis 后，`‖L⁻¹ε‖²` 对任何 Σ 都服从 χ²_D，
   期望恒为 D，**四臂 loss 自动同尺度可比**。实测 `hdgn_fixed` 的 loss 收敛到 1.0000，正是该理论值。
2. **Frobenius 正则不是可选项**。只最小化 `‖L⁻¹r‖²` 的话，把 L 放大到无穷即可让 loss→0，是平凡解。
   `λ1‖L‖_F²` 正是堵这个洞。

### 2.3 参数量分解（用于对照设计）

论文的 `AdaptiveDDPM(T=16, D=43)` 实例化到本数据维度后：

| 组成 | 参数量 | 占比 |
|---|---|---|
| 可学协方差因子 L（688×688） | 473,344 | **52.5%** |
| 网络主干 | 427,696 | 47.5% |
| 合计 | 901,040 | 100% |

**L 占了一半以上的参数**。因此若只比较论文 iid DDPM（428k）与 AdaptiveDDPM（901k），
赢了也分不清是「协方差有用」还是「参数多一倍」。这直接决定了下一节的四臂设计。

---

## 三、实验设计与 LOB-Bench 评估层

### 3.1 四臂对照

| 臂 | Σ | L 是否可学 | 主干 | 隔离的变量 |
|---|---|---|---|---|
| `iid` | I | 否 | H | 论文 baseline（任务 1） |
| `iid_wide` | I | 否 | 加宽至与 `hdgn_learned` 同总参数 | 参数量的价值 |
| `hdgn_fixed` | 样本协方差（固定） | 否 | H | **协方差本身的价值**（零额外参数） |
| `hdgn_learned` | LLᵀ | 是 | H | 论文方法（任务 2） |

四臂共享：网络结构、参数初始化种子、数据、数据顺序、优化器、β 调度、评估随机种子。
**唯一变量是 Σ**。损失统一为 Mahalanobis 形式（iid 时 L=I 退化为 MSE）。

`hdgn_fixed` 是设计里最关键的一臂：它与 `iid` 参数量完全相同，只是拿到了正确的协方差。
若它胜过 `iid`，则协方差的价值与参数量无关。

### 3.2 LOB-Bench 评估层

指标实现照抄权威实现 `lob_pipeline/lob_bench/metrics.py`（2026-08-07 核对）：

| 指标 | 定义 |
|---|---|
| `wasserstein` | real+gen 合并后 z-score，再 `scipy.stats.wasserstein_distance` |
| `ks` | 同样先合并 z-score，再 `scipy.stats.ks_2samp().statistic` |
| `l1` | 分箱计数各自归一化成概率后 `\|p−q\|.sum()/2`（即总变差距离，上限 1） |

注意 `l1 = 1.0` 表示两个分布**完全不重叠**，是饱和值，不是「归一化后的 1」。

**feature 覆盖**：WS-21 中有 12 个可从 43 列直接计算，9 个不可（需要 order-id 追踪或 message type）：

| 覆盖（12） | 不覆盖（9） |
|---|---|
| `spread`, `orderbook_imbalance`, `log_inter_arrival_time`, `ask_volume_touch`, `bid_volume_touch`, `ask_volume`, `bid_volume`, `vol_per_min`, `ofi`, `ofi_up`, `ofi_stay`, `ofi_down` | `log_time_to_cancel`, `limit_{ask,bid}_order_depth`, `{ask,bid}_cancellation_depth`, `limit_{ask,bid}_order_ticks`, `{ask,bid}_cancellation_ticks` |

这个划分不是任意的：覆盖的全是**盘口状态的函数**（截面统计量），不覆盖的全需要**订单身份追踪**。
43 列宽表刻意丢掉了 order_id 与 message type，丢的正好是这一整类。对本实验可接受，
因为噪声协方差影响的正是截面相关结构，恰好落在覆盖的那 12 个上。

### 3.3 两个必需的参照基线

没有这两条线，任何分数都无法解释：

| 基线 | l1 | wasserstein | ks | 含义 |
|---|---|---|---|---|
| **FLOOR**（真实 vs 真实，held-out 对半劈） | 0.1815 | 0.0228 | 0.1762 | 有限样本下的最好水平，是真正的 0 点 |
| **N(0,Σ) 未训练**（直接采样，零训练） | 0.4271 | 0.1699 | 0.3934 | 只有二阶统计量、无任何高阶结构的水平 |

`N(0,Σ)` 这条线尤其重要：它同时是 `hdgn_*` 臂的采样起点 x_T，
所以任何 HDGN 结果必须**优于 0.4271** 才算训练带来了增益。

---

## 四、首轮结果与已发现的问题（进行中）

### 4.1 论文原始主干在本数据上不足

用论文原配置（`trunk=paper`：`[688+32] → 256 → 256 → 688`，两层 ReLU MLP，无归一化无残差），
`iid` 臂训练 43,200 步的结果：

| 训练步 | loss | l1 | wasserstein | ks |
|---|---|---|---|---|
| 300 | 0.8134 | 0.8170 | 1.2452 | 0.6422 |
| 1,200 | 0.7407 | 0.8170 | 1.2348 | 0.6929 |
| 7,200 | 0.7393 | 0.8186 | 1.2437 | 0.6917 |
| 19,200 | 0.7427 | 0.8183 | 1.2426 | 0.6906 |
| 43,200 | 0.7328 | **0.8178** | 1.2467 | 0.6917 |

**l1 从 0.8170 到 0.8178，两个数量级的训练步数换来零改善。**
loss 只从 0.81 降到 0.73，意味着模型仅解释了约 27% 的噪声方差——基本没学动。

这本身是一条有价值的发现：论文原始主干是为 `input_dim = T×D = 100×2 = 200` 设计的，
本数据 `input_dim = 688`，同样的 256 宽两层 MLP 会把所有臂一起压在饱和区，
失去区分度。**在得出「iid 学不动」的结论前，必须先排除容量瓶颈**，否则会把
实现层面的限制误报成方法层面的结论。

处理方式：把主干宽度/深度参数化（`--trunk deep`：残差 + LayerNorm + SiLU），
**对四臂完全相同**，因此不影响「唯一变量是 Σ」的对照。同时保留 `--trunk paper`
以便报告论文原配置下的结果作为对照。

### 4.2 待跑

- [ ] `trunk=deep` 下的四臂主实验（判定容量瓶颈是否解除）
- [ ] 训练步数扫描 → 任务 3 的 quality-vs-cost 曲线
- [ ] NFE 扫描（采样步数维度的成本）
- [ ] iid 长跑（4× 预算）判定能否追上 HDGN

---

## 五、计算环境（可复现信息）

| 项 | 值 |
|---|---|
| 登录节点 | **无 GPU**（driver 12070 不支持，且 BriCS 规则禁止登录节点 GPU 计算） |
| 计算节点驱动 | 565.57.01（CUDA 12.7） |
| u6gb 默认 python | `/home/u6gb/kangli.u6gb/miniforge3/bin/python3`，torch **2.11.0+cu130** → **与驱动不匹配，无法用 GPU** |
| 实际使用 | `/lus/lfs1aip2/projects/public/s5e/quant_team/quant/miniforge3/bin/python3.12`，torch **2.8.0+cu129**，`cuda.is_available()=True` |
| 运行方式 | attach 到运行中的分配 5931446（4 节点，`--overlap --exact --gpus=1`），未新建排队作业 |

**GPU 闸门检查**（attach 前必做）：5924043 显存 68.3/97.9 GB 且 sm 100%（满载，不碰）；
5931446 显存 34.2/97.9 GB 且 sm 全 0%（怠速，可用）。但宿主显存会动态涨到 68 GB，
曾撞上瞬时峰值导致 CUDA OOM，需 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 并容忍重试。

论文 repo 的 `import ot`（POT）在本环境缺失。处理方式是 import 前把 `ot` 塞进 `sys.modules`
作为 stub，从而加载论文的类定义原件——既不改论文文件，也不往别人的 conda 环境里装包。
`ot` 只用于论文自带的评估，本实验的评估走 LOB-Bench 的 scipy 实现。
