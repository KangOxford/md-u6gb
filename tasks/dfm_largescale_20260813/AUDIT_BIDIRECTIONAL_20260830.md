# 审计三：causal → bidirectional——冻结的是参数，不是行为

应用户问（2026-08-30）：「attention matrix from causal to bidirectional，
我怀疑这里多少有点问题。」结论先行：**这是三个审计对象里最重的一个。
修正器把按 causal 目标预训练的 mamba3 参数，直接放进对称化的 decay 结构里跑
——参数一字未动（trunk_l1_shift = 0 验证的就是这个），但计算的函数彻底换了。
「主干冻结」给人的安全感是错位的：冻结保证了参数恒等，没有任何东西保证
行为在分布内。修正器的全部上下文智能都建立在这个未经训练的行为之上。**

（本模型无 attention matrix，等价物是 mamba3 SSD 的 decay mask——问题同构：
causal mask 换成对称 mask。）

## 1. 代码实况

| 项 | 位置 | 事实 |
|---|---|---|
| 机制 | `patches/models~mamba3.py.patch` | `bidirectional=True` 把 SSD decay mask 对称化：`W[l,s] = exp(−|s_l − s_s|)`（causal 原形：只对 s ≤ l 累积） |
| 路径限制 | 同 patch | 仅纯 JAX SSD 路径；Triton/CUDA kernel 不支持，强行开会 raise |
| 生产开关 | `dfm_correct_runner.py:892` | `build_model(bidirectional=True, dfm_residual=True, ...)`——修正器**永远**双向 |
| 训练侧同 | `dfm_train_worker.py:398-411` | 训练与推理同构双向；注释自认「a different function」，且记录过一次事故：拿双向模型测 AR 曲线，得到的全是无意义数 |
| 数值验证 | `test_mamba3_bidirectional.py` | chunked == brute force、causal 默认不变、decay 有界——**验证的全是数值等价性，没有一条涉及分布** |
| 恒等验证 | runner 日志 `sum|trunk−pretrained| = 0` | **参数**恒等。行为无任何验证 |

## 2. 问题拆解

### 2.1 decay 参数的语义错位

mamba3 的每层 decay 率（Δ/A 系参数）是在 causal 目标下学的：
「过去的信息以多快衰减」。对称化之后，同一组衰减率被用于**未来方向**——
这个用法在预训练的任何一步里都没有出现过。B/C 投影与门控同理。

### 2.2 状态统计整体漂移，且逐层放大

causal 下位置 l 的 state 聚合 s ≤ l 的贡献；对称下聚合**全序列**——
求和项数近乎翻倍，每层输出的量级与协方差结构都不是预训练时的那个。
patch 的测试里专门有一条「naive form explodes / bidirectional decay is
bounded」——作者处理了**数值爆炸**，处理的是能不能算，不是算出来的
分布对不对。L 层堆叠让第一层的漂移成为第二层的 off-distribution 输入，
逐层累积。

### 2.3 双重 off-distribution 叠加

修正器喂给主干的还是 **corrupted 序列**（field 度量加噪）——预训练同样
没见过。于是主干运行在「没见过的输入 × 没见过的结构」上。
而审计一已证：残差分支无上下文，**修正器的全部上下文能力都来自这个
双重出格的主干输出**。

### 2.4 −0.53 里有多少是「给 bidirectional 打补丁」？——可分离，从未分离

P 残差学到的第一件事，很可能是把 bidirectional 漂移拉回 decoder 能读的
范围，其次才是修正 corruption。分离手段**已在代码里**：
`--random-p --random-p-scale 0.0` = P=0、纯双向前向（runner:869-877 的
注释明说这是「bidirectional resampling works 与 post-training works 的
分离」）——**但 REPORT 与全部下游结论里没有这一格的数**。
teacher-forced −0.53、生成侧全部指标，至今无法拆开「结构效应」与
「学到的修正」。

## 3. 为什么这与复合误差直接相关

修正器的水平抬升（draft 0.023 → corr 0.411）需要一个「它系统性地把
合理 token 改错」的机制。审计一给了信号缺失（残差只含边缘信息），
本审计给了地基缺陷：**决定「每个位置该是什么」的上下文表征，来自一个
从未被训练过的函数**。两者相乘，正好解释「大改（64%）、改错方向
（与 real 一致率反降）」的观测。

## 4. 修法（按成本排序，与 R2 组合）

| 方案 | 做法 | 代价 | 备注 |
|---|---|---|---|
| **b4 校准层（推荐先做）** | 每层 SSM 输出后加零初始化 per-channel affine，R2 训练时随残差一起学 | 参数 ~L×2×1024，与 B7b 同量级；零初始化保恒等启动 | 把「结构漂移校正」显式交给新参数，不让 P 独自扛 |
| b1 两遍 causal 拼接 | 正向 causal + 序列翻转再 causal，hidden 拼接/平均后进 decoder | 2× 前向；每遍的**计算**都是参数被训练过的用法 | 反向序列内容上仍 off-distribution，但衰减语义保持 |
| b2 轻量解冻 | bidirectional 下给主干加 LoRA/FiLM 并训练 | 破坏 2A「主干比特级冻结」的卖点 | R2 反正重训，可作第二格 |
| b0 消融先行 | 跑 P=0 纯双向格（开关现成） | 一次推理 | **无论选哪条修法都先跑它**——它标定现状里结构效应的大小 |

## 4.1 P=0 消融结果（2026-08-31 补，§4 的 b0 已跑）

25 ticker × 2026-01（n_seq=32，state=b7b_off_s42、`--random-p-scale 0.0`）：

| 字段 | 纯双向 P=0 level | 学习版修正器 level | draft level |
|---|---:|---:|---:|
| event_type | **+1.284** | +0.185 | +0.020 |
| direction | **+0.423** | −0.017 | +0.012 |
| price_rel | +0.346 | +0.312 | +0.137 |
| size | +0.304 | +0.944 | +0.163 |
| log10_dt | +0.278 | +0.201 | +0.050 |

（学习版列 = tau_t0/lg488b，60 ticker 同月同管线；量级可比、参照 draft 略有差）

**第二种子复现（2026-08-31 晚，p0bidir_s43，31 ticker）**：event_type +1.232、
direction +0.413、price_rel +0.363、size +0.314、log10_dt +0.280——与 s42
逐字段同量级同方向，slope 同样全正（+0.02~+0.10）。分解结论两种子成立。

三个定论：
1. **双向结构效应是水平损伤的主源**：P=0 时 event_type 的 level 是学习版的
   7 倍、draft 的 64 倍。中途数据（改动率 62.6%、一致率 0.513→0.335）
   已示：破坏力属于「未训练双向前向 + per-position 重采样」的机制本身。
2. **P 残差的主要工作是补偿这个自伤**（+1.28 → +0.19），补完仍比 draft 差
   一个量级——teacher-forced −0.53 的相当一部分是在给架构缺陷打补丁。
3. **斜率修复确实是学出来的**：纯双向格 slope 全正（+0.02~+0.12，无修复），
   学习版才有负 slope（price_rel −0.044）。
   （size 一列学习版反而更差（0.94 vs 0.30）——lg488b 与 b7b_off 的 state
   差异所致，作跨 state 噪声看待，不入定论。）

**对修复的直接推论**：b4 校准层（或干脆放弃对称 mask，走 b1 两遍 causal）
从「建议」升为「必做」——R2 只换训练分布的话，模型仍要先花容量填这个坑。

## 5. 与另两份审计的合流

- 审计一（P）：残差无上下文 ⇒ 上下文全靠主干 ⇒ 本审计的地基问题被放大。
- 审计二（embedding）：主干输入端的表征天花板。
- 三者合成一句话：**修正器 = 边缘查表（P/LN） + 未训练的双向行为（主干） +
  无几何的 embedding**。R2 换训练分布之外，b4 校准层 + P 的门控化
  （AUDIT_P_MATRIX §3）是同一次重训里应当一起上的两个架构修正。
