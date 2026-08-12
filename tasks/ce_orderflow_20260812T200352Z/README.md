# Order Flow 上的 Compound Error：从零重建的分析线

建档 UTC 2026-08-12T20:03Z。

## 这条线要回答的唯一问题

> **Discrete Flow Matching 后训练，能不能降低 sigma-0 生成器在 order flow 上的 compound error？**
> 如果不能，**为什么不能**？

## 测量对象：只看 order，不看 state

| 看 | 不看 |
|---|---|
| 模型**直接生成**的 order 字段：`event_type`、`direction`、`price_rel`、`size`、`log10_dt` | 撮合引擎重建出的订单簿状态 |
| 这些字段的分布、均值、以及它们随生成位置 $m$ 的演化 | LOB-Bench 的 21 个 feature（`spread`、`log_depth`、`bid_volume`…） |

**为什么划这条线**：state 层的每一个偏差同时含两个来源 —— 模型生成错了，和撮合/重建方式不同。这条研究线上被撤回的最大一批数字（spread 17.9×、log-depth 15.5×）正是死在「LOBSTER 重建 vs 模拟器回放」这个混淆上。**order 层没有这个混淆：模型吐什么就是什么。**

本目录**不引用**任何 21-feature 的结论，也不从旧目录继承任何数字。所有数字在这里重新算。

## 分析清单（每个一个独立文件夹，互不耦合）

| 编号 | 问题 | 状态 |
|---|---|:-:|
| **A01** | **order flow 上到底有没有 compound error？** | ✅ **有，但只有 `size` 与 `log10_dt`**。见 `A01_does_ce_exist/REPORT.md` |
| A02 | 有的话，DFM 后训练能不能压低它？ | ⬜ **前提已确认成立**；A01 §3.4 的预览是「能，且随机方向做不到」，待正式判定 |
| A03 | ~~生成长度 500 是不是根本不够长？扫 horizon~~ | 🔁 **方向被 A01 改写**：误差前置且饱和，加长 horizon 只会摊薄斜率并抬高 floor。改为**在前 500 步内加密**（`step` 20→5、`w` 40→20）并查饱和机理 |
| A04 | Learnable P 到底有没有必要？（P=0 消融 vs learned vs random） | ⬜ |
| A05 | 修正步数 N、起始噪声 $t_0$ 的必要性 | ⬜ |
| A06 | DFM 实现本身是否有问题（corrupt schedule / 掩码 / 残差接法） | ⬜ |
| A07 | 14.7% 非法消息率的根因 | ⬜ **已提级** —— 它是所有水平类结论的天花板，且**恒定**（与 horizon 无关），恒定意味着多半是可定位的实现问题而非训练不足 |

**编号一旦分配不改**。每个文件夹自带 `PLAN.md`（先写）、`code/`、`out/`、`REPORT.md`（后写）。
**禁止跨文件夹复用中间结果**：A02 要用到的量，A02 自己重算一遍。耦合会让某一处的错误静默传播到全部结论。

## 数据

`/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/dfm-post-training-20260801/post_training/dfm/artifacts/rollouts/`

| npz | n_seq | n_gen | 外推倍数 | 对照臂 |
|---|---:|---:|---:|---|
| `dfm_s2a_frozen_t080_L500_2a3500` | 64 | 500 | 2× | `dfm_s2a_a2_t080_L500_2a3500`（随机 P） |
| `dfm_h1000_learned` | 64 | 1000 | 3× | `dfm_h1000_random` |
| `dfm_h1500_learned` | 64 | 1500 | 4× | `dfm_h1500_random` |
| `dfm_h3500_learned` | 32 | 3500 | 8× | 无（待补） |

每个 npz 含 `real_msgs` / `draft_msgs` / `corr_msgs`，形状 `(n_seq, n_gen, 6)`，
列为 `time, event_type, order_id, size, price, direction`，外加 `ref_mid`、`dates`。
**三条流逐条配对**（同一个 `real_id`、同一份 draft），所以对照可以在序列级配对。

外推倍数 = (500 条条件 + n_gen 条生成) / 500 条训练窗口。

## 术语（本目录内固定含义）

| 术语 | 定义 |
|---|---|
| $m$ | 生成位置（第几条消息），**不是秒**。修正器会改变时钟，用秒做横轴会把时序改善误报成误差改善 |
| $D_m$ | 位置 $m$ 处生成分布对真值分布的 KL，除以真值熵 $H$ 归一 |
| **floor** | 真值流自己劈两半算出的同一个量。**不是 0** |
| **excess** | $D_m -$ floor。0 表示与抽样噪声不可区分 |
| **水平 (level)** | excess 对 $m$ 的平均 → 「误差有多大」 |
| **斜率 (slope)** | excess 对 $m$ 的回归斜率 → 「误差是否沿 rollout 累积」 |
| **compound error** | **就是斜率**。平的曲线哪怕水平很高也只是恒定偏差，不是 compound |

## 相关

- 标准化流程：`PROTOCOL.md`
- 前一轮（含 state 层）的记录：`../dfm_compound_error_20260808T001752Z/`，本目录**不继承其数字**
- DFM 代码：`sigma-0-worktrees/dfm-post-training-20260801/post_training/dfm/`
