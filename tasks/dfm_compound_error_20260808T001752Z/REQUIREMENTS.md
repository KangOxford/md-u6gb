# 复合误差实验的需求与口径（活文档，持续 refine）

最后更新：UTC 2026-08-12。本文件记录**用户提出的需求与优先级**，以及每条需求当前的落实状态。
结论性数字不在这里，在 `RESULTS.md` / `ALEX_COMPOUND_ERROR.md` / `BREAKEVEN.md` /
`../../sigma-0-worktrees/action-chunking-20260809/docs/action_chunking/ACTION_THREE_STEPS.md`。

---

## R-1（2026-08-12，最高优先级）测量对象的优先级：先 order flow，后 stats

> 「这个 Compound Error 优先看它生成的 Order Flow，其次再看它生成的 Stats。
> 那些 LOB-Bench 的 21 Features 是关于它的 Stats。
> 我真正需要的是关于它生成这个 Orders，比方说把它取均值之类的，
> 它实际上是可以得到一个 Compound Error 之类的曲线的。」

### 两层测量对象的定义

| 层 | 是什么 | 具体量 | 优先级 |
|---|---|---|---|
| **一层：order flow** | 模型**直接生成**的 order 字段 | `event_type`、`direction`、`price_rel`、`size`、`log10_dt`（Δt） | **主** |
| 二层：stats | order flow 经**撮合引擎重建**后的订单簿统计量 | LOB-Bench 的 21 个 feature（`spread`、`log_depth`、`bid_volume`…） | 次 |

### 为什么这个排序是对的（不只是「更直接」）

二层的每一个偏差都同时含有两个来源：**模型生成错了** 与 **引擎回放/重建方式不同**。
本任务被撤回的最大一批数字（spread 17.9×、log-depth 15.5×）正是死在
「LOBSTER 重建 vs 模拟器回放」这个混淆上。**一层没有这个混淆，模型吐什么就是什么。**

推论：一层的结论可以单独成立；二层的结论**必须**配一个同引擎回放的真值对照才能读。

### 落实状态

| 需求 | 状态 | 产物 |
|---|:-:|---|
| 一层：order flow 上是否存在复合误差 | ✅ 已做 | `ACTION_THREE_STEPS.md` STEP 1；4× 外推处 **5/5 字段**都有 |
| 一层：后训练能否压低它 | ✅ 已做 | STEP 2；`size` 在 2×/3×/4× 全部显著压低，2× 处斜率转负 |
| 一层：**均值**的偏移 | ⚠️ **部分** | STEP 3 只取了 rollout **末端**的 $z$，**没有画 $z(m)$ 整条轨迹** ← **待补，见 T-1** |
| 二层：21 feature 谁 compound | ✅ 已做 | `FEATURES_OVER_T.md`、`figs/fig_lvl_slope_feat21_*.png` |
| 二层：后训练能否压低 | ✅ 已做 | `LONG_ROLL_500x500.md`、`AUDIT_AND_RERUN.md` §6.2 |

### T-1（待办，直接接 R-1）画 $z(m)$ 的整条曲线

$$z(m) \;=\; \frac{\mathrm{mean}_{\rm gen}(m) - \mathrm{mean}_{\rm true}(m)}{\mathrm{std}_{\rm true}(m)}$$

对每个 order 字段、每个 horizon 各画一条，draft 与 post 叠在同一张图上，
配日块 bootstrap 的误差带。数据已经在 `act/act_g{1000,1500,2000,3500}.json` 里，
只需在 `action_compound_error.py` 里把逐 m 的 `mean/std` 存出来（现在只存了末端）。

**为什么必须单独做**：$D_m$ 用**真值的矩**做标准化，模型整体平移后形状仍然匹配，
所以 $D_m$ 对「形状对、中心错」几乎不敏感。实测反例：8× 上 `log10_dt` 的
$D_m$ 斜率 +0.0011（CI 含 0，判「无复合误差」），**而它的均值偏了 1.607σ**。
**均值曲线不是 $D_m$ 的替代品，是 $D_m$ 测不到的那一半。**

同时要报的三个量（都是 $m$ 的函数，不要只取末端）：
`mean`（中心漂移）、`std`（离散度是否炸开）、`非法率`（当前 14.7%，与 horizon 无关）。

---

## R-2（沿用）判据口径，四条缺一不可

违反任何一条，实测都会给出「不能降低」甚至反号：

| 口径 | 违反时 |
|---|---|
| **横轴 = 生成步（事件轴）**，不是秒 | 同一份数据六项判据从「全好」翻成「全坏」 |
| **Stage 2A**（残差，主干逐位未动） | 2B 解冻主干，因果 CE 0.6827→2.6723（3.9×），几乎不修正 |
| **按交易日分块 bootstrap**，≥20 天 | 单日 n=64 给零宽度 CI + 相反结论 |
| **草稿误差 > 0.605 nats** | 低于阈值 corrector 净伤害 |

## R-3（沿用）不看 LOB-Bench 的池化 metric

不是「不优先」，是**已被证伪**：真实数据逐窗口打乱时间轴后，池化 spread KS
从 0.2393 变成 **0.0000（满分）**，排在真实 AR 模型之前。逐 $m$ 的 $D_m$ 才有意义。

## R-4（沿用）关注长尾与时间依赖

尾部：`kl_tail()`、`out_of_range_mass()`、`tail_mass_ratio`。
依赖：超额 MI、VR(k)、ACF 半衰期，**逐 lag 分开报**，不对 lag 求和（会掩盖 lag 结构）。

## R-5（沿用）结果一落地就写 md + 图；持续更新 Notion，多页而非单页

---

## 变更记录

| 日期 | 变更 |
|---|---|
| 2026-08-12 | 建档。新增 R-1（order flow 优先于 stats）与 T-1（$z(m)$ 曲线）；把原先散在 `PLAN.md` / `COMPLETION.md` 里的口径收敛成 R-2~R-5 |
