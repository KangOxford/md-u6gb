# R7 设计定稿与实施计划

创建：2026-08-12T03:18Z
状态：设计定稿待对抗审查；未建 worktree，未改代码。

---

## 1. R7 要解决的问题

R6.1 在三项分布匹配指标（WS-21 / KS-21 / L1-21）上已经全胜 26tok，但**第 4 项 reference
成功率落后 18.07 pp**（78.0736% vs 96.1435%）。R7 只攻这一项，不动其他部分。

**主判据（用户 2026-08-12 裁定）**：`exact-target` —— 撤单是否命中模型真正想引用的那一单。
`any-live-ID` 与 `live-ID + price consistency` 作为附报，防止跨版本比较时口径漂移。

选 exact-target 的直接后果：**不能再用 26tok 的 96.1435% 当追赶目标**，那个数是
resolver 兜底出来的（L1 exact-ns 只有 75.3516%，L2 同价档毫秒最近邻兜底 +20.7919%，
且完全丢掉 reference time 的反事实仍有 87.9616%）。26tok 的 exact-target 数**根本算不出来**，
因为它的产物没保存原始 `price_ref/size_ref/time_ref`。

---

## 2. 数据上限（已钉死，不要试图突破）

| 边界 | 值 | 来源 |
|---|---:|---|
| 可见目标占全部 touch | 27,789 / 31,630 = **87.8565%** | 3,841 条目标订单早于 condition 窗口 |
| 纳秒 `ref_age` 在可见目标内唯一性 | **100.0000%**（27,789/27,789，误选 0） | 已两条独立实现互证 |
| `side+price+ref_size` 在可见目标内唯一性 | 86.3003% | 同上 |

**所以 87.8565% 是 R7 的天花板。** 缺口不是算法问题，是初始化信息缺失：可见历史里没有那
3,841 笔旧订单的逐单创建时间，`ref_index` 同样指不到它们。要突破只有一条路——**初始化簿时
携带窗口前订单的逐单 ID/price/size/time**。这是一个独立的工程决定，本轮**不做**，但必须在
报告里写明天花板的位置，避免把 87.86% 当成失败。

分事件天花板（纳秒键、可见目标内全 100%，下列是占全部事件的比例）：

| 事件 | 上限 |
|---|---:|
| partial CANCEL | 64/76 = 84.2105% |
| DELETE | 27,205/30,849 = 88.1876% |
| EXECUTION | 520/705 = 73.7589% |

---

## 3. 字段设计

> **v2 修订（2026-08-12T03:30Z，依据 `results/EXPERIMENTS.md`）**：
> **`ref_size` 已砍掉。** 实测它的边际收益只有 **+0.48 pp**（85.8253% → 86.3003%），
> 因为消息自身的 `quantity` 已经携带几乎相同的信息（对占 97.9% 的 DELETE，
> `quantity == remaining` 恒成立、`== original` 也有 99.67%）。
> v2 字段：`type, side, price, quantity, ref_age_ns, ref_index`。

touch 事件（CANCEL / DELETE / EXECUTION）字段（v1 原案，`ref_size` 已废）：

```
type, side, price, quantity, ref_size, ref_age_ns, ref_index
```

| 字段 | 含义 | 定稿理由 |
|---|---|---|
| `price` | 被操作订单的价格 | **复用，不新增 `price_ref`**。真实 touch price 与目标订单 price 一致；R6 的 resolver 已在用 `(side, price, ref_code)`。26tok 额外预测的 `price_ref` 与消息自身 price 有 38.60% 不一致且从未参与 lookup，是纯浪费 |
| `quantity` | **本次**修改量（原 `size` 改名） | 与 `ref_size` 必须分开：partial CANCEL 中 `quantity == ref_size` 是 **0/64**，EXECUTION 中也只有约 69.6% |
| `ref_size` | 被引用订单的**原始** size（静态） | 选 original 而非 remaining：oracle 只差 0.0432 pp（毫秒口径 94.4079% vs 94.3611%，微秒口径完全相同），却要维护动态状态、训练与解码都更复杂 |
| `ref_age_ns` | 当前 touch 时间 − 被引用 NEW 创建时间，**完整纳秒** | 由现有 `dt_ns` 累加即可恢复每个 NEW 的 `created_at_ns`，**不需要**每条消息再带一份绝对时钟。纳秒是唯一能在可见目标内做到 100% 的精度（μs 98.04%、ms 94.18%、s 87.48%）|
| `ref_index` | 倒数第 k 个 NEW，`1..241`，另设 `OUTSIDE_WINDOW` | 熵最小：倒数 NEW rank **6.05 bit** < 向后消息距离 6.99 bit < 绝对 data index 8.54 bit。绝对 index 依赖窗口位置，语义不稳 |

`dt_ns` 与 NEW 事件的编码**保持 R6 不变**。词表已有 `0..999` 区间，`ref_index` 不需扩表。

---

## 4. 匹配层

> **v2 修订（2026-08-12T03:30Z）：三通道投票已被自己的实验否决，改为「约束优先 + ref_age 消歧」两层。**
>
> **否决理由**：T（`ref_age`）与 C（`ref_size`）**同源**——都从 reference 字段读值、
> 都在 `(side, price)` 池内定位。模型若整体 copy 错一笔订单，二者会一致地指向它，
> 凑成两票"共识"。实测 T+C 同时混淆时**撤错 21.2710% 的单，且全部来自 `consensus`**
> （`single_wrong` = 0）。错误伪装成高置信共识，比低置信兜底更危险。
> 四种裁决规则（多数决 / 共识须含 I / T+C 折叠 / 全一致）都救不了：后三种确实把
> 撤错单压到 0，但把命运绑死在 I 上——I 一错位，命中从 86.5055% 崩到 **0.2051%**。
> **问题不在规则，在于只有两个独立信息源。**
>
> ### v2 定稿：约束优先 + ref_age 消歧
>
> ```
> 第 1 层  约束（零 token 成本，读消息自身字段）
>   可行集 = { o : o.side == msg.side,
>              o.price == msg.price,
>              o.remaining == msg.quantity      (DELETE)
>              o.remaining >= msg.quantity      (EXECUTION / partial CANCEL) }
>   |可行集| == 1  → 直接接受，不读任何 reference 字段
>   |可行集| == 0  → miss（实测 0 次）
>
> 第 2 层  消歧（仅当 |可行集| > 1）
>   可行集内按 ref_age_ns 最近邻；距离 > τ 或平局 → ambiguous，不撤单
>
> 第 3 层  ref_index 交叉验证（可选，默认关闭）
>   仅在第 2 层候选距离接近 τ 时启用，用于抑制「混淆到可行集内」的残余风险
> ```
>
> **实测分工**：第 1 层直接定案 **23,850 次（85.83%）**，第 2 层只需消歧 **3,939 次（14.17%）**。
>
> **为什么这两层才是真独立**：约束层由**消息自身**的 `side/price/quantity` 驱动，
> T 层由 **reference** 字段驱动。模型即使 reference 全错，约束层仍然正确工作。
> 实测约束层把 copy 型错误的撤错单压掉 **72.5%**（36.1438% → 9.9500%）。
>
> **残余风险**：混淆到**可行集内**的订单时约束完全失效（14.1747%，与无约束相同）。
> 这个数正好等于约束无法唯一确定的比例，只能靠 `ref_age` 预测质量或第 3 层压低。
>
> **`quantity` 硬约束是零成本的 +21.97 pp**（`side+price` 63.8562% → 85.8253%），
> 且「唯一但选错」= 0、「可行集为空」= 0。这一条**同样适用于 R6.1 现有的 78.0736%**，
> 属于可以免费送给 R6 接手方的改进。

---

### 以下为 v1 原案，保留备查（已被上方 v2 取代）

### 4.1 为什么不用串行

用户最初的设计是串行四步：`T 纳秒 → 粗精度回退 → price+quantity → data index 做 double check`。
两处已被实验否决或修正：

**(a) 粗精度回退无效，删除。** 截断匹配的失败模式是 **pred 落进一个空桶**，不是「桶里候选太多」；
放宽精度治的是后者。实测 δ=1 ns 就把 ns 精度从 100% 打到 **0.0000%**，退到 μs 只救回 49.0194%。

**(b) `ref_index` 不该放最后做 double check。** 数据说它在**覆盖率**上一点不增加（纳秒键单独已
100% 唯一），它的唯一价值是模型把 `ref_age` 预测错时的**独立第二预测通道**。串行放最后只能
否决、不能纠错——前面的通道一旦给出错误答案就已经落地了。

### 4.2 定稿形态

三个通道**各自独立**产生候选，互不依赖：

| 通道 | 键 | 弃权条件 |
|---|---|---|
| **T** | `ref_age_ns` 最近邻 | 最近候选距离 > **τ**（默认 1 μs）→ 弃权 |
| **C** | `(side, price, ref_size)` 内容键 | 匹配数 ≠ 1 → 弃权 |
| **I** | `ref_index` 指针（倒数第 k 个仍 live 的同方向 NEW） | 越界 / 指向已失活订单 / `OUTSIDE_WINDOW` → 弃权 |

裁决：

```
候选集 = {T, C, I} 中未弃权者给出的 order_id
├─ ≥2 个通道给出同一个 id        → 接受，provenance = consensus
├─ 恰好 1 个通道给出候选          → 接受，provenance = single_channel_{T|C|I}
├─ ≥2 个通道给出不同 id 且无多数  → ambiguous，**不撤单**
└─ 全部弃权                       → miss，**不撤单**
```

**关键性质：不存在「永不失败」的路径。** 任何时刻都可能判 ambiguous 或 miss。
这是刻意的——26tok 的 96.14% 之所以虚高，正是因为它的 L2 无阈值最近邻永远能返回一笔。
τ 把「找不到」和「硬撤一单」之间的取舍变成一个**显式参数**；26tok 的病等价于 τ = ∞。

### 4.3 τ 的选取依据

由 Δ_min（同 `(side, price, original_size)` 干扰订单的最近创建时间距离）分布：

| Δ_min ≤ | 占可见目标 |
|---:|---:|
| 100 ns | 0.1619% |
| **1 μs** | **2.6161%** |
| 1 ms | 6.1823% |

τ = 1 μs 意味着：只要模型 `ref_age` 误差 < 1 μs，97.38% 的目标不可能被干扰单抢走。
实施时按 `τ ∈ {100 ns, 1 μs, 10 μs, 1 ms, ∞}` 扫一遍，把曲线放进报告，不写死一个值。

---

## 5. 必报指标（防止口径漂移）

| 指标 | 为什么必须报 |
|---|---|
| **exact-target accuracy** | 主判据 |
| any-live-ID rate | 与 26tok 的 96.1435% 同口径对照 |
| live-ID + price consistency | 与 26tok 的 73.8585%、R6.1 的 78.0736% 同口径对照 |
| 各通道单独命中率（T / C / I） | 判断哪个通道在真正干活 |
| 通道间冲突率 | 冲突高说明某通道在系统性犯错 |
| ambiguous 率 / miss 率 | 这两个数是「诚实的失败」，压低它们不能靠放宽 τ |
| **wrong-target rate** | 撤错单的比例，任何时候都要单列 |
| token 成本 | 新字段使每条 touch 变长，影响上下文能装多少消息，进而影响公平对比 |

---

## 6. 实施步骤

| # | 步骤 | 产出 | 状态 |
|---|---|---|---|
| 0 | 三通道投票 oracle 实验 + τ 扫描 | `results/three_channel_oracle.json` | 进行中 |
| 1 | 对抗性审查（subagent）打这份设计 | `review/adversarial_*.md` | 进行中 |
| 2 | 按审查结果修正设计 | 本文件 v2 | |
| 3 | 建 R7 worktree + 分支 | `sigma-0-worktrees/varlen-R7-*` | |
| 4 | tokenizer：新增三字段 + 无损 roundtrip 测试 | `src/varlen_R7/lossless_R7.py` | |
| 5 | resolver：三通道 + τ + 投票 + provenance | `src/lob/varlen_inference_R7.py` | |
| 6 | 冒烟：小规模训练 + 生成 + roundtrip 100% 闸门 | `logs/smoke_*.log` | |
| 7 | 正式训练 | W&B run | |
| 8 | 推理 + paired-255 生成 | 765 文件 | |
| 9 | LOB-Bench + 五项指标 | `results/bench_*.json` | |
| 10 | 报告 + Notion | `results/REPORT.md` | |

**闸门（不通过不进下一步）**：
- 步骤 4 后：100% 精确 encode/decode roundtrip，否则不训练（这是 R6 的硬判据，R7 继承）。
- 步骤 6 后：冒烟必须跑通完整链路，包括 checkpoint 保存与 resume。
- 步骤 7 前：`squeue` 去重检查；训练脚本必须含 checkpoint + resume（CLAUDE.md P0）。
- 训练进程必须 `setsid` / `nohup ... & disown` 与 shell 脱钩（R6 今天两次死于此）。

---

## 7. 与 R6 的边界

R6 全部移交，见 `/projects/public/u6gb/R6_HANDOVER_20260812.md`。
R7 **不碰** `varlen-R6p2-stepunits-20260811T0520Z` worktree、不碰 job `5980745` / `5980502`。
R7 需要自己的 allocation。
