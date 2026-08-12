# R7 词表设计

创建：2026-08-12T03:40Z
状态：设计稿。已吸收对抗审查 D6 / D8 / D10 / D15。未建词表、未改代码。
所有数字标注了来源：**实测**（本轮跑出来）/ **源码**（读到的常量）/ **待定**（需要构建时才知道）。

---

## 0. R7 是什么

R7 = SP500 LOB 消息流**无损变长词表**的第 7 代。它**只攻一件事**：让撤单/成交消息引用到
**正确的那一笔**历史订单（exact-target），不动架构、不动分布匹配那三项指标的编码。

背景：R6.1 在 WS-21 / KS-21 / L1-21 上已经三项全胜 26tok，但 reference 的 exact-target
只有 **78.0736%**。26tok 那个看起来漂亮的 96.1435% 是撮合器兜底刷出来的，不是预测准确率。

---

## 1. 继承自 R6 的编码机制（不改）

### 1.1 核心参数（源码 `src/varlen_R6/lossless_R6.py`）

| 常量 | 值 | 含义 |
|---|---:|---|
| `BASE` | **1024** | 所有数值字段的进制 |
| `NUMERIC_FIELDS` | `(dt, price, size, ref)` | 走长度前缀编码的字段 |
| `SIGNED_FIELDS` | `{price}` | 编码前过 zigzag |
| `TYPEDIR_SLOTS` | **11** | (event_type, side) 的联合符号 |
| `T_SEC_HI_SLOTS` | **48** | t_sec 高位；zigzag 后需 ceil(2×23399/1024)=46，取 48 留余量 |
| `DTNS_SLOTS` | **1000** | dt 的纳秒余数 0–999，恰好全覆盖、**永远单 token** |

`TYPEDIR_PAIRS` 把事件类型与方向合成一个符号：NEW/PCAN/DEL/EXEC_V 各有 ask/bid 两槽，
EXEC_H（类型 5）无挂单方向占单槽，CROSS（类型 6）两槽，共 11 槽。
**理由（源码注释）**：分开存要两个 token/条 = 全流的 25.3%，却只送出约各 1 bit；
一个 token 装 log2(vocab) bit，超过 90% 的位置被浪费。联合符号信息量完全相同。

### 1.2 三条编码路径（解码器按序测试）

```
[HEAD[v]]                      1 token      v 因语料频率赢得专属 ID
[SHORT+hi, DIG+lo]             2 tokens     hi = v//1024 < short_width 且 v < 1024²
[LEN_k, DIG+d₁ … DIG+d_k]      1+k tokens   任意 v，k = 2..max_digits 个 base-1024 位
```

**canonical 是硬性的**：能走短路径的值必须走短路径，解码器**拒绝**长形式。
这让 encode/decode 互为双射，模型无法用两串不同的 token 表达同一事件。
长度由 token 流自己携带，所以编码是"长到装得下值"，而不是"把值截断到装得进编码"——
这正是 R5 之前那版丢掉 62,275,273 条 DT 和 31,235,758 条 REF 的原因。

### 1.3 ID 区段分配（源码 `build_layout`，按 cursor 顺序）

```
take(11)                    typedir        (event_type, side) 联合符号
take(1)                     dt_zero_id     dt == 0 的专用 ID
take(3)                     special        PAD, BOS, EOS
for name in (dt, price, size, ref):
    take(head_sizes[name])  head           高频值专属 ID      ← 构建时按语料频次定
    take(short_widths[name]) short         SHORT+hi
    take(1024)              dig            DIG+lo / 数字位
    take(max_digits[name]-1) len           LEN_k
take(48)                    t_sec_hi
take(1024)                  t_sec_lo
take(1000)                  dtns           dt 纳秒余数（R5 的 t_us 段回收改用）
```

`build_layout` 末尾断言 `cursor == total_size`，**填不满就是硬错误**，不允许留静默空洞
（空洞后来会表现为不可达的 token ID）。

**固定开销**（与语料无关的部分）：
`11 + 1 + 3 + 4×1024 + 48 + 1024 + 1000 = 6,183` 个 ID。
R5 总词表 15,847，故 head + short + len 三类合计 **9,664** 个 ID 由构建时的语料统计决定。

### 1.4 ref 字段的双命名空间（R7 必须继承，见 §4.1）

```
ref > 0   →  同价档队列里倒数第 k 笔（R6 语义）
ref < 0   →  raw_ref_code(order_id) = -(zigzag(order_id) + 1)
```

负数命名空间承载**窗口外**的引用：目标订单早于编码流时，直接编码它的原始 order_id。
源码 `derive_ref_codes_R6` 的说明：*"This closes the lossless gap left by R6, which
silently omitted every unresolved reference."* 残留未解析引用时抛 `LosslessEncodingError`。

**这是 100% 无损 roundtrip 闸门得以通过的机制**，R7 不能用哨兵值取代它。

---

## 2. R6 的 `ref` 语义：同价档排名，不是全局排名

R6 把引用指针从 R5 的**全局 NEW 排名**改成**同价档队列排名**，这是一次有实测依据的修正：

| 排名口径 | 中位 | p99 | max | 训练/生成一致性 |
|---|---:|---:|---:|---|
| R5 全局 NEW 排名 | 10 | 86 | 121 | **差**：训练窗口约 1,080 个 NEW，生成条件窗口只有 125 个，**浅 8.6 倍** |
| **R6 同价档排名** | **1** | **4** | **9** | **好**：队列深度在两个场景下同样浅 |

源码实测同价档正排名范围 **[1, 133]**。R5 全局排名的失败表现为撤单引用成功率 87.10%。

> **对抗审查 D6 已采纳**：R7 **不得**把指针改回全局排名（PLAN v1 曾写 `1..241` 的
> "倒数第 k 个 NEW"，该数无出处，且正是 R6 已经否决掉的形态）。R7 沿用同价档语义。

---

## 3. R7 的字段变更

### 3.1 变更表

| 字段 | R6 | R7 | 依据 |
|---|---|---|---|
| `type`, `side` | typedir 联合符号 | **不变** | — |
| `price` | 相对半 tick，signed | **不变** | touch 的 price 在 LOBSTER 语义下就等于目标订单的 price，无需 `price_ref` |
| `size` → `quantity` | 本次修改量 | **仅改名** | 语义澄清；**并升级为硬约束**（见 §3.3） |
| `ref` | 同价档排名 / raw 负码 | **不变** | §2 |
| `ref_size` | — | **不加** | 实测边际仅 **+0.4750 pp**（85.8253%→86.3003%），不值一个字段 |
| **`ref_age_ns`** | — | **新增，条件编码** | 把可见目标内的唯一识别率推到 **100%**；条件编码见 §4 |

### 3.2 为什么砍掉 `ref_size`（实测）

| 键 | 唯一识别率 | 新增 token |
|---|---:|---|
| `side + price` | 63.8562% | 0 |
| **`side + price + quantity` 硬约束** | **85.8253%** | **0** |
| `side + price + ref_size` | 86.3003% | +1 字段 |

对占 97.9% 的 DELETE，`quantity == remaining` **恒成立**（27,205/27,205），`== original`
也有 99.67%。`ref_size` 想编码的信息，`quantity` 早就携带了。
（`ref_size` 唯一明显更强的是 partial CANCEL，但它只占 76/31,630 = 0.24% 的权重。）

### 3.3 `quantity` 从"输出字段"升级为"解析硬约束"

这不是词表变更，是**解析语义**变更，零 token 成本：

```
可行集 = { o : o.side == msg.side, o.price == msg.price,
           o.remaining == msg.quantity   (DELETE)
           o.remaining >= msg.quantity   (EXECUTION / partial CANCEL) }
```

实测：目标订单满足约束 **100%**（DELETE），干扰订单只有 **20.57%** 通过。
"唯一但选错" = **0**，"可行集为空" = **0**。

---

## 4. `ref_age_ns` 的编码：成本分析决定了必须条件编码

### 4.1 实测分布（27,789 个可见引用）

| 分位 | ref_age (ns) |
|---|---:|
| p1 | 10,490 |
| p10 | 99,011 |
| p25 | 705,823 |
| **p50** | **24,927,765**（约 25 ms） |
| p75 | 209,801,956 |
| p90 | 684,312,338 |
| p99 | 2,523,704,503（约 2.5 s） |
| max | 11,177,041,731（约 11.2 s） |

base-1024 位数分布：**2 位 7,759 条 / 3 位 18,399 条 / 4 位 1,631 条**。

### 4.2 HEAD 路径对 `ref_age` 基本无效（实测）

| HEAD 预算 | 覆盖率（只需 1 token 的引用） |
|---:|---:|
| top-16 | 0.1188% |
| top-256 | 1.3279% |
| top-1024 | 4.0915% |
| top-4096 | **15.1463%** |

**27,789 个样本里有 27,676 个互不相同的值**。纳秒时间差几乎不重复，所以频率表这条路走不通——
这和 `dt` 的纳秒余数（0–999，1000 槽全覆盖、永远单 token）完全不同。

### 4.3 无条件编码的成本（实测）

| `short_width` | 平均 token/条 | 分布 |
|---:|---:|---|
| 16 | 3.760 | 2:542, 3:7217, 4:18399, 5:1631 |
| 256 | 3.628 | 2:4206, 3:3553, 4:18399, 5:1631 |
| **1024** | **3.500** | 2:7759, 4:18399, 5:1631 |

**每条 touch 要多花约 3.5 个 token。** touch 约占消息流的一半（31,630 条 touch /
255 序列 / 250 条 continuation ≈ 0.496），所以整条流的平均消息长度会涨约 **+1.74 token**。

这是**不可接受**的：R6 每条消息均值约 6 token，涨到约 7.7 是 **+29%**；
在固定 13,000 token 上下文下，可装消息数从约 2,167 降到约 1,680（**−22.5%**）。
条件窗口变浅会直接压低"目标可见"的比例，也就是压低 R7 自己的天花板。

### 4.4 定稿：条件编码（conditional emission）

**只在约束层无法唯一确定时才编码 `ref_age_ns`。**

```
编码侧：
  算出可行集（side, price, quantity 约束）
  |可行集| == 1  →  不发 ref_age（0 token）
  |可行集| >  1  →  发 ref_age_ns

解码侧：
  用同样的约束算可行集
  |可行集| == 1  →  不读 ref_age，直接定案
  |可行集| >  1  →  读 ref_age，在可行集内最近邻
```

**成本降为 `0.1417 × 3.5 ≈ 0.50 token/touch`**，整条流平均只涨约 **+0.25 token（+4%）**，
比无条件编码省 **7 倍**。

**代价与风险**：编码器与解码器必须对"可行集大小"给出**逐位相同**的判断，否则 token 流
会错位（解码器多读或少读一个字段）。这是一个真实的 train-inference 一致性风险。

缓解：R6 已有同类依赖（`ref` 的同价档排名同样依赖簿状态），所以这不是新引入的耦合类别。
但必须加一道**编码器/解码器可行集一致性测试**作为 roundtrip 闸门的一部分（见 §6）。

---

## 5. 窗口外引用怎么编（对抗审查 D8，必须回答）

**12.14%（3,841 / 31,630）的 touch 目标早于 condition 窗口。** 这些引用：

- `ref` 走**负数 raw 命名空间**（`raw_ref_code`），R6 已有机制，**R7 继承不改**。
- `ref_age_ns` **不发**：解码侧根本没有这笔订单的 `created_at_ns`，发了也无法使用。
- `quantity` 约束仍然适用（簿里有这笔订单的 remaining，只是没有真实 id）。

> **不得引入 `OUTSIDE_WINDOW` 哨兵**。哨兵会让原始 order_id 无法还原，
> 直接违反"100% 精确 encode/decode roundtrip"这道自设闸门——设计层面就过不去。

---

## 6. 无损性闸门（继承 R6，不放宽）

R7 的词表在进入训练前必须通过：

1. **精确 roundtrip**：语料上 encode∘decode = 恒等，100%，一条不漏。
2. **canonical 拒绝**：对每个能走短路径的值，构造其长形式，解码器必须**拒绝**。
3. **可行集一致性**（R7 新增）：编码侧与解码侧在同一状态下算出的可行集大小逐条相同，
   否则条件编码会让 token 流错位。
4. **窗口外可还原**：负数 raw 码 → order_id 的往返恒等。

任一不通过，不进训练。

---

## 7. 与 R5 / R6 的对比

| | R5 (v5) | R6 | **R7** |
|---|---|---|---|
| 词表规模 | 15,847 | 待确认 | R6 + `ref_age` 字段的 head/short/len 段 |
| `ref` 语义 | 全局 NEW 排名 | 同价档排名 + 负数 raw | **不变（同 R6）** |
| 引用解析 | 全局排名查表 | `(side, price, ref_n)` 查簿 | **约束优先 + ref_age 消歧** |
| `quantity` 的角色 | 输出 | 输出 | **输出 + 硬约束** |
| touch 额外 token | 0 | 0 | **约 +0.50（条件编码）** |
| 可见目标唯一识别 | — | 63.8562%（side+price） | **85.8253% 零成本 → 100% 加 ref_age** |

---

## 8. 待定 / 需要构建时确定

| 项 | 状态 |
|---|---|
| `head_sizes` / `short_widths` / `max_digits` 的具体预算 | **待定**：由 SP500 2022–2025 语料频次统计生成；`ref_age` 的 HEAD 建议给最小预算（实测覆盖率极低） |
| `ref_age` 的 `max_digits` | 实测最大值 11,177,041,731 需 4 位 base-1024；建议给 **5** 位留余量 |
| `ref_age` 的 `short_width` | 建议 **1024**（平均 3.500 tok，比 16 省 0.26 tok/条）；但 SHORT 段本身要占 1024 个 ID，需与词表预算权衡 |
| R6 词表 JSON 实体路径 | **未找到**。worktree 的 `artifacts/vocab_R6/` 不存在，`tasks/` 下也未命中；构建 R7 词表前需要先定位或重建 R6 词表 |
| τ（最近邻距离阈值） | **不在设计阶段写死**。对抗审查 D4 指出 τ 必须由训练后的 `ref_age` 残差经验分布定，且可能应该用相对门限而非绝对 ns |

---

## 9. 数据来源

| 数字 | 脚本 | 结果文件 |
|---|---|---|
| `ref_age` 分布 / token 成本 / HEAD 覆盖 | `code/ref_age_token_cost.py` | 本文 §4（stdout） |
| `quantity` 约束唯一识别率 | `code/quantity_constraint_probe.py` | `results/quantity_constraint.json` |
| `ref_size` 边际收益 | 同上 + `code/three_channel_oracle.py` | `results/three_channel_oracle.json` |
| 约束层与 ref_age 的分工 | `code/constrained_resolver.py` | `results/constrained_resolver.json` |

源码常量全部来自
`/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/varlen-R7-constrained-20260812/src/varlen_R6/lossless_R6.py`
（worktree 基于 R6.2 的 `d519014`）。
