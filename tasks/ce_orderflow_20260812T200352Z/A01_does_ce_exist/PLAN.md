# A01　order flow 上到底有没有 compound error

写于 UTC 2026-08-12T20:10Z，**在跑任何东西之前**。判据一旦写定，跑完不改（G1）。

---

## 1. 问题

> 在 sigma-0 预训练生成器**自由生成**的 order flow 上，五个 action 字段
> （`event_type`、`direction`、`price_rel`、`size`、`log10_dt`）里，
> **哪些字段的误差会随生成位置 $m$ 累积**？

注意本分析**只测预训练 draft 这一条臂**。「DFM 能不能压低它」是 A02 的事。
把两件事分开的理由：如果 draft 上根本没有 compound error，A02 的问题就不成立
（没有东西可降），此时该做的是 A03（换更长的 horizon）而不是 A02。

## 2. 判据（YES / NO 的定义，写死）

对每个字段、每个 horizon：

$$\text{excess}_m \;=\; D_m^{\rm draft}-D_m^{\rm floor}$$

**判 YES（该字段存在 compound error）当且仅当：**

> `slope(excess_m)` 的**日块 bootstrap 95% CI 完全在 0 以上**。

**判 NO 分两种，必须区分：**

| 判定 | 条件 | 含义 |
|---|---|---|
| `NO-detected` | CI 含 0，**且**本设置的检出阈值 $\delta^\star$ 小于该字段斜率的量级 | 真的没有 |
| `NO-power` | CI 含 0，**但** $\delta^\star$ 大于等于该字段斜率的量级 | 分辨不出，不能说没有 |

$\delta^\star$ 由 G4 的注入实验给出，**不是事后估的**。

**辅助但不参与判定**的两个量（G6，分开报）：
- `level` = mean(excess_m) —— 误差有多大
- `z(m)` = $(\mathrm{mean}_{\rm gen}(m)-\mathrm{mean}_{\rm true}(m))/\mathrm{std}_{\rm true}(m)$ 及其斜率 —— 中心漂移

$z$ 必须单独测：$D_m$ 用**真值的矩**做标准化，模型整体平移后形状仍匹配，
$D_m$ 对「形状对、中心错」几乎不敏感。

## 3. 估计量（G2）

按交易日分层，把序列劈成不相交的两半 $A,B$（固定种子）。对每个格点 $m$：

$$D_m^{\rm draft}=\tfrac12\big[\mathrm{KL}(P^{\rm true}_A\|P^{\rm draft}_B)+\mathrm{KL}(P^{\rm true}_B\|P^{\rm draft}_A)\big] \Big/ H_m^{\rm true}$$
$$D_m^{\rm floor}=\tfrac12\big[\mathrm{KL}(P^{\rm true}_A\|P^{\rm true}_B)+\mathrm{KL}(P^{\rm true}_B\|P^{\rm true}_A)\big] \Big/ H_m^{\rm true}$$

- 两侧都是 $n/2$ 对 $n/2$，KL 的有限样本偏差**按构造相同**，在 excess 里精确抵消。
- model 侧两边**不共享序列**，避免经验直方图联动压低膨胀。
- 分箱：**真值分位数** 40 箱 + 2 个越界箱 + 1 个 **ILLEGAL 箱**（$K=43$）。
  真值在自己的分位箱上按构造均匀 ⇒ $H^{\rm true}=\log 40$ 恰好 ⇒ 任何偏离都是模型误差。
- 类别字段（`event_type` 4 类、`direction` 2 类）用类别本身做箱，外加越界箱与 ILLEGAL 箱。
- 非法定义：`event_type ∉ {1,2,3,4}` 或 `direction ∉ {−1,1}` 或 `size < 1` 或 `price ≤ 0` 或非有限。
  **进 ILLEGAL 桶，不 drop** —— drop 等于把「模型输出了非法东西」这个失败模式从指标里抹掉。

**窗口与格点**：位置 $m$ 处取 $[m-w, m+w]$ 合池，$w=40$；
**格点只取全宽窗口区** $m\in[w,\;R-w-1]$（G5：右边界截断会让样本量下降 → KL 虚高 → 末端翘起 → 凭空造出正斜率）。
截断版同时算一份放进 `out/` 做对比，但**不参与判定**。

**拟合起点** `FIT_FROM = w`（即第一个全宽格点），事先声明，防止起点附近的近原子分布造假尖峰把 OLS 符号带反。

## 4. 重抽（G3）

- 单位 = **交易日**，有放回，200 次。
- 先把 `counts[arm, half, day, grid, bin]` 存下来（分箱计数是精确充分统计量），
  重抽退化成 day 轴加权求和，精确且便宜。
- **所有臂共享同一个 `draws` 矩阵** ⇒ excess 是逐副本配对的差。

## 5. 对照与伪影排除清单（G5，每条独立）

| # | 对照 | 排除什么 | 预期（写在跑之前） |
|---|---|---|---|
| **C1** | **floor 臂**（true_A vs true_B） | KL 正偏差、真值自身非平稳、样本量随 $m$ 的变化 | floor 的斜率应 ≈ 0；若显著非零，**整个测量作废**，先修管线 |
| **C2** | 全宽窗口 vs 截断窗口 | 右边界样本量下降造成的末端翘起 | 截断版斜率应 ≥ 全宽版；差多少要量出来 |
| **C3** | **打乱位置**：把 draft 沿 $m$ 逐序列随机置换 | 「随 $m$ 上升」是否真的来自位置结构 | 打乱后斜率应落回 floor 水平；否则说明斜率来自别处 |
| **C4** | **恒定偏移注入**：gen = true 但整体替换 $\delta_0$ 比例的消息为随机合法值 | 检验管线对恒定缺陷是否正确给出**零斜率** | level 上升、slope ≈ 0 |
| **C5** | **线性 drift 注入**：替换比例随 $m$ 线性增长到 $\delta$ | **标定功效**，给出检出阈值 $\delta^\star$ | 单调可检出；记录最小可检 $\delta$ |
| **C6** | **no-op canary**：gen 直接置为 true 本身 | 管线自洽性硬断言 | excess 必须逐格点 $\equiv 0$（浮点容差 1e-12），否则非零退出 |

C6 不通过 ⇒ 立即停止，不报任何数字。

## 6. 规模

四个 horizon 全跑：$n_{\rm gen} \in \{500, 1000, 1500, 3500\}$
（外推 2× / 3× / 4× / 8×，$n_{\rm seq}$ = 64/64/64/32，20 个交易日）。

**horizon 必须一起看**：训练窗口只有 500 条消息，在 2× 处测不到 compound error
只说明外推不够远，不说明模型没有。

## 7. 产物

```
A01_does_ce_exist/
  PLAN.md                    ← 本文件
  code/a01_ce_existence.py   ← 自包含，不 import 旧代码
  code/a01_selfcheck.py      ← C4/C5/C6 合成自检
  out/a01_g{500,1000,1500,3500}.json
  out/a01_selfcheck.json
  figs/
  REPORT.md
```

**不 import 旧的 `action_compound_error.py`**：那份代码没有 floor、没有功效标定、
格点包含截断窗口。复用它等于把它的三个缺陷继承进来，且日后改动会静默改变本结论（G7 的独立性要求）。
