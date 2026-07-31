# 在 valset_v1 上的 IsoFLOP 分析：最优模型规模的算力标度

*基于 256 点全轨迹验证集交叉熵 · 2026-07-31*

---

## 执行摘要

把 33 条训练链的验证集交叉熵按固定算力切片重新组织，可以直接读出"给定算力预算下最优的模型规模"。核心结论是 **N\* ∝ C^0.44**，即算力每翻十倍，最优参数量增长约 2.8 倍。这个指数与 Chinchilla 的 0.5 同量级但略低，意味着在本任务上，把算力更多地投入数据而非参数，比通用语言建模更划算一些。

这个数字必须带着两个条件读。第一，它依赖于用哪种方法从每个算力切片里定位最优点，不同方法给出 0.40 到 0.47 的范围。第二，最低的两个算力切片存在无法修复的数据缺口，任何方法在那里都不可靠。本文把这两点量化清楚，并给出推荐的报告口径。

## 1. 数据基础

分析建立在 256 个测量点上，每个点是一个训练 checkpoint 在冻结验证集（`valset_v1`，30,720 样本）上的完整交叉熵。这些点来自两批评测：

| Source | Points | What it covers |
|---|---|---|
| Terminal window (evaluated earlier) | 132 | the last checkpoints of each chain, where models are near-converged |
| Early backfill (evaluated 2026-07-31) | 124 | early checkpoints, D_tokens down to 2.8e8, where models are still heavily undertrained |
| **Total** | **256** | 33 chains, 12 model sizes (2.63M to 293M params) |

补评的意义在于把曲线的欠训一侧填了出来。以 78M 和 120M 为例，此前各自只有 3 个点，全部集中在训练末段；补评后各有 21 个点，覆盖从 step 170 到收敛的完整轨迹。验证集交叉熵的动态范围也随之从窄区间扩展到 0.6003 至 2.5119。

每条链内部按 log(算力) 做线性插值，即可读出该链在任意算力处的损失。算力口径为解析式 `C = 6ND`（N 为参数量，D 为已消费 token 数），全表统一，因此切片之间可比。

## 2. 方法与一个必须处理的失败模式

IsoFLOP 分析的标准做法是：固定一个算力 C，把所有能达到该算力的链在该点的损失取出来，得到一组 (N, L) 点，然后拟合抛物线，抛物线的顶点就是该算力下的最优规模 N\*。对多个 C 重复，再对 (log C, log N\*) 做回归，斜率即标度指数。

直接套用这个流程会踩一个坑。在低算力切片上，能达到该算力的链绝大多数是大模型（大模型参数多，即使只训了很少 token 也能积累可观算力），而这些大模型此时严重欠训、损失极高。抛物线拟合是最小二乘，这些高损失点主导了拟合，把曲率拉高，顶点被推到所有实测点之下：

| C | Lowest measured L in slice | Fitted vertex L* (all-points) | Undershoot |
|---|---|---|---|
| 2.21e18 | 0.6448 | 0.5384 | **0.1064** |

穿底 0.1064 nats 是单点测量噪声（95% 置信区间半宽约 0.002 nats）的 53 倍。这不是测量误差，是拟合外推的产物：顶点落在了没有任何数据的位置。后果直接可见——该切片的 L\* 比更高算力切片的 L\* 还低，而"算力更多、可达损失更低"是硬性物理约束，违反它说明拟合失效。

值得强调的是，此前脚本里的 bracketed 判据（要求顶点左右都有数据点）**没能拦住这个失败**：该切片顶点左侧有 3 个点、右侧有 12 个点，形式上通过了判据，实际却穿底。臂数平衡不等于拟合可信。

## 3. 稳健化：三种顶点估计的对照

针对上述失败模式，实现了三种估计方式并逐切片对照。判据是顶点穿底幅度不得超过 0.02 nats（10 倍测量噪声），以及曲率不得跳变到相邻切片中位数的 5 倍以上。

| Mode | Definition |
|---|---|
| full | quadratic least squares on all points in the slice (the original approach) |
| window | quadratic fit using only points with L ≤ L_min + 0.15 nats (valley neighbourhood) |
| weight | quadratic fit on all points, weighted by exp(−(L − L_min)/0.10) |

逐切片结果如下（N\* 单位为百万参数，under 为顶点穿底幅度，单位 nats）：

| C | L_min obs | full N* / L* / under | window N* / L* / under | weight N* / L* / under |
|---|---|---|---|---|
| 3.93e17 | 0.9133 | no minimum | no minimum | no minimum |
| 9.31e17 | 0.6790 | 14.7 / 0.6435 / +0.0355 ✗ | insufficient points | 21.6 / 0.6783 / +0.0007 (curvature flag) |
| 2.21e18 | 0.6448 | 19.8 / 0.5384 / **+0.1064 ✗** | 20.3 / 0.6410 / +0.0038 ✓ | 19.2 / 0.6338 / +0.0109 ✓ |
| 5.23e18 | 0.6268 | 33.1 / 0.6170 / +0.0098 ✓ | 33.1 / 0.6170 / +0.0098 ✓ | 32.4 / 0.6199 / +0.0069 ✓ |
| 1.24e19 | 0.6113 | 43.5 / 0.6113 / −0.0000 ✓ | 43.5 / 0.6113 / −0.0000 ✓ | 43.7 / 0.6113 / +0.0000 ✓ |
| 2.94e19 | 0.6017 | 66.3 / 0.6016 / +0.0002 ✓ | 66.3 / 0.6016 / +0.0002 ✓ | 66.8 / 0.6015 / +0.0002 ✓ |

窗口法的行为正是稳健方法应有的样子：在数据健康的三个高算力切片上，它给出的 N\* 与全点法**完全相同**（33.1M、43.5M、66.3M，一位小数都不差），因为那里所有点本就落在谷底邻域内；只在唯一病态的切片上起作用，把穿底从 0.1064 压到 0.0038，L\* 从 0.5384 回到 0.6410，单调性随之恢复。

由此得到三组标度指数：

| Mode | Slope over all valid slices | Slope over slices passing acceptance | Slices passing |
|---|---|---|---|
| full | 0.4402 (n=5) | 0.4030 (n=3) | 3 of 5 |
| **window** | **0.4425 (n=4)** | **0.4425 (n=4)** | **4 of 4** |
| weight | 0.3576 (n=5) | 0.4691 (n=4) | 4 of 5 |

窗口法是三者中唯一自洽的：它的所有切片都通过验收，因此"全部"与"通过验收"两个口径给出同一个数，不需要事后剔除任何切片。全点法必须剔除两个切片才能自洽，剔除前后相差 0.037（0.4402 对 0.4030）。加权法对被标记切片最敏感，两个口径相差 0.11。

## 4. 一个无法修复的数据缺口

自然的补救思路是给低算力切片补充小模型的早期数据点，让左臂变厚。这条路走不通，原因是那些 checkpoint 已经不存在。

逐链枚举磁盘上实际保存的 checkpoint 步数，发现各链最早保存点的分布极不对称：

| Chain | Earliest checkpoint on disk | Resulting C_min |
|---|---|---|
| 78M / 120M / 350M | step 170 to 340 | 1.3e17 to 7e17 |
| 4M | step 37,670 | 2.1e18 |
| 10M | step 53,250 | 5.9e18 |
| 6M | step 65,440 | 5.2e18 |

训练时的 `max_to_keep` 轮转删除了小模型的早期 checkpoint，而大模型的恰好保留了下来。这个不对称正是低算力切片只有右臂的根源：在 C = 2.21e18 处，唯一能提供左臂的小模型压根没有那么早的存档。

磁盘上确实还有 180 个从未评过的 checkpoint，已在评测中，但逐链核对后确认它们**全部落在各链已有算力区间之内或之上**，没有任何一条链的 C_min 会因此下降。它们的价值是把链内采样从 7 至 10 个点加密到 15 至 30 个，从而降低 log(算力) 线性插值的误差，间接提升顶点估计精度，但不改变 bracketing 结构。

因此，最低的两个算力切片（3.93e17 与 9.31e17）应当在报告中明确排除，而不是用任何方法去"抢救"。

## 5. 推荐的报告口径

综合以上，建议按下面的方式报告标度指数，而不是给出单一数字：

主值取窗口法的 **0.44**（四个通过验收的切片，无需事后剔除）。稳健性区间取三种方法在各自自洽口径下的跨度，即 **0.40 至 0.47**。同时说明两点：低于 2e18 的算力切片因小模型早期 checkpoint 缺失而不可靠，已排除；顶点验收采用"穿底不超过 10 倍测量噪声"与"曲率不跳变"两条判据。

这样报告的好处是，读者能看到结论对方法选择的敏感程度，而这个敏感程度本身就是数据覆盖不足的诚实反映。掩盖它而只报一个四位小数的数字，会给出超出数据支撑的精确感。

## 6. 交付物

| Path | Content |
|---|---|
| `valset_eval/valset_ce_256_master_table_20260731T161800Z.csv` | 256-point trajectory table (132 terminal + 124 backfill) |
| `valset_eval/valset_ce_256_fitready.csv` | fit-ready schema (C = 6ND, loss column carries validation CE) |
| `valset_eval/valset_isoflop_256_*_parabolas.png` / `_summary.png` | IsoFLOP parabola panels and valley summary (all-points fit) |
| `valset_eval/valset_isoflop_robust.py` | three-mode robust vertex estimation with acceptance criteria |
| `valset_eval/valset_isoflop_robust_256.json` | per-slice results and slopes for all three modes |
| `valset_eval/manifest_densify180.json` | 180 unevaluated on-disk checkpoints (densification, in progress) |

---

*分析：2026-07-31。评测运行于 Isambard-AI GH200 节点，验证集为冻结的 `valset_v1`（构造与零泄漏证明见 `VALSET_V1_REPORT.md`）。*
