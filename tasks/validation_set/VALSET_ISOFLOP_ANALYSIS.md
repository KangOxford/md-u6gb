# 在 valset_v1 上的 IsoFLOP 分析：最优模型规模的算力标度

*基于 436 点全轨迹验证集交叉熵 · 2026-08-01*

---

## 执行摘要

把 33 条训练链的验证集交叉熵按固定算力切片重新组织，可以直接读出"给定算力预算下最优的模型规模"。点估计是 **N\* ∝ C^0.46**，与 Chinchilla 的 0.5 同量级。

但这个数字的不确定度很大，而且大到足以改变结论的性质：对训练链做 bootstrap 重采样，**指数的 95% 置信区间是 [0.12, 0.56]**（标准差 0.135）。也就是说，现有数据能确认指数为正、量级在 0.5 以下，但**无法把它定到小数点后一位**。区间之所以这么宽，根源是一个可以精确指出的结构问题：每个算力切片的抛物线谷底，左侧只由 1 到 3 条训练链支撑，右侧却有 7 到 12 条；而左侧那几条恰恰是决定顶点位置的关键。这个不对称来自小模型早期 checkpoint 在训练时已被删除，无法通过再评测弥补（见第 4 节）。

需要特别说明的是，"不同拟合方法给出 0.43 到 0.49"这个区间只反映方法选择的敏感性，**不是不确定度**，两者相差近一个数量级。本文把三种敏感性（方法、采样密度、抽样）分别量化，并给出诚实的报告口径。

## 1. 数据基础

分析建立在 436 个测量点上，每个点是一个训练 checkpoint 在冻结验证集（`valset_v1`，30,720 样本）上的完整交叉熵。这些点来自三批评测：

| Source | Points | What it covers |
|---|---|---|
| Terminal window (evaluated earlier) | 132 | the last checkpoints of each chain, where models are near-converged |
| Early backfill (evaluated 2026-07-31) | 124 | early checkpoints, D_tokens down to 2.8e8, where models are still heavily undertrained |
| Densification (evaluated 2026-08-01) | 180 | every remaining on-disk checkpoint, tightening within-chain sampling |
| **Total** | **436** | 33 chains, 12 model sizes (2.63M to 293M params) |

补评的意义在于把曲线的欠训一侧填了出来。以 78M 为例，此前只有 3 个点、全部集中在训练末段；补评加加密之后有 63 个点，覆盖从 step 170 到收敛的完整轨迹。验证集交叉熵的动态范围也随之扩展到 0.6003 至 2.5119。加密评测把磁盘上所有剩余 checkpoint 都评了，逐尺寸点数如下：

| Size | 0p2M | 1M | 4M | 6M | 10M | 14M | 23M | 46M | 78M | 120M | 200M | 350M |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Points | 11 | 12 | 26 | 49 | 55 | 54 | 42 | 45 | 63 | 45 | 18 | 16 |

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

逐切片结果如下（436 点数据；N\* 单位为百万参数，under 为顶点穿底幅度，单位 nats）：

| C | L_min obs | full N* / L* / under | window N* / L* / under | weight N* / L* / under |
|---|---|---|---|---|
| 3.93e17 | 0.9133 | no minimum | no minimum | no minimum |
| 9.31e17 | 0.6790 | 14.7 / 0.6435 / +0.0355 ✗ | insufficient points | 21.6 / 0.6783 / +0.0007 (curvature flag) |
| 2.21e18 | 0.6448 | 19.8 / 0.5383 / **+0.1064 ✗** | 20.4 / 0.6409 / +0.0039 ✓ | 19.2 / 0.6337 / +0.0110 ✓ |
| 5.23e18 | 0.6264 | 33.1 / 0.6168 / +0.0096 ✓ | 33.1 / 0.6168 / +0.0096 ✓ | 32.4 / 0.6198 / +0.0067 ✓ |
| 1.24e19 | 0.6112 | 43.7 / 0.6113 / −0.0001 ✓ | 43.7 / 0.6113 / −0.0001 ✓ | 43.9 / 0.6113 / −0.0000 ✓ |
| 2.94e19 | 0.6017 | 70.0 / 0.6009 / +0.0008 ✓ | 70.0 / 0.6009 / +0.0008 ✓ | 70.7 / 0.6008 / +0.0009 ✓ |

窗口法的行为正是稳健方法应有的样子：在数据健康的三个高算力切片上，它给出的 N\* 与全点法**完全相同**（33.1M、43.7M、70.0M，一位小数都不差），因为那里所有点本就落在谷底邻域内；只在唯一病态的切片上起作用，把穿底从 0.1064 压到 0.0039，L\* 从 0.5383 回到 0.6409，单调性随之恢复。

由此得到三组标度指数：

| Mode | Slope over all valid slices | Slope over slices passing acceptance | Slices passing |
|---|---|---|---|
| full | 0.4534 (n=5) | 0.4342 (n=3) | 3 of 5 |
| **window** | **0.4618 (n=4)** | **0.4618 (n=4)** | **4 of 4** |
| weight | 0.3710 (n=5) | 0.4887 (n=4) | 4 of 5 |

窗口法是三者中唯一自洽的：它的所有切片都通过验收，因此"全部"与"通过验收"两个口径给出同一个数，不需要事后剔除任何切片。全点法必须剔除两个切片才能自洽，剔除前后相差 0.019。加权法对被标记切片最敏感，两个口径相差 0.12。

### 3.1 对采样密度的敏感性

把轨迹从 256 点加密到 436 点（即把磁盘上所有剩余 checkpoint 都评出来）后，三种方法的指数一致上移：

| Mode (acceptance-passing slices) | 256 points | 436 points | Shift |
|---|---|---|---|
| full | 0.4030 | 0.4342 | +0.031 |
| window | 0.4425 | 0.4618 | +0.019 |
| weight | 0.4691 | 0.4887 | +0.020 |

位移来自链内插值精度的提升：加密前每条链用 7 至 10 个点跨两个数量级算力做 log 线性插值，谷底附近的损失被系统性高估，顶点因而偏左。加密后最高算力切片的 N\* 从 66.3M 移到 70.0M，其余切片基本不动。方法之间的跨度从 0.066 收窄到 0.055。

加密的最后 4 个点（10M 链的中段）对全部结果没有任何影响，三种方法的指数与置信区间逐位相同，说明相对于这批存档，分析已经收敛。但需要如实指出的是，从 256 点到 436 点这一整段加密使指数移动了约 0.02，说明估计对采样密度本身尚未收敛。因此报告时应给出区间而非四位小数，并注明该区间不含"继续加密可能带来的进一步位移"这一项。病态切片则完全不受加密影响（全点法穿底仍是 0.1064），再次印证那是结构性的臂不平衡，不是采样问题。

### 3.2 抽样不确定度：指数其实没有被定住

方法敏感性和采样密度敏感性都不回答这个问题：换一批训练 run，还会得到同样的指数吗？训练链是自然的重采样单位，每条链是一次独立实验。对 33 条链做有放回重采样、逐次重跑整条流程（插值、窗口法定顶点、回归斜率），得到指数的抽样分布：

| Statistic | Value |
|---|---|
| Point estimate (full sample) | 0.4618 |
| Bootstrap mean ± sd (2,000 resamples) | 0.4325 ± 0.1348 |
| **Bootstrap 95% CI** | **[0.1216, 0.5573]** |
| Jackknife max shift (leave one chain out) | 0.0360 (chain 4M-s5) |
| Jackknife median shift | 0.0023 |

留一法与 bootstrap 的结论看似矛盾：去掉任何单条链，指数最多只动 0.036，中位数只动 0.002，非常稳；但有放回重采样却给出 ±0.135 的标准差。这个矛盾正好指出问题所在。留一法每次只丢一条链，同尺寸的其它种子会顶上；bootstrap 一次可能丢掉某个尺寸的全部链，而那正是致命的，因为各切片的左臂构成极其单薄：

| C | N* | Chains forming the left arm | Chains forming the right arm |
|---|---|---|---|
| 2.21e18 | 20.4M | 3 (all of one size, 4M) | 12 |
| 5.23e18 | 33.1M | 3 (10M, 14M, 23M, one seed each) | 11 |
| 1.24e19 | 43.7M | **1 (a single 23M chain)** | 11 |
| 2.94e19 | 70.0M | 2 (both 46M) | 7 |

在 C = 1.24e19 处，整条左臂只有一条链。抛物线顶点的位置由左右两臂共同约束，右臂再厚也无法弥补左臂的缺失，因此只要重采样没抽中那一条，该切片的顶点就失去左侧约束、位置大幅漂移，斜率随之剧烈变化。

这个左右不对称不是偶然，它是第 4 节所述数据缺口的直接后果：小模型的早期 checkpoint 已被删除，能在低算力处提供左臂的小模型点本就所剩无几。换句话说，宽置信区间与病态切片是同一个原因的两种表现，一个体现在单个切片的顶点穿底上，一个体现在整体指数的不确定度上。

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

磁盘上确实还有 180 个从未评过的 checkpoint（已评完 176 个，余 4 个 10M 中段点待补），但逐链核对后确认它们**全部落在各链已有算力区间之内或之上**，没有任何一条链的 C_min 会因此下降。它们的价值是把链内采样加密（见 3.1 节的实测效果），降低 log(算力) 线性插值的误差，但不改变 bracketing 结构，因此病态切片的穿底幅度在加密前后一字未变。

因此，最低的两个算力切片（3.93e17 与 9.31e17）应当在报告中明确排除，而不是用任何方法去"抢救"。

## 5. 推荐的报告口径

综合以上，建议的写法是：**最优模型规模随算力增长的指数点估计为 0.46，对训练链做 bootstrap 得到的 95% 置信区间为 [0.12, 0.56]**。三种敏感性应分别交代，因为它们量级差了近一个数量级，混为一谈会误导读者：

| Source of variation | Magnitude | What it means |
|---|---|---|
| Fitting method (full / window / weighted) | 0.43 to 0.49 | how the vertex is located within a slice |
| Trajectory sampling density (256 to 432 points) | +0.02 | not yet converged; denser sampling would still move it |
| **Chain resampling (bootstrap)** | **± 0.135** | **the dominant term: would another set of runs agree?** |

同时应说明：低于 2e18 的算力切片因小模型早期 checkpoint 缺失而不可靠，已排除；顶点验收采用"穿底不超过 10 倍测量噪声"与"曲率不跳变"两条判据。

由此得出的实质结论应当谨慎表述。数据支持"最优规模随算力单调增长、指数为正且小于 1"，也与 Chinchilla 式的 0.5 相容，但**不足以把指数定到小数点后一位**，更不足以断言它显著低于 0.5。把 0.46 当作一个精确测量来引用，会给出远超数据支撑的确定感；而报出置信区间，既诚实，也自然引出下一步该做什么（第 6 节）。

## 6. 若要把指数定住，需要做什么

置信区间的宽度直接由左臂链数决定，因此收窄它只有一条路：在低算力区增加小模型的数据点。评测已经无能为力（磁盘上的存档全部评完了），唯一的办法是**重新训练几个小模型并保存早期 checkpoint**。

具体地说，在 C = 2.21e18 这个切片上，目前左臂只有 4M 一个尺寸。若补上 6M、10M、14M 三个尺寸在该算力处的点，左臂将从 3 条链（1 个尺寸）变成 12 条链（4 个尺寸），顶点的左侧约束会大幅加强。这些模型原本都训练过，问题只是早期 checkpoint 被 `max_to_keep` 删了，所以重训只需跑到相应步数（6M 约 3 万步、10M 约 2.5 万步、14M 约 2 万步）即可停止，不必训到收敛，且必须把 checkpoint 保留策略改为全量保存。

在动手之前值得先做一次代价评估：这三个尺寸各三个种子共九个 run，即使只跑前 20% 的步数，也是一笔可观的算力。是否值得，取决于 rebuttal 是否真的需要一个定到小数点后一位的指数，还是"指数为正、与 0.5 相容"这一较弱但可靠的结论已经够用。这是一个需要人来拍板的取舍，不是技术问题。

## 7. 交付物

| Path | Content |
|---|---|
| `valset_eval/valset_ce_436_master_table.csv` | **primary** 436-point trajectory table (132 terminal + 124 backfill + 180 densify) |
| `valset_eval/valset_ce_436_fitready.csv` | fit-ready schema (C = 6ND, loss column carries validation CE) |
| `valset_eval/valset_isoflop_432_*_parabolas.png` / `_summary.png` | IsoFLOP parabola panels and valley summary (all-points fit) |
| `valset_eval/valset_isoflop_robust.py` | three-mode robust vertex estimation with acceptance criteria |
| `valset_eval/valset_isoflop_robust_436.json` | per-slice results and slopes for all three modes |
| `valset_eval/valset_isoflop_bootstrap.py` | chain-level bootstrap of the scaling exponent |
| `valset_eval/valset_isoflop_bootstrap_436.json` | bootstrap distribution, sd and 95% CI |
| `valset_eval/valset_ce_256_master_table_20260731T161800Z.csv` | pre-densification snapshot, retained for the 3.1 sensitivity comparison |
| `valset_eval/valset_isoflop_robust_256.json` | pre-densification robust results |
| `valset_eval/manifest_densify180.json` | densification manifest; all 180 evaluated |

---

*分析：2026-07-31 起，432 点口径定稿于 2026-08-01。评测运行于 Isambard-AI GH200 节点，验证集为冻结的 `valset_v1`（构造与零泄漏证明见 `VALSET_V1_REPORT.md`）。*
