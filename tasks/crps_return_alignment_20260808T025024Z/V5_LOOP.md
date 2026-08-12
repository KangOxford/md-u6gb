# V5 求解循环（滚动记录）

> 目标：按 `PROBLEM_V5.md` 模式 N 让后训练臂在 **Primary = P(r_250|C)**（raw fair CRPS + shape qL1 + tails）上
> 显著优于预训练基线，且 spread/depth/OFI/Validity 非劣效。
> 纪律：A1 改前改后 git commit；A6 长尾必报；A7 bootstrap 按天块；A8 结果落地即记；A9/A12 每阶段新开 Notion 子页。
> 本文件按时间顺序滚动追加，**每个结果在产生当刻写入**。

## 锚点（循环开始前已知，2026-08-11T21:30Z）

| 事实 | 数值 | 来源 |
|---|---|---|
| 当前最佳臂 cond_a035 | horizon qL1 −31.5% (z=−2.60)，sd 0.563→0.714，CE +0.084% | CURRENT_STATE §3 |
| cond_a035 否决项 | **spread KS +57.5%**（1-tick 占比 0.302 vs real 0.037） | CURRENT_STATE §3 |
| 重加权上限（R71 rejection oracle） | energy 0.0822→**0.000287（0.17× null）**，sd→0.929，qL1→0.0552，ESS 41.5%，支撑外真实质量仅 **0.25%** | rejection_oracle.json |
| 直接 CRPS-PG/温度/logit-bias | 全部无效或更差 | CURRENT_STATE §5 |
| 复现地板（member 间 CV） | qL1 cv ~9–15% @n=192 | replication_floor.json |
| 可用算力 | job **5980502** 4 节点 16 卡全空闲（~15.5h 余量），attach only；5980745 满载勿动 | gpu_status 2026-08-11T20:29Z |
| 离线训练素材 | hp_baseX_s9101{1,2,3}：n=2000 contexts × 3 seeds，**含 data_tokens**（生成 token 已保存） | data/ 盘点 |

## 阶段规划（滚动更新）

| 阶段 | 内容 | 状态 |
|---|---|---|
| S6 | **V5-Primary 记分板**：在 hp_{baseX,a035X,a025X}_s9101{1,2,3}（n=2000×3 seed）上按 h=250 计 raw fair CRPS / qL1 / sd / raw+std tails + real-real 地板 + 按天块 bootstrap；顺带 spread gate 快查 | ⏳ 进行中 |
| S7 | **Arm W：终点密度比 weighted-MLE 蒸馏**（V5 §0.2 I-projection 的训练化）：w=clip(p_real(r_250\|c)/p_model(r_250\|c))，self-normalized，对已存 rollout tokens 做加权 CE 蒸入 decoder kernel（KL 锚 b=0 还原），IPF 迭代 2–3 轮；attach 5980502 生成新 rollout 复评 | 待 S6 |
| S8 | **spread guard**：测 Arm W 的 spread KS/1-tick 占比；若塌缩→加 spread-visitation 约束（状态占据入权重或约束） | 待 S7 |
| S9 | 500+500 冻结协议 + 全闸门表 + 独立复现 → 判定 | 待 S8 |

**S7 为什么选终点密度比而不是继续 placement tilt**：R71 已证明该重加权把 energy 拉进零假设（0.17×），
且尾部支撑缺口仅 0.25%（A6 长尾风险低）；它直接作用于 V5 Primary 量 r_250（placement 是代理，代理已证明会
引发 spread 塌缩）。A7 bootstrap 风险点：分母 p_model 随更新漂移→每轮用当前策略重估（IPF）；
log-w 截断 + ESS 必报；权重按 **rollout 整条**给，不逐 token。

---

## S6 记录区（结果落地即写）

### S6 结果（2026-08-11T21:55Z，DEV 级：2 天 / 1500 contexts / 3 seeds，context-cluster bootstrap）

| 臂 | raw fair CRPS | ΔCRPS vs base | qL1 (shape) | ΔqL1 | sd/real | raw tail q95 | q99 | q99.8 | std tail z2/z3/z4 |
|---|---|---|---|---|---|---|---|---|---|
| base | 5.972e-05 | — | 0.1613 | — | 0.550 | **0.138** | 0.20 | 0.111 | 1.35/1.10/1.60 |
| a025 | 5.791e-05 | −3.0% (z=−1.87) | 0.1236 | −21.1% (z=−2.73) | 0.663 | 0.280 | 0.20 | 0.111 | 1.13/1.14/0.93 |
| a035 | 5.590e-05 | **−6.4% (z=−3.73)** | 0.1117 | **−30.2% (z=−4.32)** | 0.714 | 0.409 | 0.244 | **0.000** | 1.48/1.14/1.20 |

- **发现 1**：a035 的 shape 改善（−30%）传导到 Primary raw CRPS 只剩 −6.4%——raw CRPS 被 sd=0.714 的弥散缺口支配。对照 oracle 上限（重加权后 energy 0.17× null、sd 0.929）：**Arm W 在 raw 上的理论余量远大于 a035 已实现的 −6.4%**。
- **发现 2（A6 长尾）**：raw economic tail 是重灾区——base 在 real q95 外只有 13.8% 应有质量；a035 到 40.9% 但 **q99.8 外精确为零**（0/4500）。标准化尾部 ~1.0–1.5 正常 ⇒ 尾部缺口＝尺度缺口的尾部投影，不是形状问题。q99.8 行计数极小（real 3 例/gen 4500 draw），粒度 0.07，解释需谨慎。
- **发现 3（统计基建红旗）**：全部 hp 高功率集只覆盖 **2 个交易日**（2026-01-07/08）→ 天块推断不可用，本表全部 z 为 context-cluster（DEV 级）；2 天的 split-half 地板只有一种切法（qL1 floor 0.109，单值非分布）。**S9 终局判定必须换多日索引重新生成**。
- 剂量单调：a025 全面居于 base 与 a035 之间 ⇒ tilt 方向真实有效，不是噪声。
- 产物：`v5_primary_h250.json`；评分器 `run/mid_training/score_v5_primary.py`（commit d7da188 + 修正 commit）。

**S6 图**（`figs/`）：`v5s6_density.png`（合并律对比，log 密度）、`v5s6_tails.png`（raw 尾部生存曲线：base/a035 在 ~3×sd 断崖归零，real 平滑延伸过 q99.8——尾缺=支撑终止而非形状）、`v5s6_scoreboard.png`（Δ 汇总）。

## S7 记录区

- 02:24Z：dump 第三次挂载成功（登录侧 setsid nohup srun ×4 客户端，每节点 2 seed 并行）。前两次失败教训见 learnt_lessons L358（srun 客户端随会话死；step 内 setsid 被 cgroup 清场）。
- 03:0x–03:15Z：dump 故障三层剥洋葱（①detached 挂载不可见→前台复现否决；②残留 manifest 防覆盖；③BATCH=48 在 CE/plog 段 OOM=静默半成功）→ 修复：BATCH=8 + .done+plog 双条件闸门 + **tmux 会话 v5wdump 托管客户端**（会话三次死亡的最终解）。batch8 首轮 4 seed 落地（done+plog=24）；余 4 seed tmux 重跑中。
- 权重预览（4 seed/768 rollouts）：**ESS 59.5%、clip 0%、w∈[0.50,8.14] V 形轮廓**（中心 0.5× 反粘滞、尾部 2–8× 补尾）；完美蒸馏上限 sd 0.677→0.977、qL1 0.196→0.087（图 `figs/v5s7_weights.png`）。

### S6b 全臂记分板（2026-08-12T03:35Z，12 臂 × 3 seeds，n=2000，DEV 级）

| 臂 | ΔCRPS(raw) | ΔqL1(shape) | sd/real | 判读 |
|---|---|---|---|---|
| a035 | **−6.9% (z=−4.67)** | −27.1% | 0.714 | 现最佳 raw |
| a050 | −6.6% (z=−4.18) | −36.4% | ~0.76 | raw 与 a035 打平，shape 更好 |
| **cond_fix** | −5.9% (z=−3.24) | **−44.5%** | **0.818** | 被埋没的强臂：shape+scale 双最佳 |
| tilt_p0020 | −5.3% | −5.3% | 0.793 | scale 强 shape 弱 |
| a200 | −1.7% | **+57.5%**（毁形） | 0.852 | 过冲：sd 最高但 shape 崩 → raw 收益消失 |
| rnd2（随机方向对照） | −2.5% | **−43.4%** | 0.645 | ⚠️ 随机方向也能拿 shape 大改善——qL1 单判据可被运气攻破 |
| mrnd（匹配随机） | **+15.2%** | +4.7% | 0.560 | 伤害对照正常 |
| hist/soft_hist/soft_tf | −1.7/+2.5/−0.8% | ≈0/+6/−7% | 0.58–0.62 | 无效族 |

**核心解读**：decoder 级干预的 raw CRPS 封顶 ≈−7%（四个不同构造的臂聚在同一水位，剂量加倍只换 shape 不换 raw）——与 R73「twist 方向 decoder 只能表达 2%」互证：**表达力墙是 raw 上限的主人**。对照 oracle（重加权 energy −99.7%）：通往 Primary 大幅改善的路必须加**状态依赖容量**（S7b adapter），而非在 decoder 内换目标。S7a 加权 MLE 降级为机制对照（测 cos-vs-tilt 与是否破 −7% 顶）。产物：v5_primary_h250_all.json。
- 8-seed 正式权重（batch8 全量，n=1536 rollouts）：**ESS 50.4%、clip 0%、w∈[0.43,11.8]**；完美蒸馏上限 **sd 0.639→1.019、qL1 0.156→0.0404**（低于 2 天 split-half 地板 0.109）。产物 v5w_weights.{npz,json}。
- dump 语义判定：现有 plog 的第二半段=**teacher-forced 真实延续**（深窗口邻域距离 248 的反例定案），非 on-policy——「on-policy dump」工程子任务已下放 subagent（DUMP_GEN_LOGITS 二次调用同一 CE 块于 cond+生成序列，验收=label 与生成 tokens 深窗口精确匹配 >95%）。

**S6b 更正（03:45Z，Safety 面回填）**：cond_fix 的「shape+scale 双最佳」判定被 LOB-Bench 聚合面**推翻**——其聚合 KS 0.302 ≈ base 的 2 倍（spread 单项 0.484 vs base 0.197；最坏三项 limit_bid_order_depth/bid_cancellation_depth/limit_bid_order_ticks 各 +0.35~0.38）。完整损伤谱：spread KS base 0.197 → tilt 0.187（无损）→ a035 0.311（已知否决项，+0.114）→ a050 0.354 → fix 0.484。**结论不变但更尖锐**：placement-tilt 家族的 Pareto 前沿上 a035 已是损伤最小的有效点，无任何现有臂能过 V5 模式 N（Primary 改善 + 全特征非劣效）双门。S7 密度比臂的核心开放问题由此聚焦为：**按终点收益率整轨迹重加权（不推特定价格槽）能否以更小微结构损伤换取同等 Primary 增益**。新臂纪律：落地即跑 LOB-Bench 聚合+spread 单项（lobbench2 评分已有基建，CPU 即可）。
- 多日评估索引建成：`data/v5_eval_idx_multiday.txt`（**20 天 × ~100 ctx = 1984**，从原始 selftrain 工件文件名提取，seed=20260812）。S7 起臂评估切换到此索引（天块 bootstrap 在 DEV 期即有效），base/a035 同索引重生成作配对对照；S6 发现 3 的整改闭环。
- 用户指令确认 S7b 主线 = **全量微调 post-training**（load j5705912@69378，全参数）。训练器 `wmle_full_ft.py` 落库：目标 = Σw·CE(生成窗|自身前缀)/Σw（I-projection 蒸馏，self-forcing）交替 anchor_λ·CE(真实序列)（DFM 2B 教训使锚不可选）；数据 = dump 侧 SAVE_SEQ_ARRAYS 落盘的 (tokens,books) 数组（与被审计 CE 块逐位同源）；产物 = apply_step 布局的普通 checkpoint。S7a decoder-only 降级为同权重同目标的**容量对照**。

## S7b 记录区（全量微调主线，用户指令 09:0xZ 确认）

- 09:47Z **全产物 dump 8/8 落地**（books/tokens/plog/gplog/seq × 191ctx × 8seed；gen-dump 深窗验收 1.000；seq 数组与 CE 块逐位同源）。正式权重（最终 rollouts）：ESS ~50%、ceiling sd→0.981 / qL1→0.051。
- 09:50Z **全量微调启动**（tmux v5wft @nid010371）：`wmle_full_ft.py` ARM=wm_ft_a，lr=1e-5、3 epochs（432 gen 步 + 432 anchor 步交替）、anchor λ=1.0、每 150 步存档（allocation 到期最多损 150 步）。数据 144 train / 48 hold batches。目标汇报物（用户点名）：**return 分布是否学得更好** → 多日索引（20 天）上 wm_ft vs base 的 raw fair CRPS / qL1 / sd / 双尾 + 天块推断。
- 10:55–11:06Z **全量微调完成**（wm_ft_a：432 步 3 epochs 零中断；四次启动迭代修复 load_metadata/ck['model']/160GiB OOM→M=2 梯度累积；稳态 3.9s/it）。[final] hold gen −0.48%、hold real −10.2%（后者为共享 context 污染指标，真安全闸=评估期 ceFIX）。中间存档 step100/200/300/400 齐。曲线 `figs/v5s7b_ft_curve.png`。
- 11:07Z 评估生成全线飞行：base 多日 3 seed **已完成**（hp_mdbase_s9100{1,2,3}）；wm_ft 3 seed + a035 参照 1 seed 生成中（~11:45 落地）；评分即出**首个汇报结果**。

### 🎯 S7b 首个结果（2026-08-12T11:5xZ，20 天 × 1984 ctx × 3 seeds，真天块推断）

| 臂 | raw fair CRPS | ΔCRPS | qL1 | sd/real | raw tail q95/q99/q99.8 | std tail z2/z3/z4 |
|---|---|---|---|---|---|---|
| base（预训练） | 5.319e-05 | — | 0.1196 | 0.621 | 0.219/0.25/0.333 | 1.06/1.57/2.42 |
| **wm_ft_a（全量后训练）** | **5.045e-05** | **−5.1% (天块 z=−3.20)** | 0.1169 (−3.3%, ns) | **0.685** | **0.286**/0.20/**0.583** | 1.19/1.28/**1.83** |

- **回答目标问题「return 分布学得是不是更好」：是。** 20 天天块推断下 raw 条件律显著更近（−5.1%，z=−3.20）。
- **改善构成与 a035 互补**：wm_ft 全部来自尺度+尾部（sd 0.621→0.685；raw q95 尾质量 0.219→0.286；q99.8 0.333→0.583；std z4 过重尾 2.42→1.83 反而更正），shape 未动（qL1 ns）——terminal 密度比臂修的正是 S6 诊断的「弥散缺口+尾部断崖」，placement 臂修不到的部分。
- 剂量观察：432 步只吃到 tilt 的一小口（ceiling sd→0.98），**方向已证实、剂量可加**（更多 epochs/IPF 轮）。
- 待补安全面：ceFIX CE gate + LOB-Bench spread/21 特征（接棒 allocation 上跑）。图 `figs/v5s7b_first_result.png`。

### S7b 安全面结果（12:0xZ）：全绿

| 闸门 | base | wm_ft_a | 判读 |
|---|---|---|---|
| ceFIX CE（同日不同 contexts，24 批配对） | 0.560674 | **0.514440（−8.2%）** | AR 能力不仅无损反而改善（DFM 2B 是 +290% 的灾难对照） |
| LOB-Bench 聚合 KS / L1（mdwmft vs mdbase, seed91001） | 0.1295 / 0.2111 | **0.1210 / 0.1962（−6.6% / −7.1%）** | 聚合改善 |
| **spread KS**（a035 的否决项） | 0.1834 | **0.1805** | **无损**（a035 是 +0.114） |
| 21 特征恶化>0.01 计数 | — | **1/21**（仅 vol_per_min +0.013） | ask/bid volume 类反而 −0.03~−0.04 大幅改善 |

**结论**：wm_ft_a 是全任务**第一个 Primary 显著改善（raw CRPS −5.1%, z=−3.20）且 Safety 面近乎零代价**的臂——终点密度比+全参数+CE 锚的组合同时避开了 placement 家族的 spread 塌缩与 DFM 的 AR 损毁。注意口径：CE 与 LOB-Bench 均为同两日分布内、LOB-Bench 单 seed；多日 CE 与 3-seed 特征齐档在下轮补。

### S8 组合臂结果（12:0xZ，20 天 × 1984 × 3 seed）

| 臂 | ΔCRPS | qL1 | sd | raw tail q95/q99/q99.8 | std z2/z3/z4 |
|---|---|---|---|---|---|
| wmft | −5.1% (z=−3.21) | 0.117 | 0.685 | 0.286/0.20/0.583 | 1.19/1.28/1.83 |
| **combo（wmft+a035 tilt）** | −5.5% (z=−2.82) | 0.132 (+12% ns) | **0.845** | **0.562/0.483/1.333** | **1.00/0.90**/1.75 |

判读：①尺度叠加超线性（tilt 在 wm_ft 上给 +0.160 vs 在 base 上 +0.093）；②**尾部大胜**——raw 尾质量 2-4×、std z2/z3 近完美校准（A6 维度领跑臂）；③bulk shape 小退（qL1 +12% ns）致 raw CRPS 与 wmft 打平（CRPS 对 bulk 敏感尾部权重低）；④非 a200 式崩坏（a200 是 qL1 +57% 毁形）。成败悬于 spread（LB 评分中）。教训雏形：固定 tilt 叠加是「借 bulk 换 tail」，IPF 重估才是无损继续拉尺度的正道——两者互补而非互斥。

**S8 组合臂终判（13:1xZ）：Safety 否决。** spread KS 0.1834→0.2905（+0.107，与 a035 在 base 上的 +0.114 几乎原封叠加——wm_ft 起点不吸收 tilt 损伤）；12/21 特征恶化>0.01；聚合 KS 0.1452 反超 base。**定格价值＝尾部可达性证据**（raw q95 0.562/q99 0.483 可达且 std z2/z3 近完美——尾部不是结构不可达，是方式问题）。正道确认＝IPF。
- 13:1xZ **IPF r2 权重落地**：wm_ft_a 自身 rollouts 上 ESS **65.4%**（r1 50%）、w∈[0.52,5.04]（更温和）、sd_before 0.684 与多日评估 0.685 跨集合复现；ceiling sd→0.983 / qL1→0.040 依旧。**IPF 收敛信号正确**（分布靠近→权重收敛于 1）。
- 13:2xZ **wm_ft_b 训练启动**（IPF r2：--ckpt wm_ft_a 续训，weights=v5w2，train 93000-3 / hold 93004-5，同超参 3 epochs）。tmux 在本节点失效（/tmp 不可写、TMUX_TMPDIR 不生效）→ 改阻塞式后台 srun（孤儿存活模式）。

### wm_ft_b 首启 OOM 事故与迁移（13:1x–13:3xZ）

**症状**：[init] hold 评估通过（weighted-genCE 0.59771 与权重审计吻合）后，第一个 `step_gen` backward 申请 51.07GiB 即 RESOURCE_EXHAUSTED。round-1 同几何（micro=2，实测峰值 ≈30 常驻 + 51 单张量 ≈ 81GiB < 0.92×95.6 = 88GiB 池）能跑通，本应无事。

**根因＝节点被同门实验占满，且时间线是「夹击」**：nid010272 全 4 卡各 88.2/97.9 GiB，被 4 个非本任务 PID 持有；其落位时刻恰在我 [init]（前向小池）之后、第一个 backward（大申请）之前——邻居的预分配把物理余量压到 ~9.6GiB。进一步排查：`ldm-sft`（5992007.115，12 GPU）占 5992007 其余 3 节点 → **5992007 四节点全满**。

| 5992008 节点 | GPU0/1/2/3 已用 (GiB) | 判定 |
|---|---|---|
| nid011165 | 19.1 / 31.6 / 88.2 / 88.2 | 满 |
| nid011166 | 40.0 / 31.7 / 88.2 / 88.2 | 满 |
| **nid011167** | 26.9 / 40.2 / **0.003 / 0.003** | **GPU2/3 物理空** |

**处置**（闸门按卡判不按节点判）：迁移到 5992008/nid011167，`CUDA_VISIBLE_DEVICES=2` 钉死空卡，不碰邻居两张忙卡；参数不变（micro=2 无需缩，51GiB 申请在空卡 88GiB 池内自然成立）。**教训**：物理 GPU 闸门必须在 launch 前一刻实探（srun nvidia-smi 逐卡），stale 的 gtop 快照在多会话共用 chain 分配的环境下几分钟就失效。

### wm_ft_b（IPF r2）训练完成（14:1xZ）：hold 加权目标全程未降

288 步（96 批 × 3 epochs）无故障跑完，3.9–6.3 s/it，saves @100/200/final → `ckpt/wm_ft_b/69378`。

| 量 | init | step100 | step200 | final(288) |
|---|---|---|---|---|
| hold weighted-genCE | 0.59771 | 0.60345 | 0.60491 | **0.60540 (+1.29%)** |
| hold realCE | 0.53549 | 0.52838 | 0.52359 | **0.51976 (−2.94%)** |
| train weighted-genCE | — | 0.580 | 0.646 | ~0.59（批间波动大） |

**过程判读**：train 加权 CE 大降（最低 0.456）而 hold 平台微升 → r2 加权信号在 train seeds 上被记忆、跨 seed **不迁移**；净方向被 λ=1.0 的 real 锚主导（realCE 单边 −2.9%，注意此值与 train 共享 contexts、有污染，真闸门是 ceFIX）。与「IPF 一轮即近不动点」假设一致（r2 权重已温和：w∈[0.52,5.04]，ESS 65%）。**终审在生成分布**：mdwmftb 三 seed（91001 GPU2 / 91002 GPU3 并行，91003 接续）vs base/wmft 记分板 + 安全闸门。若 sd/尾部无进展 → 结论「IPF 收敛于 r1，剩余缺口换旋钮（λ↓/lr↑）」；若小进展且安全全绿 → 剂量仍可加。

### wm_ft_b 安全面先行落地（14:5xZ）：CE 持平 wm_ft_a，LOB-Bench 聚合再改善，spread 进入监视区

| 闸门 | base | wm_ft_a | wm_ft_b | 判读 |
|---|---|---|---|---|
| ceFIX CE（192 契机 24 批配对） | 0.560674 | 0.514440 | **0.514661** | vs wmft 配对 t=+1.05（不可分辨）；vs base t=−59.9（−8.2% 全额保留） |
| LOB-Bench ks21 / l1_21（s91001） | 0.1295 / 0.2111 | 0.1210 / 0.1962 | **0.1158 / 0.1779** | 聚合 −10.6% / −15.7%，两轮单调改善 |
| 时间特征 | inter_arrival 0.0610 / time_to_cancel 0.1929 | 0.0567 / 0.1944 | **0.0269 / 0.1441** | r2 最大惊喜：时间律大改善（−56% / −25%）|
| OFI 族 | 0.103–0.146 | 0.088–0.125 | **0.080–0.112** | 全线再进 |
| **spread** | 0.1834 | 0.1805 | **0.1913 (+0.0079)** | 低于 0.01 阈但方向转差（wmft 是无损）→ 监视项 |
| 恶化>0.01 计数 | — | 1/21 | **1/21**（仅 vol_per_min） | 与 wmft 同 |

**注**：ceFIX 持平 = 参数在可泛化方向上几乎没动（与 hold-gen 平台互证）；但 LOB-Bench 时间/OFI 特征的实质改善说明 r2 并非零效应——动的是**生成分布的时间结构**而非 AR 条件律。Primary（sd/尾部）等 91003 齐后终判。

### S10 IPF r2 Primary 终判（15:1xZ，20 天 × 1984 × 3 seed）：分布面大动，「CE 不动点」是错觉

`v5_primary_multiday_r3.json`，图 `figs/v5s10_ipf_r2.png`。

| 指标 | base | wm_ft_a (r1) | **wm_ft_b (r2)** | 天花板 |
|---|---|---|---|---|
| raw CRPS | 5.319e-5 | **5.045e-5（−5.1%, z=−3.21）** | 5.140e-5（−3.4%, z=−1.89） | — |
| **qL1 shape** | 0.1196 | 0.1169（ns） | **0.0687（−40.5%, z=−5.40）** | 0.040 |
| sd_ratio | 0.621 | 0.685 | **0.728** | 0.983 |
| raw tail q95/q99/q99.8 | 0.219/0.25/0.333 | 0.286/0.20/0.583 | **0.300/0.317/0.75** | →1 |
| std z2/z3/z4 | 1.06/1.57/2.42 | 1.19/1.28/1.83 | **0.93/1.07/1.75** | →1 |

**判读**：
1. **hold-CE 平台误导了过程判断**：teacher-forced CE（开环、real 前缀）平台 ≠ 生成分布（闭环、误差复合）停滞。r2 在 CE 一位不动（配对 t=1.05）的同时把 shape 砍掉 40%、sd 再爬 0.043、z2/z3 校准近完美（0.93/1.07）——**全任务第一个显著 shape 改善**（此前所有臂 qL1 全 ns），也是 V5 §0.2「NLL 与闭环律解离」命题的最强实证。
2. **代价**：raw CRPS 点估计从 −5.1% 回吐到 −3.4%（z=−1.89，掉出 2σ）。CRPS 权重集中在 bulk 位置；shape/scale 拉开的同时 bulk 位置微移。**wm_ft_a 与 wm_ft_b 构成两点 frontier**：CRPS 最优 vs 分布最优。
3. 安全面（前节）：CE −8.2% 保留、LB 聚合再进（ks21 0.1158）、spread +0.0079 监视中、1/21 恶化。
4. **剂量-响应两轮汇总**：sd +0.064/+0.043（递减但实），qL1 0/−0.048（r2 突现），spread −0.003/+0.011（转差）。**下一步 = IPF r3**（wmftb rollouts 重估权重 → wm_ft_c），观察 sd 是否继续爬、spread 是否破 0.01 闸——spread 破闸即停在 frontier。

### S11 IPF r3 启动（15:2x–15:4xZ）

- **v5w3 dump**：6 seeds（94000-94005）双卡并行 22 分钟完成（7.5 min/seed，热 farm + JIT 缓存比 r2 快 ~3×），全部过完备性闸门（24 shards × plog/gplog/seq）。
- **v5w3 权重审计**（wm_ft_b 自身 rollouts，n=1152）：ESS **69.0%**（50.4→65.4→69.0，边际 +15pp→+3.6pp 递减）；**w∈[0.085, 8.14] 首次出现双向修正**——min 0.085 说明模型已在某些收益率区域**过度**产出被强降权（此前两轮全是欠覆盖上调）；ceiling sd→0.979 / qL1→0.0436 稳定；clip 0.26%。
- **wm_ft_c 训练启动**：--ckpt wm_ft_b + v5w3，train 94000-3 / hold 94004-5，同超参（lr 1e-5, λ=1.0, 3 epochs, micro=2），nid011167 GPU2（launch 前逐卡实探：GPU2/3 均 2 MiB 空）。预注册停机准则不变：spread 破 0.01 或 sd 增量 <0.02 即停在 frontier。

### S11 IPF r3 终判（16:1xZ）：停机条件触发，frontier 定格三臂

训练 288 步无故障（hold gen 平台 +1.15% 与 r2 同型）。`v5_primary_multiday_r4.json`，图 `figs/v5s11_dose_response.png`。

| 指标 | base | r1 wm_ft_a | r2 wm_ft_b | **r3 wm_ft_c** |
|---|---|---|---|---|
| raw CRPS Δ | — | **−5.1% (z=−3.15)** | −3.4% (z=−1.96) | −1.1% (z=−0.81, ns) |
| qL1 shape | 0.1196 | 0.1169 | 0.0687 | **0.0653 (−42.9%, z=−5.77)** |
| sd_ratio | 0.621 | 0.685 | **0.728** | 0.694 **（回退 −0.034）** |
| raw q95/q99/q99.8 | 0.22/0.25/0.33 | 0.29/0.20/0.58 | **0.30/0.32/0.75** | 0.26/0.18/0.33（尾质量回吐至 base 级） |
| std z2/z3/z4 | 1.06/1.57/2.42 | 1.19/1.28/1.83 | 0.93/1.07/1.75 | **0.99/1.000/1.42（z3 精确 1）** |
| ceFIX CE | 0.5607 | 0.5144 | 0.5147 | 0.5166（vs a 配对 t=+6.39 **首次显著恶化**，绝对量 +0.4%） |
| LOB-Bench ks21 | 0.1295 | 0.1210 | 0.1158 | **0.1092（三轮单调）** |
| spread KS | 0.1834 | 0.1805 | 0.1913 | **0.1729（反超 base，r2 之忧收回）** |
| inter_arrival / time_to_cancel | 0.061/0.193 | 0.057/0.194 | 0.027/0.144 | **0.015/0.069（−76%/−64%）** |

**停机判定**：sd 增量 −0.034 < 0.02 → **STOP**（预注册准则）。r3 的双向权重（w_min 0.085 压制项）过度矫正：raw 尾质量与 sd 回吐、CRPS 衰减出显著区、CE 首次显著蚀损（虽小），同时把标准化校准推到精确（z3=1.000）、把微观结构（spread/时间律）推到全场最优——**「量纲内自洽、量纲外收缩」**。

**Frontier 定格（全部安全绿）**：
- **wm_ft_a** = CRPS 最优臂（−5.1%, z=−3.2）
- **wm_ft_b** = 分布+尾部最优臂（qL1 −39%、sd 0.728、raw q99.8 0.75、z2/z3 0.93/1.07）——**对「return 分布学得更好」问题的头号答案**
- **wm_ft_c** = 微观结构+校准最优臂（LOB-Bench 全面最优含 spread、z3 精确校准；代价 raw 尾与 CRPS）

**机制注**：三轮曲线呈单峰（sd/尾部在 r2 见顶），IPF 不动点在分布意义上位于 r2–r3 之间；r3 的压制项与 real-CE 锚正面冲突的预测（S11 启动注）被 ceFIX t=+6.39 证实。若未来要更逼近天花板（sd 0.98），旋钮是 λ↓ 或非对称 clip（只上调不下压），而非第四轮同构 IPF。

### S12 S9 级冻结复现：可行性勘察（16:4x–17:0xZ）

目标 = 十条中第 10 条（冻结复现：不相交 stock-day + 不同 seed）应用于 frontier 的**改善主张**（1–4 条全套）。注意诚实定位：wm_ft_b 不冲全套 Solved（sd 0.728 距等价区域远，权重天花板本身 0.98），复现的对象是「显著改善」而非「已对齐」。

**数据勘察结果**：
| 复现轴 | 可行性 | 依据 |
|---|---|---|
| 不同 seed（同 20 天） | ✅ 立即可行 | 95001-3 重生成 base+wmftb 即可 |
| 日不相交（GOOG 2026-02） | ❌ 被数据挡住 | 消息库有 shard_2026-02，但宽账本库（recon_2026-05）只有 GOOG 2026-01——Feb 的 500 档重建不存在 |
| 换票（同月不同 stock，如 NVDA/MSFT 2026-01） | ⚠️ 待验证 | 宽账本库有全 SP500 2026-01；合法性取决于 base（j5705912）训练票池是否含该票（SP500 多票训练则 in-distribution）；需查 wandb b30675li 的数据配置；且 collect_rollouts.sh 硬编码 GOOG（55-57 行）需参数化 |

**算力现状**：两条 chain 共 32 卡全部被同门实验占用（5992008 四节点 86GiB 满档；5992007 三节点 89GiB + nid010272 中等占用）。不挤占。已挂登录侧 sacct 监视器（零 job step）盯同门 step 集合变化，空卡出现即起 seed-复现生成（base+wmftb × 95001-2 双卡并行 ~1.5h）。LOB-Bench 3-seed 补档（8 个缺口）CPU 顺序跑中（nid011167，不占卡）。

### S12 执行记录（17:1x–17:4xZ）

- **复现票 NVDA→MSFT**：宽账本库 483 票有 npy 分片但 NVDA/AAPL/TSLA 缺席；MSFT（可选票中最大市值）替换。`v5_repl_idx_msft.txt` 建成（1000 索引，dataset 枚举 + seed 20260812 均匀采样；builder 提交 `aa9f22e`）。
- **sbatch 双轨**：32 卡持续满占 1.5h+ 触发 A11 hard 条款 → **Job 6000743**（`v5-s9-repl-msft`，1N4G 2.5h，base+wm_ft_b × seeds 95001-3，每 run 独占一卡两波；幂等 .done 跳过使 attach/sbatch 双轨竞速安全）。发现并修正：会话用户实为 **kangli.u6gb / account brics.u6gb**（skill 模板里的 kangli.s5e 过时）。监控 6 档在跑，/moveon 状态已存。
- **LOB-Bench 3-seed 补档完成**（8 缺口全填）：seed 方差可忽略（ks21 极差 ≤0.0014），单 seed 结论全部在 seed-平均口径下成立——base/wmft/wmftb/wmftc 的 ks21 均值 0.1291/0.1211/0.1164/**0.1095** 单调；spread 均值 0.1848/0.1797/0.1919/**0.1725**（wmftb +0.007 阈内、wmftc 反超 base 确认）。安全面对齐缺口关闭。
