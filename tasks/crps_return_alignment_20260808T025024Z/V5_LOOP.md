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

### S12 终判：S9 冻结复现（MSFT 2026-01，18:5xZ）——shape 复现，scale 不迁移

用户指定 nid010384（锁 `crps-wm_ft`）四卡两波跑完 6 run（1000 契机 × 20 交易日 × 3 新 seed 95001-3，base 与 wm_ft_b 同索引配对；MSFT 从未参与任何训练/选型/调参决策，训练期止 2025-12-31）。`v5_repl_msft.json`，图 `figs/v5s12_replication_msft.png`。排队备份 6000743 因幂等 .done 将自动秒退。

| 指标 | MSFT base | MSFT wm_ft_b | GOOG 对照（base→wmftb） | 迁移判定 |
|---|---|---|---|---|
| **qL1 shape** | 0.2468 | **0.1533（−31.9%, z=−2.67）** | 0.120→0.069（−40.5%, z=−5.4） | ✅ **复现** |
| raw CRPS | 6.124e-5 | 6.020e-5（−1.7%, z=−0.83） | −3.4%（z=−1.96） | ⚠️ 方向同、不显著 |
| sd_ratio | 0.767 | 0.678（**反向**） | 0.621→0.728（正向） | ❌ 不迁移 |
| raw q95/q99/q99.8 | 0.34/0.57/0.50 | 0.32/0.23/0.17 | 上升 | ❌ 不迁移 |
| std z2 | 0.802 | **1.045**（\|1−x\| 0.20→0.05） | 1.06→0.93 | ✅ 复现 |
| std z3/z4 | 0.96/0.67 | 1.19/0.47 | 改善 | ❌ 混合偏负 |

**补测（用户点名，01:0xZ）IC 与 direction accuracy**（3-seed 集合均值 vs 真实 r_250，天块 bootstrap）：GOOG——Pearson IC base +0.1246 → wmftb **+0.1421**（ΔIC +0.0175，z=+0.41 ns）、DA 53.83% → **55.16%**（+1.3pp，z=+1.02 ns）；MSFT——IC +0.1543 → +0.1626（z=+0.21）、DA 55.76% → 55.97%（z=+0.12）。四个 delta 全部同号为正但均不显著。**按设计**：加权 MLE 重塑的是条件律的宽度/形状/尾部而非均值位置（V5 明文「不是点预测收益率」）；结论 = 方向信息无损（可能微升），分布效应（5.5σ shape）不来自也不带来点预测能力变化。

**S13 全 horizon 轮廓（用户点名，01:3xZ）**：从既有 rollout 的逐消息 mid 路径直接提取（零重生成，`horizon_profile.py`，per-member 缓存），h ∈ {10,25,50,100,150,200,250}，GOOG 多日集：

| h | 10 | 25 | 50 | 100 | 150 | 200 | 250 |
|---|---|---|---|---|---|---|---|
| sd base→wmftb | 0.83→**0.91** | 0.78→**0.94** | 0.75→0.87 | 0.69→0.80 | 0.67→0.77 | 0.64→0.75 | 0.62→0.73 |
| qL1 base→wmftb | 0.131→0.075 | 0.158→**0.063** | 0.152→0.097 | 0.145→0.097 | 0.142→0.078 | 0.132→0.077 | 0.119→0.068 |
| t99 base→wmftb | 0.50→0.60 | 0.58→**0.85** | 0.50→**0.85** | 0.32→0.43 | 0.30→0.40 | 0.23→0.35 | 0.27→0.32 |

三个发现：**① 修复迁移到全部 horizon**，三指标七档全改善，短程零伤害——不是终点过拟合；**② 峰值在 h=25–50 而非训练的 h=250**（sd@25 0.94 近完美、t99@25-50 0.85）：终点重加权学到的是逐步机制（更真实的增量波动），短程先受益；**③ base 的弥散赤字随 h 累积增长（0.83→0.62，复合欠弥散），wmftb 把衰减斜率放平**——与 LOB-Bench 时间特征改善互证。产物 `v5_horizon_profile.json`、`figs/fig_midtrain_horizon.pdf`（已入 Overleaf）。注：点估计（DEV 级），推断口径仍以 h=250 的天块 z 为准。

### S14 大规模化启动（8 票混合中训练，02:1x–02:4xZ，用户指令 faed77bc）

**数据勘察定界**：500 档宽账本仅 2026-01（483 票）+ 2026-05 有；训练年份 2022-25 的月 shard 只有 10 档账本，rollout 生成需全深度起始账本做撮合回放 → **「4 年」被数据挡住**（需另跑账本重建管线，未启动）。canonical 八票中 NVDA/AAPL/TSLA 无宽账本 → 名单定为 **GOOG, MSFT, AMZN, META, AMD, INTC, NFLX, JPM**（2026-01 全齐）。

**设计**（全部提交 `94c0032`/`e15c4b9`/`7c8dc0f`）：
- 每票 600 train + 500 eval 契机，不相交；GOOG/MSFT 额外排除既有评估集（多日 1984、ceFIX 192、旧 train 191、S9 复现 1000）防污染。八票索引全建成（池 2.5 万~22.6 万窗）。
- Dump：seed 编码 `96000+票序×10+k`（k=0-3 train、4-5 hold），单前缀 v5m → 训练器零改动装载；48 run 摊 5998835 十六卡（每卡串行 3 个）。
- **逐票密度比权重**（S9 教训的直接修复：GOOG 估的比率把 MSFT sd 推反方向），每票自归一后合并单 npz（root 名消歧）。
- 训练：wm_ft_multi 单卡全参数（数据规模 17×，模型不变），lr 1e-5、λ=1.0、1-2 epochs（~2400 批/epoch ≈ 6.7h）；hold 探针改为跨池等距 16 批（原头部切片在多票池下只测第一个票）。

## S14 mid-train-scale（多票混合，用户 09:5xZ 点名启动）

**设计**：4 票（GOOG/AMZN/META/JPM 2026-01，各 ~200 契机 × 6 seeds 96000-5）各自估密度比权重 → 混合全参数训练 `wm_ft_mt`（base 起点，与单票 wm_ft_a 同构）；**MSFT 排除在训练外当判官**（单票版 sd 0.767→0.678 反向）。基建：dump 驱动 STOCK/IDX/NSEQ 参数化（`90e4825`+同门补 STOCK 直通）、训练器 --prefix/--weights 逗号配对（`090d58d`）、三新票索引（`855c0818`）。四票 dump 四卡并行 ~47 分钟完成，完备性闸门全过。

**每票权重普查（机制级发现，10:0xZ）**：

| 票 | ESS | sd_before | 蒸馏天花板 sd |
|---|---|---|---|
| GOOG | 61.1% | 0.663 | 0.971 |
| AMZN | 66.6% | 0.706 | 0.989 |
| META | 81.4% | 0.800 | 0.909 |
| **JPM** | **91.7%** | **0.890** | 1.013 |

**base 的欠弥散是票依赖的连续谱**（GOOG 重灾 → JPM 几乎无病），与 R1×SP500 时代「GOOG 曝光量是瓶颈」互证——这就是单票 scale 修正不迁移的病理本体：每票需要的剂量不同，单票权重把 GOOG 的剂量硬套给了 MSFT。per-ticker 权重正是对症药。10:1xZ `wm_ft_mt` 训练启动（~391 train batches × 3 epochs ≈ 1170 步，~2h，nid010272 GPU0）。判据：held-out MSFT 上 sd 是否翻正 + GOOG 不退步 + ceFIX/LB 安全。

**结论（十条第 10 条的诚实读法）**：wm_ft_b 的**shape 改善是真效应**——量纲无关的分位形状匹配在陌生票上以可比幅度复现（−40%→−32%，z 弱化主要因 n 半减），且 base 在 MSFT 上 shape 缺口更大（0.247 vs 0.120）时改善依旧成立；z2 校准同样复现。**scale/raw 尾部改善是 GOOG 特化**——同一组权重在 MSFT 上把 sd 推反方向（0.77→0.68）：终点密度比是在 GOOG 条件分布上估的，修的是「GOOG 的弥散缺口」，不是一个普适的「加宽旋钮」。跨票通用的 scale 修正需要 per-ticker 权重或多票混合训练（下一线索）。Mode-N 全套 Solved 不主张；「return 分布学得更好」的主张定格为：**shape+校准跨票成立，scale 修正票内成立**。

### S14 事故与恢复（03:0x–03:5xZ）：无锁扇出被 16-GPU 同门作业挤压

**事故链**：首轮扇出（5998835，launch 前用户 gtop 快照显示全空）落卡后，同门 16-GPU 作业（每节点一个 86GB 进程）抢先/同时预分配 → 我的 16 个 dump 被压进 ~10GB 残池：生成活着、CE pass 全灭（0/9）。清场过程三连挫：①pkill 模式自匹配杀掉探针；②计算节点 **ps-shim 只支持 -p 单查**，`ps -u` 全被拦 → 模式杀全部空转；③/proc cmdline 扫描第一版又自匹配自杀，第二版（运行时拼接字面量）被 auto-mode 分类器拦截（合理：字符串混淆+kill -9 形似恶意）。**转机**：同门作业退场后，幸存的 wave-1 进程池自动长到 88GB、CE 由败转胜（s96004：14 败→17 连胜）——**残局自愈**。

**新协议（已固化 `v5m_fanout.sh`，commit 内含）**：①逐卡 30 分钟物理等待门（不再信任任何快照）；②闸门审计 + 幂等补跑为唯一收敛机制，不再外科手术杀进程；③双写者由 collect_rollouts 的 manifest FileExistsError 天然仲裁（先到者赢，后到者该 seed 报错让位）。

**当前布局**：干净扇出跑在 **6000409**（16 卡，23h，逐卡门版）；5998835 幸存孤儿继续贡献（自然终结 ~40 分钟，监听中）；**6000412 锁定 nid010779 为训练+评估基地**（唯一配额锁随 5992007 到期从 010384 迁移；锁表被双注册表迁移复活的三条旧锁已按配额裁定清理）。

### S14 素材与评估 base 侧全落地（12:0x–12:4xZ）

- **48/48 dump 过闸**（pass-2 于 6000409 零失败收官；总账：chaos 期白捡 12 + pass-2 补 36）。
- **逐票权重全景**（n=1152/票，合并 28,800 条）：ESS GOOG 56%/MSFT 57%/AMZN 73%/META 84%/AMD 74%/INTC 53%/NFLX 71%/JPM 90%；**欠弥散排行：GOOG 0.574 最狠、MSFT 0.607 次之**（大盘科技系统性欠弥散），JPM 0.880 最接近真实；八票天花板 0.90–1.06 ⇒ 逐票修正理论上可把全员拉到 ~1。
- **wm_ft_multi 训练起跑**（nid010943 GPU3，从 base 起跑保持单剂量实验纯度）：2400 train / 1200 hold 批整装载，lr 1e-5、λ=1.0、1 epoch、save-every 200。
- **base 臂评估生成 16/16 完成**（8 票 × 2 seeds × 500 契机，6000409 一波扫完）。multi 臂等 checkpoint 后复用同脚本。

### S13 修正（用户点名方向图后，14:5xZ）：方向能力在中程 horizon 显著提升

此前「IC/DA 无显著变化」的结论是 **h=250 单点判定的假阴性**。全 horizon 扫描（GOOG 多日，3-seed 集合均值，天块 bootstrap）：

| h | ΔDA | z | ΔIC(×100) | z |
|---|---|---|---|---|
| 50 | **+4.05pp** | **+2.40** | **+4.7** | **+2.03** |
| 100 | **+4.17pp** | **+2.57** | +3.5 | +0.81 |
| 150 | +1.8pp | +0.97 | +2.6 | +0.51 |
| 250 | +1.3pp | +1.00 | +1.8 | +0.40 |

方向增益的 horizon 轮廓与分布修复峰值（h=25-100）重合——终点密度比训练把中程的条件分布修好之后，其集合均值的方向信息同步变准。h<25 两臂 DA 皆低于 50%（tick 量子化：符号被一档跳动主导而均值排序仍正确，IC 为正）。教训入库：**单点 horizon 的「无效应」不许外推，先扫轮廓再下结论**。图 `figs/fig_v5m_direction.pdf`；七图全部编入 `midtrain_figs.tex`（Overleaf 7501b74）。

### 方法卡片（用户四问定稿，15:0xZ）：On-policy 终点密度比蒸馏

**定义**：老师 = 自样本 × w(r)=p̂_real/p̂_θ（「终点被扭成真实分布的自己」），学生 = 同模型全参数；on-policy = 材料为当前策略自采 rollout；safe = real-CE 锚 + I-projection 最小改动本性 + 事后非劣效闸门。**无 return 预测头无 critic**：被训练的是 action（消息 token）分布，r 从终点 state（账本 mid）算出、只进权重。**马尔可夫链只在前向经历**：likelihood-ratio 梯度不反传穿引擎（不可微无碍）；全轨迹共享同一 w（目标是终点边缘分布，不做逐步信用分配）。**Sparse terminal reward**（每轨迹一标量，(w−1) 中心化 + log 截断 ±2.5 控方差），但效果 dense（序列级权重重塑逐步条件分布 → 全 horizon 改善 + 方向中程 +4pp）。三恒等式：REINFORCE 等价（绕过引擎）、I-projection（安全面的理论根源）、可计算天花板（每轮对上界报账）。Notion 方法卡片页 3bb12c45-68fd-81b5-86e0-ecb0acd5de72。

### S15 终判：wm_ft_multi 全管线闭环（训练→inference→benchmarking→scoring，16:5xZ）

**训练**：2400 步 1 epoch 零故障（futex 卡死一次换卡+线程帽复活；hold gen 全程下降 −0.95%，多票池独有）。**Inference**：8 票 × 500 契机 × 2 seeds × 双臂 = 32 run 全绿（GPU3 被 ram_repro 压制的 4 缺 rid 移卡补齐）。**Benchmarking**：LOB-Bench ks21 0.1397→0.1247（−10.7%）、**spread 0.196→0.181 反而改善**、1/21 恶化；ceFIX **0.513864 = 全任务历史最优**（base 0.5607、a 0.5144、b 0.5147）。**Scoring**（v5m_scoreboard.json，天块 z）：

| 票 | ΔqL1 | z | sd | 票 | ΔqL1 | z | sd |
|---|---|---|---|---|---|---|---|
| GOOG | −35.2% | −2.21 | 0.78→0.67 | AMD | −23.5% | −1.89 | 0.66→0.66 |
| MSFT | −39.5% | −2.13 | 0.50→0.48 | INTC | −0.8% | ns | 0.70→0.78 |
| AMZN | −19.6% | −1.29 | 0.83→0.79 | NFLX | −17.0% | −1.85 | 0.87→0.94 |
| META | −39.3% | −3.18 | 0.79→0.68 | JPM | −7.2% | ns | 0.87→0.78 |

**判读**：① **shape 普适化达成**——八票全降、四票 ≥1.85σ、混池标准化 qL1 −26%（0.138→0.102）：逐票权重把 shape 修复推广到了全横截面（S9 单票权重做不到的事）；② **sd 轴一剂未兑现**（混池 0.760→0.734 微降）——与单票弧线同构：r1 剂量先出 shape、sd 要第二剂（且矩阵图显示 GOOG h=100/250 raw 尾部大幅外扩=尾质量先于二阶矩恢复）；③ 安全面三绿创纪录。**自然下一剂 = multi-IPF r2**（从 wm_ft_multi 重采重估逐票权重再训一轮，对照单票 r2 出 sd 的先例）。产物：v5m_scoreboard.json、fig_v5m_matrix.png（全红版，Overleaf 84fb9d2）、ceFIX_wm_ft_multi、lobbench2/hp_v5me{b,m}_GOOG。注意口径：eval 集 n=500×2seed 比多日集噪；跨集比较无效，只读集内配对 delta。

## S16 (2026-08-13 21:55Z) 图册补上八票记分板图；surge 面板上线
- 新图 `figs/fig_v5m_scoreboard.pdf`（脚本 `code/make_v5m_scoreboard_fig.py`，数据 `v5m_scoreboard.json`）：左 = 每票 ΔqL1% 棒 + day-block z（≥1.85σ 实色四票，pooled 标准化 −25.9% 灰棒），右 = sd 前后哑铃对 1.0 虚线（pooled 0.760→0.734，INTC/NFLX 朝真走、GOOG/META/JPM 反向）。
- 已接入 midtrain_figs.tex（atlas 与 matrix 之间，图册现 16 张），tectonic 编译过，Overleaf push 7793b02。
- 核查结论：此前 15 张图的 PDF/PNG 资产全部已跟踪已推送（main == origin/main），本次唯一缺口就是记分板无图形版。
- 工程面板 https://midtrain-ledger-x7k4q9.surge.sh 部署成功（HTTP 200，16.7KB，含泳道图/前沿表/八票 ΔqL1 条/画廊，已脱敏）。

## S17 (2026-08-13 22:2xZ) multi-IPF r2 起跑（4 卡档）；用户所指 16 卡被 mink3 锁挡
- 用户贴出 6006424（ctx2k-base-alloc, 4N16G 全 idle）授意使用；但锁表显示 nid010667/670/672/715 全部有 claude-mink3 live 锁（"P4 scaling sweep 16-way, user authorised 2026-08-13"）。按「别人 lock 的不碰」+ R1 双查，未占用，交用户仲裁。gtop 显示 🔓 与锁表不一致，疑 gpu_locks.json / nodelocks.json 双注册表未合并显示。
- 改在自有节点 nid010943（6000412，剩 10h17m，锁表 crps-wm_ft live）起 r2 素材：`v5m2_fanout.sh`（槽数=4×节点数泛化版），48 run = 8票×6 seeds（98000+t*10+k），DUMP_PREFIX=v5m2，CKPT_OVERRIDE=ckpt/wm_ft_multi（69378 layout 复用）。4 槽×12 连跑，r1 实测 ~20min/run → dump ≈4h；全周期估 ~7.5h。
- nid010943 现驻：GPU0-2 各 743MB ram_repro SD3 进程（0% util）、GPU3 52MB spinner（100% util）+ JAX 探针，全为他会话实验，不杀；闸门按「锁表∧剩余显存≥需求」通过（94.9G 空闲）。GPU3 槽若拖速，折份额给 0-2（幂等）。
- 后续链：build weights（v5m2_w_*，逐票，分母=新素材）→ wmle_full_ft 从 wm_ft_multi 续训 → ckpt/wm_ft_multi2 → eval fanout（seeds 97201/97202, tag v5me2）→ scoreboard/matrix r2 版。
- 24 格账目（S16 后补算）：qL1 22/24 改善 vs sd 仅 10/24 朝 1；GOOG/META/JPM 全 horizon sd 反向；短 horizon 过冲（NFLX h25 qL1 +47%）——r2 的靶就是这三票的 sd 和短端过冲。

### S17a (2026-08-13 23:5xZ) GPU3 wedged，12 seed 折给 GPU0-2 长闸门救援
- slot3 首查即暴露：12 个 rid 全部秒退 exit=1，per-seed 日志根因 `jax.errors.JaxRuntimeError: INTERNAL: no supported devices found for platform CUDA`——GPU3 新 CUDA context 建立失败。卡上驻着 ram_repro 的 52MB 100% util spinner + JAX 探针；`nvidia-smi -i 3 -q`：Compute Mode Default、有 "Pending: Enabled" 挂起项。判定 wedged，root 权限的 reset 不可得，spinner 是他会话实验进程不杀。
- 处置：`v5m2_rescue.sh`——12 个搁浅 (ticker,seed) 平分三份挂到 GPU0-2，闸门 60s×300 次（5h 耐心），各卡主链结束显存回落 743MB 地板即自动接力；不与主链并卡，幂等可重投。
- 修订 ETA：dump 主链 ~03:45 完，救援 ~05:05 完 → 权重 ~05:25 → 训练 ~07:25 → 评估（16 run ÷ 3 卡）~09:10 → 记分 ~09:30；6000412 约 09:50 到点，余量 ~20min。若窗口不够：训练带 step 链 checkpoint、eval/dump 幂等，均可换分配续跑，不丢工作。教训归档：物理闸门（显存<1000MB）判不出 wedged 卡——显存空 ≠ context 可建，首查协议再次救场。

### S17b (2026-08-14 02:0xZ) 会话重启后接管；r2 训练 argv 钉死
- 会话重启带走后台驱动，但 16 条 v5m2-dump step 01:15 起以孤儿身份自行重启（srun 客户端树 setsid 存活，幂等跳过已完成 seed，尸体 seed 顺序覆写无撞车）；与 mink3-sweep16 同分配共存（它 88.2G/卡 + dump 足印 6.6G ≈ 94.8G < 95.6G）。哨兵 v1 的 15 分钟日志静默判据误报（单 run 30-40min 无日志输出属正常），v2 改用 squeue step 计数判活。
- 锁表机制已按用户令废除：tasks/node_status → node_status_deprecated_20260814；memory 已更新（feedback_node_lock_usage_policy）。
- **r2 训练输入（照抄 r1 argv 换代）**：`ARM=wm_ft_multi2 LR=1e-5 EPOCHS=1 LAMBDA=1.0 run_wmle_ft.sh --ckpt $T/ckpt/wm_ft_multi --step 69378 --prefix v5m2 --weights v5m2_weights.npz --train-seeds <每票k0-k3共32个98xxx> --hold-seeds <每票k4-k5共16个> --eval-every 100 --save-every 200 --micro 2`，跑法 = srun --overlap 单卡（需 ~空卡 + XLA_MEM_FRACTION 0.92）+ OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8（r1 futex 教训）+ --job-name=v5m2-train。权重构建 = build_v5m2_weights.sh（新写，CPU）。

### S17c (2026-08-14 10:2xZ) 七小时空转的根因 + 第三次重铺（这次验证过）
- 复盘：03:07 的"重启"发 send-keys 到当时不存在的 tmux 会话（claude-nid010463 是 10:0x 本次 resume 才创建的），命令根本没执行；我数到的 9 条 step 是垂死旧波。**违反了"起完 40 秒内验 step 在且在算"的军规**，代价 7h（素材停在 32/48，6006424 与 6000412 相继到期）。教训并入 stmux 记忆：send-keys 之后必须 capture-pane 验回显，或直接用 §4.0 的 setsid nohup。
- 10:15 检查点：论文数学节已推送（d6536b2：tilt 两推导、I-projection 链式分解、wMLE=蒸馏、REINFORCE、IPF 不动点、taxonomy、algorithm 浮动件）。
- 恢复：6007121（u6gb-4-node-chain，20h43m，nid010463/010485/011084/011090，16 卡全空，我就在 010463 上）`setsid nohup` 起 driver5，40 秒验证 16 条 v5m2-dump step 在跑 ✓。缺口 16 seed ≈ 2 轮 × 20min。哨兵 v4：48/48 后自动跑 build_v5m2_weights.sh。

### S17d (2026-08-14 11:1xZ) 9 seed 三层连环事故破案 + 修正重投
- 三个时代三个死因，全靠 member_0/inference.log 尸检定案：**01:15 波**＝跑到 73% 被 02:53 团灭外杀（无辜）；**10:22 波**＝manifest FileExistsError 撞车，疑似 tmux 里那条"以为没发出去"的 driver4 延迟点火与 driver5 成双驱动互踩；**10:57 波**＝rescue2 索引文件名手误 `v5m2_train_idx_`（正确为 r1 命名 `v5m_train_idx_`，票内索引与轮次无关）→ FileNotFoundError 秒死。
- 两个复发教训：① `run_v5w_dump.sh` 恒 exit=0（尾部 echo 吞状态），"[r2] exit=0" 九连全是假绿——退出码不可作产物证据，必须数 .done；② 我先验归因"thundering herd"是错的，真证据在 member 层日志，"同槽后续成功"只因后续是不同 seed。
- 11:1x 修正重投：确认零残留 step（单驱动）、尸体 .part5、sed 修 idx 路径、9 条错峰 step 40s 台阶起跑、42 秒验证 9/9 在列。哨兵 v6 值守，48/48 自动建权重。

### S18-勘误（2026-08-14 14:4x，对抗复核发现）
- S15 里 LOB-Bench 「1/21 恶化」表述不准：原始符号计数为 **8/21 恶化**（其中 7 个幅度 ≤+0.0033
  属种子噪声量级，超噪声的实质恶化才是 1/21 = bid_volume +0.0193）。聚合 −10.77% 与宽基性
  （13/21 改善、leave-one-out 稳）经独立重算确认。勘误由隔离批评者 agent 提出（V2 判决）。

### S18-勘误2（2026-08-14 15:0x，隔离批评者 V4）
- S9/S12 的「MSFT dispersion 被推反（0.77→0.68）」**不成立**：全部效应来自单一 context
  （id 95946，01-30）的两条爆炸 rollout（base 26.5σ / arm 18.1σ）；6000 点剔 2 点后两臂
  sd 0.593 vs 0.592 持平。正确结论 = **dispersion 修正不迁移（不动）**，非反向。z2 校准
  「复现」同为该伪影（day-boot z=−0.28）。shape 复现经全切法确认，稳健幅度 −17~−21%
  （头条 −32/−38% 为乐观端）。逐票权重的动机不受影响（非迁移本身已足够）。论文与图册
  措辞已更正。附带新现象：生成会出 20σ+ 爆炸轨迹，评测须报告单 context 敏感性。

## S19 (2026-08-14 15:0x) 对照训练收官 + 第二批对抗判决 + P4 架构落地

**训练**：wm_ft_multi2 全 2400 步完成，final ckpt 落地；hold real-CE 全程闸内（最终
−1.32%，方向为好）。注意事项：训练日志 13:59Z 冻在 step1600（stdout 传输死亡），但
step1800..2400 checkpoint 全部按时落盘——日志是传输不是自述，以产物裁决。

**D0 链 a**（8 ckpt × 1024 seqs）：selfcheck 精确 0；clip-binding 3.75%(s200) →
4.53%(s1400)，600 步后饱和 ~4.5%；token 中位 |logρ| ~7e-6；corr(Δseq,w) 全程负
（−0.09..−0.14）。预注册开臂条件（>10% 且集中高 w）两条均不满足。链 b（1800..2400）待跑。

**第二批批评者（V5/V6/V7）**，两条内部前提被驳倒：
- V5：GOOG 弧线三主张全 CONFIRMED；stop rule 有 git 时间戳证明真预注册；发现引用陷阱
  （round2=r3.json、round3=r4.json、r2.json 是被否决侧臂）；红旗：随机方向对照 rnd2 的
  qL1 也 −43%（DEV 集）——qL1 幅度非机制特异。
- V6：「ESS 50-69→80-90」与「ceiling 近完美」按原表述 REFUTED；真值：ESS 中位 72.1→76.9%
  非均匀；ceiling GOOG 0.873 / AMD 0.893（<0.9 = round two 的数学上界）、INTC 1.060 会
  过冲；clip 仅 0.083% 触发全在下界。
- V7：「NFLX h25 1.47 过冲」REFUTED（真值 1.09）；唯一 >1.1 的 GOOG h25 (1.29) 其 base
  本来就 1.18；单条 blow-up 承载 22-38% cell 方差，剔一条后两臂所有 h25 cell ≤1.0——
  **短视界过冲整体是爆炸轨迹伪影**，Arm G 动机除名；8 票 pooled base sd = 0.76
  （0.62 是 GOOG 单票集的，台账勘误）。

**架构落地**：RESULTS.md（去噪台账）+ BELIEF.md（信念文件 v1）+ A1/A2（两信任层成文）
+ dream Stop hook（ledger_repo/.claude/settings.json → code/dream_gate.sh）安装。

**评测**：v5me2 eval fanout wave 1（14 slots）在跑；slots 9/11 因邻居占卡（nid011084
GPU1/3）延后补跑。fanout 脚本新增 SKIP_SLOTS 参数（向后兼容）。

## S20 (2026-08-14 16:xx) D0/D1 终判、tokens-per-step 勘误、Arm B 三次发射

**D0 全链终判**（12 ckpt × 1024 seqs，双段 selfcheck 均精确 0）：binding 饱和 4.5-4.7%
（峰 4.72%@s1800），corr(Δ,w) 全程负。预注册开臂条件两分支均不满足 → **Arm P 关闭，
C1 无理，C2 本设置成立**（clip 对 ~95.5% 位置是 no-op）。

**D1 判决 + 口径勘误**：cos̄ = 0.068/0.189/0.461 @ micro 2/8/32，注册分支在 cos̄(2) 上
触发 → Arm B 开。随后发现 **micro 是梯度累积 chunk 不是 batch**：真实 1 step = 1 item
= 8 seqs = **104k tokens（K=4）**，预训练 gap 10× 非 40×；真实训练步画像是 cos̄(8)=0.189。
勘误红字入 plan Answers；C6 信念 H→M（症状仍在但更温和）。

**Arm B 三次发射教训链**：
1. v1 漏抄 --train-seeds（wmle 默认是 r1 老 seed）→ "no training items" exit 2 假绿身亡；
2. v1.5 补 seeds 但 --micro 16 在 chunk 语义下是 **no-op**——实际是 lr-only 变体，
   发现后 kill（driver kill 不停远端计算——「srun 客户端死计算继续」反向验证；
   经由 overlap step pkill 清理，pattern 匹配到自身 cmdline 自杀 exit 144 但目标已清）；
3. v2 用新参数 --group-items 4（洗牌后并 4 item/step = 32 seqs = 416k tokens，正好是
   D1 实测 cos̄=0.461 档；lr 2e-5=√4 缩放）在 GPU2 起 → **OOM 51 GiB**：GPU0/1/2 被
   三连号进程组（88G×3, 99% util，另一会话的三卡计算）接管，物理闸门没重查；
4. v3 挪到真空的 GPU3，存活。ckpt/日志假启动产物均已 trash/改名（never rm）。

**eval 进度**：8/16 .done（s9/s11 善终后 GPU 被上述三卡任务接管属正常交接）。

## S21 (2026-08-14 17:0x) 三臂主判决（partial+robust）+ D2 双版本 + 尾部画像反转

**R2 主判决**（PRELIMINARY，批评者 V8 在跑）：pooled std sd 0.760→0.734→0.792，7/8 票
向真实；§7 剔单轨迹敏感性通过（0.720/0.709/0.765，每票 r2≥r1）；shape 保持、hold-CE 更好。
预注册 fixed 线（≥0.80）差 0.008 → 落 partial 出口。出口重打分：Arm G 不开（V7 已杀其
前提），对症臂改为 E11——GOOG 全样本 0.868 精确顶到 V6 天花板 0.873（剔后 0.764 并报）。

**D2 v1→v2**：v1 的 scipy t.fit 把 ν 全钉在 1.99±0.00（方差存在边界的 MLE 病理）——
tail-LL 腿可信（t 完胜 KDE 8/8，MSFT 上 KDE 崩至 −27），ν 腿仪器报废。v2 三仪器交叉
（有界 profile likelihood + Hill + 按日标准化）：**ν-gap 7/8 显著且方向全体反转预设**——
日标准化后 real ν 8-50（真实重尾主要是日间波动混合），gen ν 3-6（**生成器日内条件尾
更肥** = blow-up 家族的定量化）。尾部失配 = 条件过肥 × 跨日状态调制不足，两因子正交。
E11 两腿门开，但权重设计改为「日标准化空间 t-ratio × 尺度通道分离」。
「generator too Gaussian」的旧工作假设被自家修好的仪器驳倒——如实记录。

**Arm B**：G=4 在真空卡上重现 51.07 GiB OOM（32-seq item 触发的 XLA 物化，谜题入账
不追）；G=2（16 seqs=208k tokens/step，恰为 E9 原注册量级）过首步在跑，lr 1.4e-5=√2。

## S22 (2026-08-14 17:4x-18:1x) — Arm B lands: shape-for-dispersion trade; bench extra-arm defect found+fixed
- Arm B train done (1200 steps, drift −1.54%); eval fanout 16/16 on NEW chain alloc (old one rotated out mid-day); two sentinel false alarms were MY monitoring bugs (setsid fork makes $! the dead parent → verify via squeue/artifacts; .done lives at member_0/.done — layout guessed, not read), compute itself clean both times.
- bench v1's first unregistered-arm use exposed a stage-4 hard-code: bench_v5me2b.json packaged the CONTROL board under B's tag (caught because pooled m2 matched control bit-for-bit). Fixed via EXTRA_ARMS (empty = bit-identical legacy); plus-file never overwrites source-of-record.
- Verdict computation (armB_verdict.py, §7 excision + joint day-block z): dispersion B beats control excision-ROBUSTLY (0.804 vs 0.765 excised, z +2.27; control's fullsample number was inflated by its own GOOG +12.9σ blow-up); shape gives back most of dose one (+21% qL1, misses 0.1122). AMZN sd 1.029/t998 5.50 = one −8.2σ rollout (excised 0.998). Neither arm passes both bars. V9 critic dispatched (isolated, told to rebuild the bootstrap itself).

## S22 — 2026-08-15 05:5x–06:1x consolidation window (while material + zk/ak run)

- PROTOCOL §8 pre-registered BEFORE the round-3 material audit: power-before-bars
  (V11's DOA lesson), μ̂-tilt gate (V10 channel + V11 measurement), repeat-offender
  context registry {149580} + paired drop-context view (V12), per-dose constructor
  license re-earn = E4 conjecture → executable rule, straddle/stale-count/blow-up-mass
  wording rules. Ablation arms upgraded to 4 seeds pre-launch (97503/4, 97603/4
  registered — B1.13: attribution is decision-critical).
- abl_verdict.py pre-written + compile-checked; ctx id "149580" format verified
  present in GOOG eval ids (registry match is string equality — checked, not assumed).
- BELIEF rewritten (cycle 08-15 #1): B1.12 (E11 verdict), B1.13 (seed-spread
  discipline) new; B1.2/B1.6/B1.10/B1.11 updated; C6 final; posteriors current.
- Plan: +2 Q→A pairs (E11 verdict, V12); master table E11 → FAILED; E4 erratum
  resolved pointer. STATE.md rewritten to the event-driven watch shape.
- A1: 3 additive UPDATEs (round-2 landed + ceiling refuted; blow-up family case #3/#4
  two-species; header scope note). Panel refreshed with Phase B band (four-arm 4-seed
  board, C1–C6 verdicts, E11 FAIL row, in-flight strip) + V2 erratum fix (8/21 raw);
  redeployed to surge (21.2 KB live).
- Ledger 72818f7 pushed (sigma0-midtrain); code 917f18b pushed (sigma-0 branch).
- Ops: allocations checked — all 40 dumps + 2 trains on 10–13h-remaining allocations;
  6007121 expires 06:57Z with nothing critical (4 stale bash steps die harmlessly).
  Watch healthy: material 39/80 at 05:55, zk/ak ETA ~07:4x.
- S22b 06:3x: material first wave 72/80 — slots 36-39's cards were occupied mid-run
  (zk/ak + DFM sweep took 6014307), busy-gate skipped by design. My first audit cried
  72/80 ALL-short: wrong filename layer (plog/gplog guess vs real data_*/ subdirs) —
  layout-is-not-a-data-source again; corrected audit: 72/72 clean vs v5m2 shape.
  Retry: 8 dumps re-launched on 6011842's idle cards (gtop-verified), 8/8 live at 40s.
  Monitor dedupe: ghost bwthk67uv stopped, single channel bl8jk4ee0.
- S22c 12:2x re-attach: 3h stall discovered (watch v1 + all steps dead). Failure
  census: retry4 = manifest leftover (3rd occurrence — the guard works, my
  clearing discipline lagged); zk eval wave1 8/8 = FUSE family, mixed with dumps
  on same nodes; ak wave1 8/8 SUCCESS on quiet nodes = the clean control that
  pins the trigger to same-node collect exits. Adopted layout rule: one node one
  class, batches start/end together. Cleared 16 partials, relaunched 28 steps
  (zk 16 isolated / ak 8 / dumps 4), night_watch2 + new Monitor armed.
- S23 15:1x-15:3x ROUND-3 LAUNCH CHAIN: material 80/80 green -> S8.4 audit
  (nu-gap 3/8, t license lapses, KDE by rule; dose-2 converged the gap 7/8->3/8)
  -> 48k KDE weights (ESS .73-.93, ceilings 0.91-1.05, GOOG/AMD sub-0.9 cap gone)
  -> seeds 97701-4 registered -> wm_ft_multi3 training live (4800/1200 split
  confirmed). Timeout expected ~3h40m; --start-step resume patched+committed
  (4f2c779). V13 process note: the constructor decision ran through the
  REGISTERED gate (material-side), not the eval-side nulls — category error
  caught by the critic before it entered the record.
- S24 22:5x DOSE-3 TRAINING COMPLETE (4800/4800). The auto-handoff "failed
  forward": my heredoc triple-escaping broke the GPU probe ([: $u -lt: unary
  operator expected → CUDA_VISIBLE_DEVICES=0,1,2,3), but JAX just used the
  first visible card and the resumed run trained 3450→4800 cleanly. final
  hold real CE 0.5517 (drift ≈ −1.9%, inside gate, improving direction);
  ckpt wm_ft_multi3/69378. Lesson (same family as "don't replicate env blocks
  from memory"): generated-script escaping is untestable by eye — probe-then-
  hardcode beats clever inline probing. Session died with nid010723 as
  predicted; reborn on nid010110. final_eval_chain launched (verdict 97701-4
  1-deep serial halves; trajectory s3000/3600; 8-seed reruns with self-clear).
- S25 23:4x-00:0x INODE QUOTA WALL + DOSE-3 PREVIEW WIN. 51.2M files hard cap
  hit (space only 112.8T/200T): every NEW file creation on Lustre refused —
  eval pipeline dead (6,600 csv per run), git dead, Edit tool dead (tmp
  files). Discovery: OVERWRITING existing files still works (no new inode) =>
  ledger appends resumed via python open("a"). Verdict computed on a tmpfs
  copy of the stack with np.savez monkeypatched to best-effort (module
  shadowing via sys.path failed twice — the monkeypatch at the numpy layer is
  the robust move). DOSE-3 PREVIEW (2-seed): ALL bars pass with margin — sd
  z +4.47, level clears 0.80 in BOTH computations (0.8895/0.8432, first arm
  ever), qL1 −39% z −2.55, toward-1 8/8, tilt corrective z −4.75, t998 →
  0.9375. FUSE root cause also closed: orphan squashfuse mounts accumulate
  (45 on nid011037, unremovable — pids alive/recycled), exhaust the mount
  quota, new setups fail (farm empty => assert len(message_files)>0); halves
  died, their exits freed quota, later halves lived. 011037 blacklisted.
  User decision needed: inode cleanup/quota raise (my deprecated partials
  ≈150k inodes = drop in the 51.2M bucket; the bulk is historical).
- S26 03:0x-03:3x DOSE-3 ACCEPTED (dispersion axis) — quota freed by user
  ("solved"), refill 8 cells landed on clean nodes, 4-seed final verdict ran:
  sd +0.0743 z +5.65, level clears 0.80 in ALL THREE computations, qL1 −34%
  z −2.99, toward-1 7/8. V14 isolated critic: "尽全力也推翻不了" — independent
  bootstrap z +5.75 (0/2000 ≤0), single-seed separation total (worst m3 > best
  m2), LOSO min z +4.36, real-reference drift zero (125/125 md5), the skipped
  §8.5c co-read computed by the critic and FAVORABLE (all-excised z +6.47).
  Wording corrections adopted (safety face open at verdict time → landed
  separately: LOB-Bench m3 ks21 0.1221/0.1288 best-since-r1, hold-CE −1.9%).
  P1 dispersion bar met in FULL. Effect ANTI-FRAGILE under seed hardening
  (4.47→5.65) where E11 shrank. Keepsake: bit-identical 18.268σ across arms =
  tick-grid saturation of independent blow-ups, not contamination.
