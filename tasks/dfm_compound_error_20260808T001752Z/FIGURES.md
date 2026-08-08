# 图索引

哪张是定稿、哪张是中间态。文档里引用的编号以此为准。

## 定稿（可直接用）

| 图 | 内容 | 对应章节 |
|---|---|---|
| **`fig8_final_baseline_final.png`** | **定稿基线**：四视图并排（逐 index / 时间轴 × LOBSTER-vs-sim / sim-vs-sim），n=3136 | **R15** |
| `fig7_time_aligned_ar_baseline_t50.png` | 时间轴上的 $\widetilde D(T)$ + 两条时钟发散 + 每秒实现波动率 + 消息率 | R7 |
| `fig6_shuffle_falsification_ar_baseline.png` | 代理模型证伪：池化 KS/L1 给零时间结构代理满分 | R5 |
| `fig5_time_dependence_ar_baseline.png` | ACF 五条 + 方差比 VR(k) | R4 |
| `fig1_compound_curves_v8.png` | 九观测量逐 index 曲线，**含 excess-over-floor 修正后的 CI** | R14 |
| `fig3_tails_v8.png` | 尾部贡献 + 尾部质量比 + 前向/反向 KL | R10 |
| `fig4_snapshots_v8.png` | 标准化分布在四个 rollout 深度的快照 | R2 |

## 中间态（保留作为修正轨迹的证据，不要直接引用）

| 图 | 为什么保留 |
|---|---|
| `fig1/2/3/4_*_ar_baseline.png` | 最初一版（R2/R3），其量级已被 R7/R8/R15 撤回 |
| `fig2_normalisation_v9.png` | R11 的证据：把 raw 网格真正锚在 $m{=}0$ 之后，归一化只改 0.5%–19% |
| `fig1/2/3/4_*_v7.png` | R12 的证据：类别通道补上地板与 CI |
| `fig1/2/3/4_*_oor_check.png` | R10 的证据：`tail_contribution` 改名 + 越界质量 |
| `fig7_time_aligned_ar_baseline_t10.png` | R7 的敏感性：把时间网格缩到 10 分位（T=0.069s）结论不变 |
| `fig8_final_baseline_probe128.png` | R15 前的 128 条验证跑，地板高于信号，说明为何必须放大到 3136 |
| `fig5_time_dependence_refactor_check.png` | 抽出 `viz.py` 之后数字逐位复现的验证 |

## 待产出

| 图 | 内容 | 触发条件 |
|---|---|---|
| `fig9_arms_*.png` | AR 基线 vs DFM 后训练，三族 + 物理速率并排 | `dfm_correct_runner.py` 产出 rollout 后，跑 `compare_arms.py` |
