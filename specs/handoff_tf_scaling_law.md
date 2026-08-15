# Handoff：Transformer scaling law 全流程（训练 → 双轴评测 → 拟合），含会话溯源

写于 UTC 2026-08-02T20:35Z。本文档给出这条工作线的完整索引、术语定义、复现命令与运维坑，并记录产生这些结果的对话会话标识，以便日后回到原始上下文查证任何一个决策是怎么做出来的。所有路径均为绝对路径。

本文档与同目录下的 `handoff.md` 是**两条不同的工作线**，互不覆盖。`handoff.md` 记录的是 valset_v1 验证集构造与 IsoFLOP 标度分析（会话 `79e7e513-c9d4-4f7e-adf4-9c761190316e`）；本文档记录的是 Transformer 架构的 scaling law 参数拟合（会话 `9d4b47e0-6a05-418a-a2d9-968507ecc663`）。两者共用同一个验证集实体数据 `shard_valset_v1_30720.squashfs`，但目标、产物与结论各自独立。

## 1. 会话溯源

这条工作线的全部对话记录在同一个会话文件里。会话期间经历过多次上下文压缩与客户端重连，记录始终追加写入同一个文件，没有分裂。

| 项 | 值 |
|---|---|
| Session ID | `9d4b47e0-6a05-418a-a2d9-968507ecc663` |
| 记录文件绝对路径 | `/lus/lfs1aip2/projects/public/u6gb/.claude/projects/-lus-lfs1aip2-projects-public-u6gb/9d4b47e0-6a05-418a-a2d9-968507ecc663.jsonl` |
| 文件格式 | JSON Lines，每行一条记录（用户消息、助手回复、工具调用与返回各占若干行） |
| 记录条数 | 5,554 行（会话仍在继续时该值会增长） |
| 文件大小 | 7,716,354 字节，约 7.4 MB（同上，会随会话增长） |
| 首条记录时间 | 2026-07-29T12:59:13Z |
| 末条记录时间 | 2026-08-02T20:27:49Z |
| 工作目录 | `/lus/lfs1aip2/projects/public/u6gb` |
| 任务起点 | Notion 页 "transformer scaling law"，指令原文「需要跑transformer的所有的实验」 |

恢复这个会话继续对话：

```bash
cd /lus/lfs1aip2/projects/public/u6gb
claude --resume 9d4b47e0-6a05-418a-a2d9-968507ecc663
```

在记录文件里检索特定内容。文件是纯文本 JSON Lines，可以直接用 `grep` 单文件检索，这对 Lustre 元数据没有压力。检索时优先使用高选择性的锚点，例如 Slurm 作业号、checkpoint 目录名、Notion 页面 ID，而不是常见词。

```bash
JSONL=/lus/lfs1aip2/projects/public/u6gb/.claude/projects/-lus-lfs1aip2-projects-public-u6gb/9d4b47e0-6a05-418a-a2d9-968507ecc663.jsonl

grep -c 'CHECKPOINT_EVERY=0'   "$JSONL"    # 补尾崩溃的诊断与修法这一段
grep -c 'TOKEN_MODE'           "$JSONL"    # 26tok 静默故障这一段
grep -c '5848061'              "$JSONL"    # 本轮 attach 补跑 60 点这一段
grep -c 'label-block'          "$JSONL"    # 「加轨迹点只买 β 不买 α」的推导这一段
```

**容易混淆的一点**：本次会话过程中出现过形如 `a67833bf-fac1-48f4-a9bf-8d1f3452d121` 的 UUID，它是后台任务的临时工作目录标识（位于 `/local/user/1483804540/claude-1483804540/-lus-lfs1aip2-projects-public-u6gb/<uuid>/tasks/`），**不是** session ID，磁盘上没有对应的 JSONL 文件。唯一的 session ID 就是上表中的那一个。判别方法是看 `.claude/projects/-lus-lfs1aip2-projects-public-u6gb/<uuid>.jsonl` 是否存在。

## 2. 术语与量的定义

阅读后文之前需要先固定这些量的含义。

**N（参数量）**：模型的可训练参数总数，单位「个」。取自训练日志实测值，不是标称名。标称名 `0p2M`…`200M` 只是网格标签，与实测 N 有系统偏差（例如标称 0.2M 的实测 N 是 2,037,171）。后文一律用标称名做索引、用实测 N 做计算。

**D（训练量）**：模型从训练开始到该 checkpoint 为止见过的 token 总数，单位「个 token」。算法是 `D = global_step × 1,664,000`，其中每步 token 数 `1,664,000 = 全局 batch 128 × 每样本 500 条消息 × 每条消息 26 个 token`。全网格该常数一致，因此 D 与 step 严格成正比。

**L（损失）**：held-out 集上的 next-token 交叉熵，单位 nats/token。两把尺子各自算一个 L，见下。

**Chinchilla 标度律**：拟合的函数形式是

```
L(N, D) = E + A · N^(-α) + B · D^(-β)
```

其中 `E` 是不可约损失（把模型和数据都推到无穷时的渐近下界，单位 nats/token），`A · N^(-α)` 是参数量不足带来的额外损失，`B · D^(-β)` 是训练量不足带来的额外损失。`α` 与 `β` 无量纲，值越大表示该方向的边际收益衰减越快（即「加参数/加数据更划算」）。`A`、`B` 是量纲配平用的系数，本身没有解释意义。

**tail_frac（拟合窗口）**：每条训练 run 只取轨迹末尾多大比例的 checkpoint 进入拟合，取值 0 到 1。`tail_frac = 0.25` 表示只取最后 25%（这是预注册的口径），`1.0` 表示全轨迹。这个参数决定了 D 轴的杠杆长度，是后文所有「窗口」的含义。

**D 跨度（decades）**：进入拟合的所有点里，最大 D 与最小 D 之比取以 10 为底的对数，单位「数量级」。它衡量 D 轴的杠杆有多长。跨度太短时 `β` 无法识别，会撞到求解器的边界。

**两把尺子**。`test` 指 Jan-2026 前向 held-out：模型训练区间是 2023-01-01 到 2025-12-31，测试数据是 2026 年 1 月 2 日到 1 月 31 日的 487 支股票，时间上严格在训练之后，衡量的是「向未来外推」的能力。`valset` 指 valset_v1 同分布 held-out：从训练语料同一时间范围内抽出、从未被任何训练消费过的 30,720 个样本窗口，衡量的是「同分布泛化」的能力。

**valset 的两个口径**。`full` 指全部 30,720 个样本。`y2325` 指其中年份落在 2023 到 2025 的 13,741 个样本（占 44.7%）。之所以要分开，是因为 valset_v1 当初是为 Mamba3 队列（训练区间 2022-01 到 2025-12）构造的，其中 55.3% 的样本来自 2022 年；而 Transformer 队列只训练 2023 年起的数据，那 55.3% 对它是「向后时移」而非同分布。逐样本损失全部落盘、provenance 又带年月，所以两个口径离线切分即可得到，零额外 GPU 成本。

**LPT 调度**：Longest Processing Time first。评测 worker 从共享清单里按参数量从大到小抢任务，目的是避免收尾阶段某张卡单独拖一个大模型。副作用是队列被打断时残缺的必然是清单尾部，即最小的模型。

**curtail**：训练脚本的 `CURTAIL_EPOCHS` 参数，含义是「本次训练跑到第几步为止」，配合 cosine 学习率调度让 LR 在该步衰减到 0。每个尺寸有各自的值（见第 5 节表）。补尾时**必须填原值**，不能填剩余步数。

**TOKEN_MODE**：环境变量，取值 `24tok` 或 `26tok`，决定分词器把一条 LOB 消息编码成几个 token。本队列全部用 `26tok` 训练，对应序列长度 `500 × 26 = 13000`。

## 3. 这条工作线做了什么

会话跨越五天，包含三个前后依赖的阶段。

**第一阶段，把 31 个训练 run 全部跑完。** 网格是 11 个尺寸（标称 0.2M 到 200M）× 3 个随机种子（5、42、137），其中 120M 与 200M 因算力预算只跑 2 个种子，合计 31 个 run。接手时已完成 16 个，其余 12 个是被墙钟截断的半成品（需要补尾到 LR 衰减为 0 的终点），加上 200M:s5 完全没跑。集群当时偏好大 job，30 分钟的小 job 排队要等 13 小时以上插不进去，因此改用 pilot 占位模式：申请一个大而长的 allocation，在壳内串行跑多个同形状实验，把 31 次排队压缩成 9 次。

**第二阶段，做两条轴的 held-out 评测。** 训练全程是 train-only（`NO_VALIDATION=True` 且 `SKIP_TEST_EVAL=1`），没有算过任何 held-out 损失，而拟合的因变量正是 held-out 损失，所以必须训练后单独评测。test 轴评了 234 个 checkpoint 得到 224 个可用点，valset 轴评了 325 个 checkpoint。评测的成本瓶颈是固定开销而非算力：无论模型大小，单点耗时都是约 590 秒，时间几乎全在数据加载（顺序过一遍 30,720 个样本），所以规划要按 checkpoint 数量算，不能按参数量算。

**第三阶段，拟合并交叉验证。** 用同一个拟合器（`fit_test_ce_kang.py`，零改动，只传 `--arch transformer`）在两条轴、三个窗口上各拟合一次，再加四组去离群点的敏感性检验。中途发现 valset 轴的 `β` 撞上界无法识别，根因是首版清单只采了每条 run 尾部 25% 的点，run 内 D 只跨 0.103 个数量级；加密清单覆盖整条轨迹后 D 跨度提到 2.23 个数量级，`β` 变为内点解。

## 4. 最终结果

### 4.1 完成度

| 阶段 | 数量 | 说明 |
|---|---|---|
| 训练 | 31 / 31 | `0p2M×3 1M×3 4M×3 6M×3 10M×3 14M×3 23M×3 46M×3 78M×3 120M×2 200M×2` |
| test CE 评测 | 224 点 / 30 run | Jan-2026 前向 held-out，487 ticker |
| valset CE 评测 | 325 点 / 31 run（含 31 个终点） | 30,720 样本全量，双口径离线切分 |
| 拟合产物 | 13 个 JSON | test 3 窗口 + valset 6（双口径 × 三窗口）+ 4 个敏感性 |

唯一的保留是 `4M:s42` 训练发散：它跑完了全部步数，但损失在中途崩掉，终点交叉熵 2.371 而同尺寸另外两个种子是 0.688。该 run 被排除出全部拟合，所以准确说法是「31 个跑完，30 个可用」。

### 4.2 三把尺子对照（tail_frac = 1.0）

| 尺子 | n | D 跨度 | α | β | E |
|---|---|---|---|---|---|
| test（Jan-2026 前向 held-out） | 224 | 2.39 | 1.853 [1.751, 1.891] | 0.988 [0.957, 1.047] | 0.5385 [0.5334, 0.5505] |
| valset full（30,720 全量） | 276 | 2.23 | 0.805 [0.519, 1.205] | 0.892 [0.817, 0.978] | 0.5802 [0.5581, 0.5928] |
| valset y2325（13,741 同分布） | 276 | 2.23 | 0.805 [0.534, 1.196] | 0.991 [0.889, 1.046] | 0.5017 [0.4814, 0.5136] |

方括号内是 500 次 bootstrap 的 95% 置信区间。对照 Mamba3 v6（test 轴）：`E=0.5519`、`α=2.0`、`β=0.679`；TF 的 `E` 与之相差 0.6%。

### 4.3 valset 完整结果（双口径 × 三窗口）

| 口径 | 窗口 | n | α | β | E |
|---|---|---|---|---|---|
| full | 0.25（预注册） | 106 | 0.427 [0.122, 0.816] | 2.000（撞上界，不可用） | 0.5876 |
| full | 0.75 | 214 | 0.802 [0.570, 1.104] | 1.558 [1.289, 1.705] | 0.6033 |
| full | 1.00 | 276 | 0.805 [0.519, 1.205] | 0.892 [0.817, 0.978] | 0.5802 |
| y2325 | 0.25（预注册） | 106 | 0.365 [0.075, 0.765] | 2.000（撞上界，不可用） | 0.4958 |
| y2325 | 0.75 | 214 | 0.919 [0.495, 1.205] | 1.554 [1.179, 1.760] | 0.5153 |
| y2325 | 1.00 | 276 | 0.805 [0.534, 1.196] | 0.991 [0.889, 1.046] | 0.5017 |

test 轴同样三个窗口：`tail=0.25` 时 `α=1.874`、`β=0.050`（撞下界）、`E` 区间退化到 `[0, 0.559]`，同样不可用；`tail=0.75` 是 `α=1.827 [1.304, 2.109]`、`β=1.196 [0.933, 1.527]`、`E=0.5487 [0.5394, 0.5575]`。**预注册的 0.25 窗口在两条轴上都不可用**，原因是该窗口内 D 只跨 0.99 个数量级，D 轴杠杆不足以同时定出 `β` 和 `E`。这个差异本身是结论，不是错误。

### 4.4 四条结论

`β` 是稳健的，两把尺子给出同一个数：test 0.988 对 valset-y2325 0.991，几乎逐位相同。这说明数据指数不依赖于用哪个 held-out 集衡量。

评测口径只改 `E`，不改指数：从 full 换到 y2325 让 `E` 掉 0.078（0.5802 到 0.5017），而 `α` 完全不动（0.805 到 0.805）。两个口径的逐 run 差值 `Δ = full − y2325` 在所有 6M 及以上的模型上都落在 +0.089 到 +0.096 nats、极平；4M 与 1M 是 +0.098 到 +0.120；0.2M 才涨到 +0.139 到 +0.164。这个「向后时移惩罚」随模型变大而缩小并趋于约 0.089 的常数。

`α` 在两把尺子上不重叠（1.85 对 0.81），机制已定位：valset 的 N 曲线到 200M 仍在下降（终点 0.607 比 `E` 高 0.027），`E` 只能外推，其置信区间宽一倍（0.035 对 0.017），`α` 继承了这份不确定性；test 曲线在 14M 到 120M 已经压平（0.561 到 0.551），`E` 被直接钉住。

**加轨迹点只买 `β`，不买 `α`。** 拟合器的 bootstrap 是 label-block over logical runs，即按 30 个逻辑 run 整块重采样。`α` 描述参数量到损失的关系，它的独立观测数是尺寸/run 的个数，往已有 run 内部加 checkpoint 是零新增独立信息。本轮实测：补 60 个点后 `β` 的置信区间窄了 2.8 倍（从 [0.859, 1.303] 到 [0.817, 0.978]），`α` 的置信区间宽度纹丝不动。**要收紧 `α` 只能加种子或加尺寸，继续加 checkpoint 是白花 GPU。** 这是本轮对后续实验规划最直接可用的一条。

### 4.5 数据质量：发散与 loss spike 的判别

`4M:s42` 的训练发散在 valset 轴独立复现：`0.818@10550 → 6.423@15610 → 2.371@19840`，与 test 轴（step 12510 起 `0.90 → 2.60 → 6.37`）是同一次事故的两把尺子。判据是「终点损失比同尺寸各种子终点的中位数高出 1.5 倍以上」，此处比值 3.45。排除它有双轴证据，不是单点判断。

另有两个 loss spike：`4M:s137@2730` 交叉熵 3.378、`6M:s137@12700` 交叉熵 1.034。它们下一个点即恢复，终点与同尺寸中位数完全一致（比值 1.00），属于训练中的瞬时不稳定而非发散。这两个点**保留**在拟合内，因为拟合器用的是 log-Huber 损失，本就为压制离群点设计。去 spike 的敏感性检验共 8 组配置，所有 `α`、`β`、`E` 都落在含 spike 版本的置信区间内，最大位移是 full 口径 tail=0.75 的 `α` 从 0.802 变 0.680（仍在 [0.570, 1.104] 内），结论不依赖这两个点。

11 个尺寸的终点 valset 交叉熵严格单调、无一处倒挂：

```
0.2M 1.133 → 1M 0.761 → 4M 0.688 → 6M 0.662 → 10M 0.649 → 14M 0.636
     → 23M 0.631 → 46M 0.627 → 78M 0.625 → 120M 0.613 → 200M 0.606
```

## 5. 网格参数表

`curtail` 是该尺寸的目标总步数，`N` 是实测参数量，`bsz/GPU` 是训练时的单卡 batch（评测时沿用，且必须整除 30,720）。

| 标称 | d_model | N（实测） | bsz/GPU | curtail | 种子 |
|---|---|---|---|---|---|
| 0p2M | 64 | 2,037,171 | 16 | 7,438 | 5 / 42 / 137 |
| 1M | 128 | 3,226,931 | 16 | 14,877 | 5 / 42 / 137 |
| 4M | 192 | 6,777,523 | 16 | 19,836 | 5 / 42 / 137（s42 发散） |
| 6M | 256 | 10,499,379 | 16 | 49,590 | 5 / 42 / 137 |
| 10M | 320 | 15,122,355 | 16 | 43,605 | 5 / 42 / 137 |
| 14M | 384 | 20,646,451 | 16 | 37,620 | 5 / 42 / 137 |
| 23M | 512 | 35,400,691 | 8 | 56,430 | 5 / 42 / 137 |
| 46M | 768 | 72,714,547 | 8 | 39,330 | 5 / 42 / 137 |
| 78M | 1024 | 125,449,011 | 8 | 29,754 | 5 / 42 / 137 |
| 120M | 1280 | 192,601,395 | 4 | 45,828 | 5 / 42 |
| 200M | 1664 | 320,363,571 | 4 | 65,664 | 5 / 42 |

## 6. 文件索引

### 6.1 拟合产物

目录 `/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/scaling_law_plots/`

| 文件 | 内容 |
|---|---|
| `fit_TF_testce_tail{025,075,10}_20260801.json` | test 轴三个窗口的拟合结果 |
| `fit_TF_valset_{full,y2325}_tail{025,075,10}_v2.json` | valset 轴双口径 × 三窗口 |
| `fit_TF_valset_{full,y2325}_tail{075,10}_v2_nospike.json` | 去 spike 敏感性检验 |
| `valset_ce_tf_fit_ready_v2.csv` | valset full 口径拟合输入，318 点 |
| `valset_ce_tf_y2325_fit_ready_v2.csv` | valset y2325 口径拟合输入，318 点 |
| `valset_ce_tf{,_y2325}_fit_ready_v2_nospike.csv` | 上二者去掉两个 spike 点的版本，316 点 |
| `test_ce_sp500_jan2026_tf_fit_ready_clean.csv` | test 轴拟合输入，224 点 |
| `fit_test_ce_kang.py` | 拟合器本体，**本工作线零改动**，传 `--arch transformer` 即可 |

每个 `fit_*.json` 的结构是四个顶层键：`protocol`（拟合协议与 bootstrap 设置）、`input`（输入文件与去重审计）、`selected_data`（进入拟合的点数、run 数、N/D/L 的范围）、`bootstrap`（置信区间）。点估计在 `pooled` 键下。

### 6.2 评测产物

目录 `/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/valset_eval/results_tf26_20260801T145139Z/`

| 文件 | 内容 |
|---|---|
| `valce_<label>.json` | 单个 checkpoint 的评测结果（均值、bootstrap 区间、准确率、样本数、墙钟） |
| `valce_<label>_sampleloss.npy` | 该 checkpoint 的逐样本损失，float32 × 30,720，**双口径切分就靠这个** |
| `valset_ce_tf_master_table.csv` | 325 点双口径主表 |
| `lock_<label>/` | 工作队列的抢占锁，成功后不删除（完成标记是 json 的存在，不是锁的消失） |

`<label>` 的命名规则是 `tf-<尺寸>-s<种子>` 表示终点，`tf-<尺寸>-s<种子>@<step>` 表示轨迹中间点。

### 6.3 工具脚本

目录 `/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/valset_eval/`

| 文件 | 用途 |
|---|---|
| `valset_ce_eval_tf.py` | 评测主体。由 `valset_ce_eval.py` 改四处而来：`EXP_DIR` 改到 exp_O8（Mamba3 的 `init_train.py` 没有 transformer 分支）、强制复制的参数键表补入 `model_type/n_heads/d_ff/dtype/use_flash/remat/use_rope/rope_base/token_mode/msg_seq_len`、删掉 O8 签名里没有的 `book_ablation` 形参、元信息打印行扩充 |
| `make_manifest_tf.py` | 生成评测清单。`--trajectory` 加轨迹点，`--lo-frac` 控制采样下界（0.75 只取尾部窗口，0.02 覆盖整条轨迹） |
| `manifest_tf.json` / `manifest_tf_dense.json` | 108 点清单 / 277 点加密清单 |
| `manifest_tf_missing60.json` | 本轮补跑的 60 点清单 |
| `parallel_valset_tf.sh` | 单节点 4 卡工作队列启动器。**开头会清残锁**，只能在确知无 worker 存活时用 |
| `valset_addnodes_tf.sh` | 横向扩容 worker。**绝不清锁**，带显存 gate，attach 与 sbatch 可混投同一队列 |
| `valset_tf_missing60.sbatch` | 独占 1 节点补跑入口 |
| `build_valset_tf_fit_ready.py` | 从评测结果一步生成两个口径的拟合输入 |
| `check_trajectories_tf.py` | 逐 run 打印轨迹并按 1.5 倍判据标出发散 run |
| `summarize_valset_tf.py` | 生成双口径主表 |

目录 `/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/`

| 文件 | 用途 |
|---|---|
| `scaling_train.batch` | 训练主体 |
| `scaling_pilot.batch` | pilot 占位执行器，壳内串行跑多个同形状实验 |
| `scaling_topup_submit.sh` / `scaling_attach_topup.sh` | 补尾提交 |
| `eval_test_ce_sp500.batch` | test 轴评测 |
| `attach_eval_worker.sh` | test 轴的节点级自取任务 worker |

**这些文件全部未纳入 git**：`/lus/lfs1aip2/projects/public/u6gb` 仓库的 `.gitignore` 忽略了整个 `tasks/` 目录，实测 `git ls-files` 返回 untracked、`git check-ignore` 返回 IGNORED。若要提交必须用 `git add -f`。exp_O8 那批文件在另一个仓库（`/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade`，分支 master）下，同样需要单独确认。

### 6.4 数据

| 路径 | 内容 |
|---|---|
| `/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/squashfs/output/shard_valset_v1_30720.squashfs` | valset 实体数据 |
| `/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/squashfs/output/provenance_valset_v1_30720.npz` | 逐样本溯源，含年月，双口径切分靠它 |
| `/lus/lfs1aip2/projects/public/s5e/quant_team/lob_preproc_sp500_squashfs/shard_2026-01.squashfs` | test 集实体数据 |
| `/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_O8_self_attention/checkpoints/` | 全部训练 checkpoint |

W&B 项目：`https://wandb.ai/oxford-lob/neurips-transformer-scaling-runs`

## 7. 复现命令

以下命令按依赖顺序排列。运行前须知：拟合的 bootstrap 用 joblib 扇出，**必须在计算节点跑**，login 节点的 cgroup 会把 worker 杀掉；且必须把 BLAS 线程数钉为 1，否则忙节点上 OpenBLAS 默认开 128 线程会撞 `RLIMIT_NPROC`。

生成评测清单：

```bash
cd /lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/valset_eval
PY=/lus/lfs1aip2/projects/public/s5e/quant_team/quant/miniforge3/bin/python3
$PY make_manifest_tf.py --trajectory --n-mid 10 --lo-frac 0.02 --out manifest_tf_dense.json
```

跑评测。独占节点走 sbatch：

```bash
cd /lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/valset_eval
OUT_DIR=<结果目录> sbatch valset_tf_missing60.sbatch
```

或 attach 进已有 allocation（每节点一个 srun，见第 8 节第 5 条）：

```bash
export MANIFEST=/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/valset_eval/manifest_tf_dense.json
export GATE_MB=2000 GATE_TIMEOUT_S=39600 MAX_PASS=6
for N in nid010076 nid010077 nid010078 nid010079; do
  srun --jobid=<ALLOC> --overlap --nodelist=$N --nodes=1 --ntasks=1 --cpu-bind=none \
       /lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/valset_eval/valset_addnodes_tf.sh \
       <结果目录> &
  sleep 8
done
```

核对完成度、判定轨迹、生成拟合输入：

```bash
cd /lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/valset_eval
PY=/lus/lfs1aip2/projects/public/s5e/quant_team/quant/miniforge3/bin/python3
$PY check_trajectories_tf.py <结果目录>
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  $PY build_valset_tf_fit_ready.py <结果目录> --tag _v2
```

跑拟合（在计算节点上）：

```bash
srun --jobid=<ALLOC> --overlap --nodelist=<节点> --nodes=1 --ntasks=1 --cpu-bind=none \
     bash /lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/scaling_law_plots/run_fit_valset_tf_v2.sh
```

## 8. 运维坑（全部实际踩过）

**第 1 条，`TOKEN_MODE` 未 export 会让数字静默出错。** `lob/encoding.py` 在 **import 时**读这个环境变量选分词器，默认 `24tok`。用 24tok 布局评 26tok 训练的模型不会报错，只是交叉熵从约 0.57 变成 12.9、准确率从 89% 变成 12%。指纹是输出 CSV 里 `seq_len=12000` 而非 13000。**写进配置文件或 checkpoint 参数继承表都救不了**，因为编码模块在模型加载之前就已 import 完毕，只能在最外层 launcher export。本次会话在三个不同入口各踩了一次。

**第 2 条，半成品被判为完成。** 评测按 step 追加写 CSV，进程被杀会留下「非空但截断」的文件。完成判定必须是「不同 step 的数量 ≥ 请求数量」，不能是 `[ -s file ]`。同理，工作队列的完成度**不能数目录里的文件数**：`results_tf26_*` 目录里同时躺着两份清单的产物，按文件数判会提前收工，必须按「本清单的每个 label 是否有 json」判。

**第 3 条，`jid` 列写空字符串会静默丢光所有行。** pandas 把空串读成 NaN，`groupby` 默认 `dropna=True`，109,575 行全部消失且不报错。必须写成 `<尺寸>-s<种子>` 这样的非空值。

**第 4 条，停止追踪不等于停止进程。** 后台任务工具的 stop 只解除追踪，srun step 仍在跑、仍在写产物。必须 `scancel` 那个 step 并按文件时间戳确认。本次有一个 worker 在我以为已停止后又跑了 1 小时 36 分钟写坏数据。

**第 5 条，attach 时一次申请多个 task 会报 `Error configuring interconnect`。** Slingshot 每节点的并发 CXI service 有上限，若该 allocation 已挂 batch 加若干 step 就会撞限额。绕法是拆成每节点一个 `--nodelist` 的 srun。这是 attach 进被反复使用的 allocation 的通用做法。

**第 6 条，`sed -i` 会重置文件权限。** 它的实现是写临时文件再 rename，权限按 umask 取，所以 `chmod +x` 之后再 `sed -i` 就丢了执行位，srun 报 `execve: Permission denied` / exit 13。脚本改完要么重新 chmod，要么统一用 `bash <script>` 调用。

**第 7 条，就地改写正在运行的 shell 脚本会毒化运行中的作业。** bash 按字节偏移惰性读脚本，改写会让已在跑的实例读到错位的内容。必须写临时文件再 `os.replace` 原子改名。本次曾因此毒化 8 个运行中的实验。

**第 8 条，补尾的两条硬规则。** `CURTAIL_EPOCHS` 必须填**原值**而非剩余步数（填剩余会让 `start_epoch = state.step // steps_per_epoch` 落到不存在的 epoch，循环零执行、返回码 0 但零训练）；补尾必须 `CHECKPOINT_EVERY=0`（restore 之后第一次 checkpoint 保存的跨节点同步会触发 NET/OFI `Operation not permitted` 继而 SIGABRT，崩溃时刻可由「checkpoint 间隔 ÷ 每秒步数 + 启动时间」精确预测，23M 预测 29.7 分钟、实测 29 分 26 秒）。禁用中间保存后 11 战 11 胜。

**第 9 条，独立 worker 必须屏蔽多节点环境变量。** JAX 的分布式初始化读 `SLURM_NNODES`，若 attach 时它是 4，四个本应独立的单卡 worker 会去 join 同一个协调服务，报 `RegisterTask DEADLINE_EXCEEDED`。worker 内必须 `export SLURM_NNODES=1 SLURM_NTASKS=1 SLURM_PROCID=0`。

**第 10 条，FUSE 挂载点不能复用。** 挂载路径要带主机名与 PID，否则复用已死挂载的路径会让 `squashfuse` 直接失败。

**第 11 条，显存 gate 是 attach 的必备礼节。** `--overlap` 只让 Slurm 允许 step 共存，**不提供任何显存保护**。worker 必须先轮询 `nvidia-smi`、连续两次读到低于阈值才启动。本轮的 16 个 worker 在 gate 上等了 23 分钟，直到占卡的任务跑满 9 小时 55 分钟自然结束才开闸，全程未抢占。gate 里若要检查队列剩余量，务必降频（本实现是每 10 轮一次），否则每分钟对整份清单做 stat 就是 Lustre 元数据风暴。

## 9. 未竟事项

**收紧 `α`**。当前 valset 轴 `α` 的置信区间是 [0.519, 1.205]，偏宽。按第 4.4 条的结论，唯一有效的办法是增加独立 run：给 120M 与 200M 补第三个种子（137），把 30 个逻辑 run 提到 32 个；更彻底的做法是在 200M 之上再加一个尺寸，把 N 轴的杠杆拉长。继续加 checkpoint 无效。

**Mamba3 的 2023-2025 口径对照**。Mamba3 队列已有的逐样本损失 `.npy` 文件同样可以离线切出 y2325 口径，从而得到与 Transformer 严格同底的跨架构对照。这不需要任何 GPU，只是还没做。

**test 轴的加密**。test 轴目前是 224 点、D 跨 2.39 个数量级，已经够用；若要与 valset 轴逐窗口严格对齐，可以按同样的 `--lo-frac 0.02` 重采一遍。优先级低。

## 10. Notion 同步

结果已推送至 Notion 页 "transformer scaling law"。

| 项 | 值 |
|---|---|
| 页面 URL | https://www.notion.so/3ac12c4568fd80da8b8cd49424a49d96 |
| 页面 ID | `3ac12c45-68fd-80da-8b8c-d49424a49d96` |
| 本轮追加 | 20 个 block：完成度 callout、三把尺子对照表、valset 双口径三窗口表、四条结论 callout、发散与 spike 判别段、产物路径 code block、补跑执行记录 |
| 推送方式 | Notion MCP，`API-patch-block-children` |

推送 Notion 表格时要注意：`table` 类型的 block **必须在创建时**把全部 `table_row` 作为 `children` 一起传入，事后不能往里追加行；且 `has_children: true` 只说明有子块，不保证行数正确，必须回读子块核对。

---

*本文档写于 2026-08-02。若需查证任何结论的推导过程或某个决策的当时理由，请按第 1 节的方法回到会话记录。*
