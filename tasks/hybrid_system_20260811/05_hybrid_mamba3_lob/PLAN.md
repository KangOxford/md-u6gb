# 05 · Hybrid Mamba3 × Nemotron 用于 LOB 消息流建模 — 主计划

**建档时间**：UTC 2026-08-11T20:30Z
**指令来源**：Notion 页 `build-hybrid`（`3b912c45-68fd-8074-9f3c-eec3c125f9f4`）第 (1)–(8) 条 + 常驻规则 A1–A12
**代码工作区**（A1）：`/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/hybrid-mamba3-nemotron-20260811`
**分支**：`feat/hybrid-mamba3-nemotron-20260811`（基于 sigma-0 `5c168ed`）
**文档工作区**：本目录 `/lus/lfs1aip2/projects/public/u6gb/tasks/hybrid_system_20260811/05_hybrid_mamba3_lob/`

> 本文件是**滚动更新**的活文件。规则 A8：每当第一次拿到结果就立刻写进来，不是等全部做完再回头补。

---

## 0. 一句话目标

把 Nemotron-H 的「少量全局 attention 层插进大量线性递归层」这一混合配方，移植到 sigma-0 现有的**纯 Mamba3** LOB 消息模型上，在**同数据、同 token 预算、同硬件**的前提下，看四项指标能否**同时不劣、至少一项显著更好**。

判定对象不是「hybrid 好不好」这种泛问，而是一个可证伪的命题：

> **命题 H**：在 LOB 消息流上，把 Mamba3 层堆叠中的少数几层替换为全局因果 attention，能在不增加显著计算成本的前提下，改善那些**依赖精确回指历史某条具体订单**的能力（撤单/删单/成交指向真实挂单），且不损害整体 perplexity 与 LOB-Bench 分布匹配。

这条命题之所以值得测，是因为 LOB 消息流有一个文本没有的性质：**引用型消息**（cancel/delete/execution）在语义上必须指向历史上某一条具体的、还活着的挂单。固定大小的递归状态压缩历史时，这种「精确回指」正是最先被压掉的信息；而 attention 的本职就是按内容直接寻址任意历史位置。所以 Nemotron 配方在这里不是「跟风混合」，而是**对症**。

---

## 1. 八步映射到具体动作

| 步 | 指令原文 | 具体动作 | 状态 |
|---|---|---|---|
| (1) | 新建 worktree，跟 Nemotron 设置，用 mamba3 | worktree `hybrid-mamba3-nemotron-20260811`；配方见 `research/C_nemotron_recipe.md` | ✅ 完成 |
| (2) | 先给 baseline 建 memory | `memory/project_hybrid_mamba3_{baseline,code_map}.md`，MEMORY.md 已加指针 | ✅ 完成 |
| (3) | 找到好的 baseline 结果，汇成 md | `results/BASELINE.md`（锁定 M1 + 预注册 P1–P4 + 噪声底 + 陷阱） | ✅ 完成 |
| (4) | 造 hybrid 模型 | commit `c0bcd15`/`5090421`/`636512c`；CPU+GPU 双冒烟通过，参数量与已发表基线逐位对上 | ✅ 完成 |
| (5) | 训练它，看能否长时间稳定训 | 冒烟 1–3 失败（三处白名单 / 残留挂载 / 显存争用），第 4 次进入训练循环 | 🟡 进行中 |
| (6) | 8 只股票 2022–2025 上训练 | 数据配置与 baseline 逐字相同，48 月 × 8 票已挂载成功 | 🟡 随 (5) |
| (7.1) | message-level perplexity | baseline 侧待补（B1） | ⬜ 未开始 |
| (7.2) | 方向准确率 + return IC | **baseline 侧已补测完成**，见 `results/baseline_backfill/` | 🟢 baseline 已就位 |
| (7.3) | LOB-Bench | baseline 侧已有（0.20714/0.10458/0.16451） | 🟢 baseline 已就位 |
| (7.4) | refer order success rate | baseline 侧有 cancel+delete 混算数；execution 拆分需新写（B3） | 🟡 部分 |
| (8) | 26 tok plan | 前置审计已完成（三候选设计 A/B/C），见 §5 | 🟡 待选型 |

---

## 2. 算力策略（A11 / A3）

规则要求**先 attach 在跑的节点，实在不行才新排队**。当前快照（UTC 2026-08-11T20:15Z）：

| Job | 节点 | 剩余 | 16 卡状态 | 可用性 |
|---|---|---|---|---|
| `5980502` | nid[010053,010371,010473,011179] | 15h49m | sm=0%、显存 0.0/85.5 GB | ✅ **全空闲，首选 attach** |
| `5980745` | nid[010580,010631,010669-010670] | 16h45m | sm=66%、显存 92.3 GB/卡 | ❌ 满载，禁止打扰 |
| `5992007/8` | — | PENDING | — | 后备（排队中的 4 节点链） |

**物理闸门**（CLAUDE.md 强制）：`--overlap` 不等于显存安全。每次 attach 前必须重新跑一次 `gpu_status.sh` 确认目标节点仍是零 PID / 近基线显存；历史上出现过「17:14 看是空的，两分钟后训练进程回来了」的情况，所以**探针必须在 CUDA 初始化前再验一次**。

attach 命令形态（由 `sbash` 技能生成，禁止 `scancel`）：

```
srun --jobid=5980502 --overlap --exact --nodes=1 --ntasks=1 --cpu-bind=none ...
```

---

## 3. 两条贯穿全程的关注线

### 3.1 长尾（A6）

LOB 消息流的边际分布是重尾的：绝大多数事件集中在最优买卖价附近的小额挂撤单，而真正影响价格的事件在尾部。**平均 perplexity 会把尾部完全洗掉**——一个只会复读「在 best bid 挂 100 股」的模型可以拿到很漂亮的平均 CE。

所以每一项指标都要给出**分层版本**，而不只是总平均：

```
                总平均 (会骗人)
                     │
     ┌───────────────┼───────────────┐
     ▼               ▼               ▼
  头部事件         中段           尾部事件
 (best bid/ask)  (2-5 档)    (深档 / 大额 / 罕见类型)
   占比 ~大        ~中           占比 ~小但影响大
```

具体做法：按 (a) 价格档位距 mid 的距离、(b) 订单规模分位、(c) 消息类型频率三个轴分箱，每箱单独报 CE 与 success rate。**hybrid 若真的赢，最可能赢在尾部**，因为尾部事件恰恰是「必须精确回指某条老挂单」的那些。

### 3.2 Bootstrapping（A7）

这个词在本任务里有两层含义，两层都要处理：

**(i) 自举式生成的误差累积（exposure bias）**：(7.2) 要求条件前半段、生成后半段。生成时模型吃的是自己吐出来的 token，与训练时吃真实 token 是**两个不同的输入分布**。已知教训（记忆 `feedback_train_infer_input_mismatch`）：曾经出现过「生成时喂全零 book、训练时喂真 book」，导致指标随训练**反向恶化**。所以：
- 生成回路必须与训练回路共用同一套输入构造代码，不允许各写一份；
- 报告 metric 随 rollout 步数 k 的曲线（而非只报终点），若曲线单调恶化，说明是自举误差而非模型容量问题。

**(ii) 统计自举（bootstrap CI）**：四项指标都要给置信区间，否则「hybrid 比 baseline 好 0.3%」毫无意义。LOB 数据有强自相关，普通 bootstrap 会低估方差，必须用**块自举（block bootstrap）**，块长取自相关时间尺度以上。判定「更好」的门槛是**配对块差的置信区间不跨 0**，而不是两个点估计的大小比较。

---

## 4. (7.4) refer order success rate 的定义（本任务自定，需固定下来）

引用型消息成功率，指生成序列中每一条 cancel / delete / execution 消息，其引用的 order id（或等价的价格-数量-方向三元组）**是否指向一条在该时刻确实存在于簿上的活挂单**。

$$
\text{SuccessRate}_{\text{type}}=\frac{\#\{\text{该类型消息中引用了活挂单的}\}}{\#\{\text{该类型消息总数}\}},\quad \text{type}\in\{\text{cancel},\text{delete},\text{execution}\}
$$

要点：
- 分母只数模型**自己生成**的该类消息，不含条件段；
- 判定需要一个**撮合引擎**在生成过程中维护实时簿状态，sigma-0 有 `src/matching_engine`，优先复用；
- 这是最能体现 hybrid 优势的指标（见 §0 命题 H），也是最可能**baseline 从未测过**的指标——若未测，需要给 baseline checkpoint 补测，否则没有对照。

---

## 5. (8) 26 tok plan 的位置

26 tok 是消息编码方案的变更（当前主线为 24 tok / 变长 BPE 支线另计）。它与 hybrid 架构是**两个正交的变量**，同时改会导致归因不可能。因此排在 (7) 之后：先在固定编码下把 hybrid vs baseline 的结论钉死，再单独引入 26 tok。计划细节待 (7) 结论出来后展开。

---

## 6. 记录与汇报（A8 / A9 / A12）

| 产物 | 路径 | 更新时机 |
|---|---|---|
| 本计划 | `PLAN.md` | 每个决策点 |
| baseline 汇总 | `results/BASELINE.md` | B 路调研落地即写 |
| 阶段报告 | `results/STAGE_*.md` | 每阶段结果第一次出现时 |
| 图 | `figures/*.png` | 每张图生成即入库 |
| Notion | 每阶段一个**子页**（A9 要求 pages 而非 page） | 每阶段收尾 |

---

## 7. 变更记录

| 时间 UTC | 变更 |
|---|---|
| 2026-08-11T20:30Z | 建档；worktree 建立；三路调研发出；算力快照记录 |

---

## 8. 事故记录：正式训练在 step 12,735 被 SIGKILL（2026-08-12T04:35:50Z）

`sacct` 证据：

```
5980745.249 | CANCELLED by 1483804540 | ExitCode 0:9 | 02:46:01 -> 04:35:50 | 01:49:49
```

`0:9` = SIGKILL，取消者是本账号 uid。**未执行过 `scancel`**（明令禁止）。真实机制是
**生命周期耦合**：该 `srun --overlap` step 是我会话里后台 shell 的子进程，上一个
Claude Code 进程退出时把它一并带走。

> **教训**：attach 的风险不只是共享分配的显存争用，更致命的是**会话一没、step 就死**。
> `sbatch` 提交的作业由 slurmctld 拥有，与会话无关。
> **判据：短任务（微基准、评测）可以 attach；长训练必须 sbatch。**

### 损失与止损

| 项 | 值 |
|---|---|
| 已完成 | 12,735 / 32,000 步（39.8%） |
| **已落盘 checkpoint** | 3000 / 6000 / 9000 / **12000** |
| 目录 | `checkpoints/j5980745_2ska8a8q_5980745` |
| 意外收获 | **baseline 也有 3000/6000/9000/12000** → 可做等步数受控对照，不必等重训 |

### 修正后的执行序

1. hybrid 训练**重新 sbatch**（独立生命周期），跑满 32,000 步
2. bench hybrid **@12000**
3. bench baseline **@12000**（不能拿 @32001 比，会把架构差异与训练长度混在一起）
