# Handoff：valset_v1 验证集与 IsoFLOP 标度分析（含会话溯源）

写于 UTC 2026-08-02T20:15Z。本文档的用途有两个：一是给出这条工作线的完整索引，二是记录产生这些结果的对话会话标识，以便日后需要时可以回到原始上下文查证任何一个决策是怎么做出来的。所有路径均为绝对路径。

## 1. 会话溯源

这条工作线的全部对话记录在同一个会话文件里。会话期间经历过多次上下文压缩与客户端重连，但记录始终追加写入同一个文件，没有分裂。

| 项 | 值 |
|---|---|
| Session ID | `79e7e513-c9d4-4f7e-adf4-9c761190316e` |
| 记录文件绝对路径 | `/lus/lfs1aip2/projects/public/u6gb/.claude/projects/-lus-lfs1aip2-projects-public-u6gb/79e7e513-c9d4-4f7e-adf4-9c761190316e.jsonl` |
| 文件格式 | JSON Lines，每行一条记录（用户消息、助手回复、工具调用与返回各占若干行） |
| 记录条数 | 约 3,350 行（会话仍在继续时该值会增长） |
| 文件大小 | 约 9.5 MB（同上，会随会话增长） |
| 首条记录时间 | 2026-07-29T12:56:55Z |
| 末条记录时间 | 2026-08-02T20:12:27Z |
| 工作目录 | `/lus/lfs1aip2/projects/public/u6gb` |

**如何恢复这个会话继续对话**：

```bash
cd /lus/lfs1aip2/projects/public/u6gb
claude --resume 79e7e513-c9d4-4f7e-adf4-9c761190316e
```

**如何在记录文件里检索特定内容**：文件是纯文本 JSON Lines，可以直接用 `grep` 单文件检索，这对 Lustre 元数据没有压力。检索时优先使用高选择性的锚点，例如提交哈希、Slurm 作业号、Notion 页面 ID，而不是常见词。

```bash
JSONL=/lus/lfs1aip2/projects/public/u6gb/.claude/projects/-lus-lfs1aip2-projects-public-u6gb/79e7e513-c9d4-4f7e-adf4-9c761190316e.jsonl

grep -c 'manifest_backfill124'  "$JSONL"    # 早期补评这一段
grep -c 'leak_350M-s5'          "$JSONL"    # 泄漏检验这一段
grep -c '3b012c4568fd8155'      "$JSONL"    # 推送到 Notion 的交接页这一段
```

上面三条命令返回的命中数会随会话继续而增长，因为后续讨论同一话题时同样的字符串会被再次写入，所以命中数本身不是稳定标识，只用来定位话题所在的位置。若要读取某一段的完整对话内容，可以用 Python 逐行解析并按时间戳过滤：

```bash
python3 - <<'EOF'
import json
JSONL = "/lus/lfs1aip2/projects/public/u6gb/.claude/projects/-lus-lfs1aip2-projects-public-u6gb/79e7e513-c9d4-4f7e-adf4-9c761190316e.jsonl"
KEY = "bootstrap"                      # 想找的关键词
for line in open(JSONL):
    rec = json.loads(line)
    txt = json.dumps(rec, ensure_ascii=False)
    if KEY in txt:
        print(rec.get("timestamp", "?"), rec.get("type", "?"), txt[:200])
EOF
```

**注意一个容易混淆的点**：本次会话过程中出现过若干形如 `30b9411a-...`、`4a8c0a39-...`、`13e47414-...` 的 UUID，它们是临时工作目录与后台任务的标识，**不是** session ID，磁盘上没有对应的 JSONL 文件。唯一的 session ID 就是上表中的那一个。

## 2. 这条工作线做了什么

会话跨越四天，包含三个前后依赖的阶段。

**第一阶段，建立并验证冻结验证集 `valset_v1`。** 从 S&P 500 限价订单簿语料中取出 5,367,734 个从未被任何训练接触过的样本窗口，一次性冻结。零泄漏由三层独立证据支撑：全量消费核查（逐一提取全部训练记录的实际步数，确认最深消费为 16.63%，低于 20% 的排除线）、逐样本位置核验（每个样本在三个数据种子的排列中的位置均在 20% 之后）、以及一个独立的行为学实验（在 78M 与 350M 两个模型上比较"确定见过"、"确定没见过"、"验证集"三组数据的交叉熵，确认模型对见过的数据没有任何损失优势，且验证集在构成调整后与未见数据不可区分）。

**第二阶段，把验证集用于 scaling-law 分析。** 对 33 条训练链在磁盘上现存的全部 436 个 checkpoint 评测验证集交叉熵，按固定算力切片做 IsoFLOP 分析，得到"最优模型规模随算力增长"的标度指数。

**第三阶段，量化该指数的不确定度并定位其边界。** 发现点估计 0.46 的链级 bootstrap 置信区间宽达 [0.12, 0.56]，根因是每个算力切片的抛物线左臂只由 1 到 3 条训练链支撑，而这源于训练时 checkpoint 保留策略删除了小模型的早期存档，无法通过再评测弥补。

## 3. 文档索引

以下文档构成完整的交付，按阅读顺序排列。路径均在 `/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/` 之下。

| 文档 | 内容 | 面向的读者 |
|---|---|---|
| `VALSET_V1_REPORT.md` | 验证集的构造方法、零泄漏三层证明、统计性质、使用方法、质量审计、行为学泄漏检验结果 | 需要理解这把尺子是否可信的人 |
| `VALSET_ISOFLOP_ANALYSIS.md` | IsoFLOP 分析正文：方法、逐切片结果、三层敏感性分解、根因、后续方案 | 需要引用标度指数的人 |
| `hand_off_valset_isoflop_436.md` | IsoFLOP 分析的技术交接：术语定义、复现命令、运维坑、方法教训 | 需要接手继续做分析的人 |
| `handoff.md` | 本文档：会话溯源与全局索引 | 需要回溯任何决策来龙去脉的人 |

英文版：`VALSET_V1_REPORT_EN.md` 是验证集报告的完整英文翻译，用于对外分发。

## 4. 数据与代码索引

| 路径 | 内容 |
|---|---|
| `artifacts_valset_v1_j5790795/` | 验证集索引清单、逐样本解码、三档嵌套子集、构造 manifest、SHA-256 校验 |
| `squashfs/output/` | 验证集实体数据包，两档 squashfs 文件（359 MB 与 3.51 GB），格式与训练数据包一致 |
| `leakage_exp/` | 行为学泄漏检验：测试脚本、三组样本索引、逐 batch 损失、结果 JSON、年份分层分析脚本 |
| `valset_eval/valset_ce_436_master_table.csv` | 436 点全轨迹主表，每行一个 checkpoint 的完整评测结果 |
| `valset_eval/valset_ce_436_fitready.csv` | 拟合输入格式，损失列装验证集交叉熵，算力列为解析式 6ND |
| `valset_eval/valset_isoflop_robust.py` | 三种顶点估计方法与噪声定标的验收判据 |
| `valset_eval/valset_isoflop_bootstrap.py` | 标度指数的链级 bootstrap |
| `valset_eval/valset_isoflop_436_20260801T122539Z_*` | IsoFLOP 抛物线族图、谷底汇总图、逐切片结果 JSON |

## 5. 关键数字速查

| 量 | 值 |
|---|---|
| 验证集样本数 | 5,367,734（占全域 1.661%） |
| 零泄漏边界 | 全部训练最深消费 16.63%，排除线 20% |
| 泄漏检验，78M：见过与未见过的交叉熵之差 | −0.0002 nats，95% CI [−0.0073, +0.0070] |
| 泄漏检验，350M：同上 | −0.0015 nats，95% CI [−0.0060, +0.0030] |
| 泄漏检验，验证集与未见数据之差（构成调整后） | 78M +0.0036、350M +0.0013，两者置信区间均含零 |
| IsoFLOP 测量点总数 | 436（132 终末 + 124 早期补评 + 180 加密） |
| 标度指数点估计 | 0.4618 |
| 标度指数 95% 置信区间 | [0.1216, 0.5573] |

## 6. 唯一尚未执行的后续实验

若需要把标度指数定到小数点后一位，唯一可行的办法是重新训练 6M、10M、14M 三个规模并保留早期 checkpoint，从而把低算力切片的左臂从 3 条链（单一规模）扩充到 12 条链（四个规模）。这三个规模原本都训练过，重训不需要训到收敛，只需跑到对应步数（6M 约 3 万步、10M 约 2.5 万步、14M 约 2 万步）。最关键的前提是必须先把 checkpoint 保留策略改为全量保存，否则重训一遍会再次删掉早期存档。

是否投入这九个 run 的算力，取决于结论需要多强，是一个需要人拍板的取舍。截至本文档写作时尚未获授权，因此未执行。详细论证见 `VALSET_ISOFLOP_ANALYSIS.md` 第 6 节。

## 7. Notion 同步

`hand_off_valset_isoflop_436.md` 已推送至 Notion，位于页面 "fit scaling law on validation loss" 之下。

| 项 | 值 |
|---|---|
| 子页面 URL | https://app.notion.com/p/Handoff-valset_v1-IsoFLOP-436-3b012c4568fd81558d05c8a9bbdbc718 |
| 子页面 ID | `3b012c45-68fd-8155-8d05-c8a9bbdbc718` |
| 父页面 ID | `3ad12c45-68fd-80ee-8f6a-e656a3761028` |
| 推送方式 | Notion REST API，integration 名为 `cc`，token 路径由环境变量 `$NOTION_TOKEN_PATH` 指定 |

## 8. 相关提交

本条工作线在 `/lus/lfs1aip2/projects/public/u6gb` 仓库的提交记录如下，按时间倒序。注意 `tasks/` 目录在 `.gitignore` 中，因此这些文件都是用 `git add -f` 强制加入的。

| 提交 | 内容 |
|---|---|
| `15d41a7` | IsoFLOP 分析的技术交接文档 |
| `ec69a02` | 436 点最终表，磁盘上全部 checkpoint 评测完毕 |
| `f85fdaf` | 标度指数改用 bootstrap 置信区间报告，而非方法间跨度 |
| `3ecf6ba` | 加密到 432 点，含采样密度敏感性分析 |
| `66670a5` | 稳健顶点估计与双值报告口径 |
| `37cd710` | 256 点全轨迹表与首版 IsoFLOP 拟合 |
| `554f147` | 弹性评测队列的空闲节点 fan-out 脚本 |

---

*本文档写于 2026-08-02。若需查证任何结论的推导过程或某个决策的当时理由，请按第 1 节的方法回到会话记录。*
