# subagent 断线恢复 —— 任务计划

**日期**：2026-09-05  **仓库**：`KangOxford/md-u6gb`
**实现所在**：`/lus/lfs1aip2/projects/public/u6gb/.claude/agent_registry/`
（该路径是 VAST `/home` 的符号链接，不在 Lustre 上——这一点是有意的，见 §2）

---

## 1. 起因

2026-09-04 夜里九个 subagent 随会话一起没了。次日 SessionStart 钩子报出来的是：

```
progress: 1705 lines at progress.md
prompt:   prompt.txt
```

九行一模一样，路径还是相对的。看上去像「工作还在」，实际上一个字都不在那里。

---

## 2. 核验：哪些已落地，哪些只是声明

全部为实测，不是读代码推断。

| 声明 | 实测 | 结论 |
|---|---|---|
| transcript 只在节点本地 tmpfs，断线即失 | `/home/.../subagents/agent-*.jsonl` 有 **603 份**，横跨 **70 个 session 目录**，逐条消息追加 | $\color{red}{\textsf{声明错误}}$ ——存在持久副本 |
| 九个 agent「什么都没存下」 | 九个**全部**有 416 KB–5.6 MB 的完整 transcript | $\color{red}{\textsf{声明错误}}$ |
| 九个 agent 的 prompt 丢了 | prompt **就在同一条 registry 行里**（`prompt_inline`，1369–2349 字符） | $\color{red}{\textsf{声明错误}}$ ——报告没去读它 |
| `sol_notebook_fixes` 「NO TRANSCRIPT」 | 它有 **5.6 MB** transcript，两处都有 | $\color{red}{\textsf{标签错误}}$ ——已用追加行更正 |
| `work/<slug>/progress.md` 是 layer 2 | `plan_measurement` 确有 21 行真实内容；另外 8 个为空 | $\color{green}{\textsf{部分落地}}$ |
| `agent_reg.sh rescue` 从 tmpfs 抢救 | 能跑，本次抢救 20 份 | $\color{green}{\textsf{已落地}}$，但只是冗余 |
| SessionStart 钩子自动触发 | `settings.json` 里确实挂着，本次实测输出正常 | $\color{green}{\textsf{已落地}}$ |

**layer 2 为什么会空**：registry 行里自己写着——
`"prompt.txt writes failed on a full filesystem at 18:30Z"`。当时 Lustre 项目 inode
打到硬顶。现在 registry 已经搬到 VAST `/home`（22.68 亿 inode 空闲），
而 Lustre 侧仍是 50,463,435 / 51,200,000（98.6%）。**这不是保留问题，是放置问题。**

---

## 3. 做了什么

| # | 项 | 文件 | 状态 |
|---|---|---|---|
| 1 | `pending` 改成三源报告（banked / prompt / transcript），并且只在三源全空时才说不可恢复 | `reg_report.py` | $\color{green}{\textsf{done}}$ |
| 2 | 未记录 work 目录时按 slug 约定推断，但**标注 `[INFERRED]`** | `reg_report.py` | $\color{green}{\textsf{done}}$ |
| 3 | transcript → 机械重建（无模型参与） | `extract_progress.py` | $\color{green}{\textsf{done}}$ |
| 4 | `recover` 生成可直接交给替补 agent 的 RESUME 包，**重入不重做** | `reg_recover.py` | $\color{green}{\textsf{done}}$ |
| 5 | `fail` —— 死掉 ≠ 做完，两者必须可区分 | `agent_reg.sh` | $\color{green}{\textsf{done}}$ |
| 6 | `add` 把 prompt 同时写进文件与行内 | `agent_reg.sh` | $\color{green}{\textsf{done}}$ |
| 7 | `verify` 六项检查 + `--self-test` 证明每一项都能变红 | `reg_verify.py` | $\color{green}{\textsf{done}}$ |
| 8 | SKILL.md 更正核心事实 | `skills/agent-rescue/SKILL.md` | $\color{green}{\textsf{done}}$ |

**查重（用户要求）**：新增摘要抽取前先查过——`journal.sh` 是**前瞻**记录（agent 自己写），
`rescue` 只做拷贝，仓库内 `.claude/**/*.py` 无任何读 transcript 的东西。
所以 `extract_progress.py` 的**回溯**抽取不重复。且它必要：九个 agent 的 layer 2 全空，
从 transcript 重建是唯一能取回它们的路。

---

## 4. 验证结果

| 验证 | 命令 | 结果 |
|---|---|---|
| 九个全部可恢复 | `agent_reg.sh pending` | 9/9 `RECOVERABLE` |
| 产物真实非空 | `stat` 逐个 | prompt 1369–2353 B；重建 11–26 KB；RESUME 2.8–3.9 KB |
| 失败显式记录 | 重建文件表头 | 3/3/3/0/0/0/0/1/1 条 `is_error` |
| **重入不重复执行** | 连跑两次 `recover --all` | 第一次 **9 built**，第二次 **0 built / 9 noop** |
| 自检能变红 | `reg_verify.py --self-test` | **4/4 变红** |
| 真实数据 | `agent_reg.sh verify` | **6/6 通过** |
| 端到端 add/done/fail | 冒烟 agent | 落两处、done 后消失、fail 单独成节 |
| 钩子 | `session_start_hook.sh` | 抢救 20 份 + 正常报告 |

**自检自己先红过一次**：头一版 `--self-test` 里两条「绿得可疑」，
查出来是自检改的是**原始行**，而那行后面被 `done` 行覆盖，合并后根本不在 running 集合里——
**自检犯的正是它要抓的那一类错**。改成从合并视图取目标后 4/4 变红。

---

## 5. 剩余项

| # | 事 | 为什么还没做 |
|---|---|---|
| R1 | `recover` 生成的包**没有真的交给替补 agent 跑过一遍** | 需要起 agent；本轮范围是机制本身。包是静态文件，正确性靠 `verify` 与人读 |
| R2 | 机械重建**保留不了 agent 的判断**——它给全部工具轨迹，不给「哪一步重要」 | 这正是 layer 2 存在的理由；两层要一起留 |
| R3 | 本会话自己跑在 `SLURM_JOB_ID=6324128 step 8`（nid010561） | 分配到期这条会话就没了。**这正是上面那张三事件表里的第 3 种**，属于已知、已有恢复路径 |
| R4 | `.recover_stamp.json` 只比对 size+mtime | 原地等长改写检测不到。对追加型 JSONL 不会发生 |
| R5 | 九个包里的活**还没续跑** | 恢复机制与续跑是两件事；用户没要求这轮续跑 |

---

## 6. 没动的东西

- 其他会话：一个没碰。
- 已有巡查：`gpu_watch_15min.log` 03:56:24 还在更新（本轮 03:59 查），**未新起、未停掉**。
- `registry.jsonl`：只追加，没改没删。旧脚本按既有约定 `cp` 成 `.bak_<时间戳>`。
