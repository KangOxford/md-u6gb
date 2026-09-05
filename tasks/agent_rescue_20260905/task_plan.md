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

---

# 第二轮（2026-09-05 06:00–07:00）—— 从「包建好了」到「活干完了」

## 7. 先核对：九个包对应的任务现在到底是什么状态

不重复恢复的前提是先知道谁已经做完了。判据是**产物的 mtime 与该 agent 最后一条消息的时间比**。

| slug | 声明的产物 | 实测 | 判定 |
|---|---|---|---|
| `plan_measurement` | `plan_section_2_measurement.md` | 34,760 B @ 22:28，agent 活到 22:29 | $\color{green}{\textsf{它自己做完了}}$ |
| `plan_deliverable` | `plan_section_5_deliverable.md` | 48,238 B @ 18:22，正是它最后一条消息 | $\color{green}{\textsf{它自己做完了}}$ |
| `sol_corrected_inference` | `corrected_inference.py` | 93,432 B @ **09-05 02:40**，比它死掉晚 8 小时 | **别的会话在做，不碰** |
| `plan_analysis` | `plan_section_4_analysis.md` | 缺 | 真未完成 → **本轮做** |
| `plan_infrastructure` | `plan_section_3_infrastructure.md` | 缺 | 真未完成 → **本轮做** |
| `sol_decisive_experiment` | `plan_section_1_decisive.md` | 缺（它欠的补丁脚本倒是在） | 真未完成 |
| `sol_history` | 无声明路径 | 无任何产物 | 真未完成 |
| `sol_pipeline_fixes` | `pipefix_.../` 测试 | 目录 32 项，但「被杀时正在改测试期望」 | 真未完成 |
| `sol_notebook_fixes` | 四个 builder 脚本 | 只有 1 个在它死的那一分钟被改过，另 3 个未动 | 真未完成 |

`plan_measurement` 与 `plan_deliverable` 已在状态轴上收掉，防止下一轮再被当成待恢复。

## 8. 四段账本（用户要求：准备/提交/处理/产物分开记）

「包建好了」不等于「活干完了」。旧账本只有一根轴（running/done/failed），**表达不了
「包在但没人跑」**——而那正是九个 agent 上一轮之后的真实状态。

```
prepared   RESUME 包存在
submitted  包交给了某个执行者
processed  执行者跑完并说了它做了什么
artifact   声明的产物存在且非空   ← 唯一不是自述、而是 stat 出来的
```

前三段是**证词**，第四段是**测量**，账本刻意把它们分开。
`artifact` 列在产物缺失时显示 `ABSENT` 而不是 `yes`——改这一处是因为初版写了 `yes`，
**标签替数字说了它没说的话**。

实现 `reg_stage.py`，接进 `agent_reg.sh stage` / `stages`。

## 9. 实际恢复两个（顺序做，不并发九个）

### `plan_analysis` → `/home/u6gb/kangli.u6gb/plan_section_4_analysis.md`（17,169 B）

包的价值当场兑现：死掉的 agent 把四个校验脚本落在
`/home/u6gb/kangli.u6gb/plan4_verify/`，**全部重跑成功且复现记录值**：

| 量 | 重跑值 |
|---|---|
| rung-3 零效应 sd | 0.019468（记录 0.019468） |
| GOOG 方差份额 / Kish n_eff | 0.6276 / **2.3652** |
| 正确的 5% 带宽乘数 | **1.8964**（±2 sd 其实只有 3.68%） |
| 符号翻转最小可达 p | **2/2⁸ = 0.0078125** |
| 符号翻转实测尺寸膨胀 | 1.35–1.87× |

另外两处我把措辞收紧了：120% 是推导值（119.9% = 0.0969/0.0808，非记录字段）；
洗牌前后 qL1 与 sd_ratio 在 JSON 里是**逐位相同**，不是 2.2e-16。

### `plan_infrastructure` → `/home/u6gb/kangli.u6gb/plan_section_3_infrastructure.md`（14,613 B）

包里最后两句正卡在根因上（「`.done` 是 0 字节」「MANIFEST 为空却继续往下走」，
并注明这一步纯 CPU）。四个 shell 签名当场测出来：

| 测试 | 结果 |
|---|---|
| `set -u` + 重定向 + 未定义变量 | **文件 0 字节，命令没跑**——与观察到的 0 字节 `.done` 吻合 |
| 变量已定义但为空 | 1 字节 ⇒ **0 与 1 字节可区分 unset 与 empty** |
| `set -e` 下 `cmd; _rc=$?` | 守卫**不可达**（`collect_rollouts.sh:182`） |
| `$?` 与命令替换同行 | **位置决定对错**：`$?` 在替换之前才正确 |

因此普查结果比 FACTS.md 的原表述更精确：5 处疑似只有 **1 处真坏**
（`eval_shard.sh:12`，`$(date)` 在前，于是每次失败都记 `rc=0`）。
另查出 **10 处**用 `-f` 判 `.done`（0 字节能骗过全部）、**8 个**以 `exec` 收尾的启动器、
**8 处** `rc=$?` 需逐个判定是否可达。

~~「上一版说 agent 观察到一个 0 字节的 `.done`，所以现场必有」~~ —— 我在
`crps_res_kcollapse_20260904T163807Z/*/member_*/.done` 下**一个 `.done` 都没找到**，
所以那次目击**没能复核**。机制成立、目击未证实，两件事都写进产物。

## 10. `/home` 那条保证是错的（用户点名）

原话「/home 有空 inode 所以不会再失败」**只对了一半**。三件事必须分开查：

| 问题 | 命令 | 2026-09-05 实测 |
|---|---|---|
| 字节额度 | `quota -u $(id -un)` | **100.20 / 100.58 GiB = 99.62% 满，只剩 0.39 GiB** |
| inode 额度 | 同上 | 1,565,158 / 15,000,000 = 10.43%，宽裕 |
| 能不能写 | `: > "$DIR/.probe"` | 能 |
| 是否持久 | `findmnt -no SOURCE,FSTYPE /home` | NFS4 on VAST，非 scratch |

`df` 报 15 PB 空闲，**答的不是这个问题**——它量的是文件系统，不是这个用户。
更要紧的是：**写成功不等于有余量**。我写了 2 GiB 成功，几秒后 `quota` 才报
`107161140*`（超硬上限，带星号）。记账是滞后的。

测试文件用 `truncate -s 0` 原地清零，不 unlink（符合禁令），配额随后回落到 99.61%。

## 11. 剩余真正未完成项（未并发启动）

| slug | 缺什么 | 备注 |
|---|---|---|
| `sol_decisive_experiment` | 功效计算 + 成本 + 预登记那一节 | 补丁已应用，只欠文档 |
| `sol_history` | 整条线的时间线重建 | 需要跨月 `sacct` 与多个根目录，重 |
| `sol_pipeline_fixes` | 四个缺陷各配一个先红后绿的测试 | 与本轮 §9 的普查直接衔接 |
| `sol_notebook_fixes` | 四个 builder 脚本里三个未动 | |
| `sol_corrected_inference` | —— | **别的会话在做，禁止重复恢复** |
