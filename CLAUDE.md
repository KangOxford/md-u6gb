# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# 🚨🚨🚨 ISAMBARD-AI / LUSTRE Cluster Safety (Highest Priority, 2026-05-08)

**Isambard-AI is a BriCS national-level HPC, with 1000+ users sharing the Lustre parallel filesystem. Metadata storms triggered by LLM agents have already caused the entire group's jobs to be suspended once (BriCS administrators documented this explicitly in `/lus/lfs1aip2/projects/public/s5e/quant_team/quant/isambard-requirement`). Any violation of the rules below may lead to the team being banned again. These rules take precedence over all other content in this file, including "First Principles."**


## 1. Absolutely Prohibited Commands (DO NOT EXECUTE)

The following commands can create metadata storms for the Lustre MDT and are **forbidden, even for single "investigative" executions**:

```bash
ls -lR <path>            # Any recursive ls
ls -R <path>
ls -1td <dir>/*          # ls + time sort, hits every inode
ls | grep | sort         # ls followed by pipe (typical anti-pattern for finding checkpoints)
ls -la <large_dir>       # Directories with > 50 entries are considered large; do not try if size is unknown
find / ...               # Any broad find starting from /, /projects, or /home
find /projects ...
find /home ...
du -sh <large_path>      # Recursive stat
tree <lustre_path>       # Recursive listing
locate ...               # System-level search (depends on global index scanning)
**/*.ext                 # Deep recursive glob
*.py                     # Glob expansion in large directories (>10 files is unsafe)
watch ls /path           # Continuous metadata load
while true; do ls; sleep 5; done
inotifywait <lustre_path>
```

### 1.1 `scancel` is FORBIDDEN — one exception: the training已经废了 (2026-08-06, 例外 2026-08-20)

**🚨 2026-08-20 用户令，唯一例外：训练本身已经失败时，允许直接 `scancel`，不必等批准。**
「已经失败」指的是**继续跑下去不可能产出可用结果**的状态，要有日志证据：

| 允许直接杀 | 证据 |
|---|---|
| loss 变 `nan` / `inf` 且不再恢复 | 日志里连续多条 `loss=nan`（本次：基准段 44 条，跨 4400 步） |
| loss 单调爆炸、永不回落 | h192 那种 3075→4878 的形态 |
| 进程还活着但一步都不推进 | 步数长时间不变 |

**杀之前必须同时处理接续链**：先杀 deferred submitter，再杀训练段。反过来的话，
训练段一死，submitter 立刻从同一个坏检查点再起一段。

**形式仍受限**：只用 `scancel <jobid>` 这一种写法。**不许带任何 flag，不许带 `.<step>` 后缀**
——2026-08-06 的事故正是 `scancel -f 5924043.27` 被 Slurm 丢掉 `.27` 变成杀整个分配。

除此之外的一切情况（卡死、跑得慢、要腾资源、别人的作业）仍然是**报给用户，我不动手**。



**Claude must never run `scancel` in any form.** Not on a job, not on a job step,
not with `-s`, `-f`, `-u`, `--me`, or any other flag. If a job or step needs to be
stopped, **tell the user and let them do it**.

```bash
scancel <jobid>              # FORBIDDEN
scancel <jobid>.<stepid>     # FORBIDDEN — see below, the step form is not safe either
scancel -s KILL ...          # FORBIDDEN
scancel -f ...               # FORBIDDEN
scancel -u <user> / --me     # FORBIDDEN
```

**Why the step form is also forbidden.** `scancel <job>.<step>` looks scoped to one
step, but **adding any flag makes Slurm drop the `.<step>` suffix and act on the whole
job**. On 2026-08-06 a stuck training step was not responding to plain
`scancel 5924043.27`, so `scancel -s KILL 5924043.27` and `scancel -f 5924043.27` were
issued. Slurm answered
`scancel: error: Kill job error on job step id 5924043: No error` — note the missing
`.27` — and killed the entire 4-node allocation that had 22 hours left on it. It
survived only because that job happened to carry `Requeue=1`. The two spellings differ
by one flag and the blast radius differs by an entire allocation, so the whole family
is off-limits rather than "use it carefully."

**Do this instead**: report the stuck job/step id to the user with what is wrong, and
wait. A stuck step wastes minutes; a cancelled allocation wastes hours and re-queues
behind everyone else.

**Trigger Reflex Check**: Whenever I feel like using `find` / recursive `ls` / `du -sh` / `tree` / large directory glob to "understand project structure" or "find files," **STOP**. Use the safe alternatives below, or ask the user directly.

### 1.2 删除一律改成改名 —— `rm` / `rm -rf` 禁用（2026-08-13，用户第二次强调）

**任何要删掉东西的地方，改成 `mv` 加 `_deprecated` 后缀。** 不停下来等确认，
直接改名就继续干活——改名可逆、删除不可逆，而两者的代价几乎相同。

```bash
rm -rf runs/foo/checkpoints                       # FORBIDDEN
mv runs/foo/checkpoints runs/foo/checkpoints_deprecated_20260813T2230Z   # 这样做
```

适用范围是**一切**：checkpoint 目录、日志、中间产物、"反正是空的" 的目录、
"反正是我自己刚才写坏的" 文件。理由不是怕删错重要数据，而是
**"这个能删" 这个判断本身就是我最容易出错的地方**：崩溃的 run 到底写没写
checkpoint、那个目录是不是别的会话在共用、上一次是不是真的失败了——这些在
删除的瞬间都变成不可验证的。改名把判断错误的代价从 "永久丢失" 降到 "多一个目录"。

时间戳要带上，否则重跑第二次时 `_deprecated` 会撞名。清理由用户决定，不由我决定。

**🚨 2026-09-01 用户第三次强调，补一条执行细则：不许因为「等 `rm` 的批准」而让实验停下来。**
用户原话：「do not use the rm but use the rename (mv) do not because of waiting my approval
of rm to stop the experiments.」

犯法的形态是这样的：清理挂载残渣 / 腾目录时**先发一个 `rm`**，被权限层拦下，
然后**停在那里等批准**——而这段时间 GPU 空转、接续链断掉。
`mv` 从一开始就不需要批准，所以**根本不该出现那个等待**。

机械规则：**要清理什么，第一反应就写 `mv`，不写 `rm` 再改。**
它同时满足两件事——可逆，且不产生任何需要等待的审批点。

### 1.2.1 subagent 也受 `rm` 禁令，而且光在 prompt 里写一句没用（2026-09-03 用户令）

**事发**：起了 5 个 subagent，**五个 prompt 里我都写了 `NEVER rm (use mv with a timestamp)`**，
其中一个照样发了 `rm -rf BC`（它自己几分钟前建的暂存目录），被权限层拦下，用户点名。

**所以真正的教训不是「记得告诉 subagent」——我告诉了。** 是这条规则在
「清理我自己刚建的暂存目录」这个场景下不触发：那不像在删数据，像在收拾桌子。
禁令写在 prompt 顶部的安全清单里，而 `rm -rf BC` 出现在一条长命令的开头，
写的时候脑子在想 B/C 对照怎么搭，不在想禁令。**与「高频词的禁令最容易失效，
因为它出现在不需要思考的位置」是同一个机制。**

**两条修法，第二条才是根治的：**

1. **prompt 里把场景写出来，不要只写规则。** 不写「NEVER rm」，写
   「NEVER `rm` — including scratch, staging and temp directories you created
   yourself, and including `rm -rf` on a directory you are about to recreate」。
   规则要覆盖它将会用来说服自己的那个借口。

2. **把「需要清理」这件事从设计里去掉。** 要重复搭建一个暂存目录，
   **每次用一个新的唯一名字**，就不存在要清空的东西：

   ```bash
   BC=BC_$(date -u +%H%M%S); mkdir -p "$BC/run/mid_training"   # 不清理，换名字
   ```

   真的必须让开某个固定路径时才改名：
   `mv BC BC_deprecated_$(date -u +%Y%m%dT%H%M%SZ) 2>/dev/null || true`

**连带**：让 subagent 写的**代码**也守同一条。缓存失效、覆盖旧产物、
supersede 一个文件——一律改名或原地覆写，不许 unlink。
我给它的规则和它写进仓库的规则必须是同一条，否则禁令只管到我这一层。

### 1.3 sigma-0 一律以 junming 身份 —— 检查点是 `gh` 的 active 账号（2026-09-05 用户第三次点名）

用户原话：**「alwys work as junming rather than kang oxford in the sigma-0.
you need to record this in claude.md」**。

**这不是 push 前的一次检查，是在 sigma-0 上工作的默认身份。** 凡是会在
`KangOxford/sigma-0` 留下署名的动作——commit、push、开 PR、发或改 PR 评论、
改标题正文、提 issue、加标签、任何非 GET 的 `gh api`——一律以
**junming / anjunming1202** 出现。

#### 两套凭据，只对一半等于没对

2026-09-05 实测的违规形态：

| 决定什么 | 由谁决定 | 当时实测 |
|---|---|---|
| commit 的 Author / Committer | `git config user.name` | `junming` ✅ |
| **PR author、PR 评论作者、HTTPS push 的认证身份** | **`gh` 的 active 账号** | **`KangOxford`** ❌ |

三个 sigma-0 检出的 git 身份**全对**，同一时刻 PR#77 上发出去的
**6 条评论、1 次改标题、1 次改正文、2 次改评论全部署名 KangOxford**。
`git log` 里一个字都看不出来。

**所以「我检查过身份了」在没写明查的是哪一半之前，不构成任何证据。**

#### 为什么前两版规则没拦住

规则写成「**每次 `git push` 之前**检查」。但走 `gh` 的写操作**根本不经过 push**，
而且发评论是我一天做几十次的动作、push 是一天几次——**规则挂在低频动作上，
高频动作就永远不触发。**

#### 机械做法：在 sigma-0 上做任何留痕动作之前

```bash
gh api user --jq .login            # 必须回 anjunming1202
gh auth switch --user anjunming1202
git config user.name               # 必须是 junming <anjunming0819@outlook.com>
```

用 `gh api user` 而不是 `gh auth status`：它回的是**这次请求实际会用的身份**，
一行、可判等、不用人眼从多行里挑。

`gh` 的 active 是**进程外的全局状态**，别的会话切过它我这边看不出来，而 LDM
（`KangOxford/Large-Discovery-Models`）必须用 KangOxford（`anjunming1202` 对它
无写权限，实测 403），这个开关会被来回拨。**所以是每次写之前查，不是会话开头查一次**；
也不要在还没要动 sigma-0 时提前把别人切走。

| 仓库 | 身份 |
|---|---|
| `KangOxford/sigma-0` | **junming / anjunming1202** |
| 其余一切（`Large-Discovery-Models` 等） | **KangOxford** |

**已经用错身份发出去的评论改不回来**——GitHub 不允许改评论作者。发现时如实报给
用户，不要删掉重发假装没发生过。


### 1.3.1 sigma-0 的活先看 PR#60（2026-09-03 用户令，保留）

推 sigma-0 之前先
`gh pr view 60 --repo KangOxford/sigma-0 --json state,mergeable,statusCheckRollup`，
确认它的状态与检查项，再决定这次要推的东西是叠在它上面还是另开分支。

**为什么要盯 PR#60**：sigma-0 的 PR 是**堆叠**的（一条 PR 的 base 是另一条 PR 的分支，
如 #60 的 base 就是 `feat/midtrain-return-alignment-evidence-20260818`）。不先看栈顶
就 push，新分支会从错误的 base 长出来，等到开 PR 时才发现 diff 里混进了下层 PR 的提交。

### ~~1.3.2 每次 push 前的两条硬性检查（2026-09-03 用户令）~~ 已被 1.3 取代，原文保留

#### 原 1.3 每次 push 前的两条硬性检查（2026-09-03 用户令）

**每一次 `git push`（以及每一次开 PR / 发 PR 评论）之前，必须同时满足：**

| # | 检查 | 具体做法 |
|---|---|---|
| 1 | **身份按仓库定，不是一刀切**（2026-09-03 用户订正） | **`KangOxford/sigma-0` 用 junming**：`git config user.name` = `junming <anjunming0819@outlook.com>`，`gh auth status` 的 active = **anjunming1202**。**其余一切仓库（含 LDM / Large-Discovery-Models）用 KangOxford。** 不对就 `gh auth switch --user <账号>` 切过去再 push |
| 2 | **sigma-0 的活先看 PR#60** | https://github.com/KangOxford/sigma-0/pull/60 —— 推 sigma-0 之前先 `gh pr view 60 --repo KangOxford/sigma-0 --json state,mergeable,statusCheckRollup`，确认它的状态与检查项，再决定这次要推的东西是叠在它上面还是另开分支 |

**为什么要盯 PR#60**：sigma-0 的 PR 是**堆叠**的（一条 PR 的 base 是另一条 PR 的分支，
如 #60 的 base 就是 `feat/midtrain-return-alignment-evidence-20260818`）。不先看栈顶
就 push，新分支会从错误的 base 长出来，等到开 PR 时才发现 diff 里混进了下层 PR 的提交。

**身份检查为什么必须每次做**：`gh` 的 active 账号是**进程外的全局状态**，别的会话切过它、
或者某次 `gh auth switch` 之后忘了切回来，我这边完全看不出来——直到 PR 的 author 字段
写出来才发现。所以是每次 push 前查一遍，不是会话开始时查一遍。

**2026-09-03 订正：这条原本写成「一律 junming」，是错的。** 正确的是按仓库分：

| 仓库 | 身份 |
|---|---|
| `KangOxford/sigma-0` | **junming** / anjunming1202 |
| 其余一切（`Large-Discovery-Models` 等） | **KangOxford** |

### 🚨 sigma-0 一律 junming（2026-09-05 用户第三次重申）

用户原话：**「always work as junming rather than kang oxford in the sigma-0」**。

**在 sigma-0 里，KangOxford 这个身份不该出现在任何地方** —— 不只是 `git push` 那一下：

| 位置 | 必须是 |
|---|---|
| `git config user.name` / `user.email` | `junming` / `anjunming0819@outlook.com` |
| commit 的 Author 与 Committer | junming |
| `gh` 的 active 账号（开 PR、发评论、加标签） | **anjunming1202** |
| PR 正文、评论、issue 里的署名 | junming |

**为什么反复失守**：`gh` 的 active 账号是**进程外的全局状态**，别的会话切过它我这边完全看不出来，直到 PR 的 author 字段写出来才发现。所以是**每次 push / 每次开 PR / 每次发评论之前查一遍**，不是会话开始时查一遍。

**不要用 `gh auth switch` 去改全局 active**（那会打断别的会话正在做的事）。用单命令凭据：

```bash
TOK=$(gh auth token --user anjunming1202)
git -c "http.https://github.com/.extraheader=Authorization: Basic $(printf 'x-access-token:%s' "$TOK" | base64 -w0)" \
    push <remote> <branch>
```

这样全局 active 不动，本次推送用对身份。**LDM / Large-Discovery-Models 反过来用 KangOxford，同样用单命令凭据，不切全局。**

发现的经过：LDM 那条线上 `anjunming1202` 对 `KangOxford/Large-Discovery-Models`
**没有写权限**（`gh api repos/... --jq .permissions.push` 返回 n/a，push 报 403），
于是「必须用 junming 推」与「notebook 必须推到远端才算交付」两条规则直接冲突——
按前者做，notebook 就交付不了。这种冲突本身就是规则写得太宽的信号：
**一条身份规则如果在某个仓库上让人无法工作，那它的适用范围一定被写宽了。**

## 2. Safe Alternatives

```bash
# Lustre native metadata-batched find (far safer than GNU find)
lfs find <narrow_path> -name "*.py"

# Read known exact paths directly
cat /known/exact/path
head -N /known/path
tail -N /known/path
grep "pattern" /known/single/file        # Single file grep is always OK
grep -r "pattern" /small/src/tree        # ONLY if src tree has < 100 files
grep -r "pattern" /path --include='*.py' # Must use --include to narrow scope

# To find the latest checkpoint → read the breadcrumb file, do not ls the checkpoint dir
cat <run_dir>/latest_checkpoint.json

# If you don't know what's in a directory → ask the user, do not probe with ls
```

`ls /known/dir` should only be used if **fewer than 50 entries are expected**; if unsure, ask the user.

## 3. Checkpoint / Training Script Mandatory Requirements

Any batch script or training code must satisfy:

1. **Write to node-local storage, not Lustre**: All checkpoints, tensorboard logs, and training logs must be written to `$TMPDIR` or `/local/scratch/` during the run. Only `rsync -a $TMPDIR/... /projects/...` after the job finishes.
2. **Use Breadcrumb files for resume**: Rank 0 must **atomically write** a small `latest_checkpoint.json` (containing path + step) after every save. Auto-resume must read this file. **Strictly forbidden**: using `ls -1td` / `ls | grep | sort` on the checkpoint directory.
3. **Per-job isolated Output directories**: `OUTPUT_DIR=/projects/.../runs/${SLURM_JOB_ID}/` to avoid directory-level lock contention.
4. **Stage model weights first**: Rank 0 should copy weights to `$TMPDIR` and then broadcast. Multiple ranks are forbidden from reading large weight files from Lustre simultaneously.
5. **Use local cache for W&B, but keep runs online**: set `WANDB_DIR=$TMPDIR` and `WANDB_MODE=online` for training/eval jobs that should be tracked. Do not write W&B tokens into `CLAUDE.md`, Notion pages, shell scripts, or committed files; use the existing login state or a protected environment/secret source.
6. **Tighten Checkpoint intervals**: Save every N steps instead of every step. Choose N such that "recovering once wastes < 10% of job time."

### 🚨🚨🚨 4.-3 计划优先，GPU 次要——不许照着 gtop 临时找活（2026-09-04 用户令）

**用户原话**：「you cannot just open gtop every time, find available GPUs, and ad hoc run
tasks… The key is to stick to the plan, not the GPUs… GPU availability is a secondary
issue; the most fundamental priority is the plan file.」

这条**修正**了 §4.0 一系列空卡规则被我执行成的样子。那些规则说的是「空卡的边际成本
是零，别拿成本当不做的理由」——**依然成立**。现在明确为错的是另一件事：
**让空卡决定做什么**。空卡是用来执行计划的；**等卡是可以的，因为卡空着就去跑一件
计划外的事不行。**

`gtop` 的节奏因此从 15 分钟**放宽到约 30 分钟**——重点不再是抢下每一个窗口。

#### 写计划的流程（不是可选的）

1. **五个 subagent 起草计划**，章节互不重叠，各写各的文件（合并时才不炸 context）。
2. **另外五个 subagent 对抗评审**这些计划文件。
3. 评审过了才开始执行，用当时空着的卡。
4. **计划文件随工作推进持续更新**：进展标绿
   `$\color{green}{\textsf{done}}$`，被证明搞错的**用删除线保留**（~~原文~~），
   不要删掉——**错在哪的记录本身就是产物**。

**大部分时间应该花在计划上，不是执行上。**

#### 为什么必须用 subagent 做对抗检查

用户原话：「If you don't use subagents, you tend to assume whatever you've done is
correct.」

**当天就在我自己身上验证了两次。** 重测数据落地后我一小时内就把结论推上讲稿，
事后补做的对抗检查在我自己的分析里找出三处缺陷（噪声底借自另一个配置、跨 block
比较不安全、方差差 8.5 倍还用合并检验），外加一个 20 倍的学习率混淆——它让那个
头条对比根本不是我声称的那个对比。

四条规则，每次都做：**① 永远先 profile ② 永远维护 plan 文件 ③ 永远跑对抗检查
④ 永远找文件删、找代码重构。**

## 4. Job Submission Pacing

### 🚨🚨🚨 4.-2 巡卡必须是一个常驻 Monitor，不是我记着去做（2026-09-03 用户令，第四次犯）

**事发**：整个 2026-09-03 的会话里，我只在「用户提醒」或「我自己要起作业」时跑过 gtop，
**一次周期性的检查都没做过**。装上 Monitor 后**第一轮就报**：60 张真空卡 + 12 个作业
在等 Priority。用户原话：「have you followed the restrictions in claude.md, check gtop
every 15 mins, you should have a monitor do this」。

**下面 4.-1 的 R1 早就把这个失效方式写清楚了**——「时间驱动的规则对任务驱动的我永远
不触发」——而我读过那一节，仍然没装监视器。**所以 R1 的修正（挂到我一定会做的动作上）
是不够的：它仍然依赖我在那个动作发生时想起来。唯一可靠的办法是把检查交给一个不是我的
东西去执行。**

**硬规则：每个会话开始时（或恢复后）立刻挂一个常驻 `Monitor`，15 分钟一轮，
两侧同查，只在可行动时发事件。** 挂之前不许开始别的工作。

```
Monitor(persistent=true, timeout_ms=3600000, description="idle GPUs vs PENDING, both sides")
  每 900 秒：
    gtop --once  →  只数逐卡行（头行的 idle 把 held 也算进去，见 4.0.1）：
        awk '/^ ▸ job/{j=$3} /^   nid/{n=$1}
             /GH200/{ if ($0 ~ /idle/ && $0 ~ /mem +0\.0\//) {
               match($0,/\[[0-9]\]/); print j, n, substr($0,RSTART+1,1) } }'
    squeue -u $USER -t PENDING -h -o "%.10i %.26j %.4D %R"
    发事件的条件（其余时候保持沉默）：
      · DependencyNeverSatisfied ≥ 1        → 死作业，我不能 scancel，立刻报给用户
      · 真空卡 ≥ 4 且 Priority/Resources ≥ 1 → 本可 attach 却在排队
      · gtop 无输出                          → 检查本身失效了，也要报
```

**为什么条件里必须带 PENDING**：供给侧一个人不构成决策。「有空卡」不是事件，
「有空卡**而且**有活在等」才是。反过来「我的队列空了」也不是空转的理由——
派单清单是**账户的** PENDING 队列，不是我私人的 backlog。

**收到事件后的动作**（照 4.0 既有分工）：本战线的排队作业直接 attach，不问；
别的战线的**立刻报给用户 + 给出 job id**，不 scancel、不起副本（会与排队的那份
同写一个输出目录）。**「要问一下」不是什么都不做的理由——报告本身要立刻发。**

### 🚨🚨🚨 4.-1 为什么 4.0 这条规则我反复不执行（2026-08-18，第三次犯，必须先读这一节）

**事发**：2026-08-18 17:48–19:5x，**36 张自有的卡整整空转 2 小时 10 分（6.30 kW）**，
同时队列里 10 个作业在等。用户原话：**「这是有病还是咋的？这么多空的不用，你在那排队。」**
这是同一个错误的第三次（2026-08-14「你看着空的 GPU 不用，你去排队是吧」、2026-08-16 XVLA 四连发）。

**前两次我都在 4.0 这里加了更强的措辞，然后又犯了。所以问题不在措辞，在触发机制。**
四条根因，按造成的损失排序：

#### R1 —— 我把 GPU 当成「一个我在做的任务」，而不是「一种绝不能空着的资源」

注意力转到分析/写代码时，**GPU 从我的世界模型里整个消失了**。4.0 写的是
「每 15–30 分钟巡一次」——那是一条**时间驱动**的规则，而我的执行是**任务驱动**的：
我不会在写分析写到一半时因为「过了 20 分钟」而中断自己。**规则的触发条件与我的实际
运行方式不匹配，所以它永远不会触发。**

> **修正：把巡卡挂到一个我一定会做的动作上，不挂在时钟上。**
> **硬规则：每一次我要输出一段长回复、或开始一项不用 GPU 的工作（分析、写文档、改代码）
> 之前，先查一次卡。** 这个触发条件我逃不掉，因为长回复和分析是我一直在做的事。

#### R2 —— 只起了一波，剩下的没排队

21 个 bench，12 个位置。我起了 12 个，然后**就不管了**。两小时后前一波跑完，
没有任何东西接上去。

> **修正：作业数 > 位置数时，剩余的必须当场写成待起清单并说明何时起。
> 「起了一波」不算完成，「所有作业都已落地或已排好队」才算。**

#### R3 —— 看到空队列，我的第一反应是「查故障」而不是「填卡」

`squeue` 里没有我的 step，我读成「它们死了」，然后去翻日志查死因。
**它们其实是跑完了。** 而查死因花掉的那几分钟，卡是空的。

> **修正：空队列的动作顺序是固定的——先填卡，再查上一批怎么样了。**
> 这两件事不冲突，但**填卡必须在前**：查故障需要几分钟，填卡需要几秒。
> 而且无论上一批是死是活，下一批都要起，所以查故障根本不是填卡的前置条件。

#### R4 —— 一条线被挡住，我把所有线都停了

inode 配额打满 → 我写不了一个新的测试文件 → **我升级给用户并停下了全部工作**，
一停两小时。但 inode 满**根本没有挡住 GPU 计算**：那 9 个 bench 在 inode 满的情况下
照样把产物写完了（改写已存在的文件不消耗 inode，而它们的计算全在节点本地盘）。
**我把「建不了新文件」过度推广成了「什么都做不了」。**

> **修正：升级给用户之前必须先回答两个问题——(1) 这个阻塞具体挡住了哪一步？
> (2) 其他步骤还能不能走？** 只有当答案是「全都走不了」时才停下来等。
> **等用户回话的时间里，卡照样在烧。**

### 🚨 4.-0.5 gtop 必须由常驻监控每 15 分钟查一次（2026-09-03 用户令）

**不许靠「我记得查」。** 每个会话开始时、以及每次会话恢复后，**第一件事**就是确认
监控在跑；没有就立刻起一个，间隔 **15 分钟**，同时查空卡与 PENDING 队列。

会话拆卸会杀掉会话内的监控，而**拆卸不留痕迹**——事后看不出它什么时候停的。
2026-09-03 实测：连着两次恢复我都没重启监控，中间整段是无监控状态，直到用户点名。
所以「起过一个」不算数，**每次恢复都要重新确认**。

监控必须满足四条，缺一条它就会在最该报的时候沉默：

| # | 要求 | 为什么 |
|---|---|---|
| 1 | **探测失败 ≠ 0 空卡** | `gtop` 崩溃时 `grep -c` 返回 0，会被读成「没有空卡」，正好把浪费藏起来。要先认 `采集失败` / `Traceback`，报 `PROBE FAILED` |
| 2 | **带 `--timeout 60`** | 默认探测超时会让 gtop 整个崩掉（「N 个采集失败，读不到卡」） |
| 3 | **报逐卡位置，不报头行计数** | 头行把 held 也算进 idle；按 §4.0.1 要给出「哪个分配的哪个节点的哪张卡」 |
| 4 | **一并报 PENDING 与近期失败** | 供给侧一个人不构成决策（§4.-1 R1）；秒级 FAILED 不会自己浮出来 |

**为什么这条必须是一个进程，而不是一条我要记住的规则。** 这是 §4.-1 R1 的第四次复发，
而 R1 早就把病因写对了：规则挂在时钟上，我按任务驱动运行，**不会因为「过了 15 分钟」
而中断自己**。R1 给的修法是「挂到一个我一定会做的动作上」——**那条修法同样失效了**，
因为它仍然依赖我在写长回复前想起这件事，而写长回复时我在想内容。
2026-09-03 实测：连续几小时做 CPU 侧的活（打分、写 notebook、推送），期间一次没查卡，
用户点名时 **60 张空卡、5.37 kW 空转，同时 7 个作业在排队**。

**监控只报不做。** 它绝不提交、绝不 attach、绝不 scancel。这既是 BriCS 禁令的边界
（禁的是**无人值守下持续发起动作**，只读轮询并写日志不算），也满足 §4.0
「后台检查必须常驻，且绑资源不绑任务」。两条在这里不冲突，但**只有在它不发起动作时才不冲突**。

**静默必须有含义。** 无事时不写行，但每小时写一条 heartbeat。否则「日志是空的」
分不清是「集群很闲」还是「看守早死了」——与 [[reference_ps_is_a_shim_on_isambard]]
里「返回空要先分清是查过没有还是根本没查」同源。

**别的战线的排队作业不许直接 attach 副本**（会与排队的那份同写输出目录）——
报给用户 + 给出 job id，由用户决定。**但报告本身要立刻发，不许因为「要问一下」就什么都不做。**

现成的实现：`/lus/lfs1aip2/projects/public/u6gb/nb_build_pr22/gpu_watch.sh`，
日志 `.../nb_build_pr22/logs/gpu_watch.log`，跑在 `tmux -L cl43` 的 `gpuwatch` 窗口。

**必须活过会话的那一份放 tmux 或 Slurm 作业，不要放 `setsid nohup`。**
tmux 的记录器要设时限（如 12 小时）且**不自我重启**——登录节点常驻 agent 受 BriCS
禁令约束；时限到了由下一次会话重新起。日志写 Lustre，任何节点可读。

---

### 🚨🚨 4.0 硬标准：**每一次 `sbatch` 之前必须先跑 `gtop`，有空卡就 attach，绝不排队**（2026-08-14 用户定）

**这是命令，不是建议。没有例外。**

```bash
gtop                 # 先看：自己名下的分配里，此刻哪些卡是空的
                     # 空卡存在 → 用 srun --overlap 上去跑，不许 sbatch
                     # 真的一张空卡都没有 → 才允许 sbatch
```

**是 `gtop` 不是 `sgpu`，两者分工不同，别用错**：

| 工具 | 回答的问题 | 用在这里对不对 |
|---|---|---|
| **`gtop`** | **此刻**每张卡的物理占用（util / 显存 / 锁 / step） | ✅ 就是它。唯一不来自 Slurm 记账、直接读卡的测量 |
| `sgpu` | 过去 N 小时谁跑过、断了几次（甘特图账本） | ❌ 历史账面，回答不了「现在有没有空卡」 |

而且**要不断地查**，不是查一次就完事：排队等待期间要持续 `squeue` / `gtop` 轮询，
一旦自己名下出现空节点，立刻把工作挪上去，不要继续干等。

**判据（从 `gtop` 直接读，这是唯一不来自记账的测量）**：

| 信号 | 含义 |
|---|---|
| `steps: .batch only · nothing computing` | **整个分配的 GPU 全空**，最该抢的目标 |
| GPU `0%` 且显存 `0.0/95.6G` | `idle`，真空 |
| GPU `0%` 但显存 > 64 MiB | `held`，有人预留了显存，**不是空的** |
| `🔓` 无锁 | 可用（无锁节点即便在跑任务也可以用） |
| `🔒 <别人>` | 别人声明过，不碰 |
| `DEAD(闲置 N min)` | 锁还在但长期空转，问过用户再动 |

**血泪教训（2026-08-14）**：H3 复刻的两个作业在队列里干等了一整天，其间用户账户
名下有 **40 张 GPU 空转**（`6006783` 8 张、`6006424` 16 张、`6000409` 16 张，全部
`nothing computing`，各自还剩 5–10 小时）。我一次都没查过 `sgpu`，直接 sbatch 然后
等。用户原话：**「你看着空的 GPU 不用，你去排队是吧？」**

这条规则本来就以「Reuse active allocations」的形式写在下面，我没执行。现在提到 4.0
并升级为硬标准：**`sgpu` 是 `sbatch` 的前置条件，跳过它就是违规。**

**血泪教训第二例（2026-08-16，用户再令 "always think about using attach before any
sbatch"）**：XVLA 四连发（2 评测 + 2 训练重启）全走 sbatch 排队，同一时刻自己名下
`6022465` 的 **16 张卡显存 1-4 MiB 彻底空转、还剩 13h49m**。错误机制是把前一天的
「假窗口黑名单」当成了免检章——黑名单是**时点判定，会过期**：昨天它忙、今天它空。
三条修正入规：
1. **每一次 sbatch 前逐卡重查显存**（`nvidia-smi --query-gpu=memory.used` 经
   `srun --overlap` 到目标节点，或读 gtop 逐卡行），gtop 头行的 idle 计数把
   held（显存 86G+ 驻留）也算进去，**头行不可作判据**；1-4 MiB 才是真空。
2. **1N/1GPU 的小任务（eval/冒烟/benchmark）一律 attach-first**——暴露面小、
   分钟级起跑；sbatch 只留给需要独立 walltime 保障的多节点长段。
3. attach 版与排队版并存时用**不同 TAG/输出目录**防结果目录冲突，取先完成者。

**attach 起来之后必须真的在算。** 一个 attach 的 srun step **随启动它的会话一起死**，
所以要用 `setsid nohup srun ... &` 让它脱离会话进程组，否则会话一重启，卡就白占着
空转。起完 40 秒内用 `squeue -s -j <alloc>` 确认 step 在，并且 `gtop` 里 util 起来了；
没起来就是没在算，要么修要么让开。

attach 的具体机制见下面 4.1 起的各条。**注意其中的 `nodelock` 部分已于 2026-08-14 废止**。

### 🚨 4.0.2 巡查必须是常驻 monitor，不能靠我自觉（2026-09-03 用户令）

**用户原话：「check gtop every 15 mins, you should have a monitor do this」。**

§4.-1 的 R1 已经写明这条规则为什么不触发：**它是时间驱动的，而我是任务驱动的**——
我只在「要提交作业」或「要卡」时才查 gtop，两个触发点在时间上不重合。2026-09-03
又犯一次：整轮工作里只在需要卡时查过，没有任何定期检查。**结论是这件事不能靠自觉，
必须有一个进程替我做。**

**这与 §5 / §8.2 的「禁止在登录节点常驻 agent」不冲突**：那两条禁的是常驻
**AI agent** 和 **auto-restart 包装器**。一个每 15 分钟跑一次 `gtop` + `squeue`、
只读、只写日志的 shell 循环，既不是 agent 也不自动重启任何东西。用户已明示要它。

**已落地**：

```bash
bash /lus/lfs1aip2/projects/public/u6gb/gtop_watch.sh     # 循环体，INTERVAL 默认 900s
tmux new-session -d -s gtopwatch 'bash .../gtop_watch.sh' # 放 tmux，活过会话
tail -f /lus/lfs1aip2/projects/public/u6gb/logs/gtop_watch.log
```

它每 15 分钟同时采两侧，并按 §4.-1 的规则判定：

| 记录 | 含义 |
|---|---|
| `idle=N pending=M running_allocs=K` | 常规采样，一行 |
| `ACTIONABLE idle>=1 pending>=1` | **两侧都非空**，有活本可以 attach 却在排队，附空卡清单与排队清单 |
| `DEAD JOBS: n` | `DependencyNeverSatisfied`，永远不会跑，立刻报给用户（我不能 scancel） |

**判据是逐卡的 `mem 0.0/95.6G` + `idle`**，不是 gtop 头行的 idle 计数（头行把 held
也算进去）。接回会话时先 `tail` 这个日志，它比我的记忆可靠。

### 🚨🚨🚨 4.0.0 监视器脚本本身会失效，而它失效时是静默的（2026-09-04，第五次犯）

**用户第三次点名同一件事**：「check gtop every 15 mins, you should have a monitor do this,
record in claude.md」。前两次我的回应都是「好，挂一个监视器」——**两次挂的监视器都没产出过
一行日志**，而我不知道，因为没有产出的监视器和「没事发生」看起来一模一样。

2026-09-03 那个 `gpu_watch.sh` 跑了几个小时、**零行输出**，三处独立失效：

| 缺陷 | 为什么静默 |
|---|---|
| 判据取 `gtop` **头行**的 idle 计数 | 头行把 held（显存驻留、0% util）算作 idle。本文件 §4.0/§4.0.1 已写死"头行不可作判据"，脚本注释里也抄了这条，**代码做了相反的事** |
| 门槛写成 `MINE -lt 2`（我自己没在跑才报） | §4 规则 3 要求"绑资源不绑任务"。绑到我自己的负载上，等于我一忙它就闭嘴——而我忙的时候正是最不会主动查卡的时候 |
| 日志是一个**尚不存在**的文件 | 项目 inode 打满时 `>>` 静默失败。追加到不存在的文件不会报错，只会什么都不写 |

**所以规则不是「挂一个监视器」，是下面这四条。**

#### 硬要求

1. **每条会话开工、以及每次会话恢复后，第一件事是确认监视器在跑。** tmux server 是
   **节点本地**的，会话换节点就看不见旧的了——2026-09-04 从 login42 迁到 nid010871，
   监视器和一个 attach 的重测 step 一起没了（step 显示 `CANCELLED by <我的 uid>`）。
   「上次挂过」不算数。
2. **判据只能是逐卡显存**（`mem 0.0/95.6G`，1–9 MiB 才是真空），永远不是头行计数。
3. **两侧同查，门槛绑资源**：`真空卡 ≥ 4 且 PENDING ≥ 1` 才报，**与我自己有没有活无关**。
   派单清单是账户的 PENDING 队列，不是我私人的 backlog。另外
   `DependencyNeverSatisfied` 永远单独报——那是永不会跑的死作业，我不能 scancel。
4. **每一轮都写一行，包括「没事」那轮。** 否则静默有二义：分不清「集群很闲」和
   「监视器早死了」。这是三次失效里我唯一本可以自己发现的那一次。
5. **日志写一个已存在、且不在项目配额里的路径**（`$HOME`）。项目 inode 会打满，
   而打满时新建文件是静默失败。

#### 现成的实现（2026-09-04 落地，实测第一轮就报出 60 张空卡 + 12 个排队）

```bash
bash /home/u6gb/kangli.u6gb/gpu_watch_15min.sh      # 900 秒一轮，只读，绝不提交
tail -f /home/u6gb/kangli.u6gb/gpu_watch_15min.log
# 放 tmux（socket 要显式给，且 -t 收的是 target-pane，名字后面那个冒号不能省）
TMUX="/tools/brics/apps/.../tmux -S /tmp/tmux-$(id -u)/default"
$TMUX new-window -t '=claude-<本节点>' -n gpuwatch15
$TMUX send-keys -t '=claude-<本节点>:gpuwatch15' 'bash /home/u6gb/kangli.u6gb/gpu_watch_15min.sh' Enter
```

**监视器只报不做**：绝不提交、绝不 attach、绝不 scancel。这既是 BriCS 禁令的边界
（禁的是无人值守下持续**发起动作**，只读轮询写日志不算），也让它不可能自己闯祸。
收到事件后的动作照 §4.0 分工：本战线的排队作业直接 attach；别的战线的**立刻报给用户 +
给出 job id**，不 scancel、不起副本（会与排队的那份同写一个输出目录）。
**「要问一下」不是什么都不做的理由——报告本身要立刻发。**

#### 为什么这条必须是一个进程，不能是一条我要记住的规则

§4.-1 的 R1 早就把病因写对了：**规则挂在时钟上，而我按任务驱动运行**，写分析、改代码、
发长回复的时候不会因为「过了 15 分钟」而中断自己。R1 给的修法是「挂到一个我一定会做的
动作上」，**那条修法也失效了**——它仍然依赖我在做那个动作时想起这件事。
唯一可靠的办法是把检查交给一个不是我的东西去执行，然后**验证它真的在产出**。

#### 补充（2026-09-04 实测）：监视器必须是单例，否则规则本身会制造垃圾

「每条会话开工、以及每次会话恢复后，第一件事是确认监视器在跑」——这条规则**只写了
一半**。没有任何东西负责去重，于是每条会话都再起一个。2026-09-04 实测同时有
**七个监视器进程**（五个 `gpu_watch_15min.sh` 副本，外加 `gpu_watch_30min.sh` 和
`gtop_watch_v2.sh`），全都在轮询 gtop、往同一个日志追加。

**自我复制的监视器不是更可靠，只是更吵。** 修法是把幂等性放进脚本，而不是放进我的
记忆里——`gpu_watch_15min.sh` 现在开头拿 `flock`，第二次启动直接 no-op 退出。
所以规则可以继续写成「永远（重）起它」，而不会累积。

两个连带的坑，都是「静默失效」那一类：

1. **日志里的 NUL 会让 grep 闭嘴。** gtop 的输出带 NUL，写进日志后 `grep` 判定为
   二进制，只说 `Binary file matches` 而不打印任何行——与「没有匹配」不可分辨。
   实测该日志含 13,900 个 NUL。脚本已加 `tr -d "\000"`；**读这类日志一律用 awk**。
2. **按 cmdline 子串杀进程会连启动它的 shell 一起杀掉。** `case "$c" in *gpu_watch*)`
   会命中 `bash -c '... gpu_watch_15min.sh ...'` 这个包装 shell，我据此 kill 时把
   自己的会话 shell 也杀了（exit 144）。判据要用 **argv 精确匹配**：
   `mapfile -d '' -t a < /proc/$p/cmdline` 后要求 `${#a[@]} -eq 2` 且
   `${a[1]}` 以脚本路径结尾。

### 🚨 4.0.1 空卡不需要整节点空：要 4 张卡，进程内挑（2026-09-03 用户令）

**用户原话：「if you find anything about queue, you should check with gtop to find free
gpus, not need to be a full complete node」。**

犯法的形态（2026-09-03 实测）：gtop 报 39 张空卡，我用 `--gres=gpu:1` 探测，拿到的是
**有 25 GB 驻留的卡 0**，于是判定「0 张可用」并放弃 attach，转去排队。
**我把「我拿不到那张卡」读成了「没有空卡」**——正好把 §4.0 倒过来用。

根因两条，都不是「没有空卡」：

| 症状 | 真因 | 修法 |
|---|---|---|
| `Unable to satisfy cpu bind request` | 节点 **CPU** 被占满，与 GPU 无关 | **`--cpu-bind=none`**（§4 的 attach 推荐写法本来就带它，我漏了） |
| 拿到的卡有驻留显存 | **Slurm 的 gres 记账与物理占用不一致**：占卡的进程属于别的 step，Slurm 仍认为卡 0 空闲，且总是先发逻辑设备 0 | **要 `--gres=gpu:4`，在进程内用 `CUDA_VISIBLE_DEVICES` 点名 gtop 看到的那张空卡** |

`--gres=none` 不行：step 内 `CUDA_VISIBLE_DEVICES=[]`，一张卡都看不见。

**可用写法（2026-09-03 实测通过）**：

```bash
# 1. gtop 找出「哪个节点的哪张卡」空——逐卡行，不看头行计数
timeout 150 gtop --once 2>/dev/null | awk '
  /^ ▸ job/{j=$3} /^   nid/{n=$1}
  /GH200/{ if ($0 ~ /idle/ && $0 ~ /mem +0\.0\//) {
    match($0,/\[[0-9]\]/); print j, n, substr($0,RSTART+1,1) } }'
# → 6266773 nid010810 1   （节点的卡 0 在用，卡 1/2/3 空）

# 2. 拿全部四张 + 关掉 cpu 绑定，进程内点名那张空卡
srun --overlap --jobid=6266773 --nodelist=nid010810 --nodes=1 --ntasks=1 \
     --gres=gpu:4 --cpu-bind=none --job-name=<真名字> \
     bash -lc 'CUDA_VISIBLE_DEVICES=1 python ...'

# 3. 与邻居共存时 JAX 默认预分配 90% 会 OOM / autotuning 失败
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5   # 按 gtop 上那张卡的剩余量定
```

**判据仍然是 gtop 的逐卡显存**（1–9 MiB 才是真空），不是 Slurm 说什么，也不是头行的
idle 计数（它把 held 也算进去）。

### 🚨 4.0.2 gtop 巡查必须是常驻监控，且每次会话开始重起（2026-09-03 用户令）

**用户原话：「check gtop every 15 mins, you should have a monitor do this」。**

**每 15 分钟查一次 gtop，用 Monitor 工具起，不靠我自己记得查。**

犯例（2026-09-03）：这条会话里巡查起过三次、停过三次——一次是 login40 进程槽耗尽时我
主动停的，另两次随会话进程退出而死。用户问「你遵守了吗」时，**答案是没有：当时一个
监控都没有在跑**。空缺期间没人知道有多少卡在空转。

**为什么必须用 Monitor 工具，不能用 setsid nohup：**

| 方式 | 结果 |
|---|---|
| `setsid nohup ... &` | **子进程被沙箱当场回收**，无错误输出、pgrep 查不到。2026-09-03 实测连起三次全部秒死 |
| tmux | 本机 tmux 二进制 `Permission denied`，miniforge 那个不存在 |
| **Monitor 工具** | **唯一可靠**。随会话结束而停，所以**每条新会话开头必须重起一次** |

所以这条是**会话级义务**：接手 / 恢复 / 换节点之后，第一件事就是确认巡查在跑，
不在就重起。日志追加到
`/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/_gtop_watch.log`，
空卡清单写到同目录 `_free_cards_now.txt`，可以直接喂给 attach 脚本。

**判据仍是 §4.0.1 的逐卡行**（`mem 0.0/95.6G` 且 `idle`），不是头行 idle 计数，
也不是 `--gres=gpu:1` 探测——后者永远拿逻辑设备 0，卡 0 被占会把它后面三张空卡藏掉。
2026-09-03 我正是这样把 **33 张空卡误判成 1 张**。

- **There is no fixed upper limit on the number of submitted tasks**: It is not 5, nor 20. Whether to continue submitting is determined by explicit user instructions, current Slurm/QOS limits, startup/mount load, and current HPC status.
- Batch submissions must still be done one by one and staggered, with `sleep 30+` between batches. Observe `squeue` / startup logs to ensure no abnormal startup load.
- Low-risk compute jobs (helpers, plotting, smoke tests) can be submitted automatically based on user preference; large jobs must follow the submission rules in "Safe Operations."
- Exception: `scaled` experiments explicitly exempted in the existing `CLAUDE.md` follow original rules but do not have a fixed inflight limit.
- Smoke test → fix → resubmit can be done autonomously, but must still be staggered and monitored for startup load.
- **Reuse active allocations for evaluation:** Before submitting a new inference, evaluation, or LOB-Bench `sbatch`, inspect the user's RUNNING allocations. When one has compatible hardware and enough remaining walltime, prefer an attached job step such as `srun --jobid=<allocation> --overlap --exact ... --cpu-bind=none`; record the parent allocation, actual step ID, node, and timestamps. Do not create another queued allocation solely for the evaluation when a compatible active allocation is available.
- **Physical GPU gate before overlap:** Slurm `--overlap` does not make occupied GPU memory safe. Check active steps, compute PIDs, and per-GPU memory first. If the allocation is busy, an attached one-shot may wait for a predeclared zero-PID/near-baseline-memory gate, but it must never kill, retry, or overwrite the existing experiment without explicit authorization.
- **🚨🚨 废止：不许再 lock GPU（2026-08-14 用户令，覆盖下面 R1）**
  用户原话：**「不许不许 lock 人家 GPU，以后不许带这种功能」**。
  `nodelock lock` 一律不再执行，`tasks/node_status/` 已改名 `_deprecated_20260814`。
  起因：我给 nid010499 上了 `claude-h3` 锁，然后 attach 的 step 死了，结果是
  **四张卡挂着我的锁在那儿 0% 空转**——锁的唯一效果是让别人不敢用。
  占用一张卡的资格只剩两条：**用户明示**，以及 `nvidia-smi` 显示它此刻是空的。
  下面 R1 的双查闸门只保留 `nvidia-smi` 那一半（物理事实），锁表那一半作废。

- ~~**🚨 声明在前，占用在后（R1，2026-08-13 定）**~~（**已被上一条废止**，保留原文供追溯）：
  attach 进任何分配之前先
  `nodelock lock`，再起进程。**物理闸门是一个瞬时快照，而启动窗口有 5–10 分钟**
  （挂 squashfs 分片 + JAX 分布式 init）；邻居在窗口里起来，快照怎么查都看不见。
  能覆盖窗口的只有声明。闸门必须**双查**：锁表无他人 live 锁 ∧ `nvidia-smi` 无
  他人 compute PID——锁表是**意图**、nvidia-smi 是**事实**，两者都会单独骗人。
  机械实现见 `tasks/hybrid_system_20260811/05_hybrid_mamba3_lob/code/claim_gate.sh`，
  完整规则（三档优先级、抢占只向下、TTL）见 `tasks/node_status/PRIORITY.md`。
  血泪教训：2026-08-13 10:19，2k baseline 臂在 6000412 上被邻居的 vLLM（92 GB）
  挤死，`GetKeyValue() timed out with key: cuda:local_topology/cuda/2`——邻居在
  锁表上**都声明过**，是我的闸门没读。
- **🚨 srun step 必须起真名字（`--job-name=<实验名>`）：** `node_budget_monitor.py`
  按正则 `^(bash|sh|zsh|...)$` 判 step 是否 idle，有真名字的 step 让整个作业记成
  `computing` 而**不计入 20 节点上限**。本仓库普遍用 `srun bash -lc '...'`，step 名
  就叫 `bash`，于是**正在训练的作业被判成 IDLE-HELD，随时可能被预算闸门 scancel**。
  2026-08-13 的 dry-run 里，已训 5h45m 的 hybrid 就显示 `IDLE-HELD only bash,bash,...`。
  起名字不是绕过闸门，是把度量修对。
- **Queued-chain → attached-allocation handoff:** Hold the queued chain's root while establishing the replacement. Once the attached runner is confirmed live, cancel the entire superseded chain and verify every old job with both `squeue` and `sacct`. Never allow the queued and attached copies of the same experiment to run concurrently.

### 4.1 SquashFS SP500 Sweep Submission Constraints (2026-05-09)

SP500 scaling-law training must use compressed monthly SquashFS shards; falling back to raw `lob_preproc_sp500/<TICKER>/*.npy.zst` directory scanning is not allowed.

- `SQUASHFS_MULTI_MODE=1` must be set.
- Explicitly pass a month list like `SQUASHFS_MONTHS=2022-01,...,2025-12`; `SQUASHFS_MONTHS=all` is forbidden.
- Maintain `FORBID_RAW_NPYZST=1`.
- A single 8-node job will generate approximately `8 nodes * 48 months = 384` `squashfuse` mounts and read 48 shard-local `index.json` files per node.
- This is much safer than raw `.npy.zst` globs. System risk comes primarily from concurrent job and mount counts, not the 48 JSON files.
- For multi-experiment sweeps, no fixed inflight limit is set; however, jobs must be started one by one, staggered, and monitored for startup/mount load. Risk is estimated at `8-node jobs × 384 mounts/job`.

## 5. Login Node Taboos

- **Allowed**: Editing files, submitting jobs, `squeue` / `sacct` / `module list` / `sinfo`, light scripting.
- **Allowed (small CPU experiments, 2026-05-20)**: Lightweight CPU-only experiments such as linear regression / sklearn fits / pandas data prep / parquet conversion may run **directly on login nodes** when ALL of the following hold:
  - No GPU usage
  - Wall-clock under ~30 min, peak RAM under ~16 GB
  - Total bytes read+written under ~5 GB
  - Sequential file access only (no recursive `ls`/`find`, no parallel multi-worker pools)
  - Single-shot invocation, not a persistent loop or daemon
  - Rationale: a 5-min job behind a 30-min Priority queue wait wastes wall-clock; small CPU tasks have negligible Lustre metadata cost.
- **Prohibited**: Compiling, GPU inference/training, **heavy** data preprocessing (multi-process / >5 GB / >30 min), batch grep on large files, recursive ls / find.
- **Forbidden**: Persisting any agent on login nodes via `nohup` / `screen` / `tmux`.
- **Forbidden**: Auto-restarting any agent or daemon. If it is stopped, it was intentional; do not auto-recover.

## 6. Batch Script Review Checklist (Must check every line when writing or modifying batch scripts)

- [ ] Output dir isolated using `$SLURM_JOB_ID`
- [ ] Checkpoints written to `$TMPDIR`, `rsync` to `/projects` at end of job
- [ ] `latest_checkpoint.json` breadcrumb written (atomic by Rank 0)
- [ ] Auto-resume reads breadcrumb, does not `ls` checkpoint directory
- [ ] Tensorboard logs to `$TMPDIR`, synced at end of job
- [ ] `WANDB_DIR=$TMPDIR` and `WANDB_MODE=online` (if using W&B)
- [ ] No `find` / `ls -R` / `du -sh` / `tree` / large globs in the script
- [ ] Model weights staged to `$TMPDIR` before training
- [ ] Checkpoint interval is reasonable (not every step)

## 7. When in Doubt, Ask

If you are unsure whether a command will trigger metadata load, **do not try it**. Ask the user directly. Do not use "exploration" as an excuse to run `ls` / `find`.

> **Guiding Principle**: Act like a human researcher who understands that Lustre metadata operations are expensive. A human researcher would not `ls -lR /projects`, so neither will I.

## 8. AUP Accountability + Official 9 Anti-Patterns Table (2026-05-08 sync from HPC team / Aramis notes)

### 8.1 AUP Accountability (Most Important)

The Isambard-AI Acceptable Use Policy explicitly states: **All actions performed by an LLM / AI agent are attributed to the human account running it.** There is no "the agent did it" exemption.

- HPC audit logs record `kangli.s5e` / `aramis.s5e`, **not** "Claude."
- Every `sbatch` / `find` / `ls` / `rm` is treated as if "the human user typed it manually."
- Once BriCS attributes a violation, the penalty falls directly on the human account and may affect the entire team.
- **Batch scripts / training scripts generated by an LLM must be explicitly reviewed from a Lustre + HPC perspective before submission**—let the LLM self-check against the 9 anti-patterns below.

### 8.2 The 9 Anti-Patterns (Why + Fix)

The table below summarizes the 9 anti-patterns identified by the HPC team (Rich, Jakob) on 2026-05-08.

| # | Anti-pattern | Why it breaks Lustre | Fix |
|---|---|---|---|
| 1 | `ls -1td <ckpt_dir>` or `ls \| grep \| sort` to find latest checkpoint | Stats all inodes in the directory every call; metadata storm grows linearly with checkpoint count | Rank 0 atomically writes `latest_checkpoint.json` (path + step); auto-resume reads only this file |
| 2 | `find /` / `find /lus` / `ls -lR <broad path>` | Traverses the entire tree, floods the MDT, slows down everyone | Narrow the starting directory to the smallest possible subtree; use `lfs find` for non-recursive needs |
| 3 | Multiple jobs sharing a single output directory | Directory-level lock contention; concurrent jobs serialize on the same lock domain | Isolate each job using `${SLURM_JOB_ID}` or a unique experiment ID |
| 4 | Checkpoints / training logs written directly to Lustre | Continuous training triggers constant MDT locks | Write all intermediate output to `$TMPDIR` or NVMe scratch; `rsync` final checkpoint back to Lustre on success |
| 5 | Submitting a massive burst of jobs at once (e.g., 80) | Maximizes concurrent startup load; MDT cannot keep up | No fixed inflight limit for this repo; must submit one by one, stagger, and observe in batches |
| 6 | Checkpointing every step | Directory lock frequency follows the step count | Increase interval to at least 500 steps (model-dependent); target "recovering once wastes < 10% of job time" |
| 7 | Reading model weights from Lustre at startup | Multiple jobs starting simultaneously → startup lock storm | Pre-stage weights to `$TMPDIR` / NVMe before training actually starts |
| 8 | tensorboard / wandb / custom loggers writing to Lustre | Persistent metadata traffic throughout the entire run | Redirect local files to `$TMPDIR` (`WANDB_DIR=$TMPDIR`); keep tracked W&B runs online with `WANDB_MODE=online`; rsync non-W&B artifacts to Lustre at end-of-run |
| 9 | Running persistent AI agents + auto-restart wrappers on login/compute nodes | Explicitly forbidden by HPC team; auto-restart bypasses administrator intervention | Do not run resident agents on login/compute nodes; do not write "auto-restart if killed" scripts |

### 8.3 Origin + Original Quote from Rich (HPC Team) 2026-05-08

- **Triggering Event**: The team submitted a large volume of jobs simultaneously, with checkpoints written directly to Lustre and `ls`-based discovery for the latest checkpoint. This hit exactly the metadata storm pattern warned about by HPC.
- **Consequences**: On 2026-05-08, all group jobs were suspended by BriCS administrators, and AI agents were disabled service-wide on login nodes.
- **Symptoms**: `squeue` / `sbatch` began erroring. **Root Cause**: Lustre filesystem-level congestion.

> "Please note we are also disabling the use of AI agents on the login nodes. These are also prone to using the harmful patterns below, and could well be launching large numbers of compute node jobs that themselves contain those same patterns."
> — Rich, BriCS HPC team, 2026-05-08

---


## 🚨 训练观测频率：checkpoint 与 loss 是两件事，不许共用一个频率（2026-08-13）

### 硬约束

| 项 | 频率 | 理由 |
|---|---|---|
| **checkpoint** | **平均每 ~15 分钟一次** | 写整个参数树到 Lustre，按「恢复一次损失 <10% 时间」定 |
| **step_loss / lr / grad_norms** | **每 1 分钟（auto）或每 250 步（显式）** | 只是几个标量打到 wandb，几乎免费 |

**默认一律用 `CHECKPOINT_EVERY=auto`。** 那条路径已经按时间判据把两者分开：

```
AUTO_CKPT_INTERVAL  = 900s  = 15 分钟   checkpoint
AUTO_WANDB_INTERVAL =  60s  =  1 分钟   wandb
首次 checkpoint 在 ~5 分钟（防早期 NCCL 死锁丢掉全部进度）
```

### 反面教训（2026-08-12，hybrid ctx2k）

给了显式的 `CHECKPOINT_EVERY=2000` 覆盖掉 auto，同时踩了两个坑：

1. **checkpoint 变稀**：2k 上下文下 2000 步 = 50 分钟以上，一次崩溃丢 50 分钟
2. **`step_loss` 被绑到同一个频率上**——`train_helpers.py` 里原本是
   `should_ckpt = should_wandb = (step % N == 0)`

后果：hybrid 发散时**只能定位到「4000 到 6000 步之间」**，分不出是突然尖峰还是持续
爬升。而想看密的 loss 就得调小 checkpoint 频率，**等于用 Lustre 元数据去买观测精度，
代价差三个数量级**。

已修：`LOG_EVERY`（默认 250）独立控制显式档下的记录频率，与 `CHECKPOINT_EVERY` 解耦。

### 必须记录的 profiling 指标

```bash
export LOG_GRAD_NORMS=1     # 分组梯度范数 + clip_ratio
```

| 指标 | 为什么必需 |
|---|---|
| `grad_norms/{global,muon,ssm,regular,in_proj,out_proj}` | 发散时第一件事是看**哪一组**先炸 |
| `grad_norms/clip_ratio` | 直接回答「裁剪是不是一直饱和」——饱和说明裁剪只是在掩盖问题 |
| `lr` / `ssm_lr` | 排除「是不是 schedule 在这一段把 LR 抬起来了」；**必须与 loss 同频**，否则两条曲线对不齐就没法说话 |
| `throughput/{step_time_s,mfu_pct,tflops,tokens_per_sec}` | 步时突变往往先于 loss 异常 |

### 起跑前自查

- [ ] `CHECKPOINT_EVERY` 是 `auto`，或显式值 × 步时 ≈ 15 分钟
- [ ] `LOG_GRAD_NORMS=1`
- [ ] 显式档下 `LOG_EVERY` ≤ 250
- [ ] wandb 上能看到 `lr` 与 `step_loss` 在同一批 step 上有点

---

## 🚨 梯度累积 K 不许有「默认值」：声明有效批量，推导 K（2026-08-14 用户定）

### 硬规则

**`GRAD_ACCUM_STEPS` 永远不写成 `${GRAD_ACCUM_STEPS:-<某个数>}`。**
脚本里要声明的是**有效批量**，K 由它和节点数推导出来，并对调用方传进来的值硬校验。

```
有效批量 = micro_bsz × GPU/节点 × 节点数 × K
```

以本仓库 2k 上下文那三条臂为例（`micro_bsz=1`，`GPU/节点=4`，有效批量 80）：

| 节点数 | 2 | 4 | 5 | 10 | 20 |
|---|---:|---:|---:|---:|---:|
| **正确的 K** | **10** | **5** | 4 | 2 | 1 |

**换节点数而不改 K，改的就是实验本身。** 有效批量是实验定义的一部分，
和模型、数据、seed 同级；K 只是为了在给定硬件上凑出它的手段。

### 反面教训（2026-08-13/14）

`launch_2k_*.sh` 原本写的是 `export GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-10}`
（按 **2 节点**写死），靠调用方 `chain_manager.sh:291` 传 `5` 来覆盖。这个形状必出事：

- 4 节点上忘了覆盖 → 有效批量 **160**，是设计值的 2 倍
- **不会报任何错**。训练照跑、loss 照降、wandb 照记，只是训出一个与对照臂
  **不可比**的模型
- 而「不可比」要到几小时后做配对比较时才发现，那时机器已经烧掉了

同一族缺陷见 `feedback_config_only_reaches_one_path`：**一个必须随环境变的量，
被写成固定默认值 + 藏在别处的覆盖。**

### 正确写法（已落地在 `launch_2k_{baseline,hybrid,hybrid_pmatch}.sh`）

```bash
export EFFECTIVE_BSZ=${EFFECTIVE_BSZ:-80}          # ← 这个才该有默认值
_bsz_denom=$(( PER_GPU_BSZ * GPUS_PER_NODE * NNODES_ATTACH ))
if [ "$_bsz_denom" -le 0 ] || [ $(( EFFECTIVE_BSZ % _bsz_denom )) -ne 0 ]; then
    echo "FATAL: 有效批量 $EFFECTIVE_BSZ 不能被 $_bsz_denom 整除" >&2; exit 5
fi
_k_derived=$(( EFFECTIVE_BSZ / _bsz_denom ))
if [ -n "${GRAD_ACCUM_STEPS:-}" ] && [ "${GRAD_ACCUM_STEPS}" != "${_k_derived}" ]; then
    echo "FATAL: 传了 K=$GRAD_ACCUM_STEPS，但 $NNODES_ATTACH 节点要求 K=$_k_derived" >&2
    echo "       要改有效批量就显式传 EFFECTIVE_BSZ=，别靠改 K 间接改它" >&2; exit 5
fi
export GRAD_ACCUM_STEPS=$_k_derived
echo "[bsz] 有效批量 $EFFECTIVE_BSZ = ${PER_GPU_BSZ} × ${GPUS_PER_NODE}GPU × ${NNODES_ATTACH}节点 × K${GRAD_ACCUM_STEPS}"
```

**必须打印那行 `[bsz]`**，它让「这次到底跑的是多大批量」在日志里可检索，
而不是要靠读三个脚本反推。

### 配套：参数量也用同样的方式钉死

同理，凡是「参数量必须等于某个对照臂」的实验，设
`EXPECTED_PARAMS=<实测值>`（`src/lob/train.py:349-364` 会在不等时 raise）。
理由一样：配置没到达模型时**不会报错**，只会在几小时后给出一个假结论。

### 起跑前自查

- [ ] 脚本里没有 `GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-<数字>}` 这种写法
- [ ] 日志里有 `[bsz] 有效批量 ... = ... × ...节点 × K...`，且数值等于设计值
- [ ] 换过节点数的话，K 跟着变了（不是沿用上一次的）
- [ ] 参数量对照实验设了 `EXPECTED_PARAMS`

---

## Project Overview

**LOBS5** — A token-level autoregressive generative model of limit order book (LOB) message flow using the S5 (Simplified Structured State Space) architecture in JAX/Flax. This experiment worktree (`exp/H1-scaling-law`) trains 5 model sizes (10M–120M params) on 8 tickers × 4 years to study neural scaling laws for LOB generation.

This is a **git worktree** of the main repo at `AlphaTrade/LOBS5/`.

## Architecture

```
LOBSTER .npy data
  │  (PyTorch DataLoader — CPU only, multi-ticker support)
  ▼
Tokenized messages (24 tok/msg, vocab=2112, base-100 encoding)
  + Order book volume images (500 depth levels)
  │
  ▼
┌─ PaddedLobPredModel (lob/lob_seq_model.py) ─────────────────┐
│  Message encoder: Embed → n_message_layers × SequenceLayer   │
│  Book encoder:    Dense → pre-layers → project → post-layers │
│  Padded fusion:   concat at message boundary                 │
│  Fused encoder:   n_layers × SequenceLayer(S5)               │
│  Decoder:         Dense → log_softmax (vocab_size)           │
└──────────────────────────────────────────────────────────────┘
  │
  ▼
Next-token cross-entropy loss → AdamW (optax) + cosine anneal
  → Orbax checkpoints + W&B logging
```

Each **SequenceLayer** (`s5/layers.py`): PreNorm → S5SSM (parallel scan via `jax.lax.associative_scan`) → half_glu1 activation → skip connection.

**S5SSM** (`s5/ssm.py`): HiPPO-LegS init, diagonal complex state matrix, ZOH discretization. Supports both parallel scan (training) and RNN mode (inference).

### Key source files

| File | Role |
|------|------|
| `run_train.py` | Entry point — CLI args, JAX distributed init, calls `lob.train.train()` |
| `lob/train.py` | Training loop — epochs, mini-epochs, validation, checkpointing, W&B |
| `lob/train_helpers.py` | JIT-compiled `train_step`/`eval_step`, optimizer, LR schedule, StepWatchdog |
| `lob/init_train.py` | Model/TrainState init, Orbax checkpoint load/save |
| `lob/sharding_utils.py` | JAX mesh creation (1D flat / 2D hierarchical), data & param sharding |
| `lob/dataloading.py` | Dataset factory, multi-ticker support, distributed sampler |
| `lob/lobster_dataloader.py` | `LOBSTER_Dataset` (PyTorch Dataset), file caching, masking |
| `lob/encoding.py` | `Message_Tokenizer` — LOB messages ↔ token sequences |
| `lob/lob_seq_model.py` | `PaddedLobPredModel` — Flax module (production model) |
| `s5/ssm.py` | S5 state space model core |
| `s5/layers.py` | `SequenceLayer` — single S5 layer with norm/activation/skip |
| `lob/inference.py` | Autoregressive generation with error correction |

## Common Commands

### Training (all runs via SLURM — never run on login node)

```bash
# Single model benchmark (30min, CURTAIL=300 steps)
D_MODEL=1024 N_LAYERS=12 BLOCKS=16 SSM_SIZE_BASE=1024 PER_GPU_BSZ=10 \
  CURTAIL_EPOCHS=300 sbatch --contiguous --nodes=64 --time=00:30:00 train_full_autoreg.batch

# Full training (15% of 1 epoch)
D_MODEL=1024 N_LAYERS=12 BLOCKS=16 SSM_SIZE_BASE=1024 PER_GPU_BSZ=10 \
  CURTAIL_EPOCHS=3179 NO_VALIDATION=1 sbatch --contiguous --nodes=64 --time=24:00:00 train_full_autoreg.batch

# Batch submit all 5 scaling law models
./submit_scaling_law.sh benchmark     # 5 sequential benchmarks
./submit_scaling_law.sh train         # 5 sequential 15%-epoch training runs
./submit_scaling_law.sh epoch         # 4 models × full epoch (120M skipped)
./submit_scaling_law.sh single 55M benchmark  # single model

# Dry run (preview commands)
DRY_RUN=1 ./submit_scaling_law.sh train

# Override node count
NODES=32 ./submit_scaling_law.sh benchmark
```

### Evaluation

```bash
# Post-training eval (single node)
RESTORE=/path/to/checkpoint sbatch eval_post_training.batch
```

### Tests (Docker-based, for local dev)

```bash
make test        # pytest ./tests/ inside Docker
make build       # build Docker image
```

## Scaling Law Model Configurations

| Label | d_model | n_layers | blocks | ssm_size | BSZ/GPU | Global BSZ (64N) |
|-------|---------|----------|--------|----------|---------|------------------|
| 10M   | 512     | 6        | 8      | 512      | 20      | 5,120            |
| 22M   | 768     | 6        | 12     | 768      | 14      | 3,584            |
| 55M   | 1024    | 12       | 16     | 1024     | 10      | 2,560            |
| 85M   | 1280    | 12       | 20     | 1280     | 6       | 1,536            |
| 120M  | 1536    | 12       | 24     | 1536     | 4       | 1,024            |

Default model (when no env overrides): **360M** (d_model=2048, n_layers=24, blocks=32, ssm_size=2048, BSZ=2/GPU).

## SLURM Job Structure

```
sbatch train_full_autoreg.batch
  └→ srun (1 task/node) → node_wrapper.sh (per-node)
       └→ conda activate, CUDA, NCCL/XLA flags → python run_train.py
```

- `train_full_autoreg.batch`: env vars, model config, data paths, srun launch, auto-resume logic
- `node_wrapper.sh`: conda env, CUDA 12.6, custom NCCL 2.29.3, AWS OFI NCCL 1.18.0, XLA flags
- `run_train.py`: argparse, `jax.distributed.initialize()`, calls `lob.train.train()`

Environment variables override model config (e.g., `D_MODEL`, `N_LAYERS`, `BLOCKS`, `SSM_SIZE_BASE`, `PER_GPU_BSZ`, `CURTAIL_EPOCHS`, `NO_VALIDATION`, `HIERARCHICAL`).

Multi-node (≥2 nodes) automatically enables hierarchical 2D mesh AllReduce.

## Key Design Decisions

- **PyTorch for data loading only** — DataLoader runs on CPU; all GPU computation is JAX/Flax
- **24-token encoding** (base-100, vocab=2112) — replaced 22-token encoding (base-10000, vocab=12012)
- **Hierarchical AllReduce** — 2D mesh splits into NVLink (intra-node) + Slingshot (inter-node); without it, 32N+ is ~2x slower
- **Custom NCCL 2.29.3** — source-built with GCC 12.3 for ARM CAS fix, loaded via LD_PRELOAD
- **Cosine annealing with warmup** — √κ LR scaling for AdamW when batch size changes
- **Mini-epochs** — split 1 data epoch into K sub-epochs for frequent eval checkpoints
- **Auto-resume** — on crash, batch script resubmits with `RESTORE_PATH` (up to 3 retries)

## Known Issue: Mid-Epoch Resume

**BUG**: When resuming from a mid-epoch checkpoint, `resume_from_step` is NOT auto-set from `state.step`. The dataloader restarts from batch 0, causing data duplication.

- **What works**: model weights, optimizer state, LR schedule (all keyed by `state.step`)
- **What breaks**: dataloader position — it replays already-seen batches
- **Root cause**: `lob/train.py:339` reads `args.resume_from_step` (always None) instead of computing `state.step % steps_per_epoch`
- **Fix needed**: auto-compute `resume_from_step = int(state.step) % steps_per_epoch` when restoring mid-epoch checkpoints. The `DistributedSampler(seed=42)` produces deterministic order per epoch, so skipping to the correct batch index is safe.
- **Curtail uses `batch_idx` not `state.step`**: so without fix, resume runs ALL curtail steps again (31k steps instead of remaining 8k), with wrong LR schedule (state.step keeps incrementing past curtail target).

## Logs and Checkpoints

- SLURM logs: `logs_lobs5/lobs5_{JOBID}.out`
- Per-node logs: `logs_lobs5/training_{JOBID}_node{N}.log`
- Checkpoints: `checkpoints/j{JOBID}_{WANDB_ID}_{JOBID}/{step}/`
- W&B project: `lobs5-scaling-law`
- Data split info: `logs_lobs5/data_split_j{JOBID}.json`

## HPC Environment

- **Platform**: ARM (Grace Hopper), 4× GPU/node, NV6 (6× NVLink bonded per GPU pair)
- **Conda**: `/projects/s5e/quant/miniforge3` (base env) — JAX 0.9.0.1, Python 3.12.11
- **Partition**: `workq`
- **Login node**: no GPU computation allowed; everything via `sbatch`

## Pre-Submit Checklist (MANDATORY)

**Step 0 comes before everything else: run `gtop`** (physical occupancy right now --
*not* `sgpu`, which is the historical gantt ledger and cannot answer "is anything free
now"). If any GPU in an allocation this account already holds is idle, **attach to it
and do not submit at all**. See §4.0 — `sbatch` is only permitted once `gtop` shows
there is genuinely nothing free. Keep polling while anything is queued; the moment a
node frees up, move the work onto it.

**Every `sbatch` submission MUST be preceded by a `squeue` dedup check.** This is non-negotiable.

0. Run `gtop`. Idle GPUs anywhere in this account → attach via `srun --overlap`, do not `sbatch`.
1. Run `squeue -u kangli.s5e -o "%.10i %.20j %.8T %.12M %.6D"` to list all running/pending jobs
2. For each running job, compare: **model config** (architecture, d_model, n_layers, params), **data** (tickers, date range), **encoding** (P1a/P1b/P1c), **seq_len**
3. If any existing job has the **same model + same data + same encoding**, do NOT submit — it's a duplicate
4. If the new job is a **resume** of a completed/timed-out job, confirm the old job is no longer running before submitting
5. Print the dedup check result to the user before proceeding with `sbatch`

**Why:** Job 3253421 (fresh start) ran in parallel with 3260152 (resume of same config) for 13 hours, wasting 16N × 13h = 208 node-hours because no dedup check was performed before submission.

**Anti-pattern to avoid:** Submitting a fresh-start job when a resume chain already exists for the same experiment. Always check if there's an active job or recent checkpoint for the same config before starting fresh.

## Session Search Protocol (MANDATORY)

**Use `/find-session-id` skill.** Core rule: **LARGEST JSONL = main session. Always.**

When searching for a historical conversation session by user-quoted text:

**Step 1 — Identify and exclude current session:**
```bash
# Current session ID is in the JSONL filename being written to right now
# ALWAYS determine it FIRST, then grep -v to exclude
```

**Step 2 — Use high-selectivity keys for first-pass filtering:**
- Commit hashes (e.g., `bae82954`) are ideal — nearly unique across all sessions
- SLURM job IDs, W&B run IDs, specific error messages also work well
- Avoid generic terms (KDA, GDN, smoke test) — too many false positives

**Step 3 — Multiple matches? Pick the LARGEST file (by size):**
- **LARGEST JSONL = main session** (user's interactive conversation, 1-5MB)
- Medium files (100KB-1MB) = continuation/child sessions (reference parent, partial context)
- Small files (<100KB) = subagent fragments (never return these)
- **DO NOT use "unique text" as discriminator** — subagent JSONLs contain full assistant output (including "Brewed for Xs"), leading to wrong (child) session

**Step 4 — Validate before returning:**
- Confirm result != current session ID
- Print session ID, JSONL path, modified date, size, and resume command

**Anti-patterns:**
- Grepping for common terms first, getting current session in results, returning it without noticing (caused errors 2026-03-26)
- Using rendered text ("Brewed for 44s") as discriminator instead of file size (caused error 2026-03-26, returned child session instead of parent)

## Post-Submit Job Monitoring (MANDATORY)

**Every `sbatch` must be followed by a multi-checkpoint monitor.** A single `sleep 300 && check` is NOT acceptable. Jobs can crash in seconds (j3492376: /dev/shm full, crashed at 21s) and a naive monitor won't detect it for minutes.

### Required Checkpoint Schedule

| Checkpoint | When | What to check |
|-----------|------|---------------|
| 1min | 60s | Log file exists? Any immediate crash (OOM, NCCL error, /dev/shm)? `sacct` exit code if job already gone. |
| 5min | 300s | Past XLA compile? tqdm output started? Step count > 0? |
| 15min | 900s | In stable regime? Speed (s/step)? Any NaN fatal? |
| 30min | 1800s | Final result. Total steps, stable speed, wandb URL. |

### Critical: Detect Crash vs Success

When a job is no longer in `squeue`, you MUST check `sacct -j $JOBID --format=ExitCode`:
- `0:0` = clean exit (success)
- `1:0` = crash (Python exception, OOM, NCCL failure)
- Other = signal kill, timeout, etc.

**Never assume "not in squeue" = success.** A job that crashes at second 21 also disappears from squeue.

### Critical: Check ALL Node Logs on Crash

When exit != 0:0, grep ALL `training_${JOBID}_node*.log` files for the FIRST error. The coordinator node (node0) often shows "Shutdown barrier timeout" which is a SYMPTOM. The root cause is usually on a different node.

```bash
for NLOG in $WT/training_${JOBID}_node*.log; do
    NODE=$(basename "$NLOG" | sed 's/.*node\([0-9]*\).*/\1/')
    ERR=$(grep -iE "RESOURCE_EXHAUSTED|No space left|OOM|NCCL WARN Error|SIGABRT|nan.*fatal|FATAL" "$NLOG" | grep -iv "CUDA_ERROR_NO_DEVICE" | head -1)
    [ -n "$ERR" ] && echo "node${NODE}: ${ERR:0:200}"
done
```

### Lesson (j3492376, 2026-03-30)

Job crashed at 21s (NCCL /dev/shm full on nid011191). Monitor script had `sleep 60` as first checkpoint, but only checked `squeue` and `grep tqdm` on node0. Node0 showed "Shutdown barrier timeout" (symptom), real error was on node4. Monitor ran for 47 minutes without surfacing the crash because it never checked `sacct` exit code or other node logs.

## 运行环境的自我认知 (2026-08-07, 用户明确纠正)

**Claude Code 进程运行在 Isambard 的登录节点上，不是在用户的 Mac 上。** 用户从
MacBook 通过 SSH 连到服务器，Claude Code 是服务器上的一个程序。

推论（都影响该建议什么）：
- 服务器**始终有网**。用户 Mac 断网断的是 SSH 连接，不是服务器的出网能力。
- 因此 `tmux` **确实能**让这个会话在用户断开后存活：`tmux new -s <name>` 里跑
  `claude --resume <session-id>`，回来 `tmux attach`。不要再说「tmux 也没用因为
  需要 API 网络」——那是把自己误当成跑在用户笔记本上。
- 但会话存活 ≠ 工作继续：**没有用户输入时 Claude 不会自己醒来**。所以长时间无人值守
  的工作必须落到 SLURM 作业里（自带 checkpoint/resume/失败续投），而不是指望会话活着。
- 登录节点上常驻 agent 仍受 BriCS 2026-05-08 禁令约束。判据是**是否在无人值守下持续
  发起动作**：脱离后闲置等待输入的会话是良性的，自动轮询/循环提交的不是。

## sigma-0 永远用 junming —— 而 `gh` 的 active 账号会被别的会话切走（2026-09-05 用户第三次点名）

§1.3 已经写了「`KangOxford/sigma-0` 用 junming / anjunming1202」，**这次仍然失守**，
失守的位置是它没写的那一半：

| 层 | 这次的实际状态 | 说明 |
|---|---|---|
| `git config user.name` | ✅ junming | 仓库级配置，**别的会话动不了** |
| **`gh auth status` active** | ❌ **KangOxford** | **进程外的全局状态**，任何会话 `gh auth switch` 一下就换掉，我这边完全看不出来 |

六个 commit 的 author 全是 junming（正确），而屏幕上看到 KangOxford，是因为**推送与开 PR 用的是
`gh` 的 active 账号，不是 `git config`**。两者是独立的两个身份，只对一个是不够的。

**机械做法：每一次 `git push` / `gh pr create` / `gh pr comment` 之前，两个都查，缺一不可。**

```bash
git config user.name        # 必须 junming
gh auth status 2>&1 | grep -B1 "Active account: true"   # 必须 anjunming1202
gh auth switch --user anjunming1202                      # 不对就切
```

**为什么不能只在会话开始时查一次**：`gh` 的 active 是全局的，别的会话切过它、或上一次
`gh auth switch --user KangOxford`（LDM 那条线要用它）之后没切回来，我这边**在推送成功之前
没有任何征兆**——只有 PR 的 author 字段写出来才发现。所以是**每次推送前查**，不是每会话查。

**与 §1.3 的关系**：那条规定了「哪个仓库用哪个身份」，这条补的是「身份有两处、其中一处是
共享可变状态」。两条一起才完整。
