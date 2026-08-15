## 一句话总结

OpenPhil Coscientist 是一个 "AI co-scientist"(自动科研助手)多 agent 系统,它最特别的模块 `evolution` 让系统改写自己的代码。你的 heuristic learning(`meta-learning-evolution` skill,加上 EvoEnv / EvoCurr / RAGEN 那条自进化 RL 研究线)做的是用 GA 加 LLM 去进化"解任务用的损失函数 / 奖励 / 启发式"。两者其实是同一个 "LLM 提议 → 评估 → 择优" 外循环的两个端点,差别集中在 "用什么评估" 和 "进化的对象是谁",而且这两点正好互补。

---

## 1. OpenPhil Coscientist 在做什么

整体上它是搭在 Claude Agent SDK 上的多 agent 科研系统:FastAPI 跑在 8765,Streamlit UI 跑在 8501。人提交一个科研任务,系统返回叙事报告加 markdown 交付物,事件流通过 SSE 实时回传。它有两个 runtime。

第一个是 **research runtime**:一个 supervisor agent(opus)负责跟人对话、规划、用内置 `Task` 工具把活派给 subagent(目前只有三种角色:supervisor、generalist_researcher、data_analyst),再汇总。过程里有 HITL(human-in-the-loop)checkpoint:每隔若干次工具调用会暂停让人审。它还有一个 memory 层(三层 JSONL,agent / project / global,用 BM25 召回),用来跨 session 复用 lesson,避免重复劳动。

第二个是 **evolution runtime**,也是它最特别的地方:系统改写自己的代码、prompt、role 定义和工具。LLM 在一个 git worktree 沙箱里 `edit` 文件,跑 `run_tests`,然后调用 `propose_merge`;敏感路径(`scaffold/`、`evolution/`、`pyproject.toml`、`Dockerfile`、`docker-compose.yml`)的改动会先过一道 `pytest` smoke 二值门,再交给人审,人 approve 之后才 `git merge --ff-only` 落到主干。因为源码是 bind-mount 的,合并后的改动下一次调用就生效。

---

## 2. 反直觉的关键事实:这里的 "evolution" 不是进化计算

对整个 repo 做关键词 grep(`fitness` / `score` / `reward` / `loss` / `benchmark` / `population` / `GA`)几乎全空。也就是说,这里的 "evolution" 不是遗传算法那套,而是 **LLM 驱动、人审 gate 的自我代码修改**,属于 AlphaEvolve "直接编辑代码" 这一脉,但刻意去掉了适应度函数、种群、交叉变异。唯一的自动评估是敏感路径上的 `pytest` smoke 测试,只有通过 / 失败两态;真正的录用标准是人的 approve 加一次可回滚的 `git merge`(系统还会存 `revert.patch`)。

这条事实是理解它和你工作关系的关键:它有完整的 "自我修改机器"(worktree、edit、sandbox、merge、回滚),但**没有自动的数值适应度环**。

---

## 3. 你的 Heuristic Learning 在做什么(对照)

你的 `meta-learning-evolution` skill 进化的是任务侧的损失函数、奖励函数、config、启发式。它的分工很清楚:GA 负责 "已知组件的权重和组合"(结构化数值搜索,没有 API 成本),LLM 负责 "发明新原语"(需要领域知识和创造力)。最佳形态是两级循环:GA 内层每一代都跑,LLM 外层只在停滞(连续 N 代无提升)时介入,发明 1 到 2 个新项注入 GA 的词表。评估是**自动的数值 fitness**(Sharpe、IC、winrate_margin),有种群、有世代、对结构化 genome 做 crossover / mutation,还有 diversity pressure 防止 LLM 探索坍缩(否则它会锚定第一个成功解,产出 20 个近似变体)。

你最近在 Notion 上的 EvoEnv、EvoCurr、RAGEN-2 笔记(自进化 RL、课程进化、环境合成,带 reward 塑形、互信息 MI 诊断、template collapse 分析)是同一个家族:都是 "外循环优化内循环",都靠某种可计算的信号驱动择优。

---

## 4. 关系:同构的外循环,互补的端点

两者都是同一个抽象循环的实例:一个外循环不断**提议候选物,评估它们,留下好的**,其中 LLM 充当(至少一个)变异算子。这正是 FunSearch、AlphaEvolve、AI co-scientist 和你的 GA 加 LLM 系统共享的 "generate, evaluate, select, vary" 范式。

形式上记成:$x_{t+1} = \mathrm{Select}\big(\mathrm{Vary}(x_t),\ \mathrm{Eval}(\cdot)\big)$。区别在于 `Vary`、`Eval`、`Select` 各自被实例化成了什么。

| Axis | Coscientist "evolution" | Your heuristic learning |
|---|---|---|
| Object being optimized | The research system's own code, prompts, roles, tools | A downstream task's loss / reward / config / heuristic |
| Variation operator (`Vary`) | LLM file edits only | GA (weights, combinations) plus LLM (new primitives) |
| Evaluation signal (`Eval`) | Binary `pytest` smoke plus human judgment | Automated numeric fitness (Sharpe, IC, winrate_margin) |
| Selection / acceptance (`Select`) | Human approve, then `git merge --ff-only` | Fitness ranking, top-K survive across generations |
| Population | None (one candidate per command) | Yes (many genomes per generation) |
| Loop cadence | Episodic, human-in-the-loop, one command at a time | Autonomous, many generations until stagnation |
| Diversity mechanism | Human steering | Explicit diversity pressure after N stalled rounds |
| Lineage | AlphaEvolve edit-surface, plus Reflexion / ExpeL (planned) | FunSearch / AlphaEvolve / classic GA plus LLM proposal |

Note: "evolution" in coscientist is LLM-driven self-modification, not evolutionary computation. There is no fitness function or population in its code.

---

## 5. 合起来怎么用(actionable)

第一,可以把 coscientist 看成 "AlphaEvolve 减掉评估器":它有完整的自我修改机器,却没有自动适应度,全靠人审。而你的 heuristic learning 恰好就是 "那个评估器加 GA",只不过你作用在任务启发式上,而不是作用在能自改的 scaffold 上。两边的缺口正好对得上。

第二(组合 A,把你的环接进 coscientist):把你的两级 fitness 环(GA 内层加 LLM 外层)接到 coscientist 的 evolution runtime,就能把它从 "人审自编辑" 升级成真正的闭环自改(AlphaEvolve-complete)。前提是先给 coscientist 的自改定义一个可计算目标(比如某个研究任务的产出质量分、单位 token 成本、smoke 加扩展测试的通过率)。

第三(组合 B,把 coscientist 的 gate 接进你的系统):反过来,coscientist 的 HITL worktree merge 加 `revert.patch` 是一个比 "盲信一个 fitness 数字" 更安全的录用 gate。当你的 fitness 会被 game(reward hacking,或对 AMZN / BLK / APA 这三个 ticker 过拟合)时,可以借鉴这种 "自动信号筛一遍,人审拍板,可一键回滚" 的结构。

第四(直接对应你 skill 里的 anti-pattern):你的 skill 明确写了 "用 LLM 做权重优化会探索坍缩,产出 20 个近似变体"。coscientist 的 evolution 恰恰是纯 LLM edit、没有 GA,所以它做**结构性**改动(加一个 critic role、加一个 phase、改 prompt)很合适,但做**数值 / 权重**自调会正好踩这个坑,而这正是你 GA 层的强项。这给了一个清晰的分工建议:结构改动交给 LLM-edit 式 evolution,数值调参交给 GA。

第五(第二条桥,经验式学习):coscientist 的 memory 层加上计划中的 R2 reflection-lessons(ExpeL / Reflexion 式,从经验里抽 lesson 存进可召回的库)是另一种 "学习",它积累的是文字化的启发式,而不是被进化的代码。这跟你 RAGEN-2 / EvoEnv 里的经验诊断同属 "经验式启发学习"。所以 heuristic learning 跟 coscientist 有两条接触线:一条是对 artifact 的进化搜索,另一条是对 lesson 的经验积累。

---

## 6. 一句话映射

Coscientist 进化的是 "做科研的 agent 自己",评估靠人。Heuristic learning 进化的是 "解任务的启发式 / 损失",评估靠自动 fitness 加 GA。两者缺口互补:把自动 fitness 接进 coscientist 就得到闭环自改;把人审加可回滚 gate 接进 heuristic learning 就得到防 reward hacking 的安全录用。

---

*由 Claude Code 生成。依据:本地 repo `/projects/public/u6gb/OpenPhil_coscientist`(research / evolution runtime 源码)与 skill `meta-learning-evolution`,以及 Notion 上 EvoEnv / EvoCurr / RAGEN-2 研究线。生成日期 2026-06-22。*
