# Continual Learning: Plasticity Loss & Regime Drift

> Grok conversation
> Source: https://grok.com/c/327201c8-4198-4277-992a-d855afa281bf
> Exported: 2026-08-26

---

## User

322 🔎 Continual Learning 深研问题清单：可塑性丧失 × Regime Drift × LOB 基础模型（复制自 #321 子页 5 + 新增问题准备 · 供独立 Deep Research 用）

<aside>

🧭
这一页是什么：第一部分原样复制自 #321 的子页面 5「Continual Learning 缺什么：可塑性丧失（AI 老化）与无标注的 Regime Drift」，作为起点；第二部分是新增的「深研问题准备」：研究目标与边界、9 个主题的问题清单（每题附子问题、判定标准、检索起点）、我的先验假设 H1 到 H5、与 σ0 的决策接口、检索关键词、可直接粘贴的 Deep Research prompt，以及输出格式要求。清单里的年份与会议以检索结果为准，我标注的起点可能有误。

</aside>
第一部分 · 原页内容（复制自 #321 · 子页面 5）
Wang 最关心的两个问题（转述，对应图 9 到图 10）
1. 领域过度关注遗忘，更深的问题是模型逐渐陈旧并永久失去可塑性，他称之为 AI 的「老化问题」。几乎所有 Continual RL 方法都站在稳定性与可塑性困境（Stability vs Plasticity Dilemma）的稳定一侧，如 Elastic Weight Consolidation、经验回放、Progressive Networks，主要在防灾难性遗忘。但实验显示网络不只会遗忘，还会僵化：训练推进时神经元休眠、表征坍缩、梯度失去意义、memory buffer 过满；即使移除全部防遗忘约束，网络也既学不进新的也忘不掉旧的，只是「死了」。灾难性遗忘是网络活着时才有的下游症状，可塑性丧失会杀死 AI；目前既没有解释这一过程的机制理论，也没有原则性的解法。
2. 真实环境不会弹出「任务已切换」的标记。世界在无标注下持续漂移，Task 或 Regime 是人为定义的概念，金融市场是最好的例子；更深的问题是 Continual Learning 里连「任务」都没有清晰定义。更进一步，也许正确答案是停止用 Task 思考，转而构建不断检测分布信号并据此调节自身学习过程的策略，不必先把变化定义成某项任务、某个事件、某种情境或一次 regime 切换。
交叉验证与延伸
* 可塑性丧失有一支正在快速增长的文献：Dohare 等人在 Nature 2024 发表的「Loss of plasticity in deep continual learning」（Sutton 组，提出 continual backpropagation，通过持续重置低效用单元维持可塑性）；Sokar 等人 ICML 2023 的 dormant neuron 现象；Nikishin 等人 ICML 2022 的 primacy bias；Lyle 等人 2023 年关于「理解神经网络可塑性」的分析。Wang 的表述与这支文献一致，但他把它推到「没有机制理论」这一更强的判断（目前确实只有现象层面的诊断指标，如有效秩、活跃单元比例、梯度范数，和重置类启发式解法）。
* EWC（Kirkpatrick 等 2017）、经验回放、Progressive Networks（Rusu 等 2016）确实都属于「稳定性一侧」的方法，这一分类是准确的。
* 「Task 没有清晰定义」与 task-free continual learning、online continual learning 的文献呼应；在金融里对应的是 regime detection（隐马尔可夫、变点检测、在线 Bayesian 变点检测）与自适应学习率、自适应窗口等方法。
* 数学上可以把 Wang 的第二点写成：不假设离散的 regime 变量 $$ z_t $$（取值于有限集合），而是让学习过程本身依赖一个连续的漂移统计量 $$ s_t = D(P_{t-w:t},\, P_{t-2w:t-w}) $$（某种分布距离），学习率或记忆写入强度写成 $$ \eta_t = f(s_t) $$。这与 GDN 的 gating（决定擦除多少旧状态）在形式上是同一个问题的两个尺度：token 级 vs 训练级。
* wiki 内相关词条：#24（持续学习为何仍是最大难题）、#27（RL 是否通往持续学习）、#58（仅靠上下文做不出真正的持续学习）、#86（LoRA 作为经验内化层）、#253（预训练 VLA 天生抗灾难性遗忘）。Wang 的结论与 #58 的立场一致：在 context 里堆信息不等于持续学习，因为可塑性是权重层面的问题。
对我而言的要点
* 部署一个预训练 LOB 模型意味着必须回答「模型随时间陈旧」的问题：评估应按时间切分（训练期之后的年份、含 2020 年 3 月、2024 年 8 月等高波动时段），报告性能随「距训练集末尾时间」的衰减曲线。
* Test-time training 与 continual pretraining 是两条候选路线（我之前关心过 TTT），两者都要面对可塑性丧失；continual backprop 类方法值得作为 baseline。
* 把「漂移统计量驱动学习率」做成一个小实验：在 sigma-0 的逐日流式微调里，用验证 NLL 的滑动窗口变化作为 $$ s_t $$，对比固定学习率与 $$ \eta_t = f(s_t) $$ 在高波动月份的表现。
第二部分 · 新增：深研问题准备
0. 研究目标与范围边界
一句话目标：弄清对一个在 NASDAQ 订单流上预训练、之后要持续服役多年的 LOB foundation model（σ0/LOBS5），「可塑性丧失」与「无标注 regime drift」这两个问题在 2023 到 2026 年的文献里分别有多确定的机制解释、多可靠的解法、多可复用的评估协议；并把结论落到 σ0 的三个具体决策上：持续预训练配方、评估协议、架构层面（GDN gating / TTT）的对应实验。
纳入范围：deep RL 中的 plasticity 文献；监督与自监督持续学习；LLM continual pretraining；时间序列与金融的非平稳适应；test time training 与 fast weights；在线学习理论（dynamic regret、变点检测）。
排除范围：纯 class incremental 图像分类 benchmark（Split CIFAR 一类）的排行榜式比较，除非它给出机制性结论。
时间范围：优先 2023 到 2026；经典工作只作为锚点。
1. 问题清单（9 个主题）
每题给出：子问题、判定标准（我期望的答案形态）、检索起点。
Q1 可塑性丧失的现象学与度量
子问题：
1. 在监督学习、深度 RL、LLM 继续预训练三种设定下，「可塑性丧失」分别被怎么定义与度量？常用指标：dormant neuron 比例、有效秩 / stable rank、权重范数增长、梯度范数与梯度多样性、Hessian 曲率 / sharpness、「对新目标的拟合速度相对于新初始化模型」。这些指标之间的一致性如何？有没有哪一个被证明是充分或必要的？
2. 可塑性丧失是否随训练时长单调？是否存在「先快后慢」的阶段结构？
3. 与灾难性遗忘的关系：能否在不遗忘的前提下单独观察到可塑性丧失（Wang 的「网络死了」）？有没有把两者放在同一实验里分离的工作？
判定标准：一张「指标 × 设定 × 是否可复现」的表；明确回答「业界是否有公认的可塑性度量」。
检索起点：Lyle 等「Understanding plasticity in neural networks」（ICML 2023）；Lyle 等「Disentangling the causes of plasticity loss in neural networks」（2024）；Sokar 等 dormant neuron（ICML 2023）；Nikishin 等 primacy bias（ICML 2022）；Abbas 等「Loss of plasticity in continual deep RL」（CoLLAs 2023）；Dohare 等（Nature 2024）。
Q2 机制理论：Wang 说「没有机制理论」，2026 年还成立吗？
子问题：
1. 候选机制清单及其证据强度：ReLU 死亡 / 单元饱和；表征秩坍缩；权重范数增长导致有效学习率衰减（尤其在 normalization 之后）；曲率方向减少（Lewandowski 等「curvature explains loss of plasticity」）；特征学习退化到 lazy / NTK regime；梯度多样性下降；优化器状态（Adam 二阶矩）陈旧。
2. 这些机制中哪些互相蕴含、哪些可以独立复现？有没有「单一充分机制」的反例？
3. 有没有理论工作（不只是实验）给出可塑性随时间衰减的定量刻画，比如线性网络、随机特征模型或高维极限下的可解结果？
4. 与 Lipschitz、谱范数、Muon 类优化器（保持谱）的联系：谱控制是否被证明能防止可塑性丧失？
判定标准：给出每个机制的证据等级（理论证明 / 受控实验 / 相关性观察），并明确回答 Wang 的判断是否过时。
检索起点：Lewandowski 等（2023 到 2024）；Lyle 等「Normalization and effective learning rates in RL」（2024）；Kumar、Marklund、Van Roy「Maintaining plasticity via regenerative regularization」（2023）；Elsayed 与 Mahmood 的 UPGD（2024）；本 wiki #108 Muon 谱滤波、#211 MuonSSM、#33 Lipschitz 连续性。
Q3 解法谱系与在大模型上的可迁移性
子问题：
1. 按机制给解法分类：重置类（continual backprop、ReDo、shrink and perturb、plasticity injection、周期性 reset 如 SR-SPR / BBF）；正则类（L2 toward init / regenerative regularization、Wasserstein 正则、谱 / 秩正则、weight decay 设计）；归一化与激活类（LayerNorm 位置、CReLU、学习率与范数的解耦）；架构类（Hare and Tortoise 双网络、neuroplastic expansion、MoE 新增专家）；优化器类（Muon 型正交化、UPGD）。
2. 各类解法主要在什么规模、什么设定上验证过（是否以小网络 RL 居多）；有没有在 100M 参数以上、序列模型、continual pretraining 上验证的？
3. 代价：重置类是否损害已学能力；正则类是否降低上限；哪种对「漂移而非切换」的环境更合适。
4. 在 LLM continual pretraining 里，rewarming、replay 比例、无限学习率调度（infinite LR schedule）这些工程解与上面的机制解法是什么关系？它们解决的是遗忘还是可塑性？
判定标准：一张「解法 × 针对机制 × 验证规模 × 代价 × 适合的漂移类型」的表；明确哪些可以直接搬到 sigma-0（GDN / Mamba3 骨干，几百 M 参数，长序列）。
检索起点：Dohare 等（Nature 2024）；Sokar 等 ReDo；Ash 与 Adams shrink and perturb（NeurIPS 2020）；Nikishin 等 plasticity injection（NeurIPS 2023）；Schwarzer 等 BBF（ICML 2023）；Lee 等 Hare and Tortoise（ICML 2024）；Gupta 等「Continual pre-training of LLMs: how to (re)warm your model」（2023）；Ibrahim 等「Simple and scalable strategies to continually pre-train LLMs」（2024）；本 wiki #78 PEFT 持续学习全景、#86 LoRA 经验内化层。
Q4 规模与预训练是否「免疫」：可塑性丧失是不是主要发生在从头训练与 RL 里？
子问题：
1. 有没有证据表明预训练的大模型在持续微调 / 持续预训练中不出现可塑性丧失，只出现遗忘（对照本 wiki #253：预训练 VLA 天生抗灾难性遗忘、2% 回放近零遗忘）？
2. 模型宽度、深度、参数量与遗忘、可塑性的关系（Ramasesh 等「Effect of scale on catastrophic forgetting」ICLR 2022；Mirzadeh 等「Wide neural networks forget less catastrophically」ICML 2022；Kalajdzievski「Scaling laws for forgetting when fine-tuning LLMs」2024）。是否有「scaling law for plasticity」？
3. 如果规模有保护作用，机制是什么（过参数化提供了未使用的容量？表征更线性可分？）；这种保护在多年漂移下是否会耗尽？
判定标准：能回答「对一个 300M 到 1B 的预训练 LOB 模型，可塑性丧失是不是真实风险」，并给出证据方向与缺口。
Q5 无标注漂移下的学习策略：从 task free CL 到「漂移信号驱动的学习率」
子问题：
1. task free / online continual learning 的主流做法（Aljundi 等 CVPR 2019 起）如何在没有任务边界时决定「何时巩固、何时重置」？
2. 变点 / 漂移检测方法（ADWIN、DDM、Page Hinkley、BOCPD、KL 或 MMD 窗口检验）与学习率或记忆写入强度耦合的工作有哪些？有没有理论保证（dynamic regret、adaptive regret、variation budget 下的 tracking bound）？
3. 「停止用 Task 思考」的形式化候选：非平稳在线学习（Besbes、Gur、Zeevi 的 variation budget，2015）、tracking the best expert（Herbster 与 Warmuth，1998）、adaptive regret（Hazan 与 Seshadhri，2009）、Bayesian 潜 regime（HMM、jump model）；哪一种最适合 LOB 数据（既有缓慢漂移又有突变）？
4. 元学习学习率 / hypergradient（Baydin 等 hypergradient descent 2018，及 D-Adaptation 一类无参数方法）能否作为 $$ \eta_t = f(s_t) $$ 的实现？
判定标准：给出 2 到 3 个可直接实现的 $$ \eta_t = f(s_t) $$ 方案，附各自的假设与失败模式。
检索起点：Aljundi 等（CVPR 2019）；Gama 等 concept drift 综述（ACM CSUR 2014）；Bifet 与 Gavaldà ADWIN（2007）；Adams 与 MacKay BOCPD（2007）；Besbes、Gur、Zeevi（Operations Research 2015）；Hazan 与 Seshadhri（ICML 2009）。
Q6 序列架构内部的「可塑性」：gating、TTT、fast weights 与训练级持续学习是同一个问题的不同尺度吗？
子问题：
1. GDN / Mamba 的 gating（token 级擦除与写入）与训练级的「何时忘、何时学」在数学上是否同构？有没有工作把线性注意力与 delta rule 的状态更新解释为在线学习（在线梯度下降 / 在线回归）？TTT layers 与 Titans 就是这么建模的。
2. TTT layers（Sun 等 2024）、Titans（Behrouz 等 2024）、Nested Learning（Behrouz 等 2025，多时间尺度的嵌套优化，据我所知发表于 NeurIPS 2025）是否解决了「context 里学到的东西无法沉淀到权重」（本 wiki #58 的立场）？它们在长时间流式数据上的可塑性有没有被测过？
3. 在流式部署里，token 级 gating 能否替代一部分训练级适应（即「不重训、只靠状态」能撑多久）？失败的时间尺度是多少？
判定标准：一张「机制 × 时间尺度（token / 序列 / 训练） × 是否沉淀到权重 × 已验证的数据长度」的对照表；给出在 σ0 上做对比实验的最小设计。
检索起点：Sun 等「Learning to (learn at test time): RNNs with expressive hidden states」（2024）；Behrouz 等「Titans: learning to memorize at test time」（2024）与「Nested Learning」（2025）；Yang 等 Gated DeltaNet（ICLR 2025）；Schlag、Irie、Schmidhuber「Linear transformers are secretly fast weight programmers」（ICML 2021）；本 wiki #208 GDN-2、#288 线性注意力 × SSM 趋同、#300 Sparse Delta Memory。
Q7 时间序列与金融里的持续学习：现有做法与证据
子问题：
1. 金融 ML 的非平稳处理主流是什么：滚动重训（cadence 多久）、加权窗口、regime 切换模型、在线学习；有没有对比「周期性重训 vs 持续预训练 vs 在线适应」的受控实验，尤其在 LOB 或高频数据上？
2. 时序基础模型（Chronos、TimesFM、Moirai 等）在分布漂移下的退化与适应方法（time series 的 test time adaptation）有哪些？
3. 有没有公开的「金融 regime 漂移 benchmark」（按事件切分的评估协议）？如果没有，最接近的替代是什么？
4. 业界公开表述（XTX、Two Sigma、Jane Street、Hudson River Trading 等的博客、演讲、论文）里对模型陈旧化的处理方式，有哪些可引用的一手来源？
判定标准：能回答「σ0 的重训或持续预训练 cadence 应该由什么统计量决定」；给出可用的公开事件切分方案。
检索起点：López de Prado「Advances in Financial Machine Learning」（concept drift、backtest overfitting）；DeepLOB 及后续工作的重训策略；Nystrup、Kolm、Lindström 的 jump models；本 wiki #291 TSFM 范畴错误、#231 波动率粗糙与长记忆、#227 Forecast Collapse。
Q8 评估协议：如何在 σ0 上同时测遗忘与可塑性
子问题：
1. 标准 CL 指标（平均性能、forgetting measure、forward / backward transfer；Lopez-Paz 与 Ranzato 2017，Chaudhry 等 2018）如何移植到自回归 NLL 与 LOB-Bench 分布得分上？
2. 「可塑性」在金融数据上的操作性定义：给定一个新时段，比较「继续训练的旧模型」与「同等算力的新初始化模型」达到同一验证 NLL 所需步数（Lyle 式定义）；要多少算力才能得到稳定估计？
3. 时段切分的候选事件（日期与影响需核实）：2016 到 2018 年美股 tick size pilot；2020 年 3 月疫情；2021 年 1 月 meme stock；2022 年加息高波动；2023 年 3 月 SVB；2024 年 8 月 5 日波动率冲击；2025 年 4 月关税冲击；2025 年 SEC Rule 612 tick size 改革（生效日期需核实）；NASDAQ 自身的规则与撮合变更。
4. 需要报告的曲线：性能随「距训练集末尾时间」的衰减；不同 regime 标签下的分层性能；可塑性指标随持续训练步数的变化。
判定标准：一份可以直接写进论文 Experiments 节的协议草案。
Q9 理论与形式化：「任务」到底该怎么定义，还是应该放弃它
子问题：
1. 文献里对 continual learning 的形式化有哪几种（任务序列、非平稳 MDP / POMDP、drifting distribution with bounded variation、Bayesian 潜变量）；哪种能覆盖「没有边界的漂移」？
2. 在这些框架下，「最优策略」是什么（例如 variation budget 下的最优遗忘率、Bayesian 下的最优后验折扣）；有没有把最优遗忘率与学习率或 gating 联系起来的结果？
3. 对 LOB 这类数据可以提出的形式化：$$ P_t $$ 随 $$ t $$ 漂移，$$ \sum_t D(P_{t+1}, P_t) $$ 受 variation budget 约束，目标是 dynamic regret；在这个设定下 GDN gating 的最优值是什么？
判定标准：写出一个 σ0 可采用的形式化定义，以及它对应的理论上限与下限。
2. 我的先验假设（供 Deep Research 检验）
* H1：可塑性丧失主要是从头训练与深度 RL 里的现象；大规模预训练模型在持续预训练里主要面对的是遗忘与漂移，而不是「网络死了」。若为真，σ0 的重点应放在 Q5、Q7、Q8。
* H2：LOB 数据的漂移是「慢漂移（微观结构、参与者构成）+ 突变（事件、规则变更）」的叠加；两者需要不同机制（慢：持续预训练 + 漂移信号学习率；突变：变点检测 + 短期适应或 regime 条件化）。
* H3：GDN 的 gating 实现了 token 级的「忘」，但不能替代训练级的「学」；只靠状态不重训，性能会在数周到数月尺度上明显退化（需要数据）。
* H4：谱控制（Muon 类）与 L2 toward init 是最有可能不损害上限的可塑性保持方法；重置类方法在预训练模型上代价过大。
* H5：Wang 的「没有机制理论」在 2026 年只对一半：曲率、有效学习率、秩坍缩已有较强的机制解释，但没有统一理论，也没有能预测「何时死」的定量刻画。
3. 与 σ0 的决策接口（研究结论要落到这五条上）
1. 持续预训练配方：rewarm 学习率多少、replay 旧数据比例、是否用漂移信号调学习率、是否加谱或范数正则。
2. 评估协议：Q8 的时段切分与指标，写入 sigma-0 的评估脚本与论文 Experiments 节。
3. 架构实验：GDN gating 统计量能否作为漂移探测器；TTT 或 Titans 式层是否值得加进骨干做对比。
4. 服役策略：重训 cadence 由哪个统计量触发；何时接受「状态适应」、何时必须重训。
5. 论文叙事：把 Wang 的「金融是 CL 最典型环境」作为动机，把 Q8 协议作为贡献之一。
4. 检索关键词与关键作者
英文关键词：loss of plasticity; plasticity loss deep RL; dormant neurons; primacy bias; continual backpropagation; regenerative regularization; effective rank collapse; curvature plasticity; effective learning rate normalization; continual pre-training LLM rewarming replay; task-free continual learning; online continual learning; concept drift detection ADWIN BOCPD; non-stationary online learning dynamic regret variation budget; test-time training layers; Titans learning to memorize at test time; nested learning multi-timescale optimization; fast weights delta rule online learning; time series foundation model distribution shift test-time adaptation; regime switching jump model finance; limit order book non-stationarity retraining。
中文关键词：可塑性丧失；持续学习；灾难性遗忘；概念漂移；变点检测；在线学习动态遗憾；测试时训练；持续预训练；市场状态切换。
关键作者与组：Richard Sutton 组（Dohare、Mahmood）；Clare Lyle、Razvan Pascanu、Will Dabney（Google DeepMind）；Evgenii Nikishin；Ghada Sokar、Utku Evci；Alex Lewandowski；Saurabh Kumar、Benjamin Van Roy；Mila 的 continual pretraining 组（Gupta、Ibrahim、Irina Rish）；Yu Sun（TTT）；Ali Behrouz（Titans、Nested Learning）；Rahaf Aljundi（task free CL）；Albert Bifet、João Gama（concept drift）；Marcos López de Prado。
5. Deep Research prompt 模板（可直接粘贴）
英文版（主要来源是英文文献；把下面整段贴进去即可，末尾的假设与输出要求都已包含）：
```
You are researching continual learning for a pre-trained foundation model of limit order book (LOB) data that must stay in service over years of non-stationary markets. Two problems, as framed by Zhangyang "Atlas" Wang (XTX Markets, 2026): (a) loss of plasticity ("AI aging"): networks that not only forget but stiffen (dormant units, representation collapse, meaningless gradients) and end up unable to learn new things or to forget old ones; (b) unlabeled regime drift: the world never announces a task switch, so "task" is an ill-defined concept and the learner should instead detect distribution signals and adapt its own learning process.

Answer the following nine questions with citations (title, authors, venue, year, arXiv id), distinguishing evidence from deep RL, supervised or self-supervised continual learning, LLM continual pre-training, and time series or finance. Prioritize 2023 to 2026. For every claim, mark the evidence level: theory, controlled experiment, or correlational observation.

Q1 Phenomenology and measurement of plasticity loss. Which metrics (dormant unit fraction, effective or stable rank, weight norm, gradient norm and diversity, curvature or sharpness, fitting speed relative to a fresh network) are used, how consistent are they, and is any of them shown to be necessary or sufficient? Can plasticity loss be observed separately from forgetting?

Q2 Mechanisms. List candidate mechanisms (dead units, rank collapse, weight norm growth reducing the effective learning rate, curvature loss, lazy or NTK regime, optimizer state staleness) with evidence level. Is there any quantitative theory predicting when plasticity is lost? Does spectral control (Muon type optimizers, spectral norm regularization) provably or empirically prevent it? Is the claim "no mechanistic theory exists" still accurate?

Q3 Remedies. Classify remedies (resets such as continual backpropagation, ReDo, shrink and perturb, plasticity injection, periodic resets; regularizers such as L2 toward initialization, regenerative regularization, Wasserstein or spectral regularization; normalization and activation choices; architectural expansion; optimizer designs). For each: scale and setting where validated, cost to already learned ability, suitability for slow drift versus abrupt switches. How do LLM continual pre-training practices (rewarming, replay ratio, infinite learning rate schedules) relate to these remedies, and do they address forgetting or plasticity?

Q4 Scale and pre-training. Do large pre-trained models still lose plasticity under continual fine-tuning or pre-training, or do they mainly forget? Summarize evidence on model width, depth and size versus forgetting and plasticity, including any scaling laws for forgetting. If scale protects, what is the mechanism and does the protection run out under multi-year drift?

Q5 Learning without task labels. How do task-free and online continual learning methods decide when to consolidate or reset? Which works couple drift or change point detection (ADWIN, DDM, Page Hinkley, Bayesian online change point detection, KL or MMD window tests) to the learning rate or memory write strength, and with what guarantees (dynamic regret, adaptive regret, variation budget tracking bounds)? Which formalization (variation budget, tracking the best expert, adaptive regret, latent regime models) best fits data with both slow drift and abrupt shocks? Can hypergradient or meta-learned learning rates implement a drift-driven learning rate?

Q6 Plasticity inside sequence architectures. Is the token level erase and write gating of Gated DeltaNet or Mamba mathematically the same problem as training level continual learning at a different time scale? Which works interpret linear attention and delta rule state updates as online learning (test-time training layers, Titans, Nested Learning)? Do these mechanisms consolidate what is learned in context into weights, and have they been evaluated on long streaming data? How long can state-only adaptation substitute for retraining before performance degrades?

Q7 Time series and finance. What do practitioners and papers actually do about non-stationarity in financial and LOB models (retraining cadence, weighted windows, regime models, online learning)? Are there controlled comparisons of periodic retraining versus continual pre-training versus online adaptation on LOB or high frequency data? How do time series foundation models degrade under distribution shift and what test-time adaptation methods exist? Is there any public regime drift benchmark for finance? Cite first party industry sources where available.

Q8 Evaluation protocol. How should standard continual learning metrics (average performance, forgetting measure, forward and backward transfer) be adapted to autoregressive negative log likelihood and to distributional benchmarks such as LOB-Bench? Propose an operational definition of plasticity on financial data (steps needed by a continued model versus a fresh model of equal compute to reach a target validation loss) and the compute needed for a stable estimate. Propose a time sliced evaluation using verifiable market events (US tick size pilot 2016 to 2018, March 2020, January 2021, 2022 rate hikes, March 2023, 5 August 2024, April 2025, the 2025 SEC Rule 612 tick size change; verify dates and effects).

Q9 Formalization. Survey formal definitions of continual learning (task sequences, non-stationary MDPs or POMDPs, drifting distributions with bounded variation, Bayesian latent regimes). Which cover boundary free drift? What is the optimal forgetting rate or discount under each, and is it linked to learning rates or gating? Propose a formalization for LOB data with a variation budget and a dynamic regret objective.

Hypotheses to test and give a verdict on (supported, refuted, or undetermined, with the deciding evidence):
H1 Plasticity loss is mainly a from-scratch and deep RL phenomenon; large pre-trained models in continual pre-training mainly face forgetting and drift rather than "dead" networks.
H2 LOB drift is a superposition of slow drift (microstructure, participant mix) and abrupt shocks (events, rule changes), and the two need different mechanisms.
H3 Gating in Gated DeltaNet implements token level forgetting but cannot replace training level learning; state-only adaptation degrades within weeks to months.
H4 Spectral control (Muon type optimizers) and L2 toward initialization are the plasticity remedies least likely to hurt the performance ceiling; reset based remedies are too costly on pre-trained models.
H5 The claim "no mechanistic theory of plasticity loss exists" is only half true in 2026: curvature, effective learning rate and rank collapse have strong mechanistic explanations, but there is no unified theory and no quantitative prediction of when a network dies.

Output: (1) an executive summary of at most 300 words; (2) findings per question with citations and evidence levels; (3) a table of remedies (method, mechanism targeted, validated scale and setting, cost, suitable drift type); (4) a table of mechanisms (mechanism, evidence level, key references); (5) a list of open problems; (6) a verdict on H1 to H5; (7) a draft evaluation protocol for a 300M to 1B parameter LOB model. Flag anything you could not verify. Do not merge results from different settings without saying so.

```text
中文版要点（如果用中文工具）：把上面 9 个问题与 5 个假设原样翻译即可；强调三件事：按设定区分证据来源、每条结论标注证据等级、所有论文给出 arXiv id。
6. 输出格式要求与注意事项
* 区分设定：RL、监督、LLM 预训练、时序，很多结论不跨设定迁移。
* 区分遗忘与可塑性：文献常混用，要求作者明确定义。
* 优先一手来源，核对 arXiv id；本清单里的年份与会议可能有误，以检索结果为准。
* 对 Wang 的两个判断（「没有机制理论」「任务没有清晰定义」）给出明确的是、否或部分成立，附依据。
* 回来后把结论按第 3 节的五条决策接口写回：σ0 的评估脚本、论文 Experiments 节、以及这一页下方新开的「结论」小节。
7. 结论（预留，Deep Research 回来后填写）
（此处预留。）

only look at tweets do a deep research
```

---

## Grok

只看 X 推文的深研结论如下。Atlas Wang / XTX 原话在公开推文里**没有找到**；下面全部是从业者与作者在 X 上的一手表述，不是论文精读。

## 执行摘要（≤300 词）

X 上 2024–2026 的共识几乎对齐 Wang 的第一点：持续学习有两半，遗忘被过度测量，**可塑性丧失**才是网络“死掉”。Sutton 组把这写成 Nature 论文并反复强调；Zyphra（Beren Millidge）在 GPT 风格 5M–314M 模型上给出迄今最硬的 LLM 证据：可塑性按拟合新语言的速度度量，onset 服从 $T \propto P^{0.83}$，且**平稳预训练混合物里同样出现**——规模只推迟，不免疫。[⁠@RichardSSutton](https://x.com/RichardSSutton/status/1827409881890222214)

机制上 X 共识是“有候选、无单一充分原因”。Zyphra 测了 dormant MLP、lazy/collapsed attention、参数范数，**没有 smoking gun**；ICML 2026 海报把谱坍缩推到因果位；梯度派批评 activation 式 dormant 指标在现代网络失效。解法谱系仍以重置（continual backprop）和双时间尺度（Nested Learning / Titans / TTT / Delta-Mem sidecar）为主，作者自己也承认重置“次优”、Muon 是否就能治好还没跑完。[⁠@ZyphraAI](https://x.com/ZyphraAI/status/2069835021943497041)

Wang 第二点（无标注 regime、不要用 task 思考）在 X 上被金融/流式 ML 侧面印证，但**几乎没有 XTX/JS/Two Sigma 的可引用内部做法**。量化推文停留在“月度重训太慢、应按绩效/波动触发、模型有保质期”，没有 LOB-Bench 式协议。对 σ0：H1 被 Zyphra **部分证伪**（预训练+LLM 也会死）；H3 在推文里是主流架构赌注（快权重扛会话、慢权重不能替代多年重训）；评估必须同时报遗忘和“相对新初始化的拟合速度”。

---

## Q1 现象学与度量

X 上实际在用的操作性定义：

| 度量 | 谁在推 | 设定 | 一致性 |
| --- | --- | --- | --- |
| 对新数据的拟合速度（相对新初始化 / 相对早期 checkpoint） | Zyphra：循环 8 语后微调越南语，越慢越死 | LLM 持续训 5M–314M | 被当作主指标 |
| Dormant 比例（激活阈值） | Sokar / Castro / Evci ReDo 线 | 深度 RL | 2025 被梯度派打脸 |
| 梯度健康（神经元是否还能学） | GraMa / NeurIPS 2025 | 深度 RL | 声称激活指标在 LayerNorm/非 ReLU 下失效 |
| Attention 坍缩 / lazy heads | Zyphra 审计 GPT 风格 | LLM | 相关但“不能完全解释” |
| 谱坍缩 | ICML 2026 poster | 深度持续学习 | 作者主张驱动因素，推文无复现细节 |

关键区分被反复强调：遗忘 = 旧知识丢了；可塑性丧失 = **学不动新的，也不一定忘得掉旧的**。[⁠@ZyphraAI](https://x.com/ZyphraAI/status/2069835009415090328)

单调性：Zyphra 描述为先改善再“稳步、不可逆退化”，不是单纯从第一步就死。有没有“公认度量”？**没有。** Millidge 明确写：用小网络文献里的指标去相关 LLM 可塑性，**找不到 smoking gun**。[⁠@BerenMillidge](https://x.com/BerenMillidge/status/2069841781894857158)

能否与遗忘分离：Zyphra 在**无任务切换的平稳混合预训练**里仍看到可塑性下降，这是 X 上最接近“网络活着但在变僵”的证据。是否“既学不进也忘不掉”的死态，推文没有单独消融。

证据等级：受控实验（作者自述）+ 相关性观察。无独立复现推文。

---

## Q2 机制：2026 年“没有机制理论”还成立吗？

X 上的候选与证据强度（全部是作者/评论者陈述）：

| 机制 | 证据等级（按推文） | 要点 |
| --- | --- | --- |
| Dormant / 死单元 | 受控实验（RL）；LLM 上仅相关 | ReDo 线强；LLM 上不够 |
| 激活指标误诊 | 受控实验声称 | 应用梯度而非激活 |
| Attention 坍缩 / lazy heads | 相关性 | Zyphra：一起涨，解释不完 |
| 参数幅度增长 | 相关性 | 同上 |
| 谱坍缩 | 作者主张因果（ICML 2026；SingularClip 2026-08） | 推文无证明细节 |
| 自由度被训练锁死 | 解释性评论 | @scaling01：早期自由度最大，承诺方向后更难学 |
| Adam 陈旧 / 优化器伪影 | 开放假说 | Millidge：“如果可塑性丧失只是 Adam 的后果就好笑了”，计划测 Muon |
| 谱各向异性被 Muon 抹平 | 优化器讨论（ISO / RLVR） | 与“要不要保持谱”直接相关，未接到可塑性实验 |

Wang「没有机制理论」在 X 上的 2026 判决：**半成立。** 现象与若干相关物已清楚；Zyphra 自己写“没有干净的原因或修法”；谱坍缩是最接近“单一机制”的新叙事，但尚未成为社区共识。没有推文给出“何时死亡”的可解理论（线性网络/NTK 定量刻画在 X 上基本缺席）。[⁠@ZyphraAI](https://x.com/ZyphraAI/status/2069835021943497041)

谱控制能否防：未证明。正向信号是 Spectral Collapse / SingularClip 把谱当杠杆；反向是 ISO 讨论里有人认为 RLVR 更该**继承**预训练谱而不是 Muon 式抹平。H4 在推文里是合理优先实验，不是定理。

---

## Q3 解法谱系与可迁移性

X 上实际被点名的解法：

| 方法 | 针对 | 验证规模（推文所称） | 代价/评论 | 漂移类型 |
| --- | --- | --- | --- | --- |
| Continual backprop / 按效用重置单元 | 死单元、可塑性 | Sutton 组，偏小网络/RL | Teortaxes：次优 | 持续非平稳 |
| ReDo | dormant | 深度 RL | 2025 称在现代结构上不够 | 切换式 RL |
| Shrink-and-perturb / 周期性 reset | 可塑性 | 较少新推 | — | — |
| 快/慢双网络（Hare-Tortoise 精神；Nested Learning；Titans；TTT；Delta-Mem sidecar） | 把新知识放到快权重，慢权重稳定 | LLM/序列，论文级宣传 | 被当作“已有现成架构” | 会话–周尺度漂移 |
| FST：快通道（prompt）+ 慢通道（RL 更参） | 遗忘+可塑性 | 推文转述 Berkeley+Mila | 声称新任务上 RL 崩、FST 能适应 | 任务切换 |
| 重训期强正则 / 平滑表征 | 过早承诺 | 视觉直觉，语言不确定 | 个人推测 | 慢漂移 |
| 经验回放 + Transformer（in-context） | 可塑性 | 论文被转，非推文实验 | 猜想靠 ICL | — |
| 架构改结构（ICML 2026 poster） | 可塑性 | 未报 LLM 规模 | “能否撑到 LLM 还不知道” | — |
| LLM CPT 工程：rewarm / replay / 合并代替配比 | 主要是遗忘+配比 | OptiMer 等 | 几乎没人说这是在治“网络死亡” | 语料切换 |

对 σ0（几百 M、GDN/Mamba 骨干）推文能直接搬家的：

1. 把“越南语拟合速度”换成“新时段验证 NLL 下降速度”，作为可塑性仪表。
2. 不要只靠重置骨干。
3. 值得做的对照是 Muon/谱正则 vs Adam，以及 sidecar 快权重 / TTT 层 vs 纯 GDN 状态。
4. CPT 的 rewarm/replay 在 X 叙事里治的是遗忘与配比，不要当成可塑性解法。

---

## Q4 规模与预训练是否免疫

这是对 H1 最关键的推文块。

Zyphra：5M 到 314M **全部最终失去可塑性**；更大模型 transfer 更好、撑得更久，onset $T \propto P^{0.83}$（次线性）。平稳混合预训练同样发生 → **不是非平稳才有的病**。[⁠@ZyphraAI](https://x.com/ZyphraAI/status/2069835016763572264)

因此：对 300M–1B 的预训练 LOB 模型，可塑性丧失在 X 证据下是**真实风险**，不是“只有从头训/RL 才有”。保护机制在推文里被说成“更多未承诺自由度 / 更好 transfer”，但会耗尽。没有人给多年市场漂移的耗尽曲线。

缺口：314M 以上、已充分预训练再做多年流式微调的数据，X 上没有。H1 的“主要是遗忘不是死亡”对 frontier 生产模型仍可能对（没人在线改核心权重），对 σ0 这种会持续预训练的中等模型则危险。

---

## Q5 无标注漂移：学习率 / 重训触发

X 上几乎没有 ADWIN/BOCPD 耦合 $\eta_t$ 的作者线程，但有三条可落地的从业者启发式：

1. 绩效漂移触发：收益掉出历史带、回撤变宽 → 重测/重训。
2. 市场状态触发：波动或趋势相位明显切换后重测。
3. 计划维护：没坏也定期重测，抓慢漂移。⁠@GT_Protocol

另有一篇 2026-08 论文在 X 被广播：*When to Retrain*——在漂移、预算、延迟约束下比较重训政策。这正好是 $\eta_t = f(s_t)$ 的工程版，但推文没有公式。[⁠@Memoirs](https://x.com/Memoirs/status/2091668875054026798)

量化侧直接观点：“月度批量重训在非平稳里又慢又浪费，regime 已经走了。”[⁠@TheMintingM](https://x.com/TheMintingM/status/2092623757160628490)

形式化候选（variation budget / adaptive regret / HMM）在推文里几乎不出现。最贴近 LOB 的口语模型仍是：**慢漂移用持续更新 + 计划维护；突变用绩效/波动触发的短期适应。** 超梯度 / D-Adaptation 作为 $f(s_t)$ 的实现，X 上本轮检索没有实质讨论。

可实现的 3 个 $\eta_t = f(s_t)$（从推文翻译成实验，不是论文）：

- 验证 NLL 滑动窗口：$s_t = \Delta$NLL$_{w}$，涨则升温学习率或打开写入。失败：把短暂噪声当 regime。
- 波动/价差状态机：高波动 regime 提高 $\eta$ 或缩短重训间隔。失败：波动≠可学结构变化。
- 可塑性探针：固定一小块“未来风格”held-out，拟合速度变慢则触发 reset/正则而不是盲目加大 $\eta$。失败：探针分布选错。

---

## Q6 序列架构内部的可塑性（GDN / TTT / Titans）

X 叙事把这当成**同一问题的不同时间尺度**：

- Token/序列级：GDN 门控、线性注意力状态、TTT 更新、Titans 记忆、Delta-Mem sidecar。被说成 session 级可塑性，不覆盖慢权重。
- 训练级：改骨干才能把 context 里学会的东西沉淀。Wiki #58 的立场在推文里被重复：生产上没有前沿 LLM 在线改核心权重；用的是批量微调、LoRA、外部记忆。⁠@grok

Nested Learning（Google / Behrouz，X 上称 NeurIPS 2025）被当作多时间尺度嵌套优化的现成答案；@scaling01 直接说快网络高 weight decay、再蒸馏进慢网络，“Oh yeah, we already have this… follow Ali Behrouz”。[⁠@scaling01](https://x.com/scaling01/status/2070307798987022641)

“只靠状态能撑多久”：**推文给不出周/月数字。** 这是 H3 的缺口。σ0 最小对照实验（推文能支撑的设计）：

1. 冻结骨干，只跑 GDN/TTT 状态过 1 天 / 1 周 / 1 月持出时段。
2. 同算力继续预训练。
3. 同算力新初始化。报告三条 NLL 与可塑性探针。若 1 在数周内贴近 3 的差距扩大，H3 成立。

---

## Q7 时序与金融：X 上真正在说的做法

能引用的一手很少，且**不是 XTX/JS/HRT 技术博客**：

- 策略层：绩效带 + 波动切换 + 定期维护。
- 观点层：模型有保质期；参数可改、也可分 regime 跑多模型；难的是“是否已经换 regime”。⁠@AlphaMind101
- 基础模型苗头：M3（state-event generative FM for microstructure）、TradeFM（trade-flow FM）出现在 2026 推文，但没有漂移评测讨论。⁠@saeedamenfx
- 支付 FM（Vulcan）宣传“每笔支付成为下一笔的训练信号”——是持续学习产品叙事，不是 LOB 证据。

受控对比「周期重训 vs 持续预训练 vs 在线适应」在 LOB 上：**X 上没有。**

公开金融 regime-drift benchmark：**没有人指一个。**

这是本清单里推文最空的一题。σ0 的 cadence 统计量，X 只能建议用**验证 NLL 带 + 波动状态 + 可塑性探针**，不能替你选阈值。

---

## Q8 评估协议草案（从推文指标翻译到 LOB）

标准 CL 的 average / forgetting / FWT/BWT 在推文里几乎无人映射到 NLL。可直接写进 Experiments 的推文兼容协议：

**主指标**

- 自回归 NLL（及 LOB-Bench 分布分，若已有）。
- Forgetting：旧时段 NLL 在继续训练后的回升。
- Plasticity（Lyle/Zyphra 式）：给定新时段 $D_{new}$，旧模型继续训达到目标 NLL 的步数 $S_{cont}$，对比同等算力新初始化 $S_{fresh}$；可塑性分数 $S_{fresh}/S_{cont}$（<1 表示变僵）。
- 衰减曲线：性能 vs「距训练集末尾的日历时间」。

**时段切分**：推文**没有核实** 2016–18 tick pilot、2020-03、2021-01、2022 加息、2023-03 SVB、2024-08-05、2025-04 关税、SEC Rule 612 生效日。这些日期不能从本轮 X 检索确认，必须另查。协议上仍应按**可验证微观结构事件 + 高波动月**切，避免按“任务标签”切。

**算力**：Zyphra 用 5B tokens/语 × 多循环才看出曲线；σ0 上至少需要对每个切分做「续训 vs 新初始化」成对跑，否则可塑性分数不稳定。推文给不出具体 GPU-小时。

---

## Q9 形式化

X 上几乎不谈 variation budget / adaptive regret。出现的口语形式化：

- 智能来自 runtime experience 流（Sutton / Oak Lab / Keen），而不是静态数据集。
- Liang Wenfeng：CoT/Agent 之后下一个问题是连续学习，不是再给一次更强的离线训练。⁠@deredleritt3r
- 生产现实：适应发生在编排层、记忆、LoRA，而不是在线改满血权重。

对 LOB 可写的、与推文不矛盾的最小形式化：

不设离散 $z_t$。令 $s_t$ 为窗口分布距离或验证 NLL 变分；决策是 $\eta_t$、replay 比、以及门控写入强度。目标是跟踪非平稳 $P_t$ 的预测损失（口语版 dynamic regret）。GDN 门控是 token 级 $f(s)$，训练级 $f(s)$ 仍要另设。理论上下界：推文没有。

Wang「task 没有清晰定义」：**在 X 金融讨论里成立**；在学术 CL 推文里仍大量使用语言切换、任务循环作为探针（Zyphra 自己也用 8 语循环），说明社区口头反对 task、实验仍靠 task。

---

## H1–H5 判决（仅基于推文）

| 假设 | 判决 | 决定性推文 |
| --- | --- | --- |
| H1 可塑性主要是从头训/RL；大预训练模型主要面对遗忘 | **部分证伪** | Zyphra：LLM、平稳预训练、5–314M 全会死；但 >314M 且只做轻度 CPT 的生产模型证据不足 |
| H2 慢漂移+突变，机制应拆开 | **支持（弱）** | 交易者重测规则把“慢维护”和“波动/绩效突变”分开；无 LOB 实验 |
| H3 GDN 门控 ≠ 训练级学习；纯状态数周到数月退化 | **方向支持，时限未定** | TTT/Titans/Delta-Mem 被当成 session 级；无人给出周/月衰减数字 |
| H4 谱控制与 L2-toward-init 最不伤上限；重置对预训练太贵 | **未决，优先实验** | 谱坍缩海报 + Millidge 要测 Muon；重置被评为次优；L2-toward-init 本轮几乎无推 |
| H5 “无机制理论”只对一半 | **支持** | 相关物一堆，Zyphra 承认无干净因果；无“何时死”的定量理论 |

---

## 落到 σ0 的五条接口

1. 持续预训练配方：不要只抄 LLM 的 rewarm/replay（X 认为那是遗忘工具）。加一条可塑性探针驱动的 $\eta_t$；把 Adam vs Muon/谱正则做成一等公民消融；重置只作小比例单元或 sidecar，不要周期性洗骨干。
2. 评估协议：时间切分 + NLL 衰减曲线 + $S_{fresh}/S_{cont}$；事件日期必须另核，不能用本轮推文。
3. 架构实验：GDN 门控统计能否当 $s_t$；TTT/Titans/Delta-Mem sidecar 对照“不重训只靠状态”。
4. 服役策略：触发器三件套——验证 NLL 带、波动状态、可塑性探针。月度死周期在 X 量化舆论里已经被骂。
5. 论文叙事：Wang 的“金融是 CL 典型环境”在 X 上找不到原帖，但 Sutton/Zyphra/交易者三条线足够支撑“无标签漂移 + 模型老化”动机；Q8 协议仍可当贡献，前提是你自己把事件日期核完。

---

## 推文没核到的东西（必须标红）

- Zhangyang “Atlas” Wang / XTX 2026 演讲的原始推文、幻灯、图 9–10：本轮用户检索与语义检索均未命中。
- Jane Street / Two Sigma / XTX / HRT / Optiver 关于模型陈旧化的可引用技术陈述：无。
- SEC Rule 612、2025-04 关税、2024-08-05 等事件的日期与微观结构效应：无可靠推文。
- Dynamic regret / variation budget 与 gating 最优值的理论推文：无。
- 可塑性指标之间的充分/必要证明：无。
- 100M+ 序列模型上重置类方法的代价数字：无。

若下一轮允许离开 X，应优先核：Zyphra arXiv:2606.24752、Dohare Nature 2024、ICML 2026 Spectral Collapse、SingularClip 2608.18319、Nested Learning Google blog、Delta-Mem 2605.12357，以及 Atlas Wang 演讲的非推来源。

136 sources

---

## User

(1) continual learing with ssl
(2) continual learing with rl
(3) continual learing with ttt

---

## Grok

三条线在 X 上不是并列算法，而是**三种监督信号 × 三种时间尺度**。对 σ0 来说：SSL 是你已经在用的训练目标，RL 是可塑性文献最密的地方，TTT 是“不重训能不能撑住”的架构赌注。

## 一张对照

|  | (1) CL × SSL | (2) CL × RL | (3) CL × TTT |
| --- | --- | --- | --- |
| 学什么 | 无标签预测/对比/掩码 | 奖励驱动的策略或价值 | 推理时用当前上下文做一步学习 |
| X 上密度 | **最稀** | **最密、机制也最清楚** | **2024–2026 突然变密** |
| 主要病 | 表征漂、旧 pretext 失效 | 遗忘 + **可塑性丧失**（dormant、primacy、谱阱） | 快权重能适应，慢权重不一定沉淀 |
| 典型解 | 持续预训练、replay、表征稳健 | reset / ReDo / continual backprop / 蒸馏旧策略 | TTT layer、Titans surprise 写入、TTT-E2E 把上下文压进权重 |
| 沉淀到骨干？ | 是（改慢权重） | 是（改慢权重） | **默认否**；E2E 变体才写权重 |
| 对 LOB | 最贴 σ0 目标（NLL/掩码下一事件） | 贴“服役后还能否学新 regime” | 贴 GDN 门控 / 状态适应能撑多久 |

有人直接说：工业界夜里批量重训，然后管这叫 continual learning，TTT 被绕开了。[⁠@chiefofautism](https://x.com/chiefofautism/status/2092207614402695380)

---

## (1) Continual learning × SSL

X 上几乎没有 “continual SSL / CaSSLe” 作者线程。能站住的只有四条。

**SSL 本身不是无监督圣杯。** Hu Xu：今天的 SSL 仍由人规定任务、目标和数据；标签甚至嵌进了 $X$ 的分布，更像 unimodal supervision。真正的自监督“还没到来”。对 LOB 的含义：你的下一事件/掩码目标也是人造 pretext，regime 一变，pretext 与可交易结构的对齐会 silently 坏掉。[⁠@Hu_Hsu](https://x.com/Hu_Hsu/status/2091307069840531584)

**无标签表征对分布偏移相对稳健。** ICLR 2023 工作在 X 上的结论是：相对有监督，SSL 表征在 shift 下更稳。这支持 σ0 用 SSL 预训练而不是过早接到有标签下游。但这是一次性稳健性，不是多年持续训练后仍能学。[⁠@YugeTen](https://x.com/YugeTen/status/1617508808854568960)

**“持续学习”这个词正在被用烂，像几年前的 SSL。** 定义不清本身就是信号：很多人把夜间 CPT、replay、LoRA 都叫 CL。评价 σ0 时必须写明：改的是慢权重还是只改状态。[⁠@tensorfiend](https://x.com/tensorfiend/status/2090081933233725913)

**LLM 持续预训练是 SSL-CL 在语言里的近亲。** Zyphra 用多语循环 + 越南语微调速度当可塑性探针，且平稳混合预训练里同样变僵。σ0 的自回归 NLL 就是金融版 SSL；他们的结论直接打击 H1：不是只有 RL 才死。

X 上缺的（必须标红）：对比学习在类别增量下的表征坍缩、CaSSLe 蒸馏、在线 SSL 的负样本库过期、金融 SSL 持续训的受控实验。这些在推文里基本不存在。

**落到 σ0：** SSL-CL 的正确实验不是换对比损失，而是

- 流式自监督目标（你已有的 NLL）
- 可塑性 = 新时段拟合速度 vs 新初始化
- replay 比例与漂移信号调 $\eta$把它当成“无标签所以免疫老化”没有推文支持。

---

## (2) Continual learning × RL

这是 X 上唯一形成完整口语理论的一条。

**问题被说成三角形：** 可塑性（快速吃新任务）、稳定性（记住旧的）、可扩展性（内存/算力不爆）。经典 RL/RLHF 被描述成单任务、忘性大、换环境就重开。[⁠@rohanpaul_ai](https://x.com/rohanpaul_ai/status/1949325606778495432)

**可塑性丧失的机制，X 上比 SSL/TTT 都具体：**

- Sutton / Dohare：持续反传，按效用重置单元（Nature 2024）。
- ReDo / dormant：深度 RL 训练中休眠单元比例上升。
- 2025 反驳：激活式 dormant 在 LayerNorm/非 ReLU 下会误诊，应该看梯度还能不能学。
- ICLR 2026（Apple/ETH）：几何视角——梯度被困在不变流形里。两种阱：冻结单元（饱和、梯度没了）、克隆单元（冗余、同向运动）。一旦进去，SGD 自己出不来；学当前任务的压力本身会把网络推进这些阱。作者问：SGD + 交叉熵是不是根本不该作为 CL 框架。⁠@MFarajtabar
- 机器人 CIDER：每新任务冻旧策略当教师，边真机 RL 边蒸馏，梯度路由把“学新”和“保旧”拆开；6 个操作任务，新技能 10–20 分钟，旧技能仍在。这是 X 上少有的物理世界稳定性/可塑性数字。⁠@heetezition

**2026 的民间理论把大规模 post-training RL 直接叫 “bitter lesson continual learning”。** 部署时按难度校准一直 RL，理论上能当 CL，只是算力贵到现在看不到。另一边：遗忘仍被说成微调/RL 的最大障碍。两条并存——RL 既被当成 CL 的希望，也被当成最会把网络训死的设定。[⁠@UnderwaterBepis](https://x.com/UnderwaterBepis/status/2092453198087025081)

**对 σ0 不要误搬：** LOB 基础模型的主循环是 SSL，不是 RL。该搬的是 RL 文献的**诊断和药**，不是 PPO：dormant/梯度健康、谱阱、冻结/克隆单元、reset vs 蒸馏旧“教师”。评估上，CIDER 的“教新技能会不会悄悄忘前五个”应改写成“学 2024-08 会不会毁 2020-03”。

---

## (3) Continual learning × TTT

X 把 TTT 拆成三代，不要混。

**A. 经典 TTT（任务级，推理时更新）。** ARC 上用每题 LoRA + 合成变换，1B 从 6% 拉到 36%。这是“每个测试题一次适应”，不是多年流式服役。[⁠@rohanpaul_ai](https://x.com/rohanpaul_ai/status/1871253431610220611)

**B. TTT 作为层（Yu Sun / Xiaolong Wang，2024）。** 隐藏状态换成一个小网络；每个 token 对这个小网络做一步梯度，把序列压进快权重。作者原话：线性复杂度、比 Mamba/Transformer 在书本尺度上更吃长上下文；视频上密采样对 Transformer 是负担，对 TTT 是数据增强。这是架构，不是训练配方。[⁠@xiaolonw](https://x.com/xiaolonw/status/1810387662060269668)

**C. Titans / MIRAS（Google，NeurIPS 前后被大量转）。** 深 MLP 记忆 + 用梯度“surprise”决定写不写；遗忘 ≈ 权重衰减正则；L2 记忆对离群点过敏，L1/Huber 更稳；宣称 2M needle 上打过 GPT-4（推文转述，未独立验证）。短时靠注意力，长时靠神经记忆。[⁠@_philschmid](https://x.com/_philschmid/status/1997227157748060277)

**D. TTT-E2E（2026 热帖）。** 把长上下文当成持续学习：注意力只看固定窗，更早的上下文用下一 token CE **写进选定权重**。声称长上下文上打过 Mamba 2 和 Gated DeltaNet，128K 比满注意力快 2.7×。有人设想：先满注意力训到 100k，溢出再 TTT 压成常数状态。另有人把它列为“第三条优雅路”：别做 harness，也别为每个任务单独 RL，训练和测试用同一种学习。[⁠@jiqizhixin](https://x.com/jiqizhixin/status/2009187297137446959)

**E. 快权重 sidecar（Delta-Mem 等）。** TTT 学进旁路快权重，一个隐状态就能恢复记忆，不靠把整段上下文留下。这是在回答 wiki #58：context 里学会的东西如何不完全蒸发。

X 上的限制同样清楚：

- 生产前沿模型没有在线改核心权重；用批量微调、LoRA、外置记忆。TTT 仍是研究原型。⁠@grok
- TTT 被说“长上下文很好、速度很糟”；E2E/逐层更新是在修延迟。
- 有人警告：推理时连续学不等于稳定智能，递归本身不够。
- 时限空白依旧：没有推文给出“纯 TTT/状态、不重训，LOB 能撑几周”。

**落到 σ0：** TTT 不是 SSL 或 RL 的替代，是第三尺度。GDN 门控已经是 token 级“忘/写”。最小实验仍是：冻结骨干只跑状态/TTT；对照同算力续训；对照新初始化。若 TTT-E2E 真能在长序列上压过 Gated DeltaNet（推文声称），这就是你骨干对照的优先级，不是再堆一层 replay。

---

## 三条线怎么接到上一页的 H1–H5

- H1（预训练大模型主要忘、不死）SSL 线：Zyphra 的 CPT/SSL 设定已经打脸。RL 线：最支持“会死”。TTT 线：不回答会不会死，只回答能不能把死推迟到慢权重之外。
- H3（门控 ≠ 训练级学习）TTT 作者和评论者基本站在 H3 一边：快权重扛会话，多年 drift 仍要写慢权重或定期 CPT。工业默认也是夜里重训。
- H4（谱 / L2-init 优于大重置）几乎全在 RL 线出现（谱坍缩、几何阱）。SSL/TTT 推文没测。
- 无标注 drift（Wang 第二点）SSL 最贴近“世界不给任务标签”。RL 实验仍大量靠任务切换。TTT 用 surprise/梯度当写入信号，形式上最接近 $s_t$ 驱动学习，但 surprise 是 token 级，不是市场 regime 级。

---

## σ0 可执行的最小矩阵

| 实验 | 用哪条线 | 看什么 |
| --- | --- | --- |
| 流式 SSL 续训 vs 固定模型 vs 新初始化 | (1) | NLL 衰减 + $S_{fresh}/S_{cont}$ |
| 续训时加少量单元重置 / 谱正则 / 冻旧教师蒸馏 | (2) 的药，SSL 的目标 | 高波动月会不会既学不进又毁旧月 |
| 冻结骨干 + GDN 状态 / TTT 层 / sidecar | (3) | 1 天 / 1 周 / 1 月持出；失败时间就是 H3 的数字 |
| surprise 或验证 NLL 当 $s_t$ 调写入强度 | (3) 的门 + (1) 的目标 | 突变月是否优于固定 $\eta$ |

X 上三条线的可信度排序：**RL 机制 > TTT 架构叙事 > SSL-CL 实证**。SSL 最贴你的目标函数，但推文几乎不帮你选超参；RL 帮你选诊断；TTT 帮你选“状态适应能替代重训到哪一步”。

89 sources

---

## User

(1) continual learing with rl, post Training
(3) continual learing with rl, distilation

---

## Grok

X 上这两条已经合成一套口语配方：**后训练 RL 负责探索与抬上限，蒸馏负责把这次学到的东西写进学生、同时挡住旧技能被冲掉。** 生产里被叫成 continual learning 的，多半是「夜间/日更 checkpoint 的 RL + 蒸馏」，不是在线改推理权重。

## 先分清三件被混用的东西

|  | 后训练 RL | 蒸馏（尤其 on-policy） | 真·在线持续学习 |
| --- | --- | --- | --- |
| 何时发生 | 预训练之后，SFT/偏好/RLVR 循环 | 学生 rollout，教师给密监督 | 推理时改核心权重 |
| X 共识 | 当前实验室主循环 | 把 RL 变便宜、把旧策略冻成教师 | 基本没在产线发生 |
| 对 CL 的角色 | 学新（可塑性） | 保旧 + 把新行为内化（稳定性） | 被 Ox Alpha 一类日更表现带起谣言 |

有人把「模型每天变好」拆成三种可能：静默换后训练 checkpoint、用线上失败做离线修补、真在线学习。多数知情者押前两种。[⁠@Hesamation](https://x.com/Hesamation/status/2092157700608053751)

---

## (1) Continual learning × RL post-training

### X 上的主叙事

后训练把基座变成策略，课表被总结成六步：imitate → compare → explore → verify → transfer → anticipate。在线 RL（REINFORCE/PPO/GRPO）吃序列级奖励；RLVR 用 checker 换掉奖励模型，被认为比 RLHF 更好 scale。瓶颈从「例子」换成「环境」。On-policy distillation 被写进同一张地图：用 7–10× 更少步数复制一个 RL 策略。[⁠@cv_usk](https://x.com/cv_usk/status/2091308807515459871)

另一条更硬的实验室结论来自 Gemini RL / GEM 库的顺序任务模拟（Zichen Liu）：

- 泛化：RL 泛化，SFT 记忆
- 稳定：on-policy 采样让 RL 忘得更少
- 几何：不同任务走不同低曲率方向（“off-principals”）
- 结构：RL 只拧小块子网络，天然有点参数隔离

作者强调：这是在 **LLM 预训练过的模型**上做持续 RL，比经典深度 RL 稳；但要「根本解决」CL，仍觉得需要超越当前网络和反传。这直接修正上一页的 H1：不是「预训练就不死」，而是「预训练让后训练 RL 比从头 RL 更适合作顺序任务」。[⁠@zzlccc](https://x.com/zzlccc/status/2012211416917287019)

民间说法更粗：scale post-training RL = bitter-lesson 版 continual learning，只是算力贵到现在还看不见那条 S 曲线。另一些人直接把「系统变好」等同于 RL，不叫 CL。[⁠@UnderwaterBepis](https://x.com/UnderwaterBepis/status/2092452741176332666)

### 后训练 RL 在 CL 里会炸的点（推文点名）

- 弱长程底座：噪声轨迹误差累积，稀疏奖励归因错，冲突教师触发遗忘。后训练本身补不干净。
- 长程规划论文的处方：先把 world-model / 长轨迹打好；奖励太稀就上 OPD；不要合并规划策略不相容的教师。OPD 在更长、更吵的设定里被说成优于结局奖励 GRPO，因为教师信号贯穿整条轨迹。⁠@rohanpaul_ai
- 可塑性几何阱（冻单元、克隆单元）主要还是一般持续训练的故事，后训练推文较少单独测「RLVR 跑半年会不会学不动」。
- 可验证性决定能不能持续 RL：Verifier’s law——可验证才好训。生活若被看成一条在线持续 RL 轨迹，人是被许多可验证奖励推着走。对 LOB 的刺：微观结构预测很少有干净 binary checker，RLVR 那套不能原样搬。⁠@zzlccc

### 对 σ0 / LOB 的翻译

后训练 RL 不是 σ0 的主循环（主循环仍是 SSL NLL）。能搬的是两句话：

1. 若你做「服役后适应」，on-policy 更新比离线 SFT 更不容易把旧时段冲掉（GEM 的主张，未在 LOB 复现）。
2. 不要把多个不相容的「教师信号」（不同 regime 的标签、不同微观结构启发式）直接混进同一次 RL；冲突教师 = 遗忘。

---

## (3) Continual learning × RL distillation

X 上这是 2025–2026 最具体的「可落地 CL 工具」。

### 机制（反复被讲成同一驾驶学校比喻）

- 监督蒸馏 / SFT on traces：看老师开车，模仿动作。学生测试时一旦偏离老师轨迹就崩（exposure bias）。
- On-policy distillation（OPD）：学生自己开，老师对学生实际走到的状态给全词表 logprob，逐步纠正。Lilian Weng：教师当过程奖励模型，密奖励，又避免 SFT 式 OOD shock。Thinking Machines：RL 的纠错相关性 + SFT 的奖励密度，数学/内部助手上更便宜地打过其它后训练。已公开用在 Gemma 2/3、Qwen3-Thinking。⁠@lilianweng

Rishabh Agarwal 把公式说死：学生吐 token → 教师给整表 logprob → 学生去匹配。这就是 DAgger 在 LLM 里的翻版。[⁠@agarwl_](https://x.com/agarwl_/status/1982880080482140372)

### 蒸馏怎么被当成持续学习

三条实现，不要混：

**A. 冻旧策略当教师（机器人 CL 正统）**

CIDER：每个新技能前冻累积策略；真机 RL 学新，同时向旧教师蒸馏；梯度路由拆开「学新 / 保旧」。6 个操作技能，每个 10–20 分钟，全部结束后平均成功率被报成 100%。caveat：短任务、单平台、有人干预。但这是 X 上最接近「教新的不毁旧的」的带数字结果。[⁠@ludwim_i](https://x.com/ludwim_i/status/2092315141555499321)

DAgger + 蒸馏的工业实验（Axis）：新校正数据 BC + 对冻旧模型的 matching loss；新旧分布成功率都 +20pp（多种子未齐）。冻结 normalizer / 混合旧数据能减轻某个已退化任务的掉点。这就是 replay 的蒸馏版。[⁠@axisroboticsin_](https://x.com/axisroboticsin_/status/2089712203884834825)

**B. On-policy / 自蒸馏当会话级 CL**

Hackathon 和评论把 OPD/OPSD、合成 RL 环境、KV compaction 并列成「模型层 CL」。有人写：agent 不该犯第二次同样的错，现在有 reflection + OPSD，可在文本和权重两边写。另有人更满：SFT、RL、OPD、prompt 优化已经够用，CL 可以马上做。[⁠@michalwols](https://x.com/michalwols/status/2074172037329723815)

反向证据同样在 X 上：SDPO（on-policy self-distillation）在教师信号稳、域内特化时快，但 **OOD 失败、遗忘/坍缩上升、放大伪影**。OPSD 被说成擅长「常规任务上的持续学」，弱于探索和开新解；那块仍是 GRPO 更好。[⁠@arxivsanitybot](https://x.com/arxivsanitybot/status/2073462689012072491)

**C. 蒸馏会让学生变差（理论警告）**

Paria Rashidinejad：更好教师 + 精确自然梯度，一大类 $f$-散度自蒸馏（含 reverse KL、JS）**仍可把学生弄差**。原因是只做 token 级局部归因，不管早期决策对未来状态的影响。他们的 DistIL（分布版 DAgger）把未来分歧折回早期 token，给单调改进和 regret 保证。实证上打过 RLVR 和自蒸馏基线。这是对「有教师就正向迁移」的直接打脸。[⁠@paria_rd](https://x.com/paria_rd/status/2064044148802605281)

长程规划那篇补刀：**规划策略不相容的多教师合并会忘。** 多教师不是免费的稳定性。

其它变体：On-Policy Delta Distillation 只蒸「推理微调教师相对基座的差」，避免把教师的通用文风和旧预训练习惯一起灌进去——普通 OPD 在强模型上有时会伤。Latent On-Policy Self-Distillation 把过去成功尝试压成 latent token 给教师副本，部署只留学生、不存经验。

快/慢网络也被说成蒸馏：小快网络高 weight decay，再蒸进慢网络（Nested Learning 口语版）。这是架构级蒸馏，不是后训练配方。[⁠@scaling01](https://x.com/scaling01/status/2070307798987022641)

### 蒸馏在 CL 里的分工（X 共识）

| 目的 | 用哪种蒸馏 | 失败模式 |
| --- | --- | --- |
| 保旧技能 | 冻旧政策 / 旧 checkpoint 当教师 + matching loss | 教师过时；多教师冲突 |
| 把昂贵 RL 压便宜 | OPD：教师打学生自己的轨迹 | 域内好、OOD 和探索差；熵塌 |
| 会话级「不再犯同样错」 | OPSD / latent 自蒸馏 | 自蒸馏放大伪影；上限不一定涨 |
| 只迁「新能力差」 | Delta distillation | 推文证据薄 |

一句话：后训练 RL 解决「有没有新行为」；蒸馏解决「新行为进学生时旧行为死不死、以及要花多少 rollout」。

---

## 两条线合成一张后训练 CL 栈

X 上能拼出来的顺序（不是论文协议，是推文配方）：

1. 预训练提供稳定底座（Liu：预训练让持续 RL 比经典 RL 可行）。
2. 新任务用 RLVR/GRPO 探索（稀奖励、要开新解时）。
3. 用 OPD 把这个 RL 教师压进学生（密信号、少步数、走学生自己的状态）。
4. 同时保持一个 冻旧教师，matching loss 挡住遗忘（CIDER / DAgger 校正）。
5. 不要把规划哲学不同的教师 merge。
6. 产线表现成「日更」，其实是 checkpoint 滚动，不是推理时写权重。

对可塑性：蒸馏偏稳定一侧。它不重置死单元，也不解决谱阱。它让你在**不洗骨干**的前提下继续加技能。探索和「网络还能不能学」仍要靠 RL 或重置/谱工具。

---

## 对 σ0 的最小可做实验（如果只借这两条）

σ0 没有语言 RLVR。可借用的是蒸馏骨架，不是 GRPO。

- 旧时段教师：每个日历/事件切分结束，冻一个教师头或整模。下一时段 SSL 续训时加对旧教师的 logits/表征 matching（CIDER 的保旧项）。看旧月 NLL 是否少崩。
- On-policy 味：匹配损失打在当前模型自己采样的订单流续写上，不要只打历史教师轨迹（OPD vs SFT traces）。
- 不要多教师乱炖：2020-03 教师和 2024-08 教师若微观结构哲学冲突，分头匹配或选一个，不要平均。
- 后训练 RL 先别上：没有可验证 checker 以前，RLVR 叙事帮不了 LOB；GEM「RL 比 SFT 更适合作 CL」只能当假说，用「on-policy 续训 vs 离线重放 SFT」做弱对照。

---

## 推文没核到的

- 后训练 RL 跑数月后的可塑性曲线（拟合速度 vs 新初始化）几乎没有。
- OPD 7–10× 步数、CIDER 100% 成功率、Ox Alpha 日更，均为作者/观察者陈述。
- LOB 或任何金融后训练 RL + 蒸馏的一手实验室推文：无。
- Thinking Machines 博客的数字细节：推文只给方向（更便宜、数学/助手上更好）。

50 sources

---

## User

(1) continual learing with rl, post Training or distiliation pikc up one

---

## Grok

选 **蒸馏**，不选后训练 RL。

对 σ0 这不是口味问题：后训练 RL 在 X 上更热闹，但你缺它成立的前提；蒸馏能直接变成持续预训练里的一项 loss。

## 为什么砍掉后训练 RL

X 上后训练 RL 当 CL 用，默认三件套：可验证奖励（RLVR）、环境、on-policy rollout。Verifier’s law 说得很难听——不好验证的任务就不好用 RL 持续训。LOB 下一事件 / NLL 没有 binary checker，也没有「对了就过」的 sandbox。把 GRPO 硬接上去，你得到的是稀疏、滞后、容易把冲突 regime 当成冲突教师的信号，正是那篇长程规划推文里会触发遗忘的组合。

GEM 那条「LLM 上 RL 比 SFT 更适合作顺序任务」成立的条件是：已经预训练好的语言策略 + 任务有奖励。σ0 的主循环仍是自监督 NLL，不是策略。先上 RL 是换问题，不是答问题。

产线里被叫做 continual learning 的日更，知情者拆成「换后训练 checkpoint」，不是推理时写权重。那条线帮不了你写 Experiments。

## 为什么蒸馏是可做的那一个

蒸馏在 X 上已经被说成后训练 CL 的稳定器，而且不依赖 checker：

- 冻旧策略当教师，新技能用 RL/BC 学，同时 matching 旧教师（CIDER：6 个真机技能，10–20 分钟一个，旧的还在）。
- 学生走自己的轨迹，教师打密 logprob（OPD / DAgger）：纠的是学生实际会偏到的状态，不是老师的示范分布。
- 校正数据 + 旧分布蒸馏，新旧两边成功率一起涨；冻 normalizer 能减轻已退化任务掉点。

搬到 σ0 时，教师不是「更会推理的 LLM」，而是 **上一个时间切分冻住的模型**。目标不是把 RL 变便宜，是让流式 SSL 续训时旧月份不崩。

这正好打你原来的决策接口：持续预训练配方（replay 的替代/补充）、评估协议（旧切分 NLL = forgetting）、不必先改 GDN。

## 只做这一个：σ0 实验草稿

**设置**

切分按日历或事件（事件日期另核）。每个切分结束存教师 $T_k$。在切分 $k+1$ 上继续训学生 $S$。

**损失**

$$
\mathcal{L}=\mathcal{L}_{\text{NLL}}(S;D_{k+1})+\lambda\,\mathcal{L}_{\text{match}}(S,T_k;D_{\text{on-policy}})
$$

- $\mathcal{L}_{\text{NLL}}$：你现有的 SSL。
- $\mathcal{L}_{\text{match}}$：对学生自己采样的订单流续写做 logits 或表征匹配，不要只匹配历史教师轨迹（这是 OPD 相对 SFT-on-traces 的那一点）。
- 一个切分只用一个教师。2020-03 和 2024-08 不要平均进同一个匹配项——X 上明确写过，不相容教师一合并就忘。

**对照**

1. 只 NLL 续训
2. NLL + 旧数据 replay
3. NLL + 冻教师匹配（本实验）
4. 同算力新初始化（可塑性探针：$S_{\text{fresh}}/S_{\text{cont}}$）

**必报表**

旧切分 NLL（遗忘）、新切分 NLL（适应）、拟合速度相对新初始化（可塑性）、$\lambda$ 扫描。CIDER 要回答的那句改成：学新月会不会悄悄把前五个月弄死。

**不要做的**

自蒸馏当唯一教师（SDPO 在 X 上被报 OOD 失败、遗忘上升、放大伪影）；多教师乱炖；还没匹配基线就上 GRPO。

## 和上一页假设的关系

- H1：蒸馏不回答「网络会不会死」，只回答「死之前旧能力能不能保住」。可塑性探针仍要对照新初始化。
- H2：慢漂移用小 $\lambda$ 持续匹配最近教师；突变切分换新教师、短窗口加大 $\lambda$ 或短时提高 $\eta$，不要把突变教师和慢教师并成一个。
- H3：蒸馏写的是慢权重，不是 GDN 状态。状态适应仍是另一组实验。
- H4：蒸馏偏稳定侧，不替代谱正则/Muon。两者正交，不要互相冒充。

后训练 RL 可以留在文献综述里当「语言/智能体社区把 CL 理解成什么」。σ0 这一页的可运行贡献选蒸馏。

---

## User

(1) continual learing with rl, distiliation, ??? (解决?) 轨迹误差累积
(2) continual learing with rl, distiliation, ??? (解决?) 隔日昨天的数据进入模型
(3) 在线持续学习 推理时改核心权重 ????
(4) 新任务用 RLVR/GRPO 探索（稀奖励、要开新解时）。????这是相同大小模型 还是很小的模型?

- 用 OPD 把这个 RL 教师压进学生（密信号、少步数、走学生自己的状态）

---

## Grok

四个问题不是同一层。先给判定，再拆开。

| # | 你问的病 | 蒸馏 / OPD 解决了吗 | 真正在解决的东西 |
| --- | --- | --- | --- |
| (1) | 轨迹误差累积 | **部分，只修学生自己这条路上的偏航** | 密过程监督，不是把长程底座变好 |
| (2) | 隔日把昨天的数据写进模型 | **不是 OPD 的职责**；那是夜间 CPT / replay / 冻教师匹配 | 离线把昨天冻成教师，今天匹配它 |
| (3) | 推理时改核心权重 | **生产上基本没有** | TTT/Titans 改快权重或一小块；产线是换 checkpoint |
| (4) | RLVR/GRPO 探索用一样大还是更小的模型 | **两种都有，不是一件事** | 探索常用同尺寸或更大专家；OPD 再压进同尺寸或更小的学生 |

---

## (1) 轨迹误差累积 — 蒸馏解决了什么

病是：多步里前一步小错 → 走进老师从没示范过的状态 → 后面全崩。SFT 蒸馏最容易中这个招（exposure bias）。结局奖励 GRPO 也中：整条轨迹只在最后打一个分，中间哪步把误差放大了说不清。

OPD 修的是**第二条**：学生自己 rollout，教师在学生实际走到的每个 token 上给全词表反馈，学生在「自己会偏到的状态」上学会往回收。驾驶学校比喻：不是看老师开，是学生开、老师在副驾喊。这能压「一偏就不可逆」。[⁠@NicolasGertler](https://x.com/NicolasGertler/status/2091786250307117114)

它**不修**的：

- 预训练长程世界模型本来就弱。推文写得很死：多少后训练也补不干净弱底座；噪声轨迹会一直累积。OPD 只是比「只看结局的 GRPO」更抗吵、更密。
- 教师自己也会错。教师在长上下文上和 verifier 的分歧会随长度涨。
- 自蒸馏（SDPO）在 OOD 上会忘、会塌。

σ0 翻译：订单流预测里的「误差累积」更接近表征/NLL 在长窗口上漂，不是 agent 多回合规划。OPD 思想能用（匹配打在**当前模型自己续写的序列**上），但不要指望蒸馏单独治好长 horizon LOB。

---

## (2) 隔日，昨天的数据进模型 — 不是同一把刀

「昨天的市场进今天的权重」是 **数据何时写入慢权重**，和时间表有关，和 OPD 的 on-policy 几何无关。

X 上实际在发生的三档：

1. 夜间/日更后训练：用昨天日志做 SFT/RL/蒸馏，早上换 checkpoint。被叫成 CL，其实是批作业。Ox Alpha「每天变好」多数人押这个。
2. 冻昨天当教师：今天的学生继续训，loss 里加对 $T_{\text{昨天}}$ 的匹配。昨天不是「再吃一遍原始 tick」，是「别把昨天会的行为冲掉」。CIDER / 校正+旧分布蒸馏是这一档。
3. 真在线：每个 tick/每个请求反传核心权重。产线基本不做。

所以：

- 要昨天的 数据分布 进模型 → 把昨天样本放进今天的 NLL/replay（普通持续预训练）。
- 要昨天的 能力 别丢 → 冻昨天 checkpoint 做教师匹配。
- OPD 不负责「隔日」；它负责「更新时走谁的状态」。你可以每天跑一次 OPD，那只是把 (2) 的批作业换成 OPD 损失。

σ0 该做的是 (2) 的 1+2：每日/每切分流式 NLL + 冻教师匹配。不要把 OPD 宣传成日更管道本身。

---

## (3) 在线持续学习、推理时改核心权重？

四层，别混：

| 层 | 推理时改什么 | 例子 | 算不算「核心权重」 |
| --- | --- | --- | --- |
| A | 不改权重，只改上下文 / RAG / 记忆 | 几乎所有线上 LLM | 否 |
| B | 改快权重、隐藏状态、一小块旁路 | TTT layer、Titans surprise 写入、GDN 门控、Delta-Mem sidecar | 否（或只改很小一块） |
| C | 把当前上下文压进选定权重矩阵（仍是这次前向里的局部更新） | TTT-E2E | 灰区：写的是选中的层，不是全模型日更 |
| D | 服务过程中 SGD 更新全部骨干，下一用户看到新权重 | 研究原型；X 上说前沿产线没有 | 是 |

X 共识：没有前沿模型在服务时连续改核心权重。用的是批量微调、LoRA、外置记忆。有人挖苦：工业界夜里训一发，白天管这叫 CL。

TTT-E2E 是目前最接近「推理时学习」的架构实验：固定注意力窗，更早上下文用 CE 写进选定矩阵。这是为了长上下文压缩，不是为了把昨天全市场写进明早的全局 θ。

σ0：GDN 门控 = B。若做 TTT 对照 = B/C。日更服役 = 换 checkpoint（D 的离线版）。不要在论文里把 B 写成 D。

---

## (4) RLVR/GRPO 探索：同样大，还是更小？然后 OPD 压进学生

两段式，尺寸策略不一样。

**探索段（RLVR/GRPO）**

跑的是「当前要抬上限的那张策略网」，通常是：

- 同尺寸 学生自己 RL（最常见的 post-training：同一个 8B/70B 接着 GRPO）；或
- 同尺寸或更大的领域专家 分开 RL（数学一个、工具一个、agent 一个）。

几乎没有「先用很小的模型 GRPO 探索、再指望它当教师」这种主流配方。小模型纯 RL 被说成比好蒸馏更不 sample-efficient。探索要的是还能试出新解；太小的网探索不动。

**蒸馏段（OPD）**

学生 **可以同尺寸，也可以更小**。X 上两种都在用：

- 同尺寸（self / 快照）：学生开训时冻一份自己当「Self 教师」，防止其它教师把人拽进陌生区导致灾难性漂。MOPD 里这是标准配件。
- 同尺寸、多专家 → 一个学生：MiMo、GLM-5、Nemotron-Cascade、DeepSeek-V4 一类配方——多个领域 RL 专家当教师，OPD 合成一个学生，用来修「数学涨了、写作掉了」的跷跷板。教师往往和最终学生同级或来自同一条管线的不同 checkpoint。⁠@cwolferesearch
- 大教师 → 小学生：经典压缩。OPD 相对 SFT 蒸馏的好处是小学生走自己的轨迹，不在老师的示范分布上装懂。Thinking Machines / Gemma / Qwen 公开叙事是这一路。也有跨家族、不同 tokenizer 的 OPD。

口令：

- GRPO：谁要探索谁就上，一般是同尺寸或专家，不是小玩具网。
- OPD：把已经探索到的行为，沿学生自己的状态，灌进目标学生；目标学生可同可小。
- 「7–10× 更少步数复制 RL 策略」指的是复制成本，不是说教师必须更大。

OPSD 被点名：常规任务持续学还行，**开新解仍是 GRPO 强**。所以栈是「稀奖励先 GRPO，再 OPD 内化」，不是 OPD 替代探索。

σ0：你没有 checker，GRPO 段先不要做。若只做蒸馏，教师 = 上一切分同尺寸冻模，学生 = 当前同尺寸 σ0。这是 CIDER/MOPD-Self 那一格，不是 70B→3B 压缩。

---

## 四句话收束

1. OPD 治的是「学生自己轨迹上的逐步偏航」，不治「底座长程不行」。
2. 「昨天进今天」靠日更数据和/或冻昨天教师，不是 OPD 这个算法名。
3. 推理时改核心权重在产线近乎不存在；TTT/GDN 是快权重或局部写。
4. GRPO 探索用同尺寸或专家；OPD 再灌进同尺寸或更小的学生——两段，不是一个模型角色。

24 sources

---

## User

轨迹误差累积 self forcing?

---

## Grok

对。**Self-Forcing 就是冲着轨迹误差累积去的**，视频自回归里的正式名字；和 OPD 是同一类病、同一类药，不是 RL 后训练那套。

## 病：teacher forcing 缺口

训练时每一步喂 **真值前文**（GT frames / GT tokens）。

推理时每一步喂 **自己刚生成的前文**。

一步小错进入条件，下一步在错的条件上再错，连续潜空间里会漂成过曝、运动循环、语义丢掉。论文里就叫 exposure bias / train–test mismatch / error accumulation。[⁠Ar5iv.labs.arxiv](https://ar5iv.labs.arxiv.org/html/2510.02283)

自回归视频扩散比 LLM 更狠：时间上自回归，空间上还要一步步去噪，两条链都会积误差。

## 药：训练时按推理那样自己滚

Self-Forcing（Huang et al.，Xun Huang 2025-06 那条）训练时 **带着 KV cache 做自回归 rollout**：生成第 $t$ 帧的条件是自己生成的 $<t$，不是 GT。再用整段生成视频上的分布匹配（DMD 等）对齐双向教师。强迫模型在训练里见到、并学会修自己的错。[⁠@xxunhuang](https://x.com/xxunhuang/status/1932107954574275059)

作者自己划边界：训练长度内（约 5 秒）看不到明显积误差；**超出训练 horizon 用滑窗外推，质量照样掉。** Self-Forcing++ 的补丁是：故意滚到教师不会的长 horizon（到 100 秒），让学生先把误差滚出来，再把噪声打回这些已退化轨迹，用强教师做分布匹配。

后来还有人说：有些「积误差」其实是教师的 **conditional bias**——双向教师按新事件给的指导，和已经实现的学生历史不相容，DMD 会把人推向「符合 prompt、但接不上自己轨迹」的模式（Delta Forcing）。所以不是所有 drift 都是学生自己积出来的。

## 和 OPD / 你前面那条栈的关系

|  | Self-Forcing | OPD |
| --- | --- | --- |
| 域 | 自回归视频扩散 | LLM / 策略 |
| 学生走谁的状态 | 自己生成的帧 | 自己的 token rollout |
| 教师干什么 | 双向短视频教师做分布匹配 | 对每个学生 token 给 logprob |
| 治的病 | 帧级 exposure bias | 轨迹 exposure bias |
| 不治的 | 超训练 horizon；教师自己偏 | 弱长程底座；OOD 自蒸馏坍塌 |

一句话：**都是 scheduled sampling / DAgger 家族**——别在 GT 条件上装懂，在自己滚出来的脏条件上学习。Self-Forcing 是视频版；OPD 是语言/策略版。

经典亲戚：scheduled sampling、professor forcing、DAgger。不是 GRPO，也不需要 verifiable reward；监督可以来自教师分布或重建，不一定来自对错 checker。

## 对 σ0 意味着什么

如果你只做 **条件于真实历史订单流的一步 NLL**（下一事件 / 下一 bin），推理和训练条件一致，**没有这条积误差**。Teacher forcing 在这里是对的。

只有在你 **用模型自己往前滚** 时才会出现同一疾病：

- 多步生成未来 book
- 用自己的预测当下一步条件
- 世界模型 / 开环仿真
- 长窗口自回归 rollout 评分布

这时才值得上 Self-Forcing：训练时对 **自己采样的续写** 算 NLL 或匹配冻教师，而不是只在 GT 历史上算。这和上一页写的 $\mathcal{L}_{\text{match}}(S,T;D_{\text{on-policy}})$ 是同一设计，只是视频圈叫 Self-Forcing。

不要指望它解决：隔日数据进权重、可塑性丧失、推理时改核心权重。那些是别的层。Self-Forcing 只闭合「训练喂 GT、测试喂自己」这一道缝。超出你训练时滚过的步数，误差照样会回来——视频论文已经写明。

21 sources

---

## User

用整段生成视频上的分布匹配（DMD 等）对齐双向教师。强迫模型在训练里见到、并学会修自己的错。 没看懂监督信号从哪里来?

---

## Grok

监督**不是**「下一帧的 GT 像素 / 交叉熵」。Self-Forcing 管的是**条件从哪来**；DMD 管的是**对整段假视频打分的场从哪来**。两件事叠在一起。

## 先拆开

|  | Self-Forcing | DMD |
| --- | --- | --- |
| 解决什么 | 训练时也用自己生成的前文，别再用 GT 前文 | 整段生成视频的分布要像「真视频 / 教师」 |
| 学生看见什么 | $\hat{x}^{<i}$（自己滚出来的） | 自己滚完的整段 $\hat{x}^{1:N}$ |
| 有没有「正确答案帧」 | 没有，当上下文用 | 没有逐帧 label |

所以你找不到 $y_t =$ 第 $t$ 帧真值这种监督。信号是分布级的。

## DMD 的监督从哪来（三条网）

现场有三个东西：

1. 学生 $G_\theta$：因果自回归，自己一帧帧滚出一段视频 $\hat{x}$。
2. 冻住的 real / 教师扩散 $s_{\text{real}}$：在真视频（或双向教师）上预训练好的分数网络。问的是：这块带噪潜变量，若来自真视频分布，去噪该往哪走。
3. 可训练的 fake critic $s_{\text{fake}}$：专门拟合学生正在生成的分布。问的是：这块带噪潜变量，若来自学生分布，去噪该往哪走。

一步训练长这样：

```text
学生自己滚  →  得到干净视频 x̂
给 x̂ 加噪   →  x_t = 前向扩散(x̂, t)
教师看 x_t  →  「真分布会怎么去噪」  (pred_real)
critic 看 x_t → 「假分布会怎么去噪」  (pred_fake)
梯度 ≈ (pred_fake − pred_real)  反传到 x̂，再反传到 G_θ
```

代码里就是这一行：grad = pred_fake_image - pred_real_image（DMD 原文 eq.7）。[⁠GitHub](https://github.com/guandeh17/Self-Forcing/blob/main/model/dmd.py)

数学上它在最小化带噪后的反向 KL：

$$
\mathbb{E}_t\big[D_{\mathrm{KL}}(p_{\theta,t}\,\|\,p_{\mathrm{data},t})\big]
$$

梯度是分数差：

$$
\nabla_\theta \propto -\mathbb{E}\big[(s_{\mathrm{real}}(x_t,t)-s_{\mathrm{fake}}(x_t,t))\,\partial\hat{x}/\partial\theta\big]
$$

$s_{\mathrm{real}}$ 来自预训练扩散（真数据流形）；$s_{\mathrm{fake}}$ 来自一边训一边追学生的 critic。

**人话：**

教师分数场 = 「真视频在这个噪声水平上该长什么样」。

Critic = 「你现在实际在生成的那坨长什么样」。

两者一减 = 「你这段假视频该往真流形搬哪里」。

学生整段视频被这块向量场推一把，于是学会：**在自己已经滚歪的前文上，下一段还要能回到真视频长得像的地方**——这就是「见到自己的错并修」。

## 真视频到底用没用

用了，但**不是 teacher forcing 那种用**。

- 教师当初在真视频上预训练 → $s_{\mathrm{real}}$ 里已经有真数据。
- Critic 要能区分真/假，训练时会看到真视频的加噪版和假视频的加噪版。
- 学生预测下一帧时，前文不是真帧，loss 也不是 $\|\hat{x}^i - x^i_{\mathrm{GT}}\|^2$。

没有「第 17 帧必须等于训练集第 17 帧」。只有「你滚出来的整段，加噪之后，教师觉得不像真的，就把你往像的方向推」。

也可以换成 SiD（对齐两个 score）或 GAN（判别器打真假）。都是整段分布对齐，都不是逐 token 标签。

## 和 OPD 对一下（同一形状、不同空间）

|  | OPD | Self-Forcing + DMD |
| --- | --- | --- |
| 学生走 | 自己的 token 轨迹 | 自己的帧轨迹 |
| 教师给的监督 | 每个位置的 logprob 表 | 带噪潜变量上的去噪方向 $s_{\mathrm{real}}$ |
| 还要一个「假分布」模型吗 | 通常不用（教师直接打学生状态） | 要，$s_{\mathrm{fake}}$ |
| 有没有 GT 下一符号 | 没有 | 没有 |

都是：**监督打在学生自己造出来的东西上，教师提供「更好分布」的方向。**

## 常见误读

- 「双向教师在看未来帧当 GT」——DMD 里教师是在看 学生视频加噪后的 $x_t$，输出的是分数/ $x_0$ 预测，不是把未来真帧抄给学生当输入。CausVid 那条线确实用过双向教师当 $s_{\mathrm{real}}$；Self-Forcing 论文把匹配目标写成 $p_{\mathrm{data}}$，实现上 $s_{\mathrm{real}}$ 仍是预训练扩散。
- 「所以没有监督」——有，是向量场，不是 label。
- 「这样就不会积误差了」——只在你训练时滚过的长度里成立；horizon 以外仍会掉。

σ0 若要类比：不要找「未来真 tick 当 label」；要对 **模型自己续写的一段订单流** 打一个「像不像真订单流」的分布分数（冻教师 NLL / 表征匹配 / 判别器）。那才是 DMD 这一格，不是下一 bin 的交叉熵。

5 sources

---

## User

；超出训练 horizon 用滑窗外推，质量照样掉。 Self-Forcing++ 的补丁是：故意滚到教师不会的长 horizon（到 100 秒），让学生先把误差滚出来，再把噪声打回这些已退化轨迹，用强教师做分布匹配。 所以这里是怎么做到的?

---

## Grok

教师从来不会生成、也不会看见 100 秒。它仍然只会看大约 5 秒的短窗。++ 的把戏是：**短教师去改学生自己滚出来的长视频里切出来的短片段**，而这些短片段里已经带上了长程积出来的脏。

## 原先 Self-Forcing 卡在哪

教师（双向短视频扩散）的可靠 horizon ≈ 5 秒。

SF 训练也只滚 ≈ 5 秒，DMD 对整段 5 秒假视频对齐。

推理若滑窗外推到 30–100 秒：条件已经离开训练分布，过曝、循环、语义丢，教师帮不上——你没法把 100 秒整段塞进只会 5 秒的教师。

## ++ 就三步（项目页原话）

1. 学生用 rolling KV 自己滚到远超教师的长度（实验到 100 秒，加算力到 4 分 15 秒）。
2. 从这段长视频里 均匀随机切一段教师长度的连续窗，把窗按扩散日程加回去噪声（backward noise），再对这个短窗做 DMD。
3. 可选：再用 GRPO + 你喜欢的奖励（光流等）后训。

推理代码和 Self-Forcing 一样，只是 KV 窗更长。

## 监督具体怎么打到「已经脏了的轨迹」上

一次迭代：

```text
学生自回归滚出 100s 干净潜变量
        x̂¹, x̂², …, x̂ᴺ     N ≫ T

均匀抽一个起点 s
        window = x̂[s : s+T]     T ≈ 教师 5s

按扩散日程加噪
        window_t = (1-t)*window + t*ε

教师 s_real 看 window_t   → 「真短视频会往哪去噪」
critic  s_fake 看 window_t → 「学生这段会往哪去噪」
DMD 梯度 = (fake − real) 只打在这个窗上
再经 rolling KV 漏回学生权重
```

关键点：

- 教师输入永远是短窗 + 噪声，和它预训练时一样，所以不需要长视频数据集、也不需要会长视频的教师。
- 窗的内容不是真 5 秒片，是学生 100 秒里切下来的 5 秒。 如果切到第 80 秒，这 5 秒往往已经过曝、漂了。教师仍按「真短视频流形」去推，等于在说：就算前文已经歪了，你这一段局部还得长得像真视频。
- 加噪（他们叫 backward noise initialization）是为了让这个脏窗和教师/学生去噪器的输入空间一致，不是「再生成一遍」。
- Rolling KV 让第 80 秒的窗仍连着前面的历史，局部纠正能通过缓存影响后面怎么滚。

所以「让学生先把误差滚出来再打噪声、用强教师匹配」不是对 100 秒整段做一次 DMD，而是：

**长 rollout 负责制造带积误差的状态；短窗 DMD 负责在这些脏状态上给恢复力。**

## 和原版 SF 差在哪

|  | Self-Forcing | Self-Forcing++ |
| --- | --- | --- |
| 学生滚多长 | ≈ 教师长度（5s） | ≫ 教师（100s+） |
| DMD 打在哪 | 整段短视频 | 长视频里随机短窗 |
| 训练时见过的「已积误差」状态 | 几乎没有 | 故意制造 |
| 教师看什么 | 短假视频 | 短假视频（但是从长假视频里切的） |
| 要长视频 GT 吗 | 不要 | 也不要 |

作者的 insight：外推掉点，是因为学生从没在「自己已经滚坏的条件」上被纠正过。把训练 horizon 拉过教师，错误会自然出现；再把这些错误状态切成教师能懂的短窗，用原来的 DMD 当恢复信号。

## 做不到的

- 教师没有对「第 1 秒和第 90 秒是否同一场戏」给全局分数，只有局部流形。长程剧情靠学生自己的 KV + 这种局部拉回。
- 切到的窗若已经彻底崩出教师流形，$s_{\text{real}}$ 的方向会变差（Delta Forcing 后来把一部分漂归因为教师条件偏置）。
- 超过位置编码上限照样没办法。
- 这仍不是「昨天数据写进明天权重」，也不是推理时改核心权重。

## 对 σ0 的对应（若你滚多步未来）

教师 = 只在短窗（比如 5–15 分钟真实历史）上可靠的冻模型。

学生 = 自己开环滚很长一段未来 book。

++ = 不要对整段长预测算一个匹配；从长预测里随机切教师能懂的短窗，把短窗加噪或直接匹配冻教师。学生因此学会：滚歪之后如何在局部回到像真订单流的样子。

一步条件 NLL（始终吃真实历史）用不上这一套。只有开环多步生成才会用到「先滚脏、再短窗拉回」。

5 sources

---

## User

DMD - 扩散模型蒸馏新范式 和Discrete tokens如何结合 看tweets

---

## Grok

X 上的共识：**原版 DMD 不能直接打在离散 token 上。** 能接上的路有四条，其中两条是「把 DMD 改成 token 条件分布」，一条是「先把 token 嵌进连续空间再 DMD」，一条是「根本不用 DMD，改用 OPD」。

## 为什么原版 DMD 和离散 token 不合

连续 DMD 的监督是分数差：

$$
\nabla \propto (s_{\text{fake}}(x_t)-s_{\text{real}}(x_t))\,\partial x/\partial\theta
$$

这要求 $x$ 可微、加噪是高斯、教师给的是 $\nabla_x\log p$。离散 token 三条都不成立：

- 采样是 $\mathrm{argmax}/\mathrm{sample}$，$\partial x/\partial\theta$ 断了。
- 前向过程是 mask / 均匀跳，不是 $x_t=(1-t)x+\ t\varepsilon$。
- 熵主要靠逐步 unmask，不是靠一条从噪声到数据的连续轨迹。有人说「楼梯没法蒸馏」：一步生成会把这条序贯熵吃掉。⁠@jacnrwalker

反驳也在同一条线上：DUO（高斯对偶做 consistency）、SDTT（masked diffusion 上渐进蒸馏）、**D-MMD**（离散版分布匹配）说明「不能蒸馏」是错的，只是不能原样搬连续 DMD。[⁠@PatrickPyn35903](https://x.com/PatrickPyn35903/status/2046989681963733079)

---

## X 上实际怎么接

### 1. Di[M]O：DMD 从「整段潜空间」改成「token 条件分布」

作者原话：和 DMD 的根本区别，是不去最小化整个 latent 上的去噪分布散度，而是最小化 **token-level conditional distributions** 的散度。灵感直接写的是 **on-policy distillation**：在学生自己走到的中间态上，对齐师生输出分布。[⁠@yuanzhi_zhu](https://x.com/yuanzhi_zhu/status/1902830933570437280)

做法（他们的线程）：

- 教师是多步 Masked Diffusion（MaskGit / Meissonic）。
- 学生一步出图。
- 对齐发生在学生生成的中间态上，覆盖学生会经过的状态（on-policy）。
- 初始化要把随机输入映射到近乎确定的目标序列，否则多 token 独立假设会散或 mode collapse。
- 辅助网估计损失里不可积的那一项，用标准 MDM loss、目标是学生一步样本。

这是「DMD 精神 + 离散 softmax」：匹配的是每个位置的 $p(\text{token}\mid\text{state})$，不是连续分数场。

### 2. D-MMD：连续 moment matching 搬到离散

Hoogeboom：你以为离散蒸馏根本上不行，其实可以。D-MMD 把连续域的 moment matching 搬过来，少步采样，质量/多样性还在，文本和图像都测了，声称可打过教师，PPL–diversity 帕累托上优于「先连续扩散再蒸馏」。[⁠@emiel_hoogeboom](https://x.com/emiel_hoogeboom/status/2056003879548940492)

冈野原的转述把监督说死了：

- 匹配的是每个噪声态上的 条件期望（从该脏状态恢复出的干净分布的均值）。条件期望对齐了，终态分布就对齐。
- 学生期望算不出：离散采样不可微，MC 方差爆。
- 所以再训一个 auxiliary 追学生期望。学生更新变成 Teacher − Auxiliary（靠近教师、离开自己当前行为）。Auxiliary 吃学生样本，所以是 on-policy。
- 因为 token 不可微，匹配的是 概率向量 / soft probabilities，不是 hard sample。相关性能靠 soft 结构表达，尽管输出仍常假设位置因子化。
- 类比 RL：auxiliary ≈ value，整段是生成问题上的 on-policy。

和连续 DMD 的对应：

| 连续 DMD | D-MMD / Di[M]O |
| --- | --- |
| $s_{\text{real}}(x_t)$ | 教师在该 mask/噪声态上的 token 分布或条件期望 |
| $s_{\text{fake}}$ critic | auxiliary 对学生分布的估计 |
| 梯度 $s_{\text{fake}}-s_{\text{real}}$ | Teacher − Auxiliary |
| $x$ 可微 | 对 softmax / 概率向量反传 |

### 3. 先把离散嵌进连续，再跑真 DMD

GIDD：不直接在 token 上扩散，把离散数据放进连续空间再扩散。然后连续 DMD/consistency 都能用。代价是多一截「连续↔离散」的编解码，蒸馏打在连续态上。[⁠@Clashluke](https://x.com/Clashluke/status/1899048979968450796)

Sticky Jump / 混合扩散同一思路：连续里去噪，再 jump 回离散 token。DMD 打连续段，jump 用通量平衡定落到哪个 token。

视频栈是另一变体：离散文本只当条件，像素/latent 仍连续，DMD 只蒸视觉（Pika：flow matching + DMD，8 步以内）。这不是「DMD 蒸 token」，是「token 当 cond，DMD 蒸连续」。

### 4. 根本不走 DMD：离散上用 OPD / 全词表 KL

语言 AR 里更主流的「分布匹配」就是 OPD：学生自己吐 token，教师给整表 logprob，KL/反向 KL。这已经是离散的分布匹配，不需要分数、不加高斯噪。Thinking Machines / Gemma 那条线。

Will Depue 补了一刀：就算没有 logits、只蒸采样 token，期望上仍是在匹配教师分布，效率大概只差常数倍（2–3×），不是词表那么多倍。那是硬标签蒸馏，不是 DMD，但说明离散上「匹配分布」不必经过连续分数。

---

## 一张图：四种接法

```text
离散 token x ∈ {1..V}
        │
        ├─① 对 pθ(token | state) 做 KL/矩匹配
        │     Di[M]O, D-MMD, OPD
        │     监督 = 教师分布 − 学生/auxiliary 分布
        │
        ├─② embed / 对偶拉到 R^d，再连续 DMD
        │     GIDD, DUO, sticky-jump 的连续段
        │
        ├─③ token 只当条件，DMD 蒸连续 latent
        │     视频/世界模型（Self-Forcing 那套）
        │
        └─④ 不蒸，逐步 unmask（多步 MDM）
              熵留在序贯过程里
```

X 上没有看到「把 Self-Forcing 的 $s_{\text{real}}-s_{\text{fake}}$ 原样打在 one-hot token 上还稳定」的成功帖。成功帖都改了匹配对象：条件分类分布或 soft 概率。

## 和你前面几页怎么接

- Self-Forcing + DMD：连续视频潜变量，教师是分数场。
- 若 σ0 / 语言是离散事件或 token：应对齐的是 ①，不是把 tick 当高斯潜变量硬套 DMD。
- Di[M]O / D-MMD / OPD 共用同一句：在学生自己走到的脏状态上匹配教师的 离散条件分布。Auxiliary = 连续 DMD 里的 fake critic。
- 「楼梯不能一步蒸完」对 masked diffusion 仍部分成立；D-MMD/Di[M]O 是少步，不是免费一步且无损。大规模仍弱于强 AR——冈野原明确写了。

**一句话：** DMD 接到离散 token，不是给 token 加高斯噪声，而是把「分数差」换成「该噪声/mask 态上的教师 token 分布 − 学生 token 分布」，反传走 softmax。X 上这件事的名字叫 Di[M]O 或 D-MMD；语言 AR 里它早就叫 OPD。

40 sources

---

## User

DMD 和 Discrete flow matching [https://arxiv.org/pdf/2505.20147](https://arxiv.org/pdf/2505.20147) FUDOKI: Discrete Flow-based Unified
Understanding and Generation via Kinetic-Optimal Velocitie

---

## Grok

FUDOKI 论文里 **没有 DMD、没有教师–学生蒸馏**。它是 Discrete Flow Matching 的一条生成模型，监督是「脏序列上预测干净 token」的交叉熵。DMD 要接上去，得另做一层少步蒸馏。

## 三套东西先对齐

|  | 连续 Flow Matching | 连续 DMD | Discrete FM（FUDOKI） |
| --- | --- | --- | --- |
| 状态 | $x\in\mathbb{R}^d$ | 同左 | $x_i\in$ 有限词表 |
| 网络学什么 | 速度 $v_\theta(x_t,t)\approx x_1-x_0$ | 少步生成器，用 $s_{\rm real}-s_{\rm fake}$ 推分布 | 在 $x_t$ 上输出每个位置的分类分布 |
| 监督 | MSE 回归条件速度 | 教师分数 − critic 分数 | $\mathrm{CE}(p_\theta(\cdot\mid x_t),\,x_0)$ |
| 采样 | ODE | 1–8 步生成器 | CTMC / Euler：按速率跳到别的 token |
| 论文角色 | 预训练范式 | **蒸馏**少步 | 预训练/适配范式 |

连续里：扩散学 score，再转成速度；FM 直接学速度。两者边际场同一件事，坐标系不同。DMD 是第三件事：已经有一个多步教师，把分布压进少步学生。[⁠@ZhihuFrontier](https://x.com/ZhihuFrontier/status/2092147885333004584)

离散里没有现成的 $\nabla_x\log p$。Discrete FM 把「速度」定义成 **概率通量**：单位时间里，位置 $i$ 从 token $z$ 跳到 token $x$ 的速率 $u_{t,i}(x\mid z)$。

## FUDOKI 实际在干什么

**路径（不是网络）**

不用纯 mask。用 embedding 距离诱导的路径：

$$
p(x_{t,i}\mid x_{0,i})=\mathrm{softmax}\big(\beta_t\, d(x_{0,i},x_{t,i})\big)
$$

$d$ = token embedding 的 L2。$t\to 1$ 时，语义近的 token 概率升高。已 unmask 的 token **还可以改**，这是相对 absorbing mask 的卖点。

**Kinetic-optimal velocity（仍是路径上的闭式通量，不是 $\theta$ 的输出）**

$$
u_{t,i}(x_i\mid z_i)=\big[p(x_{t,i}\mid x_{0,i})\,\dot\beta_t\big(d(z_i,x_i)-d(x_{0,i},x_i)\big)\big]_+
$$

质量只从「离真值更远的 token」流向「离真值更近的 token」，通量尽量小。这是 ICLR 2025 Discrete FM 后续里 Neta Shaul 海报上的闭式解，FUDOKI 拿来当腐蚀过程。[⁠@shaulneta](https://x.com/shaulneta/status/1915736656482877553)

**网络损失（很朴素）**

采 $t$，按上面的路径把干净 $x_0$ 腐蚀成 $x_t$，模型看 $x_t$（双向注意力）预测干净 token：

$$
\mathcal{L}=\mathbb{E}_{t,x_0,x_t}\sum_i \mathrm{CE}\big(p_\theta(\cdot\mid x_t),\,x_{0,i}\big)
$$

没有对 $u$ 做回归。$p_\theta(\cdot\mid x_t)$ 在采样时再被转成跳转速率。从 Janus-1.5B AR 初始化，两阶段改成离散流。理解 / 生成共用离散 token，图像有语义编码和像素 codebook 两套。

**采样**

从先验或腐蚀序列出发，逐步：算每个位置总跳出率 $\lambda_i=\sum_{z\neq x}u_{t,i}(x\mid z)$，以 $1-e^{-h\lambda_i}$ 决定跳不跳，跳到与 $u$ 成比例的新 token。所以是可逆的自我修正，不是 mask 一次定终身。

论文额外说 test-time scaling 有增益，并暗示以后可以接 RL。仍然不是 DMD。

## DMD 和 Discrete FM 怎么接（论文没写，是接口）

连续 DMD 的形状是：

$$
\nabla\propto(s_{\rm fake}-s_{\rm real})\,\partial\hat x/\partial\theta
$$

离散上 $s$ 和 $\partial\hat x$ 都不存在。要保持「教师分布 − 学生分布」这一刀，匹配对象必须换成 Discrete FM 里真正有的量：

**A. 匹配条件 token 分布（Di[M]O / OPD 形）**

教师 = 多步 FUDOKI（$p^{\rm T}(\cdot\mid x_t)$）。

学生 = 少步或一步生成器，在自己滚出的 $x_t$ 上被拉向教师：

$$
\mathcal{L}_{\rm dist}=\mathbb{E}_{x_t\sim p_\theta}\big[\mathrm{KL}\big(p_\theta(\cdot\mid x_t)\,\|\,p^{\rm T}(\cdot\mid x_t)\big)\big]
$$

这就是上一页的离散 DMD。状态必须 on-policy（学生自己的脏序列），否则又变成 teacher forcing。

**B. 匹配条件期望 / soft 概率（D-MMD 形）**

离散不可微，对学生期望再训一个 auxiliary，更新 **Teacher − Auxiliary**。匹配概率向量，不匹配 hard token。

**C. 匹配离散速度本身**

教师有闭式或学出的 $u^{\rm T}_{t,i}(x\mid z)$，学生学 $u^\theta$。连续 FM 蒸馏里有人蒸速度场；离散版就是蒸速率矩阵。X 上 Discrete FM 帖子几乎不谈这条蒸馏，但数学上最「FM 原生」。

**D. 不要做的**

把 one-hot 当 $\mathbb{R}^V$ 向量加高斯再跑原版 DMD。词表方向上的欧氏分数和 CTMC 通量不是一回事。

## 一张对应表

```text
连续 FM:     学 v_θ(x_t)           采样 ODE
连续扩散:    学 score              再转 v
连续 DMD:    少步 G_θ，s_real−s_fake

离散 FM:     路径给出 u*，网学 p(·|x_t)
             采样 CTMC 用 u(p)
FUDOKI:      度量路径 + kinetic-optimal u*
             训练 CE；不是 DMD

离散「DMD」: 多步 FUDOKI 当教师
             少步学生在自己的 x_t 上
             对齐 p(·|x_t) 或 E[x0|x_t] 或 u
```

Self-Forcing 那套「自己滚长轨迹、短窗再对齐教师」也可以搬：学生按 CTMC 滚很长的 token 序列，随机切教师能看的短窗，在窗上做 A/B。那是 Discrete FM 的 Self-Forcing++，不是 FUDOKI 原文。

## 和你前面几问的关系

- FUDOKI 解决的是 离散状态上怎么定义流、怎么允许改已经写出的 token，不是少步蒸馏。
- Kinetic-optimal 是 腐蚀路径的通量设计，让语义近的 token 之间走得省。网络并不直接输出 $u^*$。
- 若要把 DMD 接上 FUDOKI：教师用多步离散流，学生少步，监督用 token 条件分布差（Di[M]O/D-MMD），不要用连续高斯分数。
- 论文自己指的下一步是 test-time scaling / RL，不是 DMD。

一句话：Discrete Flow Matching 是离散版「直接学怎么搬质量」；DMD 是「已经会搬了，再压成少步」。FUDOKI 只做了前者。两者结合点是把 DMD 的分数差换成 $p_\theta(\mathrm{token}\mid x_t)$ 或离散速度 $u$ 的差。

16 sources

---

## User

能把这个数学问题抽象出来吗

---

## Grok

把整条线收成一个问题。细节（视频、token、LOB、RL）都是同一套对象的 specialization。

## 对象

有一个生成过程，状态在空间 $\mathcal{X}$ 上：

- 连续：$\mathcal{X}=\mathbb{R}^d$（潜变量 / 像素）
- 离散：$\mathcal{X}=\mathcal{V}^D$，$\mathcal{V}$ 有限词表（token / 事件）

时间有两层，不要混：

- 过程时间 $t\in[0,1]$：噪声 $\to$ 数据（扩散 / flow / mask）
- 展开时间 $k=1,\dots,N$：自回归 / 逐帧 / 逐事件 rollout（推理时真正走过的链）

教师 $p^{\mathrm{T}}$ 只在短窗 $N\le N_{\mathrm{T}}$ 上可靠。

学生 $p_\theta$ 必须在 $N\gg N_{\mathrm{T}}$ 上自己走。

一条学生轨迹：

$$
\hat x^{1:N}\sim p_\theta(\,\cdot\,;\,\text{自己的前文})
$$

训练分布 $p_{\mathrm{train}}$ 若用真值前文，则

$$
p_{\mathrm{train}}(\hat x^k\mid x^{<k}_{\mathrm{GT}})
\;\neq\;
p_{\mathrm{test}}(\hat x^k\mid \hat x^{<k})
$$

这就是 exposure bias / 轨迹误差累积。抽象问题不提视频或语言，只提这两条测度不等。

---

## 问题（主形式）

求 $\theta$，使得学生在 **自己到达的状态** 上，短窗边缘靠近教师（或数据）：

$$
\min_\theta\;
\mathbb{E}_{k\sim\mathrm{Unif}\{1,\dots,N-N_{\mathrm{T}}\}}
\;
\mathbb{E}_{\hat x^{k:k+N_{\mathrm{T}}}\sim p_\theta}
\Big[
D\big(
p_\theta(\,\cdot\mid \hat x^{k:k+N_{\mathrm{T}}},t)
\;\big\|\;
p^{\mathrm{T}}(\,\cdot\mid \hat x^{k:k+N_{\mathrm{T}}},t)
\big)
\Big]
$$

约束：

1. On-policy：外层期望的状态来自 $p_\theta$，不是 $p_{\mathrm{data}}$ 的真值前文。
2. 教师窗：$D(\cdot\|\cdot)$ 只在长度 $\le N_{\mathrm{T}}$ 的片段上可计算。
3. 过程一致性：比较必须在同一过程时间 $t$ 的带噪 / 腐蚀边际上做。

$D$ 还没选定。选定之后才分别叫 DMD、OPD、Discrete FM 蒸馏。

---

## $D$ 的三种坐标（同一散度，不同可计算性）

连续、可微、高斯腐蚀：

$$
\nabla_\theta D_{\mathrm{KL}}(p_{\theta,t}\|p^{\mathrm{T}}_t)
\;\propto\;
-\mathbb{E}\big[(s^{\mathrm{T}}(x_t,t)-s^{\mathrm{fake}}_\psi(x_t,t))\,\partial_\theta\hat x\big]
$$

这是 **DMD**。$s^{\mathrm{fake}}$ 是追 $p_\theta$ 的 critic。

离散、token 条件分布可求：

$$
D=\sum_i\mathrm{KL}\big(p_\theta(x_i\mid \hat x,t)\;\big\|\;p^{\mathrm{T}}(x_i\mid \hat x,t)\big)
$$

或匹配条件期望 / soft 概率（D-MMD：学生期望用 auxiliary $\psi$ 代替，更新形如 $p^{\mathrm{T}}-\psi$）。这是 **Di[M]O / OPD / D-MMD**。

离散、用流的语言：质量通量 $u_t(x\mid z)\ge 0$（$x\neq z$）满足连续性方程

$$
\partial_t p_t+\mathrm{div}(p_t u_t)=0
$$

Kinetic-optimal 是在通量代价下的特解，例如度量路径上

$$
u^\star_{t,i}(x\mid z)=\big[p_t(x\mid x_0)\,\dot\beta_t\big(d(z,x)-d(x_0,x)\big)\big]_+
$$

FUDOKI **训练**并不回归 $u^\star$，而是学

$$
p_\theta(\,\cdot\mid x_t)\approx p(\,x_0\mid x_t)
$$

再用 $p_\theta$ 重建采样用的 $u$。蒸馏若要「FM 原生」，应匹配 $u_\theta$ 与 $u^{\mathrm{T}}$；实践上等价于匹配 $p(\cdot\mid x_t)$。

三种坐标描述同一件事：**在学生自己的脏状态上，对齐教师诱导的局部传输场。**

---

## 生成（预训练）和蒸馏（少步 / 长程）是两个问题

**生成（FUDOKI / 连续 FM）**

已知路径 $p_t(\cdot\mid x_0)$，学一个局部场，使从先验积出来的边际等于数据：

$$
\min_\theta\;\mathbb{E}_{t,x_0,x_t\sim p_t(\cdot\mid x_0)}
\;\ell\big(F_\theta(x_t,t),\,x_0\text{ 或 }u^\star(\cdot\mid x_0)\big)
$$

状态可以来自数据腐蚀（off-policy 于学生生成过程）。这是密度估计 / 场学习。

**蒸馏（DMD / Self-Forcing++ / OPD）**

教师场已有，学生生成器更穷（更少步，或更长 $N$）。必须在学生轨迹上匹配教师场，否则匹配的是错误测度。这是

$$
\min_\theta\;
\mathbb{E}_{\hat x\sim p_\theta}\;D\big(F_\theta(\hat x)\,\|\,F^{\mathrm{T}}(\hat x)\big)
$$

Self-Forcing++ 只加了一条：$\hat x$ 的长度 $N\gg N_{\mathrm{T}}$，$D$ 只作用在随机短窗上。教师从不看见整条长轨迹。

---

## 四个派生问题（同一主形式的切片）

**P1 测度错配**

若把 $\hat x\sim p_\theta$ 换成 $x_{\mathrm{GT}}$，目标变成另一个问题，解不能迁移到测试。Teacher forcing / CausVid 的 DF 蒸馏属于此病。

**P2 教师短视**

$p^{\mathrm{T}}$ 在 $N>N_{\mathrm{T}}$ 上无定义或偏差。只能窗化：

$$
D(p_\theta\|p^{\mathrm{T}})=\sum_{\text{windows }w}D\big(p_\theta|_w\;\big\|\;p^{\mathrm{T}}|_w\big)
$$

窗上的最优不能保证整条链最优（教师条件偏置 / Delta Forcing）。

**P3 离散不可微**

$\partial\hat x/\partial\theta=0$。必须匹配 $p(\mathrm{token}\mid\mathrm{state})$ 或 $u$，或对偶到连续再 DMD。不能把 one-hot 当 $\mathbb{R}^{|\mathcal{V}|}$ 做高斯 DMD。

**P4 可塑性 / 隔日写入（另一时间尺度）**

上面的 $t,k$ 都不是日历时间。日历上的慢权重 $\theta_\tau$、冻教师 $p^{\mathrm{T}}_{\tau-1}$ 是同一个匹配问题换了下标：

$$
\min_{\theta_\tau}\;
\mathbb{E}_{x\sim p_{\theta_\tau}}
\big[\ell_{\mathrm{new}}(\theta_\tau;x_{\tau})+\lambda D(p_{\theta_\tau}\|p^{\mathrm{T}}_{\tau-1})\big]
$$

推理时改核心权重是第四个时间尺度，主形式不包含它。

---

## 一句话

在学生自己诱导的测度上，用教师只在短窗上可信的局部传输场（连续 score、离散条件分布、或离散速度），去对齐学生的局部传输场；长程、少步、自我修正都是这个对齐在 $(N,t,\mathcal{X})$ 上的限制。

FUDOKI 提供离散 $\mathcal{X}$ 上的场 $u^\star$ 与可学的 $p(\cdot\mid x_t)$。DMD / OPD / D-MMD 提供在 $p_\theta$ 自己的态上把该场压进更穷学生的目标。Self-Forcing++ 提供 $N\gg N_{\mathrm{T}}$ 时的窗化。其余都是坐标系。

---

## User

DMD 的监督从哪来（三条网）
现场有三个东西：

1. 学生 $G_\theta$：因果自回归，自己一帧帧滚出一段视频 $\hat{x}$。
2. 冻住的 real / 教师扩散 $s_{\text{real}}$：在真视频（或双向教师）上预训练好的分数网络。问的是：这块带噪潜变量，若来自真视频分布，去噪该往哪走。
3. 可训练的 fake critic $s_{\text{fake}}$：专门拟合学生正在生成的分布。问的是：这块带噪潜变量，若来自学生分布，去噪该往哪走。 一步训练长这样： text Copy

```text
学生自己滚  →  得到干净视频 x̂
给 x̂ 加噪   →  x_t = 前向扩散(x̂, t)
教师看 x_t  →  「真分布会怎么去噪」  (pred_real)
critic 看 x_t → 「假分布会怎么去噪」  (pred_fake)
梯度 ≈ (pred_fake − pred_real)  反传到 x̂，再反传到 G_θ
```

代码里就是这一行：grad = pred_fake_image - pred_real_image（DMD 原文 eq.7）。⁠GitHub
数学上它在最小化带噪后的反向 KL：
$\mathbb{E}_t\big[D_{\mathrm{KL}}(p_{\theta,t}\,\|\,p_{\mathrm{data},t})\big]$
梯度是分数差：
$\nabla_\theta \propto -\mathbb{E}\big[(s_{\mathrm{real}}(x_t,t)-s_{\mathrm{fake}}(x_t,t))\,\partial\hat{x}/\partial\theta\big]$
$s_{\mathrm{real}}$ 来自预训练扩散（真数据流形）；$s_{\mathrm{fake}}$ 来自一边训一边追学生的 critic。
 这个是GAN吗?

---

## Grok

**精神像 GAN，对象不是 GAN。** 三条网排成「生成器 + 真模型 + 假模型」，所以看起来像；假模型学的是 **分数 / 去噪器**，不是真假分类器。

## GAN 在优化什么

标准 GAN：

$$
\min_G\max_D\;
\mathbb{E}_{x\sim p_{\mathrm{data}}}[\log D(x)]
+\mathbb{E}_{\hat x\sim p_\theta}[\log(1-D(\hat x))]
$$

- $D(x)\in(0,1)$：这块样本是真的还是假的。
- 生成器被推的方向是 $\nabla_{\hat x}\log D(\hat x)$（或非饱和变体）。
- 散度是 JS（或 f-GAN 里选定的 $f$）。
- 没有扩散时间 $t$，也不需要给 $\hat x$ 加噪。

判别器是分类器。它不回答「若来自真分布，去噪该往哪」。

## DMD 在优化什么

$$
\min_\theta\;\mathbb{E}_t\big[D_{\mathrm{KL}}(p_{\theta,t}\,\|\,p_{\mathrm{data},t})\big]
$$

用分数恒等式，KL 对样本的梯度是分数差：

$$
\nabla_{\hat x}\mathrm{KL}
=\;s_{\theta,t}(\hat x_t)-s_{\mathrm{data},t}(\hat x_t)
$$

所以

$$
\nabla_\theta\propto
-\mathbb{E}\big[(s_{\mathrm{real}}-s_{\mathrm{fake}})\,\partial\hat x/\partial\theta\big]
$$

代码里 grad = pred_fake - pred_real 是同一件事的 $x_0$-预测写法（两个去噪器差，不是两个 logits 差）。

- $s_{\mathrm{real}}$：冻住的扩散教师，在真数据上预训练。
- $s_{\mathrm{fake}}$：扩散损失训在学生样本上，拟合 $p_\theta$ 的 score，不是 $\sigma(\text{真假})$。
- 比较发生在带噪边际 $p_t$ 上。

## 并排

|  | GAN | DMD |
| --- | --- | --- |
| 假模型输出 | $D(x)$ 真假概率 | $s_{\mathrm{fake}}(x_t,t)$ 或 $\hat x_0$ |
| 假模型损失 | 分类 / logistic | 扩散去噪（对学生样本） |
| 生成器看见的力 | $\nabla\log D$ | $s_{\mathrm{fake}}-s_{\mathrm{real}}$ |
| 目标散度 | JS（默认） | 反向 KL（mode-seeking） |
| 要不要加噪 | 通常不要 | 必须，在某个 $t$ |
| 教师 | 没有冻住的世界模型 | 有一个预训练扩散 |

相似点只有一句：都是「用一个跟着学生走的网络，减去一个代表真分布的网络，把差送回生成器」。GAN 里这两个网是同一个分类器的两种输入；DMD 里是两个去噪器。

## 为什么还是常被拿来跟 GAN 比

反向 KL 的分数形式，等价于一类 **score 参数化的判别器**。若强行写

$$
D_t(x)\;\propto\;
\frac{p_{\mathrm{data},t}(x)}{p_{\theta,t}(x)}
$$

则 $\nabla\log D_t=s_{\mathrm{real}}-s_{\mathrm{fake}}$，生成器梯度与 GAN 同构，只是 $D$ 不直接学分类，而由两个 score 的比定义。所以：DMD 是 **用两个扩散器实现的、在每个噪声水平上的 KL-GAN**，不是训练 $D:x\mapsto[0,1]$ 的那种 GAN。

后果也跟 GAN 家族一致：反向 KL 会挤模式（少步 DMD 图更干净、更少杂模）；critic 滞后会让学生追一个过时的 $p_\theta$；两边要交替更新（Self-Forcing / Lip Forcing 里学生:critic 常见 5:1）。

Self-Forcing 把 GAN 列成 **另一种** $D$，和 DMD、SiD 并列。那一格才是真 GAN（JS + 判别器）。主路径那行 pred_fake - pred_real 不是它。

## 一句话

三条网 $\neq$ GAN。

GAN：假模型说「真还是假」。

DMD：假模型说「若这是学生分布，去噪往哪」；真模型说「若这是数据分布，去噪往哪」；相减就是反向 KL 的力。

---

## User

那critic 看起来很像GAN啊

---

## Grok

像，而且应该像。差别不在「有没有一个追着学生的第二网络」，而在 **这个网络被训练成什么函数**。

## 你觉得像的那一层是真的

两边都是三人局：

- 学生造样本
- 一个冻住的「真」参照
- 一个跟着学生跑的网，专门描述「假分布现在长什么样」

生成器都被推去：**靠近真参照，离开当前的假描述**。假网都要频繁更新，否则学生在打一个过时的靶。交替更新、mode-seeking、不稳——这些手感确实是 GAN 的。

所以说「DMD 是 score 实现的 GAN」并不错，前面已经把 $D_t\propto p_{\mathrm{data},t}/p_{\theta,t}$ 写出来了：$\nabla\log D_t=s_{\mathrm{real}}-s_{\mathrm{fake}}$。

## 不像的是 critic 的作业

GAN 判别器的标签是二元的：

$$
\mathcal{L}_D=-\log D(x_{\mathrm{real}})-\log(1-D(\hat x))
$$

它学的是决策面。真样本要往 1 推，假样本要往 0 推。同一个 $D$ 同时扮演「真」和「假」。

DMD critic 的标签是去噪目标：

$$
\mathcal{L}_{\mathrm{fake}}=\mathbb{E}_{t,\hat x\sim p_\theta,\,x_t}
\big\|s_\psi(x_t,t)-\nabla_{x_t}\log p_{\theta,t}(x_t)\big\|^2
$$

实现上就是对学生视频做标准扩散损失：加噪，预测噪声或 $\hat x_0$。它 **从不看「这是真还是假」这个 bit**。真分布那一侧完全不经过它，由冻住的 $s_{\mathrm{real}}$ 单独负责。

所以：

|  | GAN 的 $D$ | DMD 的 critic |
| --- | --- | --- |
| 输入 | 干净 $x$ 或 $\hat x$ | 带噪 $x_t$，还要 $t$ |
| 输出 | 一个分数 / 概率 | 一个和 $x$ 同形状的向量场 |
| 看见真样本吗 | 每步都看见 | 训练 critic 时可以不看真样本 |
| 学的函数 | $p_{\mathrm{data}}/(p_{\mathrm{data}}+p_\theta)$（最优时） | $\nabla\log p_\theta$ |
| 「真」从哪来 | 同一个 $D$ 在真样本上的 logit | 另一个冻住的扩散器 |

最优 GAN 判别器确实隐含了两个密度的比；DMD 把这个比拆成两个 score，假的那边单独用扩散损失估。看起来像 $D$，训练目标却是「把学生分布的去噪器拟合好」，不是「把真假分开」。

## 一个直观差别

把学生突然改得很差：

- GAN 的 $D$：假样本更好认，分类损失变容易，$D(\hat x)\to 0$。它仍然只输出一个标量。
- DMD critic：学生分布变了，它的扩散损失变大，必须重新学会「这种更差的视频加噪后怎么去噪」。输出仍是整张图的方向，不是「更假了」三个字。

生成器吃到的力：

- GAN：沿「让 $D$ 升高」的方向，标量场的梯度。
- DMD：沿「教师去噪方向 − 自己去噪方向」，两个向量场相减。教师方向来自世界模型，不来自 critic。

critic 单独不能提供「真」；没有 $s_{\mathrm{real}}$，pred_fake - pred_real 不存在。GAN 里拿走真样本，$D$ 立刻废掉，因为真假是它自己对比出来的。

## 什么时候它真的变成 GAN

Self-Forcing 把 GAN 写成和 DMD 并列的第三种 $D$：这时假网络就是判别器，损失是 JS，输出标量。那一格才是字面意义的 GAN。

主路径那条 critic 是 **学生分布的在线 score 估计器**。角色上像 GAN 的假侧，函数类和损失都是扩散的。

可以记成：

> critic $\approx$ GAN 里「只负责假分布」的那一半，而且用去噪器来参数化密度，不用分类器。真的那一半不在 critic 里，在冻住的教师扩散里。

---

## User

有专门帮助区分这个的tweets吗?

---

## Grok

有。X 上最直接把两者掰开的，就是下面这几条——而且社区后来还把 **DMD 和 GAN 当成两个可以加减的损失**，这本身就说明 critic ≠ 判别器。

## 最干净的一句

David Fadaeinejad 讲 DMD 的 7/11：

> DMD 训练时三个网：冻教师 → $s_{\mathrm{real}}$；在线 fake critic 扩散 → 追 $p_{\mathrm{fake}}$，给 $s_{\mathrm{fake}}$；一步学生。推理只留学生。
> 
> **It looks like a GAN, but with score functions instead of a discriminator.**[⁠@AFadaeinejad](https://x.com/AFadaeinejad/status/2051155874148655204)

同一条线前面两帖把「像」说清楚了，而不需要分类器：

- 目标是 $\mathrm{KL}(p_{\mathrm{fake}}\|p_{\mathrm{real}})$，密度算不出，梯度却是两个 score 的差。
- $s_{\mathrm{real}}$ 把样本推向真实区域；$-s_{\mathrm{fake}}$ 把样本推离学生正在过量生产的模式。差 =「更像真的，又更不像你现在那套」。

再前一帖解释为什么必须加噪：干净的 $p_{\mathrm{real}}$ 和 $p_{\mathrm{fake}}$ 若几乎不重叠，score 没定义；加高斯之后处处重叠，而扩散模型本来就会估带噪 score。

## 社区用「DMD + GAN」证明它们是两件东西

若 critic 已经是判别器，就不会再单独加一个 GAN。

同一线程 9/11，DMD2：

> 扔掉回归损失。两个新招：假 critic 和生成器 5:1 更新，避免 $s_{\mathrm{fake}}$ 滞后；**另外加一项打在真数据上的 GAN loss，用来纠正教师自己的偏置。**[⁠@AFadaeinejad](https://x.com/AFadaeinejad/status/2051155987059327363)

Siyi Chen（DFD）：

> DMD2 里 GAN 主要是通过判别器把真数据信号打进去。DFD 把这路监督写进分布匹配梯度里，所以 **可以整段丢掉 GAN**：更稳、少一个网、质量不掉。[⁠@SiyiChen277112](https://x.com/SiyiChen277112/status/2070701152895660525)

同一作者开源帖标题就是 **No GAN**，核心改成蒸馏时直接用真数据的 score。[⁠@SiyiChen277112](https://x.com/SiyiChen277112/status/2069618970596978860)

Jiaqi Feng 讨论 1-step AR 视频：Self-Forcing 在 4 步时觉得 GAN 多余；到 1 步才「GAN is crucial」——这里的 GAN 是外挂判别器，不是那条 fake score。[⁠@FengLeader](https://x.com/FengLeader/status/2059326271893700853)

Neta Shaul 把视频蒸馏分成两派：主导方法靠 **VSD 或 GAN（distribution-based）**；他自己的 PDD 是轨迹法。DMD 被放在分布匹配那一档，和 GAN 并列，不是同一损失。[⁠@shaulneta](https://x.com/shaulneta/status/2082484283323679103)

还有工程向的：跑 DMD2 省显存时可以 **turn off the discriminator**——关掉的是 GAN 头，不是 fake score critic。[⁠@julberner](https://x.com/julberner/status/2022038457649082613)

## 这些帖合在一起的判定

| 帖子在说 | 含义 |
| --- | --- |
| “looks like a GAN, but with scores instead of a discriminator” | 结构像，函数不是 $D(x)$ |
| DMD2 = DMD **plus** GAN | 两项损失，两个假网 |
| DFD = 去掉 GAN，匹配梯度里显式加真数据 score | GAN 可删，critic 还在 |
| 4-step 可不用 GAN，1-step 再挂判别器 | 判别器是可选稳定器 |
| 省显存关掉 discriminator | 关掉的不是 $s_{\mathrm{fake}}$ |

X 上没有把 fake critic 训成「真/假 logit」的主流实现帖。它被反复写成 **trained on student outputs 的小扩散模型**。GAN 头是另一条，DMD2 才把它加回来打教师偏置。

读这三串就够：Fadaeinejad 1–11、Siyi Chen 的 DFD「No GAN」、DMD2「GAN loss on real data」。第一串把「像但不等于」说完，后两串用加减法证明 critic 和判别器能拆开。

25 sources

---

## User

DMD teach and explain tweets

---

## Grok

X 上能当教材用的 DMD 帖不多。下面按「先懂机制 → 再懂变体 → 别和 GAN 搅在一起」排。

## 1. 先读：机制讲完的那条

**David Fadaeinejad @AFadaeinejad · 11 连帖**（2026-05-04）

起点：[https://x.com/AFadaeinejad/status/2051154889225437522](https://x.com/AFadaeinejad/status/2051154889225437522)

一条一条在干什么：

| 帖 | 教的东西 |
| --- | --- |
| 1 | 50 步 → 1 步，**不抄教师轨迹** |
| 2–3 | 轨迹蒸馏 vs 分布蒸馏：同一噪声不必同一张图，集合统计像就行 |
| 4 | $\nabla\mathrm{KL}=\mathbb{E}[(s_{\mathrm{fake}}-s_{\mathrm{real}})\partial G/\partial\theta]$ |
| 5 | $s_{\mathrm{real}}$ 推向真实区；$-s_{\mathrm{fake}}$ 推离学生正在过量产的模式 |
| 6 | 干净分布不重叠 → 加噪，扩散器本来就会估带噪 score |
| 7 | 三网：冻教师 / 在线 fake critic 扩散 / 一步学生。**看起来像 GAN，用的是 score 不是判别器** |
| 8 | 原版 DMD 还加 LPIPS 回归当锚，否则反向 KL 会 mode collapse |
| 9 | DMD2：扔掉回归；critic:生成器 = 5:1；**另加 GAN 打教师偏置** |
| 10 | CausVid：双向教师 → 因果视频学生，同一套分数差 |
| 11 | Medium 长文链接 |

这是 X 上把「监督从哪来」写得最完整的一条。后面所有应用帖都默认你读过这些。

## 2. 作者口径（短、官方）

**MIT CSAIL @MIT_CSAIL**（2024-03-29，DMD 发布）

[https://x.com/MIT_CSAIL/status/1773752230488805813](https://x.com/MIT_CSAIL/status/1773752230488805813)

要点就三句：

- 一步生成器 = 回归损失锚定映射 + 分布匹配损失对齐出现频率
- 分布匹配的梯度 = 两个扩散器输出之差（真样本训一个，假样本训一个）
- COCO FID 11.49，对标 SD v1.5，快约 30×

作者 Tianwei Yin @TianweiY 后来转这条，说做视频 4 步蒸馏时「excellent overview 看这条」。机制仍以 CSAIL / Fadaeinejad 为准，他本人这条偏应用。[⁠@TianweiY](https://x.com/TianweiY/status/1865233726797688922)

## 3. 变体：什么时候多一个 GAN，什么时候丢掉

读完主线再看这些，才不会把 critic 和判别器焊死。

- DMD2 = DMD + GAN（Fadaeinejad 9/11）：GAN 是第二项损失，用来纠正教师偏置，不是把 critic 换成 $D(x)$。
- Siyi Chen @SiyiChen277112 · DFD：DMD2 里 GAN 通过判别器灌真数据信号；DFD 把这路写进匹配梯度，可以 No GAN。https://x.com/SiyiChen277112/status/2069618970596978860
- Jiaqi Feng：4 步 Self-Forcing 常觉得 GAN 多余，1 步 AR 视频才「GAN is crucial」——外挂判别器，不是 fake score。
- Neta Shaul：视频蒸馏分成 VSD/GAN/DMD 这类 distribution-based，和他的轨迹法 PDD 对立。用来记住：DMD 和 GAN 是同一档里的两个损失，不是同一个网。

## 4. 只报「用了 DMD」、不教公式的（浏览即可）

这些能看产线把 DMD 当成什么，不要当教程：

- fal：FLUX.2 Turbo = custom DMD2
- NVIDIA TMD：MeanFlow + DMD2，视频 < 3 NFE
- LongCat-Avatar / ShotStream / LPM：8 步或 2–4 步 DMD 当加速器

里面几乎不写 $s_{\mathrm{fake}}-s_{\mathrm{real}}$。

## 建议阅读顺序

1. CSAIL 发布帖（2 分钟，官方定义）
2. Fadaeinejad 1–7（核心：KL → 分数差 → 三网 → 不是判别器）
3. 同一条 8–10（DMD → DMD2 → CausVid）
4. DFD「No GAN」+ DMD2「plus GAN」（确认两项可拆）

X 上没有比 Fadaeinejad 更适合当讲义的 DMD 线程。作者原帖更短，数学展开在他那 11 条里。

30 sources

---

## User

DMD DMRL

---

## Grok

你写的 **DMRL** 在文献里对应的是 **DMDR**：*Distribution Matching Distillation Meets Reinforcement Learning*（arXiv:2511.13649，ECCV 26）。X 上标题就是 “DMD meets RL”。旁边还有两个近亲名字，先分开。

| 缩写 | 全称 | 干什么 |
| --- | --- | --- |
| **DMD** | Distribution Matching Distillation | 少步生成器对齐教师/数据分布 |
| **DMDR** | DMD Meets RL | 把奖励倾斜写进 DMD，再和 RL 联合训 |
| **DMPO** | Distribution Matching Policy Optimization | 给 **扩散 LLM** 的 RL，匹配 reward-tilted 策略，不是蒸图像 |
| $R_{\mathrm{dm}}$ / GNDM | matching 当成 reward | 把 DMD 的 KL 项改写成 RL 里的奖励 |

下面只讲 DMDR。

## DMD 原来对齐谁

$$
\min_\theta\;
\mathbb{E}_t\,D_{\mathrm{KL}}\big(p_{\theta,t}\,\|\,p_{\mathrm{data},t}\big)
\quad\Longrightarrow\quad
\nabla\propto(s_{\mathrm{fake}}-s_{\mathrm{real}})\,\partial\hat x/\partial\theta
$$

教师/数据分布是平的：整堆真实图像一视同仁。RL 若另开一阶段，常见写法是

$$
\mathcal{L}=\mathcal{L}_{\mathrm{DMD}}+\lambda\,\mathcal{L}_{\mathrm{RL}}
$$

两项打架：DMD 要覆盖全分布，RL 要挤向高奖励模式，后者容易 reward hack。

## DMDR 改的那一刀

不把奖励当第二项损失，而是 **改匹配的靶分布**（reward-tilted）：

$$
p_{\mathrm{tgt}}(x)\;\propto\;p_{\mathrm{data}}(x)\,e^{r(x)/\alpha}
$$

蒸馏变成

$$
\min_\theta\;
D_{\mathrm{KL}}\big(p_\theta\,\big\|\,p_{\mathrm{tgt}}\big)
$$

分数差里的 $s_{\mathrm{real}}$ 换成「被奖励倾斜过的真分布」的 score。学生少步模型不再均匀压缩教师，而是压缩 **教师 × 偏好**。

论文分两段：

1. 先用 reward-tilted matching + 动态蒸馏策略把少步生成器立住。
2. 再 联合 DMD 和 RL：RL 让蒸馏有偏好；DMD 当正则，挡住奖励黑客。

作者声称少步学生视觉质量和跟 prompt 的程度可以超过多步教师。Harry Yang 在 X 上直接丢：*DMD meets RL* + 该 arXiv。实践侧有人说某 Turbo 人像模型用了 DMDR，结果很难直接微调，但还能做 RL。[⁠@hyang_ust](https://x.com/hyang_ust/status/1998081320510173541)

## 和「三条网」怎么接

还是那三个人，靶变了：

```text
学生 G_θ 滚出 x̂
加噪 → x_t
s_fake      仍追 p_θ
s_real/tgt  不再是平的 p_data，而是 reward-tilt 后的场
梯度        (s_fake − s_tgt) ∂x̂/∂θ
```

RL 阶段可以继续用同一套样本打外部奖励（美学、HPS、判别器）。AdvDMD 更绝：直接拿 DMD2 的判别器当 reward model（真图高分、生成图低分），等于 GAN 头兼奖励。

$R_{\mathrm{dm}}$ 那篇把整段 KL 写成奖励，再用 group normalization（GNDM，像 GRPO 的组均值）估 $R_{\mathrm{dm}}$，然后和别的奖励加权。数学上和 DMDR 同一方向：匹配项进入 RL 的奖励槽，而不是永远当一项独立 loss。

## 和你前面那页的位置

```text
DMD          对齐 p_data
DMD2         DMD + 真数据 GAN（纠教师偏）
DMDR         对齐 e^{r} p_data，再和 RL 联立
Self-Forcing  DMD 打在学生自己的帧轨迹上
OPD          离散 token 上的同一形状（条件 KL）
```

Self-Forcing++ 的「可选第三步：GRPO + 光流奖励」就是粗糙版 DMDR：先分布匹配，再 RL。DMDR 说这两步应该共享同一个倾斜靶，不要先蒸平分布再硬拉奖励。

DMPO 不要和 DMDR 混：它是 **dLLM 推理** 的 policy optimization（用交叉熵去贴最优 reward-tilted 策略），没有 fake critic 扩散器。

一句话：DMD 问「学生分布像不像数据」；DMDR 问「学生分布像不像 **被奖励倾斜过的数据**」，并用这项匹配当 RL 的正则。X 上入口就是 Harry Yang 那条 [2511.13649](https://arxiv.org/abs/2511.13649)。

27 sources

---

## User

DMD discreate tokens

---

## Grok

原版 DMD **不能直接打在离散 token 上**。能接上的，都是换了匹配对象：不再用高斯分数差，改用 **token 条件分布差**。

## 连续 DMD 为什么断在 token 上

连续版：

$$
\nabla_\theta \propto -\mathbb{E}\big[(s_{\mathrm{real}}(x_t)-s_{\mathrm{fake}}(x_t))\,\partial\hat x/\partial\theta\big]
$$

离散序列 $x\in\mathcal{V}^D$ 三条都不成立：

- $\partial\hat x/\partial\theta=0$（采样、argmax）
- 腐蚀是 mask / 均匀跳，不是 $x_t=(1-t)x+t\varepsilon$
- 没有 $\nabla_x\log p$ 这种欧氏分数

X 上有人说「楼梯不能蒸成一步」：离散扩散的熵在逐步 unmask 里，不在一条连续轨迹上。反驳帖点名：DUO、SDTT、**D-MMD** 说明能蒸，只是不能原样搬 DMD。

## 离散上「还叫 DMD」的三件事

还是三条网，输出从向量场换成词表分布。

```text
学生自己滚/自己 unmask  →  得到离散序列 x̂（或中间态 x_t）
教师 p^T(· | x_t)        →  「真/教师在这个脏状态下会填哪个 token」
auxiliary / critic ψ     →  「学生分布在这个状态下会填哪个 token」
梯度 ≈ (ψ − p^T) 打在 softmax 上，不是打在 one-hot 上
```

### 1. Di[M]O（最像 DMD）

作者原话：和 DMD 的区别是，不去最小化整个 latent 的去噪散度，而是最小化 **token-level conditional distributions**。灵感写的就是 on-policy distillation：在学生自己的中间态上对齐师生。

- 教师：多步 Masked Diffusion（MaskGit / Meissonic）
- 学生：一步
- 辅助网估不可积项，MDM 损失，目标是学生一步样本
- 初始化不当会散或 mode collapse（多 token 独立假设）

### 2. D-MMD（矩匹配，Hoogeboom）

匹配每个噪声态上的 **条件期望**（从该脏状态恢复出的干净分布均值）。学生期望算不出，再训 auxiliary，更新 **Teacher − Auxiliary**。匹配 soft 概率向量，不匹配 hard token。作者明确说这是把连续 moment matching 搬到离散，少步、保多样性。

### 3. OPD（语言 AR 上的同形状）

学生自己吐 token，教师给全表 logprob：

$$
\mathbb{E}_{x\sim p_\theta}\big[\mathrm{KL}(p_\theta(\cdot\mid x_{<i})\,\|\,p^{\mathrm{T}}(\cdot\mid x_{<i}))\big]
$$

没有扩散时间 $t$，没有 fake score 网。离散分布匹配里最成熟的一条，不叫 DMD，数学是同一个 $D(p_\theta\|p^{\mathrm{T}})$ 打在学生自己的态上。

## 不要做的两条

- 把 one-hot 当成 $\mathbb{R}^{|\mathcal{V}|}$ 加高斯再跑 pred_fake - pred_real。词表上的欧氏分数 ≠ CTMC 通量。
- 只用 GT 前文做匹配（teacher forcing）。那是另一条测度，测时学生走自己的前文会崩——和 Self-Forcing / OPD 同一病。

可绕路：GIDD / sticky-jump 先把离散嵌进连续，**连续段**再真 DMD，最后 jump 回 token。蒸的不是 token 本身。

## 和 FUDOKI / Discrete FM

FUDOKI 学的是 $p_\theta(\cdot\mid x_t)$（脏序列上预测干净 token 的 CE），采样用 kinetic-optimal 速度。它是预训练，不是 DMD。

要蒸少步 FUDOKI：教师多步 $p^{\mathrm{T}}(\cdot\mid x_t)$，学生少步，在学生自己的 $x_t$ 上做 Di[M]O / D-MMD。那才是 **Discrete FM 上的 DMD**。

## 一张对照

|  | 连续 DMD | 离散「DMD」 |
| --- | --- | --- |
| 状态 | $x_t\in\mathbb{R}^d$ | mask/跳之后的 token 序列 |
| 教师 | $s_{\mathrm{real}}$ | $p^{\mathrm{T}}(\mathrm{token}\mid x_t)$ |
| critic | $s_{\mathrm{fake}}$（去噪器） | auxiliary 或直接 KL |
| 反传 | 通过 $\hat x$ | 通过 softmax / 概率向量 |
| 名字 | DMD / DMD2 | Di[M]O、D-MMD、OPD |

一句话：DMD 接到离散 token，是把分数差换成 **该状态下的教师 token 分布 − 学生 token 分布**，状态必须来自学生自己。X 上这件事不叫 DMD，叫 Di[M]O / D-MMD；语言里叫 OPD。

6 sources

---

## User

● 材料齐了（A01 存在性判定、A03 形状报告、PR#22、goalbox
  图）。这次全部用这条线自己的定义和数字——并且先认一个方向性错误：这条线在 PLAN 里就刻意把判定放在
  order-flow 层（模型吐出的 token 本身），不判 book-state 层，理由是 state 层混入 replay
  引擎语义、误差不可归因。我前两轮讲的正好是被这条线排除掉的那层。
  一、定义与测量（这条线里 compound error 的精确含义）
  每条消息按五个决策字段打分：event_type、direction、price_rel、size、log10_dt。在生成深度 $m$（第 $m$
  条生成消息附近的窗口）取每个字段的边缘分布，与真实分布算 symmetric cross-half
  KL、熵归一化、按构造减掉 real-vs-real floor：
  $E(m) = D_{\text{arm}}(m) - D_{\text{floor}}(m)$
  两个独立坐标：level（$m{=}440$ 处错多少）与 slope/accumulation（错误随 $m$ 长多快）。Compound error
  的定义就是 slope；一个字段算"修好"必须双坐标同时低于 draft。评测规模 487 tickers × 2 个 held-out
  月，ticker-block 配对 bootstrap。
  二、Rollout 怎么做
  预训练 AR draft（Mamba-3，j5705912 @ step 69378）条件于真实前缀，闭环续写消息流到深度 500 / 1000 /
  1500 / 3500（后来推到 4000）条消息；每 ticker
  按天分组交错取窗口（--n-seq×--group-size——就是那个"两参数相等 = 只测了一个交易日"的陷阱，后来抬到整月
  160 窗）；生成流逐深度分块，块内收字段边缘，与真实对应分布走上面的估计器。
  三、现象——draft 基线实测
  1. 存在，但只在部分字段，且量级小。 500 深度、每 100 条消息的 excess 增长率（A01，CI 排除 0
  的加粗）：size +0.024 [+0.009,+0.038]、log10_dt +0.024 [+0.011,+0.038] 显著；event_type 边缘（1.88σ
  只有一格）；direction、price_rel 多数格含 0。
  2. 最重要的结构事实：不是无界复合，是漂移到一个错误的定态。 分块斜率（size@8×：+0.013 → +0.016 →
  −0.006 → −0.002 → −0.001）显示误差几乎全部在头约 500 条生成消息内积累完，之后进入平台。斜率随
  horizon"变小"纯粹是同一条饱和曲线被拟合在更长区间上的效应（三份独立 rollout 在同窗口斜率相差
  <15%）。所以病的形象不是发散，是头 500 条内滑进一个错误的平稳分布然后停在那。
  3. 水平层面病很轻——除了 size。 fx488 修复后口径的 draft level@440：direction −0.047、price_rel
  问题：闭环生成的消息流在头约 500 条内，size 与 log10_dt
  两个字段的边缘分布从贴真实滑到一个错误定态（excess 各累积约 +0.10
  量级）并停驻；修复的验收是双坐标——slope 和 level 同时压过 draft，两月复现。教训一（PR#22
  的主结局）：slope 单独会说谎——DFM 把 slope 5/5 压赢、price_rel 绝对斜率转负，但 level 在全部 84
  个深度点上差 2.4–33 倍；白噪声也不漂移，平斜率可能是摧毁 joint 的症状（TC 保留 3.2% vs AR 的
  85%），此问题记为 unresolved。教训二：覆盖陷阱——n-seq 8 × group-size 8 恰好等于一天，headline 从 2/5
  掉到 1/5。
  这个形态对前面所有方案讨论有一个实质修正：要修的不是"阻断无界增长"，而是"把头 500
  条内滑向的错误定态拉回来"——定态漂移集中在两个字段、病因有 tokenization 层的具体嫌疑（B7b 的
  per-offset embedding 就是对着它去的活线），这比"泛泛的 exposure bias"可打击面小得多、便宜得多。
✻ Churned for 3m 22s
❯ 你就说这个问题定义本身 不要太多的细节
● 问题定义，一句一层：
  对象：模型闭环续写出的离散消息流本身——每条消息的五个决策字段（type、direction、price_rel、size、log10
  _dt），不经撮合引擎、不看任何状态量。
  量：在生成深度 $m$ 处，取字段的边缘分布与真实分布的距离，减掉 real-vs-real 地板：
  $E(m) ;=; D_{\text{gen}}(m) - D_{\text{floor}}(m)$
  Compound error 的定义：$E(m)$ 随深度 $m$ 的增长率——模型条件在越来越多自己生成的 token
  上，吐出的分布离真实越滑越远的那个速率。
  完整的问题是二维的：slope（滑多快，= compound error 本身）和 level（停在多错的地方）。只压 slope
  会被骗——白噪声的 slope 也是零；所以"解决"定义为两个坐标同时低于基线。
  实测对形态的一个修正：它不是无界复合，而是头约 500
  条消息内滑进一个错误的平稳分布，然后停驻。所以问题的准确说法是：闭环条件化让生成分布在有限深度内漂移
  到一个偏离真实的定态——要修的是把这个定态拉回真实，而不是阻断某种持续增长。
● 两问一起答，都保持定义级。
  我的 tokenization（26-token 模式，src/lob/encoding.py）
  一条消息 = 26 个 token，词表 2112，按字段分段：
  ┌───────────────────────────────────┬─────────┬──────────────────────────────────────────────────┐
  │ 字段 │ token │ 编码 │
  │ │ 数 │ │
  ├───────────────────────────────────┼─────────┼──────────────────────────────────────────────────┤
  │ event_type │ 1 │ 类型枚举 │
  ├───────────────────────────────────┼─────────┼──────────────────────────────────────────────────┤
  │ direction │ 1 │ 买/卖 │
  ├───────────────────────────────────┼─────────┼──────────────────────────────────────────────────┤
  │ price │ 3 │ 符号 + 2 位 base-1000（±$9,999.99，无损） │
  ├───────────────────────────────────┼─────────┼──────────────────────────────────────────────────┤
  │ size │ 2 │ 2 位 base-100（0–9999） │
  ├───────────────────────────────────┼─────────┼──────────────────────────────────────────────────┤
  │ time │ 9 │ Δt 秒 1 + Δt 纳秒 3 + 绝对秒 2 + 绝对纳秒 3，全 │
  │ │ │ base-1000 │
  ├───────────────────────────────────┼─────────┼──────────────────────────────────────────────────┤
  │ 引用侧（price_ref / size_ref / │ 3 + 2 + │ 同上，指向被撤/被改的原单 │
  │ time_ref） │ 5 │ │
  └───────────────────────────────────┴─────────┴──────────────────────────────────────────────────┘
  两个决定性的结构事实。其一，词表段跨位置共享：price 段被 4 个位置复用、size 段 4 个、time 段被 14
  个位置共用（Δt 的秒位和绝对时间的纳秒位是同一批 token，只靠位置区分语义）。其二，数字 token
  的序数几何有断崖：size 的 base-100 两位编码里 99=(0,99) 与 100=(1,0) 数值相邻、token
  上最远。这两处正好对上 compound error
  的落点——有显著复合斜率的两个字段（size、log10_dt）就是编码病最重的两处（time 段承载 76.6% 的
  teacher-forced 损失、序数结构崩坏 0.073 对 price 的 0.240；B7b 的 per-offset embedding
  就是冲第一处去的）。
  Benchmark 怎么做（compound error 的操作化）
  五步：
  1. 取样：held-out 月份，487 tickers，每 ticker 按交易日交错取窗口（覆盖整月 160 窗——不是某一天）。
  2. 生成：draft 模型条件真实前缀，闭环续写 500–4000 条消息。
  3. 切深度：生成流按深度 $m$ 分块，每块提取五个决策字段各自的边缘分布。
  4. 打分：每个（字段, $m$）与真实同位分布算 symmetric cross-half KL、熵归一、减掉 real-vs-real
  地板，得 $E(m)$——地板按构造扣除，有限样本偏差自动抵消。
  5. 读两个数：对 $E(m)$ 拟合直线，slope = compound error（滑多快）；level =
  $E(440)$（停在多错）。不确定度用 ticker 配对的 block bootstrap，判定要求两坐标同时低于
  draft、两个月都复现。
  Compound error 一句话：模型条件在越来越多自己生成的 token
  上时，其输出字段分布滑离真实分布的速率。实测形态：头约 500 条内滑进错误定态后停驻，病集中在 size 和
  time 两个字段。

---

## Grok

问题先改写成这条线自己的对象，不再用「无界轨迹误差」。

## 对象

闭环续写的 **消息 token 流**，五字段边缘，不经撮合、不看 book。

$$
E(m)=D_{\mathrm{gen}}(m)-D_{\mathrm{floor}}(m)
$$

- $m$：已生成消息条数
- $D$：该深度窗口上，字段边缘对真实边缘的距离（对称 cross-half KL，熵归一）
- floor：real-vs-real，有限样本偏差按构造扣掉

两个坐标，缺一不可：

- slope：$E(m)$ 随 $m$ 的增长率 ← 你们把 compound error 定义成这个
- level：$E(440)$ ← 滑停以后停在多错

「修好」= 两坐标同时低于 draft，两月复现。只压 slope 不算——白噪声 slope 也是 0。

## 实测把问题改写成哪一句

不是

> 条件在自己 token 上，误差无界复合。

而是

> 头约 500 条内，生成边缘滑进一个偏离真实的 **定态**，然后停住。病几乎只在 size 和 log10_dt。

要修的是 **这个错误吸引子**，不是阻断一条永远变陡的斜率。

## 和前面那些方案的关系（定义级）

Self-Forcing / OPD / DMD 解决的是：训练测度是 GT 前文，测试测度是自己的前文。你们的病是这个错配的 **一个有限深度、两字段、会饱和的切片**，不是视频里那种越滚越烂。

因此：

- 不需要「滚到教师不会的 100 秒再短窗拉回」那种 Self-Forcing++；训练时见到 自己写到第 100–500 条之后的状态 就够。
- 不需要连续潜变量上的 $s_{\mathrm{real}}-s_{\mathrm{fake}}$。离散侧对齐的是字段（或 token）条件分布。
- 最便宜的打击面在编码：词表跨位置共享 + size 的 base-100 序数断崖，和斜率所在字段重合。那是 tokenization 问题，不是蒸馏问题。

一句话：闭环条件化让两个字段的边缘在有限步内迁到错误定态；compound error 是迁过去的速率，验收是速率和停驻点一起被拉回。

---

## User

那这个问题我想解决compound error要怎么办?

---

## Grok

按你们已经测到的形态做，不要按「无界复合」去做。

病是：闭环后头 ~500 条里，size 和 log10_dt 的边缘滑进错误定态。要动的是 **吸引子**（level）和 **滑过去的路**（slope），而且 joint 不能毁。顺序从便宜到贵。

## 1. 先打编码（活线，且和落点重合）

斜率显著的两个字段，正好是编码最脏的两处：size 的 base-100 断崖（99 与 100 token 最远），time 段 14 个位置共用同一批数字 token。

优先做完再谈蒸馏：

- per-offset embedding（你们的 B7b）：同一数字 token 在「size 高位 / size 低位 / Δt 秒 / 绝对纳秒」不再共享向量。
- size 换几何：更高进制、对数桶、或把数值当标量旁路，避免 99→100 在词表上跳最远。
- time 拆段：Δt 与绝对时间不要同一批 id。

这一步不改训练测度，只改模型怎么读自己写出来的数字。若 level@440 的 size / time 明显掉，后面闭环训练会轻很多。

## 2. 训练时见到「第 100–500 条自己写的前文」

Teacher forcing 只在真实前缀上对，测时条件是自己的脏前缀——错配发生在头 500 条，不是 4000 条。

最小闭环：

- 真实前缀 $L$ 条，然后 自己续写 $K\in[50,500]$ 条再算 CE / 字段级匹配。
- $K$ 从短加到覆盖饱和区即可，不必 Self-Forcing++ 那种超教师 horizon。
- 损失打在 五个决策字段的边缘或条件分布 上，不要只打整句 token CE（time 已经占了 76% teacher-forced 损失，会把 size 的事淹没）。

这就是这条线上的 on-policy / Self-Forcing：状态来自学生自己的消息流，监督仍是真实字段统计或冻 draft 的字段分布。

## 3. 验收必须双坐标；禁止只优化 slope

PR#22 已经判过：DFM slope 5/5 赢、level 全面更差，joint（TC）从 85% 掉到 3%。白噪声也不漂。

任何候选都要报：

- size / log10_dt 的 slope 和 level@440
- 其余三字段 level 不恶化
- 某种 joint 探针（TC 或字段共现），防止「边缘变平、消息变噪声」

过线定义保持你们现在的：两坐标低于 draft，两月复现。

## 4. 先不要上的

- 连续 DMD（$s_{\mathrm{real}}-s_{\mathrm{fake}}$）：你们没有连续潜变量可微路径。
- 超长 horizon 窗化蒸馏：饱和已经发生在 500 内，算力应堆在这一段的 on-policy 态。
- 只靠 GDN/TTT 改快权重：那是另一时间尺度，不回答「自己写的 size token 把条件带偏」。
- 把 book-state / 撮合误差当主指标：这条线刻意排除，会把 replay 语义混进来。

## 建议的实验序

1. 只改 tokenization（B7b ± size 几何），同一套 A01 协议重测 size / time 的 slope+level。
2. 编码有增益后，加短闭环：$K\le 500$ 的自生成前文 + 字段级 CE 或对 draft 的字段 KL。
3. 若 level 仍高、slope 已平：说明吸引子还在，再加「对真实边缘 / 冻教师边缘」的分布匹配（离散 DMD/OPD 那一格），打在学生自己的深度 $m$ 窗上。

第一刀编码，第二刀有限深度 on-policy，第三刀才是分布匹配。目标不是斜率变零，是错误定态被拉回真实。

---

## User

size 的 base-100 断崖（99 与 100 token 最远） 这里我没看懂

---

## Grok

数字挨着，两个 token 却一点都不像。

size 用 **两位 base-100**：

$$
\mathrm{size}=100\cdot a+b,\qquad a,b\in\{0,1,\dots,99\}
$$

一条 size 不是一个词，是两个词：高位 $a$、低位 $b$。

| 真实数量 | 高位 token | 低位 token |
| --- | --- | --- |
| 98 | 0 | 98 |
| 99 | 0 | 99 |
| 100 | 1 | 0 |
| 101 | 1 | 1 |

99 → 100 在数量上只差 1。

在 token 上是 $(0,99)\to(1,0)$：两个位置 **都换了**，而且低位从词表的一头跳到另一头。

模型若按「token 像不像」来泛化：

- 98 和 99 低位相邻，高位相同，好学
- 99 和 100 两个槽都变，低位 99 与 0 在 embedding 里通常最远

所以数值直线上最平常的进位，在词表几何里是一次「换高位 + 低位从 99 打回 0」的断崖。闭环时一旦生成落在进位附近，下一个 size 的条件已经不在「差 1」的邻域里，边缘就容易滑。

price 的 base-1000 三位是同一类病，只是进位更稀；你们测到复合斜率在 size 上更显著，和「两位、进位更密」是对得上的。

B7b 的 per-offset embedding 管的是另一处：同一个 id（比如数字 1）既当 size 高位、又当 size 低位、还可能当时间位，向量却共享。断崖是 **进位时两个槽同时大跳**；共享是 **同一向量在不同槽表示不同量级**。两件事都会让「数量相邻」在表示里不相邻。

---

## User

# Second-Pass Deep Dive: Plasticity, Continual Pre-Training, and a sigma-0 LOB Experiment Design

## 1. What changed relative to the first pass (approx. 200 words)

The first-pass hypothesis that "plasticity loss is mainly a from-scratch and RL phenomenon" is now falsified for language models. Hernandez-Garcia, Figliolia and Millidge (Zyphra, arXiv 2606.24752) show plasticity loss in GPT-style pre-norm Transformers from 5M to 314M non-embedding parameters, and, critically, under STATIONARY multilingual training, not just abrupt task cycling. Scale only delays onset (sublinear power law, exponent 0.8269), so a 300M-1B LOB model will eventually be affected. Lampinen's essay argues the opposite emphasis (scale plus pretraining substantially reduce interference), but concedes plasticity is not eliminated; the two cohere once you separate "delayed" from "defeated."

The hypothesis that "no mechanistic theory exists" is also outdated. Joudaki et al. (ETH/Apple, 2510.00304) give a dynamical-systems definition (entrapment in invariant sub-manifolds) with proofs for frozen-unit and cloned-unit manifolds, and Wang et al. (2605.09044) offer a computable predictor (optimization readiness). Wang et al. (2505.07796) give a closed-form CPT scaling law decoupling distribution shift from LR annealing. Still missing: any plasticity measurement in state-space or linear-attention models, and any controlled LOB continual-learning study.

## 2. Goal A: per-paper deep read

### A1. "Can Scale Save Us From Plasticity Loss in Large Language Models?"

Authors: J. Fernando Hernandez-Garcia, Tomas Figliolia, Beren Millidge (Zyphra). arXiv 2606.24752v1, cs.AI, 23 June 2026.

- Operationalization: plasticity loss = degradation in the model's ability to improve on a target distribution under a fixed training budget. Concretely they periodically run a "probing task": train a discarded copy of the checkpoint on held-out Vietnamese for 5B tokens, measure validation loss every 95 steps on 1,280 sequences of length 2,048, and take the area under the validation-loss curve (AUC) versus probe steps. Lower AUC = faster adaptation. Rising AUC across cycles = plasticity loss. They normalize as 100 x (AUC_k / AUC_1 - 1).
- Protocol: multilingual next-token prediction on CulturaX (167 languages, 6.3T tokens). Eight training languages in fixed order: English, Chinese, French, Japanese, Spanish, German, Portuguese, Russian. Each appearance = a fresh 5B-token task instance; a cycle = 8 task instances; trained up to 48 cycles. Held-out probe language = Vietnamese (only Austroasiatic language, chosen to minimize transfer), 20B train / 1B eval tokens, 5B subsampled per probe.
- Model sizes: 5M, 12M, 27M, 39M, 53M, 83M, 106M, 314M non-embedding parameters. Fixed aspect ratio d_model/L = 80, head dim 64, MLP expansion 4, GeLU, tied embeddings, Qwen3 tokenizer (vocab 151,680). AdamW, beta1 0.9, beta2 0.95, weight decay 0.1, batch 0.5M tokens, seq len 2,048, 9,537 steps per task, optimizer reset and LR warmup restart each task. LR 3e-3 (5M) to 1e-3 (314M), interpolated for intermediate sizes.
- Onset scaling law: T = 1.3e-5 * P^0.8269, where T = task-instance number at onset and P = non-embedding parameters. Sublinear exponent implies diminishing returns from scaling. Power-law beat linear, log-linear and exponential fits under leave-one-out CV. Outliers at 12M and 53M.
- Stationary finding: on a stationary mixture of all 8 languages (5M, 12M, 27M), AUC still eventually rises. The 27M model crosses zero at 475B tokens (stationary) vs 320B tokens (continual), so nonstationarity accelerates but is not necessary.
- Extrapolation: a 1B non-embedding model would show onset near 360 task instances = 1.8T tokens; a 7B model near 9T tokens, consistent with Springer et al.'s overtraining results. Authors state plainly: "scale alone cannot save us."
- Correlates measured (not causal, no smoking gun): rising average parameter magnitude; dormant MLP units (ReDo-style definition, epsilon 0.01) concentrated in single layers (over 95% dormant in layer 8 of the 53M model; nearly 80% in layer 10 of the 106M); lazy heads (entropy above 0.9*H_max, H_max approx 6.9215) and collapsed heads (below 0.1*H_max). None perfectly tracks onset.
- Remedies discussed (not all tested): higher weight decay (cites Han et al., ICML 2026, "weight decay improves language model plasticity"), weight clipping, Continual Backprop, ReDo, Self-Normalized Resets, GraMa, Shrink-and-Perturb, Selective Weight Reinitialization, learnable attention-sink tokens. Their own only interventions were weight decay 0.1 and per-task optimizer/warmup resets.
- Implication for a 300M-1B LOB model on years of data: expect measurable plasticity loss after roughly hundreds of billions of tokens; scaling from 300M to 1B delays but does not remove it; budget for weight decay tuning and reset-based mitigations from the start.

### A2. "Barriers for Learning in an Evolving World: Mathematical Understanding of Loss of Plasticity"

Authors: Amir Joudaki, Giulia Lanzillotta (ETH Zurich); Mohammad Samragh Razlighi, Iman Mirzadeh, Keivan Alizadeh (Apple); Thomas Hofmann (ETH); Mehrdad Farajtabar, Fartash Faghri (Apple). arXiv 2510.00304, October 2025. Code: github.com/ajoudaki/loss-of-plasticity.

- Formal definition (Def 2.1): a manifold M in parameter space induces LoP if the loss gradient is tangent to M at every point, so gradient flow that enters M stays in M. "Functional LoP" holds irrespective of the data distribution, arising from architecture plus GD dynamics.
- Two mechanisms: (1) frozen units from activation saturation (f'(z_v)=0), which zero incoming gradients (frozen-unit manifold M_F); (2) cloned-unit manifolds M_C from representational redundancy, generalized to "equitable partitions" (equal row/column sums, not strict weight equality).
- Theorems: Theorem 2.1 proves M_F and M_C are invariant affine subspaces under GD/SGD (and SGD+momentum, Adam) if initialized on them; only weight decay can break some symmetries. Theorem 2.2 (modular cloning) extends this to networks composed of modules with forward/backward invariance and persistence. Theorem 3.1 (rank gain across one linear-nonlinear step) shows nonlinearity strictly increases Renyi-2 effective rank unless off-diagonal correlations are exactly 0 or 1.
- Central claim: low-rank compression, simplicity bias, neural collapse and the information bottleneck all steer networks toward these LoP manifolds; properties that aid static generalization cause LoP in continual settings.
- Restoring plasticity: perturbations that break symmetry. Noisy SGD (even a single step, noise magnitude 0.01 relative to gradient norm) and Dropout let MLPs escape cloning manifolds; in ViTs escape was partial. Architectural interventions can destabilize manifolds.
- Applicability to modern layers: analysis is primarily MLP/CNN/ResNet/ViT via numerical simulation. The modular cloning theorem is architecture-general in principle, but the paper does not give dedicated theorems for normalization, residual, attention or state-space layers; ViT cloning experiments are the closest to attention.

### A3. "Predicting Plasticity in Deep Continual Learning: A Theoretical Perspective"

Authors: Jiuqi Wang (UVA), Jayanth Srinivasa (Cisco), Claire Chen (Caltech), Shuze Daniel Liu (Purdue), Ali Payani (Cisco), Shangtong Zhang (UVA). arXiv 2605.09044, 9 May 2026.

- They interpret plasticity as trainability (future optimization gain on a target task) and ask whether existing diagnostics predict it. By counterexample they show representation rank and NTK rank can fail to predict trainability in both regression and classification.
- Proposed predictor: "optimization readiness," combining gradient strength and gradient reliability. They prove it lower-bounds one-step optimization gain under standard smoothness assumptions.
- Validation: on Slowly-Changing Regression and Permuted MNIST, optimization readiness ranks checkpoints by trainability more reliably than prior diagnostics, even with far fewer samples.
- Cheap for a 300M sequence model? Yes in principle: it is gradient-based (a few forward/backward passes on a probe batch), avoiding the eigendecompositions that NTK/rank diagnostics require, so it is computable per checkpoint at modest cost.

### A4. "Overtrained Language Models Are Harder to Fine-Tune"

Authors: Jacob Mitchell Springer, Sachin Goyal, Kaiyue Wen, Tanishq Kumar, Xiang Yue, Sadhika Malladi, Graham Neubig, Aditi Raghunathan. arXiv 2503.19206, 2025; ICML 2025.

- Claim ("catastrophic overtraining"): beyond an inflection point, more pre-training tokens make post-training worse. Per the paper, "the instruction-tuned OLMo-1B model pre-trained on 3T tokens leads to over 2% worse performance on multiple standard LLM benchmarks than its 2.3T token counterpart" (specifically about 2% worse on ARC after tuning on Anthropic-HH).
- Sizes/budgets: OLMo-1B and OLMo-7B; controlled OLMo-30M models across token budgets; fine-tuning LRs from 4e-6 up.
- Mechanism: "progressive sensitivity." Longer pre-training makes parameters more sensitive to perturbations (fine-tuning, Gaussian noise). Catastrophic overtraining occurs when rising sensitivity outpaces base-model improvement, producing a U-shaped C4 perplexity. Formalized in a two-layer linear model where incremental feature learning (progressively smaller singular values) causes the sensitivity.
- Relation to plasticity loss: it is a stationary-pretraining forward-interference effect (harder to adapt later), so it is the large-scale evidence that Figliolia et al. extrapolate to. It is plasticity-adjacent (sensitivity/adaptability) rather than the classic non-stationary-stream framing.

### A5. "Learning Dynamics in Continual Pre-Training for Large Language Models"

Authors: Xingjin Wang, Howe Tissue (project lead), Lu Wang, Linjing Li, Daniel Dajun Zeng. arXiv 2505.07796, 12 May 2025; ICML 2025.

- Core idea: the CPT loss curve is a transfer between two hidden pre-training curves (one that would result from continuing on D_pt, one from training from scratch on D_cpt), decomposable into LR annealing and distribution shift.
- Equation: L(t) = L_0 + A*(S1_pt + S1_cpt)^(-alpha) - C1*S2_pt - C2*S2_cpt + B*(1 - (1 + E*S1_cpt)^(-beta)). First four terms = scaling law with LR annealing; last term = power-law distribution shift. S1 = summed LR (forward area); S2 = annealing area with lambda = 0.999.
- Loss potential = ratio of final annealed LR to peak LR in the PT phase; higher loss potential means more capacity to drop loss via annealing, and PT models with higher loss potential adapt better in CPT. Distribution shift term is independent of transfer starting point and of model size.
- Replay ratio enters via the distribution-shift term (replaying D_pt data reduces the shift and the D_pt loss increase). Peak LR sets S1.
- Validated on LLaMA-like models 106M to 1.7B non-embedding params, D_pt = FineWeb, D_cpt = Knowledge-Pile, across WSD and cosine schedules.
- Worked example for LOB CPT: suppose a 300M sigma-0 base pre-trained on 2018-2022 order flow, now continuing on 2023-2024 data. Fit A, alpha, C1, C2, B, E, beta from a few short pilot CPT runs at two peak LRs and two replay ratios. Because the distribution-shift term B*(1-(1+E*S1_cpt)^(-beta)) is starting-point and size independent, estimate B and beta once from a small pilot; then choose peak LR to hit target D_cpt (new-regime) validation NLL within the token budget while keeping the predicted D_pt (old-regime) loss increase acceptable, and set replay ratio to bound that increase. Larger distributional distance (e.g., a regime break like Aug 2024) yields larger B and a steeper slope, arguing for higher replay.

### A6. "Learning Continually by Spectral Regularization"

Authors: Alex Lewandowski, Michal Bortkiewicz, Saurabh Kumar, Andras Gyorgy, Dale Schuurmans, Mateusz Ostaszewski, Marlos C. Machado. arXiv 2406.06811, 2024; ICLR 2025.

- Regularizer: keep the maximum singular value (spectral norm) of each layer close to 1, targeting the beneficial spectral properties present at initialization. Only the largest singular value is regularized (smallest stays roughly constant); computed via one power iteration.
- Targets: gradient diversity. Spectral-norm growth during training reduces effective gradient diversity and trainability; keeping sigma_max near 1 preserves it while minimally interfering with single-task performance.
- Results: across class-incremental, label-flip and pixel-permutation nonstationarities on ResNet-18 and ViT-B (approx 86M params), spectral regularization is consistently among the best and is less sensitive to its hyperparameter than baselines; also used in continual RL via a streaming conversion.
- Cost: verbatim, "On a 1080TI, training with spectral regularization is approximately 14% slower" (one power iteration per layer).

### A7. "Nested Learning: The Illusion of Deep Learning Architectures"

Authors: Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni (Google). NeurIPS 2025; arXiv 2512.24695.

- Claims for continual learning: three contributions. (1) Deep optimizers reframed as associative-memory modules compressing gradients; (2) a self-modifying learning module ("Self-Modifying Titans") that learns its own update rule; (3) a Continuum Memory System (CMS) generalizing long/short-term memory into a spectrum of update frequencies. Higher-frequency neurons adapt fast but store briefly; lower-frequency neurons hold persistent knowledge. Combined into the HOPE architecture.
- Continual-learning evidence: the paper reports "promising results" in language modeling, knowledge incorporation, few-shot generalization, continual learning and long-context reasoning. This is largely benchmark-style LM/reasoning evaluation; there is no large-scale streaming or multi-year long-horizon continual evaluation of the kind Figliolia et al. run.
- Fast-to-slow consolidation: the multi-frequency CMS design is explicitly meant to let knowledge partially recover after forgetting via a "loop process," which is the mechanism by which faster memory feeds slower memory; but the paper frames this as architectural design rather than a demonstrated slow-weight consolidation guarantee.

### A8. Essay: "What are the real problems of continual learning?"

Author: Andrew Lampinen. infinitefaculty Substack, 29 May 2026.

- Argument: pre-training and scale have substantially reduced (not eliminated) both catastrophic interference and loss of plasticity in LLMs. Evidence relied on: (i) representation-probing papers showing earlier-task info remains linearly decodable (interference concentrated at readout layers); (ii) wider models forget less (sparser, more orthogonal gradients); (iii) pre-training-times-scale interaction reduces forgetting; (iv) layer norm and weight decay reduce plasticity loss; (v) his own preprint arguing reduced interference drives scaling benefits. He cites the view that plasticity loss may be an artifact of hard task boundaries (gradual drift preserves plasticity).
- Coherence/conflict with Figliolia et al.: partial conflict. Lampinen leans on the hard-boundary-artifact view; Figliolia et al. directly rebut it by showing plasticity loss under stationary training with no abrupt boundaries. They agree scale delays the problem and that weight decay / layer norm help. Net: Lampinen is right that scale plus standard tricks push the problem far out, but Figliolia et al. show it is not an artifact and not defeated. Lampinen's genuinely open problem (positive forward transfer / cumulative learning) is orthogonal and unaddressed by the plasticity papers.

### A9. Two RL/post-training notes

- "Reinforcement Fine-Tuning Naturally Mitigates Forgetting in Continual Post-Training" (Song Lai et al., arXiv 2507.05386, Jul 2025, rev Jan 2026). Claim: across a benchmark of seven diverse multimodal tasks with Qwen2.5-VL-7B-Instruct, supervised fine-tuning causes catastrophic forgetting whereas reinforcement fine-tuning (RFT) largely preserves prior-task performance. Quantitatively, SFT "suffered from a substantial drop... FM of -10.4%... GRPO achieved an FM [Forgetting Measure] of only -2.3%, effectively preserving performance across the entire sequence." Explanation offered: RFT's selective, on-policy gradient updates change the model less destructively, complementing concurrent "on-policy data mitigates forgetting" findings.
- "AltNet: Addressing the Plasticity-Stability Dilemma in Reinforcement Learning" (Mansi Maheshwari, John C. Raisbeck, Bruno Castro da Silva, arXiv 2512.01034, Dec 2025). Claim: full network resets restore plasticity but cause an immediate performance drop. AltNet keeps two networks that periodically swap roles; the active one acts, the passive one learns off-policy from the shared replay buffer, so plasticity is restored without the recurring performance collapse of standard resets.

## 3. Goal B: verification table

| Item | Status | Source |
| --- | --- | --- |
| SEC Rule 612 half-penny tick: adoption 18 Sep 2024 | VERIFIED | SEC press release 2024-137, 18 Sep 2024 |
| SEC Rule 612 compliance date 3 Nov 2025 | VERIFIED, then SUPERSEDED | SEC 2024-137 says "first business day of November 2025" (= 3 Nov 2025). Caveat: SEC exemptive order (press release 2025-130, Exchange Act Release No. 104172, issued 31 Oct 2025) extended Rule 612 (and 600(b)(89)(i)(F)) compliance to "the first business day of November 2026" |
| TAFAS test-time adaptation, arXiv 2501.04970 | VERIFIED | Kim, Kim, Mok, Yoon; AAAI 2025; "Battling the Non-stationarity in Time Series Forecasting via Test-time Adaptation" |
| Neuroplastic Expansion, arXiv 2410.07994 | VERIFIED | Liu, Obando-Ceron, Courville, Pan; ICLR 2025 |
| Time-MoE, arXiv 2409.16040 | VERIFIED | Shi et al.; ICLR 2025 Spotlight (Top 5.1%); "For the first time, we scaled a time series foundation model up to 2.4 billion parameters"; Time-300B "comprising over 300 billion time points spanning more than nine domains" |
| Kumar et al. L2 Init, arXiv 2308.11958, CoLLAs 2025 | VERIFIED (ID/title) | Kumar, Marklund, Van Roy; "Maintaining Plasticity in Continual Learning via Regenerative Regularization"; CoLLAs venue not separately reconfirmed |
| Hare and Tortoise, arXiv 2406.02596, ICML 2024 | VERIFIED | Lee et al.; "Slow and Steady Wins the Race" |
| Self-Normalized Resets, arXiv 2410.20098 | VERIFIED | Farias, Jozefiak; ICLR 2025 |
| AdaLin, arXiv 2505.09486 | VERIFIED | "Preserving Plasticity in Continual Learning with Adaptive Linearity Injection" |
| FIRE, arXiv 2602.08040 | VERIFIED | Han, Park, Oh, Kim, Kim; ICLR 2026; "Frobenius-Isometry Reinitialization" |
| Hyperspherical normalization, arXiv 2502.15280 | VERIFIED | Lee, Lee, Seno, Kim, Stone, Choo; "Hyperspherical Normalization for Scalable Deep RL" |
| Plasticity measured in SSM / linear attention / RWKV / GDN | NOT FOUND | No paper measures plasticity loss in these architectures; SSMs appear only as adapted models in forgetting-oriented CL work (Mamba-CL 2411.15469, MambaCL 2412.00776, Exemplar-Free CL for SSMs 2505.18604) |
| Plasticity in continually pre-trained models above 300M | PARTIAL | Springer et al. 2503.19206 (OLMo-1B/7B, sensitivity framing); Figliolia et al. cite Cho et al. 2026 "Forget forgetting" (over 1B params, only about 8 tasks) |
| LOB continual learning / drift / regime study | NOT FOUND (controlled) | Generative LOB models exist (LOBS5 2309.00638; LOB-Bench 2502.09172, ICML 2025); closest partial evidence is a fine-tuning ablation in "LiT: Limit Order Book Transformer" (Frontiers in AI 2025). No controlled retraining-vs-CPT-vs-online study |

## 4. Goal C: implementation-ready design for the first sigma-0 experiment

Objective: measure the plasticity ratio of a 300M sigma-0 model on two NASDAQ stress slices using Figliolia et al.'s fixed-budget metric and Lyle et al.'s fresh-network baseline.

Slices: (a) COVID crash, 24 Feb to 30 Apr 2020; (b) 5 Aug 2024 volatility spike plus the following four weeks.

Base pre-training window: 2015-01 to 2019-12 order flow tokens (well before slice a), giving a checkpoint theta_base with no leakage into the test slices. For slice (b), also prepare a checkpoint pre-trained through 2024-07-31.

Three arms, matched compute:

1. Continued model: from theta_base, rewarm to a target peak LR (start with 30-50% of original pre-training peak LR), replay fraction rho of old-distribution tokens mixed with the target slice, total N_c tokens.
2. Fresh model: same architecture, random init, trained on the same N_c tokens (target slice, same replay-free or matched mixture), same LR schedule.
3. (Diagnostic) Rewarm-only and replay-only ablations if compute allows.

Matched-compute rule: identical token budget N, identical batch size, identical schedule shape and identical number of optimizer steps for continued vs fresh; the only difference is initialization (pre-trained vs random). This isolates plasticity from compute.

Plasticity ratio (two definitions, report both):

- Steps-to-target: R_steps = steps_fresh(NLL*) / steps_continued(NLL*), where NLL* is a fixed target validation NLL on a held-out slice window. R_steps > 1 means the continued model is faster (healthy plasticity); R_steps < 1 means the pre-trained model is slower to adapt than a fresh net = plasticity loss.
- Fixed-budget loss reduction: R_loss = (NLL_init_continued - NLL_final_continued) / (NLL_init_fresh - NLL_final_fresh) at fixed N tokens. Equivalent AUC form: use Figliolia's validation-loss AUC over adaptation steps and report continued AUC minus fresh AUC.

Seeds and CIs: minimum 5 seeds per arm (init seed and data-order seed varied together); report mean and 95% bootstrap CI (10,000 resamples) on R_steps and R_loss; declare a difference only if the CI excludes 1.0.

Diagnostics logged every N steps (suggest every 200 steps): per-layer effective rank (Renyi-2 er_2 = (tr M)^2 / ||M||_F^2 on the feature covariance), dormant-unit fraction (ReDo-style, epsilon 0.01 on normalized absolute activations), weight L2 norm (non-embedding), gradient norm, and top Hessian eigenvalue via power iteration (10-20 Hessian-vector products, Pearlmutter trick). Also log attention/mixing entropy (lazy/collapsed head fractions) as in Figliolia et al.

Expected GPU hours: a 300M model, adaptation runs of a few billion tokens each. On a single 8-GPU node (e.g., 8xH100) a few-billion-token run is on the order of low hundreds of GPU-hours; with 3 arms x 5 seeds x 2 slices that is on the order of low-thousands of GPU-hours on a single node. At roughly 2,000-GH200 scale the same sweep completes in a small number of wall-clock hours because it is embarrassingly parallel across the 30 runs. Report exact throughput once tokenizer and sequence packing (roughly 6k-token sequences) are fixed.

Decision rule:

- "Plasticity loss PRESENT" if, on either slice, R_steps < 1 or R_loss < 1 with 95% CI excluding 1.0 (the pre-trained model adapts no faster, or reduces less loss, than a fresh net at matched compute), accompanied by at least one corroborating diagnostic trend (rising dormant fraction, falling per-layer effective rank, or rising top Hessian eigenvalue).
- "Plasticity loss ABSENT" if R_steps and R_loss are both above 1.0 with CI excluding 1.0 on both slices (pre-training strictly helps).
- Otherwise "inconclusive / mild," and extend token budget.

Multi-year cyclic protocol (adapting Figliolia et al. to LOB):

- Replace the eight languages with either calendar years (2015, 2016, ... 2024) or regime-labeled slices (low-vol trend, high-vol crash, choppy range, event-driven) as task instances. A cycle = one pass through all year/regime slices in a fixed order chosen to minimize adjacency similarity (e.g., interleave calm and volatile regimes).
- Hold out one regime slice as the probe (analogous to Vietnamese), never included in cycles; the natural choice is a distinctive tail event (e.g., the Aug 2024 spike or a flash-crash window). After each cycle, train a discarded copy on the probe slice for a fixed token budget and record validation-NLL AUC.
- Run models at 34M, 100M, 300M and 617M parameters for many cycles; fit onset T (cycle at which probe AUC begins rising) versus P and estimate the LOB analog of T = c * P^k. If k < 1 (sublinear, as in language), scaling delays but does not remove LOB plasticity loss; the fitted c and k then predict the token budget before a production sigma-0 model needs reset-based intervention.
- Log the same per-layer diagnostics each cycle; because sigma-0 uses S5/Mamba/Gated DeltaNet backbones, this doubles as the FIRST plasticity measurement in state-space / linear-attention models, which the literature currently lacks.

## 5. Recommendations

1. Run the two-slice fresh-vs-continued experiment (Section 4) first at 300M; it is a low-thousands-of-GPU-hour commitment and directly answers whether plasticity loss is real for sigma-0. Benchmark that flips the decision: R_steps or R_loss crossing 1.0 with a 95% CI excluding 1.0.
2. In parallel, adopt rewarm + replay as the CPT baseline and fit the Wang et al. (2505.07796) law on a small pilot to choose peak LR and replay ratio before committing a large run. Threshold: if the predicted D_pt (old-regime) NLL increase exceeds your production tolerance, raise the replay ratio.
3. Instrument every training run now with the five cheap diagnostics plus optimization readiness (Wang et al. 2605.09044) as a per-checkpoint trainability probe; these are near-free and give early warning before a production run degrades.
4. Pre-commit mitigations in order of cost: tune weight decay upward first (cheapest, cites Han et al.), then add spectral regularization (about 14% slower) or Self-Normalized Resets / ReDo if dormant-unit fraction climbs, and reserve AltNet-style dual-network swapping or continual-backprop resets for the online-adaptation setting.
5. Run the multi-year cyclic onset curve (34M/100M/300M/617M) once the two-slice result is in; publishing the fitted exponent k would be the first plasticity onset law for SSM/linear-attention sequence models and directly sizes the reset cadence for production.

## 6. Caveats / what could not be accessed or verified

- Joudaki et al. PDF returned empty for direct text extraction; all detail here is from the arXiv HTML (v1), which was complete. Theorem numbering and the escape-perturbation results are quoted from that HTML.
- The exact fitted numeric coefficients (A, C1, C2, B, E, beta) of the Wang et al. CPT law are not reported in the extracted body; only the functional form and lambda = 0.999 are confirmed. The LOB worked example therefore specifies the fitting procedure rather than plugging fitted constants.
- Cho et al. 2026 "Forget forgetting: continual learning in a world of abundant memory" (cited by Figliolia et al. as an over-1B-parameter plasticity study on about 8 tasks) was not independently fetched; parameter and task counts come secondhand from the Figliolia reference list.
- CoLLAs 2025 venue for Kumar et al. L2 Init was not separately reconfirmed; the arXiv ID 2308.11958 and title are confirmed.
- Nested Learning continual-learning experiments: only abstract-level descriptions were confirmed; the specific benchmarks and whether any streaming/long-horizon evaluation exists were not extracted from the full text.
- No controlled LOB continual-learning study (periodic retraining vs CPT vs online) was found; the closest partial evidence is a fine-tuning ablation in "LiT: Limit Order Book Transformer" (Frontiers in AI 2025), which is an inference rather than an authors' continual-learning claim.
- GPU-hour figures are order-of-magnitude estimates pending measured throughput for roughly 6k-token sigma-0 sequences on the target hardware.
- The SEC compliance-date item requires care: 3 Nov 2025 was the originally adopted date, but the operative compliance date for Rule 612 is now the first business day of November 2026 per the October 2025 exemptive order (SEC 2025-130).

---

## Grok

这份二稿把「可塑性丧失在 LLM 上成立」钉住了。和你们刚收紧的 compound error **不是同一个实验**。先把判定改对，再跑 300M。

## 两件事不要写进同一条 decision rule

|  | Plasticity | Compound error |
| --- | --- | --- |
| 时间 | 日历 / 累计 token（CPT、多年循环） | 一次闭环里的消息深度 $m$ |
| 对象 | 慢权重还能不能学新切片 | 自己写下的 size / Δt 边缘滑向错误定态 |
| 基线 | 同架构、同预算的适应曲线 | draft 的 $E(m)$ slope + level |
| 修法 | WD、谱正则、reset、replay、rewarm | 编码、短闭环 on-policy、字段匹配 |

Figliolia 的 AUC 上升，不解释 size@500 的定态漂移。B7b 也不回答 2019 的 checkpoint 到 2024 还能不能学。

## 实验设计里必须改的一处

你们写的「ABSENT if $R>1$」会几乎必然报缺席。随机初始化、只看两个月压力切片，预训练模型理应更快——那是表征迁移，不是可塑性还在。

Figliolia 的探针是：**同一检查点年龄不同的拷贝**，同一探针预算，比 AUC。不是 pretrained vs scratch。

改成三臂里真正能证伪的一对：

1. 早检查点拷贝 $\theta_{\mathrm{base}}$（2015–2019）→ 固定预算适应当前切片
2. 晚检查点拷贝 $\theta_{\mathrm{late}}$（多训了 $T$ token 之后）→ 同一预算、同一切片、同一 schedule
3. scratch 只作下界参考，不要写进「ABSENT」

判定：

- PRESENT：$\mathrm{AUC}(\theta_{\mathrm{late}})>\mathrm{AUC}(\theta_{\mathrm{base}})$（或 $R_{\mathrm{steps}}(\mathrm{late}/\mathrm{base})<1$），CI 不含相等，且至少一项诊断同向（dormant↑、有效秩↓、optimization readiness↓）
- ABSENT：晚检查点仍显著更快或 AUC 不高出早检查点
- scratch 对比仅当 $R(\mathrm{late},\mathrm{scratch})<1$ 才是「网络几乎死了」的强结论；不要指望它单独出现

COVID 与 2024-08 仍可用；2024-08 必须用训到 2024-07-31 的晚检查点，不能用 2019 的 $\theta_{\mathrm{base}}$ 去冒充「过训后的可塑性」。

## 和 CPT 公式怎么接

Wang 的 CPT 律管的是 **旧分布 NLL 升多少、新分布降多少**，不管探针 AUC。用法保持：小规模两档 peak LR × 两档 replay 拟合 $B,\beta$，再锁生产配方。它回答「续训会不会把 2018 忘光」，不回答「还能不能学会 2024-08」。

两条都要报：旧窗 NLL（稳定性）+ 探针 AUC（可塑性）。只报一个会重演 PR#22：单一坐标说谎。

## 建议保留的五条，收成开工序

1. 先做 早 vs 晚检查点 的固定预算探针（300M，两切片，≥5 seed），再决定要不要多年 onset 曲线。
2. 并行小 CPT 拟合 replay / rewarm，不要等可塑性结论才定配方。
3. 诊断（dormant、$er_2$、$\|w\|_2$、grad、top Hessian、optimization readiness）从现在每条 run 就打日志——包括 compound-error 的 draft 训练，零额外叙事成本。
4. 缓解按成本：先加 WD，再谱正则或 ReDo；AltNet / CBP 留给真在线。
5. 34M–617M 循环 onset 放在 1 出结果之后；那才是「SSM/GDN 上第一条可塑性 onset 律」，文献表里 NOT FOUND 的那一格。

## 不要在这份实验里夹带的

- 闭环 500 条的 size/time 定态：另开 A01 协议，先编码后短 on-policy。
- SEC 612 日期：表里已写 2026-11 才是现行合规日；切片用 2024-08 波动，不要写成 tick 规则已切换。
- Nested Learning / HOPE：当叙事可以，当 300M 第一刀的 backbone 不行，没有流式可塑性证据。

一句话：可塑性实验比的是 **同一模型老了之后，同样预算还能不能贴上新切片**；compound error 比的是 **一次生成里字段边缘停在哪**。二稿的文献判断成立，把 fresh-vs-continued 的「ABSENT」改成 late-vs-early 再跑。

---

## User

你觉得他可能会有什么技术?

Continual Learning Under Non-Stationarity for a Pre-Trained LOB Foundation Model

1. Executive summary (295 words) Loss of plasticity is real, measurable, and mechanistically well understood in 2026, but the strongest evidence comes from small deep-RL and supervised-continual-learning networks, not from 100M+ sequence models in continual pre-training. The best-supported operational metric is task-agnostic: how quickly a continually trained network fits a fresh target relative to a freshly initialized copy of equal compute. Diagnostic covariates (dormant units, effective rank, weight and gradient norms, curvature) track plasticity in many settings but none is necessary or sufficient; Lyle et al. show they can decouple from each other. Mechanistically, three drivers have controlled-experiment support: unit dormancy/saturation, weight-norm growth that shrinks the effective learning rate under normalization, and curvature loss (low Hessian rank). These are distinct and partially independent. Lewandowski et al. provide a linear-network theory linking curvature loss to plasticity loss, and Muon-type spectral optimizers hold effective rank and preserve plasticity in controlled experiments. Wang's claim that no mechanistic theory exists is only half true. For sigma-0, the immediate practical answer is that large pre-trained models rarely die; they forget and drift. LLM continual pre-training shows learning-rate rewarming and 5 to 25 percent replay recover most of the benefit of from-scratch retraining, with plasticity remaining intact. Financial practice remains ad hoc: rolling retraining and exponential windows dominate, and there is no controlled comparison of periodic retraining versus continual pre-training versus online adaptation on LOB data, nor a public regime-drift benchmark. That is the gap sigma-0 can fill. The recommended path is a rewarmed, replayed continual pre-training loop with a drift-driven learning rate, a Muon-type optimizer, an explicit plasticity measurement (continued vs fresh model), and an event-sliced evaluation using verifiable NASDAQ episodes. GDN gating can act as a drift detector but is not a substitute for weight updates.
2. Findings per question Q1 Phenomenology and measurement Metrics in use. Across settings, plasticity is measured through (a) a behavioral definition, the ability to fit new targets or improve loss on new data; (b) proxy covariates: dormant-unit fraction, effective rank, weight norm, gradient norm, and curvature/sharpness. In deep RL, Sokar et al. (2023) define dormancy per neuron as a scaled activation score, showing that as much as 90 percent of neurons can go dormant [Sokar et al., ICML 2023; controlled experiment, deep RL]. Lyle et al. (2023) define plasticity as the ability of a network to reduce loss on a new target and show that losses on random targets rise while feature rank and dead-unit fraction change, but plasticity loss "cannot be reliably predicted by any single covariate" [Lyle et al., ICML 2023, "Understanding Plasticity in Neural Networks"; controlled experiment, supervised and RL]. The 2024 follow-up demonstrates that no single mechanism or covariate fully explains plasticity loss and calls for a layered view [Lyle et al., arXiv 2402.18762; controlled experiment]. Necessity/sufficiency. No covariate is necessary or sufficient. Lyle et al. explicitly note that rank and dormant units are informative but not decisive; Dohare et al. characterize plasticity loss as a general effect (in ImageNet, class-incremental CIFAR, and RL) and use a fresh-network baseline, not a covariate, to define it [Dohare et al., Nature 2024; controlled experiment]. Monotonicity/phases. In deep RL the loss is roughly monotone and cumulative over gradient steps, with a rapid early phase from early-transition overfitting (primacy bias) [Nikishin et al., ICML 2022; controlled experiment]. Dohare et al. report a slow decline on continual ImageNet over thousands of tasks without saturation. There is no evidence of stable "phases," but reset-based methods show a sawtooth pattern (plasticity restored after reset, then decays). Disentangling from forgetting. Plasticity loss is distinct from forgetting: forgetting concerns retention of old tasks while plasticity concerns the ability to acquire new ones. Dohare et al. show that networks with no forgetting constraint still lose plasticity, and Abbas et al. show plasticity loss in continual deep RL even when each task is revisited [Abbas et al., CoLLAs 2023; controlled experiment]. Elsayed and Mahmood (2024) present a single experiment (continual regression and MuJoCo) that separately measures plasticity and forgetting and finds their UPGD remedy improves both [Elsayed & Mahmood, ICLR 2024; controlled experiment]. Implication for sigma-0. Adopt the behavioral definition: for each new NASDAQ slice, train a continued model and a fresh model with matched compute and compare loss reduction. Track effective rank, dormant units, weight norm, and Hessian top eigenvalues only as diagnostics. Q2 Mechanisms

```text
| Mechanism                                                    | Evidence                                              | Refs                                                  |
| ------------------------------------------------------------ | ----------------------------------------------------- | ----------------------------------------------------- |
| Dead/saturated units (ReLU death)                            | Controlled experiment, RL; strong                     | Sokar et al. 2023; Abbas et al. 2023                  |
| Rank collapse (feature rank drops)                           | Correlational/experimental; strong but not sufficient | Lyle et al. 2023; Kumar et al. 2021                   |
| Weight-norm growth reducing effective LR under normalization | Controlled experiment; strong                         | Lyle et al. 2024 (arXiv 2407.01800); Ash & Adams 2020 |
| Curvature loss (Hessian rank)                                | Theory in linear nets plus experiment; strong         | Lewandowski et al. (arXiv 2312.00246)                 |
| Lazy/NTK drift                                               | Theoretical association; weak direct evidence         | Lyle et al. 2023 discussion                           |
| Gradient diversity loss                                      | Correlational                                         | Dohare et al. 2024                                    |
| Stale optimizer state (Adam moments)                         | Correlational/anecdotal                               | Lyle et al. 2024; Ash & Adams 2020                    |
```

Which imply each other. Dead units cause rank collapse but rank collapse can occur without dead units (Lyle et al. 2023). Weight-norm growth in normalized networks directly reduces effective learning rate and is separable from unit death [Lyle et al. 2024]. Curvature loss is shown in linear networks to be equivalent to losing rank of the Hessian; Lewandowski et al. argue this explains why pre-existing measures fail as predictors [theory plus experiment].
Quantitative theory. Lewandowski et al. derive, in linear networks, that plasticity loss arises from decline in the number of curvature directions of the loss; they propose a Wasserstein-to-init regularizer that preserves curvature and show it prevents plasticity loss in deep networks [arXiv 2312.00246; theory plus controlled experiment]. This is the only quantitative account. There is no random-feature or high-dimensional-limit theory predicting when plasticity is lost.
Spectral control. Muon-type orthogonalized optimizers (Jordan et al. 2024; Liu et al., Moonshot 2025) constrain update spectra; Lyle et al. 2024 and follow-up works show that constraining effective LR via normalization plus weight decay preserves plasticity; Elsayed & Mahmood's UPGD achieves similar effects by utility-weighted perturbation [controlled experiment, RL]. Muon's plasticity benefits are supported by experiments in LLM pre-training (Moonshot, arXiv 2502.16982) but not directly tested in continual-learning settings.
Verdict on Wang's claim. "No mechanistic theory exists" is inaccurate in 2026. There are partial mechanistic theories (curvature, effective-LR) with controlled evidence, and a unifying view is emerging (Lyle et al. 2024). But there is no single unified theory, and no quantitative predictor of when a network "dies."
Implication for sigma-0. Monitor weight norm and Hessian top eigenvalues; use Muon or spectral-norm-constrained AdamW; add regenerative L2-toward-init.
Q3 Remedies
See Table 3 below. Key points:

- Resets (continual backprop, ReDo, shrink-and-perturb, plasticity injection, SR-SPR/BBF): validated in deep RL (Atari, MuJoCo), supervised continual ImageNet, up to few-million-parameter networks. Continual backprop scales to ImageNet continual (Dohare et al. Nature 2024). Cost: temporary loss of learned ability (recovered after re-training); works for abrupt switches and slow drift alike. No validation at 100M+ or in LLM continual pre-training.
- Regularizers: L2-toward-init (Kumar, Marklund, Van Roy 2023, arXiv 2308.11958), Wasserstein-to-init (Lewandowski 2023), effective-LR control via weight decay plus normalization (Lyle 2024). Cost: minimal to learned ability; suitable for slow drift; abrupt switches may need combined resets. Validated in RL and continual ImageNet; L2-init used in continual LLM finetuning as EWC-like penalty but without plasticity measurement.
- Normalization/activation: LayerNorm placement and CReLU (Abbas 2023) help in RL; cost negligible; suited to both drift types.
- Architectural expansion: Hare and Tortoise (Lee et al., ICML 2024, arXiv 2406.02596) and Neuroplastic Expansion (Liu et al., 2024, arXiv 2410.07994) add capacity or a stable network; validated in RL; cost: extra parameters; suited to abrupt switches (Hare) and slow drift (Tortoise consolidation).
- Optimizers: UPGD (Elsayed & Mahmood 2024) and Muon-type. UPGD validated on continual regression and MuJoCo; Muon on LLM pre-training up to 15B (Moonshot). Cost: low; suited to slow drift. LLM continual pre-training practices. Rewarming plus replay (Gupta et al., arXiv 2308.04014; Ibrahim et al., TMLR 2024, arXiv 2403.08763) are engineering remedies addressing forgetting and, indirectly, plasticity by keeping effective LR high; infinite LR schedules (Ibrahim et al.) maintain a nonzero learning rate to avoid the plateau that cosine decay introduces. Evidence is from 405M and 10B models [controlled experiment, LLM pre-training]. These practices address forgetting first and plasticity second; they do not measure plasticity directly. Implication for sigma-0. Use rewarming plus 5 to 25 percent replay as the baseline; add L2-toward-init and Muon; keep reset-based methods only for targeted layers under abrupt regime shifts. Q4 Scale and pre-training
- Larger pre-trained models forget less: Ramasesh, Lewkowycz, Dyer (ICLR 2022) show forgetting decreases with scale in ResNets and Transformers under fine-tuning [controlled experiment, supervised]. Mirzadeh et al. (2022) show wide networks forget less catastrophically, attributed to lazier training [controlled experiment].
- Scaling laws for forgetting exist: Kalajdzievski (2024, arXiv 2401.05605) fits forgetting vs number of fine-tuned parameters and update steps, finding a power law increase in forgetting with both; LoRA is not safer [controlled experiment, LLM].
- Continual pre-training of LLMs (Ibrahim 2024; Gupta 2023) shows rewarm plus replay matches from-scratch performance; no dead networks reported [controlled experiment, 405M and 10B].
- Plasticity loss in pre-trained LLMs: Kotha, Springer, Raghunathan (2024, "Understanding catastrophic forgetting in LMs via implicit inference", arXiv 2309.10105) find fine-tuning shifts implicit task inference rather than destroying capabilities; plasticity is not lost [controlled experiment]. Dohare et al. (2024) show plasticity loss even in ImageNet-scale ResNets under class-incremental streams, but with from-scratch training.
- No "scaling law for plasticity" exists. Mechanism proposed for scale's protection: overparameterization keeps unused capacity and features closer to the lazy regime [Mirzadeh 2022]. There is no evidence on multi-year drift. Implication for sigma-0. At 300M to 1B parameters with continual pre-training, plasticity loss is a secondary risk; forgetting and drift are primary. Measure plasticity directly to confirm. Q5 Learning without task labels
- Task-free CL (Aljundi et al., CVPR 2019, arXiv 1812.03596): consolidation is triggered when loss plateaus or surprise spikes; memory-aware synapses updated online; no need for task boundaries [controlled experiment, supervised].
- Drift detection coupled to learning: ADWIN (Bifet & Gavalda 2007) adapts window size with false-positive guarantees; DDM/EDDM adjust learning when error rates rise; Bayesian online change-point detection (Adams & MacKay 2007, arXiv 0710.3742) gives posterior run lengths that can weight updates; Page-Hinkley used in streaming ML (Gama et al., ACM CSUR 2014). Coupling to learning rate is common in practice but formal guarantees are limited.
- Theory: Besbes, Gur, Zeevi (Operations Research 2015; arXiv 1307.5449) derive minimax dynamic regret under a variation budget V_T; optimal restart period and step sizes scale as (V_T/T)^{1/3}. Hazan & Seshadhri (ICML 2009) define adaptive regret and FLH with polylog overhead. Herbster & Warmuth (1998) tracking the best expert. These support a formalization for boundary-free drift.
- Implementable designs:

1. BOCPD-modulated learning rate: run BOCPD on the model's validation NLL residuals; set eta_t = eta_min + (eta_max minus eta_min) times posterior probability of change; assumption: piecewise-stationary; failure: slow drift under-detected.
2. Variation-budget restart: estimate V_T by cumulative KL between rolling windows; restart or rewarm every ceiling((T/V_T)^{2/3}) steps; assumption: bounded variation; failure: misestimated V_T yields over- or under-restarting.
3. Hypergradient learning rate (Baydin et al., ICLR 2018) plus drift statistic s_t as an input to an MLP that outputs eta_t; assumption: differentiable objective; failure: hypergradient noise under heavy tails. D-Adaptation (Defazio & Mishchenko 2023) is parameter-free but assumes stationary convex objectives; not suited to drift. Implication for sigma-0. Implement design 1 first (BOCPD on daily NLL); design 2 as a fallback for slow drift. Q6 Plasticity inside sequence architectures

- Gating as online learning: Gated DeltaNet (Yang, Kautz, Hatamizadeh, ICLR 2025, arXiv 2412.06464) updates a matrix memory S_t = alpha_t (I minus beta_t k_t k_t^T) S_{t-1} plus beta_t v_t k_t^T, which is one step of online gradient descent on a regression loss with forgetting factor alpha_t. Test-Time Training layers (Sun et al. 2024, arXiv 2407.04620) make this explicit: hidden state is a model updated by self-supervised gradient steps at test time. Titans (Behrouz et al. 2024, arXiv 2501.00663) learn to memorize with surprise-based updates and weight decay (forgetting gate). Nested Learning (Behrouz et al. 2025, NeurIPS 2025) generalizes to multi-timescale nested optimization with a continuum memory system. Fast weight programmers (Schlag, Irie, Schmidhuber, ICML 2021, arXiv 2102.11174) show linear attention is a delta-rule fast-weight update.
- Consolidation to weights: TTT and Titans update fast weights within a context; they do not consolidate into slow weights across contexts. Nested Learning proposes multi-level updates where higher levels update slower; the paper reports language modeling and long-context gains but no multi-month streaming plasticity evaluation [controlled experiment, LLM; no plasticity metric].
- State-only adaptation horizon: no evidence of how long state-only adaptation substitutes for retraining; likely bounded by context length and the forgetting gate.
- Minimal experiment for sigma-0: hold slow weights fixed; run GDN with (a) fixed gates and (b) learned gates on a 3-month stream; compare rolling NLL against a model with weekly continual pre-training; log gate statistics as drift signals. Implication for sigma-0. Gating implements token-level forgetting; it is not a replacement for weight updates. Use gate statistics (mean alpha_t) as a drift feature. Q7 Time series and finance
- Practice: rolling-window retraining (daily to monthly), exponentially weighted losses, HMM or jump-model regime detection (Nystrup, Kolm, Lindstrom 2020, Journal of Financial Data Science), and online learning (e.g., Hoeffding trees) are common. DeepLOB (Zhang, Zohren, Roberts 2019, arXiv 1808.03668) retrains on rolling windows; TransLOB (Wallbridge 2020) and LOBS5 (Nagy et al. 2023, arXiv 2309.00638) use fixed splits. MarS (Li et al., 2024, arXiv 2409.07486) trains once on large data. No controlled comparison of periodic retraining vs continual pre-training vs online adaptation on LOB data exists.
- TSFMs under shift: Chronos (Ansari et al. 2024, arXiv 2403.07815), TimesFM (Das et al. 2024, arXiv 2310.10688), Moirai (Woo et al. 2024, arXiv 2402.02592), Time-MoE (Shi et al. 2024, arXiv 2409.16040) degrade under distribution shift; test-time adaptation methods such as TAFAS (Kim et al., 2025, arXiv 2501.04970) and OneNet (Wen et al., NeurIPS 2023, arXiv 2309.12659) show gains via online ensembling or partial fine-tuning [controlled experiment, time series].
- Benchmarks: no public event-sliced regime-drift benchmark for finance; the closest are DeepLOB/FI-2010 splits and the "Adaptive Markets" narrative (Lo 2004).
- Industry sources: XTX's Kajaani press release (2025) confirms compute scale but not retraining policy; Two Sigma and Man AHL blogs discuss ensembles and regime-aware models but do not publish retraining cadences. Implication for sigma-0. Define the first controlled LOB comparison of periodic retraining vs continual pre-training vs state-only adaptation; publish an event-sliced protocol. Q8 Evaluation protocol (draft, Experiments-section style) Time slices (NASDAQ, verifiable events): Tick Size Pilot (October 3, 2016 to September 28, 2018); COVID crash (February 24 to April 30, 2020); GameStop episode (January 25 to February 5, 2021); 2022 rate-hike volatility (March 16 to December 14, 2022); SVB collapse (March 9 to 17, 2023); August 5, 2024 volatility spike; April 2025 tariff shock (April 2 to 9, 2025); SEC Rule 612 half-penny tick (adopted September 18, 2024; compliance date November 3, 2025 per SEC). NASDAQ matching-engine changes: none documented as regime-relevant. Metrics: (1) autoregressive NLL per token type; (2) LOB-Bench distributional scores (Wasserstein on returns, spread, queue depth); (3) 30-second and 1-minute mid-price direction IC; (4) plasticity: steps for continued model to reach the fresh model's NLL on each slice, at matched compute (fresh model trained on slice only); (5) forgetting: NLL on pre-slice data after continued training; (6) forward and backward transfer per Lopez-Paz and Ranzato (2017) adapted to NLL deltas. Compute budget: for a 300M model, each slice requires two runs (fresh, continued) of 5 to 10 percent of pre-training compute; 8 slices times 2 runs; stable plasticity estimate needs at least 3 seeds. Curves: NLL and IC versus days since end of training data; stratified by regime label; plasticity ratio (continued/fresh steps) versus continued-training steps; effective rank and weight norm versus steps. Q9 Formalization
- Formal settings: task sequences (GEM, EWC); non-stationary MDPs (Khetarpal et al. 2022 survey, arXiv 2012.13490); drifting distributions with bounded variation (Besbes, Gur, Zeevi 2015); Bayesian latent regimes (BOCPD; switching state-space models).
- Boundary-free drift: variation-budget and Bayesian latent-regime formalisms cover it; task sequences do not.
- Optimal forgetting: under variation budget V_T, optimal window length and step size scale as (T/V_T)^{2/3} and (V_T/T)^{1/3}; under BOCPD the posterior hazard rate determines discounting. Both link to learning rate; gating corresponds to the forgetting factor.
- Proposed formalization for LOB: P_t drifts with sum_t KL(P_{t+1} || P_t) <= V_T; objective is dynamic regret R_T = sum_t L_t(theta_t) minus sum_t min_theta L_t(theta). The optimal GDN gate alpha_t under quadratic loss approximates 1 minus (V_T/T)^{1/3}, i.e., a forgetting factor that decays slower under low variation and faster under high variation. Implication for sigma-0. Estimate V_T from rolling KL between daily LOB token distributions; set learning rate and gate priors accordingly.

1. Table of remedies

```text
| Method                           | Mechanism targeted       | Validated scale/setting      | Cost                | Suitable drift | Ref                          |
| -------------------------------- | ------------------------ | ---------------------------- | ------------------- | -------------- | ---------------------------- |
| Continual backprop               | Dead units, rank         | ImageNet-continual, RL; <10M | Temporary loss      | Both           | Dohare et al. Nature 2024    |
| ReDo                             | Dormant units            | Atari, RL; <5M               | Low                 | Abrupt         | Sokar et al. ICML 2023       |
| Shrink-and-perturb               | Weight norm, warm-start  | CIFAR/ImageNet; <30M         | Low                 | Slow           | Ash & Adams NeurIPS 2020     |
| Plasticity injection             | Capacity                 | Atari; <5M                   | Extra params        | Abrupt         | Nikishin et al. NeurIPS 2023 |
| SR-SPR/BBF resets                | Primacy bias             | Atari; <10M                  | Temporary           | Abrupt         | Schwarzer et al. ICML 2023   |
| L2-toward-init                   | Rank, norm               | RL, continual regression     | Slight ceiling loss | Slow           | Kumar et al. 2023            |
| Wasserstein-to-init              | Curvature                | RL, CIFAR                    | Low                 | Slow           | Lewandowski et al. 2023      |
| Effective-LR control (norm + WD) | Norm growth              | RL; validated with LayerNorm | Low                 | Both           | Lyle et al. 2024             |
| CReLU/LayerNorm                  | Dead units               | RL                           | Low                 | Both           | Abbas et al. 2023            |
| Hare and Tortoise                | Stability/plasticity     | RL                           | Extra params        | Abrupt/slow    | Lee et al. ICML 2024         |
| Neuroplastic Expansion           | Capacity                 | RL                           | Extra params        | Abrupt         | Liu et al. 2024              |
| UPGD                             | Utility perturbation     | Continual regression, MuJoCo | Low                 | Slow           | Elsayed & Mahmood ICLR 2024  |
| Muon-type optimizer              | Spectral control         | LLM pre-training up to 15B   | Low                 | Slow           | Liu et al. 2025 (Moonshot)   |
| Rewarm + replay                  | Forgetting, effective LR | LLM 405M and 10B             | Low                 | Both           | Ibrahim et al. TMLR 2024     |
```

1. Table of mechanisms

```text
| Mechanism                         | Evidence level               | Key refs                    |
| --------------------------------- | ---------------------------- | --------------------------- |
| Dead/saturated units              | Controlled experiment (RL)   | Sokar 2023; Abbas 2023      |
| Rank collapse                     | Controlled (RL, supervised)  | Lyle 2023; Kumar 2021       |
| Weight-norm growth / effective LR | Controlled                   | Lyle 2024; Ash & Adams 2020 |
| Curvature loss                    | Theory (linear) + controlled | Lewandowski 2023            |
| Lazy/NTK drift                    | Theoretical association      | Lyle 2023; Mirzadeh 2022    |
| Gradient diversity loss           | Correlational                | Dohare 2024                 |
| Stale optimizer state             | Correlational                | Lyle 2024                   |
```

1. Verdicts on hypotheses

- H1 Supported. Plasticity loss is documented primarily in from-scratch RL/supervised streams; LLM continual pre-training shows forgetting, not dead networks (Ibrahim 2024; Gupta 2023; Kotha 2024). Deciding evidence: rewarm+replay matches from-scratch on 10B models.
- H2 Undetermined (plausible). No LOB-specific evidence; theory (variation budget vs BOCPD) supports separate mechanisms, but no empirical LOB decomposition exists.
- H3 Supported (indirectly). TTT/Titans/Nested Learning do not consolidate to slow weights; no evidence of multi-month state-only adaptation; gating's forgetting factor limits horizon.
- H4 Supported. Muon-type and L2-init have low cost and preserve plasticity in controlled settings; reset-based methods incur temporary loss and are untested on pre-trained models.
- H5 Supported. Curvature, effective-LR and rank mechanisms are established; no unified theory or death-time predictor.

1. Draft evaluation protocol See Q8. Additional specification: use a base pre-training window ending December 31, 2015 for the tick pilot slice and ending December 31, 2019 for the COVID slice; continued models use rewarm to 50 percent of peak LR with 10 percent replay; fresh models train from random init on the slice with equal tokens; report plasticity ratio and NLL delta with 95 percent CIs across 3 seeds; report per-slice LOB-Bench Wasserstein distances; log effective rank, dormant units, and weight norm every 1000 steps.
2. Open problems

- No unified theory linking curvature, effective LR and rank collapse.
- No plasticity measurement at 100M+ sequence models in continual pre-training.
- No LOB or finance benchmark with event-sliced regime evaluation.
- Optimal gate vs learning rate link under variation budget is theoretical only.
- Multi-year drift effects on scale-based protection unknown.
- Consolidation from fast to slow weights in TTT/Nested Learning untested.

1. Recommended next experiments for sigma-0 (ordered by information gain per GPU hour)
2. Plasticity ratio measurement on two slices (COVID, August 2024) with continued vs fresh 300M model; 3 seeds.
3. Rewarm plus replay sweep (5, 10, 25 percent replay) with L2-init versus AdamW; measure forgetting and plasticity.
4. Muon vs AdamW continual pre-training on one slice; measure effective rank and NLL.
5. BOCPD-modulated learning rate versus constant LR on a 3-month stream.
6. GDN gate statistics as drift detector; correlate mean alpha_t with realized KL between daily distributions.
7. Unverified items

- Exact arXiv id for Kumar, Marklund, Van Roy 2023 (stated 2308.11958; not verified).
- SEC Rule 612 compliance date November 3, 2025 (from SEC adopting release summary; not cross-checked with final rule text).
- Muon plasticity benefits in continual settings (no dedicated study found).
- Nested Learning NeurIPS 2025 venue and long-stream plasticity results (not found in fetched sources).
- TAFAS arXiv id (2501.04970; not verified).
- Neuroplastic Expansion arXiv id (2410.07994; not verified).
- Time-MoE arXiv id (2409.16040; not verified).
- NASDAQ matching-engine regime changes (none documented; searched).

<aside>

🧭
一句话：XTX Markets 研究总监、UT Austin 副教授 Zhangyang "Atlas" Wang 在 2026 年两次公开讨论（3 月 12 日 Harvard AI in Finance 讲座、一个月后 Asian American Scholar Forum 的 AIX Summit 金融 AI 圆桌）中给出一组「反热潮」判断：AI 会先自动化金融里结构化、重复性的工作并压缩人才金字塔，会把顶尖与平庸研究者的差距拉得更大，会让交易信号的半衰期缩短约一个数量级；但金融很可能不会复刻 LLM 的 Scaling 路线（容量增加往往只是更精确地拟合噪声，瓶颈是市场每单位时间能产生多少真实信号），最持久的 edge 是 Infrastructure 而非 Data 或 Model，AI 无法凭空创造 Alpha，而多个相似模型同时失效（Correlated Model Failure）可能是下一次系统性事件的诱因。

</aside>
来源信息
* 原帖：小红书（rednote）视频笔记「XTX 研究总监：AI 如何重塑金融与交易」，作者 狄奥尼索斯之歌，发布于 2026 年 8 月 26 日，原帖链接（笔记 id 6a8eeab30000000006013611）。话题标签：ai、量化交易。抓取时互动数：点赞 19、收藏 29、评论 1、分享 3。
* 内容形式：19 张图（前 11 张为竖版长截图，后 8 张为横版截图）+ 1 段 38 秒视频。视频只是同一组 19 张图的配乐轮播，没有额外内容。
* 图片本身是一篇中文长文的截图：作者把 Atlas Wang 两次英文公开讨论整理成第一人称中文文稿（编者按注明仅按中文阅读习惯调整句式与段落、不改变原有观点；原文出处标注为 Harvard 与 AIX Summit 两处）。
* 讲者：Zhangyang "Atlas" Wang，XTX Markets 研究总监，负责纽约的 XTY Labs；UT Austin ECE 终身副教授（目前 on leave）；VITA 组 PI。研究横跨压缩感知、稀疏优化、Foundation Model 与神经符号 AI。
* 文稿结构：编者按 → 讲者简介 → 两次讨论的范围说明 → TL;DR 八条核心观点 → 演讲精编第一部分（Harvard，五个问答）→ 第二部分（AIX Summit 圆桌，五个问答）。圆桌由 Juhua Zhu 主持，同场嘉宾 Andrew Lo、Argyro Tasitsiomi、David Wu、Yin Luo。
<aside>

⚠️
关于「一字不落」的说明：19 张原图已全部嵌入子页面 1，原文逐字内容请直接看图（原图即原文）。本页及各子页面中的文字均为整理转述、交叉验证与分析，不是原文的逐字转录（整篇转载他人文章是我这边的硬性限制）。如需纯文本原文，建议在原帖长按复制或对原图做 OCR，再贴进子页面 1 预留的位置。

</aside>
核心观点速览（八条，转述）
1. 自动化顺序：先是结构化、重复性、依赖模式识别的工作（回测、信号生成、执行优化、初级分析），入门岗位最先；真正新颖的假设、未见情境的判断、高风险决策里的责任与信任仍靠人。
2. 差距放大：工具人人可用后，差异来自问题选择、数学品味、判断力和识别错误的能力；优秀者被放大，平庸者更快产出「看起来正确」的错误研究。
3. 金融市场是 Continual Learning 最典型的现实环境：世界不会标注 Task Switch 或 Regime Shift，只会持续漂移；理想系统应自主检测分布变化并调节学习过程。
4. Continual Learning 更深的问题是可塑性丧失（Wang 称之为 AI 的「老化问题」）：神经元休眠、表征坍缩、梯度失效，最后既学不进新的也忘不掉旧的。
5. AI 让已有信号更快失效：信号半衰期被压缩约一个数量级；市场平均更有效但更脆弱；新机会出现在不同资产、不同时间尺度、不同约束参与者之间的「接缝处」。
6. Infrastructure 是最持久的 edge：数据有半衰期、模型快速商品化，只有算力、低延迟执行栈和「承受大量失败实验」的能力会复利。
7. 金融很可能不复刻 LLM 的 Scaling 路线：语言有稳定、可压缩的结构，市场只有微弱的预测相关性；增加容量往往只是更精确地拟合噪声，瓶颈在信号本身的稀缺。
8. AI 无法凭空创造 Alpha，且可能带来新的系统性风险：多个机构用相似架构与重叠数据，可能在同一时刻同一方向犯错（Correlated Model Failure）。
与我项目的关系（简版，详见子页面 11）
* XTX 在做的事（面向金融时间序列的大规模模型、私有市场数据 + GPU 基础设施）与 $$ \sigma_0 $$/LOBS5（NASDAQ 订单流上的 LOB foundation model）是同一条赛道，这是方向上的外部确认。
* Wang 的「容量拟合噪声」警告与他 2025 年学术报告里「在极噪声数据上 scaling 仍有效」的说法看似矛盾，其实可以用「horizon 决定信噪比」和「loss 中可压缩的市场机制 vs 稀缺的收益可预测性」来调和；这正是我的 scaling law 实验（34M 到 617M+）最能回答的问题。建议把 loss 分解成不可约噪声底 $$ E $$ 与可学习部分，并分别用下游指标验证。
* 架构上，Wang 认为 SSM（含 Mamba、DeltaNet）本质仍是一维递归、缺少几何归纳偏置；对 LOB 而言，价格档位结构与跨资产关系恰好是「被 tokenization 强行线性化的拓扑」，这给 GDN 混合架构和 tokenization taxonomy 工作提供了一个新的论证角度。
* 产品角度上，回测/仿真系统正好对应 Wang 批评的「回测假设未来同分布」问题，可以把「regime drift 下的稳健评估」做成卖点；短 horizon 预测产品要明确报告 IC 随 horizon 的衰减。
* 职业上，XTY Labs 的 AI Residency 与 Wang 给学生的建议（投资于品味与判断而非工具熟练度）与我当前的路径高度匹配。
目录（12 个子页面，每页回答一个问题）
1. 原帖完整存档：19 张原图 + 38 秒视频 + 逐图导读
2. 人物与机构：Atlas Wang、XTX Markets、XTY Labs 分别是什么？
3. AI 会取代哪些金融工作？人才金字塔如何被压缩？学术界与产业界怎么选？
4. 架构层面：LLM 的一维序列范式为何难以理解拓扑？SSM 的边界在哪？只有 World Model 够不够？
5. Continual Learning 缺什么：可塑性丧失（AI 老化）与无标注的 Regime Drift
6. 市场微观结构：AI 是增加了无效，还是消灭了无效？信号半衰期为何缩短一个数量级？
7. 金融为什么很可能不复刻 LLM 的 Scaling Law？端到端 AI Trading 为何回测好、实盘差？
8. Data、Model、Infrastructure 哪个是持久 edge？Correlated Model Failure 为何是下一次系统性风险？
9. AI 会缩小差距还是放大最优秀的人？给研究者和学生的建议
10. 批判性评估：Wang 的哪些主张有证据支持，哪些是观点，哪里与他自己的其他公开表述有张力？
11. 与我的项目的关系：σ0/LOBS5、Scaling Law、GDN、产品角度、职业
12. 术语表、开放问题与延伸阅读
1 原帖完整存档：19 张原图 + 38 秒视频 + 逐图导读
2 人物与机构：Atlas Wang、XTX Markets、XTY Labs 分别是什么？
3 AI 会取代哪些金融工作？人才金字塔如何被压缩？学术界与产业界怎么选？
4 架构层面：LLM 的一维序列范式为何难以理解拓扑？SSM 的边界在哪？只有 World Model 够不够？
5 Continual Learning 缺什么：可塑性丧失（AI 老化）与无标注的 Regime Drift
6 市场微观结构：AI 是增加了无效，还是消灭了无效？信号半衰期为何缩短一个数量级？
7 金融为什么很可能不复刻 LLM 的 Scaling Law？端到端 AI Trading 为何回测好、实盘差？
8 Data、Model、Infrastructure 哪个是持久 edge？Correlated Model Failure 为何是下一次系统性风险？
9 AI 会缩小差距还是放大最优秀的人？给研究者和学生的建议
10 批判性评估：Wang 的哪些主张有证据支持，哪些是观点，哪里与他自己的其他公开表述有张力？
11 与我的项目的关系：σ0/LOBS5、Scaling Law、GDN、产品角度、职业
12 术语表、开放问题与延伸阅读

Zhangyang "Atlas" Wang（中文名通常写作 汪张扬，以本人公开资料为准）

- 身份：XTX Markets 研究总监（Research Director），负责纽约的 XTY Labs；UT Austin Chandra Family ECE 系终身副教授（Temple Foundation Endowed Faculty Fellowship），同时隶属 UT CS 与 Oden Institute；VITA 组 PI。据公开资料，他 2024 年起在 UT Austin on leave，全职在 XTX。
- 学术背景：UIUC ECE 博士（2016，导师 Thomas S. Huang）。曾任 Picsart 的 AI 研究与技术总监（2022 到 2024）、Amazon 访问研究员（2021 到 2022）。奖项包括 NSF CAREER Award、ARO Young Investigator Award、IEEE AI's 10 to Watch。
- 研究主线（他自称「a low-dimensional lens on intelligence」）：稀疏性与剪枝、Lottery Ticket Hypothesis、高效 LLM 训练与推理（如 GaLore 这类低秩梯度投影的显存高效优化）、模型压缩、神经网络理论、生成式 AI；在 XTX 则聚焦金融时间序列与市场数据的大规模基础模型。
- 近两年公开报告（用于交叉验证本帖内容）：2025 年 3 月 25 日 Princeton CSML 报告「Cracking the Market Code: Building Large Foundation Models for High-Frequency Trading」；2025 年 10 月 7 日 University of Minnesota CSE DSI 报告「Algorithmic Trading with Large-Scale Deep Learning」（Stony Brook AI Institute、Rutgers、NYU CDS 亦有同题报告）；播客 The Information Bottleneck EP19「AI in Finance and Symbolic AI」。本帖来源则是 2026 年 3 月 12 日 Harvard AI in Finance 讲座与一个月后的 AIX Summit 圆桌。
- 他在 2025 年报告摘要里的核心主张（转述）：高频市场是半结构化数据（时间序列、表格、图）上基础模型的试验场；在极噪声数据上扩大模型规模仍有帮助；学习到的特征比手工特征更稳健；严格因果性避免了现代 benchmark 常见的 train/test 污染，因而市场是度量「真实智能进步」的难得环境。这与本帖 2026 年的说法之间的张力见子页面 7 与 10。 XTX Markets
- 2015 年 1 月由 Alex Gerko 创立（从 GSA Capital 拆分），总部伦敦；名字来自线性回归中的 $X^\top X$。做市与算法交易，覆盖股票、外汇、固定收益、商品、加密资产，接入 80 多个交易场所。公开口径：每日成交超过 2500 亿美元，覆盖 35 个国家（Wang 报告摘要里的说法是每日超过 3000 亿美元、覆盖数万个金融工具、完全自动化、无人工自由裁量）；员工 250 人以上（2024 年数据）；自有资本、无外部投资人，Gerko 持股约 75%。
- 算力与基础设施：据 XTX 2025 年 1 月 22 日关于芬兰 Kajaani 数据中心的公告，其研究集群拥有超过 25,000 张 GPU 与 650 PB 可用存储；计划投资超过 10 亿欧元建设 Kajaani 数据中心园区（首期 15,000 平方米、22.5 MW，2026 年建成）。这直接对应本帖第 6 条「Infrastructure 是最持久的 edge」。
- 研究文化的公开信号：2023 年 11 月发起 1000 万美元 AIMO Prize（AI 数学奥林匹克，顾问包括 Terence Tao、Timothy Gowers）；2024 年 12 月再投入约 920 万美元资助开源数学 AI。Wang 的神经符号 AI 兴趣与此一脉相承。
- 招聘口径里反复出现的关键词：统计深度（正则化、分布偏移、平稳性）、post-mortem 式的实验叙事（模型、预期、错在哪、学到什么、怎么修）、工程岗的重 C++。 XTY Labs
- XTX 在纽约设立的机器学习部门，2024 年 2 月 22 日公开宣布成立，由 Wang 领导。定位是把学术级 ML 研究引入做市业务，运行 6 到 12 个月的 AI Residency Program，组织扁平，研究成果进入伦敦核心量化团队。
- 「Research Director」在这里的含义：领导面向交易的 ML 研究议程（数据、模型、评估、基础设施），而不是管理某一条交易线。 数字口径提醒 不同来源对 XTX 日成交额（2500 亿 vs 3000 亿美元）、员工数与 GPU 数的说法随时间变化；以上均注明了出处年份。这些数字来自之前的 Deep Research（Wikipedia、MarketsWiki、XTX 官方公告、PRNewswire、各高校报告页），引用前建议回到一手来源核对。

Wang 最关心的两个问题（转述，对应图 9 到图 10）

1. 领域过度关注遗忘，更深的问题是模型逐渐陈旧并永久失去可塑性，他称之为 AI 的「老化问题」。几乎所有 Continual RL 方法都站在稳定性与可塑性困境（Stability vs Plasticity Dilemma）的稳定一侧，如 Elastic Weight Consolidation、经验回放、Progressive Networks，主要在防灾难性遗忘。但实验显示网络不只会遗忘，还会僵化：训练推进时神经元休眠、表征坍缩、梯度失去意义、memory buffer 过满；即使移除全部防遗忘约束，网络也既学不进新的也忘不掉旧的，只是「死了」。灾难性遗忘是网络活着时才有的下游症状，可塑性丧失会杀死 AI；目前既没有解释这一过程的机制理论，也没有原则性的解法。
2. 真实环境不会弹出「任务已切换」的标记。世界在无标注下持续漂移，Task 或 Regime 是人为定义的概念，金融市场是最好的例子；更深的问题是 Continual Learning 里连「任务」都没有清晰定义。更进一步，也许正确答案是停止用 Task 思考，转而构建不断检测分布信号并据此调节自身学习过程的策略，不必先把变化定义成某项任务、某个事件、某种情境或一次 regime 切换。 交叉验证与延伸

- 可塑性丧失有一支正在快速增长的文献：Dohare 等人在 Nature 2024 发表的「Loss of plasticity in deep continual learning」（Sutton 组，提出 continual backpropagation，通过持续重置低效用单元维持可塑性）；Sokar 等人 ICML 2023 的 dormant neuron 现象；Nikishin 等人 ICML 2022 的 primacy bias；Lyle 等人 2023 年关于「理解神经网络可塑性」的分析。Wang 的表述与这支文献一致，但他把它推到「没有机制理论」这一更强的判断（目前确实只有现象层面的诊断指标，如有效秩、活跃单元比例、梯度范数，和重置类启发式解法）。
- EWC（Kirkpatrick 等 2017）、经验回放、Progressive Networks（Rusu 等 2016）确实都属于「稳定性一侧」的方法，这一分类是准确的。
- 「Task 没有清晰定义」与 task-free continual learning、online continual learning 的文献呼应；在金融里对应的是 regime detection（隐马尔可夫、变点检测、在线 Bayesian 变点检测）与自适应学习率、自适应窗口等方法。
- 数学上可以把 Wang 的第二点写成：不假设离散的 regime 变量 $z_t$（取值于有限集合），而是让学习过程本身依赖一个连续的漂移统计量 $s_t = D(P_{t-w:t},\, P_{t-2w:t-w})$（某种分布距离），学习率或记忆写入强度写成 $\eta_t = f(s_t)$。这与 GDN 的 gating（决定擦除多少旧状态）在形式上是同一个问题的两个尺度：token 级 vs 训练级。
- wiki 内相关词条：#24（持续学习为何仍是最大难题）、#27（RL 是否通往持续学习）、#58（仅靠上下文做不出真正的持续学习）、#86（LoRA 作为经验内化层）、#253（预训练 VLA 天生抗灾难性遗忘）。Wang 的结论与 #58 的立场一致：在 context 里堆信息不等于持续学习，因为可塑性是权重层面的问题。 对我而言的要点
- 部署一个预训练 LOB 模型意味着必须回答「模型随时间陈旧」的问题：评估应按时间切分（训练期之后的年份、含 2020 年 3 月、2024 年 8 月等高波动时段），报告性能随「距训练集末尾时间」的衰减曲线。
- Test-time training 与 continual pretraining 是两条候选路线（我之前关心过 TTT），两者都要面对可塑性丧失；continual backprop 类方法值得作为 baseline。
- 把「漂移统计量驱动学习率」做成一个小实验：在 sigma-0 的逐日流式微调里，用验证 NLL 的滑动窗口变化作为 $s_t$，对比固定学习率与 $\eta_t = f(s_t)$ 在高波动月份的表现。

<aside>

🔎
这一页已复制为独立的深研问题清单（原内容 + 9 个主题的问题、假设 H1 到 H5、决策接口、Deep Research prompt）：322 🔎 Continual Learning 深研问题清单：可塑性丧失 × Regime Drift × LOB 基础模型（复制自 #321 子页 5 + 新增问题准备 · 供独立 Deep Research 用）

</aside>

---

## Grok

没有公开配方。能推断的是 **他会买哪些能力、不会买哪些论文里的方法**。按「XTX 能养得起、和公开判断自洽」来排，不是猜仓库里的类名。

## 他几乎肯定有的（工程层，不是论文层）

**日更/周更的续训管道，不是 tick 级改核心权重。**

公开口径是基础设施复利、承受大量失败实验。这对应：冻 checkpoint、夜间用新订单流做 CPT（rewarm + 一部分旧数据 replay）、早上换线。和「推理时改 θ」不是同一件事。Wang 自己把可塑性放在权重层，不会把 GDN 门控当成续训。

**按日历和事件切的评估，不是 task 标签。**

讲座第二点就是市场不弹 regime 标签。内部一定有：距训练集末日的衰减曲线、2020-03 / 2024-08 这类切片、新旧窗 NLL 一起看。没有这个，25k GPU 也只是在拟合噪声。

**大模型教师 + 小模型执行。**

做市要延迟。Foundation 模型会用来出表征、仿真、研究信号；真正挂在执行路径上的是蒸馏过的小网或线性头。这和你们讨论的 OPD/DMD 是同一形状：教师更新慢，学生更新快、对齐教师分布而不是抄轨迹。

**多模型、强相关监控。**

第 8 条就是 Correlated Model Failure。内部技术不会是「一个 σ0 用十年」，而是一簇模型 + 何时一起失效的统计（同期回撤、同方向误差、共享数据切片）。这比 EWC 更像风险管理。

**tokenization / 盘口几何上的私有工作。**

他公开嫌 SSM 还是一维递归、缺几何归纳。LOB 的档位、队列、跨品种关系被编成线性 token 之后，信息已经丢了一层。XTY 若认真做 foundation model，技术预算会砸在「盘口怎么编」而不是再堆一层 Mamba。你们 size 的 base-100 断崖属于这一层；他那边几乎不可能还在用会进位跳词表的朴素数字编码。

## 他可能有、但会改形态的（研究层）

**谱约束优化，而不是 Nature 那套整网 reset。**

背景是稀疏、低秩、GaLore。Muon / 谱范数 / 向初始化拉的 L2，和「容量用来拟合噪声」的担忧一致：更新不要把有效秩打崩。Continual backprop、ReDo 整层重置，在 10B 级预训练权重上太贵，最多当诊断（dormant 比例高了再局部打）。

**漂移统计量调学习率，不调成离散 regime 分类器。**

他明确说停止用 Task 思考。更像：日度 NLL 或滚动 KL 当 $s_t$，$\eta_t=f(s_t)$，replay 比例跟着走。HMM「现在是哪一态」最多是分析工具，不会当训练开关。

**仿真器当第二损失，不只是下一 token。**

2025 报告强调严格因果、市场当真实智能的度量。内部很可能会用生成出来的消息流跑撮合，看价差/队列/收益分布（你们 LOB-Bench 那一格），而不是只报 token NLL。否则 scaling 只是把噪声记更清。

**神经符号约束（轻量）。**

AIMO、神经符号是公开兴趣。落到交易不会是「定理证明下单」，更可能是：硬约束（价格网格、队列守恒、不可穿越的因果），生成后再过滤或当辅助损失。和 foundation model 并列，不是替代。

## 他公开立场下几乎不会当主力的

| 学术热词 | 为何不像 XTX 主力 |
| --- | --- |
| EWC / 经验回放当 CL 全部 | 他自己划在「稳定一侧」，防遗忘不防老化 |
| 服务时 TTT 写核心权重 | 延迟、可复现、合规都炸；TTT 最多当上下文里的快权重 |
| 单一世界模型替代执行模型 | 他问过「只有 world model 够不够」；做市仍要短 horizon 决策头 |
| 纯 LLM scaling 路线 | 第 7 条写明了：容量更容易拟合噪声 |
| 无验证器的 GRPO 当主循环 | LOB 没有 math 那种 checker；RL 最多在仿真器/偏好上后训练 |

## 和你们两份深研稿的错位

第一份执行摘要写「300M–1B 很少死，主要是忘和漂」——和他 2026 的「老化会杀死 AI」比偏软。Zyphra 那篇已经证明平稳预训练也会丢可塑性，只是 onset 被 scale 推迟。他若在 XTX 看 10B×多年私有 tick，更可能站 Zyphra + 自己那侧：**会老化，只是死得慢**，所以管道里要有可塑性探针，不只 replay。

你们 compound error（头 500 条 size/Δt 滑进错误定态）他未必用这个名字，但会落在同一类「生成闭环是否还像市场」的评估里，而且会归到编码/几何，不会先上 DMD。

## 一句话

他手里的技术大概是：**私有订单流上的续训集群 + 事件切片评估 + 教师慢学生快 + 盘口怎么编 + 谱上比较老实的优化 + 漂移调 $\eta$ + 模型相关性监控**。不会是论文清单上的 CBP/ReDo/TTT 原装。σ0 能对上的不是「也做 Mamba」，是把 **编码几何、早/晚检查点可塑性、新旧窗 NLL、生成定态** 做成可复现的公开协议——这些恰好是他能做、但不会发表的那一层。

---

## User

```text
| 单一世界模型替代执行模型 | 他问过「只有 world model 够不够」；做市仍要短 horizon 决策头 |
| ------------ | ----------------------------------------- |
```

---

## Grok

这一行是在拆两件被经常焊在一起的事：**世界模型**和**下单策略**。

## 世界模型在这一行里指什么

一个能 **闭环滚订单流** 的生成器：条件历史消息，吐出后面的 type / 方向 / 价 / 量 / Δt。你们 σ0、LOB-Bench、compound error 测的都是这一层——「下一步市场长什么样」。

它回答的是 $p(\text{未来消息}\mid\text{过去})$。

不回答 $a_t=\pi(\text{状态})$：这一瞬间挂多少、撤哪一档、要不要收紧价差。

## 「只有 world model 够不够」在问什么

公开讨论里，Wang 把「端到端一个大序列模型」和「对市场结构有几何/因果约束的模型」分开。推到交易上就是：

- 把 book 编成一维 token、用 SSM/Transformer 学下一个 token，可以得到一个不错的模拟器；
- 做市的 edge 在 毫秒到秒的决策 + 约束 + 库存，这些不会自动从「会滚消息」里长出来。

所以「够不够」不是世界模型无用，而是：**仿真器合格 ≠ 执行器合格**。回测好看、实盘差，经常就是用同一个网既当世界又当策略，在历史 $p$ 上拟合 $\pi$，换制度或延迟以后 $p$ 一偏，$\pi$ 一起崩。

## 为什么还要短 horizon 决策头

做市动作的价值函数几乎是短的：队列位置、当下价差、库存偏斜、接下来几十到几百毫秒的到达。世界模型可以提供：

- 特征（对未来到达/撤单的预测）
- 反事实（若我挂在这里，生成的后续流怎么走）
- 训练数据（在仿真里打策略）

真正下单仍是一个 **浅的、可审计的头**：线性或小 MLP，输入这些预测 + 库存 + 风控硬约束。换头、换约束、换延迟假设，不必重训 300M 生成器。

这也对上他的另一条：模型商品化快，基础设施和「能承受失败实验」才复利——失败的应是决策头和执行栈，不是每次把世界模型推倒重来。

## 和你们两条线的接法

- Compound error 测的是世界模型闭环是否滑进错误定态，还没碰到 $\pi$。
- Continual pre-training / 可塑性测的是世界模型的慢权重还能不能跟上新切片，同样还没碰到 $\pi$。

「单一世界模型替代执行模型」想避免的，是第三步把 NLL 最小的生成器直接当策略。对 σ0：生成器继续做生成与仿真；执行另挂短 horizon 头；用生成器当环境去训或蒸馏这个头，而不是让同一个 AR 既滚市场又出价。

---

## User

我需要continual learning相关的

---

## Grok

Continual learning 在你们这条线上要拆成 **两个对象、两套度量**，不要共用一套 EWC/replay 故事。

## 两个对象

|  | 世界模型 $\sigma_0$ | 短 horizon 决策头 $\pi$ |
| --- | --- | --- |
| 学的是 | $p(\text{下一条消息}\mid\text{历史})$ | $a_t\mid\text{特征, 库存, 约束}$ |
| 漂的是 | 到达/撤单/size/Δt 的生成分布 | 哪些预测还能换成钱 |
| 忘了会怎样 | 旧制度下的消息流滚不像 | 旧制度下的报价还在，但期望收益变号 |
| 死了会怎样 | 新切片 NLL 降不动（可塑性） | 再怎么训头也抬不起 IC |
| 更新频率 | 日/周 CPT | 可以更快，甚至按会话重拟合 |

Wang 的「老化」主要打在左列：慢权重僵了，生成器既学不进 2024-08，也回不到 2019。右列更常是 **遗忘和过拟合切片**，很少「整头死掉」。

## 世界模型上的 CL（主战场）

要同时报三件事，缺一不可：

1. 新切片适应：早检查点 vs 晚检查点，同一预算贴上 COVID / 2024-08 的 NLL（可塑性；不要用 scratch 当「ABSENT」标准）。
2. 旧窗保留：续训后 2018–2019 的 NLL（遗忘）。
3. 闭环定态：头 500 条生成里 size / log10_dt 的 $E(m)$（这是生成器自己的 drift，不是日历 CPT）。

配方按成本：

- 基线：rewarm（峰值 LR 的 30–50%）+ 5–25% 旧流 replay。
- 漂移调 $\eta$：日度验证 NLL 或滚动 KL 当 $s_t$，不要先训练一个 regime 分类器。
- 优化：AdamW + 偏高 weight decay；有余力再上谱约束 / L2-toward-init。
- Reset / ReDo：只当 dormant 或探针 AUC 恶化时打局部，不当默认。
- GDN $\alpha_t$：当漂移特征，不当「不用更新 θ」的理由。

不要把 TTT/Titans 的快权重写成 CL。那是一次前向里的状态，跨日不巩固进 $\theta$。

## 决策头上的 CL（次战场，但更像交易）

头小、数据是「预测 → 动作 → 短窗盈亏」。这里用世界模型那套整网 CPT 是杀鸡。

更贴的是：

- 冻 $\sigma_0$，只续训头（默认）。世界模型周更，头可以日更。
- 切片失效时 重拟合头，不要先动 300M。
- 若头也要跨制度保旧行为：旧制度上的动作分布当教师，做一层小的蒸馏（OPD 形状，不是 DMD 分数场）。
- 有仿真器时才上 RL；奖励必须来自撮合后的短窗 PnL / 库存罚，没有 checker 就不要 GRPO。

遗忘在这里的定义换成：旧事件切片上的 IC / 价差捕获，不是 token NLL。

## 无标签漂移（Wang 第二问）

两边共用同一个 $s_t$，动作不同：

$$
s_t = D\big(\hat P_{t-w:t},\;\hat P_{t-2w:t-w}\big)
$$

- $s_t$ 中等：加大 $\eta$、略增新数据比例（世界模型 CPT）。
- $s_t$ 很大：先重训头 + 提高 replay；世界模型短预算探针，看晚检查点还能不能降 NLL。
- $s_t$ 来自生成定态（size 边缘 $E(440)$ 升）：先查编码和短闭环，不是先加 EWC。

不要把 $s_t$ 离散化成「现在是 COVID 态」。那正是他要停的 Task 思维。

## 最小实验序（只保留 CL 信息量大的）

1. $\sigma_0$：2019 检查点 vs 训到 2024-07 的检查点，固定预算适应 2024-08，报 AUC / $R_{\mathrm{steps}}$ + 旧窗 NLL。
2. 同一对检查点上跑 A01：闭环 500 条的 size/Δt 的 slope+level（生成器 CL ≠ CPT CL）。
3. 冻两个世界模型，只训同一决策头，看 IC 随「距世界模型训练集末日」怎么掉。
4. 有 1 的结论再扫 replay × rewarm；有 3 的结论再决定头要不要自己做 replay。

一句话：continual learning 对 σ0 是「慢权重还能不能跟上无标签漂移」；对执行头是「世界一偏，浅策略还能不能重拟合」。先测左列的可塑性与生成定态，再决定右列更不更新世界模型、只换头。

---

## User

cpt 要怎么做?

---

## Grok

σ0 的 CPT 就是：**冻一份旧检查点当起点，混一点旧订单流，把学习率重新拉起来，在新时间窗上继续做同一件 SSL/NLL**。不是换任务，也不是在线改核心权重。

## 一次 CPT 跑什么

固定四样东西，只改「从哪天接到哪天」：

- 目标：还是下一 token / 下一字段 NLL，不要临时改成 IC 或 RL。
- 数据：新窗订单流为主 + 旧窗 replay。
- 起点：$\theta_{\mathrm{old}}$（例如训到 2019-12 或 2024-07-31）。
- 预算：token 数、batch、步数与「从头训同样新数据」对齐，否则没法谈可塑性。

## 数据怎么配

$$
\mathcal{D}_{\mathrm{cpt}}
=\underbrace{(1-\rho)\,\mathcal{D}_{\mathrm{new}}}_{\text{新月份 / 新切片}}
+\underbrace{\rho\,\mathcal{D}_{\mathrm{old}}}_{\text{replay}}
$$

- 起步 $\rho=0.10$，再扫 $\{0.05,0.10,0.25\}$。
- 旧数据按时间分层抽，不要只抽最近一个月（那叫滑窗微调，旧制度会蒸发）。
- 新窗按交易日打乱到 batch 里，但 评估必须按日历序，训练打乱只为稳梯度。
- 2020-03 和 2024-08 不要和「普通 2023」平均进同一个 replay 桶还当同一分布。突变切片单独当一段 CPT，或临时把 $\rho$ 提到 0.2–0.25、peak LR 略升。

Wang / Ibrahim 那套：replay 是压 **旧窗 NLL 升多少**；$\rho$ 太低旧的掉，太高新的学不动。用短试点拟合 CPT 律里的 shift 项，再锁生产 $\rho$。

## 学习率：必须 rewarm

余弦已经退火到接近 0 的检查点，直接低 LR 续训，有效步长不够，看起来像「学不动」（假老化）。

一套能跑的：

1. 短 warmup（新阶段步数的 2–5%）拉到$\eta_{\mathrm{peak}}^{\mathrm{cpt}} = 0.3\sim 0.5\,\eta_{\mathrm{peak}}^{\mathrm{pt}}$。
2. 再余弦或 WSD 降下来，末段不要降到 0（Ibrahim 的 infinite / 非零尾，避免平台死掉）。
3. 每个 CPT 阶段 重置 Adam 动量（或至少缩小 $\beta_2$ 的历史），避免旧矩在新分布上指错方向。
4. weight decay 保持预训练量级或略增（可塑性侧最便宜的旋钮）。

漂移大时把 peak 往 0.5 靠；平稳续年份用 0.3。不要每根 tick 改 $\eta$。若要自适应：用 **日度验证 NLL 或两周滚动 KL** 当 $s_t$，只在阶段之间调 peak / $\rho$。

## 优化器与正则

第一版 CPT 用 **AdamW + 现有 WD**，先把 replay×rewarm 测完。

Muon / 谱正则 / L2-toward-init 是第二刀，用来保有效秩，不是 CPT 能不能跑的前提。

不要默认整网 ReDo；dormant 升了再局部打。

GDN 的 $\alpha_t$ 继续当模型内部遗忘，**不代替这次 $\theta$ 更新**。

## 一次阶段怎么切

按年或按事件，不要按「任务 id」。

例：

```text
θ0  : 2015–2019 预训练
CPT1: 2020 全年（含 COVID 切片单独报）
CPT2: 2021–2022
CPT3: 2023–2024-07
探针: 2024-08 固定预算适应（可塑性）
```

每一段结束存教师 $T_k=\theta_k$。下一段可选加一项轻的匹配：

$$
\mathcal{L}
=\mathcal{L}_{\mathrm{NLL}}(\theta;D_{\mathrm{cpt}})
+\lambda\,\mathrm{KL}\big(p_\theta(\cdot\mid x_{\mathrm{on-policy}})\,\|\,p_{T_{k}}(\cdot\mid x)\big)
$$

$\lambda$ 先从 0 做起；旧窗掉得凶再开。这是稳定性，不是可塑性。

## 每段必须报的数

| 数 | 含义 | 用来判 |
| --- | --- | --- |
| 新窗 val NLL | 适应 | CPT 有没有吃进新流 |
| 旧窗 val NLL | 遗忘 | replay 够不够 |
| 探针 AUC / $R_{\mathrm{steps}}$（晚 vs 早检查点，同预算） | 可塑性 | 网络有没有僵 |
| size / Δt 的 $E(440)$、slope | 生成定态 | 闭环有没有滑到错误吸引子 |
| $\\|w\\|_2$、dormant、$er_2$ | 诊断 | 僵之前的报警 |

旧窗升太多 → 加大 $\rho$ 或加教师 KL。

新窗降不动、探针 AUC 变差 → 提高 rewarm、查 WD / 有效秩，不要先加 EWC。

只有 slope 变好、level 变差 → 不是 CPT 成功（PR#22 教训）。

## 算力怎么配（300M）

- 先用 1–2B token 试点：2 档 peak LR × 3 档 $\rho$，看旧/新 NLL 前沿。
- 再锁一组超参，在 COVID 与 2024-08 上跑完整 CPT。
- 可塑性探针：从 $\theta_{\mathrm{early}}$、$\theta_{\mathrm{late}}$ 各拷一份，同一 $N$、同一 schedule、同一切片，不要拿 scratch 当「没有老化」的证据。

## 不要做成的 CPT

- 服务时每个请求反传骨干。
- 把决策头的 IC 塞进预训练 loss。
- 用 regime 分类器切任务再 EWC。
- 只在最近 20 个交易日上滑窗、还叫 continual pre-training。
- 余弦尾已经 0 还不 rewarm，然后宣布模型死了。

一句话：CPT = 旧权重出发 + 新旧数据按 $\rho$ 混 + 学习率重新抬起来再降、但别降死 + 每段同时看新窗、旧窗、探针。先扫 $\rho$ 和 peak LR，别先上 reset。

---

## User

这样每次都要看过去的 岂不是很浪费?

---

## Grok

不必每次把历史全扫一遍。浪费的是 **「从头重训」或「100% 重放旧 tick」**；CPT 要的只是一小截旧分布的锚。

## 实际在付的成本

$\rho=0.10$ 表示这段新预算里 10% 的 token 来自旧窗。

新一年若吃 50B token，replay 是 5B，不是 2015–2019 全量再走一轮。旧样本还可以：

- 水库抽样（固定容量，按年分层）
- 只留每天/每标的一个压缩窗
- 越远的年份越稀（指数遗忘，不是均匀重放）

算力大致是「只训新数据」的 $1/(1-\rho)$，10% replay ≈ 贵 11%，不是贵一倍。

## 为什么还要付这 10%

只吃新窗时，旧制度上的 NLL 会升。对 LOB 这不是抽象遗忘：2020 的模型若完全不碰 2018 的拍卖/开盘结构，生成定态会漂。Replay 是把 CPT 律里的 **distribution-shift 项**压住。

可以不回放 **原始 tick**，改回放 **旧行为**：

$$
\mathcal{L}_{\mathrm{NLL}}(\text{新数据})
+\lambda\,\mathrm{KL}\big(p_\theta(\cdot\mid x_{\text{新}})\,\|\,p_{\theta_{\mathrm{old}}}(\cdot\mid x_{\text{新}})\big)
$$

旧检查点当教师，学生只看新流，用匹配把旧策略钉住。磁盘和 I/O 比抽旧 tick 便宜，算力多一次冻教师前向。旧窗掉得不凶就 $\lambda=0$；掉得凶先加 $\lambda$，再考虑加 $\rho$。

## 什么时候可以几乎不 replay

- 相邻月份、滚动 KL 很小：$\rho=0$ 或 0.05，只 rewarm。
- 只更决策头、冻 $\sigma_0$：世界模型这轮可以不 CPT。
- 突变切片（2020-03）你本来就想偏新分布：replay 是为了别把开盘结构忘光，不是为了记住那两个月之前的每一个 size。

不必「每次 CPT 都回顾全部过去」。规则是：**距上次 CPT 漂移小 → 少回放或不回放；旧窗 NLL 开始升 → 加 5–10% 或加教师 KL。**

## 真正浪费的做法

- 每年从零训到当天（那才是全历史税）
- $\rho=0.5+$ 还当默认
- 为了 replay 把序列重新 tokenize 两遍
- 没有旧窗指标，replay 开满「以防万一」

先跑 $\rho\in\{0,0.05,0.10\}$ 加一项教师 KL 的 2×2。多数年份会停在「几乎不回放旧 tick、只偶发对冻教师打一点 KL」。那 10% 是保险，不是 CPT 的定义。

---

## User

搜索tweets

---

## Grok

X 上和「每次回看过去是不是浪费」直接相关的，不是口号，是这几条。

## Replay 被写成标配，但都是「掺一点旧的」

Chandar Lab 推 CoLLAs 2025 论文：**Revisiting Replay and Gradient Alignment for Continual Pre-Training of LLMs**（arXiv 2508.01908）。题目就承认：LLM 的 CPT 里 replay 还在，值得重测，不是已经过时。[⁠@ChandarLab](https://x.com/ChandarLab/status/2031391084006486382)

安全域 CPT 的公开配方也是同一句：在 11.8B 新 token 上做 CPT，**同时 replay general knowledge，防止忘**。没有人写「把旧语料再训一遍」。[⁠@NaseerMuzammal](https://x.com/NaseerMuzammal/status/2016550120184246430)

Suhas Kotha / Percy Liang：微调时 **replay 通用预训练数据**——不只减少遗忘，在微调域上往往更好，尤其是微调数据在预训练里本来就少。方向和「回放=纯浪费」相反：一小截旧分布能抬新任务。[⁠@kothasuhas](https://x.com/kothasuhas/status/2029983689988542742)

## 「浪费」在 X 上指的是另一件事

被骂贵的是：

- 整段历史从头再预训
- 为了蒸教师，在 API 上滚几千万条（Lambert：蒸馏红利在变小，大 RL 滚次太贵）

不是 5–10% 的旧 token。Grok 那条甚至把 **冻底座 + 扩宽/加深 + 自蒸馏** 说成比一次次全量重启更省、更能把旧算力留下。那是「别把过去算过的权重扔掉」，不是「别看旧数据」。[⁠@grok](https://x.com/grok/status/2086390831913500931)

本地模型用户的痛点也是：CPT 更新知识时 **怕忘原能力**，要的是锚，不是全量回放。[⁠@simajilord](https://x.com/simajilord/status/2048237436061544470)

## 比固定 10% 更细的用法（和「别每次全看」一致）

Barry Yao（EMNLP 2026）：CPT 时内部电路能稳定标出 **学习 / 遗忘 / 干扰**；下一步是按这些信号 **自适应分配训练量、调数据日程**，而不是均匀重放全部旧概念。[⁠@barry_yao0](https://x.com/barry_yao0/status/2090859871138619609)

这就是「旧窗 NLL 升了再加 replay / 教师 KL，没升就 $\rho\approx 0$」在 X 上的对应物。

## 对你那句的翻译

| X 上在说 | 对 σ0 |
| --- | --- |
| CPT 默认带一点 general/old replay | $\rho=5\text{–}10\%$ 或冻教师 KL，不是全历史 |
| Replay 旧通用数据甚至能帮新域 | 旧开盘/拍卖结构对新年份不是废数据 |
| 全量重训、海量教师 rollout 才叫贵 | 贵的是每年从 2015 训到今天 |
| 按遗忘信号调配数据 | 用旧窗 NLL / 电路或字段级掉点决定要不要回放 |

没有搜到「CPT 必须每次重读全部过去」的正经帖。有的是 **少量 replay 仍是 2025–2026 CPT 论文的主旋钮**，以及用遗忘信号决定看多少旧的。

36 sources

---

## User

那搜索tweets 最接近 xty xtx的 continual learning的解决方案是什么?

---

## Grok

X 上 **没有** XTX / XTY 发表的 continual learning 配方。搜 Wang、Gerko、XTY Labs，落到算法上的帖几乎是空的。最接近的是下面几类「旁证」，不是内部 runbook。

## 直接和 XTY 有关的帖在说什么

Atlas Wang 上播客（VITA / Information Bottleneck，2025-12）：题目是 **金融基础模型 + 在纯噪声里抽信号**，以及神经符号、为什么去量化而不是前沿 LLM 实验室。没有 EWC、replay、CBP、Muon。公开技术自我定位是「大规模时序基础模型 + 符号约束」，不是「我们用了哪种 CL」。[⁠@VITAGroupUT](https://x.com/VITAGroupUT/status/1998957822386839649)

Stony Brook 研讨会预告同一人：**Algorithmic Trading with Large-Scale Deep Learning**（XTX Research Director）。还是规模与交易，不是持续学习论文。[⁠@ai_sbu](https://x.com/ai_sbu/status/2029406220990263336)

一条有信息量的旁注：有人讨论「第一家从头预训 LLM 的公司」时写 **as far as I can tell xtx is not doing this**——至少在圈内印象里，XTX 不是在走「再造一个通用 LLM」那条 CL/CPT 叙事。[⁠@PingbangHu](https://x.com/PingbangHu/status/2055374821887029285)

Gerko 机器人号只转慈善数学、Lean、vibe research，零模型更新。

## 圈内把 XTX「持续学习」理解成什么

传播最广、也最接近外部对 XTX 的想象，是这类工业化描述，不是学术 CL：

> 2.5 万 GPU + 十亿欧元机房，干一件事：机器学习机器人 **研究、构建、回测、优化、重复——按小时**，直到策略在从未见过的数据上成立。

这是 **短周期实验闭环**：大量并行假设、在未看见的数据上杀死，而不是一个 foundation 权重活十年、靠 replay 防忘。和 Wang 公开讲的「Infrastructure 才是 edge、承受大量失败实验」对齐。[⁠@antpalkin](https://x.com/antpalkin/status/2076001027950592239)

没有人在 X 上替 XTX 写：rewarm 30%、ρ=10%、ReDo、BOCPD 调 η。

## 所以「最接近 XTY 的 CL 方案」在推特上长什么样

把能搜到的信号叠起来，外部能合理推断的解是：

| X 上能钉住的 | 翻译成 CL |
| --- | --- |
| 基础模型吃噪声时序，不造通用 LLM | CPT 对象是市场 token，不是 CulturaX |
| 按小时级研究–回测–淘汰 | 持续的是 **实验与头**，不是单网终身 SGD |
| 基础设施 + 失败实验复利 | 多检查点、多模型，相关失效靠风控不是 EWC |
| 神经符号 / 干净方程 | 硬约束 + 生成，不是纯分布匹配 |
| 不公开任何 replay 数字 | 配方保密；公开层只有评估哲学 |

这和 Ibrahim/Chandar 的「LLM CPT + 10% replay」不是同一条产品线。X 上更像 XTX 的，是 **高频换策略/换头 + 大模型当研究教师**，持续学习发生在「哪条假设还活着」，权重更新是批作业。

## 必须标红的

- 没有 一条 XTX 员工帖写 Continual Backprop、task-free CL、GDN 当 CL。
- 讲座里的「老化 / 无标签漂移」来自小红书整理的 Harvard / AIX 口述，不是推特原创线程。
- 「25k GPU 每小时迭代」是二手叙事，当作文化近似，不当作系统架构图。

若一定要从 X 里挑「最像 XTY 会采用的 CL」：不是 Nature 2024 那套重置神经元，而是 **大批并行短生命周期模型 + 未见数据上的淘汰 + 基础模型慢更新**。要配方数字，X 上没有；那一层他们不会发推。

35 sources

---

## User

那你能猜出来吗

---

## Grok

能猜，但要标明是推断。依据是：公开判断、算力形态、做市约束、Wang 自己的学术口味。不是仓库里的配置。

## 总结构：三层寿命，不是一个永生网络

```text
慢：市场基础模型（周～月更）     生成 / 表征 / 仿真
中：预测头与因子（日～周更）     短 horizon 信号
快：执行与风控（分钟～日，大量并行）  报价、库存、约束
```

Continual learning 主要发生在 **中、快两层被杀死和替换**；慢层是 CPT，不是 tick 级反传。这和「25k GPU 上研究–回测–淘汰」以及「模型会商品化、基础设施才复利」是同一张图。

他们几乎不会押一个 σ0 用五年、靠 EWC 续命。Correlated Model Failure 那条决定了：**必须能同时跑很多不完全同分布的模型，并测量它们何时一起错。**

## 慢层（最像「学术 CPT」的那一块）

猜是：

- 私有订单流 + 跨品种、跨场所的半结构化序列，不是从零训一个通用 LLM（和「xtx is not doing this」那条旁证一致）。
- 更新是 阶段性 CPT：新窗为主，旧窗只留一个小水库或冻教师 KL，不会每年重扫 2015 至今。
- 一定 rewarm。Wang 讲老化时强调有效学习率；余弦退火到 0 再续训，正是他会当成「假死」的操作。
- 优化器更像 带谱约束 / 低秩的 Adam 变体（GaLore、低维透镜），不是 Nature 文的整网 reset。Reset 太伤已投入的私有预训练。
- 评估按 距训练集末日的衰减 + 事件切片（2020-03、2024-08），没有 regime 分类器当训练开关。漂移统计量调的是这段 CPT 的 η 和 replay，不是 token 门控。

GDN/Mamba 若在用，是骨干，不是 CL 算法。他公开嫌一维递归缺几何——内部更可能把预算砸在 **盘口怎么编**（档位、队列、跨资产），而不是再堆一层线性注意力当持续学习。

## 中层（他真正当「持续学习」用的）

冻慢模型，头在新数据上重拟合或短 CPT。失败了换头、换特征、换窗口，慢模型不动。

这解释了为什么外部看起来像「每小时迭代」：迭代的是假设和头，不是 10B 骨干。可塑性探针若存在，也是研究侧的，不会挡执行上线。

## 快层（做市本体）

大量短寿命策略，严格的未看见数据淘汰，相关性监控。某一簇在同一方向失效就降权或下线。这是 CL 的工程实现：**用组合的出生死亡代替单网终身可塑性。**

RL 只会出现在仿真器里（他们自己的生成模型当环境），不会在实盘用无 checker 的 GRPO 改基础模型。

## 神经符号放哪

不会是「定理证明下单」。更像：

- 生成后再用守恒/网格/因果过滤
- 或辅助损失，把明显违反微观结构的样本打掉

用来减「容量拟合噪声」，和 2025 报告里「极噪声上 scaling 仍有用」不矛盾：scale 吃可压缩的机制，符号挡住不可压缩的胡编。

## 和论文清单的对应（猜他们会留下什么）

| 会像他们的 | 不会当主力的 |
| --- | --- |
| rewarm + 小 replay / 教师 KL | EWC、Progressive Nets |
| 事件切片评估、动态 regret 思维 | 先定义 Task 再 CL |
| 谱/低秩优化、偏高 WD | 默认整网 ReDo / CBP |
| 多模型 + 相关失效 | 单世界模型兼执行 |
| 编码/几何 + 硬约束 | 服务时 TTT 写核心 θ |
| 头快、骨干慢 | 纯 LLM scaling 路线 |

## 最可能被说中、也最可能猜错的两点

说中概率高：分层寿命 + 批 CPT + 未见数据上杀头 + 不公开任何 ρ。

猜错概率高：具体骨干（也许根本不是 SSM）、replay 是 5% 还是只用蒸馏、有没有内部可塑性 AUC。这些 X 上为零。

一句话：XTY 的 CL 不像 Dohare 实验室，更像 **慢教师定期 CPT、快学生不断出生死亡、几何和约束管噪声、相关性当系统性风险**。σ0 能对上的公开工作是把慢层的 CPT 和生成定态测清楚；快层那套他们不会发。

---

## User

ttt 呢?

---

## Grok

TTT 在这张图里是 **快权重**，不是 CPT，也不是 XTY 那种「每小时换头」。

## TTT 实际改的是哪一层

一次前向里，用当前窗口的自监督损失（下一 token、补洞、重建）对 **一小块参数或状态** 走几步梯度，然后用更新后的块做预测。窗口一滑，这块通常丢掉或衰减。

|  | CPT | TTT |
| --- | --- | --- |
| 写哪里 | 慢权重 $\theta$ | 快权重 / 隐状态 |
| 何时 | 日/周批作业 | 推理当下 |
| 跨日还在吗 | 在 | 默认不在 |
| 延迟 | 无所谓 | 做市很敏感 |
| 解决的漂 | 日历级制度变化 | 当下窗口里的局部非平稳 |

GDN 的 $\alpha_t,\beta_t$、Titans 的 surprise 更新、Nested Learning 的高频神经元，都是同一族：**上下文里的在线回归，不是把 2024-08 写进基础模型。**

## 对 σ0 有用的那一小段

闭环前 500 条、size/Δt 滑进错误定态——有一部分是「条件已经是自己写的脏前文」，TTT 可以在 **这一条序列内部** 让快层去贴当前 ticker、当前时段的到达节奏，而不改 300M。

值得试的最小形态：

- 慢权重冻死
- 只适应 LayerNorm / 一个低秩适配器 / GDN 的状态
- 损失仍是字段级下一 token，步数极少
- 对照：冻快层的同一模型

看的是 **这一窗的 NLL 和 $E(m)$**，不是次年 COVID 切片。TTT 赢了只说明「会话内适应有用」；旧窗会不会掉、晚检查点还能不能学新年份，TTT 回答不了。

Wang 那句「只靠 context 做不出真正持续学习」就是对着这个：可塑性在 $\theta$ 上。TTT 再强也是 #58 那种错觉，除非另有一条巩固（Nested Learning 声称有，没有多年流式证据）。

## XTY 会不会当 CL 主力

推断：**会当特征，不会当基础模型的更新器。**

理由很硬：

- 做市要稳定延迟、可复现、可审计。请求里反传一块骨干，尾延迟和种子都炸。
- 相关失效：每个账号、每条序列自己 TTT，行为更散，风控更难看。
- 他们的 edge 叙事是基础设施上的大批实验，不是「线上每个 tick 写权重」。

更像的用法：

- 研究/仿真里开 TTT，看局部漂移能不能被快层吃掉
- 执行路径上最多用 已经训练好的门控 / 固定步数的状态更新（GDN 那种），不做即时反传
- 真正的制度切换仍走慢层 CPT + 换头

## 和三层寿命怎么摆

```text
慢 θ     CPT / 周月          制度、编码、世界模型
快 W_t   TTT / GDN 门控      本窗到达强度、这个标的的微观节奏
π        日更或重拟合        报价
```

TTT 填的是中间那条缝：今天这 30 分钟和训练时不一样，但还没到「该再 CPT 了」。缝以外的事（老化、旧开盘结构、2024-08 探针）仍是 $\theta$ 的。

## 实验上一句话

先冻 $\theta$ 开短 TTT，只报本窗；若本窗 $E(440)$ 降、跨日旧窗不变差，就留在快层。若只有 TTT 才能让新年份 NLL 下降，说明慢层已经僵，该 CPT 而不是把 TTT 开得更大。

---

## User

这里是这样的：它有成千上万种不同的 scenarios 同时在跑，然后不管自己跑出来的特征是不是稳定的，只需要在不同的 generative 的 scenario 里头经过 pass bug testing 都能成立，那其实就没关系。
我可以先训练好一个大的 checkpoint，然后去 online 的 policy distillation 去强化每种不同的 scenarios，然后在不同的 scenarios 上进行测就好了。
测完了之后，这有一套完整的框架，你可以源源不断地产生新的神经网络训出来的模型，然后只要在这个框架上效果好，不管能不能可解释，都直接上线。

---

## Grok

这就是那套 **慢教师 + 场景工厂 + 在线蒸馏出一堆可扔的学生**。特征稳不稳不是验收；验收是：在生成情景里过完同一套 bug test 还能赚钱或至少不炸。

## 三件套

```text
θ*     大 checkpoint（世界模型 / 教师）     偶尔 CPT
S      成千上万 generative scenarios      源源不断
π_k    每个场景上 OPD 出来的学生            过测试就上，不过就扔
```

流程是：

1. 先把 $\theta^*$ 训到能滚出「像市场」的轨迹（你们的 NLL + 闭环定态）。
2. 用 $\theta^*$（或再加撮合）采样情景 $s\sim S$：不同库存、价差、到达强度、延迟、制度切片。
3. 在学生自己的状态上做 policy distillation：教师给密信号，学生少步、走自己的轨迹（OPD，不是抄教师动作序列）。
4. 同一只 $\pi_k$ 必须在 一组没见过的情景 上过 bug test（库存爆炸、穿价、自成交、隔夜缺口、生成器饱和后的错误定态）。过了就上线。
5. 框架不停产新 $\pi_k$。可解释性不是门禁。

这和「每小时研究–回测–淘汰」是同一台机器；CL 发生在 **学生种群的出生死亡**，不发生在 $\theta^*$ 每 tick 写一遍。

## 这一套已经替你做掉的选择

- 不靠特征是否 stationary。Stationarity 被情景覆盖代替。
- 不靠单个 $\pi$ 活三年。活的是测试框架。
- 不在推理时 TTT 写核心 $\theta^*$。适应在蒸馏阶段、在学生身上。
- 不把世界模型和执行焊死。$\theta^*$ 出情景和教师分布；$\pi_k$ 才是下单。

Wang 说的 infrastructure edge，落在这里就是：**S 的生成器 + bug test 电池 + 蒸馏流水线**。模型是耗材。

## 框架必须测到的，否则「生成情景里成立」是假的

生成器有偏，测试会一起偏。最低限度：

- 情景里要有 学生自己滚出来的状态（否则又是 teacher forcing，上线闭环会崩）。
- 要有 $\theta^*$ 会饱和的那一段（头 500 条 size/Δt 错误定态）。只在真实前缀上测过的 $\pi_k$，闭环会踩你们 A01 的坑。
- 要有 跨情景同方向失效：很多 $\pi_k$ 在同一生成器 bug 上一起过——这就是 Correlated Model Failure 的内部版。过 bug test 的定义应包括「别全簇在同一情景族上满分」。
- 真实事件切片（2020-03、2024-08）至少做抽检。生成器没见过的制度，S 再大也只是 $\theta^*$ 的回声。

「不管特征稳不稳」可以；「不管教师有没有系统性洞」不行。学生再多，洞是同一个。

## $\theta^*$ 还要不要 CPT

要，但频率低。S 和 $\pi_k$ 再勤，教师过时了，整厂都在蒸过期世界。CPT 只服务教师：小 replay 或教师 KL、rewarm、旧窗 NLL + 探针。学生照样日更、扔了就扔。

TTT 最多给学生或门控，不要给 $\theta^*$ 的在线路径。

## 和 σ0 的接法

你们缺的不是又一个 CL 算法，是把现有线收成这三件：

1. $\theta^*$：编码 + 短闭环，把生成定态拉回（否则 S 是脏的）。
2. $S$：从 $\theta^*$ + 撮合抽样，显式覆盖库存/波动/深度 $m$。
3. OPD：同一教师，按情景出 $\pi_k$，电池测试后上线。

可塑性实验仍只打在 $\theta^*$ 上（早 vs 晚检查点）。$\pi_k$ 不谈老化，谈通过率和簇相关性。

一句话：持续学习在这套里 = **教师偶发 CPT + 情景工厂不停产学生 + 测试框架当唯一门禁**。可解释和特征稳定都不是门禁；教师洞和同簇同错才是。

---

## User

你觉得我说的这一整套办法是可行的吗？很有可能是 XTX 内部的做法

---

## Grok

可行。组织形态也像 XTX。但把「很有可能就是 XTX 内部做法」说死，过了。

## 可行，条件很具体

这套要成立，必须同时真的有这四块，少一块就会在实盘上假装成立：

1. 教师真能滚市场，不只会压 NLL。 生成情景是测试的宇宙。闭环在 500 条内滑进错误 size/Δt 定态，那宇宙是歪的，过测试只说明「在歪宇宙里自洽」。
2. 蒸馏必须 on-policy。 学生走自己的状态，教师打密信号。Teacher-forcing 的 $\pi_k$ 一上线就踩暴露偏差。
3. 学生必须小、可扔。 做市延迟不允许每个情景挂一个 300M。OPD 的产出应是小头或小网；大 checkpoint 只当教师和情景机。
4. 上线门禁不能只有生成器电池。 还要影子盘、限额、同簇同向失效、真实事件切片抽检。否则「pass bug test」= 在教师的回声里过关。

这四块齐了，**不管特征稳不稳、能不能解释，过门禁就上**——作为工厂逻辑是通的。持续学习落在学生种群，教师只偶发 CPT。这比「一个永生网络 + EWC」更像能跑十年的东西。

## 和 XTX 像在哪，不像在哪

像的是制度，不是算法名词：

- 完全自动、无人工自由裁量
- 算力用来并行试大量假设并在未见数据上淘汰
- 模型商品化、基础设施（数据、仿真、实验平台）才复利
- 公开担心相关模型同时失效

这些支持「成千上万情景 + 源源不断产模型 + 过测试就上」。

不像、或至少 **推特和讲座都没钉死** 的是：

- 情景工厂是不是 学出来的 LOB 世界模型。做市商更常是历史回放 + 微扰 + 撮合，生成模型当补充，不当唯一宇宙。
- 每个情景是不是都走 online policy distillation。内部更可能是监督头、直接短窗目标、或仿真里的简单策略搜索；OPD 是 2025–26 的语言，不一定是他们的词。
- 教师是不是一个统一大 checkpoint。也可能是按品种/场所切的一簇教师，故意不全共用同一生成洞。

所以：你说的是 **XTY 会喜欢的操作系统**；OPD + 单一生成器出全部情景，是你这边合理的实现，不是能从外部证成的内部图纸。

## 这套自己会烂在哪

- 教师洞放大。 所有 $\pi_k$ 在同一生成 bug 上满分 → 相关失效。Wang 第 8 条就是这个。工厂必须强制情景和教师的多样性，不能只比谁在 $S$ 上分高。
- 生成器与真盘的缝。 真盘有你没编进 token 的制度（手续费、拍卖、612、延迟）。只在 $S$ 里过关的学生，缝上会一起死。
- 「不可解释就上」在风控上有上限。 交易可以黑箱，但炸了要能关、能归因到哪一簇 $\pi_k$、哪一类情景。不可解释 ≠ 不可下线。
- 教师老化。 学生日更掩盖不了 $\theta^*$ 过期。工厂越顺，越容易忘了 CPT 教师。

## 判断

当 σ0 的产品架构：**可行，而且应该往这走。**

当「XTX 内部就是 OPD + 生成情景电池」：**形似，证据不够。** 更稳的说法是：XTX 公开行为像「教师慢、策略快、测试框架当门禁、模型当耗材」；你把耗材生产写成了 OPD，这是目前和你们技术栈最贴的实现，不是泄密级还原。

下一步不是再猜他们用不用 TTT，而是把教师的生成定态和情景电池做成硬门禁——工厂能转之前，宇宙得先不像假的。

---

## User

<aside>

💡
HRT AI 负责人：AI 正在如何重构量化公司
访谈来源：Bloomberg Odd Lots
主持人：Joe Weisenthal、Tracy Alloway
嘉宾：Iain Dunning，Hudson River Trading AI 负责人
原视频：[https://www.youtube.com/watch?v=py9jbwsbLrU](https://www.youtube.com/watch)
在首次采访七个月后，Odd Lots 再次对话 HRT AI 负责人 Iain Dunning。第一次对话更关注 HRT 怎样用深度学习预测短期价格；这一次，讨论转向了更现实的问题：当 AI 真正进入量化公司之后，公司本身会发生什么变化？ HRT 已经不只把 AI 当作写代码的工具，Agent 开始参与研究和实验，公司也在建立真实量化任务的人机 Benchmark；与此同时，算力、Token、数据中心和招聘标准都在被重新定义。
Iain 长期把自己视为一个务实、怀疑 AI 宏大叙事的工程师，但过去一年的变化让这种怀疑越来越难维持。HRT 的算力、模型覆盖范围和研究能力都在快速增长，以至于他开始担心自己是不是陷入了所谓的 “AI-induced delirium（AI 狂热症）”。这期访谈真正值得看的，是 HRT 已经发生的变化：AI 正在从研究工具变成量化公司的生产基础设施，并开始重构研究、风险、资本投入和人才标准。
TL;DR： Iain Dunning 的核心观点
1.
2.
3.
4.
5.
6.
7.
8.
访谈精编
HRT 为什么开始相信 AI 正在加速
Joe： 我终于实现了长期愿望：同一位嘉宾做两次节目。每次录完，我都会想起本该追问的问题。去年我们采访过 HRT 的 AI 负责人 Iain Dunning，讨论交易公司怎样实施 AI、怎样使用 GPU。这次继续。
Joe： 上一轮的问题相对容易，这轮会更难。HRT 是交易公司，你负责 AI。理论上，你们能不能像 High-Flyer 推出 DeepSeek 那样，利用现有技术栈训练一个通用大模型，做出 DeepSeek 的竞争者？
Iain： 我认为可以。我们擅长训练模型，拥有大量算力，也有研究人员反复完成研究循环，这些能力足以追赶一段距离。但真正到达前沿显然非常困难。投入一些努力，也许可以做到 DeepSeek 某个阶段的水平；再向前，我不愿声称 HRT 能做到。那需要极高的资本投入。
Joe： 公司内部会不会认真讨论：“我们其实可以做这件事”？
Iain： 我们想过。也许最合适的时间已经错过。现在美国和中国都在推出大量开放模型。DeepSeek 最初像一道突然出现的闪电，让人第一次看到有竞争力的开放模型；现在发布模型的团队已经很多。
我不知道开放模型未来会怎样。它们通常仍落后于真正前沿，而前沿推进得太快。怎样持续追上去是个难题。很多人相信开放路线能够跟上，我没有那么确定。
Tracy： 我看了你的 Twitter。你最近写道：“我每天都担心自己陷入某种 AI 诱发的狂热症，但多项经验指标看起来呈指数增长，所以最好假设我们正在冲向某种终局。”先证明你没有陷入 AI 狂热症，再解释你说的终局是什么。
Iain： 我现在说话越来越像一个旧金山人。我大约从 2016 年开始做 AI，当时在 DeepMind。那里的文化让我震惊，因为已经有一批真正的信徒，而我绝对不是。我天生怀疑，很长时间都在抵抗这种信念。
但外部世界里，一些关于进步速度的经验指标确实很惊人。再看 HRT 自己：明年拥有的算力相对于今年、后年相对于明年，都像指数增长。今天能做的事，是我过去没有认真设想过的。我希望自己能说：“我是有远见的人，早已看见未来并为它建设。”事实不是。我更像一个务实工程师，一步步推进，然后突然回头说：“一年里竟然发生了这些，那意味着什么？”
也许多种技术正在汇合，一切都在加速。当然，也可能只是狂热症。
Joe： 普通用户已经能感到模型一年间能力变化巨大。从交易应用看，2026 年能做、而 2024 年的你无法预料的事是什么？
AI 怎样进入 HRT 的研究流程
Iain： 一个变化是，训练和运行模型的算力投入都大幅增加。更重要的是，同一套技术、同一个统一方法可以同时用于全球每个股票市场、期货市场、加密市场和期权市场。HRT 正在这样做。
我不会声称只有我们在 AI 交易上取得真正进展。很多同业也在大规模投入，而且大家同时在推进。这最终意味着什么？我们不可能永远提高市场预测能力；竞争必然形成某种强制机制，大家不断投入，利润率最后可能接近零。
Joe： 你的意思似乎是，HRT 能够越来越准确地预测市场，而且预测期限也越来越远，但同业也在做相同的事。最终会不会像电影《Highlander》那样只能剩下一个赢家？
Iain： 这正是让我困惑的尺度问题。与此同时，大型实验室的算力扩张也公开可见，看上去同样呈指数变化。Anthropic 今天又发布了模型，版本间隔似乎越来越短。我说这些时确实听起来发烧了。但屋里每个人多少都能感到这股热度。
Tracy： 我一直不理解《Highlander》“只能剩一个”的设定，故事里明明已经有两个，他们可以共存。不过先不拆你的比喻。
Tracy： 新模型发布后，你们在 HRT 第一件事做什么？怎样把它与现有模型比较？
Iain： 我们最主要的用途是加速自己的研究。它可以写代码，也可以提出实验构想、监控实验。
去年 Anthropic 某一批模型出来时，我们经历过一次假启动。快速试用后，大家以为终于跨过了分界线。整整一周，团队处在一种亢奋状态，感觉 AGI 已经到了。最后却很空虚，因为我们发现它还不能真正增强人类研究员。
Opus 4.5 出来时，同样的感觉再次出现。这一次我们发现，它真的接近六个月前以为模型已经达到的状态。最近几轮发布的差异更细微，但我们越来越清楚模型还会犯哪些错，而且错误集合在缩小。
过去几周，我们一直尝试设计客观指标，把模型在真实量化研究中的能力与人类比较，包括构思交易信号等任务。传统量化研究有时像手工设计指标，为什么不让 AI Agent 做同样的事，再与人类进行一场小型对战？目前它也许达到实习生水平。
一年后会到哪里？我不想做大胆预测，但我也不愿假定它仍然只是实习生。
AI 会不会终结“因子故事”
Joe： 即便传统量化交易已经存在几十年，策略背后通常仍有一种人的直觉。例如便宜股票往往表现更好。我们未必完全同意它过去为什么有效，但这个说法至少不违反直觉。
现在是否已经到了可以跳过“给人讲一个合理故事”的阶段？只做严谨回测：某种关系经过一百万种检验仍然有效，我们不再解释原因，直接交易。
Iain： 我觉得今天已经进入那个世界，像是某种“后后后资本主义”（post-post-post-capitalism）。现在看到一些计划夏季 IPO 的公司及其估值，我会想：“基本面到底是什么？这些数字究竟表示什么？”一种犬儒看法是，所有东西都变成赌博市场，公开市场也一样。
但玩笑归玩笑，市场确实由流量构成，有买有卖，价格就是成交出来的价值。买方更多，价格上涨。模型很擅长直接从数据中提取这种关系。
Joe： 经典的数据挖掘例子是：“股票代码以 P 开头的公司星期二表现更好。”数据也许这样显示，但完全没有道理，所以人不会交易。未来会不会变成：我们反复检验后确认 P 开头的股票星期二确实上涨，于是决定投入资金，完全不再问为什么？
Iain： 我觉得可能。说出口很疯狂，也像一种狂热症，但它有可能是真的。只是在什么期限上成立，我无法预测。
极短期限里，大家已经接受这种情况。没有人合理期待一个人看着订单簿，就能准确说出一分钟后的股价；但我们接受神经网络能做到。那么，这套逻辑为什么一定会在某个更长期限突然失效？如果模型读取全部数据，把人无法同时容纳的信息留在上下文里，我为什么必须理解它？
这种想法很奇怪，让人觉得失去控制。但也许数学早就在提醒我们，人类本来就不擅长数学。AI 在数学证明上远胜多数人并不奇怪。全球有数千种工具，在很长的时间尺度上相互作用，人类也可能从来不擅长市场，只是过去默认有少数人擅长。也许那只是一个暂时状态。
当模型不再需要向人解释自己
Tracy： 上次你说模型缺乏可解释性，但在你们的短期交易范围里可以接受。我们当时还拿“魔法模型”开玩笑；“魔法”在这个节目里是个危险词，因为大家很容易想到“魔法盒子”。又过了半年，你是否更理解模型为什么能在短期成功？
Iain： 我们做过一些诊断，能看见部分可以理解的东西。它像一个极其复杂的物体，你观察其中一个切面，发现这个切面符合人的理解，于是增加一点信心。但信心也可能是错觉，因为你只是在一个超复杂对象上切了几刀。
我们曾观察到一种涌现现象，模型像是从第一原则理解了迷因股。例如量子计算股票和加密相关股票在模型的股票空间里彼此靠近。从基本面看，它们没有理由属于一类；但用某种分析透镜查看模型时，可以明显感觉模型知道它们相关。还有一些 WallStreetBets 喜爱的公司也位于附近，我不打算点名。
这只是一扇很小的窗。我们还看过其他切片，完全无法理解。但我又凭什么否定模型？
Joe： 模型说这些公司在高维空间里彼此相邻。
Iain： 对。也许它看见了我没有看见的结构。
Iain： HRT 的“魔法模型”外面包着很多安全层，因为我们做的是较高频率的双向交易，系统里有大量自动风险检查。
我不知道怎样把同样逻辑推广到长期主观交易。那里很难建立同等清楚的风险检查和分层防御。我们可以严格约束短期模型，因为问题定义清楚，不会在单只股票上持有几个月的巨大特异性风险。正因为这些控制，我们晚上睡得着。
但如果是基本面多空策略，模型要建立一个持有三个月的头寸，并且故意朝某个方向承担很大风险，同样的防护方式怎样应用？我没有答案。假如把全部控制权交给一个神奇预测，这首先就是一个 AI 风险管理问题。
算力正在变成一种资本密集型业务
Joe： 上次你说过一句很重要的话。AI 交易的投资者总在寻找当前瓶颈，因为瓶颈往往拥有最高定价。你说芯片本身不是 HRT 的主要限制，真正困难的是选址、供电和电力接入。现在情况怎样？假如我挖走一批 HRT 员工，拿到一批 GPU，仍然很难找到地方把它们运行起来吗？
Iain： 在短交付周期内，场地非常难找。假如我现在要求第四季度在北美交付一套装有六千块 Blackwell GPU 的完整设施，我不确定市场上有价格合理的现成产品。除非有人放弃租约，我们正好接手。
需要区分芯片和完整容量。有电力的话，今年可以拿到 Blackwell 芯片；但芯片、机柜、网络、冷却、供电全部组合起来的方案未必能拿到。进入 2027 年，首批 Rubin GPU 也会售罄。到那时也许能找到数据中心外壳，但想优先获得新 GPU，现在就必须进入队列。
我在算力规划上的最大失败，就是长期低估 HRT 需要多少 GPU。怀疑让我下单太慢，结果一直追赶。同业也经历了相同过程：最初只有一座自己的数据中心，算力需求增长后，只能到任何有容量的地方寻找资源。我们也不能挑剔，哪里有一兆瓦就先拿下，哪怕合同条款对我们不够有利。
Tracy： 你们怎样采购这些资源？竞争听起来非常激烈。HRT 在挪威之类的地方不是已经有一座很大的数据中心吗？
Iain： 有，但不够。我们同时联系 Neo Cloud、超大规模云厂商和其他供应方，保持持续对话，它们彼此也在竞争。
这些 Neo Cloud 背后还有一场更隐蔽的竞争，因为每家公司都在寻找场地和电力。也许场地与电才是真正稀缺的资源，Neo Cloud 只是把它们包装成算力再卖给我们。我不了解它们全部采购过程，但它们确实会突然通知：“一份租约刚空出来，请在今天结束前决定是否签长期合同。”
合同不是现货算力，而是八千块 GPU、三年、四年甚至五年的付款承诺。双方可以谈一半预付，也可以按年付款。商业条款组合很多，双方都承担信用风险，交易非常复杂。
Joe： 再谈谈交易对手风险。你以 HRT 的名义要求大量数据中心容量。现场很多金融从业者知道 HRT，但在旧金山等地方，它并不是家喻户晓的公司。数据中心需要确认你们未来能一直付账。你怎样证明自己是可靠租户？
Iain： 这一直是一场舞。现在容易了一些，因为我们做过足够多交易，市场开始认识我们。但我们遇到过各种问题。有人会问：“你们发过债券，评级是多少？”也有人不愿把某个站点太大比例的容量卖给我们。
如果我们拿走一个站点的全部电力权利，后来破产，数据中心可能很久都找不到另一个大租户填满容量。所以双方都要判断风险。供应商想要客户，市场上也有很多买方，但愿意一次承诺巨大规模并提前支付大量现金的客户没有那么多。
反过来，我们也会看部分供应商的 CDS，判断怎样影响合同。也许我们每小时多付三百五十美元，同时花十美分买一份类似保险的信用保护。对一个高杠杆 Neo Cloud，业务中断并非不可想象。我不点名。
双方保持谨慎很合理，因为这个市场几乎从零突然出现。一年前，我们还没有提出这些采购要求，很多供应商也不存在。整个链条里最稳的“岩石”大概是 Nvidia。它资本雄厚，不会突然消失，也确实在生产 GPU。我们与 Nvidia 的关系很好，这也是重要因素。
HRT 开始向芯片和算力金融化延伸
Tracy： 现在选择 GPU 的自由有多大？如果你更在意延迟或吞吐量，能否买到对应芯片，还是只能拿到什么用什么，或者干脆自己设计？
Iain： 可以选择，也可以自己做。很多公司在开发推理芯片，因为推理芯片在技术上比训练芯片简单。HRT 和多家同业都有硬件团队处理这个问题，而且部分流程可以外包，所以没有听起来那么吓人。对我们来说，这是活跃研究领域。
现在几乎每家公司都在谈与 Broadcom 合作。别人说自己在“与 Broadcom 合作”，通常意味着他们在做推理芯片。
Joe： Amazon 有 Trainium，Google 有 TPU。未来会不会出现一块 HRT 品牌芯片？
Iain： 我们不会把它对外销售，但这个思路是对的。
另一方面，Jensen Huang 从不睡觉。Nvidia 收购了 Groq，并基于收购增加了很有吸引力的产品线。市场上还有其他方案，例如 Etched。推理芯片的设计空间较小，竞争者足够多以后，不清楚每家公司是否仍有必要自研。
训练端不同。Nvidia 的护城河仍然极深，另一个选择也许是 Google。但使用 TPU 意味着与 Google 建立紧密关系，带来供应商锁定。算力极度饥渴的实验室会接受任何可用资源。Anthropic 同时使用 TPU、Trainium 和 GPU，因为每一种都需要。
Joe： 稍后我们会采访 Compute Exchange 的 CEO Carmen Li。他们试图为算力容量建立像石油一样的金融市场，包括算力期货。假如交易所提供 H100 或其他基准芯片的流动合约，代表运行这些芯片的成本，这会成为 HRT 有用的金融工具吗？
Iain： 有可能，这与我没有提前规划好算力需求的失败直接相关。如果我能锁定未来某个日期交付算力的价格，就能对冲自己下单太晚、价格上涨的风险。
2026 年内存价格上涨很快，我们已经遇到具体项目，让我后悔没有早一个月下单。所以风险是真实的，工具也可能有价值。
但长期算力期货能否形成足够流动性仍要观察。短期算力期货对我有什么用，我也不确定。首先，“算力”本身很难定义。假如要求实物交割，交割物是什么也不清楚。数据中心合同期限很长，每个站点还需要大量定制工程。
HRT 接入一个 Neo Cloud 站点时，要设计它怎样连接其他站点。每家供应商的网络和文件系统不同。GPU 之外还要考虑数据放在哪里，是在同一座设施里，还是硬盘集中在邻近站点。每个项目都很特异。
我用一百二十八块 GPU 做不了什么，需要的是数千块，这才是我们的最小交易单位。很难把整套资源拆成标准化小单位。也许现货产品容易一些；长期合约怎样设计，我还要继续学习。
Tracy： 我看过 Compute Exchange 的预览。界面允许用户选择具体数据中心和所需算力类型，参数有几十项甚至上百项。我们之后可以请 Carmen 演示。
当 AI 成为研究员的日常生产工具
Tracy： 你们现在的 Token 支出有多大？希望比 Joe 的个人用量高。
Iain： 我也很难给出精确平均值。我的团队里，每名员工每天大约花一百到两百美元。有些人在短期高峰会达到每天一千美元，但这种使用很突发。
Tracy： 你会更喜欢那些人吗？他们理论上生产率更高。你会鼓励其他人增加用量吗？
Iain： 会关注，也会鼓励理解他们为什么这样用。有些人会进入实验高峰，也可以叫 AI 狂热症期，这很容易理解。我们总想知道，高用量者是不是找到了其他人还没有掌握的方法。
这是一项从前完全不存在的新费用。金额还没有大到令公司担心，但它作为独立成本项出现，本身很值得注意。
Joe： 消费级模型经常过度谄媚。工程和量化研究里也会发生吗？模型会不会对研究员说：“这个想法太聪明了，你已经接近破解市场，再多花一点 Token 就成功了”？Claude 很喜欢说：“这里正在发生真正重要的工作，这个论证非常好。”
Iain： 会，而且现象很有意思。我们刚开始新一届暑期实习。过去的实习项目里，我们发现新人进入量化交易环境时压力很大。行业没有一本真正有用、读完就能上手的教科书，所以他们会向 AI 提问。
AI 总会以异常高的频率强调某些概念，而领域专家未必会这样选重点。冬季实习项目里，许多实习生反复使用一批很技术化的量化金融术语，但正式员工几乎从不这样说。追溯之后，我们发现最初的种子来自 AI。它像一种心智病毒，把相同表达复制给一群新人。
但我们的 Token 支出肯定还会增加，我们确实从中获得价值。生产率提升也许还不到两倍，但我与一位同行聊过，他认为团队生产率提高了 50%。如果真能提高 50%，每天为一个员工支付一百美元当然划算。
我不知道 Token 贫困的人怎样与 Token 富裕的人竞争。假如两个人同样聪明、同样善于利用资源，其中一个人始终带着一个能提高 50% 效率的副驾驶，而得到它只需要花钱，差距会迅速扩大。
成功者赚到更多钱，就更愿意承担每人每天一千美元的 Token 成本，从而跑得更快。这会形成“有”和“没有”的差异，并可能不断复合。在改善速度是关键变量的领域，结果可能越来越赢家通吃。这套故事有逻辑，当然也可能只是我的狂热症。
AI 正在重写量化人才标准
Tracy： AI 行业另一条大新闻是人才竞争，每家公司都在追同一批天才工程师。HRT 现在感受到的竞争怎样？
Iain： 格局变了。只要履历足够好、表现得足够有冲劲，仍然可能凭一个很初步的想法拿到风投。这种现象以前也存在，某种意义上就是 YC 的哲学；但现在的金额和 FOMO 程度很惊人。
创业本身成为招聘竞争的一部分。候选人会想：“为什么不自己创办公司？”有时他甚至还没有具体想法，只是觉得应该创业。
大型实验室则面对剩余上行空间的问题。假如最领先的两家已经达到一万亿美元估值，再向上还能到哪里？这会影响新候选人的远期乐观，也会影响已经在实验室工作、考虑离开的人。
这些实验室快速扩张，已经越来越像大型科技公司。人数大量增加，文化也发生变化，某些实验室的变化对 HRT 招聘有利。有段时间，我们感觉竞争极其激烈；现在也许更接近平等竞争。
不过，我与很多本科生交谈，他们对未来并不乐观，反而很担心。毕竟模型已经接近初级员工水平。
Joe： 现在你希望候选人带来什么？Anthropic 的面试题可能要求设计 GPU Kernel，或者优化数据中心内 GPU 的配置。你怎样重新定义人才？
Iain： 第一件事是接受开放式考试。我们希望允许候选人在面试中使用 AI，因为假装员工日常不会使用它，迟早会变得不现实。
量化行业一直有一种人物原型：数学理论家或弦理论家走进长岛某处，出来时手里拿着 Alpha。但我们的经验更复杂。一个人提出想法之后，具体怎样实现？过去这会限制纯理论型候选人的价值。现在 Claude 也许能完成实现工作。
因此，我们可能会接受更多理论家、梦想家和善于提出构想的人，相信 AI 可以承担部分落地。这是招聘标准的变化。
我最近一直拿 文字型人才（wordcel） 和 技术/数理型人才（shape rotator） 这组说法开玩笑：wordcel 的时代也许要来了。“Prompt Engineering”这个词现在听起来已经很老派，但把自己真正想要的东西说清楚、排除混杂因素，仍是一项可以学习的能力，而且在人群中分布很不均匀。AI 让这项能力的价值快速上升。
我愿意认为自己属于擅长描述问题的人。当然，也可能仍是 AI 狂热症在说话。
暂无评论

</aside>
<aside>

💡
HRT AI 负责人：深度学习如何预测短期价格
访谈来源：Bloomberg Odd Lots
主持人：Joe Weisenthal、Tracy Alloway
嘉宾：Iain Dunning，Hudson River Trading AI 负责人
*原视频：[https://www.youtube.com/watch?v=ADfpBrl8Avo](https://www.youtube.com/watch)*
Iain Dunning 是 Hudson River Trading（HRT）的 AI 负责人，加入 HRT 前曾在 DeepMind 工作。HRT 是全球领先的自动化自营交易与做市机构之一，覆盖股票、期货、期权、债券和加密资产。过去十多年，HRT 持续把深度学习引入核心交易系统：神经网络直接读取报价、成交、撤单等底层市场事件，生成价格预测，再由独立的执行与风控系统将这些预测转化为交易。
这期访谈主要讨论 HRT 如何用深度学习直接预测短期市场，以及这种方法为什么能够成立。 Iain 回顾了 HRT 从人工特征工程转向大规模神经网络的过程：减少人为预处理，把更底层的市场数据直接交给模型，让机器自己学习表征；这种方法依赖海量市场事件和大量重复决策，即使单次预测优势很小，也能通过规模累积成利润；与此同时，模型仍被包在严格的执行、监控和风险控制体系之内。
TL;DR：Iain Dunning 的核心观点
1.
2.
3.
4.
5.
6.
7.
8.
9.
10.
访谈精编
HRT 到底怎样赚钱
Joe ： 今天的嘉宾是 Hudson River Trading 的 AI 负责人 Iain Dunning。他此前在 DeepMind 工作，交易和 AI 两方面的经历都很扎实。
Iain ： 我很高兴来这里。我同意，行业里的神秘感被夸大了，尽管大家为什么愿意保持这种神秘感也不难理解。我们可以直接越过它。
Joe ： 先问最基础的问题。Hudson River Trading 是一家怎样的公司，它怎样赚钱？
Iain ： 我们是一家量化、自动化、自营交易公司。这几个词连在一起很长。我更愿意把我们理解成市场的服务提供商，最清楚的例子就是做市。任何人、任何时间、任何地点想买卖一种产品时，如果有人随时准备和他成交，这对市场有实际用途。对我们来说，这些产品包括股票、期货、期权、加密资产和债券。
假设你能造出一台神奇机器，随时为任何工具报出买价和卖价，而且报价是市场上最紧的。别人愿意和你交易，因为他们找到了对手方，也拿到了较低的价差。我们则从价差里赚一点钱。它很像在压路机前捡硬币。只有当那台机器能正确判断每种资产值多少钱，并告诉我们压路机什么时候会过来，我们才能持续捡到这些硬币。
从这个角度说，我们是一种高度复杂的中间商。Amazon 自己不生产大部分商品，却通过连接交易双方提供服务，成为一家有价值、有利润的公司。我们做的事也类似，只是我们在不同时间和地点之间，为不同对手方转移股票、债券和其他资产。
从人工特征工程到机器直接学习市场
Tracy ： AI 与过去的算法交易或量化交易有什么不同？这是对既有方法的渐进改良，还是交易方式发生了结构性变化？
Iain ： 我不想夸大 HRT，因为这个行业很不透明，我们不知道同类公司具体在做什么。我只能谈自己的经历。我们做这类交易已经二十多年。传统方法与其他量化公司相似：研究员根据人的直觉手工设计特征。比如，订单簿失衡，想买的人比想卖的人多，于是推断价格很快会上涨。
你可以让很多聪明人仔细琢磨，像制作高级机械表一样，把一个个部件精心做出来，然后用线性回归等相对简单的数学方法把这些预测变量组合起来。我参加会议和招聘很多年，到今天你仍能在网上看到有人说，金融里只能这样做。他们会说市场噪声太大、非平稳性太强，复杂模型不可能有效。但在我看来，这种信念没有证据支持，也不符合我们的实际经验。
理想状态是把市场问题交给一台没有人类偏见的机器。我自己不知道该怎样交易股票，我个人只买宽基 ETF。但如果你能把所有市场数据放进一个盒子，让它从这些数据中自行学习，它会找到手工构造特征永远找不到的东西。我们大约从 2013、2014 年开始这样做，在行业里算比较早。
过去十年，这条路线和金融以外的 AI 一样出现了曲棍球杆式增长。你可以从模型规模和投入算力看到这个变化。开始时，它只是传统方法的补充，两者组成混合系统；后来，直接从数据学习的模型完全超过了传统方法。现在，我们的交易全部由这台“神奇机器”驱动，它读取我们的全部数据。
我反复用“读取全部数据的神奇机器”这个说法，是因为 ChatGPT 的训练方式也是读取从互联网收集并连接起来的大量数据，再训练一个统一模型，最终产生某种涌现能力。这与人先用自己的市场直觉构造预测模型有实质区别。
Tracy ： AI 的价值有多少来自执行能力，比如用数百、数千块 GPU 快速处理大量数据；又有多少来自它发现了过去方法捕捉不到的复杂模式或错价？
Iain ： 两者都有。坚持“做一个线性回归就够了”的人忽略了金融市场到底生成了多少数据。这里的数据不是一条价格时间序列，而是市场发生的每一个底层事件：有人报价、成交、撤回报价。底层事件合在一起，数据规模可以与互联网相比。
我们在 AI 上得到过一个带着苦涩意味的教训：不要花太多力气做特征工程和预处理。把底层数据全部放进去，选择一种能利用互联网规模数据的计算方式。2010 年代的计算机视觉也走过同一条路。以前人们手工设计图像边缘检测器，再把这些检测器组合起来。那个方法不是毫无价值，但后来完全输给了另一种路线：准备大量 GPU，用相对通用的神经网络直接处理数据。
至于模型究竟找到了什么，是过去方法发现不了的，我们很难给出清楚解释。模型的可解释性不强，我认为这可以接受。我们的典型持有期是几分钟、几小时，最长往往也只是几天。假如让我看 Tesla 的订单簿，我能比随机猜测更准确地说出一分钟后的 Tesla 股价吗？不能。如果一个系统在这件事上已经明显超过人类，你还能期待它提供哪一种符合人类直觉的解释？
Joe ： 先把这里说清楚。你用这台在实验室里“长出来”而非被逐条编程的机器交易 Tesla、Nvidia 等股票。它不是聊天机器人，技术也不同。你的意思是，你们现在能比十年前更准确地预测 Nvidia 今天下午或明天的价格。过去有人说，股票市场不是国际象棋或围棋，所以机器无法用相同方式预测。但你认为这些模型至少在短期限内确实拥有预测能力。
Iain ： 是的。直到今天，我自己仍觉得这件事有点难以相信。有效市场假说已经进入每个人的脑子。别人告诉你他能预测一小时后的股价，你的本能反应是怀疑他在虚张声势。但这些模型确实能预测。
调和这种直觉的办法，是承认它们的预测其实很差。我们通常不用“准确率”描述交易模型，但你可以把它想成 50.1%。它只比随机好一点点。
Tracy ： 可是，只要规模足够大，多出来的一点准确率就能显著改变利润。
Iain ： 对。你反复做足够多次，时间一长就能看见这枚硬币存在轻微偏置。这个能力也不需要诉诸魔法。市场很美，因为许多参与者带着不同效用、风险偏好和约束相互作用。你只能从他们在市场里采取的行动观察这些差异。模型吸收大量微小的买卖流量信号，再向前外推。
Joe ： 对机器能否预测股价的怀疑其实有些奇怪。机器读取数据，发现某组数据出现后第二天更可能上涨。人类每天都在做同样的事。分析师根据少得多的数据，也会给 Tesla 或 Nvidia 一个目标价。既然人类可以预测，为什么拥有更多数据的计算机反而不能？
Tracy ： 一部分怀疑来自模型向后看。模型偶尔不能识别或应对大的制度变化。人们会认为，人类的思考更灵活、更能适应环境，也更容易看见文化变化。你们怎样应对大的模式断裂？
机器正在学习人类难以解释的表征
Tracy ： 再具体谈谈数据。很多人介绍 AI 时会用公关语言，说自己拥有大量罕见数据、另类数据，因此能把 AI 用得更好。你们实际看什么？哪类数据最有用？
Iain ： 我刚进入这个行业时，最反直觉的一点是：如果你预测一分钟或一小时后的价格，最有用的东西远远领先其他数据，就是普通市场数据，也就是从交易所购买的数据流。价格并不高。有人把它想成竞争护城河，但这些数据费没有高到无法负担。在加密市场这个“西部世界”，每个人也能自己采集交易所数据。
这些数据最接近所有参与者真实意图。人们到市场里报价、买入、卖出。数据流里是一连串很小的事件：某人以某个价格和数量报了一笔单。身份是匿名的，但这些事件是最原始的材料，而且数量巨大。每只股票、每个期货品种每天都会产生数百万个事件。
大家容易迷恋 Twitter 数据之类的东西。Bloomberg 的产品里也卖 Twitter 数据，当然可以买。新闻偶尔会在交易时段突然出现，推动价格离开原位。但冷静计算，它相对于整个市场的巨大活动量仍是低频事件。日内预测首先要看市场数据。
当期限延长到几天，所谓另类数据才真正进入模型。这里的“另类”是相对于市场数据而言，包括 SEC 文件、新闻、资产负债表和券商报告。市场上有一大片供应商出售各种数据。预测期限变长后，Sharpe 本来就较低，很难把增加的 Sharpe 精确归因给某个数据集。
数据市场在某种意义上也很民主。也许有人秘密收集很独特的数据，但我甚至不负责采购，邮箱里仍然经常收到最新另类数据集的推销。我认为很多数据未必有预测力，不过显然存在买方。
Joe ： 你见过最疯狂的数据是什么？
Iain ： WallStreetBets 时代之后，很多供应商迅速推出从 Reddit 抽取的数据，还试图在原始帖子之外做进一步提炼。但回头想想，迷因股往往是在事情发生之后被讨论得更多，而不是之前。所以我不知道这些东西到底能提供多少预测价值。
Joe ： 你刚才提到可解释性。我一直想不通一个问题，不只在金融里。DeepMind 做出了超过最强棋手的围棋系统，国际象棋引擎也早已超过人类。但据我所知，并没有真正优秀的 AI 国际象棋教练。引擎能碾压你，却不能清楚地说：“你刚才这一步堵住了车的线路，若干步之后会付出代价。”Chess.com 的自然语言解释也很初级。为什么 AI 能把问题做得极好，却无法用普通语言解释自己做了什么？
Iain ： 神经网络在某种意义上只是一大团数字。训练这些模型时，我们有意摆脱几乎全部先验结构。它们学习问题的方式可能与人完全不同。它们也许在内部进行某种“推理”，但我对这个词很谨慎。人们还会说神经网络拥有“想象力”，这种拟人化很危险。它们用一种本质上不同于人的方式处理信息，所以很难把内部过程重新映射成人类的解释。
也有一些有趣的反例。Anthropic 做过 Golden Gate Claude 实验，让模型对金门大桥产生强烈兴趣，无论问什么都绕回金门大桥。这说明模型并非完全无法探测。但超过某个程度之后，想把它映射成人的思考方式仍然很困难。AI 安全研究很想解决这个问题，虽然它与我的工作关系没那么直接。
Joe ： 如果这个问题能解决，许多岗位会获得明显的生产率提升。你们的模型与大语言模型不同，但两者都依赖海量数据和算力。在大模型实验室工作过的人，能否把训练经验直接带到你们这里？
Iain ： 2025 年可以。2020 年时我不会这样回答。这个变化让我很意外。我们的数据是很长、按顺序排列的信息流，任务是根据这段信息向前推断。早期 AI 研究更像图像分类：“这是不是热狗？”后来音频和机器人研究与我们稍微接近。到了 LLM 时代，双方的问题突然很相似：都需要回看很长的历史和上下文，都拥有大量数据，也都希望高效处理这些数据。
两边还都需要快速提供推理结果。大型语言模型面对数百万个在网页里输入问题的人，用户希望尽快得到回答。我们的模型也必须及时作出预测，否则预测失去价值。因此，我们对问题的思考方式越来越接近前沿 LLM 实验室。区别在于数据模态。它们主要处理文本，我们处理同样庞大、难以直接解释、具有顺序关系的 Token 流，只是我们的 Token 是市场事件。
这很有意思，因为只要还有论文公开，我们就能从中获得启发并比较做法。但市场仍是一个独立问题，每天都有自己独特的挑战。
HFT 的竞争已经不只是速度
Tracy ： 你说数据让金融更民主。2010 年代，人们称大型投行为“流量怪兽”，因为它们能看到大量客户订单，并利用流量优化融资成本和其他费用。数据与 AI 能否复制这种优势，让 HRT 也成为一个小型流量怪兽？
Iain ： 市场里仍有一些趋势让我担心。我们心目中的理想市场结构，大概是所有人都在一个中心化交易所里成交。但现实并没有向那个方向发展。大量交易发生在场外、暗池或准暗池。有些交易领域里，“人在那个房间里”仍然是巨大优势。从 AI 的角度看，这几乎是一种反 AI 结构：流量数据被藏起来，样本又很稀疏，无法交给机器学习。
不少数据最终会被集中报告，但延迟太长，无法用于实时决策。AI 依赖数据，所以从长期看，这会成为一个问题。你仍然需要进入实际发生交易的房间。
Joe ： 这正是我对物理基础设施好奇的地方。我向 ChatGPT 提问时，不在乎模型是在得州的哪个地方训练，只要回答能回来。但高频交易至少在执行端要求设备与交易所共址，线路越短越好。你们的硬件栈与前沿大模型实验室相比，有哪些相同和不同？
Iain ： 从整体上看，两者有不少相似之处。可以用延迟和吞吐量来理解：延迟是反应所需时间，吞吐量是在给定时间内可以做多少计算。交易确实要求低延迟。2010 年代早期，《Flash Boys》塑造了一种印象，认为行业主要靠延迟套利。好消息是，大部分延迟已经被套利完了。缩短信号线也许还剩一点优势，但相对于过去已经很小。大型量化交易公司不再把线路缩到绝对最短作为主要竞争点。坦白说，我觉得那部分工作很无聊。
现在更重要的问题是：在给定反应时间内，你能不能成为最聪明的参与者。这是一条曲线。如果你用一秒作出交易决定，它最好是一个非常好的决定，那么花一秒就没有问题。如果你必须在一微秒内反应，你做不了太多计算，但仍要给出这一微秒内最好的回答，只是我们可以接受它比一秒版本稍弱。
训练方面，我们同时使用云，也建设自己的训练数据中心，基本结构与 Google 等公司相似，只是规模小得多。与普通公司相比我们可能很大，但不是 Google 或 Meta 那种数十亿美元级别。
推理端不同。设备要放在交易所附近，我们必须仔细考虑功耗和延迟。HRT 有硬件团队，会设计 FPGA，也会设计自己的芯片，同时使用现成 GPU。目标是在每一种反应速度上，尽可能作出最聪明的决定。
FPGA 是现场可编程门阵列。各种设备的延迟和吞吐量不同。GPU 的吞吐量极高，这也是它们有用的原因。但市场的数据流比较窄。LLM 可以把成千上万名网页用户的请求凑成批次再处理，我们没有这种奢侈。市场按照自己的速度运行，我们不能暂停一会儿再追赶，必须一直留在场内。
因此出现了很有意思的设计问题：GPU 延迟相对高，返回一次结果要花时间，但一块 GPU 也许能同时处理整个股票市场。怎样把这种能力与最快的反应结合起来？HRT 有完整团队专门研究怎样从同一个智能模型里，以不同速度取出不同答案。
Iain ： 今天我们大量精力都投在这里，而不再是把宾夕法尼亚乡间的微波塔重新对准一点。后者也是一个很酷的工程问题，但已经解决了。我认为大家已经找到了从新泽西到芝加哥最直的线路。
Tracy ： Joe 刚才提到有人质疑 CME 与 Google 的云合作。Don Wilson 在我们节目里公开表达过一种担忧：把撮合放在云上时，你提交两笔订单，却无法确定哪一笔先成交，重新掉进了黑箱。也许这是延迟问题。你认为它是真问题吗？
Iain ： 这是我会担心的事。我们的总体理念是，市场应尽量透明、公平。让不同参与者平等接入是好事，不能让少数人通过奇怪技巧获得速度优势。但市场同时需要可靠性。如果订单先后到达，交易所却用不同次序成交，这不是合理的市场运行方式。工程团队需要投入很多工作保证顺序确定，市场设计也应把它当成基本要求。
这种差异在全球交易所里很常见。我们交易很多国家的产品。有些交易所的硬件极其出色：即便两笔订单只差一纳秒，哪怕订单来自一百个不同网络端口，系统也不会处理错顺序，因为它们建立了精密的时间戳机制。
另一个极端是某些加密交易所，感觉像一个刚学会 JavaScript 的孩子搭了网站。你发送订单后，甚至不知道它是否收到，只能五分钟后刷新账户余额，看看钱还在不在。市场是什么样，我们就应对什么样的市场。但我们明确偏好两件事：平等的接入和可预测的结果。让所有人把精力用在怎样把线路缩得更短，对社会并没有太大价值。
真正的护城河：算力、电力与系统工程
Joe ： 当你与大模型从业者交谈时，大家会争论主要约束到底是什么：电力、GPU、人才，还是训练数据。你们现在与理想状态之间，最大的约束是什么？
Iain ： 从长期战略规划看，电力显然是硬约束。我们要建设新的 GPU 训练数据中心时，土地通常不难找，困难在于有没有足够电力。HRT 同时使用云服务和自建数据中心，所以电力限制、供电谈判已经进入我们的规划。
我们有一座位于寒冷地区的数据中心，想进一步扩建。数据中心合作方很好，但他们会说，下一批容量需要先与电网谈判。很多时候，这才是瓶颈。
GPU 在过去确实短缺，但现在，如果我们下单，通常能在合理时间内收到，不一定次日送达，却不再是扩容周期里最长的一根杆。
Tracy ： GPU 最紧张是什么时候？
Iain ： 2023 年下半年很糟，正好是 Nvidia Hopper 一代。Bloomberg 昨天报道 Nvidia 的会议，数字大概是 Hopper 级 GPU 已生产一百万块，而 Blackwell 级已经生产四百万块。供应明显扩大，但我也不认为它们形成了卖不掉的库存，新产能同样迅速被吸收。
真正难的仍是电力。这件事让我很震撼。我是千禧一代，大学时气候变化是公共讨论的中心。现在很多人为了快速启动数据中心，大量购买燃气轮机，直接放在建筑外面发电。要及时获得电力，这几乎成了唯一办法。看到这种场景，我会想：我们到底在干什么？
大家宣称未来还要建设多少数据中心。只要用纸笔估算一下用电量，我就不知道这些数字怎样成立。Sam Altman 等人讨论过每个时间单位需要新增多少发电能力，但数字仍然很吓人。HRT 在整个 AI 用电版图里只是较小参与者，我们谈的是数十兆瓦，不是数吉瓦。数十兆瓦已经超过很多城镇的用电规模，而我们仍然很难以合理价格找到电。
Tracy ： 如果 GPU 紧缺已经缓解，延迟也不像过去那样重要，交易公司的竞争优势来自哪里？
Iain ： 人才仍是约束，市场竞争很激烈。我们需要的人既要是优秀研究员，也要是优秀工程师。进入 AI 时代，这两个身份越来越难分开。研究不再是先在白板上写想法，再把编码交给其他人。任何研究构想都与实现方式紧密连接。这类人很稀缺，我们也愿意为他们支付高薪。
更隐蔽的优势是把全部组件接起来。工程团队能不能采集所有数据、正确记录，再把它送进 GPU 训练数据中心？我们的数据集以 PB 计。仅仅存下这么多数据，并从存储位置可靠地传到世界另一处训练中心，就是巨大的工程工作。训练运行本身又很昂贵。
模型训练完成以后，还要把它真正部署出去。听起来答案只是“每件事都要做好”，似乎很平淡，但事实就是如此。我的团队叫 AI 团队，重点是训练模型。这是重要环节，却不足以独立构成交易系统。如果 HRT 没有其他团队负责取得数据、把数据送给模型、把决策传回市场，并在市场突然繁忙时维持全部系统，我们什么也做不了。
规模在这里有明显好处。我很难想象 2025 年从零创办一家 HRT 式公司，因为建立足够的工程规模需要巨大的前期投入。我们的同业也在工程上投入很多，而且会继续投入。
AI 正在把量化交易推向更长周期
Joe ： 《金融时报》最近写到，HRT 之类的公司正在进入更慢的交易，原本做慢速交易的公司则试图加快。你们也许能相当有把握地预测下一小时，幸运时预测下一天，一个月则过于遥远。你们模型的预测期限是否在扩大？
Iain ： 在扩大。外界即便知道 HRT，也常停留在 2020 年以前的印象，认为我们纯粹是一家高频交易公司。我们现在既做高频，也做中频，中频已经是很大的业务。
假设我对五天后的某只股票有明确观点，准备买入。我不会一次买完，而会在五天内逐步获得头寸。什么时候买最好？我有一个模型能告诉我未来一小时内哪一个时点价格更合适。于是，短期模型可以为长期交易提供执行决策，而且这种关系能够一层层向下传递。
Tracy ： 当交易期限变长时，你们的基本工作还是提供流动性，只是库存持有得更久，还是赚钱的理由也发生了变化？大家想到对冲基金时，通常会想到方向性策略，而非流动性服务。
Iain ： 到了更长期限，做市服务的比喻确实会失效，再继续用就太勉强了。此时更接近主动获取流动性。订单簿里有人挂单说要卖股票，我们判断长期价值更高，就主动买走，因此会跨过买卖价差并支付交易成本。当然，你也可以在做市时带一点方向性倾斜，慢慢获得头寸。
不过，任何交易都有对手方，对方也有自己的交易理由。我喜欢用国际象棋和围棋作比较。它们是零和游戏，只有一个赢家，一方加一，另一方减一。交易之所以能够持续，是因为它在效用意义上可以是正和。钱在成交瞬间守恒，交易所还收一点费，所以从货币算甚至略为负和；但参与者的效用不同。
我的工资进入 401(k)，自动购买 ETF。我四十年内都不会动它，所以对成交价格相差一美分不敏感。做市商赚到这一美分会高兴，我也高兴，因为我及时拿到流动性，没有跨过很大的价差。双方都获得了自己需要的东西。
这也是为什么不能用 AlphaGo 的方式理解市场。如果市场只有 HRT 和所有竞争对手相互伏击，大家都在比谁更聪明，最后会形成一个巨大的僵局，没有人愿意先交易。真实市场很活跃，因为即便我们主动跨过价差，对手方也确实有理由出售。五天后如果我们判断正确，他未必后悔，因为他可能只是在对冲另一个头寸，根本不在乎五天后的股价。
因此，它仍然可以理解成一种服务。我们只有在别人想交易时才能赚钱。如果没有人交易，我们就没有存在的理由。不同参与者拥有不同动机、目标和时间范围。
Tracy ： 工程师通常喜欢开源，也愿意参与 AI 研究生态。交易公司则更重视保护模型、数据和其他专有信息。HRT 怎样平衡这种冲突？
Iain ： 坦率说，很多年前，这确实让我们在招聘时处于劣势，尤其面对刚毕业的博士。他们会说：“去 Google，我还能发表研究，行业知道我是谁，我也保留职业选择；去 HRT 或类似公司，就像走进一道帷幕后面，多年不再出现，外界只能相信我在那里做了聪明的事。”
我当时没有很强的反驳，只能说写论文被高估了，我自己做过，年纪大一点以后你不会那么在意。
现在情况变了。大科技公司付钱让研究员持续发表的黄金时代基本结束。大型 AI 实验室公开的论文往往已经过时，或者不是最重要的工作。真正处于前沿的人不能分享自己在做什么，研究变得非常保密。这个问题在某种程度上替我们自己解决了，人才开始承认知识产权需要保护。
我甚至看到 AI 实验室的人公开思考竞业协议。这是一个惊人的转变。过去硅谷对此非常反感，加州事实上也限制竞业，还会以纽约交易行业使用竞业为由批评我们。
现在的大笔人才薪酬也在购买知识产权。个人不会把代码或文件偷走，但如果一家公司同时招走五个知道“汤是怎么熬出来的”的人，他们拥有大量流程知识。你会突然更认真地考虑保护这些知识。HRT 花很长时间培训员工，员工也需要很久才能产生完整生产力。假如别人可以把这套知识立即拿走，对我们当然不利。
AI 交易真正难的地方：风险控制与验证
Tracy ： 回到你最开始说的压路机。大家现在很兴奋 Agent 型 AI。我会想到金融史上 Knight Capital 的事故。它的一个算法失控，买入大约七十亿美元的股票，十五分钟左右就毁掉了公司。你们设置了哪些防线，避免重演 Knight Capital？
Iain ： 每一期新人培训，我们都会讲“Knightmare”，用 K 开头。HRT 有多位前 Knight 员工。一家成功交易公司在十几分钟内结束，这个故事一直困扰我们，我们尽量吸取其中每一个教训。
核心是分层防御。AI 模型不会直接向纽约证券交易所发送订单。它提供某种计划，后面仍由传统、经过人工审计和风险检查的系统执行。日常运行中，我们沿整条链设置很多层合理性检查。
更高层面上，新版本发布有严密流程。我们做上线前测试和审计，专门避免 Knight/ KCG 那类场景。交易日内还会对神经网络做持续的合理性检查，确认输出值落在预期范围。检查系统可能比市场数据流慢一点，却足以重新确认模型的数值稳定性等基本条件。
这里讨论的主要不是亏钱或赚钱，也不是传统金融风险，而是操作风险。整个组织保持很深的偏执。这与普通 AI 世界仍然不同。普通产品有时会接受一定失败率，把它计入成本；在交易里，一次失控就可能毁掉一切。
我们当然担心损失，但更担心做出监管者不能接受的行动。监管信任一旦失去，要很久才能恢复。HRT 在很多国家交易，我们尊重各地监管者，也认真遵守各自复杂规则。谁都不想因为一次操作错误被赶出一个国家。监管文化对错误的容忍度很低，我们也必须如此。
只要公司十年后还留在场内，长期利润远高于“快速行动、打破东西”带来的短期好处。我们希望行动快，但不能打破市场。
预测能力最终取决于期限和数据密度
Joe ： 我还有很多问题，最后先问一个以后值得单独做一期的话题。就业报告发布后的第一秒发生了什么？数字出现在屏幕或网站上，市场立刻大幅移动。人还没读完报告，更没来得及讨论“就业好不好、工资数据怎样”，价格已经变了。从 HRT 的角度，事件发布后的第一毫秒发生了什么？公司财报也可以作为例子。
Iain ： 我们有低延迟 Bloomberg 标题数据流。重要新闻会带星号之类的标记。处理方法可以从手工逻辑到 AI 模型：最简单的是查找关键词，复杂一点就把内容交给模型。
有件事我仍然很难理解。一些期权交易公司雇了数千人做人机结合交易。某只大股票，例如 Nvidia，可能有十个人专门盯着它的期权。他们看新闻流，界面上准备了绿色和红色按钮，一旦事件出现就迅速点击。这是一种真实工作。
我们曾在公司黑客松里模拟它：接上 PlayStation 手柄，让员工练习对事件快速反应。它很难，却是一种可以训练的技能。按照有效市场的逻辑，这项工作应该可以交给 AI。
真正困难的是，不能直接接入 ChatGPT。对日常使用来说它很快，对市场而言延迟仍然太高。更重要的是，你不能用 ChatGPT 做真正的历史回测。它训练时看过美联储主席的每次讲话，也知道讲话之后发生了什么。你怎样证明它面对下一次 Fed 讲话还会正确反应？
传统金融研究用回测检查策略过去表现。但这里的历史样本全在训练集里。学术论文会尝试处理这个问题，声称模型仍然有效并做各种校正，可这些模型很强的原因之一就是它们记住了大量训练内容。既然如此，历史“预测”为什么可靠？如果有人说：“我让 ChatGPT 分析了每次 Fed 讲话，十次里答对九次。”我反而会问，既然它都见过，为什么不是十次全对？
今天仍有很多人参与相对高速的股票和期权交易，尤其是小众产品。把全部信息实时整合起来仍然困难。人们讨论 2028 或 2030 年出现 AGI，但当下仍有许多人在手工交易股票和期权。我也还没有完全调和这两件事。
Joe ： 我很喜欢 Iain 对怀疑论的回应。很多人说 AI 能解决国际象棋，却不能解决股票市场。我一直不满意他们给出的理由。股票当然不像一个可以完全求解的棋局，但人类也是靠识别模式在市场赚钱，为什么硅基系统不能做同样的事？
Tracy ： 而且，我们已经有多年高频和算法交易的历史，参与者也确实赚到了钱。Iain 对期限的解释是我最重要的收获。你必须让 AI 方法匹配实际可用的数据。市场数据绝大部分集中在短期，秒的样本比分钟多，分钟又比天多。数据也天然偏向即时变化，而不是遥远历史。
Joe ： 金融里经常有人说：“过去十九次出现 S&P 500 死亡交叉，有十七次随后下跌。”任何严肃数据科学家都会嫌弃这种样本。十九个观测值还远远谈不上规律。
Tracy ： 但“死亡交叉”太适合写标题了。
Joe ： 对记者的建议是，绝不要放弃在标题里写“死亡交叉”的机会。我还很高兴听到，线缆长度不再是主要问题。过去有关 HFT 的讨论总是围绕把线路缩到极限，已经很乏味。GPU 市场也比几年前宽松。与此同时，即便 HRT 这种规模，电力也成为主要限制，这让人怀疑聊天机器人行业设想的扩张会不会撞上现实墙壁。
Tracy ： AI 实验室的文化变化也很有意思。并不是交易公司变得更开放，而是实验室自己变得更封闭、更重视专有知识。
Joe ： 这次回答了不少问题，当然还留下更多问题，以后应该再请 Iain 回来。
暂无评论

</aside>
<aside>

💡
XTX 研究总监：AI 如何在噪声中预测市场
访谈来源： The Information Bottleneck
主持人： Ravid Shwartz-Ziv、Allen Roush
嘉宾： Atlas Wang，UT Austin 教授、XTX Markets 研究总监
原视频链接：
*[https://www.youtube.com/watch?v=mq3QkSfrkPk](https://www.youtube.com/watch)*
Zhangyang “Atlas” Wang 是 XTX Markets 研究总监，负责纽约的 XTY Labs，同时也是得州大学奥斯汀分校拥有终身教职的副教授。XTX 是全球领先的算法交易与电子做市机构之一。Atlas 的研究横跨压缩感知、稀疏优化、Foundation Model 和神经符号 AI；在 XTX，他利用机构的私有市场数据、GPU 基础设施和跨学科团队，研究面向金融时间序列的大规模模型。
这期访谈主要讨论 XTX 如何用 AI 从高度嘈杂的市场中提取微弱但可兑现的预测优势，以及神经网络能否进一步被压缩成公式、规则和人类可读的知识。Atlas 解释了金融 Foundation Model 与语言模型的根本差异：它处理价格、成交量和逐笔交易等连续数值流，在预测相关性接近零的环境中，通过海量重复交易把微小优势转化为利润。访谈还延伸到数据为何可能比算法更重要、金融为什么尚未建立合成数据飞轮，以及私有数据、算力和“会写代码的数学家”如何构成金融 AI 的核心壁垒。
TL;DR：Atlas Wang 的核心观点
1.
2.
3.
4.
5.
6.
7.
访谈精编
用噪声预测噪声
Allen： 高频交易对速度有多敏感？机构是否必须靠近交易所的数据中心和海底光缆节点？ASIC、FPGA 也很重要，因为决策时间有时按纳秒计算。
Atlas： 速度确实重要。生成式 AI 也在改变这个行业，这也是今年 NeurIPS 展厅和 Workshop 对金融 AI 如此热情的原因之一。
Ravid： 不谈公司内部模型，只说行业共性，哪些研究问题最重要？
Atlas： 问题类型不多，但每一种都需要研究得很深。最常见、最实用，也最接近 Quant 入门的问题，是多变量时间序列预测。
高频场景中，你从交易所接收价格、成交量、逐笔成交记录，以及每只股票的其他元数据。模型可以同时观察许多股票，形成某种 Foundation Model。目标是预测每只股票在一段时间后的价格或至少方向。那段时间叫 Horizon，通常由执行策略决定。
形式上，它仍是经典的 ： 是未来价格， 包括当前与历史价格及相关信息。难点在于数据的噪声极大。
高频交易不缺数据。交易所每个 Tick 都在发送数据。我们缺的是高质量信号，也没有办法简单判断“今天的市场数据比昨天更好”。这里的质量问题不是机械错误或传输错误，数据团队会尽量保证原始记录真实、准确。市场本身由噪声主导，所以任务很像用噪声预测噪声。
实际预测与未来收益的相关性非常接近零。这也是个人投资者很难长期战胜市场的原因之一，S&P 500 是很强的被动基准。Quant 团队投入大量工作，只为让预测能力略高于零。
高频机构每天和每年执行的交易数量达到天文量级。单笔交易只有很小的边际优势，但重复足够多次后，大数定律会把微小概率优势转化成足以支付奖金的实际金额。这就是业务模型。
Ravid： 市场不是确定性系统。即使掌握全部信息，许多参与者的判断和行为仍会相互作用，模型也未必能给出稳定预测。这是否构成根本限制？
Atlas： 市场按定义就是多方博弈，参与者太多，不可能得到解析解。
高频交易过去不太担心其他人针对你的具体决策，因为交易 Horizon 太短，对手来不及专门设局。近几年情况有所变化，机构开始使用更高级的方法，多方博弈也逐渐成为现实问题。涉及具体做法的部分我不方便展开。
XTX 如何用 AI 预测市场
Ravid： 从外部看，量化公司过去对模型很保守，常用的方法也比较简单，没有追随机器学习前沿。现在不少机构开始采用新架构。你认为这种变化会持续吗？
Atlas： 生成式 AI 对交易的收益已经得到证明。XTX 和少数其他公司交付过实际效果，人们会相信自己看到的结果。NeurIPS 上的变化也很明显。
整个金融行业都对 AI 兴奋，因为任何能增加利润的工具都会得到关注。不同机构的使用方式不同：有些用 LLM 自动化工作流；有些让 Foundation Model 阅读互联网和社交媒体，从情绪等另类数据中提取公司信号；XTX 和部分竞争者则自己训练模型。
我们训练的 Foundation Model 不以语言为对象，而是处理交易所提供的大量多变量时间序列。这些连续数值流和符号语言不是同一种数据。
各家公司对“AI”的定义和目标不同，但机构都在尝试使用它，而且一些公司已经得到收益。银行和交易公司正在建立 AI Center，也在集中招聘 AI Researcher。
金融行业跟进技术浪潮通常比科技行业慢，变化也更平滑。并非所有投资都会成功，科技公司的大模型项目也一样。但金融 AI 已经出现成功案例，这会让更多公司持续投入。
金融从业者不会被论文说服，他们要看 Annual Return。现在回报证据已经出现，行业会继续向这个方向移动。
最重要的可能不是算法
Ravid： 哪类问题适合用神经网络解决？符号学习能否推广到更广范围？
Atlas： 符号学习属于“为推理建立数学基础”这个更大的问题。推理包含大量逻辑和离散结构。神经网络怎样学到这些结构，包括 Chain of Thought 为什么有效，仍像一个奇迹。
我们正在和同事研究，基于代码的表示能否看作多种逻辑的叠加，希望现有理论能够延伸过去。实践上的推理研究也会继续，不需要等理论全部完成。两边满足的是我不同的研究兴趣。
Ravid： 长期看，Chain of Thought 和今天所谓的 Reasoning Model 是必需的吗？还是未来模型可能绕过它们？
Atlas： 先要明确“推理”指什么。如果你说的是 Chain of Thought，我不确定这套算法在数据足够好时是否必不可少。所有算法都可以看成在高维数据空间里搜索。好的架构和优化方法提供更好的归纳偏置与初始化，让搜索更快；如果你能从目标分布持续得到优质样本，随机采样也可能到达目标。我没有相关理论，只能谈自己的经验。
我在 XTX 也参与大模型训练。这个经历让我的注意力从高度定制的算法和模型设计，逐步转向问题设定与数据分布。后两者往往带来更大的性能提升。

<aside>

🟧
「逐步转向问题设定与数据分布」= 把精力从「改架构」挑到「改题目 + 改数据」
他在说自己经验重心的转变：以前花时间在高度定制的算法与模型设计（custom architectures and algorithmic tricks），在 XTX 训大模型之后发现，性能提升更多来自问题设定（problem formulation）和数据分布（data distribution）。
问题设定：预测什么目标（target）、horizon 多长、用什么条件信息（conditioning / context）、用什么损失、怎么评估、什么算成功 —— 即在定义「这道题本身」（task specification）。
数据分布：训练数据取自哪个 universe、哪些时段与市场状态，如何采样、加权、过滤、去重、构造标签、切分 train / val / test —— 即在定义「模型看到的世界」（data curation and mixture）。
为什么他认为这两个更重要

- 接上一段的论点：所有算法都可以看成在高维数据空间里搜索，架构与优化只是提供更好的归纳偏置与初始化，让搜索更快；能持续从目标分布拿到优质样本时，随机采样也可能到达。→ 架构决定「多快找到」，问题设定与数据分布决定「要找的是什么、能不能找到」。
- 与他后面的「架构大收敛」一致：数据足够丰富时，不同算法会逐渐逼近同一个目标（数据分布的某种联合熵），架构差异被摊薄。
- 英文可直接用：Most of the measurable gains came from re-specifying the problem and reshaping the training distribution rather than from customizing the architecture. 和他对 SSM 的「缺归纳偏置」批评矛盾吗 不矛盾，两句话作用在不同瓶颈上：数据还能改时，先改题目与数据（边际收益最大）；当数据的信号上限由市场决定、无法再加时（金融的情形），才靠结构先验提高样本效率。 落到你的 LOB 项目
- 问题设定：horizon 选择、target 是 mid-price return 还是整条条件分布、标签构造、评估用 IC 还是 CRPS / energy score、是否严格因果。
- 数据分布：symbol 与时段的 universe、事件重采样与加权、tokenization 决定的数据表示、train / test 是否跨 regime 切分。
- 可写进论文的实验：同参数、同 FLOPs 下并列三组消融 —— formulation ablation、data-mixture ablation、architecture ablation，分别报告 ΔIC / ΔCRPS，用来支撑 gains come from formulation and data curation rather than the backbone。

</aside>
Ravid： 这和 JEPA 式学习有什么关系？JEPA 把生成过程与学习过程分开，内部状态随时间演化。这个状态也可以看作某种符号模型。
Atlas： 你们是不是要求每位嘉宾都回答 JEPA？节目可以改名叫 JEPA Podcast。
我读过 JEPA 论文，也听过 Yann LeCun 三四次相关演讲。我喜欢 Model Predictive Control，也认同他把 MPC 与 JEPA 表示结合起来的想法。它还和 Koopman Operator 等更早的深度动力学研究紧密相连。
但 JEPA 不是解决深度学习全部问题的万能函数。每种学习算法都要作出假设，这意味着它会牺牲对部分真实数据的适应性。仍然是那句话：所有算法都有错误，其中一些有用。JEPA 属于有用的算法。
Allen： 它会成为 World Model 的核心架构吗？
Atlas： 我不是 World Model 专家。我更相信一种“架构大收敛”：数据足够丰富时，不同算法可能逐渐逼近相同的目标，也许是数据分布的某种联合熵。不同方法只是走向它的不同路径。
Ravid： 也就是说，数据足够好时，不同算法会得到同一类解？
Atlas： 理论上和实践上要分开。神经网络的 Universal Approximation Theorem 在上世纪90年代初就告诉我们，网络可以表示任何函数。如果知道这一点就足够，过去30年也不需要发明 ResNet、Faster R-CNN 等架构。
实验结果还取决于优化能否稳定、超参数是否合适、架构能否适配机器学习系统，以及它对 DDP、FSDP 等分布式训练方式是否友好。这些设计选择对深度学习进展产生了远超其理论地位的影响。
算法各有适用程度，有些更适合今天的硬件和软件生态。产业研究中，我不执着于算法在抽象意义上“正确”还是“错误”，而会判断它在当前条件下是否合适。
Ravid： 实时研究中，那些小调整和算法选择会决定你能不能先做出来。等一条路线成功后，人们很快会找到第二种、第三种做法。
Atlas： 对。我也观察到这种现象。第一个可行方案最难，一旦有人证明任务能完成，替代路线会迅速出现。
为什么市场还无法被合成
Ravid： 你怎么看 Synthetic Data？我看到 Berkeley 的 Alex Dimakis 讨论过一个结果：如果总答案数相同，500个问题、每题两个答案的数据集，可能比1000个问题、每题一个答案的数据集训练效果更好。后者的问题更多，前者却可能包含更多可学习信息。
Atlas： 这个具体例子我第一次听说，需要再看论文。Alex Dimakis 几个月前还分享过一项相关工作：只用一个问题做强化学习。模型先回答，再被要求换一种方法、更有创造性地回答，反复迭代后，小型基础模型也能显著自我改进。
这让我想到高中练数学。最初我想刷大量题，老师却让我使用一本精选题集，同一道题先做一次，再重新做，直到解法比参考答案更好。我没有因此成为优秀数学家，但这段经历让我看到，钻深一道题有时比扩大题目多样性更有效。
一道题只尝一口就放下，很多推理价值还没有被提取。反复咀嚼同一问题，会迫使模型寻找不同路径。虽然我还没有细读你提到的论文，但这个结果符合我的经验，也值得研究。
Atlas： 从实践看，Synthetic Data 是少数领域才拥有的奢侈品，主要包括语言、视觉，也许还有语音。
这些领域先用了很多年把判别模型做好。研究者从分类、识别开始，继续解决标注、检测和分割，并积累大规模高质量数据。ImageNet 最初不是为生成任务建立的，LAION 早期也依赖图文匹配和描述数据。
只有判别模型足够成熟后，我们才进入生成模型阶段，并重用判别阶段建立的数据集。联合分布最难学习，模型只有学到较好的联合分布后，才有能力生成足以反哺训练的数据。
<aside>

🟧
「联合分布最难学」= 判别只要学边界，生成要学整个数据的形状
判别模型只需要条件分布 $p(y\mid x)$，很多时候甚至只需要条件均值 $\mathbb{E}[y\mid x]$；生成模型要的是联合分布 $p(x)$ 或 $p(x,y)$ —— 数据里所有变量之间的依赖、多峰、偏度、尾部、长程相关都要刻画。在分解 $p(x,y)=p(y\mid x)\,p(x)$ 里，判别只负责左边那一项，生成还要把 $p(x)$ 整条学下来。
难在三个具体的地方

- 维度诅咒：判别任务只需一个低维充分统计量或决策边界；密度估计要在高维空间里把整个数据流形定位出来，样本需求随维度爆炸。
- 误差会累积：自回归 / 多步采样时，每步的小偏差在 roll-out 中复合（compounding error, exposure bias），尾部与罕见事件最先失真。
- 评估困难：likelihood 高不等于样本可用，也没有判别任务那种便宜的 ground truth 与 accuracy 式指标。 英文可直接用：Discriminative learning only needs a conditional; generative modeling needs the full joint, including higher-order dependencies and tails, and sampling errors compound. 为什么「学好联合分布」是反哺训练的前提 合成数据想当训练数据用，前提是 $p_\theta\approx p_{\text{data}}$。分布错了却拿它的样本继续训，模型只是在自己的偏差上再训一遍 → 自我消耗循环与分布坡缩（self-consuming loop, model collapse）：方差先收缩、尾部先消失。 语言与视觉做到了，是因为先花了十几年把判别任务和数据资产做好（ImageNet、LAION 这类标注与图文配对），信号丰富且相对稳定，联合分布可学；高频市场做不到，是因为它的联合分布由多方博弈生成、可预测性近于零、而且非平稳 —— 所以他说「谁能可靠地生成每天的股市，应该直接去开对冲基金」。这也呼应他后面那句「不同算法最终逼近数据分布的某种联合熵（joint entropy）」。 落到你的项目（关键区分）
- 你的 mid-training 目标是条件分布匹配：让 $p_\theta(r\mid x)$ 逼近 $p_{\text{data}}(r\mid x)$（CRPS / energy score）。比点预测难，但仍然只是条件分布。
- LOB 世界模型 / order-flow 生成器要的是联合分布：$p(\text{message}_{1:T})$，即事件类型 × 价格档位 × 数量 × 时间间隔 在长序列上的联合，难度高一个量级 —— 这正是他说「金融还没有合成数据飞轮」的位置。
- 因此建议先把生成器定位成诊断工具（stress test、反事实 regime、评估协议）而不是训练数据来源；若真要反哺，必须报告 collapse 诊断：尾部覆盖、自相关与跳档位相关矩阵、真实 vs 合成预训练的下游 IC 对比。
- 可写进论文：Synthetic order flow can bootstrap training only if the generator approximates the joint law of the message stream; matching low-order moments is insufficient, and training on samples from a mis-specified joint induces self-consuming degradation.

</aside>
我现在研究的是一个不幸运的领域：高频交易。如果有人知道怎样训练生成模型，可靠地生成每天的股票市场，请来找我。能做到这一点，也许你应该直接开自己的 Hedge Fund。
符号是压缩的终极形式
Atlas： 我长期关注低维性。博士阶段的背景是统计信号处理与优化，最早研究压缩感知，后来做稀疏优化、低秩结构和非线性流形。深度神经网络兴起后，原来的方向越来越难发论文，我也进入了过参数化神经网络。但我很快发现，深度学习反而提供了更多研究低维性的机会，例如模型剪枝、Lottery Ticket Hypothesis、低秩方法和 Mixture of Experts。
近几年，我和学生也做了大规模 Foundation Model 的高效训练与推理算法，其中一些工作得到开源社区采用。不过要选过去一年我最喜欢的成果，我会选一篇暂时没有太多实际应用的纯理论论文《Why Neural Network Can Discover Symbolic Structures with Gradient-based Training: An Algebraic and Geometric Foundation for Neurosymbolic Reasoning》。它研究神经网络为什么能够通过梯度下降学习符号方程。
深度学习理论已经研究过梯度动力学怎样把网络推向稀疏、低秩、低熵等结构。难点在于，梯度下降是连续动力系统，而符号结构是离散目标。连续过程怎样学到离散结构，并不显然。
Ravid： 先退一步。所谓符号神经网络是什么？我们为什么要关心？
Atlas： 逻辑、规则和关联关系都可以用符号表达。我想知道神经网络为什么不只学到一个巨大的黑箱函数，反而可能映射到一个低维符号空间。对我来说，这是低维性的终极形式。
剪枝可能把模型压缩50%；低秩微调相对于完整矩阵可以小几百倍。但更彻底的压缩，是把神经网络压缩成非神经网络，把它学到的内容写成符号、公式或教科书里的知识。人类能写下来、背诵和传递的知识，是极高效的压缩形式。我希望把神经网络转换成离散、可读的符号，而不是把一个网络蒸馏成另一个网络。
Allen： 你提到关联和符号，我会想到 Association Rule Mining。我刚接触机器学习时，学过 Apriori 等算法，它们和数据库的关系让我着迷，像是在自动构建数据库。零售商甚至可以用关联规则发现顾客怀孕，并提前推送尿布广告。
你也谈到降维和流形。我认为无监督学习长期被神经网络低估了。降维和低维表示上的聚类都很重要。Segment Anything Model 所做的事情，就接近理想图像聚类算法：它把图像按语义分成合理区域。
婴儿的大部分早期经验也没有标签。他们先发现桌上的苹果和玻璃杯不是同一个物体，再逐步建立概念。
Atlas： 我现在每天都能观察这件事。我有一个一岁的孩子，我会不停指着苹果说“这是苹果”。我不知道他究竟怎样学习，但我确实在努力提供监督信号。
Allen： 从信息量看，你提供的标签仍然很稀疏，远少于他接触的感官数据。
Ravid： 我也相信自监督、无监督和降维方法，但不能把“人类这样学习”推得太远。我们并不知道大脑先天编码了多少信息。少量外部数据也许建立在很强的先验之上。
Allen： Moravec's Paradox 提供了一些线索。人类经过数百万年进化，已经很擅长行走和奔跑；对机器来说，这些能力反而很难学。
Atlas： 所有理论都错，只是其中一些有用。稀疏、低秩、流形和符号从不同数学角度描述低维性，我不认为这些数学对象按原样埋在大脑里。大脑当然存在生物学稀疏性，但人学习时不会在脑中显式计算  范数。
Ravid： 物理实现和可用的最优解是两回事。优化过程、表示方式、数据类型都会约束实际解。我们不能因为人类可能以某种方式学习，就直接要求模型照搬。不过降维、压缩和无监督学习仍是一条值得推进的路线，理由不需要依赖人类模仿。
Atlas： Yann LeCun 很早就用飞机作过类比：人类造的飞机不会像鸟一样扇动翅膀。只要信息论仍然成立，对低维结构的追求就不会消失。
神经网络如何发现符号
Ravid： 你希望符号学习在实践中解决什么问题？
Atlas： 最直接的收益有两个：效率和可解释性。
运行 LightGBM 树通常比运行神经网络容易得多。在资源受限或对延迟敏感的 CPU 任务里，例如网络与拥塞控制，符号方程或决策树比神经网络、强化学习策略更适合部署。
我们早期和相关团队合作，先训练强化学习模型控制 CPU 环境中的网络拥塞，再用符号回归把策略转换成决策树。CPU 上的速度提高了约400至500倍。我们试过的网络压缩方法只能带来10倍或20倍改善，符号转换达到了数百倍。
之后我们转向可解释性。在 OpenAI Gym 的简单视觉游戏环境里，我们训练基于 CNN 的强化学习策略，再把它转换成符号树。模型先以无监督方式把颜色块绑定为对象，再用关联关系和逻辑算子组合这些基本单元并作出决策。
这个过程类似中层视觉：算法自动把图像切分成具有语义的对象块，再让决策模块在这些对象上运算。实验只用了类似简化 Minecraft 的图像，我不认为它能直接扩展到复杂视觉控制基准。但它证明了一种可能性：神经网络学到的策略可以分成感知模块和可组合逻辑。SAM 一类模型也提供了把连续感知转换成离散对象的接口。
Flatiron Institute 的研究者和 Miles Cranmer 等团队也做了很多 Symbolic Regression 工作。我们很喜欢这些成果。大约三年前，我和学生开始反复追问：神经网络作为 Universal Approximator，理论上当然能拟合公式，但它为什么会倾向于干净的符号形式？
牛顿万有引力定律中的关系会出现漂亮的 ，而不是 。如果牛顿用今天的神经网络拟合数据，模型可能给出一个很难读的函数，不会自动告诉他答案就是平方反比。连“干净、简单的函数”该怎样定义，都包含很强的人类先验。
Atlas： 我们从合成数据上的多项式学习开始，目标函数含有输入之间的组合代数关系和逻辑关系。团队在基本群运算和环结构假设下证明：如果数据生成过程遵循代数结构，神经网络可以用梯度下降准确发现底层符号结构。
这让我很兴奋，因为连续动力学最终到达了离散目标。证明使用测度空间分析和函数上的 Wasserstein Gradient Flow。论文包含完整数学细节，这里不展开。
<aside>

🟧
这段在讲什么：用「代数结构 + 测度空间上的梯度流」证明神经网络能发现符号结构
对应他前面提到的那篇纯理论论文（Why Neural Network Can Discover Symbolic Structures with Gradient-based Training）。整段的逻辑是：先把「符号结构」定义成代数结构，再证明连续的训练动力学会收敛到它。

1. 为什么从「合成数据上的多项式学习」开始 多项式是可以完全掌控 ground truth 的玩具任务：目标函数由哪些输入、以何种方式组合生成（compositional algebraic and logical relations）完全由你规定，于是「模型有没有发现真结构」变成一个可判定的命题，而不是靠看可解释性图拍脑袋。真实数据里你永远不知道真结构是什么，所以理论只能从 synthetic 开始。
2. 「基本群运算与环结构」是什么，为什么必须是它们

- 群（group） $(G,\cdot)$：一个集合配一个二元运算，满足封闭、结合、有单位元、有逆元。例：模 $p$ 加法 $\mathbb{Z}/p\mathbb{Z}$、置换群。
- 环（ring） $(R,+,\times)$：两个运算 —— 加法构成交换群，乘法有结合律，两者由分配律连接。多项式天生活在环里，所以「多项式学习」与「环结构假设」是同一件事的两种说法。
- 为什么选代数：它能同时装下「代数关系」和「逻辑关系」—— 布尔逻辑就是 $\mathbb{F}_2$ 上的多项式（XOR = 模 2 加法，AND = 乘法）。于是「符号 / 逻辑」有了严格定义，而不是含糊的直觉。
- 关键前提：数据生成过程本身遵循代数结构（the data-generating process respects the algebraic structure）。有了这个前提，正确解在参数空间里带有对称性（群作用下不变），训练动力学才有机会被这种对称性「吸」过去 —— 这也是 modular arithmetic 上 grokking 研究里网络自发学出傅里叶 / 表示论特征的同一类现象。

1. 为什么「连续动力学到达离散目标」值得兴奋 梯度下降是连续状态的动力系统（continuous dynamics），符号结构是离散、组合的对象（discrete target）。两者之间没有显然的桥：连续优化通常只保证收敛到某个低损失点，不保证这个点「长得像一条公式」。这篇的贡献就是在受限设定下把这道 discrete–continuous gap 补上：continuous gradient dynamics provably converge to a discrete, symbolic solution.
2. 「测度空间分析 + Wasserstein Gradient Flow」是什么技术 这是宽网络的 mean-field（平均场） 视角：把网络看成神经元参数上的一个概率测度 $\mu$，输出写成 $f_\mu(x)=\int \phi(x;\theta)\,d\mu(\theta)$。当宽度趋于无穷、步长趋于零时，SGD 的轨迹收敛到测度 $\mu_t$ 在 Wasserstein-2 空间上的梯度流，满足连续性方程 $\partial_t \mu_t = \nabla\cdot\left(\mu_t \nabla \frac{\delta L}{\delta \mu}\right)$。 好处很直接：损失在参数空间里非凸，但作为测度的泛函往往是凸的，于是可以谈全局收敛。代表工作是 Chizat–Bach 与 Mei–Montanari–Nguyen 的 mean-field 分析。所以他说「用测度空间分析和函数上的 Wasserstein Gradient Flow」，意思是：不再跟踪某一组权重，而是跟踪权重的分布怎样流动。
3. 边界（他下一段自己承认的） 这是存在性证明（existence proof），不是构造性方法：知道训练后的网络里存在符号结构，却没有可靠办法把它提取出来 —— 与 Lottery Ticket Hypothesis 同构的困境。假设也很强：合成数据、代数生成过程、宽网络极限。
4. 对你的 LOB 工作有什么用

- LOB 里确实存在硬代数约束：订单簿的数量守恒、价格档位的全序与 tick 网格、mid-price 与 spread 的恒等式、事件类型对簿状态的确定性作用（近似一个幺半群作用）。这正是「data-generating process 遵循代数结构」的现实版本。
- 可做的检验：训练后探测模型是否恢复了这些恒等式（probing / symbolic regression on hidden states），或直接把它们写进架构或损失作为硬约束 —— 这与前面「归纳偏置」那条连成一条线：结构已知时，与其让模型硬学，不如把结构编码进去。
- 可写进论文：When the data-generating process obeys known algebraic constraints, encoding them into the architecture improves sample efficiency, whereas an unconstrained sequence model must rediscover them from data.

</aside>
Ravid： 你证明了梯度下降能够在离散结构问题上找到正确解？
Atlas： 我们证明了它能够找到一个解，但这仍是第一步，假设很多。最大的限制是它属于存在性证明，没有给出构造方法。我知道符号结构存在于训练后的网络中，却不能直接把它提取出来。
这和 Lottery Ticket Hypothesis 很像：你知道初始化网络里存在一个中奖子网络，却没有可靠方法提前把它挑出来。我们还需要把定理推进到可验证的实际问题，并找到不经过过参数化训练、直接到达符号方程的方法。后者是我想继续研究的 Holy Grail。
Ravid： 你会继续用严格数学证明具体问题，还是放松假设，用近似方法解决更一般的问题？
Atlas： 两条路并行。我们一边做经验性的 Proof of Concept，一边做理论，谁也不用等谁。我喜欢严格理论，但严格理论直接推动经验进展的情况很少。
Allen： 我记得一篇2023年前后的 ACL 论文。作者让神经网络自己生成可解释、成本很低的 Spam Classifier。模型提出关键词或启发式规则，最终形成一组 if-else 条件，近似一棵决策树，实验效果还不错。
直接要求模型生成符号表示，当然没有你刚才的理论那么优雅。你怎么看？
Atlas： 这是一个有意思的方向，但效果取决于模型在相关维度上的 Alignment。语言模型无论收到什么问题都会生成文本，输出数据的质量依赖问题质量。很多论文难以复现，也和 Prompt、基础模型及数据条件有关。
从广义上说，让语言模型生成符号语言也属于 Neuro-symbolic AI。生成的语言可以交给特定领域的符号检查器。我们做过一类工作：让 LLM 提出规划方案，在有合规要求的领域里把方案送给安全检查器，再用检查器反馈进行类似 DPO 的训练，使模型提高合规率。
只要基础模型、问题和反馈数据都足够好，这个流程能形成正向的数据飞轮。现阶段还没有保证，但接口本身有研究价值。
会写代码的数学家
Ravid： 过去进入量化研究的典型候选人可能是数学博士。未来会不会出现更多真正训练过 AI 模型的 ML Researcher？团队会形成怎样的能力组合？
Atlas： 我也负责招聘。团队有一句招聘 Quant Researcher 的话，不是我发明的：我们要“会写代码的数学家”，或者“会讲数学的工程师”。两者也许指的是同一种人。
前沿金融 AI 公司的招聘偏好越来越接近前沿 AI Lab。我们的候选人经常同时拿到 OpenAI、Gemini 等团队的 offer，这在日常招聘中很常见。
金融仍然偏好数学与统计基础。掷1000次硬币之类的题，你可能还会在某家公司的面试里遇到。原因是金融数据噪声很大。即使沿用 Foundation Model 的配方，研究者仍需在结构清晰的语言数据和噪声更强的时间序列之间搭桥，其中涉及大量数学变换。
我的同事包括顶尖数学家、物理学家、Quant Researcher、计算机科学家和电气工程师。没有人拿着“交易学博士”进入行业，大家都来自其他学科，再把数学和代码用于市场问题。
Ravid： AI 工具已经会写代码。未来学生是否还需要学矩阵变换、线性代数和微积分？过去30年，尽可能多学数学通常是更好的职业选择，但这条规律未来是否继续成立并不确定，纯 AI/ML Research 也面临同样问题。
Allen： 所以大家应该尽快考水管工资格证。
Atlas： 那也取决于具身智能 Agent 多快取得突破，没有职业百分之百安全。
Allen： 高强度蓝领工作可能还能维持更久。人类已经能制造巡飞弹等武器替代部分士兵，但机器人完成复杂水管维修反而更难。
为什么选择 XTX
Atlas： 还可以寻找数据不足、私有数据壁垒很强的领域。金融属于这一类。通用公共模型很难穿透机构内部的知识壁垒。金融业在几十年中积累的私有数据和经验具有价值，公共知识不能轻易覆盖它们。
Ravid： 有些比赛给 Gemini、ChatGPT、DeepSeek 一个模拟交易账户，让它们交易两个月。你怎么看？
Atlas： 实验很有意思，但我不会把自己的钱交给它们。
我十年前读博士时几乎没考虑过金融。当时我和许多人一样，觉得量化金融就是线性回归、Lasso 和决策树，薪酬很高，但研究技术不够前沿。我那时想做计算机视觉和机器学习。
现在金融业正在发生真正的变化，市场盈利会持续施加压力，也会筛掉无法创造回报的方法。一个领域可以充满 Hype，却找不到真正赚钱、让业务飞轮转起来的公司。金融 AI 已经完成了概念验证，并进入更成熟的阶段。有人证明 AI 能赚钱，行业将因此扩大投入。这里的研究问题很多，也很难解决，这正是吸引我的地方。
金融 AI 会提供更多岗位，一部分团队会成功。尤其是刚毕业的 AI Researcher，求职时应该认真考虑这个行业。我们需要研究人才，薪酬也相当高。
我曾在加入大模型团队和进入现在的公司之间选择。如果去 Foundation Model Lab，只要有人指导，我相信自己能够学会 Pre-training 或 Post-training。但在 XTX，由于独特数据、充足 GPU 和人才密度，我能够研究全世界只有几十个人接触得到的问题。我一直希望自己的研究路径保留独特性，所以对现在的选择很满意。
暂无评论
</aside>
<aside>

💡
XTX 研究总监：AI 如何重塑金融与交易
狄奥尼索斯之歌
狄奥尼索斯之歌
Aug 26, 2026, 8:44 PM
3 listened
本文整理自 Atlas Wang 2026 年两次关于 AI 与金融的公开讨论：3 月 12 日在 Harvard University 的 AI in Finance 讲座，以及一个月后 AIX Summit 金融 AI 圆桌后的文字整理。正文继续采用原文第一人称表达，仅根据中文阅读习惯调整部分句式与段落，不改变原有观点。
原文：Harvard ｜ AIX Summit
Zhangyang “Atlas” Wang 是 XTX Markets 研究总监，负责纽约的 XTY Labs，同时也是得州大学奥斯汀分校的终身副教授。XTX 是全球领先的算法交易与电子做市机构之一。Atlas 的研究横跨压缩感知、稀疏优化、Foundation Model 和神经符号 AI；在 XTX，他利用机构的私有市场数据、GPU 基础设施和跨学科团队，研究面向金融时间序列的大规模模型。
相比此前更聚焦 XTX 技术路线的访谈，这两次讨论更多涉及 AI 与金融的长期边界，包括金融工作的自动化、AI 对研究能力差距的影响、金融能否复制 LLM 的 Scaling 路线，以及 Data、Model 和 Infrastructure 中什么才是更持久的竞争优势。Atlas 还谈到 Continual Learning、模型可塑性和 Regime Drift，并强调：AI 可以帮助机构更快提取微弱信号，但 Alpha 依然稀缺，市场也不会因此变得容易预测。
TL;DR：Atlas Wang 的核心观点
1.
2.
3.
4.
5.
6.
7.
8.
演讲精编
第一部分：AI in Finance，以及更大的问题
2026 年 3 月，我有幸在 Harvard University 做了一场关于 AI 在量化交易中作用的讲座，当然也重点介绍了 XTX Markets 正在开展的工作。
讨论很快超出了交易本身。现场学生提出的问题延伸到了现代机器学习的前沿、学术界与产业界的关系，以及未来十年 AI 可能如何重塑职业和决策。我很喜欢现场的活力和求知欲。
后来有几个人希望我整理一版更有结构的讲稿，所以我把当时讨论的一些想法写在下面。
哪些金融工作可能被 AI 取代？为什么？
我会先看那些本质上依赖结构化数据做模式识别的岗位。初级信贷分析、合规监控、常规记账，AI 已经能做得更快、更便宜，而且往往更好。量化研究和系统化交易中的一部分工作也属于这一类。回测、信号生成和执行优化等环节，迟早会被自动化和商品化。
就目前而言，最先被自动化的是入门层级的工作，比如初级分析师。高级岗位可能也不远了。
不过，说得公平一点，即使在量化与交易中，仍有一层能力是 AI 没有解决的。提出真正新颖的假设，或者理解一个真正未曾见过的情境，仍然是人的工作。历史上不止一次出现过量化研究崩溃，而主观判断和基本面研究胜过所有基于数据训练的模型。
更广泛地说，我们正在看到人才金字塔被压缩。过去需要十个人的团队，有了 AI，可能只需要两个人。这不是消失，而是重新定义。
人类判断在哪些地方仍然不可替代？我认为是那些高风险、强关系型的工作，比如并购、困境资产和董事会层面的战略。处在重大转型交易中的客户不只想要正确答案，还希望有人愿意接住这个答案，并对它负责。这种信任和责任很难被取代。
对同时考虑学术界和产业界的学生来说，两者应该怎样选择？学术界在 AI/ML 研究上有什么独特优势？
这个问题我自己也想了很久。我的粗略判断是：如果一项研究需要产业界无法承受的耐心，学术界更有优势。
产业研究天然更看重短期回报。它必须拿出回报，现实就是这样。深层理论或十年后才可能有用的问题，往往得不到足够投入。状态空间模型（State-Space Model，SSM），比如 Mamba、DeltaNet 等状态空间与循环序列模型，都来自大学。最终取代现行最佳实践的想法，几乎总要经历很长的孕育期。产业界等不起，学术界可以。也可以回想一下神经网络在 2012 年以前经历了什么。
所以我的经验法则是：如果你喜欢基础理论，或者研究的问题要一两年以上才可能见到回报，学术界更合适。反过来，如果你想做 LLM、Agent、量化金融，或者任何当下热门的方向，产业界更合理。
LLM 难以理解解空间的拓扑，是否会从根本上限制推理和规划？
我的直觉是肯定的。一维序列范式与高维几何和拓扑推理之间存在根本错配。把集合、树、图等结构展平成 Token 序列，等于给原本没有线性顺序的数据强加一种线性次序。拓扑不关心先后，而关心连接、孔洞、分支和空间的形状。这些都无法自然地用一维序列表达。
模型最多只能隐式地逼近这些结构，或者借助复杂的位置编码，但我认为这些方法既脆弱又难以解释。做规划时，如果表示本身无法编码解空间的拓扑，模型其实是在盲目规划：它在做插值，而不是推理。
SSM 确实很有意思，其中很多工作也来自学术界。但我不会夸大它的作用。SSM 本质上仍是一维递归。Mamba 的输入依赖动态机制（Input-dependent Dynamics）确实是重要进步，但更丰富的循环状态，并不意味着模型天然拥有理解几何结构所需要的归纳偏置。你仍在要求一个序列模型从数据中学出拓扑，这个跨越很难。

<aside>

🟧
归纳偏置（inductive bias）= 模型「先天」带的偏好
有限数据永远不足以唯一确定一个函数，模型只能靠架构、参数化、损失、优化器里预置的假设去挑一个 —— 这些假设就是归纳偏置。它不是模型学到的东西，而是模型学之前就自带的「偏见」。
英文可直接用：the set of assumptions built into a model — architecture, parameterization, objective, optimizer — that determines which hypotheses it prefers when the data alone does not single one out.
贝叶斯视角：偏置 = 先验 $p(f)$，学习 = $p(f\mid\mathcal D)\propto p(\mathcal D\mid f)\,p(f)$；没有先验就没有泛化（no-free-lunch）。
标准例子（偏置就写在架构里）

- CNN：局部性 + 平移等变（locality, translation equivariance, weight sharing）
- RNN / SSM：因果时序 + 递归状态压缩（causal ordering, Markovian state compression, recency prior）
- GNN / 等变架构：置换等变、关系结构与对称群（permutation equivariance, SE(3)-equivariance）
- Transformer：结构偏置很弱（注意力对集合近乎置换等变，次序全靠位置编码补）→ 偏置弱就靠数据量换
- 正则化与优化也是偏置：L2 偏好小权重、early stopping、dropout、SGD 的 implicit bias 这一句到底在说什么 Wang 的论点：Mamba 的 input-dependent dynamics 提高的是容量 / 表达力（capacity, expressivity），不是结构先验（structural prior）。一维递归天然假设「数据有线性顺序」，而集合 / 树 / 图 的真实结构是连接关系、不是先后关系；于是模型只能「从数据里硬学拓扑」，样本效率差、易脆、难解释。 关键区分：capacity ≠ inductive bias（能表示 ≠ 学得到）。可直接写进论文：A richer recurrent state increases capacity, not the structural prior; a 1-D recurrence still imposes a sequential order on data whose true structure is relational. 对你 LOB 工作的用法 LOB 的价格档位结构、跳资产关系正是被 tokenization 强行线性化的拓扑，因此可以论证「加 relational / geometric inductive bias（GDN 混合、跳档位 attention、等变层）比单纯放大参数更有效」。论证句式：At matched parameters and FLOPs, the hybrid variant improves X, indicating a better-matched inductive bias rather than additional capacity.

</aside>
从架构上看，更好的方案可能会直接在合适的几何域中工作，比如图网络（Graph Network）或等变架构（Equivariant Architecture）。
不过，我们不必只从架构入手。工具调用和外部记忆已经减轻了一部分压力，因为它们把拓扑问题外部化了。模型不必把所有几何结构都压进权重或隐藏状态，而可以查询外部记忆，或者调用天然理解这些结构的工具，比如图数据库、符号推理器或搜索算法。模型由此变成协调者，而不是亲自完成几何推理的主体。这可能无法从根本上补上差距，但在实践中很有意义。
只建立 World Model 够吗？Agent 是否需要进入真实环境学习？
不够。在任务和奖励越来越难验证的情况下，很多任务的目标不只是把世界预测得更准确。它们还涉及价值判断、社会关系和策略选择。就算模型能够完美描述世界，如果没有结构化的评估或验证机制，也推不出可靠的策略。自动形式化一类方法正试图把模糊任务转成至少部分可检查的问题。
与其继续“榨 RL 这颗柠檬”，也可以让具身 Agent 进入环境，哪怕先进入 Sandbox，从真实、可验证的奖励中学习。这样可以获得真实奖励和因果反馈；与来自静态数据的合成奖励或代理奖励相比，这些信号稳健得多。问题是，真实世界的交互既慢又贵。更可行的路线可能是混合系统，让模拟、形式验证和现实世界交互共同训练。
Continual Learning 还缺什么？
这是一个令人兴奋、发展很快的领域。我目前最关心两个问题。
第一，整个领域过度关注遗忘，更深的问题却是模型会逐渐变得陈旧，最后永久失去可塑性。我们才刚开始触碰这个问题。我把它称为 AI 的“老化问题”。几乎所有 Continual RL 方法都偏向“稳定性—可塑性困境”（Stability–Plasticity Dilemma）中的稳定一侧，比如 弹性权重固化（Elastic Weight Consolidation）、经验回放（Replay）和渐进式神经网络（Progressive Networks）。这些方法主要都在防止灾难性遗忘。
实验显示了更令人担忧的现象：网络不只会遗忘，还会僵化。随着训练推进，神经元开始休眠，表征发生坍缩，梯度越来越失去意义，Memory Buffer 也可能变得“太满”。即使移除所有防遗忘约束，它们既学不进新东西，也忘不掉旧东西，只是“死了”。灾难性遗忘至少还是一个网络活着时才会出现的下游症状；可塑性丧失会杀死 AI。我们既没有解释这个过程的机制理论，也没有原则性的解决方法。
第二，真实环境不会弹出一个标记，告诉你“任务已经切换”。世界在没有标注的情况下持续漂移，Task 或 Regime 本来就是人为定义的概念，金融市场就是很好的例子。更深的问题在于，我们甚至没有对 Continual Learning 中的“任务”给出清晰定义。
如果再往前推一步，也许正确答案是停止用 Task 来思考，转而构建这样的策略：它不断检测分布信号，并据此调节自己的学习过程，不必先把变化定义成某项任务、某个事件、某种情境或一次市场状态切换。
第二部分：AI 会怎样重塑金融与交易？
就在分享 Harvard 那次讲座一个月后，我又有机会参加 Asian American Scholar Forum 组织的 AIX Summit，与 Andrew Lo、Argyro Tasitsiomi、David Wu 和 Yin Luo 一起讨论这个领域最前沿的一些问题。
Juhua Zhu 主持了这场一小时的对话，几位嘉宾提出了不少尖锐而新鲜的观点。会后，很多听众继续来找我们讨论。他们的热情促使我再次把其中一些想法写下来，与更多人分享。
市场微观结构：AI 增加了无效，还是消灭了无效？
先把“无效”这个词说清楚，因为它经常被滥用。所谓市场无效，并不是说价格错了，而是分散的信息进入价格需要时间，也需要资本。这是一个物理问题，不是一个 AI 问题。
AI 加快了信息被价格吸收的速度，也提高了模型能够提取的模式复杂度，但它没有改变市场无效存在的根本原因。
在市场微观结构层面，AI 极大压缩了单个信号的半衰期。五年前，一个不错的统计优势也许能存活一年。今天，业内所有人都知道，它的寿命短得多。从发现、部署到衰减，这个反馈循环已经收紧了一个数量级。
结果是，市场平均来看变得更有效，但也更加脆弱。所有人都更接近噪声下限，剩余的信号越来越高维，也越来越难解释。
我不认同市场会单向走向完全有效。每次技术发生变化，新的无效都会出现在接缝处：不同资产类别之间、不同时间尺度之间，以及约束不同的参与者之间。AI 不会消灭无效，只会重新分配无效。
数据、模型和基础设施，哪个才是最持久的优势？
基础设施。我愿意为这个与时下流行看法不同的答案辩护。
数据是一种带有半衰期的商品。今天的私有数据，迟早会被复制，或者其中的优势会被套利掉。模型确实正在商品化，前沿架构往往几个月内就会公开。只有基础设施会复利。严肃研究所需的大规模算力、低延迟执行栈，以及运行那些足以让小公司破产的大量实验的能力，都需要大量资本，很难复制，也支撑着所有上游工作。
长期获胜的机构，是那些有能力承受大量错误的机构。这是一个基础设施问题。
关于 AI Trading，哪一种路线可能无法兑现，哪一种信念可能是错的？
我想到的是那种从原始数据直接生成仓位的端到端系统。这类系统在回测中常常表现得很好，到了实盘却经常让人失望。
这个行业还有一种常见现象：真正拥有可用系统的人很少公开谈论它，而公开谈论的人往往没有真正可用的系统。先把这个现象放在一边。如果问题确实是端到端系统无法在实盘中交付，原因并不神秘。
回测假设未来大致来自与过去相同的分布，市场却不断违反这个假设。深度网络很擅长拟合你展示给它的任何分布，包括那些不会重现的部分。端到端系统可以在回测中表现很好，到了实盘却会失效。
语言模型的 Scaling 能够奏效，因为语言中的信号丰富而且相对稳定。在市场里，Scaling 往往只是把噪声记得更精确。最先出问题的是信噪比。模型表达能力越强，这个问题反而越严重。
这主要不是工程问题，而是这个领域自身的结构特征。瓶颈不在算力，而在数据本身的熵，以及真正可预测信号的稀缺。金融不像语言那样存在一条会持续奖励更大模型容量的 Scaling Law。它会碰到一个下限，这个下限由市场每单位时间能产生多少真实信号决定。无论怎样建模，都无法凭空制造更多信号。
把金融视为另一个等待“GPT 时刻”的领域，是一种类别错误。语言包含很深的可压缩结构：语法、语义，以及数万亿 Token 中反复出现的模式。市场只有微弱的预测相关性，因为任何简单、可重复的结构都会被套利掉。两者处在完全不同的信息论环境中。
照搬 LLM 的路线——收集更多数据、投入更多 Fine-tuning、继续扩大模型——在金融中不只是效果更差，还可能反过来伤害你，因为增加的容量往往被拿去拟合噪声。
AI 不会以广泛、可靠的方式预测市场。它能做的是从更复杂的数据中多提取一点信号，也许比竞争者稍快一点。这是一门真实存在的生意，但不是魔法。任何向客户承诺更多的人，都只是在兜售故事。
AI 会缩小能力差距，还是放大最优秀的人？
它会放大差距，而且差距扩大的速度比很多人意识到的更快。
“AI 让所有人都能使用工具，所以人人都能完成过去只有专家才能做的工作。”这个朴素叙事把方向说反了。当工具免费时，差异不再来自有没有工具，而来自使用工具的人，或者说他们的品味。品味不会被民主化，它会复利。
平庸的研究者借助 AI，可以大量生产看似可信、实际错误的工作。这些错误很难发现，因为输出看起来很专业。优秀的研究者借助 AI，会在真正重要的问题上快十倍。两者的差距没有缩小，而是在爆炸式扩大。处在分布中部的人会遇到真正的麻烦，顶端的人不会。
对学生而言，不要在工具熟练度上投入过多，因为每个人都会拥有工具。应该投资于那些无法规模化复制的能力：判断什么是真实的、数学品味，以及在简单问题随处可得时仍愿意用一年攻克一个难题。
AI 会改变什么，又不会改变什么？请各举一个例子。
会改变的是交易公司的经济结构。少数拥有强大基础设施的机构，将捕获不成比例的价值。
不会改变的是 Alpha 的根本稀缺性。AI 无法在不存在信号的地方创造信号。市场依然具有对抗性、非平稳性，也依然会不断让人保持谦逊。忘记这一点的机构，会为没有忘记这一点的机构提供利润。
还有一个补充判断：下一次系统性事件最可能的诱因，不是违规交易员，也不是交易中的“乌龙指”，而是多个模型同时失效。许多机构运行同一种架构的不同变体，使用彼此重叠的数据，并在同一时刻、同一方向上犯错。监管者还没有准备好，行业也没有。
Reads
</aside>

---

## Grok

这三期访谈把你那套工厂 **拆成两半**：一半被 HRT 坐实，一半被 Wang 直接泼冷水。

## 被坐实的：不可解释 + 过测试就上

Iain 说得很直：短 horizon 已经在「不必给人讲故事」的世界里；50.1% 重复足够多次就行；模型是一团数字，切几刀能看懂的部分也可能是错觉；**魔法模型外面必须包很多安全层**，因为他们做的是高频双向交易。新版本有上线前审计，盘中对输出做合理性检查，模型 **不直接往交易所发单**。Knightmare 是培训固定节目。

这和你说的「不管特征稳不稳、能不能解释，框架上效果好就上」是同一门哲学——**门禁在分层风控和操作检查，不在因子叙事**。HRT 还加了一句你没写但必须有的：预测和控制分离。神经网络出计划，后面仍是人工审计过的执行与限额。学生 $\pi_k$ 再黑，也不能自己拥有下单权。

算力叙事也同构：电力和整机柜才是瓶颈；优势是把数据采集、训练、部署、盘中忙碌时系统还活着接在一起。这就是「源源不断产模型」的工厂，不是某一个永生网络。

## 被泼冷水的：生成情景当宇宙

Wang 几乎点名你的情景工厂：

> 谁能可靠地生成每天的股市，请来找我。能做到，也许该直接开对冲基金。
> 
> 合成数据是语言/视觉的奢侈品。金融还没有合成数据飞轮。
> 
> 联合分布最难学；生成器偏了再拿它的样本训，是自我消耗。

所以：**「在不同 generative scenario 里过 bug test 就能上」——若 scenario 来自学出来的 $\theta^*$，Wang 认为这正是金融做不到的那一步。** 测试宇宙和真盘的缝，不是工程细节，是领域结构。

HRT 应对制度断裂时，公开讲的也不是「用世界模型造一万个平行市场」，而是：底层事件数据极多、短期限样本密、模型直接读全部市场数据；另类数据和长 horizon 另说。情景更像 **历史回放 + 微扰 + 真盘影子**，不是生成器闭环当唯一考场。

Wang 还说端到端「原始数据 → 仓位」回测好看、实盘差，因为网会记住不会重现的分布。你的 $\pi_k$ 若只在 $\theta^*$ 的回声里过关，就是这个失败模式的工厂版。

## 所以「很像 XTX 内部」要改口

更像的是 **HRT 的生产哲学 + XTX 的研究口味**，不是某一家的源码：

| 你的模块 | 访谈更支持谁 | 修正 |
| --- | --- | --- |
| 大 checkpoint 读全市场事件 | HRT + XTX 都有 | 保留；问题设定和数据分布 > 换骨干（Wang） |
| 成千上万并行假设 | HRT「统一方法打所有市场」+ XTX 基础设施 | 保留；并行的是实验和头，不是一个策略永生 |
| 不可解释就上 | HRT 短 horizon 明确接受 | 必须加：模型不直连撮合、盘中数值检查、分层防御 |
| 生成情景当主测试宇宙 | **Wang 反对金融合成飞轮** | 主宇宙应是历史切片 + 真盘影子；生成器只做压力和洞的探测 |
| 每情景 OPD 出学生 | 两边都没点名 OPD | 可行实现，不是访谈里的词；HRT 更像「预测头 + 传统执行」 |
| 教师偶发更新 | Wang 老化 / 无标签漂移 | 要 CPT，但不能用生成数据反哺教师（collapse） |

XTY 若真有「情景工厂」，最大可能是：**回放真实事件流、改库存/延迟/拥塞参数**，生成模型最多补罕见尾部，且那些 $\pi_k$ 权重更低。把 $\theta^*$ 的 rollout 当成全量 $S$，和 Wang 自己的判断打架。

## 对你这套的可行性，改一句

- 工厂形态可行，而且和 HRT 公开的「神奇机器 + 安全层 + 系统工程」一致。
- 生成器当唯一考场不可行——这是 Wang 写明的金融结构限制，不是你还没把 bug test 写全。
- 「就是 XTX 内部」仍然过满。 外部能钉住的是：短 horizon 放弃叙事、模型与执行分层、基础设施和实验吞吐是 moat、合成市场不可靠、相关失效是下一场系统性事故。OPD 产耗材学生是你这边和 σ0 最贴的实现。

可执行的收口：$\theta^*$ 继续做世界模型，但 $S$ 的主集是 **真实held-out 日 + 参数化扰动**；生成 rollout 只进「压力电池」，过了不能单独获准上线。教师禁止用自己的合成消息再 CPT。风控层按 Iain 的 Knightmare 清单写死。这样才同时吃到两家访谈，而不是只吃到「不可解释也行」那一半。

---

## User

那这里的continual learning 到底要怎么做呢?

---

## Grok

在你这套工厂里，continual learning 不是一个算法，是 **三条更新环，各写各的、各验各的**。混成一条「又 replay 又 TTT 又 EWC」就会假忙。

## 三条环

```text
环 A  教师 θ*     周～月     世界还像不像
环 B  学生 π_k    日～周     这个情景还能不能过门禁
环 C  快状态      前向内     本窗局部非平稳
```

CL 的主体是 **B 的出生死亡**。A 偶发、贵、要防老化。C 不是持续学习，是推理时的门控。

---

## 环 A：教师 CPT（真正的权重持续学习）

**做什么：** 新真实订单流为主，旧流只留小锚，rewarm 后再降 LR，但不要降到 0。

**数据：**

$\rho=0\sim0.10$ 的真实旧窗，或冻旧 $\theta^*$ 对 **真实新窗** 打 KL。

禁止用 $\theta^*$ 自己滚出来的消息再训 $\theta^*$（Wang：没有合成飞轮；否则 compound error 会写进教师）。

**切段：** 按日历或事件，不按 task id。COVID、2024-08 单独报，不当一个「regime 类」。

**每段必看：**

| 看 | 过线 |
| --- | --- |
| 新窗 NLL | 下降 |
| 旧窗 NLL | 升幅可接受 |
| 早 vs 晚检查点、同预算探针 | 晚的不能明显更难学（老化） |
| 闭环 $E(440)$ size/Δt | 不能比 CPT 前更歪 |

旧窗升 → 加 $\rho$ 或教师 KL。

探针变差 → 加 WD / 谱约束 / 提高 rewarm，不要先整网 reset。

只有 slope 好看 → 当没过（PR#22）。

GDN 门控、TTT **不代替** 这一环。

---

## 环 B：学生种群（工厂的持续学习）

**做什么：** $\theta^*$ 冻住。每个情景 $s$ 上 OPD 出一个小 $\pi_k$。过门禁上线，过期或失效就扔。

**情景主集不是生成宇宙：**

- 主：真实 held-out 日 + 库存/延迟/拥塞微扰
- 辅：$\theta^*$ 生成的压力轨（含头 500 条错误定态）——只能否决，不能单独批准上线

蒸馏必须学生自己走状态。Teacher-forcing 的 $\pi_k$ 禁止进生产队列。

**门禁（HRT 那层，不是因子故事）：**

- 生成压力 + 真实切片都过
- 输出落在数值箱里（盘中合理性）
- 不直连撮合，经限额/风控
- 同簇 $\pi_k$ 不得在同一情景族上一起满分（相关失效）

这里不谈可塑性，谈 **通过率、存活天数、簇相关性**。特征稳不稳、能不能讲故事，不是门禁。

教师一旧，整厂学生再勤也是在蒸过期世界——所以 B 勤、A 仍要偶尔转。

---

## 环 C：本窗快适应（可以有，不算 CL）

冻 $\theta^*$，只动 LN / 低秩 / GDN 状态，几步自监督。只报 **这一窗** NLL 和 $E(m)$。

跨日旧窗变差，或只有把 C 开很大新年份才能学 —— 回去转 A，不要把 TTT 当成持续学习。

---

## 无标签漂移怎么接到三条环

一个统计量，三个动作：

$$
s_t=D(\hat P_{\text{近}},\hat P_{\text{更早}})\quad\text{真实日度 token / NLL，不用生成器}
$$

| $s_t$ | 动哪 |
| --- | --- |
| 小 | 只刷 B；A 按月例行 |
| 中 | B 全量重蒸；A 的 $\eta,\rho$ 略加 |
| 大（事件切片） | 先重拟合/重蒸 B + 影子盘；A 开短 CPT，看探针 |
| 生成 $E(440)$ 升 | 修编码或短闭环，不是加 EWC |

不要训练一个「现在是哪次 regime」分类器再切任务。

---

## 不要做的

- 一个 $\theta$ 又当世界又当仓位，端到端上线（两边访谈都反对）
- 用合成订单流 CPT 教师
- 推理时反传骨干当「在线 CL」
- 只优化遗忘或只优化 slope
- 拿 scratch 对比证明「没有老化」

---

## 开工顺序

1. 把 A01 的生成定态压到可接受，否则 B 的辅情景是脏的。
2. 定 B 的门禁：真实切片为主、生成压力为辅、分层风控、簇相关。
3. 跑一次 A 的试点：$\rho\in\{0,0.05,0.10\}$ × 两档 rewarm，报四张表。
4. C 做冻骨干对照，能留下再留。

一句话：持续学习 = **真实数据上偶发 CPT 教师 + 真实/微扰情景上不停产可扔的学生 + 风控当唯一门禁**。生成器是压力机，不是世界的替代品，也不是教师的训练集。

---

## User

技术路线呢?

---

## Grok

按你们现有栈写一条能开工的技术路线，不换赛道。

## 栈上已经有的

- 教师：LOB 基础模型（Mamba-3 / GDN 混合），26-token 消息，词表 2112
- 病：闭环头 500 条，size / log10_dt 滑进错误定态（A01）
- 评测：字段边缘 $E(m)$、slope+level、487 tickers、held-out 月
- 产品形态：$\theta^*$ 出世界，$\pi_k$ 过门禁上线

路线按依赖排，后一层吃前一层的输出。

---

## Phase 0 — 把教师变成可用的世界（先修编码与定态）

目标：生成宇宙不要明显歪，否则后面情景和蒸馏全是假的。

1. B7b per-offset embedding：数字 token 按槽位拆向量（size 高/低位、Δt、绝对时间不再共享）。
2. size 几何：换进制 / 对数桶 / 标量旁路，消掉 99→100 的双槽跳变。
3. 短闭环训练：真实前缀后自生成 $K\le 500$ 条，字段级 CE 或对冻 draft 的字段 KL。
4. 验收：size、log10_dt 的 slope 和 level@440 双过 draft；TC/joint 不崩。

过不了就不要开工厂。这一期没有 CL，只有「教师配得上被蒸」。

---

## Phase 1 — 教师 CPT（环 A）

目标：$\theta^*$ 能跟真实日历走，且不变死。

- 数据：新真实流 $(1-\rho)$ + 旧真实水库 $\rho\in\{0,0.05,0.10\}$；或 $\rho=0$ + 冻旧教师 KL（只在真实新窗上）。
- 禁止用自己 rollout 再训 $\theta^*$。
- Schedule：rewarm 到预训练 peak 的 0.3–0.5，余弦/WSD，尾 LR≠0；每段重置 Adam 矩。
- 优化：先 AdamW + 略增 WD；秩掉了再加谱约束或 L2-toward-init。
- 诊断：dormant、$er_2$、|w|_2、optimization readiness。
- 验收四张表：新窗 NLL、旧窗 NLL、早/晚检查点同预算探针、闭环 $E(440)$。

切段按年/事件，不按 task 标签。GDN $\alpha_t$ 只记日志当漂移特征。

---

## Phase 2 — 情景与门禁（工厂骨架）

目标：有一个不是生成器独裁的考场。

**情景主集 $S_{\mathrm{real}}$**

- held-out 真实日，按 ticker×日抽样
- 微扰：库存、延迟、队列深度、开盘/拍卖掩码
- 必含：深度 $m\in\{100,500,1500\}$ 的真实前缀再接学生闭环

**情景辅集 $S_{\mathrm{gen}}$**

- $\theta^*$ rollout，故意走到饱和区
- 只用于否决，不能单独批准

**门禁（预测与下单分离）**

- $\pi_k$ 只出计划（公平价、急度、尺寸上限）
- 传统执行+限额+盘中数值箱（HRT 层）
- 同簇同向：一组 $\pi_k$ 在同一情景族上不得一起满分
- 真实事件切片抽检（COVID、2024-08）

这一期可以还没有 OPD，先用简单头在 $S_{\mathrm{real}}$ 上跑通电池。

---

## Phase 3 — OPD 产学生（环 B）

目标：一个教师，源源不断小 $\pi_k$。

- 学生：小 MLP / LoRA 头 / 蒸馏小网，延迟预算写死
- 数据：学生在 $s\sim S_{\mathrm{real}}$ 上自己走状态
- 损失：$\mathrm{KL}(\pi_k(\cdot\mid x_{\mathrm{on-policy}})\,\|\,\pi_{\mathrm{T}}(\cdot\mid x))$ 或字段级软标签；教师是冻 $\theta^*$ 上的预测头，不是消息生成器直接当 $\pi$
- 稀奖励、要开新解时：仿真里短 RL（GRPO 类）只打教师，再 OPD 压进学生——学生不大改骨干
- 生命周期：过门禁上线 → 存活天数、簇相关超标或真实切片掉点则下线重蒸

CL 在这里就是队列：每天/每周批量产、测、杀。不解释特征。

---

## Phase 4 — 漂移调度（无 task）

$$
s_t=\mathrm{KL}(\hat P_{\mathrm{近}}\,\|\,\hat P_{\mathrm{更早}})\quad\text{真实日度 token}
$$

| $s_t$ | 动作 |
| --- | --- |
| 低 | 只刷 B |
| 中 | B 全量重蒸；A 的 $\eta,\rho$ 上调一档 |
| 高 | B + 影子盘；A 开短 CPT + 探针 |
| $E(440)$ 升 | 回 Phase 0，不动 EWC |

可选：日度 NLL 残差上的 BOCPD 只调 A 的 peak LR，不切任务。

环 C（TTT / 只适应 LN）冻 $\theta^*$ 对照；只留「本窗增益、跨日不伤旧窗」的变体。

---

## 和论文名词的对应（避免再搅）

| 路线上的块 | 用 | 不用当主力 |
| --- | --- | --- |
| Phase 0 | 短 Self-Forcing / 字段匹配 | 连续 DMD、超长 Self-Forcing++ |
| Phase 1 | Ibrahim 式 rewarm+小 replay、Wang CPT 律试点 | CBP/ReDo 默认、合成反哺 |
| Phase 2 | 真实切片+微扰、分层风控 | 生成器当唯一宇宙 |
| Phase 3 | OPD、小头 | 端到端 $\theta^*\to$ 仓位 |
| Phase 4 | $s_t\to\eta,\rho$、学生换血 | regime 分类器、线上反传骨干 |

---

## 季度级里程碑

- Q1：Phase 0 过双坐标；CPT 试点三档 $\rho$ 出四张表。
- Q2：$S_{\mathrm{real}}$ 电池 + 一个监督头跑通门禁；生成压力只做否决。
- Q3：OPD 流水线日产 $\pi_k$；相关失效监控上线。
- Q4：漂移调度接 A/B；早/晚检查点可塑性做成固定回归。

人员上按 Wang 的「会写代码的数学家」切：一人盯 tokenization+定态，一人盯 CPT+探针，一人盯情景电池+风控，一人盯 OPD 队列。骨干别每个季度换。

一句话：先让 $\theta^*$ 在真实闭环里不像假市场，再在真实情景上蒸可扔的小 $\pi_k$，用 $s_t$ 决定本周刷学生还是本月动教师。这就是这条线的技术路线。

---

## User

搜索一下awm 和refl 以及seedance2.0 的posttraining方法 tweets

---

## Grok

Request was interrupted by the user.

---

## User

继续

---

## Grok

接着把路线落到 **损失、接口、失败回退、前八周**。前面四期顺序不变。

## 四个接口（对上 σ0 决策页）

```text
θ*   : generate(prefix, m) → message_{1:m}
head : predict(state) → 短 horizon 目标分布（方向 / 到达 / 公平价）
π_k  : plan(state, inventory) → 报价计划（不发单）
gate : risk(plan, limits, box) → 发 / 改 / 拒
```

训练只碰前三个；第四个只升级规则和阈值，不走梯度。任何把 gate 学进 $\theta^*$ 的实验单独开，不进主干。

---

## Phase 0 损失（字段，不是整句 CE 独大）

一条消息 26 token，五个决策字段。闭环段 $t>L$：

$$
\mathcal{L}_{0}
=\sum_{f\in F} w_f\,\mathrm{CE}(p_\theta(f\mid x_{<t}^{\mathrm{own}}),\,f^{\mathrm{real}})
+\lambda_{\mathrm{kl}}
\sum_{f}\mathrm{KL}\big(p_\theta(f\mid x^{\mathrm{own}})\,\|\,p_{\mathrm{draft}}(f\mid x^{\mathrm{own}})\big)
$$

- $x^{\mathrm{own}}$：真实前缀 $L$ + 自己写的 $K\le500$
- $w_{\mathrm{size}},w_{\Delta t}$ 明显高于 type/direction（time 已占 teacher-forced 损失的 76%，再均匀 CE 会把病淹没）
- $\lambda_{\mathrm{kl}}$ 先 0，level 降、joint 崩再开

B7b 与 size 几何是架构 diff，同一套 $\mathcal{L}_0$ 重跑 A01。过线仍是双坐标 + TC。

---

## Phase 1 损失（真实流 CPT）

$$
\mathcal{L}_{A}
=\mathbb{E}_{x\sim\mathcal{D}_{\mathrm{new}}\cup\rho\mathcal{D}_{\mathrm{old}}}
[\mathrm{CE}(\theta;x)]
+\mu\,\mathbb{E}_{x\sim\mathcal{D}_{\mathrm{new}}}
[\mathrm{KL}(p_\theta(\cdot\mid x)\,\|\,p_{\theta^*_{\mathrm{old}}}(\cdot\mid x))]
$$

- $\rho\in\{0,0.05,0.10\}$，$\mu$ 与 $\rho$ 不要同时从大开；先扫 $\rho$，旧窗仍升再加 $\mu$
- $x$ 只来自真实 tick
- 每段结束：ckpt_A_YYYYMM + 四张表进同一 JSON（新/旧 NLL、探针 $R$、$E(440)$）

探针脚本冻两份拷贝，**同一 $N$、同一 schedule、同一切片**，禁止拿 scratch 当 ABSENT。

---

## Phase 2 情景对象（先写 schema）

```text
Scenario:
  ticker, date, start_seq
  prefix_real[L]
  perturb: {inv, delay_us, queue_scale, session_mask}
  depth_m: 100|500|1500
  split: real_holdout | stress_gen
  label_for_gate: box bounds, fat-finger, self-trade
```

电池输出不是「这个 $\pi$ 的 Sharpe」，是：

- 计划违规率（穿价、超库存、自成交）
- 短窗 IC / 价差捕获（只在 real_holdout 上算成绩）
- 簇：同一 perturb 族里 top 学生的符号相关

stress_gen 只有 fail 票，没有 pass 票。

---

## Phase 3 OPD（学生必须小）

教师侧：冻 $\theta^*$ + 冻 head（Phase 1 末的预测头）。

学生：LoRA 或 10M 级头，输入 = 教师表征或薄状态，输出 = 与 head 同形状的计划分布。

$$
\mathcal{L}_{B}
=\mathbb{E}_{s\sim S_{\mathrm{real}},\;x\sim\pi_k}
\big[\mathrm{KL}(\pi_k(\cdot\mid x)\,\|\,\mathrm{sg}[\pi_T(\cdot\mid x)])\big]
$$

- $x$ 学生自己滚；停梯度打在教师
- 需要开新解时：只在仿真里对 教师头 做短 RL，再 OPD；学生不吃稀奖励
- 产物：pi_k 二进制 + scenario_hash + 门禁报告；队列按日 GC

上线路径：π_k → gate → 执行。禁止 π_k.send().

---

## Phase 4 调度器（一个文件）

```text
s_t = KL(token_hist[t-w:t] || token_hist[t-2w:t-w])   # 真实日
if E440_size 升:     job = phase0
elif s_t > τ_high:   job = A_short_cpt + B_redistill + shadow
elif s_t > τ_mid:    job = B_redistill; bump η,ρ one notch
else:                job = B_only
```

阈值 $\tau$ 用试点期的日度 KL 分位数定，不训练分类器。

---

## 失败回退

| 症状 | 回退到 |
| --- | --- |
| level 好、TC 崩 | 关 $\lambda_{\mathrm{kl}}$，查 size 编码，禁止再上 DFM 式抹平 |
| CPT 后 $E(440)$ 比以前差 | 丢掉该段 A，回上一 ckpt，$\rho\uparrow$ 或 $\mu\uparrow$ |
| 探针 $R<1$ 且 CI 不含 1 | A 加 WD/谱；B 照常产，但标记教师「老化」 |
| 一簇 $\pi_k$ 同向满分成绩 | 整簇下线，扩 $S_{\mathrm{real}}$ 扰动轴 |
| 真实切片挂、生成压力全过 | 门禁权重切到 real；生成器降权（已是辅集） |
| TTT 本窗好、旧日伤 | 关环 C |

---

## 前八周（一个人也能推）

**W1–2** B7b + size 几何，只跑 A01 双坐标。

**W3** $K=200$ 短闭环，字段加权 CE。

**W4** $K=500$；写死「双坐标 + TC」报表。

**W5** CPT 试点：1–2B token，3×$\rho$，两档 rewarm，出四张表。

**W6** Scenario schema + real_holdout 电池，监督头当假 $\pi$。

**W7** 门禁箱 + 簇相关；stress_gen 接上但无 pass 权。

**W8** 一条 OPD 最小环：单 ticker 族、学生 LoRA、日产 1 个 $\pi_k$ 走完 GC。

八周内不要上 Muon、不要上 GRPO、不要换骨干、不要合成反哺。

---

## 和现有工单怎么并

- A01 / PR#22 → Phase 0 验收，不再当 CL 指标
- B7b → Phase 0.1
- 可塑性早/晚检查点 → Phase 1 探针脚本
- 决策头 IC → 只在 real_holdout 上报，进 Phase 2/3
- GDN $\alpha_t$ → Phase 4 的旁路特征，不当 job 开关

论文可写的三刀（Wang：formulation / data / arch）：同 FLOPs 下并排 **编码+闭环**、**CPT 配比**、**骨干替换**。预期前两刀 Δ 大于换 Mamba 变体。

下一步最小的一步：W1 的 per-offset embedding 出 A01 表，不要先搭 OPD 队列。工厂在教师还歪的时候转，只会量产能过假考场的 $\pi_k$。

---

## User

搜索一下awm 和refl 以及seedance2.0 的posttraining方法 tweets

---

## Grok

X 上这三件事不要焊成一条官方配方。**ReFL 有清楚算法；Seedance 公开的是 1.0 的 post-training 骨架，2.0 几乎不写训练；AWM 在推特上主要是另一件事（智能体世界模型）。**

## ReFL（有算法、有帖可对）

ReFL = **Reward Feedback Learning**（ImageReward / Xu et al.）：冻奖励模型，在扩散去噪偏后的一步直接把 $-\,r(x_0)$ 反传到生成器，避开 PPO 要的似然。

视频侧两条延伸，X/论文都在用这个名字：

- RewardDance（ByteDance Seed，视觉奖励缩放）：奖励模型当生成式 VLM，ReFL 对齐；文中写明 Seedance-1.0 T2V 上 1B RM +28%、26B RM +49% vs SFT。抗 hacking 靠大 RM 保持奖励方差。
- PRFL：奖励打在 带噪 latent、任意 $t$，不解码 VAE。解决「ReFL 只能在快结束时打像素、运动学不到」的问题。

和你们 DMD 的差别：DMD 是两个 score 之差（分布匹配）；ReFL 是 **一个标量奖励的梯度**（偏好/质量）。Seedance 1.0 报告里的视频 RLHF 更像 ReFL：模拟推理、早停去噪预测 $x_0$、多维 RM 打分、直接最大化线性组合；和 DPO/PPO/GRPO 比过，报告写这种最好。RM 与生成器 **多轮交替**，不是在线一直更新 RM。蒸馏后的 refiner 上也再打一层同样的奖励。

X 上几乎没有「ReFL」热帖；方法活在论文和 Seed 技术报告里，不在 Seedance 2.0 的演示推。

## Seedance 2.0（推特几乎不讲 post-training）

官方发布（2026-02-12）讲的是：统一音视频联合架构、最多 9 图 + 3 视频 + 3 音频、复杂交互可用率、物理。**训练阶段表没有 2.0 专帖。**

能拼上的后训练信息：

| 来源 | 说了什么 |
| --- | --- |
| Seedance **1.0** 报告 | 预训练 → continue training → 精选 SFT → **视频专用 RLHF（多维奖励）** → 蒸馏；RL 还打在超分 refiner 上 |
| RewardDance | 1.0 的 T2V RL 用的就是 ReFL + 可缩放 RM |
| X @bookwormengr | 「SeeDance 2 - RLHF」短线：RLHF 让视频跟 prompt、更好看；给过 pretrain vs RLHF 对比图（低互动，非正式） |
| X @ned714 | Seedance **2.5** 优势写成「海量视频数据做 post-training + 成本」——仍无算法名 |
| 其余 Seedance 2.0 帖 | 几乎全是提示词/成片，0 条 SFT/ReFL/GRPO 配方 |

2.0 相对 1.0，公开增量是 **多模态参考和物理**，不是换了一套新的 RL 字母。合理默认：骨架仍是 **SFT → 多维奖励 ReFL/RLHF → 少步蒸馏**；2.0 多半把奖励维扩到音频同步、多参考一致性、物理可用率。这是推断，X 上没有 Seed 员工画 2.0 的 loss。

## AWM（推特上不是 Seedance 的模块）

检索里和 post-training 沾边的 **AWM = Agent World Model**（Snowflake / UNC，arXiv 2602.10090），不是视频世界模型：

- 从情景名生成 可执行 环境：SQLite + MCP 工具 + 校验代码，交互时不用 LLM 假状态
- 智能体 RL：GRPO，Qwen3-4B/8B/14B，每步 1024 并行环境
- 奖励：步级格式（非法工具 −1 提前停）+ 任务级（库状态差 + LLM judge）
- 卖点：合成环境上训，三个真实 bench 都比 base 涨；代码环境比 LLM 模拟环境信号干净

X 上 Vivek 的笔记帖把流程写全了。另有一个同名 AWM 是 LLM **权重指纹**（对 SFT/CPT/RL 都鲁棒），和后训练方法无关。

**没有** 搜到 ByteDance 把 Seedance 2.0 的后训练叫做 AWM。若你指的是「用世界模型造情景再 RL」，那是 Snowflake 这条 agent 线，和 Seedance 的 ReFL 不是同一篇。

## 三块怎么摆

```text
Seedance 后训练（公开能钉的）
  SFT 精选成片
  → ReFL / 视频 RLHF：多维 RM 打 x0（1.0 已写；2.0 未官方展开）
  → 蒸馏少步；refiner 再打奖励

AWM（Snowflake）
  合成可执行环境工厂
  → GRPO + 步级/任务级混合奖励
  → 迁移到真实 agent bench

和 σ0 工厂的类比
  AWM ≈ 你的情景工厂（但他们用真代码+DB，不用学出来的市场联合分布）
  ReFL ≈ 对生成器打标量偏好，不是 OPD
  Seedance 蒸馏 ≈ 少步学生；教师仍是扩散，不是 AR 订单流
```

X 上结论就一句： **ReFL 是视频后训练的真算法名；Seedance 2.0 的推特不教训练；AWM 在推特上是智能体合成环境 + GRPO。** 没有一条帖把三者写成 Seedance 2.0 的同一套 post-training。

61 sources

---

## User

AWM 可能是某种posttraining 的算法

---

## Grok

对，**有一个后训练算法就叫 AWM**，而且正好和 ReFL、Seedance 同一条视觉生成线。不是 Snowflake 的 Agent World Model，也不是权重指纹。

## AWM = Advantage Weighted Matching

中文帖标题就是「优势加权匹配：让扩散模型的强化学习与预训练对齐」。

核心判断：LLM 的预训练和 RL 都在打 logprob；扩散这边拧了——

- 预训练：DSM / Flow Matching（对 干净目标 的分数/速度匹配）
- Flow-GRPO / Dance-GRPO / DDPO：优化 反向一步转移 的高斯 loglik

他们证明 DDPO 等价于在 **带噪目标** 上做匹配，方差被噪声放大，所以比预训练目标难优化。

AWM 的做法：

- 损失仍是预训练那条 DSM / FM
- 每个样本乘 优势：好样本正权重，坏样本负权重
- 策略梯度一致，但不换目标

和 Flow-GRPO 比：SD3.5-M、FLUX 上 GenEval/OCR/PickScore 质量持平或更好，GPU 小时最高大约 **8–24×**（OCR 上 17.6 vs 416 小时那种量级）。

一句话：AWM 是 **用优势给预训练匹配损失加权的扩散 RL**，不是另起一个 PPO 似然。

## 和 ReFL、Seedance 怎么排

|  | 监督从哪来 | 优化什么 |
| --- | --- | --- |
| **ReFL** | 冻 RM 对 $x_0$ 打标量，梯度反传到生成器 | $-\,r(x_0)$，通常偏后天 |
| **AWM** | 组内奖励 → 优势 $A$ | $A\cdot\mathcal{L}_{\mathrm{FM/DSM}}$ |
| **Flow-GRPO** | 组相对奖励 | 反向转移 loglik（和预训练不一致） |
| **DMD** | $s_{\mathrm{fake}}-s_{\mathrm{real}}$ | 反向 KL / 分布匹配，不是奖励 |

Seedance 1.0 报告写的是「预测 $x_0$ + 多维 RM + 直接最大化奖励」，更像 **ReFL 族**，没点名 AWM。RewardDance 也明确写 Seedance-1.0 用 ReFL。

AWM 是后出现的「别用 GRPO 似然、回到匹配损失」方案，论文实验在 SD3.5 / FLUX，**不是 Seedance 官方配方**。

2.0 推特仍然不写用了 AWM 还是 ReFL。

## 同名干扰（别并进配方）

- Agent World Model：合成可执行环境 + GRPO，智能体后训练的数据工厂
- Accurate Weight-Matrix fingerprint：查血统，抗 SFT/CPT/RL，不是训练算法
- WebArena 帖里的 AWM pass/fail：环境名，不是损失

你要的「某种 post-training 算法」就是 **Advantage Weighted Matching**：预训练目标 + 优势加权。和 ReFL 并列，一个打奖励梯度，一个打加权匹配；Seedance 公开站 ReFL 更近。

29 sources

---

## User

ReFL冻 RM 对 $x_0$ 打标量，梯度反传到生成器$-\,r(x_0)$，通常偏后天AWM组内奖励 → 优势 $A$$A\cdot\mathcal{L}_{\mathrm{FM/DSM}}$ 详细解决这一类方法

---

## Grok

这一类方法要解决的是同一件事：**扩散/流模型预训练的目标，和「这段视频好不好」对不上。** 先把共同问题和两条修法写清楚。

## 共同的裂缝

预训练（FM / DSM）学的是：给定噪声 $x_t$、时间 $t$，预测速度或噪声，让

$$
\mathcal{L}_{\mathrm{FM}}
=\mathbb{E}_{x_0,t,\varepsilon}
\big\|v_\theta(x_t,t)-v^*(x_t,t,x_0)\big\|^2
$$

尽量小。这保证样本落在数据流形附近，**不保证**「更跟 prompt、更好看、物理对」。

后训练想最大化 $\mathbb{E}[r(x_0)]$。直接对整条去噪链做 PPO，要逐步 loglik，贵、方差大，且和 $\mathcal{L}_{\mathrm{FM}}$ 不是同一个函数。于是出现两条捷径：

- 把奖励变成对 $x_0$ 的可反传标量 → ReFL
- 不换损失，只按好坏给 $\mathcal{L}_{\mathrm{FM}}$ 加权 → AWM

两者都避免「反向转移高斯 loglik」（Flow-GRPO / DDPO 那条）。

---

## ReFL：奖励当损失

流程：

1. 从噪声滚到某个偏后的 $t$（或直接预测 $\hat x_0$）
2. 冻住的奖励模型 $r_\phi(\hat x_0,\text{prompt})$ 打一个标量
3. 损失 $-\lambda\,r_\phi(\hat x_0)$，梯度穿过去噪网络（有时再加一项 $\mathcal{L}_{\mathrm{FM}}$ 防崩）

$$
\nabla_\theta\,\mathbb{E}[r]
\approx
\mathbb{E}
\Big[
\frac{\partial r_\phi}{\partial \hat x_0}
\frac{\partial \hat x_0}{\partial\theta}
\Big]
$$

Seedance 1.0 / RewardDance 就是这一族：多维 RM（对齐、运动、审美）线性合成，直接最大化；和 DPO/PPO/GRPO 比过，报告写这条又快又好。RM 与生成器多轮交替，不是每步在线改 RM。

**为什么「偏后天」：**

太早的 $x_t$ 几乎是噪声，$r_\phi$ 看不出运动/审美，梯度是噪声。太晚才打分，结构已经定了，只能修纹理。经典 ReFL 折中：随机一个较后的 $t$，预测 $x_0$ 再打分。

**视频上的病：**

像素 RM 要 VAE 解码，显存炸；只打接近完成的帧，**运动学不到**。PRFL 把 $r$ 改成对任意 $t$ 的 latent 过程奖励，整条链都能回传，不必解码。

**会坏在哪：**

- Reward hacking：生成器专攻 RM 的洞（过饱和、假锐利）
- RM 小、方差塌 → 模式崩塌（RewardDance 用到 26B RM 就是压这个）
- 和预训练目标打架：只最大化 $r$、不加 $\mathcal{L}_{\mathrm{FM}}$，流形会歪

和 DMD 不同：DMD 的监督是 $s_{\mathrm{fake}}-s_{\mathrm{real}}$（分布），ReFL 是 **一个偏好标量**。没有「真数据分数场」，只有「人/RM 觉得好」。

---

## AWM：预训练损失 × 优势

不引进 $-\,r(x_0)$ 当主损失。仍用 $\mathcal{L}_{\mathrm{FM}}$，但每个样本有一个来自组内奖励的权重。

对同一 prompt 采 $G$ 条轨迹，得奖励 $r_1,\ldots,r_G$，组内标准化：

$$
A_i=\frac{r_i-\mathrm{mean}(r)}{\mathrm{std}(r)+\epsilon}
$$

然后

$$
\mathcal{L}_{\mathrm{AWM}}
=\mathbb{E}_i\Big[\,A_i\cdot\mathcal{L}_{\mathrm{FM}}(x_0^{(i)},t,\theta)\,\Big]
$$

$A>0$ 的样本：和平时一样，把匹配损失压下去（更贴这条「好」轨迹的流形方向）。

$A<0$ 的样本：符号反了，等于把模型从这条「坏」轨迹的速度场推开。

这是策略梯度在匹配目标上的实现：好轨迹的 FM 方向加强，坏轨迹减弱。预训练和 RL **共用一个损失形状**，只改样本权重。所以比 Flow-GRPO（优化反向一步 $p(x_{t-1}|x_t)$ 的 loglik）方差小、收敛快——论文里相对 Flow-GRPO 有将近一个数量级的 GPU 小时优势。

**和 ReFL 的真正差别：**

|  | ReFL | AWM |
| --- | --- | --- |
| 主损失 | $-\,r(\hat x_0)$ | $A\cdot\mathcal{L}_{\mathrm{FM}}$ |
| 梯度语义 | 「让 $x_0$ 往 RM 高分挪」 | 「好样本的速度匹配加强，坏的减弱」 |
| 要不要可微 RM | 要（$r$ 必须对 $x_0$ 可导，或用直通） | 不一定；$r$ 可以是 OCR、规则、不可微打分 |
| 和预训练 | 两条损失要加权拌 | 一条损失 |
| 典型病 | hack RM、只修后期观感 | 优势估计差、组太小方差大 |

AWM 更接近 LLM 里「CE × 优势」的 GRPO，只是 CE 换成了 FM。ReFL 更接近「直接对输出打分再反传」，没有「这条轨迹的预训练损失」这一项。

---

## 这一类里还该放谁

同一裂缝上的光谱：

1. DDPO / Flow-GRPO / Dance-GRPO把去噪当 MDP，对逐步高斯打 PPO/GRPO。和预训练目标不一致，AWM 要替代的就是它。
2. ReFL / RewardDance / Seedance RLHF$x_0$ 上可微奖励。工业视频后训练最常见。
3. PRFLReFL 搬进全程 latent，修「偏后天」。
4. AWM保留 FM，只加权。奖励可不可微都行。
5. DMD / 分布匹配没有 $r$，用分数差对齐教师或数据。解决的是「像不像分布」，不是「好不好看」。

Seedance 公开站 2；AWM 是 4 对 1 的纠正。两者可以叠：RM 出 $r$ → $A$，再 $A\cdot\mathcal{L}_{\mathrm{FM}}$，而不把 $-\,r$ 当主损失——那就是「ReFL 的奖励、AWM 的目标」。

---

## 落到 σ0 / 离散 token 时不要生搬

订单流没有 $x_t$ 和 VAE。同类思想的翻译：

- ReFL 类：冻一个打分器（字段分布、LOB-Bench、短窗 IC），对 整段生成消息 打标量，梯度穿过学生。离散上分数对 token 不可导，通常要 REINFORCE/GRPO，或对 logits 做直通。这就是「稀奖励打教师」。
- AWM 类：损失仍是字段 CE / 流匹配（若走 discrete FM），按该段的 $A$ 加权。好 rollout 的 CE 照常降，坏 rollout 反号。这比上 DMD 更贴「预训练目标别换」。
- 不要：用生成消息再训教师（自我消耗）；也不要用连续 ReFL 的「偏后 $t$」当离散深度 $m$ 的借口——你们的病在头 500 条，监督必须覆盖那段，而不是只打终点。

选型：有可微、稳的 RM（审美、对齐）→ ReFL。奖励是规则/OCR/撮合 PnL、不可微 → AWM 或 GRPO。既要对齐分布又要对齐偏好 → 匹配项（AWM/DMD）+ 稀疏奖励，不要只留 $-\,r(x_0)$。

---

## User

没有。两份报告全文和参考文献都查过了：Seedance 1.0 没提(时间上也不可能，1.0 是 2025 年 6 月，AWM 是 2025 年 9 月底才出的)；Seedance 2.0 的引用列表里也没有 AWM，后训练相关它只引了自家的 DanceGRPO 和 RewardDance。另外 AWM 不是字节的工作，是 UCAS、Adobe Research、HKU、MIT 的(Shuchen Xue 等，注意和 DanceGRPO 一作 Zeyue Xue 不是同一人)，论文里反而把 DanceGRPO/Flow-GRPO 当作被批评的基线。
它值得放进这条脉络里看，因为它正面攻击的就是 DanceGRPO 那套形式。核心两点：
理论：DDPO 系方法(Flow-GRPO、DanceGRPO 都建在其上)优化的逐步高斯似然，被证明等价于在做"以带噪数据为条件"的 denoising score matching。以带噪 x_s 而非干净 x0 为条件的估计量无偏但协方差严格更大(多出 d 倍 κ(s,t) 项)，所以方差大、收敛慢。这解释了为什么这类 RL 和预训练目标(干净数据的 score/flow matching)不一致会吃亏。
方法：把策略从"逐步反向转移 p(x_t-1|x_t)"换成"序列级 π(x0|c)"，用 flow matching 损失(即 ELBO)作 log likelihood 的替代，套 GRPO 式目标：组内相对 advantage 加权、重要性比率用新旧模型 FM 损失之差的 exp 估计(共享噪声与时间步降方差)、KL 直接在 velocity 空间对参考模型算。效果上就是 advantage 加权的 score/flow matching：好样本把速度场往目标拉，坏样本推开，目标函数和预训练完全同形。附带的工程红利是训练与采样解耦：采样随便用什么 ODE/SDE 采样器和步数，训练只需在少数几个前向加噪时间步上算损失(实验里采样 20 步、训练 4 步)，也不像 DDPO 系被绑死在 Euler Maruyama 上。结果：在 SD3.5M 和 FLUX 上与 Flow-GRPO 同质量，GenEval 0.95 打平但 8 倍提速，OCR/PickScore 最高 24 倍 GPU 时省。
所以现在视觉生成 RL 后训练大致三条路线：ReFL 系可微奖励直接反传(Seedance 1.0 的选择)、DDPO/GRPO 系反向过程似然比策略梯度(DanceGRPO、Flow-GRPO)、以及 AWM 这类前向过程 advantage 加权回归(源头可追到 Lee et al. 2023 的 reward weighted regression)。2026 年这第三条线还在快速展开，比如 Reinforce Adjoint Matching(2605.10759)、AdvantageFlow(2605.26013)和统一 path-space 视角的分析(2608.14430)。Seedance 2.0 内部有没有吸收这类想法无从判断，报告不披露方法，引用面上看它仍停在自家 GRPO + RM scaling 的叙事里。
Sources:

- Advantage Weighted Matching (AWM)
- Seedance 1.0
- Seedance 2.0

---

## Grok

你这次的文献关系是对的，前面不该把 AWM 说成 Seedance 用过的东西。

## 该改掉的

- Seedance 1.0（2025-06）时间上不可能引 AWM（2025-09）。
- Seedance 2.0 参考文献里没有 AWM；后训练只认自家 DanceGRPO + RewardDance。
- AWM 是 UCAS / Adobe / HKU / MIT（Shuchen Xue 等），一作也不是 DanceGRPO 的 Zeyue Xue。论文里 DanceGRPO / Flow-GRPO 是 被批评的基线。
- 2.0 内部吸没吸收这条线，报告不披露，引用面仍停在 GRPO + RM scaling。

## AWM 作为第三路线，机制按你写的收束

DDPO 系（Flow-GRPO、DanceGRPO）优化逐步反向高斯似然 $\log p(x_{t-1}\mid x_t)$。这被证成：等价于 **以带噪 $x_s$ 为条件** 的 denoising score matching——无偏，但协方差比以干净 $x_0$ 为条件的估计量多出 $d\cdot\kappa(s,t)$，所以方差大、和预训练（干净数据上的 FM/DSM）不一致。

AWM 换的是策略对象，不是再加权同一个逐步似然：

- 策略看成序列级 $\pi(x_0\mid c)$
- $\log\pi$ 的代理 = FM 损失（ELBO）
- 目标仍是 GRPO 形：组内相对优势
- 重要性比：$\exp(\mathcal{L}_{\mathrm{FM}}^{\mathrm{old}}-\mathcal{L}_{\mathrm{FM}}^{\mathrm{new}})$，共享噪声与时间步
- KL 打在 velocity 空间 对参考模型

于是好样本把速度场往目标拉、坏样本推开，损失和预训练同形。工程上采样与训练解耦：采样可用任意 ODE/SDE 和步数，训练只在少数前向加噪 $t$ 上算损失（他们实验里采样 20、训练 4），也不绑死 Euler–Maruyama。

质量与 Flow-GRPO 打平（GenEval 0.95），算力低一个数量级，是效率故事，不是「另一套审美」。

源头放在 reward-weighted regression（Lee et al. 2023）是对的；2026 年同族还有 Reinforce Adjoint Matching（2605.10759）、AdvantageFlow（2605.26013）、path-space 统一分析（2608.14430）。

## 三条路线（2026 视觉生成 RL）

```text
① ReFL 系     可微 RM 对 x0（或 latent）直接反传
               Seedance 1.0；RewardDance 是 RM 缩放
② DDPO/GRPO 系 反向逐步似然 + 组相对优势
               DanceGRPO、Flow-GRPO；Seedance 2.0 引用停在这里
③ 前向匹配 × 优势  FM/DSM 当 loglik 代理，A 加权
               AWM 及 2026 后续；批评的就是 ② 的目标错位
```

① 和 ③ 都可以不碰逐步高斯；差别是 ① 的主梯度来自 $r$，③ 的主梯度仍来自匹配，$r$ 只进 $A$。② 和预训练目标裂缝最大，AWM 打的就是这条缝。

对 σ0：离散 token 没有 $x_t$ 的 $\kappa(s,t)$ 那项，但「RL 目标别换成和预训练不同形的逐步似然」仍然成立——字段 CE / discrete FM 加权，比硬上逐步 token GRPO 更接近 ③。Seedance 2.0 不能当「字节已经选了 AWM」的证据。

---

## User

RAM refinforch adjoint maching , flow grop, 我把这些路线和那个 IM，把这些工作的放进去，应该放在哪合适？ DMD,

---

## Grok

「那个 IM」我按最可能的两个解来放：一是 **Adjoint Matching（AM，RAM 的母方法）**，二是你若指别的请对一下名字。其余都可以钉死位置。

## 先分两个正交轴，再挂算法

```text
轴 R  有没有奖励 r（偏好 / OCR / 规则）
轴 T  优化的是哪一段过程、哪一种似然
```

| 轴 T | 反向逐步似然 $p(x_{t-1}\\|x_t)$ | 前向匹配 $\mathcal{L}_{\mathrm{FM/DSM}}$ | 分数差 / 分布 |
| --- | --- | --- | --- |
| **有 r** | **② GRPO 系** | **③ 前向×优势/校正目标** | DMD+RL 杂交 |
| **无 r** | （少见） | 预训练本身 | **④ DMD 系** |
| **可微 r 直接反传 $x_0$** | — | — | **① ReFL 系**（第三种监督，不走似然） |

① 不占「似然」格：监督是 $\partial r/\partial x_0$。

---

## 各工作落点

**Flow-GRPO → ② 正中心**

ODE 改 SDE 才能采样 → 逐步反向转移上打 GRPO。DanceGRPO 同格。Seedance 2.0 引用停在这格。

旁支仍在 ② 里：Stepwise-Flow-GRPO（按 $t$ 给 credit，不改目标）。

**AWM → ③**

$\mathcal{L}=A\cdot\mathcal{L}_{\mathrm{FM}}$，策略对象改成 $\pi(x_0\mid c)$。批评的就是 ②。

**AdvantageFlow → ③，AWM 旁边**

同样前向加权最小二乘；负优势会非凸，用 rollout 正则稳住。明确写「不像 Flow-GRPO 优化反向过程」。

**AM（Adjoint Matching，Domingo-Enrich et al. 2025）→ ③ 的理论父节点**

KL 约束最大奖励 ⇒ 最优过程只 **tilt 干净终点**，前向加噪定律不变。用伴随状态把 $\nabla r$ 送进速度场。要可微奖励或反向伴随扫。

若你说的 IM 是这个 AM，就放这里。

**RAM（Reinforce Adjoint Matching，2605.10759）→ ③，AM 的不可微版**

同一条最优性：加噪定律不变，只改 $p(x_0)$。用 REINFORCE 把「奖励梯度」换成「奖励加权的分数项」，得到 **用 r 校正过的预训练回归目标**：采 $x_0$、打分、按预训练那样加噪、回归。没有 SDE rollout、没有反向伴随、也不要 $\nabla r$。对 Flow-GRPO 是「同质量、步数最多约 50× 少」，和 AWM 一样打 ②，但机制是 **改 target** 不是 **改样本权重**。

```text
③ 里再拆一层
  AWM / AdvantageFlow     损失形状不变，乘 A
  AM / RAM                损失形状仍是回归，target 被 r 校正
```

两条都是前向过程、都和预训练同形；一个加权，一个改靶。

**DMD → 单独的 ④，不要塞进 ①②③**

监督是 $s_{\mathrm{fake}}-s_{\mathrm{real}}$，最小化带噪反向 KL，**没有 r**。解决的是「学生分布像不像教师/数据」，不是「好不好看」。

和 ③ 容易混：都回归速度场——DMD 的靶来自分数差，③ 的靶来自数据（或 r 校正后的数据）。

DMD 的杂交挂在 ④ 和 R 轴的交界，**不要改 DMD 的主格**：

- DMDR：蒸馏时并行一条 RL（ReFL / DPO / GRPO 都能插）
- RTDMD：$\mathrm{KL}(p_\theta\|p_{\mathrm{teacher}}^{r\text{-tilt}})$ 拆成匹配项 + 奖励项

那是「④ 做冷启动 / 正则，再叠 ① 或 ②」，不是 AWM。

**① ReFL / RewardDance / Seedance 1.0 RLHF**

可微 RM，$-\,r(\hat x_0)$。和 ③ 都不用逐步高斯，但梯度语义不同。

---

## 一张图（往 Notion 里贴）

```text
① ReFL 系          可微 r → ∂x0
                   ImageReward, RewardDance, Seedance 1.0 RLHF, PRFL

② 反向 GRPO 系     log p(x_{t-1}|x_t) + 组内 A
                   Flow-GRPO, DanceGRPO, Stepwise-Flow-GRPO
                   Seedance 2.0 引用面

③ 前向匹配 × r     预训练同形
                   ├ 加权：AWM, AdvantageFlow
                   └ 改靶：Adjoint Matching, RAM
                     （2026 path-space 分析罩这一支）

④ 分布匹配         s_fake − s_real，无 r
                   DMD, DMD2, AC-DMD
                   叠 r：DMDR, RTDMD
```

---

## 选用时怎么记

- 要少步学生、对齐教师分布 → ④，不是 RAM。
- 奖励不可微（OCR、规则、撮合）→ ③ RAM/AWM 或 ②；有算力打 ②，要和预训练同形打 ③。
- 奖励可微、只在乎 $x_0$ 观感 → ①。
- Flow-GRPO 已经是 ② 的代表基线；RAM/AWM 是写「② 目标错位」时必须并排的对照，不是 Seedance 引用里的方法。

若「IM」不是 Adjoint Matching，把全称发一下，我只改那一格。

20 sources