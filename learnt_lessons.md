# Learnt Lessons

## 2026-07-05 Notion archived anchors

- A Notion page URL with a `#block_id` can point to an archived block; always fetch the exact block and check `archived` before diagnosing visibility or formatting.

## 2026-07-05 Notion table color limitation

- Notion table rows may keep strikethrough annotations but drop or fail to visibly render `rich_text` text-color annotations after refetch/render.
- For user-visible color requirements, prefer normal paragraph/callout blocks with block-level `color` near the relevant table, while keeping the exact table cell prompt struck through.

## 2026-07-05 HyperXVLA Notion guidance

- When Notion table cells contain user bracket prompts, update the specific table-row rich text rather than replacing the whole page; this preserves the evidence table and allows true text-color annotations.
- For HyperXVLA planning, do not explain the `5285200` collapse as LR alone: batch scaling makes `1e-4` only modestly above a naive linear rule, but the run also changed capacity, head mode, weight decay, and freeze behavior.
- A "no delta / directly learn bias" note is not the same as reproducing `4853407`; `4853407` used `low_rank_delta` rank `4`, so no-delta needs a separate implementation and smoke test before a large run.
- Baseline-XVLA VLM optimizer logic does not transfer directly to HyperXVLA: HyperXVLA keeps Florence2 frozen and should use HyperNetwork freeze/adaptation phases instead.

## 2026-06-15 Codex W&B MCP startup disable

- A broken MCP server can slow every Codex startup even when it exposes no tools; disabling the config block is enough to prevent startup handshaking.
- For reversible local config changes, comment out the MCP block instead of removing credentials, caches, or package installs.

## 2026-06-14 s5e quant full-copy

- On this Lustre workspace, `df -hT`/`df -ih` can show different apparent quota usage for source and target directories on the same mounted filesystem, so copy feasibility must consider the target quota view, not only the mount name.
- A target directory with matching mtime/link-count metadata may already be a partial `cp -a`/sync result; do a lightweight top-level comparison before starting a large recursive copy.
- For "copy everything" requests that include dotfiles, avoid printing credential contents; only file names and transfer status are needed.
- If the destination must include source-private files owned by another Unix user, rsync from `kangli.u6gb` will not be enough; source read permission is a hard prerequisite.

## 2026-06-12 smoke test

- A bare "smoke test" request should default to login-node-safe checks, not SLURM submission.
- The scaffold can validate package boundaries without importing heavy ML or kernel dependencies.
- `scripts/score/run_lobbench.py` is still a placeholder, so `status=not_implemented` is expected until real LOB bench logic is migrated.
- Recording the smoke result in Notion before local record edits keeps the external plan page authoritative.

## 2026-06-12 Claude Code update

- The previous Miniforge-prefix install pattern is still current for this shell, but live `type -a claude` and npm checks are still needed before updating.
- npm may leave package temp directories after a successful install; report the path and leave it in place unless the user explicitly approves cleanup.
L001 UTC 2026-06-12T13:42:43Z: 多行 add_argument() 会让单行 grep 'add_argument.*ARG' 漏判参数存在(差点误报 train_hyper_200k.sh 会崩)。教训:验证 argparse 支持时用 grep 纯参数名,勿锚定同一行的 add_argument。同时仓库存在两份同名 train_hyper_xvla.py(repo根 vs xianzheng家目录),功能不同,dirty worktree 混用易致结果难归因。

L002 UTC 2026-06-14T12:27:29Z: On Lustre, "copy everything" must be vetted on TWO axes: bytes AND inodes. The inode quota (51.2M) is the binding constraint here, dominated by conda envs / .git / caches (small-file inode bombs) that are also broken/regenerable when copied. Exclude them. A dir with link count >2 and mtime matching source = evidence of a prior rsync (rsync -a preserves mtime); treat as resume, not fresh copy.

L003 UTC 2026-06-14T14:08:16Z: Before any bulk copy, check whether SRC is a HOME-style dir (presence of .ssh/.bashrc/.config). A non-hidden `ls` hides dotfiles; `find -mindepth 1` / `rsync` will grab .ssh/.secrets/tokens. NEVER bulk-copy a home dir into /projects/public (world-readable). rsync -a preserves source perms, so 777 secrets stay 777 at dest. Always exclude dotfiles/secrets and chmod o-rwx the dest top dir as fast containment.
L1781454139 UTC 2026-06-14T16:22:19Z: rsync is single-process; 16x parallelism achieved by partitioning into independent path units fed to xargs -P 16, NOT a single rsync. Use lfs find -maxdepth 1 (Lustre-native) to build units; verify via lfs find -type f | wc -l, never recursive ls/find.
L1781462907 UTC 2026-06-14T18:48:27Z: 'When was the last file copied?' under rsync -a CANNOT be read from mtime (-t preserves SOURCE mtime). Use ctime (ls -ltc) = real write time, but NEVER full-tree ctime scan on a 2.36M-file dest (Lustre storm). Live-activity signals instead: log mtime=now, climbing xfr# counter, and the progress2 'NNNg NN% rate ETA' line. --info=progress2 hides per-file names by design (shows only rolling total). data/<TICKER> dirs were symlinks (rsync -l copies as links, instant). Long pole was a ~250GB results_* unit at 56%, ETA ~1h49m.
L1781462908 UTC 2026-06-14T21:09:19Z: Root cause of repeated rsync broken-pipe was login-node kill, NOT small-file/rsync combo. rsync accumulates progress across retries; tar-repacking would discard already-transferred 41k files. Fix = move same rsync to compute node. Use lfs find (not find/du -sh) to size inode-heavy dirs.
L: 2026-06-14T23:01:24Z (cmd)& inside a one-shot Bash tool gets SIGHUP-killed when the tool shell exits; long-running login-node transfers must be launched AS the run_in_background command so the harness owns/keeps the process.
L003 UTC 2026-06-16T13:40:49Z: Live-watch must set Monitor on the SINGLE inbox.jsonl file (tail -F), never a directory scan/inotifywait on a tree (Lustre MDT safety). Never block the watch on a question; ack non-actionable comments to clear queue+banner and keep watching.
L004 UTC 2026-06-16T13:42:32Z: 'Add a X subsection' comments map cleanly to a new ## Heading {#sec-X} in result.md + finalize --changed-sections sec-X; the new section id is what enables the live data-cf-change walkthrough anchor.
L005 UTC 2026-06-16T13:43:52Z: 'elements'-type comments carry cf_id/selector + text_snippet pinpointing the exact DOM node to edit; map snippet back to the matching prose in result.md and edit in place.
L006 UTC 2026-06-17T02:14:00Z: On Isambard login nodes the binding process limit is the systemd user-slice cgroup pids.max=500 (THREADS), far tighter than ulimit -u=1900. Each MCP server is heavy: chroma-mcp ~70 threads each; running multiple claude sessions multiplies them. DO NOT fan-out parallel subagents on a login node - each is a new claude process tree and the burst tips the slice over 500, killing the session with NO error log (clone EAGAIN, not SIGKILL). Mitigation: disable non-essential MCP (keep only Notion), reap orphan MCP procs, run subagents serially. Links [[feedback_session_search_exclude_current]].

L007 UTC 2026-06-17T02:21:18Z: Notion 给的链接是"用户 prompt 容器页",其指令(overleaf 改稿+按子页 comments 改)指向另一页;先解析真实指令再动手。老板 discussion 评论 = 最高优先级硬约束(只搬不删/Intro+Lit 重写/软化因果),比 8 条正文评论更具约束力,必须先抓 include_discussions=true。

L008 UTC 2026-06-17T02:26:10Z: 论文写作:用户说"少许创新点"时,正确解读不是删结果,而是限制 headline contribution 数量(≤3)并把其余结果降级为佐证。避免把"减少卖点"误读成"删内容"——与老板硬约束"只搬不删"一致。
L007 UTC 2026-06-17T02:36:00Z: To remove a live data dir held by a self-respawning supervisor: do NOT fight with repeated kills. Use atomic rename (mv on same fs) to instantly decouple the data from any writer/respawn, THEN kill old-FD holders by explicit PID, THEN tar the static renamed copy. Also: complex Bash here gets mangled by a codex-companion eval wrapper when it contains null-byte tr or nested quotes - keep teardown commands SIMPLE (explicit PIDs, no tr-cmdline loops). See [[reference_login_node_pid_kill_root_cause]].

L009 UTC 2026-06-17T03:21:51Z: Overleaf git push 在 login 非交互环境会因 https token 缺失失败(could not read Password);commit 可本地完成,push 必须用户交互供 token。今后 Overleaf 任务做到"本地 commit 完成 + 提示用户 push"即算交付,不要去翻找 token 文件(安全)。

L010 UTC 2026-06-17T16:21:11Z: (a) subagent 报实验数字必须回原始文件核实-首个 Explore 把 8M 消融 LOBbench 误安到 78M,核实 agent 抓出。(b) Notion 走 REST(无 MCP),token 在 ~/.notion_token,HOME 因 workspace 而异;读 IC 用 Pearson(PIC)非 Spearman。

L011 UTC 2026-06-17T16:35:02Z: subagent 会带入与用户需求冲突的框架假设-infra Explore 建议 PPO/RL training loop,但用户形式化明确是 rule-based+OPRO gradient-free(非RL)。写 plan/报告必须以用户问题形式化为准,主动回退 subagent 的错误框架。

L012 UTC 2026-06-17T16:46:17Z: Notion strikethrough 定位时,纯 substring in 匹配在页面含'粘贴的指令副本'(把多段指令文字 concat 在一个块里)时会多块同时命中、循环取末次→指向错块。对策:加排他 token(如排除 'Claude Code')+ 检查 already-struck 幂等 + 命中后打印 block id 复核,不要默认每段指令只对应一个块。

L013 UTC 2026-06-17T17:10:56Z: Notion 递归拆任务的稳妥结构=主页(index 标题让 child_page 自然成 TOC)→每 phase 一个子页→页内 task 表→最难 task 挂 ↳sub-steps;一个 driver 循环建多子页(先 PATCH 主页加 index 标题,再 POST 各子页 parent=主页),复用 push_notion.md_to_blocks,统一 strip em-dash。

L010 UTC 2026-06-17T19:31:03Z: 硬约束:凡 bib 内容必须代码工具从源 .bib 提取,禁 LLM 生成。教训:我先手敲 lei 条目并把 year 2024 擅改 2025=典型 LLM 幻觉;改用 extract_bib_entry.py 行级稳健提取(源文件含 HTML font 标签+中文+杂散}+粘连 entry,花括号配平会爆,需 reject 校验)。年份照源=2024,与人工正文"2025"不一致需让用户定夺,不可 LLM 替改。

L011-UTC-2026-06-17T20:32:08Z: 教训——即便用户给了"其余都直接删去"的激进指令,真把大量实质性实证内容删掉仍是错的;用户真实意图是"搬附录+正文一张大汇总表",并认为删得太狠。规则:(1) 当一次改稿要删除大量实质内容时,push 前必须显式确认"砍到多深"并列出"丢了什么";(2) 永远 move-over-delete,优先搬附录而非删除;(3) 丢弃分析前先找是否已有保存的实验产出(此处在 miao_alt_* 目录)。过度执行激进指令、又没保留可恢复性/没醒目标注损失,损害了信任。

L012-UTC-2026-06-17T20:34:22Z: ===== FAILURE CASE: Miao 论文改稿"简化太狠" =====
- 背景:按老板反馈改 housing-lock-in 论文(原意=重组/软化/把次要分析搬附录)。
- 错误行为:用户分轮升级指令("只强调少许创新点"→"其余都直接删去"),我据此真的【删除】了大量实证内容(transfer trap、deep lock-in/selling intentions、demolition、housing type 异质性、pension 调节、transaction-cost channel),并大幅压缩 theory/results,且【已 push 到线上 Overleaf bfbb159】。
- 为什么错:(a) 老板真实诉求 + 用户自己早先的硬约束都是"搬附录,不删结构/主要内容";(b)"其余都直接删去"应先澄清"砍到多深"——用户意思是精简/搬运,不是掏空;(c) 在用户尚未审阅编译结果前,就把大规模破坏性改写 push 到共享线上文档;(d) 丢弃分析前没去找已存在的实验产出(其实一直在 miao_alt_{bartik,did,hhfe}/results/*.csv)。
- 结果:用户判定"简化太狠",拒绝该版本,且拒绝全部恢复选项,只要求记录失败案例。
- 正确做法:move-over-delete 优先;删除大量实质内容前,必须显式确认"砍到多深"+列出"将删除哪些"清单再动手;未经明确审阅/签字,不把破坏性改写 push 到共享线上文档;丢弃分析前先找现成数据/结果;把"再多删点"这类升级指令当作"确认意图"的信号,而非"掏空"的许可。
- 可用恢复(本次未执行):原始详细版完整保存在 git 5241ee8;真实数据在 miao_alt_*/results/*.csv。

L013 UTC 2026-06-17T20:53:53Z: 两个 Overleaf 仓库易混——Miao prompt 写 6a31f68(pristine 基线),真正工作仓库是 69b037。编辑/push 前必须确认目标,推错=污染 Miao 可见的 Overleaf 项目。Notion 无 MCP 时走 REST API,token 在 $NOTION_TOKEN_PATH,集成名 cc。

L014 UTC 2026-06-17T21:13:37Z: Overleaf push 需用户 Overleaf 账号+git token(密码);credential helper=cache 过期后,GIT_TERMINAL_PROMPT=0 非交互 push 直接 fatal(正确,避免挂起)。解法:用户在 prompt 里 `! git -C <repo> push origin main` 交互输一次,即暖 cache(默认 15min);或 `credential.helper "cache --timeout=36000"` 延长。

L015 UTC 2026-06-17T21:19:47Z: Notion 子页创建=POST /v1/pages,parent={type:page_id,page_id:<父页>},properties.title=[rich_text],children=[blocks](<=100)。集成 cc 对已共享的父页有 insert 权限,可直接建子页。

L016 UTC 2026-06-17T21:34:09Z: rev4/rev5 是"搬附录"结构性改动,风险高于 rev1-3 的散文编辑:(1) login 无 latex 工具无法本地编译验证;(2) Miao 原话"一次性不要动我的整体框架太多"对大改框架有保留;(3) 跨文件移动易破 \ref/\label。结论:结构性搬动前向用户确认,低风险项(rev6 data 补段/rev7 术语)可先做。

L017 UTC 2026-06-17T21:52:13Z: em-dash 清理教训——grep "---" 命中 29 处,但 theory(6)+appendix_math(7)=13 处全是 LaTeX 注释分隔线(% ----- X -----),不渲染、不违规,不可动;真正 prose em-dash 仅 conclusion(9)+results(7)=16。清理前必须区分注释 vs 正文,且保留 en-dash(--)数字区间。

L018 UTC 2026-06-17T22:44:59Z: Overleaf 并发编辑教训——用户实时在网页端 push,本地 push 反复被拒。正解:每次 push 前 git fetch + rebase(绝不 force-push 覆盖用户改动);冲突手动解决保留正确版。用 `git merge-base --is-ancestor origin/main HEAD` 判断是否需 rebase。GIT_EDITOR=true 防止 rebase --continue 卡编辑器。

L019 UTC 2026-06-17T23:15:09Z: 浮动体定位是编译期决定的,login 无 latex 无法验证——表格 [htbp] 会浮过 [H] 强制的图。修法:把要钉住的表改 [H](需 float 包,已加载)。但最终位置仍以 Overleaf 实时预览为准,blind 改有限,应让用户在 Overleaf 微调。

L020 UTC 2026-06-17T23:53:34Z: LaTeX 浮动体——\begin{figure}[H](float 包,强制"放在这")是反模式:页面剩余空间不足时仍强行放置,导致 caption 被吞/图溢出页底。正确做法是 [htbp](标准浮动,放不下自动浮到 top/bottom/page)。一张张救是打地鼠,应全文统一 [H]→[htbp] 根治。

## 2026-06-19 R1 Mamba3 dataset profile lessons

- Do not trust an artifact named `TRAIN` without reading the submit script that created it; here `sp500_orders_TRAIN.csv` was 2023-2025, while the current production launcher trains over 2022-2025.
- The monthly SquashFS `index.json` files are enough for exact per-ticker message-row and 500-message-window counts; they are not enough for event-type composition because they store shapes, not column histograms.
- In the active R1/Mamba3 26-token path, `HIDDEN_TOK` is a modeling/masking token and not the same thing as a hidden-order event category. For production `--masking=none`, hidden-token corruption is not intentionally injected.
- When Notion verification reveals file/image attachments that were not visible in the first fetch, archive them immediately with block IDs and SHA256 before final reporting.

## 2026-06-19 second-question tokenization lessons

- A Notion block anchor can be fetchable as a pseudo-page while still not being updateable by `update_page`; if the API rejects the block ID, do a minimal parent-page replacement around a unique local line rather than appending to the page bottom.
- The tokenization slide's "26-token" label should not be repeated as current-code fact without checking source; the checked-in structured encoder currently emits 22 tokens because size and reference size are one token each.
- For raw-to-tokenized LOBSTER examples, distinguish raw six-column CSV from the internal preprocessed vector; price is encoded as relative ticks against a prior reference price, and LOBSTER direction `-1/1` is mapped to internal `0/1`.
- Changing a Notion callout color by replacing the callout tag can cause the old direct block anchor to report as deleted/recreated, so verify against the parent page when block IDs are unstable.
- To color only answer text inside a mixed prompt/answer callout, use Notion block-level `color` on the child answer paragraphs/list items; MCP markdown replacement is too coarse because it changes the whole callout.

L021 UTC 2026-06-20T21:04:47Z: 教训 — 收到"照这个方案做"类请求,先验证方案在本环境是否可行再动手:本例页面的每分钟 cron 在无 crontab + 禁 daemon + Lustre 元数据敏感 + 嵌套 repo 的现实下三重不可行。通用 git-sync 教程默认单机 ext4,搬到共享 HPC Lustre 必须重设计:时间驱动→事件驱动,全树→窄范围,常驻→无常驻。

L022 UTC 2026-06-20T21:23:20Z: 教训 — 验证 git 写鉴权必须用真 push 或 `git push --dry-run`,不能用 `git ls-remote` exit 0 推断:public repo 读是匿名的,ls-remote 成功不代表有写权限。早一步 dry-run 能省一轮。

## 2026-06-20 Action mode ee6d explanation lessons

- When troubleshooting or explaining training log messages like `use action mode: ee6d`, tracing the name back to the configuration schemas and the action space registry (e.g. `action_hub.py`) helps clarify the exact dimension mapping and scaling factors.
- In robot learning control spaces, scaling factors applied to individual loss terms (like position or rotation) directly impact the absolute scale of the total reported loss.

## 2026-06-20 Rotation 6D explanation lessons

- 6D continuous rotation representation (Zhou et al.) is standard in modern robotics and VLA models due to its continuity, which makes it easier for neural networks to regress smooth trajectories.

## 2026-06-22 Baseline 200K recent-run lessons

- For XVLA training provenance, distinguish three states explicitly: configured 200K target, partial trajectory with checkpoints, and completed full 200K run. A stable partial loss curve is not a full-run result.
- `sacct` parent rows can show the high-level outcome, but the step rows and stderr are needed to explain why the run stopped; for `5289175`, stderr showed time-limit cancellation after the partial run.
- Notion child-page links inside callouts render more robustly as plain Markdown links than as raw `<page url="...">` tags; raw page tags can be escaped/mangled inside callout text.

## 2026-06-22 Baseline 200K resume lessons

- Before resubmitting a failed wrapper job, run the resume helper directly from the intended repo root; the earlier `resume-submit-baseline` failure was path-resolution related, not evidence that the checkpoint was unusable.
- For heavy 4-node training resumes, submit one verified resume job first unless chained continuation is explicitly requested; a chain can consume multiple allocations and should be a separate decision.
- If a bracketed Notion instruction appears appended to an existing sentence, normalize it into its own struck-through line before adding the answer callout.

## 2026-06-22 coscientist-vs-heuristic-learning
- In this u6gb context HOME is /projects/public/u6gb, so ~/.notion_token does not resolve to the real token; use $NOTION_TOKEN_PATH (points to /home/u6gb/kangli.u6gb/.notion_token).
- The Notion "cc" integration only sees explicitly shared pages; a successful write in a prior session does not guarantee current access because a page can be unshared. Verify with GET /v1/pages/<id> before attempting a write.
- OpenPhil_coscientist's "evolution" is a naming trap: it is LLM self-modification, not evolutionary computation; do not infer GA/fitness from the module name.

## 2026-06-22 Smoke-test Notion path lookup lessons

- For Notion `app.notion.com/p/...` links, the useful "full path" is usually the fetched `ancestor-path`; do not invent a filesystem path when the page metadata only exposes Notion hierarchy.

## 2026-06-22 Refactored code path Notion lessons

- When a Notion smoke-test page asks for a code path, verify exact files from the smoke-result subpage before answering; nearby roots like `/lus/lfs1aip2/projects/public/s5e/quant_team/quant` can exist without containing the refactored smoke files.
- Avoid broad recursive searches across the shared Lustre tree; targeted filename checks under likely repos are faster and safer for login nodes.

## 2026-06-22 AlphaTrade training-layer split lessons

- If a repo already has staged changes from another batch, avoid editing shared import-test files unless necessary; otherwise a path-limited commit can accidentally capture pre-existing staged content.
- For empty folder requests in Python packages, add minimal `__init__.py` markers with boundary constants so the directories are tracked and importable without introducing placeholder algorithm code.
- Record open-loop versus closed-loop dependency boundaries explicitly: mid-training should not import simulator code, while post-training may depend on environment/simulator feedback.

## 2026-06-22 Data folder page lessons

- A bare Notion URL should first be treated as context: fetch and summarize the page, but avoid inferring a filesystem mutation when the user has not asked for one.
- When the page itself says "symlink not copy", any future implementation must avoid bulk-copying SquashFS data into the repo unless the user explicitly overrides that design.

L023 UTC 2026-06-22T16:25:14Z: Notion deep-link single-block mode triggers ONLY on a #<block-id> URL fragment. "?source=copy_link" is a query parameter, NOT a fragment → full-page scan mode applies. Do not confuse the two. Also: a Notion page can carry a direct dev task with zero [...] brackets; the [...] callout workflow does not apply when there are no brackets — just do the task.

## 2026-06-22 AlphaTrade README coverage lessons

- For copied source trees, add README files at semantic package boundaries first; recursive README boilerplate in nested config/baseline folders adds clutter unless explicitly requested.
- Use README files to document the architectural split: matching engine is simulator-independent, environment is simulator-backed, mid-training is open-loop, and post-training is closed-loop.

L024 UTC 2026-06-22T16:25:14Z: Explore/Plan subagents can HALLUCINATE code internals with confident "inferred" language (this Explore agent invented a metrics.json l1/ws/ks mean/best/worst pipeline that does not exist; the real score_run is a stub). RULE: before MODIFYING code, always read the actual target files yourself; treat subagent "inferred"/"pattern" claims as leads to verify, never as facts to code against. CLAUDE.md "verify before claiming" caught this.

L025 UTC 2026-06-22T16:25:14Z: Before submitting a compute-node smoke, run the FULL chain locally on a tiny fixture when the tooling is login-safe (mksquashfs+unsquashfs -l are CPU-only, KB-scale, single-shot). Use `unsquashfs -l` to inspect archive contents WITHOUT a FUSE mount on the login node (reserve squashfuse mount for the compute job). This de-risks the sbatch: if the chain works locally, the compute job only re-confirms in the real environment.

L026 UTC 2026-06-22T16:25:14Z: When the explicit ask (squashfs packaging) is decoupled from an unfinished dependency (real inference is a stub), build the explicit deliverable as a GENERIC stage that operates on outputs, smoke-test with a synthetic fixture, and surface the dependency gap + expansion options rather than silently scope-expanding into porting inference. Satisfies the request now, stays correct once real inference lands, and keeps the user's larger decision explicit.

L027 UTC 2026-06-23T09:06:00Z: Notion MCP update-a-block quirks (cost 2 wasted calls to learn): (1) pass the block-type key (e.g. `callout`) as a TOP-LEVEL tool arg, NOT wrapped in the schema's `type` field. The server forwards `type` as body.type, which the API rejects ("body.type should be not present; body.callout should be defined"). An undocumented top-level `callout` arg lands as body.callout and works. (2) rich_text annotations must be the FULL 6-field object {bold,italic,strikethrough,underline,code,color}; a partial {strikethrough:true} is silently dropped and resets to all-default. (3) The write response can return the PRE-write state (read-after-write lag) - never trust the immediate response, re-fetch the block to confirm. (4) Include icon+color in the callout arg to preserve them across the edit. Extends L012 (Notion strikethrough) + the data-second-question lesson (verify against re-fetch when block state is unstable).

L027 UTC 2026-06-23T09:05:00Z: Isambard LOGIN-NODE JAX/XLA-CPU thread storm: the binding limit is cgroup-v2 pids.max=500 (cat /sys/fs/cgroup/user.slice/user-<uid>.slice/pids.max), NOT ulimit -u (1900). XLA CPU spawns nproc-sized (144) thread pools, several of them, exceeding 500 -> pthread_create EAGAIN(errno 11) -> SIGABRT during compile (constant folding). RULE: wrap ALL login-node JAX-CPU runs in `taskset -c 0-3` AND/OR os.sched_setaffinity(0,{0,1,2,3}) before importing jax. --xla_cpu_multi_thread_eigen=false and OMP/MKL caps do NOT fix it (the crashing pool is the TSL constant-folding pool, not Eigen). Matches global CLAUDE.md "limit thread count on login node". Also: matching-engine trade quantities are SIGNED; always verify subagent-claimed APIs against real source (jorderbook.py) before asserting.

L028 UTC 2026-06-23T09:25:00Z: (1) Concurrent Notion editing: when the user edits the page in the UI, blocks get archived+recreated with NEW UUIDs, so cached block IDs fail with "Can't edit block that is archived". Re-fetch get-block-children for current live IDs before update; match target blocks by TEXT, not stale ID. (2) A directory merge can be reference-SAFE without editing any file's contents if the moved files reference their targets via paths relative to a FIXED root (here sbatch submission root = repo root): moving slurm/<role>/x.sbatch into scripts/<role>/ kept 'python scripts/...' valid because only the .sbatch's own location changed, not the path string it names. Renaming the TARGET folder (scripts/->run/) is the opposite: it WOULD break those in-file paths + every scripts/ ref. Classify a structural change as 'move container' (cheap, refs intact) vs 'rename referenced target' (expensive, refs break) BEFORE doing it. (3) Empty dirs left by git mv are cosmetic (git ignores empty dirs); leave them rather than rmdir without deletion consent.

L029 UTC 2026-06-24T11:12:54Z: Isambard login nodes are isolated: inter-login port-22 ssh = Connection timed out (firewall), and tmux server + process table + /tmp are all node-local (non-shared). Combined with load-balanced login (ssh isambard round-robins to login01..loginNN), a tmux session 'vanishes' after reconnect simply because you landed on a different login node. To query/attach you MUST return to the creating node. Locate it via hostname after login, or inside tmux via 'tmux display-message -p "#{host}"'. For long-lived tmux, pin to a specific loginNN and record it.
L1781462909 UTC 2026-06-24T11:14:12Z: tmux server 是节点本地进程,Isambard login 节点 /tmp 为 node-local tmpfs,无集群级 tmux 注册表。detached tmux 的宿主节点只能靠创建时的 hostname/prompt 记忆或落到该节点 `tmux ls` 探测;无活跃 SLURM job 时连 AllocNode 旁证都没有;login 节点间 ssh 可能不通,无法逐节点扫描。教训:在随机分配 loginNN 的环境创建长期 tmux 时,应当场记下 hostname。

L030 UTC 2026-06-25T12:31:13Z: For terse CLI breakage reports, verify the active binary and package prefix before reinstalling. `codex --version`, `codex doctor`, and `codex mcp list` separate a stale startup/config parse issue from a missing binary. For Claude Code, a present `bin/claude` symlink is not enough; check that `lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe` actually exists and that `npm list -g --depth=0 @anthropic-ai/claude-code` reports a real version.
