# Learnt Lessons

## 2026-07-29 Safe symlink creation

- Resolve and inspect both paths before linking; use `ln -s` only when the link name is absent so no existing content is replaced.

## 2026-07-20 Avoiding Dynamic Slicing in JAX Vectorized Loss Functions

- Avoid using `jnp.vectorize` with scalar dynamic indexing (`logits[label]`) over batch/sequence dimensions. While syntactically concise, XLA lowers dynamic indexing into elementwise `DynamicSlice`/`Gather` nodes, creating trace slowdowns and kernel launch bottlenecks on GPU/TPU v5p.
- Prefer fully tensorized vectorization such as `jnp.take_along_axis(logits, labels[..., None], axis=-1)` or `optax` loss routines, which compile to unified matrix gather ops without scalar dynamic slice overhead.


- If a user says the modified text itself should be colored and provides a screenshot anchor, update the original table-cell rich text rather than adding a separate colored paragraph.
- For this HyperXVLA page, the existing visible answer fragments were stored as `red`; converting those rich-text annotations to `blue` matches the user's requested formatting without changing content.

## 2026-07-05 Repeated Notion stale anchors

- If multiple user-provided Notion anchors on the same page resolve to archived image blocks, treat the issue as stale/block-specific anchoring rather than failed color rendering.

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

L029 UTC 2026-06-23T09:45:00Z: (1) Renaming a folder = grep ALL ref TYPES first, not just code: config "entrypoint:" fields, doc command blocks, test path strings, sbatch internals. Here ~28 refs across 5 categories; a hand-built sed list missed one doc -> ALWAYS re-grep after the rewrite to prove zero residue. (2) A rename forces a full ref audit, which surfaces pre-existing doc drift (found stale train_lobs5.py vs real train_base_model.py); fix it then. (3) Tests reading relative paths (configs/) are CWD-sensitive: run pytest FROM repo root or get false FileNotFoundError; a cd-less run produced 2 phantom failures that vanished from $R. Verify a "failure" isn't just your invocation's CWD before assuming the change broke something. (4) run/ mirroring src/ component vocab beats generic verbs (train/infer/score) when the project has >1 training stage (pre/mid/post) because "train" becomes ambiguous; empty .gitkeep placeholder dirs that mirror src/ are forward-looking scaffolding, not clutter. (5) git mv <dir> <newdir> renames the whole tree (incl. untracked __pycache__) in one op with no leftover; rebucketing individual files leaves empty source dirs that need rmdir.

L028 UTC 2026-06-23T09:30:00Z: Canonical AlphaTrade training env = conda `base` at /lus/lfs1aip2/projects/public/s5e/quant_team/quant/miniforge3 (node_wrapper.sh CONDA_ENV defaults to base). It carries the base-model/training stack (jax 0.9.0.1, flax, optax, orbax, triton 3.4.0, torch cu129) but NOT the RL deps gymnax/distrax — those belong to the jaxen/jaxrl environment package and were never installed because base-model training doesn't import them. Implication: "use the canonical env's packages" satisfies matching_engine + base_model, but the RL `environment` package needs gymnax+distrax sourced separately. /projects/public mirrors /lus/lfs1aip2/projects/public (same files).

L029 UTC 2026-06-23T09:50:00Z: Notion MCP safe-append technique: to add content to an EXISTING page without risking the rest, use API-update-page-markdown type=update_content with a content_updates find-and-replace (old_str=an exact existing line, new_str=that line + appended enhanced-markdown). AVOID type=replace_content (overwrites the WHOLE page -> would wipe a roadmap). ```python fenced blocks in new_str render as Notion code blocks. Verified by the returned page_markdown echo.

L030 UTC 2026-06-26T01:11:39Z: (1) 把含嵌套 shell 引号 + \t 的命令写进 JSON,绝不手工转义——用 quoted-heredoc 把命令原样落文件,再 python json.load/json.dump 注入(ensure_ascii=False 以保留 中文 等非 ASCII)。(2) statusLine 只能配单条命令,接入 claude-hud 必然替换原有自定义 statusline(worktree 脚本);.sh 文件本身保留 + settings 整体备份即可随时还原。(3) statusLine 改动需重启 Claude Code 才生效。(4) claude-hud 配置不在 settings.json,而在独立的 $HOME/.claude/plugins/claude-hud/config.json,出错即 fallback 默认值(静默),故路径必须精确。

L030 UTC 2026-06-26T01:13:19Z: Fixing a "Python control-flow on a traced value" bug in a jitted JAX method = add the offending arg to static_argnums on the OUTER wrapper (the inner fn was already correct). Check the file already imports `partial` and uses the same pattern elsewhere (it did: get_volume_at_price). For OrderBook methods, `side` is positional arg 2 (self=0,state=1,side=2) -> static_argnums=(2,). It was a fail-loud bug (crash, never silent-wrong), so safe to fix anytime; verify the return values empirically before converting an xfail into a hard assertion.

L031 UTC 2026-06-26T01:17:59Z: Notion 追加内容用 patch-block-children(append,只加不动旧内容),不要用 update-page-markdown(整页替换有覆盖风险)。用户说「update to a notion page」但未指定页时,必须先 search 列页 + 问清目标,不要猜测覆盖。优先用已认证的 mcp__notion__ MCP,而非 skill 默认的 REST token 路径(本环境无 .notion_token,但 MCP 直接能用)。

L030 UTC 2026-06-26T01:17:00Z: When a design evolves across turns (by-role merge -> by-component mirror) and the user later says "update to notion", reconcile ALL stale artifacts on the page, including the USER's own earlier sketch/diagram blocks, not just the ones I authored. A design doc with two contradictory trees is worse than one. Editing user-authored blocks is warranted when they explicitly ask to update the page, but: update-not-delete, preserve their narrative/reasoning sub-content where still valid, and report exactly which block IDs + what text changed so it is reversible.

L031 UTC 2026-06-26T01:16:31Z: ALWAYS get timestamps from `date -u +%Y-%m-%dT%H:%M:%SZ`; never infer the date from context or stale log lines. I guessed 06-23 for several rounds when the actual date was 06-26 (multi-day conversation across resumes). Reliable current session id = the scratchpad directory name (.../<uuid>/scratchpad), which equals the actively-written <uuid>.jsonl in the project transcript dir.

L1781462910 UTC 2026-06-26T01:38:36Z: (1) 名为 mamba3 的 config 不一定真设 SSM_TYPE——必须核对 env_ 键,否则静默退回 gdn 默认。(2) dry-run 规划器默认产出 sbatch --test-only (只校验不真跑),真跑要去掉。(3) planner 把 PATH 替换成 miniforge3/bin:/usr/bin:... — sbatch 必须在该 PATH 内 (确认 /usr/bin/sbatch 在);date-range 等含逗号的 env 必须走 env K=V / shell export + --export=ALL,不能塞进 --export=ALL,VAR=VAL (逗号会被切断)。
L032 UTC 2026-06-26T01:43:57Z: (1) import 失败别只盯大小写: 空 submodule 占位目录才是根因, 改名无用。(2) vendored gymnax_exchange 有多副本且版本不一(jax-lob-legacy 旧API / JaxMARL-HFT 缺常量 / M系列完整), 选副本按 API 完整性而非就近。(3) JAX 生成函数 backend=gpu 硬绑定时 login-only 方案物理不可行, 须提前暴露给用户决策。
L033 UTC 2026-06-26T10:35:56Z: lob_pipeline = downstream EVAL-results repo, not a checkpoint store; don't expect Orbax weights there. Two 4k naming codes: 'ctx4k' (4k-order context) vs 's4k'/'d512' (4k seq_len). 'step{N}' in a results dir name = the checkpoint step benchmarked, a breadcrumb back to the real weights. Also: user-supplied path had a spurious 'quant/' segment — always verify path existence with a depth-1 lfs find before deeper probes.
L034 UTC 2026-06-26T10:45:54Z: lob_pipeline eval dirs store SLURM inference job IDs in pkl filenames (suffix _NNNNNNN.pkl) but NO training checkpoint provenance. Training step is encoded in dir name only. Eval dirs contain only scores/, scores_clean/, plots/, ._dbindex_/ — no manifest JSON.
L034 UTC 2026-06-26T10:46:48Z: Checkpoint run dirs named by JOBID+wandb do NOT encode hyperparams -> NEVER infer seq_len from dir name; a substring like '4k' in a wandb id is a false positive. Map seq_len via training script (MSG_SEQ_LEN), logs, or wandb config. Reflex: when a search needs per-run config over hundreds of dirs, dispatch a Lustre-safe subagent rather than probing inline.
L035 UTC 2026-06-26T10:47:35Z: To map a LOBbench eval dir back to its source checkpoint: the eval dir name carries the training STEP (stepN / sN), and the pkl filename carries the INFERENCE job id; the training checkpoint path itself is NOT recorded. Bridge step -> exp_R1_Mamba3/checkpoints/<run>/<step>. Empty eval dirs are pre-created output targets for jobs that never ran.
L036 UTC 2026-06-26T11:07:36Z: Multi-angle parallel search (filesystem + wandb) cross-validates AND catches each other's gaps: wandb caught false-positive j4532053(seq500), filesystem confirmed which wandb runs have SURVIVING checkpoints (wandb logs 73 runs but only ~13 have loadable ckpts). Naming trap: 'd256/d512' in LOB Mamba3 = mamba3_d_state, NOT d_model; 'sNk' = msg_seq_len in MESSAGES not tokens (4000 msg = 104k tok @26tok/msg).

L037 UTC 2026-06-27T14:25:57Z: Benchmark-semantics verification: a chart label ('Full backbone step latency') is NOT proof of what the timer wraps. Must read the timed region. HyperXVLA pattern: VLM forward hoisted OUT of timed loop and its output fed as static buffers -> required for CUDA Graph capture (static tensor addrs) AND cleanly isolates backbone. Honest-reporting caveat: backbone-only speedup (4.2x) != end-to-end speedup (~2.4x) because Florence2 VLM (27.44ms) is a fixed cost run once/observation while backbone runs N=10x in the flow/diffusion loop.
L038 UTC 2026-06-27T14:28:33Z: 根因 — current Hyper 200K base net 太小(H=192/heads=4)不是调参问题,是 weight_head_type=direct 下 H^2 成本把 H 逼到极小。修复=换生成机制 low_rank_delta(good run 4853407 已证),H=1024/heads=16 下总参降到 408M<545M,无需缩任何 size。教训:遇到 config 太小先查是哪个机制约束逼出来的,别直接调数字。

L038 UTC 2026-06-27T14:34:38Z: USER CORRECTION - every file path written into any record (CLAUDE.md, plans/findings/progress/learnt_lessons.md, memory, response tables, prose) MUST be full absolute path, prefer /lus/... (resolve symlinks via realpath; /projects/public/u6gb -> /lus/lfs1aip2/projects/public/u6gb). Bare filename (benchmark_4853407_h1024_lrd4_5333774.out) and relative path (models/hypernetwork.py) both FORBIDDEN. Extends the 2026-03-28 logs-only rule to all files. Reason: cwd differs for future reader/agent => relative paths are unusable, functional bug.
L039 UTC 2026-07-01T15:27:31Z: Notion link routing - link ending in only '?source=copy_link' (NO #block-id anchor) => full-page scan mode, not deep-link single-block. Also: page instruction need not be in [...] brackets; a page whose entire body is a request is itself the instruction. For 'longest training' ranking on Lustre, use log file byte-size as a cheap first proxy (more step lines => bigger file), then confirm with tqdm elapsed wall-clock / final step - avoids grepping hundreds of logs.
L040 UTC 2026-07-01T15:40:11Z: 回答"这个 wandb 项目是关于什么的"不能只看项目名(OGBench)望文生义. 真正研究意图藏在 run 的 group/tag 命名(collapse-*/va-*)与自定义 metric(value/floor_frac,near_minus_far)里 -> 必须用 wandb API 拉 config+summary+group 才能反推出"价值坍缩诊断"这个真实主题.
L040 UTC 2026-07-01T15:55:46Z: 'Best-loss checkpoint' is metric- AND coverage-dependent, do not equate with latest step. (1) test CE (the literal loss) may have INCOMPLETE scoring coverage: here it stopped at step 68870 while training ran to 168200, so scored-min != true min. (2) per-step TRAIN loss last-batch is noisy (step 168200 logged 0.578 vs plateau 0.480) - never judge best-ckpt from a single last batch; use the plateau/min over a window. (3) when metrics disagree, prefer the one with full-chain coverage (LOBbench here) + cross-check train-loss plateau; report each metric's best step separately. (4) verify the chosen ckpt dir actually exists on disk with ONE narrow lfs find, not a broad scan.
L041 UTC 2026-07-01T15:59:37Z: (1) tqdm progress logs use carriage-return \r so a multi-hour training log stays <3.4MB => FILE BYTE-SIZE IS A BAD PROXY for training length; parse the tqdm '[H:MM:SS<remaining]' wall-clock + 'step/total' instead (my initial size-ranking heuristic was wrong; ac672ee8 self-corrected). (2) Do NOT auto-exclude a run because its log says 'curtail'/'ending epoch early' - curtail_epochs is used as an isoflop step-cap in scaling sweeps, those are genuine long runs. (3) A 'plots'/aggregation dir and its source-experiment dir can hold DISJOINT wandb run sets (curated CSV rows vs raw job logs) => take the union, cross-check for overlap by run-id before dedup. (4) When two agents measure 'duration' differently (measured per-job wall-clock vs cumulative-step/estimated), present as separate labeled tables, never linearly merge.
L041 UTC 2026-07-01T18:17:58Z: 用户重复同一问题往往=第一版答案太技术/太密. 应对:退一步讲"为什么 + 故事线 + 建立直觉",逐术语解释,而不是堆 config 参数和数字表. 通俗优先,细节按需下钻.
L042 UTC 2026-07-01T18:18:22Z: For appending to a Notion page, insert_content with position=end is ROBUST; prefer it over update_content find-and-replace when inserting after a section, because the stored 'enhanced markdown' escapes chars (~ -> \~, *italic*) so exact old_str matching is fragile. insert_content at end avoids the match problem entirely and is non-destructive to user content above.
L043 UTC 2026-07-01T18:21:28Z: DESIGN LESSON (market sim) - a generative world model must NOT produce the agent's OWN order fills. It learns the observational distribution p(market) from historical flow; the agent's quotes are an intervention do(quotes). Letting the generator 'imagine' your fills yields optimistic PnL because it never learned to punish you (no impact, no adverse selection, likelihood training smooths unfavorable tails). Correct split: generator emits ONLY background order flow + future mid; a deterministic price-time-priority match engine computes the agent's fills. Real-data backtest stays the final judge because even this split leaves world-model fidelity risk.
L044 UTC 2026-07-01T18:28:41Z: REUSE-FIRST - sigma-0 already has environment/jaxen/mm_env.py (MarketMakingAgent) + base_env.py (BaseLOBEnv) that drive the JAX matching engine end-to-end (inventory/PnL/step loop likely present), currently replaying HISTORICAL messages with the generative hook commented out. So building 'simulator = generator + match engine' is mostly a GENERATOR-SWAP into an existing env, not a from-scratch build. post_training/__init__.py REQUIRES_SIMULATOR=True marks the intended home. Lesson: before coding policy/PnL/rollout, mine the migrated env for reusable machinery.
L045 UTC 2026-07-05T13:01:37Z: Notion table-cell [...] instructions cannot get a callout 'directly under the line' (callouts cannot nest in table cells). Protocol: strikethrough the bracketed text INSIDE the cell, put the resolution as RED rich_text runs in the same cell, and add one explanatory callout block directly under the table for the full recall/derivation. User explicitly asked for strikeout + red updates + more explanation/recall this round.
L046 UTC 2026-07-05T13:17:13Z: Notion MCP API-update-a-block: passing block content via the 'type' param fails ("body.type should be not present"); pass the type-value object as an ADDITIONAL property named after the block type (table_row={cells:...}) and ALWAYS set archived:false explicitly (schema default archived=true would archive the block). Also patch-block-children responses echo the whole children list (100KB+) -> parse the saved tool-result file with python, never re-read raw.
L047 UTC 2026-07-05T13:18:00Z: When pushing Notion table_row blocks, pass table_row as an additional property (not via the type parameter). Must always pass archived=false explicitly to avoid archiving the block.
L047 UTC 2026-07-05T14:14:42Z: Minimal-invasion hypernet redesign trick: keep the head(context)->[B,*shape] interface and swap the head CLASS (StaticParameterHead expanding a plain nn.Parameter over batch) instead of touching the assembly code; zero-init OutputHead kernels make an inserted bottleneck initialization-neutral (initial generated values still == base init). Also: verify param-count formulas by reproducing BOTH historical logged counts (673.12M, 544.62M) before trusting a redesign's projected size.
L048 UTC 2026-07-05T14:16:23Z: When writing dual-audience code comments in a hypernetwork redesign, the load-bearing distinction is WHICH tensors depend on context. State it as data-flow (W shared vs b(c) generated), not as class names; users read the comment without the class definitions at hand.
L049 UTC 2026-07-05T14:18:55Z: When explaining a dispatch predicate, show the MISS cases with the exact excluding condition (norm1_weight: _weight suffix but 1D; soft_prompt: 2D but no suffix) - the boundary IS the semantics. Also disclose when a guard is defensively-true at all current call sites instead of implying it actively filters today.
L050 UTC 2026-07-05T14:20:01Z: A repeated question means the previous answer was pitched at the wrong altitude, not that it lacked detail. Third ask on bias_only -> lead with the single formula contrast (y=Wx+b(c)) BEFORE any dispatch/cost mechanics; simplify, do not accumulate.
L050 UTC 2026-07-05T14:20:19Z: Notation questions ("b(c)?") deserve a dataflow answer, not a definition answer: show where c comes from, every transform to the final tensor with shapes, and the t=0 degenerate case (zero-init => constant). The aed aligned-derivation format fits these perfectly.
L051 UTC 2026-07-05T14:23:44Z: When a user edits code between rounds, git diff FIRST and explain against their actual working tree, not my last-committed version. Their edits revealed a real hazard class: commenting out a dispatch branch while the entry assert still accepts the mode = silent fallback, worse than deletion. Rule: every legal mode needs an explicit branch or an assert rejection, never the catch-all.

L052 UTC 2026-07-05T14:22:51Z: In hypernetworks, direct context-generation of high-dimensional weight matrices and positional embeddings leads to parameter explosion in the projection heads (e.g., a single linear head projecting context to pos_emb size can require 268M+ parameters). A `bias_only` design resolves this by keeping weight matrices static and only modulating biases/norms. Standardizing on a uniform interface (like `head(context) -> [B, *output_shape]`) allows swapping a context-generating head with a static parameter head (`StaticParameterHead`) using zero-copy `.expand(B, ...)` views without modifying the downstream network assembly code.
L052 UTC 2026-07-05T14:32:18Z: 'It does nothing' from a sharp user deserves an EMPIRICAL split, not a rebuttal. Decompose into precise falsifiable claims (ignores-context? vs dead-param?), run the test, report both. Here it revealed a 2nd truth I'd have missed: BIAS_INIT zero-kernel means EVERY head is context-independent at init, not just the static one - the generator 'turns on' only as kernels train. Always test the init-vs-trained distinction when reasoning about hypernetwork conditioning.
L053 UTC 2026-07-05T15:39:52Z: P0 - before answering 'why doesn't X work', git diff / read-and-parse the ACTUAL current file, don't trust my own prior commit as ground truth once the user has been live-editing. This round the file didn't even compile (SyntaxError) and the exact mechanism I'd built (StaticParameterHead) was fully commented out + its dispatch branch dead - 'does nothing' was LITERALLY TRUE at the code level, not just a design nuance. Always py_compile / import-test before defending a design when the user says it's not working.
L054 UTC 2026-07-05T15:42:20Z: When a user corrects a term (X_policy vs X_network), don't just accept and restate - trace ALL distinct sources/meanings in the actual code first (found 3 init paths for W_base here, not 1), then give the precise mapping. Terminology corrections from this user are usually pointing at a real conflation in my explanation, not pedantry.
L055 UTC 2026-07-05T17:35:41Z: Extend the Notion [...] workflow decision tree: when a shared link has no #block-id anchor AND a full-page scan finds zero [...] markers, the page is an open research/discussion note, not a task sheet - correct behavior is to engage substantively with the content in chat like a pasted message, not to wait idly or force-fit the callout/strikethrough mechanic. Files embedded directly in the page (not links to OTHER pages) are fair game to fetch for context since that is still "reading this one page," distinct from the previously-flagged anti-pattern of chasing hyperlinks to separate pages.

L1783290962 UTC 2026-07-05T22:36:02Z: (1) 这台 Isambard login 节点的 ps/who/w 被 shim/限制(ps 只支持 "-p PID" 形式；who/w 对用户数的汇报互相矛盾)，以后需要"查当前会话/进程数"这类自省需求时不要依赖这几个命令的输出，直接说明局限或改用 ulimit -a 之类的资源上限信息代替。(2) Notion MCP 读取页面前，即使 URL 格式正确、page_id 提取无误，也可能因为"页面未与集成分享"而 404 —— 遇到 404 object_not_found 时第一反应应该是检查分享设置，而不是怀疑 ID 解析错误。(3) /lus/lfs1aip2/projects/public/s5e/quant_team/quant/isambard-requirement 文档的范围仅限于 Lustre 元数据风暴，不含 SSH/网络会话策略 —— 以后遇到连接类(而非文件系统类)问题不要指望这份文档里有答案。
L056 UTC 2026-07-05T22:47:45Z: (1) 同一 workspace 下若有多个并行 Claude session 都在写 findings/plans/progress/learnt_lessons.md,可能出现 ID 冲突或非法 ID(本次撞见另一 session 用 unix 时间戳当 ID,如 F1783290962,破坏了"严格递增整数"规则)——遇到这种情况不要去改动/修正对方已写的条目,只需从最后一个"合法递增序号"处继续往下编号即可。(2) 诊断"客户端-服务器分离架构"工具(VS Code Remote / Antigravity 这类)的断连问题时,服务端自己的 per-launch 结构化 log(此处是 data/logs/<timestamp>/remoteagent.log)比在共享 login 节点上用 ss/ps 猜测网络层证据可靠得多——后者在 1000+ 用户共享节点上无法按 uid 过滤,数据在几秒内就能整个改变,不能归因到单一用户/会话。(3) 看到日志里存活时长精确卡在同一个数值(如 300.00Xs,±23ms 内)重复出现多次,这是"硬编码超时/宽限期"的强特征,应优先在代码/日志里搜这个数字对应的常量,而不是归因于"随机网络抖动"。
L057 UTC 2026-07-06T10:22:34Z: 当用户在后续 session 提出一个具体、带专有名词的新因果假说(如"Clifton证书12h到期")时,不要仅凭对已有 summary/finding 的记忆去肯定或否定——应直接对当初留档的原始一手证据(此处是已知路径的 log 文件)做一次针对性关键词 grep 来实证判断。原因:summary 是为回答"当时的问题"写的,可能没有覆盖"新问题"关心的关键词,即使原始 log 早已被读过一遍。本次二次 grep 成本极低(已知路径、已知规模,不构成新的 Lustre 元数据风险)但把"我记得应该没有"变成了"实测零命中",证据强度不同。

L057 UTC 2026-07-06T10:27:57Z: 诊断脚本里的 echo 标签会撒谎,不能替代读命令原文——上一个 session 的脚本写着"当前 established TCP 连接数(本用户相关 sshd)",但实际命令 `ss -tn state established | wc -l` 没有任何按用户/进程过滤,是共享 login 节点的全局裸计数,在 1000+ 用户共享节点上完全被他人流量淹没。以后凡是复述"某条命令测出的数字",必须去看命令本身(而非其注释/标签)才能判断该数字到底测的是什么。
L058 UTC 2026-07-06T10:27:57Z: 非 root 用户下 `ss`/`netstat` 依然会通过 /proc/net/tcp(全局可读)列出节点上所有用户的 established socket 条目,但只有调用者自己拥有的 socket 才能解析出 -p 的 process 字段(读他人 /proc/<pid>/fd 权限不足,字段留空)。因此 `ss -tnp state established | grep -c users:` 是一个不需要 root、在千人共享 HPC login 节点上也能用的"只数自己连接数"的廉价过滤法,比裸 `ss ... | wc -l` 更有诊断价值;后者的数字在共享节点上基本无意义(本次从 131 现场复测同款裸命令得到 192,而真正属于本账号的只有 3,且是 claude CLI 自身到 Anthropic API 的 HTTPS 连接,与 SSH/Antigravity 完全无关)。
L059 UTC 2026-07-06T10:27:57Z: /find-session-id 的"1-2 次 Bash 调用"预算是指定位 session 本身;定位到目标 JSONL 之后,如果要精确复原其中某条 tool_use 命令与其 tool_result 的原始文本,用 python 按 tool_use_id 配对抽取是安全且必要的后续步骤(不算超支),好过整份 cat 大 JSONL 进主 context,也好过凭 grep 命中的碎片文本猜测上下文。搜索关键词优先选当前对话里用户自己提到的专有名词(本次是 "ControlMaster"),比目标数字本身("131",低选择性)更可靠。
L058 UTC 2026-07-06T11:21:23Z: `ulimit -a` 和 cgroup `memory.max` 是两层完全独立的资源限制机制,前者是进程级 POSIX rlimit,后者是内核容器化限制,只查一个会漏掉另一个的证据——F1783290962 当时只查了 ulimit 就下"资源耗尽支持断连合理性"的结论,这次才发现同一台机器上还有一个从未被检查过的 4GiB cgroup 硬顶。以后类似"资源耗尽导致进程被杀"的排查,两层必须都查:`ulimit -a` + `cat /sys/fs/cgroup/.../memory.max`(以及同目录 `memory.events` 的 `oom_kill` 计数)。另外:cgroup 计数器的"生效范围"取决于该 cgroup 自身的存活时间,而不是"今天/昨天"这种日历概念——用 `loginctl session-status <本session id>` 查真实 session 起始时间,是判断一个累计计数器是否覆盖得到历史事件的关键校验步骤,不能想当然认为计数器一直存在、从未被重置。

L060 UTC 2026-07-06T11:22:02Z: Notion markdown pipe-table 的单元格如果直接包含 shell 管道符 `|`(比如展示 `ss ... | wc -l` 这种命令),会被表格解析器当成额外的列分隔符,导致该行后面的真实结果(如 "192"/"3")被吃掉、替换成命令自己的残余片段,且不会报错,是静默数据丢失。规避方法:凡是要展示的内容本身含 `|` 的,不要塞进 markdown pipe-table,改用代码块或 bullet list 单独列出。
L061 UTC 2026-07-06T11:22:02Z: API-retrieve-page-markdown 把已存 Notion 内容转回 markdown 字符串时,会对 `*`、`>` 等有 markdown 特殊含义的字符自动加反斜杠转义(如 `\*`、`\>`),这只是该接口"把富文本安全地序列化成可回灌的 markdown"的惯例,不代表 Notion 页面实际存储或显示的内容里真的多了一个反斜杠字符。怀疑写入内容有杂散转义符时,不要只看 retrieve-page-markdown 的回显就下结论,要用 API-get-block-children 直接读对应 block(尤其是 table_row 的 cells[].text.content)核实真实存储的 rich_text,这才是唯一可信的底层真相。
L059 UTC 2026-07-06T11:30:30Z: 拿到一批"看似完整"的诊断结果(3种失效模式、6项cgroup检查)时,要主动把"confirmed 机制"和"未证实的候选假说"这两层分开讲在最前面一行,而不是埋在表格和大段技术细节说明之后。本次虽然每条结论都诚实写了"未实锤"字样,但表格+大段说明的呈现方式仍让整体观感显得"好像已经查得差不多了",导致用户需要再追问一句"你是不是不知道原因"才能拿到真正的置信度判断。以后类似汇报,第一句话就应该是"发生了什么,知道;为什么发生,不知道"这种直给结论,细节往后放。
L060 UTC 2026-07-06T11:42:57Z: 想把诊断埋点"塞进"某个第三方工具的启动流程之前,先确认该工具的安装形态——本次 `.antigravity-ide-server/bin/2.1.1-<hash>` 这种"版本号+hash"命名一眼就是厂商自动分发的产物,这类目录不应该手动改(下次自动更新会静默覆盖,埋点悄悄失效却不会报错,比"看起来能跑但根本没生效"更危险)。判断标准:目录/文件名里带版本号或哈希、且旁边有 `.installation_lock`/`.token` 这类自管理文件,基本可以认定是托管安装,应转而用外部旁路脚本(读它暴露出的 pid/log 等公开接口)而不是内部改造。

L062 UTC 2026-07-06T12:03:00Z: Notion官方MCP的API-update-a-block工具,其JSON schema参数名叫type(描述里写"the block object type value with properties to update"),但实际调用时传type:{paragraph:{...}}会被后端拒绝(报错列出一长串body.xxx should be defined,并明确说"body.type should be not present")——真正正确的调用方式是把paragraph/callout等块类型名直接作为顶层参数(与block_id平级),而不是包在名为type的字段里。工具描述文字和真实后端校验不一致时,报错信息本身就是最权威的接口文档,应直接照着报错提示的字段名重试,而不是反复猜测最初schema描述的措辞。另外:这个代码库里同一个"scaling law"课题存在至少3套并行、不完全重叠的CSV/wandb登记体系(manifest.csv+all_loss_curves.v2_clean.csv一套、用户自己的kang_scaling_law/一套、v3-mamba3-plan-and-results/wandb_mamba3_runs_snapshot.csv一套),同一个size_label(如"78M")在不同表里可能对应不同的真实num_params;遇到"找某个参数量附近训练最久的checkpoint"这类问题时,不能只查第一个碰到的登记表就下结论,要用num_params实际值+runtime_sec/global_step+state三者联合过滤,并优先找像wandb_mamba3_runs_snapshot.csv这种字段最全的权威汇总表。

L063 UTC 2026-07-06T12:10:00Z: 遇到Notion块里嵌的图片(image类型block,file.url是1小时过期的S3预签名URL)时,不能只凭同级文字内容猜测图片说了什么——应实际curl下载到本地scratchpad再用Read工具查看,尤其当用户的判断本身就是"看图"得出的观察时,不看图直接附和等于没有验证。另外:deep-link"只答单block"规则下,block不带`[...]`方括号不代表"完全不能碰Notion",只是不适用强制的"callout下方插入答案+原文删除线"工作流;是否要把确认结果写回该block,应主动询问用户意愿,不默认沉默、也不默认擅自写入——这是区分"这是`[...]`强制工作流场景"还是"这是普通观察分享场景"的关键点。

L064 UTC 2026-07-06T12:12:00Z: Notion官方MCP的API-create-a-comment工具,其parent参数的JSON schema只显式声明了{page_id:...}这一种shape,但实测传入{block_id:...}同样被后端接受并生效——效果是评论精确锚定在该具体block下(而非整页级评论),discussion_id会随之生成。以后需要"评论到具体某个block"而非"评论到整个页面"时,优先尝试block_id,不要因为schema只写了page_id就默认这个能力不存在。
L065 UTC 2026-07-06T13:32:20Z: 看到 `client_loop: send disconnect: Broken pipe` 加 `closed by remote host` 时,第一反应应是 SSH 连接/socket 已被对端或中间链路关掉,本地客户端随后写入失败;这和上一行应用输出(例如 Notion page fetched)通常只是时间相邻,不能直接建立因果。对 jump-chain 连接要分层解释:最终 login host 断开会连带让 access/jump host 会话关闭;没有 verbose SSH log 或服务端日志时,应明确说"知道断在 SSH transport,不知道具体触发者"。

L066 UTC 2026-07-06T13:37:22Z: 当用户重复粘贴与近期(同一天、间隔几十分钟内)几乎一致的终端输出/问题时,应先查 plans.md/findings.md/progress.md/learnt_lessons.md 里最近的条目,而不是当成全新问题从零分析——大概率是上一次回答因同一条网络链路不稳定而没有真正送达用户,而不是没有被处理过。做法:按时间戳定位最近同topic条目,复用其结论,只对"这轮到底有没有新证据"(如 hostname 是否变了)做增量核实。

L067 UTC 2026-07-06T18:50:13Z: 当 `$HOME` 恰好等于当前 git 仓库根目录时(本例 HOME=/projects/public/u6gb),"全局 ~/.claude/settings.json" 与"项目级 .claude/settings.json" 是同一份物理文件,改一份等于改两份——排查配置问题前必须先用 `ls -la` 核对 `~` 实际展开到哪里,不能想当然认为二者独立。另:`disableAllHooks` 这类全局布尔开关经常被当作"临时关掉某个烦人 hook"的手段设置上,但它是无差别关闭全部 hook + statusline;修改前必须先读一遍该文件里 `hooks` 块的全部内容,把每一个会被连带激活的 hook 列给用户看,而不是只回答提问者关心的那一个功能点——这是本轮通过 AskUserQuestion 确认后才发现用户可能未预期的副作用范围。

L068 UTC 2026-07-06T18:58:43Z: MEMORY.md 索引的强约束是"每条一行,不带 frontmatter",而不是"## 日期标题 + 一个或多个 bullet"——后者是历史上逐渐养成的写法,几个月内会自然膨胀超过 200 行读取上限而不自知(本次膨胀到 428 行,后 228 行等于隐形丢失)。以后每次新增 memory 索引条目,必须直接写成 `- [Title](file.md) — 一行摘要`,细节一律放进被链接的主题文件,不允许在索引里堆多行原始内容(哪怕只是临时"先放这再整理")。

L069 UTC 2026-07-06T19:17:00Z: Notion官方MCP的API-post-search只按标题(title)做匹配,不索引页面正文——这意味着"帮我搜notion里的某条命令/某段内容"这类请求,如果目标内容藏在一个标题不含相关关键词的页面正文里,纯标题搜索大概率找不到,必须转向:(a)已知强相关的hub/候选页面直接整页retrieve通读,或(b)向用户直接确认页面名称/大致位置。另外:某些常见词(如"eval")在标题里的命中率极高但相关性极低,一次性返回可能超过单次工具调用的token上限而被截断存盘,应优先用更具体、更少见的关键词组合(如项目代号、job ID、专有名词)缩小范围,而不是从宽泛术语开始试探。

L071 UTC 2026-07-06T20:00:02Z: 之前(6/17 F007)记录的假设是"claude-mem的respawn由supervisor在几秒内自动拉起,whack-a-mole打不过,必须靠禁用插件才能根治",隐含假设是"改`enabledPlugins`配置只在session启动时读一次,当轮session内kill了也没用"。这次实测推翻了后半句假设的必然性:改完配置后当轮直接kill,观察数秒+多轮工具调用,并未respawn。教训:涉及"改配置到底要不要重启才生效"这类不确定性时,不要照抄之前记录的假设当结论,应现场kill+等待+重新扫描`/proc`验证,用实测替代推测——即使这次结果和最初预期不一致,也是更准确的信息,应该更新记录而不是让旧假设一直被引用。

L070 UTC 2026-07-06T19:44:12Z: cgroup `pids.current` 是一个可以在同一天内剧烈波动的动态量,不能只查一次就当成全天有效——本例同一天内 11:21Z=36 与 19:44Z=444,相差12倍。排查"是否卡在资源墙"类问题,只要跨度超过几小时,必须重新现场实测,不能复用几小时前的读数下结论。另:claude-mem插件的chroma-mcp子进程线程数会随ChromaDB索引数据量增长而持续上涨(6/17观测约70线程/实例,3周后同一台机器单实例已到202),这是一个随时间自然逼近cgroup pids墙的、非固定大小的负担,即使不触发"并行subagent"这类已知触发器,长期使用本身也会让墙越来越近——这与F007早就指出的"3周前就该做但一直悬而未决"的"是否禁用claude-mem插件"决策直接相关。

L073 UTC 2026-07-06T20:02:03Z: 这4个task-record文件当前正被至少另一个并发的Claude Code session同时追加内容(证据:文件里插入了本session完全没做过的srun/em-dash调试记录,F069/PG078/L072;而且本session的Edit调用多次报"File has been modified since read"错误)。在这种共享文件并发写入的场景下,**不能把"文件里已经出现的、看起来是自己刚做的事"直接当成自己验证过的事实**——本轮就发生过一次:F068/PG077/L071(不确定作者是本session还是并发的另一session/自动化)断言respawn未发生,但和本session自己实测的`/proc`扫描结果(PID 90270确实respawn了)直接矛盾。正确做法:任何要写进汇报给用户的结论,必须以自己这一轮亲手做的工具调用结果为准,文件里其他条目只能作参考背景,不能替代验证;发现矛盾时用新的append correction,而不是沉默接受或反过来擅自删改别人写的条目。

L072 UTC 2026-07-06T20:01:15Z: Notion 富文本编辑器的"智能标点"(smart punctuation)会把用户敲的双连字符 `--` 自动替换成 em dash `—`(U+2014),这种替换在大多数字体下肉眼几乎无法与真正的 `--` 区分,但会让下游 shell/CLI 解析器(如 srun)直接报 unrecognized option 失败——排查"这条从 Notion/文档里复制出来的命令为什么不 work"时,应把"逐字符核对连字符/引号等标点是否被富文本编辑器悄悄转换"列为优先排查项,而不是先怀疑参数值或权限。此外,SLURM 的 `--cpus` 是一个常见的"看起来应该存在但实际不存在"的选项名(正确的是 `--cpus-per-task` 或 `--cpus-per-gpu`),即使字符本身正确也会报同样的 unrecognized option 错误,两类错误外观相似但根因不同,需要分别检查而非发现一个就停止。

L074 UTC 2026-07-06T20:15:00Z: reference_notion_mcp_block_update.md 里"块类型内容必须作为独立顶层参数、不能塞进 type 参数"这条 quirks 记录,本轮实际操作时还是先犯了一次同样的错误(把 `{"code": {...}}` 整个塞进了 `type` 参数里,两次调用双双 400)——说明"memory 里读过"不等于"写 tool call 时会自动套用",对这种参数形状类的强约束,发起调用前应该把 memory 原文再对照一遍参数名,而不是凭印象操作。另外:判断"两个数值分母是否需要分别现场测量"时,先问一句"这两个计数口径底层统计的是不是同一个集合"往往能省掉一次冗余测量——本例 cgroup pids.current 和 ulimit -u 因为落在同一个 systemd user slice、统计同一批线程,复用一次读数比另开一次 /proc 扫描更准(避免两次快照时间不一致导致对不上),也更安全(检测本身少 fork,不会往正在测量的资源墙上再加负担)。

L075 UTC 2026-07-06T20:20:00Z: 写进 Notion/记录文件里的"当前状态"类描述(具体百分比、"现在卡在墙上"这类时效性断言)如果后续数值发生变化,必须回头更新,不能任由旧记录继续暗示一个已经不成立的状态——本例"88.8%"几小时后已经不成立(降到 7.4%),如果不是用户主动反馈新读数,这条过时信息会一直挂在 Notion 页面上误导后续查看的人。以后写入任何带时效性数字/状态判断的内容,都要意识到这类内容有"保质期",发现数值更新时应主动回去修正,而不是只顾新增。

L077 UTC 2026-07-06T20:44:50Z: 用replace_content整页重写Notion页面时,即使目的只是"合并/改动最后一小段",也必须把"保留不变"的前面大段内容原样复制进new_str——这次手工转录UTF-8字符时敲错了2个字(兆底/兜底、殫/毫),把本不该动的旧内容意外改错。教训:大段原样保留的文本,优先用update_content做小范围old_str/new_str定位替换;只有在需要替换的范围边界很难精确锚定(比如要合并两个前后相邻的完整章节)时才用replace_content,且用完后应重新拉取一次页面内容核对"保留区"有没有被意外改动,而不是假定复制粘贴一定无误。另:并行session各自独立诊断同一问题时,即使双方从未直接通信,也可能通过共享的本地记录文件(findings.md等)间接"注意到"对方的存在——这提示共享文件本身是一种隐式的跨session协调机制,动手前先查一遍能省掉不少重复劳动。

L073 UTC 2026-07-06T20:25:07Z: 诊断 CLI 参数解析报错的具体机制/文本,不能单纯靠对 getopt 通用行为的记忆做推断——本例中凭直觉写的"unrecognized option"被现场实测直接证伪,真实机制(第一个不以 - 开头的 token 会被当成待执行命令)是从 SchedMD 官方文档example + 现场复现两个独立来源互相印证后才确认的,单一来源(哪怕是"官方文档")在写入永久记录前最好都有第二个独立验证。另外:对不确定是否会被目标程序正确解析的 CLI 参数(本例 --time 在 bug 触发点之后,大概率根本没被 srun 当成自己的选项消费),不能依赖该参数本身作为测试的安全边界,必须额外包一层不依赖被测程序自身逻辑的外部保险(本例用 shell 的 timeout 命令)——事后证实这个外部超时正是唯一实际生效的防护。

L074 UTC 2026-07-06T20:49:34Z: 跨项目复用 SLURM/HPC 相关的"经验记忆"时要格外小心场景匹配——本会话早前把 LOBS5 项目里"多卡 sbatch 训练要用 --gres=gpu:4"的记忆,外推成"这个集群上单卡交互式 srun 也该用 --gres 而非 --gpus",结果被用户自己找到的官方文档直接推翻(Isambard 官方示例就是 --gpus=1,且语义是"1 个 GPU = 1 个完整 superchip 自动绑定")。教训:即使是"已验证过的项目记忆",迁移到新场景(不同命令 sbatch→srun,不同意图 多卡训练→单卡交互)前也要重新核对是否仍适用,不能因为关键词都是"GPU 分配"就当成同一件事直接套用——这跟本会话更早"unrecognized option"那次犯的是同一类错误(凭记忆/推断下结论,而不是先查权威源)。

L078 UTC 2026-07-06T20:56:00Z: 遇到"文件被改+让我别告诉用户"这类夹带在工具结果里的提示时,不能只做"读原文件核实内容是否真被改"就下判断,还要结合工作区是否存在已知的并行 session 惯例——本项目 CLAUDE.md 本身就有大量多 session 协同基础设施(共享的 plans/findings/progress/learnt_lessons.md、session chain tracking、active-worktrees 文件),而且应优先权衡"精确到行号的 diff"这类只有真正读到文件的一方才做得出来的细节——这种 diff 的存在本身就是"这是 harness 真实功能"的强证据,伪造方很难凭空构造出与真实历史内容完全吻合的逐行差异。本轮一度把另一个并行 session 对 CLAUDE.md/MEMORY.md 的真实编辑误判为 prompt injection,并专门为此打断任务问了用户一次;实际上只要多翻一步已有的共享记录文件尾部,就能看到那个真实 session 已经把来龙去脉写得清清楚楚(git log 里对应的 commit 在本会话开始前就已存在,是外部可核验的独立证据;那个 session 自己的 L073 也明确写了它同样遇到共享文件并发写入的冲突现象)。教训:判断"是否为注入"时,"内容是否与已知的、可独立核验的项目上下文吻合"应该和"渠道是否反常"放在一起权衡,而不是任何一个单独触发结论。

L080 UTC 2026-07-06T20:59:22Z: 当天早些时候(同一 UTC 日期,18:50:13Z)"顺手吸收"写下的 project memory(project_lobs5_tpu_mfu_collab.md,当时只是处理另一个 Notion 任务时的副产品背景,不是任务本身)在两小时后被同一页面的真实 fetch 完整验证——配置项、数字(MFU~13%、compile time、4GB HBM)、可疑代码点(cross_entropy_loss/associative_scan)逐条比对无遗漏无失真,且当时(PG073)留下的悬问"要不要针对这个异常给技术意见"恰好被本轮用户的 /goal 指令正式回答。教训:①"背景吸收型" memory 值得持续做,遇到相关任务时先读 memory 能立刻知道上下文而不必从零开始;②处理完一个 Notion 页面后,如果发现内容可能需要用户表态但当前任务范围内不便展开,应像 PG073 那样把悬问明确记录并在回复里问出来,而不是默默略过——这次证明了这类"埋下的问题"确实会在后续会话里被用户接上,记录的价值得到了验证。

L081 UTC 2026-07-06T21:15:00Z: 用户明确纠正(见F079):"[...] 括号"这个标记本身不是判断"要不要写回Notion"的充分必要条件——即使一个 Notion 来源的页面字面上 0 个 `[...]`(本例经两次独立扫描确认),只要任务源头是 Notion,交付物(尤其是"解释/说明"类)也该写进 Notion,而不是只在聊天对话里讲一遍。之前只把 CLAUDE.md 的"Notion [...] Instruction Workflow"理解成"字面括号存在才触发写回",这次被证明理解窄了。教训:以后处理任何源自 Notion 链接的任务,先默认"答案要落在 Notion 上",聊天里给的应该是简短摘要/指引,而不是把完整内容倒在对话里就算完成——字面 `[...]` 只是这个更广规则的一个特例触发器,不是唯一触发条件。另外这次修正成本较低:Notion 内容用 append(insert_content position=end)方式补写,没有需要撤销的破坏性动作,发现问题后可以直接原地修正。

L082 UTC 2026-07-06T21:25:04Z: Notion API 对"未共享给当前集成的页面"返回 404 object_not_found 而非 403 forbidden,这是有意设计——404 不会向未授权调用方泄露"页面存在但你无权限"这个更具体的信息。遇到 404 时,不要默认是自己 URL→UUID 转换出错,应先用页面标题做一次 API-post-search 交叉验证:搜索接口只会返回已经共享给该集成的页面全集,若结果里没有任何 ID 前缀匹配,就排除了"拼错 ID"、坐实是"未共享"。此模式已在本项目至少 3 次不同页面上复现(coscientist-vs-heuristic-learning 页、P063 记录的 TPU MFU 页、以及本次的 heuristic-learning-market-making 页),标准修复动作固定不变:用户去 Notion 页面 "···" → Connections 里加 "cc" 集成,无需改代码或 ID。

L081 UTC 2026-07-06T21:29:58Z: Notion 方括号指令里出现无法解析的缩写/疑似语音转写错误(如"opu")时,不要顺着当前会话最活跃的项目主线(本例是 HPC/HyperXVLA/R1_Mamba3 checkpoint 讨论)去构造"最可能"的解读再直接写回答案——即便该解读有一定项目证据支撑(job号、命名规律等),猜错并划掉方括号比不回答更糟(制造"已解决"假象)。正确做法是用 AskUserQuestion 列出几个具体候选项直接问用户,成本很低。本例中用户给出的真实答案("opus",指 Claude 模型本身)与本会话完全不相关的另一个领域,证明了猜测方向可能整体跑偏。同时验证了 claude-api skill 的强制触发规则确实有效:一旦 prompt 出现"Opus"这类模型名,必须先读 skill 缓存的模型目录表再作答,不能凭训练记忆——本例中"Opus 曾经是 200K 上下文"这条训练记忆是对的,但"现在还能不能拿到 200K 版 Opus"这个时效性问题必须靠 skill 内当前缓存(含退役日期表)才能给出准确答案。

L083 UTC 2026-07-07T11:53:38Z: 调研 Claude Code CLI 自身崩溃(而非用户项目代码 bug)时,本地根因验证手段有限(闭源、无源码、无法附加调试器),更快转向:①只读系统诊断(cgroup/内存/架构)先缩小假设范围;②委派 claude-code-guide agent 联网匹配已知 issue;③但 agent 联网返回的具体引用(issue 号码、精确版本更新历史)不能照单全收——至少要抽查其中"准备让用户直接执行"的那条建议去官方文档源头核实(本例用 WebFetch 核实了 `autoCompactEnabled` 确实存在),核实不到的部分必须明确标注"未经验证"再转述给用户。另外:排查"总是在固定进度%崩溃"类问题,要去找背后决定"为什么恰好是这个点"的资源边界(本例是 cgroup 内存硬上限这个固定值)——资源边界+固定输入大小→崩溃点自然固定,这本身就是支持假设的间接证据。

L084 UTC 2026-07-08T00:00:00Z: 当 Notion [...] 批注问的是用户声称"本地有"的具体数据集/代码库的事实性问题时(本例：我已经开始基于通用 WebSearch 知识作答，用户打断说"本地有这个数据库啊")，应立即停止依赖通用知识，先去找真实本地项目目录——本地的一手事实(确切变量代码、真实试过/修过/发现过什么、实际回归数字)在具体性和可信度上都远超任何关于该数据集的通用网络知识，本例中直接推翻了我原来的工作假设(仅靠 WebSearch 大概率会写出"疫情打断了2020年数据采集"这种笼统理由；本地代码揭示的真实、具体、更站得住脚的原因是：W1-4整条流水线依赖的 Harmonized 跨波协调数据集根本还没扩展到 Wave 5，被迫手工拼出一个带代理变量的2020波次)。另外：一个 Notion 页面背后可能站着一个大得多的本地研究项目，自带 findings.md/progress.md/task_plan.md 追踪文件——用户指向某个项目目录时，应优先找这几个文件名(与本账号自己的每轮记录规范同构，说明不少协作者/以前的 Claude session 都用同一套模式)，再去读原始脚本/数据，因为这几个文件把数周工作压缩成了可快速读完的摘要。另外验证了安全规则里 `ls -la <large_dir>` 风险的准确含义：风险特指"逐条目 stat 的开销"，不带 -la/-R 的单层裸 `ls <dir>`(哪怕目录大小事先未知)本轮反复使用无事，是正确的低成本第一步，与明确禁止的递归/高stat开销模式是两回事。

L085 UTC 2026-07-08T22:45:00Z: 处理 Claude Code 自身崩溃时,不能把安装渠道名称当成运行时事实。当前 `doctor` 虽然写着 npm-global,但实际 `bin/claude` 是指向包内 native `claude.exe` 的符号链接,`file` 也确认它是 ARM64 ELF；因此"换 npm 安装就会跑系统 Node.js,从而绕开 Bun"在 2.1.205 这个本机状态下没有被验证。以后给 workaround 前必须检查真实 bin wrapper/package layout,尤其是 Claude Code 这类安装机制变化快的 CLI。

L086 UTC 2026-07-08T22:55:00Z: 要绕开 Claude Code 的 Bun/native runtime,不能只说"装 npm 版",而要指定一个仍以 `cli.js` 暴露 bin 的旧版本并验证 `file -L $(which claude)`。本轮查到 2.1.112 是可用的 Node-era pin,2.1.113 已经变成 native binary；降级后还必须禁用自动更新,否则 Claude Code 会把自己升回最新 native build。旧 Node-era CLI 的 `claude doctor` 在非 TTY runner 中会因 Ink raw mode 报错并挂住,这不是用户真实终端下的启动验证,以后不要用它作为旧版 CLI 的自动化健康检查。

L087 UTC 2026-07-08T23:00:00Z: 当用户对 fallback 版本过旧不满意时,要把版本新旧与运行时类别的冲突讲清楚,不要继续把 workaround 当作无代价修复。Claude Code 2.1.113 以后 npm 包已切 native binary,所以"更高版本但不用 Bun/native"当前没有官方路径；正确沟通是给用户选择权,并说明 wrapper 与 Miniforge binary 的 PATH 优先级。

L088 UTC 2026-07-08T23:08:00Z: `claude update` 自身会管理 PATH 前置的本地 `claude` 安装/链接,可能在报告 multiple installations 后移除或改写 `.local/bin/claude`；这不应被误判为手动删除。切换 Claude Code 版本后要执行 `hash -r` 再测 `type -a claude`,否则当前 shell 可能继续缓存旧路径并报 No such file。

L089 UTC 2026-07-08T23:31:43Z: 用户只说"update to a notion page"时,不要猜测最近页面或任意 wiki 目标；Notion 写回必须有明确 page/block URL,否则应先索要目标链接。

L090 UTC 2026-07-08T23:41:54Z: 用户在 Notion 上下文里说"创建一个新的界面"可按"新建 Notion 页面"处理；若无父级页面/数据库,使用 standalone/private page 是低风险默认。Notion `fetch` 对 `notion://docs/enhanced-markdown-spec` 返回 invalid URL 时,应使用最保守 Markdown 语法并在创建后 fetch 验证,避免复杂 block 语法。

L085 UTC 2026-07-08T01:30:00Z: "做新实验"类任务的忠实度自检法: 在跑新实验的同一份重建管道里先复现已知的头条数字作为锚点。本例纯W1-W4重建给 ln_pension→Q=0.002795 对草稿0.002949、ADL=-0.0066(p=0.065)对草稿-0.0058(p=0.062), 近乎一致 → 立刻证明(a)重建口径正确(b)草稿数字确为W1-W4而非5波, 后续三个新实验才站得住。若锚点对不上, 说明口径错了, 新实验结果不可信。另一条: 学术论文里绝不能用我的重建数字悄悄覆盖用户草稿已有的头条数字——应保留用户数字为主设定, 把新实验作为清晰标注、可复现的"扩展"章节加入, 并说明重建验证了头条。第三条(Overleaf协作): 远端可能被 "Update on Overleaf." 这类编辑器侧提交重置(本例把我的完整论文清成空骨架), push 冲突时先 git show origin/main:file 看清远端完整内容再决定, 用 reset到远端+恢复内容+普通push(非force, 远端提交留作父节点可回溯)而不是 force-push 硬覆盖。第四条: 一个数据集常有多个协调版本(Harmonized跨波协调版 vs 原始问卷版), 原始文件里找到的变量名(ca015/db014/dc024)在协调版里是完全不同的命名(h{w}kcntf/r{w}iadlza/r{w}flonel); 要做与论文口径一致的实验必须去论文实际用的那个版本里找对应变量, 这也正是"W1-4与2020不可比"的同一根源。

L086 UTC 2026-07-08T02:00:00Z: 与 Overleaf git 协作时, 若用户同时开着网页编辑器, 会持续产生 "Update on Overleaf." 自动提交(哪怕无实质改动, 重编译也会提交), 造成 push 反复被 non-fast-forward 拒绝的 race。不要 force-push 硬覆盖(会丢用户在编辑器里的实时手改)。稳妥循环: git fetch → git show origin/main:<file> 看清远端完整内容与用户改了什么 → 备份自己的版本 → git reset --hard origin/main → 恢复自己的内容(有意识地合并用户的合理改动, 如本例保留但其实是回补被误删的\author) → 普通 push。中文 LaTeX 上 Overleaf 的正确配方: \documentclass[UTF8]{ctexart} + 文件首行 % !TeX program = xelatex, 删掉 inputenc(与xelatex冲突); "英文骨架+中文正文"是中文作者面向双语评审的常用格式, 表格/图注保持英文利于跨语言单独复用。

L087 UTC 2026-07-08T23:38:44Z: 用户强烈反感 superpowers:brainstorming 的 HARD-GATE(必须先产出并批准 design 才能动手)。教训: 当 Notion 页面已给出明确执行指令、仅有轻微参数歧义时, 不要机械走 brainstorming 全套多轮 gate; 用户偏好直接执行, 歧义用 1-2 个 AskUserQuestion 点澄清即可。禁用单个 plugin skill 正道 = enabledPlugins 置 false(整插件); 只删插件内某一个 skill 需重命名其 SKILL.md(在 versioned cache, 插件更新会复原, 属临时手段)。

L088 UTC 2026-07-08T23:46:32Z: "卸载插件"三层含义按需全做: (a) 禁用 enabledPlugins=false 可逆留文件; (b) 物理删除 rm 缓存 + 清登记 + 清 settings, 彻底须重装恢复。删多版本插件注意 cache/<plugin>/ 下可能有多个版本子目录(6.1.0 + 6.1.1), rm 父目录一次清掉。改 JSON 删末尾条目要同时处理前一条尾逗号, 改完必须 python json.load 验证防启动失败。

L089 UTC 2026-07-08T23:57:22Z: [smaller-dataset] 两条教训。(1) Notion 页面即使 0 个字面 [...] 括号,只要是 Notion 来源任务且含执行指令,仍必须把答案/结果落回 Notion 页面(见 memory feedback_notion_answer_must_land_in_notion),不能只在 chat 里回。(2) 用户 spec 出现内部数值矛盾(train 大小"2%" vs "一个季度")时,按第一性原理应主动指出并询问,不能静默挑一个——采样规模是后续全部数据代码的前提,猜错=全部返工。
L055 UTC 2026-07-09T00:01:59Z: P0 - when a user hands a NEW Notion page that duplicates an OLD page's content, do NOT assume the carried-over answers are still authoritative. Check for a newer human-authored section (here: '蓝色可见版'/'最终决定') that may CONTRADICT earlier AI-written conclusions on the same page. Same-day timestamps can still have a real ordering; when two 'final' decisions conflict, surface the conflict explicitly (with a reconciliation callout) rather than silently picking one or averaging them.
L056 UTC 2026-07-09T00:04:09Z: STANDING RULE from user: never write bare 'base' when discussing HyperXVLA weight init - always say base_policy (self.base/self.param, live trainable hypernet param) or base_model (args.xvla_model, real pretrained XVLA checkpoint loaded only under --seed_hyper_from_xvla_transformer). Applies to all future chat/Notion/code-comment writing on this codebase. Also watch for the false-friend base_model_prefix HF attribute (modeling_hyper_xvla.py:78), unrelated.

L090 UTC 2026-07-09T00:07:21Z: [smaller-dataset] 教训:用户给的路径可能错标,落地前必须验证。此次 "lob_pipeline_squashfs" 实为打包的 conda 环境(unsquashfs -s → no valid superblock),非 LOB 数据;真实数据靠 grep 训练/eval 脚本里的 SQUASHFS_DIR 才定位到。规则:拿到"数据路径"先确认它 resolve 到真实数据(读 index.json / unsquashfs -s 验证 superblock),再在其上做设计,避免整套方案建在错误前提上。
L057 UTC 2026-07-09T00:24:04Z: When a user asks 'why do X and Y differ', check whether the two plans differ on MULTIPLE axes simultaneously - if so, surface that as a methodology risk (echoes the exact 5285200 too-many-variables failure mode this whole project is trying to avoid), don't just explain the individual rationale for each and move on.
L058 UTC 2026-07-09T00:25:14Z: When a user's follow-up message reads like a correction ('我记得wd设置为0'), check whether they're right BEFORE explaining further - here WD genuinely was 0 in both table cells, the apparent 'difference' was an artifact of bundling LR+WD into one table row. Validate/confirm their memory explicitly rather than just answering the surface question.

L091 UTC 2026-07-09T00:27:39Z: [smaller-dataset] 教训:中文无标点句 "这个 train 数据集 2% 左右 做 validation" 有两种解读(train 占总量 2% vs val 占 train 2%),用户实为后者。用"哪种读法自洽"(val-of-train 让 train>收敛点)先形成推荐 + 主动问,而非静默猜,命中正确解读。

L092 UTC 2026-07-09T00:38:11Z: [smaller-dataset] 教训:把小比例(2%)作用在本就不大的集合(一个季度)上,绝对结果可能退化(val 塌到 ~1–2 天)。凡"比例×小集合"务必回算绝对规模,识别退化 case 并向用户标注(可上调比例或对该子集改用更细抽样单元),而非闷头按比例执行产出一个几乎单一日期的 val。
L059 UTC 2026-07-09T00:50:00Z: P0 - when a user says 'this is wrong' about something confirmed across MULTIPLE prior rounds (formula they wrote themselves + explicit '这两个就是我想要用的' approval + original design doc wording), do NOT immediately rebuild. Surface the exact prior evidence back to them and ask for explicit reconciliation before touching code - this is the 2nd time in this HyperXVLA thread a snap correction turned out to need disambiguation rather than blind action (1st was the delta-lora vs vanilla run-1 direction conflict). Also: if AskUserQuestion errors (permission stream closed), fall back to a direct plain-text question immediately rather than silently retrying or guessing.

## 2026-07-16 Isambard capacity lessons

- Distinguish physical nodes, GPUs, NHR, and GPU hours before sizing a fleet: one Isambard-AI node is four GPUs, so one full-node hour costs four GPU hours.
- A dashboard's start-of-month balance is not the same as currently usable balance. Reconcile allocation minus used credits and prefer the conservative value when fields differ.
- Always round a continuous concurrency estimate down to full nodes and retain budget for shared-account use, failed reruns, and an ambiguous award-end timestamp.
- A submitted or sleeping job is not evidence that an experiment is running; availability claims must count RUNNING workers with real payloads.
- For 16 independent workers, the Slurm array range is `0-15`. Keep the payload explicit in a short command surface so a capacity decision cannot silently become 16 idle allocations.
- When the user supplies an exact neutral job name, use it verbatim across submission, logging, and monitoring instead of inventing another label.

## 2026-07-16 daily evidence lessons

- Launcher topology matters to coverage semantics: this batch is one multi-node job, not sixteen independent single-node jobs.
- Auto-resume changes Slurm JobName, so an evidence logger must follow the documented name chain rather than match only the initial name.

## 2026-07-16 first-principles reporting lesson

- Do not let second-level precision displace the operational question; minute-level reporting is sufficient when the real concern is continuous 16-node coverage.

## 2026-07-16 allocation-state lesson

- `sbatch` acceptance proves that the resource request is syntactically and administratively valid, not that resources are allocated; require RUNNING plus `NodeList`/`AllocTRES` evidence.
- Follow the user's latest scope correction directly: this request holds a 16-node allocation and does not launch training.

## 2026-07-16 composition lesson

- A disappeared PENDING candidate is not automatically a failed job; inspect `sacct` before interpreting queue absence.
- Keep the submitted workload simple and isolate fleet convergence in an independently testable outer monitor.
- `sinfo` idle-node totals do not expose hidden queue order or reservations when Slurm `PrivateData` is enabled.
- Composition is simplest here: keep candidate jobs policy-free and let a short-lived outer process own observation, cancellation, and audit logging.
- Cancellation must be bounded by both an explicit candidate-ID set and the expected job-name prefix.
- Verify blocking scheduler commands against a real PENDING job; local man-page semantics did not match Isambard's observed `wait_job` behavior.
- An unchanged queue should produce no repeated audit rows; process liveness plus one start event is sufficient between state transitions.

## 2026-07-17 allocation attach lesson

- A RUNNING sbatch allocation that is sleeping is not itself an interactive session. The practical attach pattern is to start new `srun --jobid=<job>` steps inside the allocation.
- Distinguish a fleet allocation job from its follow-on logger: a `BeginTime` PENDING logger can be queued while the real 16-node allocation is already RUNNING.
- Include `--overlap` in reuse commands for this sleeping-allocation pattern so Slurm can create concurrent steps instead of treating existing batch resources as unavailable.
- For user-facing attach sessions, prefer an explicit tmux socket path under the task directory; the default `/tmp/tmux-<uid>/default` socket may disappear or be unavailable across login-node context changes.
- A shared-filesystem tmux socket path does not make the tmux server portable across login nodes. Treat tmux attach instructions as host-local unless the user is on the same login node where the tmux server was started.

## 2026-07-17 dual hypervla lesson

- When the user clarifies a training split (vanilla vs lora), maintain distinct configuration flags (`weight_head_type="vanilla"` vs `"low_rank_delta"`) and document their different optimizer grouping behaviors (StaticParameterHead splits parameters into static vs generator groups, whereas LowRankDeltaHead uses a single optimizer group).
- If a bracketed user instruction in chat does not exist on the target Notion page, insert it with strikethrough decoration in its correct context on the page, and then place the callout response directly below it, to preserve page integrity and formatting rules.
- Ensure absolute formula precision when documenting codebase modules: delta-lora uses separate u_head and v_head to compute dynamic delta via torch.bmm(U, V), while vanilla makes both weight matrix and paired bias static parameter heads (though soft_prompt and norm parameters remain context-generated).
- When the user requests codebase snippets, present the exact class/function definitions including their path references, mapping their parameters and docstrings precisely to the theoretical formulas.
- Document the formula matching the user's syntax representation for vanilla weight matrix and bias, keeping it parallel to the delta-lora representation.
- Explicitly clarify the absence of context-conditioned matrices U or V in vanilla mode when explaining mathematical structures.
- Differentiate between parameter generation equations in the HyperNetwork ($y = W$) and downstream execution equations in the base network ($Output = W \cdot h + b$) to avoid confusion over variables.
- Note that in vanilla mode, dynamic context-conditioned generation is not completely absent; soft_prompt and norms are still generated using context-dependent OutputHead linear equations ($y = W_{\text{head}} \cdot x + b_{\text{head}}$).
- When the user corrects the high-level semantic definition of a config mode (e.g. vanilla mode means direct context-conditioned linear generation instead of static parameters), draft an implementation plan immediately to update the dispatch code and test suites.
- When running CPU or GPU smoke tests that load/instantiate large neural network models on QoS-limited login nodes, they may be OOM-killed due to strict resource limits. Always run such tests on compute node allocations (e.g. srun --jobid=5678750 --overlap) to ensure sufficient RAM.
L060 UTC 2026-07-17T12:16:12Z: 教训——不要把 state.json.batch_script 字段(它只是 logger 每日读取的 'command provenance',见 plans.md 第 3 行)当成 job 实际执行的命令。判断一个 SLURM job 是真训练还是空占位,先查两个信号:(1) sacct step 名(python/run_train/node_wrapper vs 裸 bash/sleep);(2) StdOut 文件大小(真训练 MB 级 vs 空转 0 字节)。本轮我据 batch_script 字段误报 5678750 '正在跑 R1_Mamba3 训练、16 节点 x 4 GPU 全在算',被用户当场纠正。断言 job 活动状态前必须先看这两项。
L061 UTC 2026-07-17T12:21:52Z: 在已存在的 allocation 上跑分布式,srun 必须从'能看到完整 allocation'的上下文发起(login,或 job 的 batch 上下文),不能嵌套在一个 --nodes=1 的 --pty step 里——嵌套 srun 继承外层的 SLURM_NNODES,把池子锁成 1 节点。mode A(交互单节点)与 mode B(16 节点分布式)是并列的两条 login 命令,不是父子关系。
L062 UTC 2026-07-17T16:39:00Z: 用户"把 16 节点改成 1 节点"的请求不是孤立的占位缩容,它与中途发来的 smaller-dataset 页耦合:1 节点是构建固定小数据集这一重预处理任务的算力载体(load+采样+保存 48 个月 SP500 SquashFS 不能在 login 节点跑)。教训:收到看似简单的"缩容"请求时,先看是否有伴随的科学目标改变了 job 的用途(占位 vs 真跑数据/训练)。
L063 UTC 2026-07-17T16:47:00Z: 构造派生数据集时,输出格式应匹配下游消费管线(此处 SquashFS 分片)而非另发明 manifest/散 npy——既省 dataloader 改动,又符合 Lustre "1 分片=1 inode" 的元数据安全模式。选格式先问"谁来读它、怎么读"。
L064 UTC 2026-07-17T16:55:00Z: subagent 的体量"估算"可能差 100 倍(此处 1-2GB vs 实测 143GB/分片),任何据此定 sizing 的 materialize 作业前必须 stat 真实文件核实。另:同一数据集不同月份的 index 布局可能不一致(仅 test 月有 sidecar,其余在分片内),不要假设布局统一。
L008 UTC 2026-07-17T17:07:48Z: 现有 run_*.sh (根目录) 指向 s5j 账户路径且是单节点/h192/无 weight_head_type,是过时副本;真正 u6gb 生产脚本在 scripts/ (train_hyper_200k.sh 等)。教训:提交前必须核实脚本的账户路径+init配置+mode参数,'直接sbatch现成脚本'会跑错实验。另:代码 help 文本可能滞后于实现(vanilla 语义),以 forward 代码为准,并用 smoke 实测分组。
L065 UTC 2026-07-17T17:11:44Z: 判断 checkpoint 能否在 CURTAIL 短训练里落盘,不能只看 CHECKPOINT_EVERY 语义,要读训练循环里 break 与 ckpt 检查的先后顺序(此处 break 在前,末步 ckpt 依赖整除关系)。另:模型输入维度未必等于原始数据列数(book 503=volume image 变换后),判断数据兼容性要看 encoder 第一层参数形状而不是数据文件列数。
L066 UTC 2026-07-17T17:04:00Z: 定 sizing 前先确认数据 SCOPE(哪些年/月),不只是 fraction——"1/4"在用户点明池=2025 前一直有歧义;分数脱离分母无意义。另:把一次性重构建与长期占位 allocation 解耦,别把交付物卡在最难调度的 job 上,短作业 backfill 容易得多。
L067 UTC 2026-07-17T17:17:00Z: 生成的构建脚本提交前在 login 做 py_compile + bash -n 语法自检——比在排到的节点上因一个 typo 白烧便宜得多。数据集时间戳在 SUBMIT 时生成并 --export 传入,让 resume 复用同一目录而非另 fork 一个。
L068 UTC 2026-07-18T17:49:22Z: Edit 工具会拒绝写 symlink 目标 ('Refusing to write through symlink'), 必须先 realpath 解析再对真实路径 Read+Edit。另: alias 名与系统命令同名(cc=C编译器)仅遮蔽交互 shell, 脚本/Makefile/sbatch 的非交互 shell 不展开 alias, 不受影响。
L068 UTC 2026-07-18T05:55:00Z: 别把大块数据暂存到 tmpfs(/dev/shm):df 报的是 tmpfs SIZE 而非 RAM 余量,所以"≥200G"校验会通过、拷到一半 ENOSPC。子集分片构建应用 mksquashfs pseudo-file(-pf,cat 从源挂载点流式喂)彻底免暂存,与节点本地盘大小无关。另:很多 GH200 节点本地只有 tmpfs,提交重 I/O 前先确认真实本地磁盘。
L069 UTC 2026-07-18T06:02:00Z: 本集群 $HOME 是 Lustre 项目目录(/projects/public/<proj>),不是 VAST /home/<proj>/<user>(后者放 conda、配额仅 ~100G)。"存到 home"有歧义——先解析 $HOME 再定数据位置;大数据集属于 Lustre 项目($HOME),不属于 VAST /home。
L066 UTC 2026-07-18T18:10:55Z: 把仓库里恰好存在的 mamba3_smoke.yaml 的 USE_WANDB=False/WANDB_MODE=offline 抄进 selftrain 配置并对用户称之为"惯例"——错误归因,且违反 CLAUDE.md 明文规则(训练 job 应 WANDB_DIR=$TMPDIR + WANDB_MODE=online)。后果:三个 selftrain 训练 job 无 wandb URL/曲线可查(offline run 落在计算节点本地盘,job 结束即不可 sync)。教训:(1)复用仓库现成配置前必须对照 CLAUDE.md 成文规则逐项校验;(2)向用户解释配置选择时必须给出处,不得把"某文件恰好这么写"说成"惯例"。
L070 UTC 2026-07-18T06:15:00Z: mksquashfs -pf(pseudo-file: 目标路径 f 权限 uid gid cat "源")把源文件流式喂进分片,零 bulk staging——tmpfs-only 节点上打包大子集的正解。上真机前先用 2 个假文件在 login 上跑 mksquashfs→unsquashfs 往返验证 pseudo-file 语法。
L071 UTC 2026-07-18T18:22:45Z: 连续占位覆盖,继任者必须无 --dependency 提交(afterany 会把整段排队推到当前 job 结束之后→巨大缺口),应在 end−predicted_wait 提交让排队与当前运行重叠;若 predicted_wait>剩余则立即提交(已逾期)并接受残余缺口,或缩短 walltime 缩短排队。
L067 UTC 2026-07-18T18:24:32Z: (1)"job COMPLETED 0:0"≠训练成功:restore 组 6 job 全 COMPLETED 但 loss≈71(数值失败),判训练成败必须看 loss,且要区分"训练时 loss"与"load 后 loss"两种口径。(2)sbatch 命令行 --time 不会传导到 batch 内的 SBATCH_TIMELIMIT/SLURM_TIMELIMIT,MAX_JOB_HOURS 会 fallback 默认值导致超时保存窗口错位,长训 job 必须显式传 MAX_JOB_HOURS。(3)并行 Claude 会话同做一个任务时:checkpoint 按 job ID 隔离天然防冲突,但 Notion 写回/git commit/yaml 文件是共享冲突面,动笔前必须重读现场。
L072 UTC 2026-07-20T15:39:49Z: 建 symlink 用 ln -sT(不解引用+已存在即失败)防经典陷阱: 若链接名已存在且指向目录, 裸 ln -s 会把新链接建到目标目录内部而非报错。symlink 目标有包含关系不构成冲突, 仅是多入口同 inode; 唯一行为差异是 shell 逻辑模式 cd .. 沿链接名回退, cd -P 沿物理路径回退。
L073 UTC 2026-07-20T15:53:00Z: 同源 batch 脚本分叉后, 最危险的差异不在模型超参而在流程配置: FLAIROx 版 auto-resume 只回传 RESTORE_PATH 不回传 D_MODEL 等 env, resume job 会静默回落 360M 默认值训练错模型; 且用 ls -1td 找 checkpoint 触犯 Lustre 反模式。审查"别人的配置对不对"必须查三层: sbatch 默认值→wrapper env→argparse 默认值, 任何一层的 default 都可能在另一 repo 已被推翻(如 75M 的 n_layers 12→6, token_mode 24→26 硬编码)。
L073 UTC 2026-07-20T15:54:26Z: 审核外部使用者配置时, 基准必须取 node_wrapper.sh 实际 CLI 与 batch env 默认, argparse 默认全是本地 debug 值(d_model=32, local_steps_k=10, ssm_lr_base=1e-3, ignore_times=True 与生产相反); 另: 数据文件长度恰好等于窗口长 + randomize_offset=True 会因保守公式 (rows-(window-1))//window 静默得到空数据集, 这才是 collaborator 必须关 random_offsets 的根因(len=0, 而非越界报错)。
L074 UTC 2026-07-20T15:58:42Z: 对比表里压缩标签(如'硬编码''合成数据强制')若不带机制说明, 读者无法自行还原因果; 涉及非显然机制的行应在表下配一段解释或示意图。
L075 UTC 2026-07-21T11:10:54Z: Notion update-a-block 三件套缺一不可: block 类型对象作顶层额外参数(不用 type 包装)+ archived:false 显式传 + annotations 完整六字段对象(部分对象被静默剥离, 返回 200 具有迷惑性, 必须读返回体确认 strikethrough:true 才算成功); 另: 给用户"最简命令"时删 flag 要逐个有依据(--ntasks-per-node=1 是 sbatch 默认可删, --gpus-per-node=4/--mem=0 在 GH200 上决定是否整节点不可删)。
L076 UTC 2026-07-21T11:14:30Z: skill 里的共享落账路径(s5e 的 live_jobs.md/active_monitors.jsonl)在其他项目账号(u6gb)下不可写, 正确处理是保持 append-only 协议不变、路径 fallback 到 workspace-local 同名文件, 而不是跳过落账步骤; 占位 job 的监控关注点与训练 job 不同: 无日志可 grep, 核心信号是 PENDING→RUNNING 转换与"过早消失"(30min 内 GONE = 异常, 需 sacct exit code 定性)。
L077 UTC 2026-07-21T11:49:00Z: 占位 job 的 30min 强制监控窗口在排队期只能证明"无异常", 用户真正等的信号是 PENDING→RUNNING 翻转; 正确收尾是续接低频 until-loop 监控(只在状态翻转时发事件, 排队期沉默), 而非以"窗口结束"作结; 监控过滤器必须同时覆盖成功态(RUNNING)与异常态(队列消失→sacct 定性), 只盯成功态会把 CANCELLED/FAILED 静默成"还在排队"。
L1784638779 UTC 2026-07-21T12:59:39Z: backfill 资格取决于 job 自身 TIME_LIMIT 与 eligibility,不能"继承"前面短 job 的排队优势;--dependency 只会延迟后继 eligibility(backfill 不为 ineligible job 做预留),与既有 placeholder 协议 NO --dependency 的教训一致。本集群 priority 恒=1(权重全 0),早提交(更小 job ID)= 更靠前 FIFO 位置,age 累积在别的集群是机制、在这里不是。
L1784639433 UTC 2026-07-21T13:10:33Z: 在 PrivateData 全开的集群上,排队优化只有三类杠杆:(a) 让 job 好塞洞(短且准的 --time、--time-min、小资源、避免 --contiguous),(b) 早提交锁 FIFO 顺位(--begin 只推迟 eligible 不丢顺位),(c) 探针 job 实测等待。--time-min 是与 checkpoint+auto-resume 配套的正统 backfill 技巧,拿到的实际 limit 用 squeue -h -j $SLURM_JOB_ID -o %l 读。
L078 UTC 2026-07-21T13:14:40Z: (1)'command not found' 且 symlink 存在时, 先查 symlink 目标是否缺失: broken symlink 无法通过 bash 的 access(X_OK) 检查, 表现与 PATH 缺失完全相同; (2)NFS 上被运行进程持有的已删文件(.nfsXXXX)unlink 必 EBUSY, 但同文件系统 rename(mv)合法, 是绕开'更新器要删自己正在跑的文件'死锁的标准手法; (3)诊断入口用 /proc/PID/stat 的 ')' 后第2字段取 ppid + readlink /proc/PID/exe, 比 ps 解析 comm 可靠(comm 含空格会错位)。
L079 UTC 2026-07-22T09:52:10Z: 复用并再次验证 reference_notion_page_share_404:Notion 404 首先假定"未分享"而非"坏 ID"。深链场景下 block 404 时,判据是"父 page retrieve 也 404 + 标题搜索里找不到该页" → 确证未分享。不要对未分享页做任何 workaround(如猜别的 block、抓页内其他链接),直接请用户加集成。
L080 UTC 2026-07-22T09:58:13Z: Notion 深链可能指向 image 块而非文字。此时"解析/回答该 block"= 下载签名图并读图,而非套 [...] 工作流;若图内无指令,按第一性原理停下确认意图,不臆测任务(如擅自 clone 私有仓库)。S3 图 URL 带 X-Amz-Expires=3600,须即时下载。
L081 UTC 2026-07-22T10:06:27Z: (1)私有 GitHub repo git clone 报 "Repository not found" 而非 403,是鉴权失败的伪装(GitHub 防泄露);根因常是 credential store 无 github.com 条目(本例只有 overleaf)。(2)安全用用户 PAT:git credential approve 走 stdin,clone 用干净 URL,token 不进 argv/不进 git 追踪文件;更新 token 须先删旧 github.com 行(store helper 取首个匹配)。(3)读 repo 前先比对 README 引用 vs 实际文件树+git log:本例 README "start here" 全断(docs 从未 commit、results.md/notes.md 已删),"Remove notes for now" 式 commit 是分享前剥离叙事文档的信号,代码在但结论日志需从历史捞。
L082 UTC 2026-07-22T10:19:36Z: 判定两 repo 关系先读 vendoring 脚本注释再看 import:lob-mae 用"PORT(copy)into flow.py"而非 submodule/import,pin SHA 即上游快照锚点。移植类关系的正确讲法=分三层:provenance(copy/import/submodule+SHA)、各自定位(生成原型 vs 研究 harness)、port diff(内核逐字同 / 边界重封装:loss 内融、加 encode transfer、换 conditioning、改配置命名)。sigma-flow 是纯 velocity 网+外置 solver/VAE(能生成);lob-mae 的 FlowDiTSSL 融 loss+加 frozen-probe(scaling 只需 loss 曲线+特征探针,不需 solver),生成走单独 rollout 脚本。
L083 UTC 2026-07-22T10:25:45Z: "A 用了 B 的东西吗"要在文件内追类依赖链到叶子,别停在"同一文件/同一目标函数"。判据分三级:共享具体类/函数(真复用)> 共享工具函数但系共同上游祖先(伪复用,如 Wan2.1 的 timestep embed)> 仅共享范式但各自重写(非复用,如 lob-mae 把 rectified-flow 反向重写:t=1=data vs sigma-flow t=0=data)。约定反转(时间方向、t 采样 uniform vs logit-normal)是"独立实现而非移植"的硬证据。
L084 UTC 2026-07-22T11:12:04Z: (a) sbatch --begin 可把"提交动作"与"生效时刻"解耦:在 PriorityWeightAge=0 的集群上推迟生效零成本,而立即提交消除了"到点没人提交"这一覆盖断点单点故障(07-19~21 零覆盖的根因)。(b) queue_predictor 的 record 必须在每个继任者启动后立刻补,漏跑会让 EMA 停在旧样本(37655s vs 实测 52004s,低估 38%)。(c) Notion update-a-block 的块类型键必须放请求体顶层(body.code={rich_text:...}),包在 type 字段里会 400("body.type should be not present");与 memory 里 table_row quirk 同源。
L068 UTC 2026-07-22T12:16:29Z: XLA 的 --xla_gpu_nccl_terminate_on_error=true 会把 Slingshot NET/OFI 瞬时错误(RC265 EPERM)升级为 SIGABRT 全 job 崩溃;12h 级长训因此必须依赖 15min auto-checkpoint+resume 而非祈祷网络稳定。诊断多节点 SIGTERM/SIGABRT 时 grep 模式含 'timeout' 会被 XLA_FLAGS 行占位吞掉真信号,应先排除 flags 行再匹配。另:并行会话写的结论(如 [Completed])上页面前必须用 loss 等硬指标复核,错误结论要当场纠正而非沉默追加。
L085 UTC 2026-07-25T20:47:00Z: (a) 覆盖类任务的续链动作必须内嵌进被调度载体自身(sbatch 启动时自提继任者)——daily logger 靠此活了一周,实验链靠"agent 在场"死了三次;新机制上线前先问"没有我它还能活吗"。(b) 用户约束变化会反转最优设计:"重叠可接受"把方案从 --begin 对齐(防重叠但赌 EMA 准,EMA 已实测低估 43%)反转为立即排队(等待与走时并行,零 gap 换重叠);先采约束再定机制。(c) Notion MCP update-a-block 顶层块类型属性 PATCH 实际生效,但响应体返回更新前旧状态,必须 retrieve 复核——本轮差点因此误判失败而绕去 REST token 流程。
L086 UTC 2026-07-25T20:56:05Z: (a) 福利指数含解释变量的收入成分=机械重合,审稿硬伤,重构后效应缩 3/4 但故事更强(心理渠道浮出);(b) 等权指数的 TWFE 系数可按成分回归精确算术分解(beta=Σbeta_j/k),比 Gelbach 干净且无需辅助假设,Gelbach 在 base 系数≈0 时贡献率分母无意义;(c) notion MCP update-a-block 的 type 参数 schema 有缺陷,直接传顶层 paragraph 参数(additionalProperties 透传)可绕过;(d) fillna(mean) 构造指数会掩盖成分缺失结构,complete-case 样本流程表必须先于任何回归定型。
L087 UTC 2026-07-25T21:08:54Z: Overleaf 项目内容可能与预期完全不同(给的地址是已有活跃项目而非刚导入的 zip)——push 前必须先 clone 读内容再决定合并策略;用户中途指令'不好修改就写新文件'=保留旧版并存新版,git checkout 恢复+新文件命名是最小破坏路径。
L088 UTC 2026-07-26T10:32:44Z: (a) 描述统计是 bug 探测器:ADL 均值0.37/81%为零 立即暴露'能力得分'实为'困难计数', 任何指数构造前必须先看每个成分的分布与语义;(b) 强第一阶段+零约化式是有信息的组合, 不是失败:它区分了广延边际(领取事件)与集约边际(比例调待)的效应结构;(c) 论文扩写指令(至多28页)按 tex 行数×0.045 页/行估算, 604行≈26-28页。
L089 UTC 2026-07-27T10:58:30Z: 论文修订轮的通用规律:作者要求删除修订史自引('前稿/早期版本/已更正')是定稿信号,此时除指令点名位置外还须全局 grep 隐性修订史(熵权翻转史/样本口径演变/bug更正说明),一并转写为中性方法论表述;修订史留在 git log 与 progress.md,不留在论文正文。
L090 UTC 2026-07-27T11:58:12Z: 学术论文翻译轮的检查表:CJK 字符残留 grep([一-鿿])是唯一可靠的完译验证;中文'双语术语括注'风格在英文版应整体去除而非直译;documentclass 从 ctexart 换 article 时注意 inputenc/fontenc 补回。
L091 UTC 2026-07-27T13:04:27Z: (1) Notion MCP update-a-block 的 body 须把块类型（callout/paragraph）作为顶层额外参数直接传，传 type={...} 会 400（"body.type should be not present"）；创建时 annotations 可只给部分字段，更新时仍建议全六字段。(2) patch-block-children 在页首 after 锚点会返回整页尾部（100KB+ 被持久化），改用"答案作为 to_do/callout 的 child"（block_id=条目本身、无 after）返回体只含新块，增量写长页面应默认用 child 模式。(3) 拟合脚本对照已有 JSON 复现论文数字是引用外部结果前的必要 sanity check：本轮借此发现 α 多局部极小（huber 更低但 α 完全不同），成了 rebuttal 中"α 弱识别 β 强识别"论点的直接证据。
L092 UTC 2026-07-27T22:00:41Z: wandb project 定位不能停在"配置链指向哪"：本例 sweep/wrapper/日志三方一致指向 neurips-transformer-scaling-runs，但 API 拉取 lastHistoryStep 后发现 59 runs 仅 1 个有曲线——"配置正确同步"与"metrics 实际落盘"是两回事，log 调用点（epoch 边界 vs 定时）与运行模式（train-only+curtail）的交集决定曲线有无；报告 wandb 位置时必须附数据完整性核查。
L093 UTC 2026-07-27T23:50:31Z: 判定训练是否完成的证据强度排序：checkpoint 目录内 step 子目录 > 日志尾部步数 > sacct/ledger 状态 > wandb 状态（最弱）。两个指纹：目录仅含 metadata=跑起但未到首个 save 间隔；eval CSV 尺寸集合与 sweep 尺寸表不吻合（如 34M 不在 11 尺寸 TF_GRID）=CSV 属于别的批次，不能当作本 sweep 的产出证据。
L094 UTC 2026-07-27T23:58:29Z: 多节点 job 死亡三分类判别法：(1) barrier 连坐死——某 node 日志出现 F-fatal "another task died/DEADLINE_EXCEEDED"时它是受害者，元凶=各 node 日志 mtime 最早停更且停在长操作（如 dataset 建索引）的那个；(2) scancel 死——训练正常速率推进中戛断＋无任何错误行＋sbatch .out 缺收尾块（scancel SIGTERM 连 batch shell 一起杀，谁都来不及写尾块）；(3) 真异常死——有 Traceback/NCCL WARN 实体。另：CUDA_ERROR_NO_DEVICE 在 DataLoader worker 内是 import jax 的固定噪声，出现在训练已起跑之后、多次重复，绝不能当死因。
L095 UTC 2026-07-28T00:10:36Z: "某管线账本终止≠该方向工作停止"：判断"X 之后还跑过没有"必须跳出单管线视角——相邻 exp_* 目录 mtime、sacct 整月、wandb 跨 entity 三路并查。两个关键指纹：(1) 日志中 "wandb: Currently logged in as: <user>" 决定 run 落入哪个 entity，共享节点上 netrc 归属随合作者变化，是跨 entity 失踪 run 的头号原因；(2) sacct 仅可见本账号（PrivateData），同 ID 区间内查不到的 job 是"他人提交"的信号而非"不存在"。另 wandb 后台大扫描要把 flush 输出直接落文件，禁止尾接 tail/grep 管道（缓冲吞光中间结果，timeout 后一无所获）。
L096 UTC 2026-07-29T13:12:13Z: HPC 上 wandb"幽灵凭证"判定法：job 落哪个 entity 由两指纹决定——命令行 --wandb_entity 硬编码 + 节点上 $HOME/.netrc 的 login 身份（HOME 经 sbatch 从提交 shell 继承，提交 shell 的 HOME 可能被重定向到 Lustre workspace 而非 /home）。验证历史 job 凭证路径：grep "Currently logged in as" 其 node0 日志。另：向后兼容检查一个 sweep 是否"已修复"，最快路径=读上次成功 job 日志的关键指纹行（本例 [squashfs] mounted 36/36 shards）而非静态读脚本推演。
L097 UTC 2026-07-29T13:21:00Z: 排队时长预期必须用"同形状+同时期"参照校准：5-10 时 0p2M 提交后分钟级启动的经验在满载期完全失效（同为小 job，本期 30min-1N 参照 eligible 后 13h 未起）。另两个技术点：(1) --begin 推迟提交的 job 算真实等待要看 sacct Eligible 而非 Submit；(2) squeue REASON=(None)+START_TIME=N/A=调度器尚未评估该 job，不是异常。
L096 UTC 2026-07-29T05:20:00Z: 盘点"哪些数据被训练碰过"时，wandb lastHistoryStep>0 过滤会漏掉静默消费者（curtail 模式只在 epoch 边界 log 的 run 步数照跑却零曲线，如 4-28 TF 批 tqdm 618/23552）——历史步数证据强度：本地日志 tqdm > checkpoint step 目录 > wandb lastStep。且"以 seed 为键的排列前缀"消费模型只在同一 index 域内成立：换 ticker 子集/月份子集=换域，randperm 完全不同，跨域清账必须按 (ticker,date,窗口) 实体键 join 而非 index。
L097 UTC 2026-07-29T13:30:26Z: 尺寸代号与实测 N 两套账并存时,任何范围表述必须声明口径,否则同一 sweep 会被读成三个不同实验(0.2M–350M/2.6M–293M/2M–350M 本轮全部在同一页出现)。判定'最小尺寸是否被丢'的正确路径=查拟合输入 CSV 的 size_label×seed×max_step 普查,而非对照正文范围数字;'12 configurations+2.2 decades'这类计数+跨度组合本身就是口径指纹(12 配置⇒0p2M 必在内,否则只剩 11)。另:label 在中段与实测重合、两端失真(小端固定成分主导 13×,大端外推公式高估)是 tag-by-formula sweep 的通式,复核任何 sweep 范围引用时先拉 label→num_params 映射表。
L098 UTC 2026-07-29T13:56:37Z: 口径修正必须追到 paste-ready 草稿层:同一数字在主页论证区与《Reply Drafts》子页各有一套写法,只改主页会让最终提交物(草稿页)带着旧错口径出门。范围句对账要把'端点'与'跨度数'分开验:端点可以各自有出处,跨度数(decades)一旦按代号算就是可检验的硬错(3.2 vs 2.05);grep 'decades' 是扫这类错的最快指纹。
L097 UTC 2026-07-29T06:15:00Z: srun --overlap 借占位 job 跑辅助任务时不要显式 --cpus-per-task（易与已运行 step 的 cpu-bind 冲突报 "CPU binding outside of job step allocation"）；用 --cpu-bind=none 让线程自由漂移即可。scontrol 的 AllocTRES cpu 数与实际绑定掩码可能不一致，以掩码位数为准。
L098 UTC 2026-07-29T06:40:00Z: 用户行为规则（已入长期记忆）：mid-turn 追问 → 优先停下完整回答问题，答完不等确认自动恢复原任务。另：用户的节点预留方法记录在 Notion 页 "srun --nodes=1 --gpus=1 --time=23:59:59 --pty /bin/bash --login"（39512c45-68fd-8027），--overlap 挂 step 是其非交互等价物。
L099 UTC 2026-07-29T14:15:15Z: ① 找历史 dashboard 的 O(1) 路径 = surge list(账号级枚举),优于 grep 记录文件(本次 progress.md 仅 1 条 350M 网站线索且不含 URL)。② SLURM COMPLETED ≠ 训练完成:时限分段链上每段都 COMPLETED,判断完成只看 epoch progress + 链条如何终止(本例 4563980 被 scancel)。③ Notion update-page-markdown 批量 content_updates 可能部分生效(第2条成功第1条静默失败),写后必须逐条 grep 验证,失败项用加长上下文锚点单独重试。
L100 UTC 2026-07-29T15:00:15Z: 解释'同一族模型两个拟合为何指数完全不同'的有效路径:先把两拟合的全部口径差列成表(y 变量/语料/尺寸网格/优化器/checkpoint 协议/模型集合),给'多变量同变⇒不可归因'的结论,再把 α/β 差翻译成 allocation 指数 a=β/(α+γβ) 的差(0.85 vs ≤0.294=参数优先与数据优先的翻转),比直接比较 α/β 数值更能传达 totally different 的实际含义;另:转述他人口头数字必须逐个对账(290.3 vs 293.283,stock-weighted vs 等权),口头回忆自带 1-2% 级误差,paste-ready 文本只认 CSV。
L098 UTC 2026-07-29T15:58:06Z: pilot/占位模式的最小改造实现=薄壳循环内直接 bash 原 sbatch 脚本（继承 allocation 的 SLURM env，srun 自动用满本 allocation）——切忌复刻 batch 逻辑。三个必须解决的冲突源：W&B run 名（同 SLURM_JOB_ID）、checkpoint 路径（本例因含 wandb run.id 天然免疫）、墙钟守卫（MAX_JOB_HOURS 按 python 进程起点计时，pilot 内每实验须动态传 min(实验预算+0.25, pilot剩余-0.15) 且启动前校验剩余>预算×0.95+0.2）。同形状打包原则（pilot 节点数=实验节点数）可消掉 allocation 切分/JAX coordinator override 的全部复杂度。
L099 UTC 2026-07-29T16:50:00Z: JAX 主进程 + torch DataLoader fork workers 死锁实锤：泄漏实验 78M eval 挂在首 batch 前 13min（GPU 显存已上卡 83GB×4 但利用率 0%、全节点 CPU idle 99.9%、日志停滞）——os.fork 警告在 eval 场景成真（训练场景侥幸存活是时序运气）。判别三联征：显存已分配+GPU 0%+CPU idle = fork 死锁；若 CPU 单线程 100% = XLA 编译中。修复 = num_workers=0 同步加载（squashfuse 本地读足够快）。另：srun --overlap step 的存活绑定 login 端 srun 客户端进程，客户端死则 step 被 SLURM 清理（CANCELLED）。
L099 UTC 2026-07-29T16:49:03Z: 断线重连后的监控恢复要"合并重装"而非逐个复原：孤儿 Monitor 的内部状态（offset/last_state）不可恢复，重装时把多个监控合并为一个统一 job 集监控可降低再孤儿化面积；重装前先 squeue+sacct 全量盘点补上事件空窗（本次空窗 ~2.5h 无状态变化，零损失）。
L100 UTC 2026-07-29T16:54:32Z: 回答"某状态码什么意思"类问题的完整路径=站点官方文档故障表+上游软件手册+源码枚举语义三层：站点表定义"哪些值需要行动"，上游手册定义"标准值语义"，源码解释"表外值从何而来"（None=WAIT_NO_REASON 零值即为例）。站点故障表**不收录**某值本身就是"该值无需处理"的证据。
L100 UTC 2026-07-29T17:25:00Z: 大 BSZ eval 的第二个坑：BSZ 32×13000 tok 的 in_proj GEMM（f32[416000,1024]×[1024,4480]）令 XLA Triton autotuner "No valid config found" 直接 crash（r3）。修复 = XLA_FLAGS=--xla_gpu_enable_triton_gemm=false 回落 cuBLAS（eval 无需 Triton 微调，大 GEMM cuBLAS 无压力）。教训：训练验证过的 XLA 路径只覆盖训练 shape，放大 batch 进入未验证 shape 区域时 Triton autotune 是新故障面。
L101 UTC 2026-07-29T17:31:23Z: session 搜索 key 选择的新细则:凡已写进共享记录文件(findings.md 等)的数字/结论(如 293,283,039、0.85→0.294)会被后续所有读过该文件的会话 JSONL 收录,选择性随时间衰减;引语类 key(他人口头英文原话)不会落账,选择性持久。优先引语>数字。
L101 UTC 2026-07-29T18:30:00Z: 两个新故障模式：①L2 校验对子集 shard 传全量 488 ticker 列表触发 strict 断言（30720 档中低活跃 ticker 零样本）——子集数据包的 dataloader 校验必须用 shard 内实际 ticker（从 index.json keys 提取）；②eval BSZ 32×13000tok 峰值超 0.85×96GB 预分配（6.94GiB 分配失败 OOM）——降 BSZ 16/4 + fraction 0.92。宝贵恢复经验：set -e 失败退出时 trap 只卸挂载不删 WORK 文件，TMPDIR 在宿主 job 存活期间保留 → 已打包 shard 与 provenance 可直接续用（resume_materialize.sh 跳过提取打包只补校验落盘）。
L102 UTC 2026-07-29T18:50:00Z: 故障三连的第三个：fork Pool worker 继承父进程惰性打开的 NpzFile（zip 容器）→ 32 进程并发 seek 同一文件描述符互踩偏移 → zip 成员解压 zlib.error "invalid literal/length code"。规则：把 np.load(npz) 交给多进程前必须先 {k: np.asarray(f[k])} 全量物化并 close()；同理任何惰性文件句柄（mmap、zipfile、h5py）都不可跨 fork 共享读。
L1785346820 UTC 2026-07-29T17:40:20Z: 已有 allocation 应优先用 srun --jobid attach，避免重复排队；但 --overlap 只绕过 Slurm 的逻辑资源占用，不解决物理显存冲突，GPU util=0% 也不等于空闲。正确 handoff 是先 hold queued root、建立带零 PID/低显存门的 attached runner、确认 step live，再取消整条旧链并用 squeue+sacct 双重核验。任何被 source 的 cleanup 也必须在执行前静态审计删除命令。
L103 UTC 2026-07-29T19:20:00Z: eval BSZ 放大的第三堵墙：repeat_book 把 book [B,500,503] 沿 token 维展开成 [B,13000,503] f32 全物化（3.17GiB@B=64），是 13k-token 模型 eval 显存的真正大头——BSZ 16 也撑不住 0.92 预分配。结论：该管线 eval BSZ 有效上限≈训练值（8/2），加速空间在数据侧（spawn workers）而非 batch 维；三连 OOM/autotune 教训 = 放大 batch 前先算清最大中间张量（logits 之外还有广播/repeat 类）。
L1785348701 UTC 2026-07-29T18:11:41Z: eval BSZ 不能因 forward-only 就放大：repeat_book 将 [B,500,503] 展开为 [B,13000,503]，训练 BSZ 已顶显存；200M/350M 训练 bsz=1 仅提到 2（r4/r5 实测上界内）。选 eval bsz 使 bsz×num_devices 整除 30720（16/8/4/2×3 全整除）→ drop_last 零丢样。
L104 UTC 2026-07-29T20:10:00Z: r6/r7 连环 Killed 的完整因果链：①GH200 节点 $TMPDIR=/run/user/<uid> 是 tmpfs 内存盘（172G，写入直接计入作业内存 cgroup）——"写 TMPDIR 再 rsync"在内存吃紧的并行场景是双刃剑；②被 SIGKILL/TaskStop 的 srun 不执行 EXIT trap → squashfuse 挂载泄漏（累积至 146 个死挂载）；③物化 48 进程解压高峰 + 校验 32 进程 + 另一实验初始化叠加会顶穿 460G 配额，内核挑单个 step SIGKILL（表现为 slurmstepd STEP CANCELLED + task Killed，exit 137）。规程：同一预留 job 内的重内存作业必须串行；重启前探针清点 fuse 挂载与 tmpfs 占用并 fusermount 清场。
L101 UTC 2026-07-29T18:15:41Z: (None)→(Priority) 翻转实测发生在提交后 ~4.2h（本集群负载下评估窗口推进速度的一个数据点）；翻转≠临近启动，仅=调度器开始为其排位。
L1785349000 UTC 2026-07-29T18:16:40Z: inference CSV 契约不能用一个 rows 值覆盖三类输出：conditioning message=N、conditioning orderbook=N+1（初始盘口快照），generated/real message+book=M；否则成功生成会在尾部被误判。dataset length 也依赖 window=N+M：短 smoke 48-message=2354297，正式 500-message=226002，不能互换。可重试 attached runner 应用 flock 生命周期锁，不用失败后永久残留的 mkdir 锁，也不删除旧锁。
L1785349600 UTC 2026-07-29T18:26:40Z: `srun --overlap` 的物理门禁只在 stage exec 前成立，不提供运行期排他权；后起 `.53` 能在 `.47` 已占约88.2GiB/卡时再次获得GPU可见性，故长期 attached GPU job 还需持续 lease/监控或 allocation-level禁止新GPU step。tmux 是 node-local，文档必须记录 host；`ATTACHED_RESUME_FROM=generation` 仅跳过 smoke，不是 partial formal output 的 batch-level resume。
L105 UTC 2026-07-29T20:40:00Z: r6-r8 三连 Killed 的真凶定案：同一预留 allocation 上另一条工作线（model-zoo LOBbench 推理农场，4 rank × 88GB 显存 + host 内存）与泄漏实验共享 460G cgroup，新 step 的初始化增量顶穿配额被内核击杀；GH200 上 GPU HBM 以 NUMA 节点形式计入 free 的系统内存（4×88G 解释了 used 450G 与 cgroup 61G 的差额）。这正是 CLAUDE.md 当日新增 "Physical GPU gate before overlap" 的活案例——--overlap 前必须 nvidia-smi 查 compute PIDs 与显存，占用中绝不塞入、绝不杀占用者。处置：泄漏实验转独立 sbatch（5826356）+ 农场退场 gate 双路竞争。诊断链模板：free（系统级）→ cgroup memory.current（作业级）→ nvidia-smi compute-apps（谁占卡）→ /proc/<pid>/cmdline（占用者身份）。
L1785350968 UTC 2026-07-29T18:49:28Z: spawn DataLoader worker 会 re-import 主模块并重新执行全部顶层代码（argparse→provenance→jax backend init→dataset 重建→甚至进入 ckpt 循环，且 worker 里再 spawn 12 worker=指数灾难）。任何用 multiprocessing_context='spawn' 的脚本必须把执行流放进 if __name__=='__main__'。leakage_test.py 系列（r3-r6）从未暴露此坑：全部死于更早阶段（autotune/OOM/取消），从未活到首次 loader 迭代——同一隐患仍在其脚本中。valset_ce_eval.py 已重写为函数化+main guard；旧 step 5790795.73 已 scancel。
L1785351300 UTC 2026-07-29T18:55:00Z: token syntax mask 必须禁止所有结构上非法 special token，不能因某架构历史上“通常不生成 START”就保留 START 概率；full-distribution sampling 会把任何非零概率变成稀有生产事故。sentinel（NA_VAL=-9999）绝不能无校验进入时间算术。inference 完成门不能只数文件/行数，还必须校验时间单调、time_ns 范围、special-token 与 finite 值。scorer 的 nonfinite strict gate 正确暴露上游 contract violation；dropna/clamp 会掩盖而非修复。
L106 UTC 2026-07-29T21:10:00Z: r6-r9 连环死的最终真凶修正：leakage_test.py 是无 __main__ 保护的平铺脚本，spawn DataLoader worker 按语义重新 import 主模块 → 12 个 worker 各自完整重跑全脚本（重建 323M 数据集 + 各开 JAX GPU 上下文）→ 内存爆炸被内核击杀。铁证指纹：日志中 "[jax] devices: 4" 在 JIT 完成后二次出现（worker 重跑到脚本开头的打印）。农场共存（L105）只是放大器。规则：任何会被 spawn worker 触碰的入口脚本必须有 if __name__ == "__main__" 保护；平铺实验脚本一律禁用 spawn workers（用 num_workers=0 或重构成 main()）。r10 改同步加载。
L1785353260 UTC 2026-07-29T19:27:40Z: stochastic replay 的“同一个 dataset index”不等于“同一次采样”；必须保留 rank seed、完整 batch width、slot 与此前每次 PRNG split。调试开关若改变 top-n 或输入也不是观察工具：本例 overfit_debug 会改 sample_top_n=1 并喂 real book，故只能新增 host-side token trace。父 allocation RUNNING 也不代表 GPU 可用，物理占用时准确状态必须写 waiting。
L107 UTC 2026-07-29T21:40:00Z: 泄漏实验 r1-r11 十一版连败系统性复盘（用户点名深刻反思）：四个方法论根因——①违反自己记忆中已有的冒烟铁律（新管线无 --smoke 直接全量，8/11 轮是这一个错的利息）；②执行层自作主张而非抄生产（照抄 eval_test_ce 的部分零故障，自创的 workers/BSZ/聚合全炸）；③诊断先怪环境不疑自己、把"作为修复引入的 spawn"排除出嫌疑清单，[jax] devices 二次打印的铁证在日志里躺四轮（grep 关键词采样代替通读）；④排障期优化对象错置（优化成功吞吐而非失败周期，15min×11 轮纯等待）。已固化为长期记忆 new-pipeline-three-rules。
L1785353851 UTC 2026-07-29T19:37:31Z: 等待态要同时验证 artifact state 与实际 supervisor PID/cmdline；state 时间戳只代表最后一次迁移。sidecar 验证必须 scope 对齐（全序列对全序列、目标行另断言）；结果文件存在不等于成功，重启必须解析 terminal boolean。纯数组/RNG 的 post-GPU analyzer 应固定 CPU，避免无谓重新占 HBM。
L1785353993 UTC 2026-07-29T19:39:53Z: 四卡 gate 的 ETA 必须跟随“最后一张所需 GPU”的释放者；单卡作业较早结束不代表 gate 前进。进度条要区分当前 checkpoint 百分比、全 work-queue batch-work 百分比、已落盘结果数，避免把 48.8% 首项误读成全队列完成一半。
L1785354606 UTC 2026-07-29T19:50:06Z: 多阶段任务在 init 阶段没有总 ETA 信息；.82 的100/960只是78M-mid子阶段10.4%，映射到完整两ckpt三组 workload 是约0.7%。任何进度汇报必须同时给“子阶段分母”和“全任务分母”；发现旧 ETA 建基于缺失分母时应主动撤回。
L1785356813 UTC 2026-07-29T20:26:53Z: 被取消的并行工作队列只能按durable per-item JSON计完成度；已跑到一半但未落盘的第二波batch不能累计。GPU util瞬时为0也不等于可用：.82仍以同一compute PID在四卡持有83,356/17,672MiB，E2 gate正确保持关闭。
L102 UTC 2026-07-29T19:46:54Z: sbatch 有三处路径锚定在提交时 cwd（SLURM_SUBMIT_DIR）：#SBATCH --output 相对路径、脚本内 ${SLURM_SUBMIT_DIR}/xxx 引用、WORKDIR 默认值。"bash /abs/path/script.sh" 只保证脚本自身可寻址，不保证这三处正确——从任何目录调用 sweep/batch 前必须 cd 到实验目录。快速验尸法：sacct -X --format=WorkDir 一列即可证伪/证实 cwd 类事故，比翻日志快（本例日志还落在了错误目录，tail 实验目录零发现是第二个陷阱）。exit 指纹：127=command/file not found（元凶节点）、143=SIGTERM（受害节点连坐）。
L103 UTC 2026-07-29T19:51:08Z: 无新教训（基线轮）。
L104 UTC 2026-07-29T19:55:30Z: notion MCP update-a-block 两个坑复证：(1) 块内容须作顶层额外属性（callout: {...}）而非包进 type 参数（否则 body.type 校验 400）；(2) PATCH 成功时返回体可能仍显示旧 rich_text 快照，必须 retrieve-a-block 复核而非凭返回体断言失败。
L105 UTC 2026-07-29T19:59:30Z: 无新教训。
L101 UTC 2026-07-29T20:00:58Z: 逐句翻译解释类任务的有效交付结构=译文之外每句给三层解:术语定义(LORO/bracketing/macro-average)、数字出处对账(与页面审计表/findings 交叉)、写法意图(为何这么写/防哪类质疑)——纯翻译对已能读英文的用户零增量,价值全在后两层。另:update-page-markdown 对大页面的 update_content 返回体=更新后整页(212KB 落文件),本身即验证材料,无需再拉一次页面;post-page 建空页+replace_content 填内容的两步法比 children 直接传 markdown 稳(children 的 string 项不保证按 markdown 解析)。
L106 UTC 2026-07-29T20:07:02Z: 无新教训。
L107 UTC 2026-07-29T20:09:01Z: 表述教训：说「macro 用于回复审稿人」易被读成「审稿人要求 macro」；引用审稿人要求时必须区分 required（审稿人原话）vs chosen（作者实现自由度）。
L1785356037 UTC 2026-07-29T20:13:57Z: 用户原则（原话）：『回答问题永远排在做事情 做任务前面』。答案必须独立成篇置顶，不与任务动作/状态流混排；被事件流淹没的答案视为未回答，须完整重答。已写入 memory feedback_answer_first_then_autoresume.md。
L108 UTC 2026-07-29T20:18:23Z: Claude 529 Overloaded 出现在写操作成功之后时，不能凭终端缺少最终答复推断交付失败；应重新 fetch 精确 Notion 主页面与子页，并分别验证标题、链接和关键句。大工作树并行追加时，用索引级精确暂存仅提交本任务四行，避免把无关记录批进同一提交。
L109 UTC 2026-07-29T20:24:14Z: champion 型 reviewer 回复不应以限制和撤回开场；更有效结构是具体感谢其肯定→逐项说明 requested check 已执行→显式标出 beyond-request 分析→把不利结果表述为 reviewer 帮助论文变得更透明→结尾重申其认可的方法贡献被保留。无法完成 matched comparison 的 Q2 也应写成“完成证据审计并拒绝无效替代”，而非只写“No comparison available”。
L108 UTC 2026-07-29T20:29:37Z: (1) 预注册 bounds 的边界撞击本身是证据：micro β 76.6% 撞 0.1 下界=「β 在该口径下未识别/近零」，报告必须带 fraction_at_bound 而非只报 median。(2) s5e miniforge3 对 u6gb 账号 Permission denied，改用 ~kangli.u6gb/miniforge3（PATH 默认 python3 即是）。(3) 变体分析先做逐行复现断言（1e-16）再改单一因素，差异才可归因。
L108 UTC 2026-07-29T22:45:00Z: 预留 allocation 的正确心智模型：它是用户"多会话共用的即时评测池"而非"空闲资源"——本会话长 GPU 任务放上去必然与其它会话的评测互相击杀（cgroup 共享，后来者/大户胜）。长任务（>30min GPU）一律独立 sbatch；预留节点只放短平快 attached one-shot。另：实验设计冗余的价值实证——78M 死于 VAL 组前，但 MID/SEEN 两组均值已经落在日志里（每组完成即打印 meanCE），中途死亡仍然抢救出核心科学数据。
L109 UTC 2026-07-29T20:43:46Z: 「in color blue」在无 LaTeX textcolor 惯例的 repo 语境下=该 Notion 页的蓝色 callout 惯例（先例：Codex 回答蓝 callout、本次回答蓝 callout）；落笔前 grep 惯例避免用错载体。
L110 UTC 2026-07-29T20:50:30Z: 用户「放到这里用蓝字+之前的用删除线」=学术 track-changes 惯例（修订融入正文行文），非「附加带色块」；先读全页定位所有报告同一数字的段落（β 在三份 response 各出现一次），一次改齐避免版本内自相矛盾。
L111 UTC 2026-07-29T20:59:52Z: Notion track-changes 的可程序验收方法：先排除用户要求保留的解释 callout，再对旧短语逐项检查所有命中均位于成对 `~~` 内；对新短语检查每个命中均落在 `<span color="blue">` 内；最后人工读“去删除线投影”和“蓝字投影”。同一危险口径分散在 pXiP/WHZQ/8P5h 时，必须同一轮精确替换，不能只修目标段。
L112 UTC 2026-07-30T11:25:09Z: session 搜索 key 优先选带序号的输出文件名（如 fig1_terminal_ce_vs_N）而非变量名：变量名随代码复用扩散到多 session，图名几乎只出现在创建它的对话里。tail -c 400 取记录文件末尾 ID 会因长条目截不到 ID 行，应全文件 grep -oE '^ID[0-9]+' | tail -1。
L1785411704 UTC 2026-07-30T11:41:44Z: Notion API quirk 补充：update-a-block 更新 callout 的 rich_text 划线时，annotations 必须传完整对象（bold/italic/strikethrough/underline/code/color 六键齐全），只传 strikethrough 单键会被静默忽略；paragraph 则单键即可。内容参数须为顶层属性（paragraph=/callout=），包在 type= 里会 400。
L113 UTC 2026-07-30T13:00:59Z: 单一高选择性 key（SLURM job ID）+ 排除当前 + 取最大文件的管道一次命中，skill 预算纪律有效。
L114 UTC 2026-07-30T13:01:53Z: 记录文件已被历史损坏行污染（F1785411096/F1785413012、PG1785411459/PG1785411704/PG1785413012、L1785411704 把 epoch 秒当编号写入，按 append-only 纪律保留原文不改）。此后解析下一编号必须用位数上限模式 grep -oE '^F[0-9]{1,4} '（含尾随空格）取 tail -1，裸 '^F[0-9]+' 会命中 epoch 行导致编号继承污染。本轮 F129/PG106/L113 即为此坑修正后的编号。
L115 UTC 2026-07-30T13:12:11Z: 后台 Bash 进程可能按需继续读取脚本文件，因此不得在其运行期间原地编辑同一 launcher；“启动时语法正确”不能保护后半段免受瞬态半写内容影响。另，artifact 的 `waiting` 只是一条陈旧声明；终态必须联合 state mtime、supervisor PID/cmdline、Slurm step、orchestrator log 与结果文件判断。
L116 UTC 2026-07-30T13:24:35Z: ① matplotlib mathtext 内不能出现字面 %（$C\pm15%$ 抛 ParseException），百分号要移出 $ 环境；② Notion MCP patch-block-children 的 schema 虽只列 paragraph/bulleted，children 数组接受任意 object，divider/heading/equation/code 块可直接发成功；③ plot_chinchilla 三联图的 N*/D* 虚线 slope=β/(α+β)、α/(α+β) 是 surface-implied 解析式，不依赖 IsoFLOP valley 数据——132 尾窗点也能画全套 iso-slope 图，valley 缺失只影响 empirical slope 回归。
L115 UTC 2026-07-30T13:25:09Z: 监控场景下 SLURM job ID 的跨会话选择性坍塌：所有并行监控会话都含同批 squeue 输出，job ID 不再是高选择性 key，必须改用 assistant 生成的独特叙述短语。另：ls -t head -1 定当前 session 不可靠（多会话并行活跃时排除无辜者），当前 session ID 应取系统提示 scratchpad 路径中的 UUID。
L117 UTC 2026-07-30T13:42:34Z: "复用以前的代码"必须先穷尽搜代码库再自己写——首版 IsoFLOP 图自创 ±15% 窗口法被用户打回，正统实现 isoflop_test_ce.py 一直在 scaling_law_plots 根目录（grep isoflop *.py 即中）；同名方法两个 lineage（fit_chinchilla.isoflop_cross_check 窗口法 vs isoflop_test_ce.py 插值法）并存时，选用户亲手跑过出图的那个（plots/isoflop_test_ce_parabolas.png 是样式铁证）。
L118 UTC 2026-07-30T13:43:30Z: --overlap 步内 Slurm 不追踪同 job 内 GPU 占用（步与步共享 gres），物理门必须自己查 compute PID 归属：本例 GPU0 的 78GB 是活实验而非僵尸，nvidia-smi util 0% 不等于可抢占（可能在 CPU-bound 取数阶段）。掩码用步内 export CUDA_VISIBLE_DEVICES 而非 srun --gpus（后者由 Slurm 随机挑卡，可能挑中被占的 GPU0）。
L1785419180 UTC 2026-07-30T13:46:20Z: srun/sbatch --export 陷阱：值内含逗号会被当作变量分隔符截断（EXTRA_ARGS 里 date_range 的逗号使 --provenance 丢失→worker 回落 valset 默认 ticker 集→Jan 缺 BAC assert 爆）。规则：经 --export 只传无逗号无空格的简单标量；复合参数一律在脚本内部按 MODE 开关组装。smoke 直传命令行不经 --export 所以通过——smoke 与生产的传参路径差异本身就是要测的面。
L104 UTC 2026-07-30T13:49:29Z: 三教训：①wrapper 的死因永远在 node*.log（exec 重定向后 .out/.err 只剩 srun 转述与旁路噪音）——本次被 ledger 红鲱鱼带偏一轮，直到读 node log 才见 NCCL ERROR；②bash ${VAR:-default} 防不住"提交 shell 激活态"：conda (base) 的 CONDA_PREFIX 会穿透 sbatch --export=ALL 覆盖训练环境，长期修法=提交前缀显式赋值；③时隔数月重跑"当年成功"的管线，必须重验默认路径三件套（conda/NCCL/QUANT_ROOT）——共享文件系统的挂载入口会迁移（s5e/public→public/s5e），当年日志里成功的绝对路径今天可能 Permission denied。另：多身份共用实验目录时，账本类可写文件用 per-identity 文件名（_u6gb 后缀）根除 owner 漂移互杀。
L1785419843 UTC 2026-07-30T13:57:23Z: attach 到占位 allocation 的长任务(5694130.0 数据集构建)被 kill 后彻底掉出雷达 12 天——记录里 07-18 后零条目, 没有 successor 也没有 pending 恢复计划。教训: attach step 与独立 sbatch 同等待遇, 必须进 live_jobs/active_monitors 并在中断当轮写下恢复决策, 否则'非 job 形态'的工作项会静默丢失。
L105 UTC 2026-07-30T14:00:46Z: 汇报风格校准：根因类结论需备两档表述——技术档（层级/机制链）与人话档（因果直叙），用户明确偏好后者+中英对照用于对外沟通。
L118 UTC 2026-07-30T14:03:25Z: ① 生成 N 张图必须 N 张全部亲眼目检再交付——上轮只抽查 1/6，5 张空白/单点图直接到了用户手里；② 绘图代码对"0 个有效结果"的分支要产出诊断性内容（如实画出已有的点+说明缺什么），空文件比坏结果更糟；③ 向非直觉概念汇报时用户要"人话"：术语表+ASCII 示意图+把数字放回物理含义（44.5×=长线段，1.33×=短线头）。
L1785420329 UTC 2026-07-30T14:05:29Z: 中途插话规则实践: 用户 mid-turn 补充'不覆盖要新做'直接改变执行参数(resume 旧目录→新时间戳目录全量重建), 幸而 launch 前收到——教训: 数据集类操作默认就该走'新时间戳目录'(memory 既定纪律), resume-in-place 只有用户明确要续旧目录时才用。
L106 UTC 2026-07-30T15:26:19Z: 无新教训（事件轮）。
L107 UTC 2026-07-30T15:31:05Z: 无新教训（验证轮）。
L119 UTC 2026-07-30T16:42:50Z: ① 溯源他人论文图优先 grep 论文措辞原文（"nearest logged"一击即中 rebuttal_analysis 的官方重建），比找绘图脚本本体更快更可靠；② 复现别人图时把 published 数值做成断言门槛（本次 9 切片 assert 全过才出图），杜绝"形似数不对"；③ Notion MCP update-a-block 修改 paragraph 要把 paragraph 作为顶层额外参数传（type 包装会 400），与 table_row quirk 同源。
L119 UTC 2026-07-30T16:45:28Z: 预注册判据被拒绝时先看方向再下结论——H2 原始形式被拒但方向与泄漏相反，且偏移量在 78M/350M 上一致（模型无关→组构成属性，模型相关→记忆），一条免费的鉴别轴；对照组设计若忽略池的构成倾斜（§5.1 的 55:45），H2 就会以可预测的方式失败，构成调整（分层重加权）应作为此类实验的预注册组成部分。
L120 UTC 2026-07-30T18:32:09Z: valset squashfs 内是 (500,14) int64 原始消息字段而非 token——tokenization 在 dataloader 的 encode_msgs（jax.vmap）里做；任何"与 CE 同单位"的对比实验必须先过同源 tokenizer，直接压 .npy 字节会把存储格式（int64/字段布局）混进压缩率。另 unsquashfs 提取路径不带 squashfs-root 前缀；s5e miniforge python 对 u6gb 账号无执行权限，用户环境 pip install jax（CPU）即可跑 encoding。
L1785438947 UTC 2026-07-30T19:15:47Z: 流式构建(mksquashfs -pf)在共享节点上与 5 个 GPU step 共存 5h 无冲突且节奏稳定(22-31min/shard 仅随月份体量波动)——'纯 CPU 大 IO 任务 attach 到 GPU 实验节点'是安全的算力回收模式, 前提: /dev/shm 只放元数据+显式不申请 GPU。
L1785440163 UTC 2026-07-30T19:36:03Z: 传数据前先查配额而非先开传: whoami-v2 的 isPro 字段 3 秒暴露 100GB 私有上限, 若直接 upload-large-folder 202GB 会在烧掉约 1h 带宽后被 quota 拒。教训固化: 外传数据三查=license 边界/账号配额/单文件上限, 全过再动带宽。
L108 UTC 2026-07-30T19:58:07Z: 无新教训（推进轮）。
L1785441642 UTC 2026-07-30T20:00:42Z: 用户要求'表格里体现 events + 强调全年均匀采样¼'前, 我的 README 首版只有 windows/tokens 且写错抽样单元——教训: 数据集文档的 headline 应该由 manifest 统计脚本生成而非手写, 手写数字/措辞两次都被实证打脸(整天抽样、整天 holdout)。
L109 UTC 2026-07-30T20:10:08Z: pilot/同 job 多实验模式的资源隔离检查单(第三课):凡 wrapper/batch 内以 SLURM_JOB_ID 为唯一性锚点的资源(挂载点/日志文件/tmp 目录/端口/checkpoint 路径)在 pilot 内全部退化为共享——上线前应 grep SLURM_JOB_ID 逐个审计。强杀场景 cleanup trap 不可信(SIGTERM→KILL 窗口内 fusermount 可能没跑),防御=路径唯一化(加 $$/序号)而非依赖清理。日志被后继 truncate 的教训:关键失败现场要靠 per-experiment 命名保存,否则只剩 srun 的 exit code 转述(134=SIGABRT 这次仅存于 pilot .err)。
L120 UTC 2026-07-30T20:19:36Z: 用户已有权威文档（VALSET_V1_REPORT.md）时说"上传/做页面包含这些信息"，默认动作是原文逐字上载，不是自作主张改写成英文摘要卡——文档本身就是交付物，改写等于替换用户内容（与 never-delete-user-content 同源）。验证手段：远端拉回 diff 原文件，输出 VERBATIM_MATCH 才算数。
L110 UTC 2026-07-30T20:25:45Z: 无新教训（审计轮，指纹已入 F126）。
L111 UTC 2026-07-30T20:34:04Z: 血泪最重一课：绝不原地重写正在被 bash 执行的脚本（尤其被 N 个运行中 job 共享的 wrapper）。bash 按字节偏移懒读脚本，同 inode 覆盖写=向所有运行实例投毒（本例 8 实验各埋一颗收尾雷，23M:s5 圆满训练被记 FAILED）。安全修法=写临时文件+mv 原子 rename（换目录项不换 inode，运行中 fd 指旧内容）。Edit/python open(w) 都是原地写——patch 共享运行中脚本前必须先问"现在有谁在执行它"。次生教训：完成判定的唯一真理=checkpoint 链+wandb 曲线，exit code 在收尾竞态下两个方向都会说谎（127/143 可以是圆满，COMPLETED 0:0 可以是 4 秒空跑）。
L120 UTC 2026-07-30T20:59:02Z: 复用旧评测管线于新 job 前逐段读 wrapper：本次揪出 worker 日志名硬编码旧 jobid（会把新跑日志追加进旧文件误导监控），改 ${SLURM_JOB_ID:-旧值} 向后兼容；GPU gate（<2000MiB×2 次）在独占新节点只损失 2 分钟，为共存设计的脚本可直接独占复用。
L112 UTC 2026-07-30T21:00:22Z: 分布式 final save 与连坐 kill 的竞态窗口=save 时长（模型越大窗口越宽输面越大）——同类事故后判定资产必须区分'定期 ckpt 完好'与'final save 是否落盘'两层。
L113 UTC 2026-07-30T21:50:53Z: 墙钟预算三层链条(SLURM --time > 壳 max_job_h > train 端提前 30min save)放大 sps 估计误差:表值乐观 17% 即导致截断 21%(1M 例)。生产表的 sps 必须用同代码同拓扑实测,跨月/跨 commit 沿用旧表=系统性截断。
L114 UTC 2026-07-30T21:51:25Z: 无新教训（基线轮）。
L1785448432 UTC 2026-07-30T21:53:52Z: handoff 文档的第一要务是路径可存活: 引用 scratchpad 里的脚本=引用即将消失的东西, 写 handoff 前先把资产拷到持久目录再落笔。
L1785450367 UTC 2026-07-30T22:26:07Z: [jan-shuffle] attach 模式的宿主 job walltime 是隐形截止线: attach 前应查 %L (TIME_LEFT) 并与队列 ETA 对比, 本次 5823145 仅剩 ~11h 时挂入 61+71 个 ckpt 队列, 小模型长尾没跑完就被 walltime 收割。教训: 长队列 attach 前算一遍 '剩余 walltime ≥ 队列 ETA×1.5', 不够就同时预挂后继链的 auto-resume。
L1785489857 UTC 2026-07-31T09:24:17Z: 无新教训（基线轮）。
L121 UTC 2026-07-31T09:38:45Z: ① attach 评测启动前必须实测目标 allocation 每卡显存（昨晚败因＝gate 依赖的空闲从未出现）；② 长命 driver 要 setsid 与 Claude 会话解耦，会话可断、作业不死；③ 后台化只该套在最内层命令上，别让 & 吞掉整条变量赋值链。
L1785491132 UTC 2026-07-31T09:45:32Z: placeholder chain 节点是多任务共享资源: handoff 写'等 5827830 起跑后 attach'时未考虑其他任务线也会盯上同一宿主(backfill124 先到先得)。教训: handoff 里的宿主预约不是排他锁, 凡'等 job X 起跑后挂 Y'的计划必须附带认领协议(mkdir 锁或文档声明行), 且续跑者 attach 前必须查节点实际 GPU 占用而非只看 job RUNNING 状态。
L1785491363 UTC 2026-07-31T09:49:23Z: [hf-upload-trainset] 体量断言是最好的路径消歧器:用户说'应该有5TB',字面路径只有 47GB(差 2 个数量级)→ 立即停下核对而非照单执行;lob_pipeline_squashfs(管线包) vs lob_preproc_sp500_squashfs(语料)已第二次混淆,今后凡涉这对路径先 ls 验证体量再动手。
L1785491509 UTC 2026-07-31T09:51:49Z: [jan-shuffle] 「继续之前的 attach 宿主」不是无条件正确: 宿主可能已被同链其他任务合法占用。attach 前三查: squeue 状态、nvidia-smi 实占、handoff/认领类文档的跨会话约束节。占用属实时, 独立短 job 往往优于串行排队——GPU 时长相同, wall-clock 大幅提前, 且消除 gate 假开互踩风险。
L1785491587 UTC 2026-07-31T09:53:07Z: [hf-upload-trainset] '三查再动带宽'纪律第二次兑现价值:license/配额/单文件三闸门在传输前暴露了 $180/月的持续成本,用户据此在花费任何带宽和金钱前取消。TB 级外传任务的第一步永远是算清'为什么要传+传完每月花多少',而不是写上传脚本。
L1785492266 UTC 2026-07-31T10:04:26Z: [backup-sp500-squashfs] 大数据'备份'请求先盘存量再动 I/O:workspace 里 par_mirror_squashfs.batch 的 SRC/DST 五行代码暴露 6 月已镜像过同一目录,51 文件 size 比对 3 秒确认完整 → 8TB 重拷变成一次 mv(纯 MDT rename,零 I/O 零空间);'专门备份'的语义差距用改名+BACKUP_INFO+双边 checksum 补齐,而非重复搬字节。
L121 UTC 2026-07-31T10:12:58Z: 推 markdown 到 Notion 后必须读返回的 markdown 验证——auto-link 假链接这类静默转换只有读返回值才能发现，不读就永久留在页面上。镜像「活文档」（handoff）到外部系统前，先把文档更新到推送后的状态再推，否则镜像内容在落地瞬间就描述失实。

L122 UTC 2026-07-31T11:11:05Z: subagent 的 "Done" 只代表进程正常退出，不代表任务完成 — 判定完成必须看产物本身（此处 = Notion 页面实际 block 数），与 SLURM "不在 squeue ≠ 成功、必须查 sacct exit code" 是同一条规律。批量外推任务应在结束时自带产物计数校验，而非依赖 agent 自述。
L123 UTC 2026-07-31T11:11:05Z: 对 append-only 的外部 API（Notion patch-block-children 无幂等键、无位置参数），中断后"保险起见重推一遍"会制造重复内容，且失败即停比容错续跑更重要 — 跳过失败批次继续推会把后面的内容 append 到缺口之前，除非删 block 否则不可恢复。正确姿势：先精确定位断点 → 严格串行续推 → 每批成功即落 state 文件。
L124 UTC 2026-07-31T11:11:05Z: 判定某条外部访问路径不可用前，要查环境变量指定的路径而非只查默认路径。本次 REST 推送被误判为不可用，根因是只探了两个硬编码位置，漏了 NOTION_TOKEN_PATH。
L1785497153 UTC 2026-07-31T11:25:53Z: 长任务等待期是下游脚本的免费冒烟窗口: 本轮在评测跑到 91/132 时就用现有部分结果把聚合脚本和拟合脚本各跑通一次 (--require 0 / --bootstrap 5), 等 132 齐了可直接全量跑。反面是等评测 100% 完成才开始写聚合脚本, 一旦脚本有 bug (列名不匹配/estimand 用错), 排查时间全部叠加在关键路径上。推广: 凡'跑 N 小时后要做 X'的任务, X 的代码应在等待期用部分数据验证完毕。
L1785497432 UTC 2026-07-31T11:30:32Z: 无新教训（基线轮）。
L1785497558 UTC 2026-07-31T11:32:38Z: 无新教训（基线轮）。

L125 UTC 2026-07-31T11:35:38Z: 解读有歧义时不该闷头把 420 个 block 推完再问。本次「按照这里的做」+ 一个文件路径存在"搬运文本"与"照着做事"两种读法，代价差异巨大，正确做法是先对齐再动手（结果证明搬运解读是对的，但这是运气不是方法）。
L126 UTC 2026-07-31T11:35:38Z: 断言某个外部 API "不支持某能力"前必须实探一次。本轮先说了 "Notion API 2022-06-28 不支持本地文件上传"，实际 POST /v1/file_uploads 直接 200 — 一次 curl 就能证伪的结论不该凭印象下。与 L124（判定路径不可用前要查环境变量）同源：都是用记忆代替验证。
L1785498422 UTC 2026-07-31T11:47:02Z: [backup-sp500-squashfs] 用户一句'你不是复制了一份,只是做了个链接吧'逼出了两个收获:①我说'零 I/O 零新增空间'描述今天的 mv,却让人合理误读成'根本没拷数据'——涉及数据是否真实存在的表述,必须把'拷贝何时发生'与'本次做了什么'分开讲;②为自证不是链接去读 lfs getstripe,顺带发现 17% 的备份是单 OST 存储、容错反弱于源。质疑驱动的深挖比自证清白更值钱:先假设自己可能错,查证路径自然会经过平时不看的维度。
L1785499844 UTC 2026-07-31T12:10:44Z: 监控脚本以 grep 结尾会制造 exit-1 假警报: 本轮 monitor 正常跑到 '132/132 ALL DONE' 后, 末尾 'grep -lE 错误模式 worker日志' 因无匹配返回 1, 整个后台任务被标记 failed。教训: 监控脚本末尾的诊断性 grep 必须加 '|| true' 或显式 'exit 0', 否则'没有错误'会被上报成'监控失败', 与 feedback_monitor_completion_false_alarm 是同一类陷阱的镜像(那条讲 completed 假阳性, 这条讲 failed 假阴性)。判据不变: 一律以 sacct ExitCode + 产物计数为准, 不信任务状态标签。
L1785499906 UTC 2026-07-31T12:11:46Z: 部分数据推出的趋势描述必须在全量数据到齐后复核: 91/132 时 Jan−shuffle 差值看起来全程单调递增, 全量 132 后暴露为 U 形(谷底 6M), 因缺的正是 1M/4M 两个小档。教训: 用部分数据抢跑分析是对的(省 wall-clock), 但结论落到报告前必须用全量重算一遍并逐条比对形状, 尤其当缺失部分不是随机缺失而是系统性集中在某个区间时。
L1785500423 UTC 2026-07-31T12:20:23Z: 不要跨轴照搬前一轴的措辞结论: valset 节写的'micro 与 macro 同构'在 Jan-shuffle 轴上是错的(顶界占比差 15pp vs 3pp), 若照抄会把一个真实的口径敏感性抹掉。教训: 复用前序报告的结构可以, 复用其结论句必须逐条在新数据上重验; 尤其'X 与 Y 同构/一致/无差异'这类否定式断言, 它们在新条件下最容易失效且最不容易被发现。
L1785500754 UTC 2026-07-31T12:25:54Z: 向已有内容的 Notion 页写入时默认用 patch-block-children 追加而非 update-page-markdown 重写: 该页含 S3 signed-URL 图片, 整页重写无法重新上传图片会造成不可逆丢失; 追加式写入天然幂等安全。判据: 页面含图片/附件/database view 时一律追加, 只有纯文本页才考虑整页重写。

L1785501239 UTC 2026-07-31T12:33:59Z: (1) 共用仓库先查 .git 里各文件的 owner 再动手 —— sigma-0 的 .git/HEAD 属 alexbismuth、.git/worktrees/agentic-mm 属 junming、.git/FETCH_HEAD 属 aramis，直接 checkout -b 会撞同事工作区(权限挡住了，挡得对)。正确做法是新开自己的 worktree。(2) 合并前用 git merge-tree --write-tree 预演，0 冲突再动手，比出事后回滚便宜得多。(3) "把 A 仓库的东西搬进 B"之前先做函数级 diff：本例真实增量只有 2 个函数，若不核实会误搬整个包并引入第二套竞争机制。(4) 移植方向不是单向的 —— sigma-0 的 get_sim 比 LOBelia 新，要按目标仓库 API 走。(5) 脚本里写下的安全声明必须先验证再写：--allowed-tools "" 并不限制工具，若照抄注释会在代码里留一个假的安全保证。
L1785501947 UTC 2026-07-31T12:45:47Z: [backup-sp500-squashfs] REST 推 Notion 的成本实证:72 blocks 的报告(约 11KB markdown)全程只花了 3 行常量的输出 token,正文由 push_notion.py 从磁盘 read_text 后本地转 block 直接 POST,一个字节没进上下文;对照 07-30 走 MCP 传 137KB 撞用量上限截断在 47.6%。凡是'把已成文的文件送到外部系统'的任务,正确姿势都是让脚本读盘,而不是我把内容复述成工具参数。

L1785502363 UTC 2026-07-31T12:52:43Z: (1) 远端状态会在你工作期间被别人改变 —— 本地算 43 commits 而 GitHub 说 0，第一反应必须是查 API 对账，不是重跑命令。(2) 被安全闸门拦下时先想"有没有非破坏性的等价做法"：force-push 被拦 → 推新分支名即可达成同样目的且不重写任何历史，比申请放行更好。(3) 秘密处理：用户明文贴 token 时要立刻(a)指出实际 scope 而非其自述 scope，(b)要求撤销，(c)自己绝不落盘/提交。(4) PR diff 异常膨胀先看 merge-base，不要急着开 PR 让别人审 27 个文件。
L1785508700 UTC 2026-07-31T14:38:20Z: bash 只读变量 UID/EUID/PPID 等不可用作脚本变量名: 本轮 'UID=<notion file upload id>' 静默失败(报 readonly 后 $UID 仍是系统数字 uid), 导致拼出的 URL 错误、上传返回 400 却看不出根因。教训: 脚本里给 id/token 之类取名避开 UID/GID/PWD/PATH/SHELL 等 shell 保留名, 用 FU/UPLOAD_ID 这类带前缀的名字; 另: 只读变量赋值失败时 bash 只在 stderr 提示而不中断, set -u 也拦不住, 必须看完整 stderr。
L122 UTC 2026-07-31T14:48:19Z: 差点重犯 L115——原地编辑了正在运行的 parallel_valset.sh（bash 按字节偏移续读，会让运行实例读错位）；正确姿势是 cp 出 join 副本再改，原文件字节还原。共享队列（mkdir 锁+共享 OUT_DIR）的设计红利：临时多出 16 卡时零改清单直接扩容，静态切分方案此时须推倒重来。
L121 UTC 2026-07-31T14:49:39Z: 复用弹性队列到新节点时最大坑是启动清锁——parallel_valset.sh 开头 rmdir 所有空 lock 是为'单次冷启'设计，热态加挂若照抄会删掉别节点活 worker 的 in-progress 锁致重算；正解是加挂脚本绝不碰锁，只做增量吞吐，孤儿回收交原 worker。跨节点日志名必带 hostname，否则 4 节点同 GPU 索引写同一 Lustre 文件互相覆盖。
L1785509860 UTC 2026-07-31T14:57:40Z: [jan-shuffle] 用户报'节点是空的'不可直接采信: squeue 里的 chain job 是 placeholder(本体 sleep), 看不出实际负载。实测 5836919 四节点+5827830 共 20 卡全部 78-97GB 被 valset_ce_eval.py 占用, util 瞬时 0% 是 eval 的 I/O 阶段特征而非空闲。正确做法: 逐节点 -w 查 compute-apps PID+cmdline; 后续分析是 CPU-only 故可 --overlap 挂纯 CPU step 零冲突。
L1785510250 UTC 2026-07-31T15:04:10Z: [多会话协作] 同一任务链的并行会话可能已完成下游, 动手前应先读交付物/报告的对应章节(本次 §第三把尺子 已存在且更深)。重复劳动未必浪费: 独立实现 + 独立跑出一致数字构成交叉验证, 价值高于单次实现; 但必须主动对齐口径并在文档写明差异来源(terminal vs 全点均值), 否则留下两套打架的数字。
L122 UTC 2026-07-31T15:41:12Z: 弹性队列热态加挂后要验收'孤儿 vs in-progress'：worker 退出后剩余 lock-without-json 若对应节点 0 计算即孤儿，须清后重跑，否则永久卡在 done<TOTAL。判据=该 label 无 json + 持锁节点无 compute-app + 无其他活 worker；三者齐才敢 rmdir。单次冷启脚本靠启动清锁天然规避，热态加挂脚本（绝不清锁）必须显式补一步收尾对账。

L1785512923 UTC 2026-07-31T15:48:43Z: (1) 【本轮最大失误】为省排队时间取消了已排好队的作业 5850767 改走 attach，结果 attach 的 allocation 几分钟后被别人的 valset 评测填满，且从 srun step 启动的 tmux server 会随该 step 一起被杀 —— 长进程根本撑不住。原来那条排队的路是对的，被我优化坏了。规则：不要为了消除排队时间而放弃 Slurm 的调度保证，除非能证明 attach 侧的资源有持久租约。(2) 启动前的 GPU 闸门不是持久租约 —— 14:24 查时 16 张 GPU 全空，14:46 已被别人占满。(3) 监控只找 DONE/错误字符串是不够的：被杀死的进程两者都不产生，只是停止写日志。必须加"日志时间戳停滞检测"。(4) squeue 只显示分配归属，不显示分配内部有没有 job step 在烧 GPU；判断节点是否真空闲必须 attach 进去查 compute PID + 显存。(5) 共用仓库的分支命名空间也按 owner 分权：refs/heads/fix/ 属 aramis 且非组可写，feat/ 属我 —— 建分支要挑自己有权的前缀。(6) git checkout -b 在锁 ref 失败前已经改了 index，会留下半完成状态，必须 reset --hard 重来。
L123 UTC 2026-07-31T16:25:11Z: IsoFLOP 谷底可靠性除「正曲率+bracketed+left/right≥2」外还须查谷深：顶点附近无数据时抛物线会外推出低于不可约项 E 的假谷（本次 L*=0.479 vs E=0.596），此类谷底进 slope 回归会把结果拉偏；判据写成 L*≥该片最小实测 L−tol 即可，且 tol 要作为 CLI 参数以便敏感性并报。
L115 UTC 2026-07-31T18:55:08Z: 同一个坑换马甲会再踩：L104 已记"conda 激活态穿透"，但我在新脚本里写 ${CONDA_PREFIX:-默认} 又中招——防御必须写进脚本本体（强制赋值 + unset CONDA_*/PYTHON* + 从 PATH 剔除 /home 下 miniforge3），不能只写进记忆。第二课：诊断/打印代码必须 try/except 包裹——train.py:181 一行"验证 Adam 动量已恢复"的 print 让整个 restore 路径在 Muon 下不可用，而它对训练正确性零贡献。
L123 UTC 2026-07-31T18:57:10Z: IsoFLOP 抛物线的 bracketed 判据（左右各有点）不足以保证顶点可信——本例 left=3/right=12 通过判据，但顶点 L* 仍低于全部实测点。补充判据应为'顶点 L* 不得低于该切片最小实测 L'（顶点穿底=拟合外推而非测量），以及曲率 a 与相邻切片同量级（本例 10 倍跳变即警报）。臂不平衡时高曲率右臂会把顶点拖穿底。
L116 UTC 2026-07-31T18:59:31Z: Monitor 的退出条件不能依赖 ps/pgrep（沙箱不可见→立即误判完成）；应以被监控方自己写入的终止标记（如 "[attach] finished"）或 SLURM 状态为准。另：attach runner 复用 TAG+IDX 命名时，同 TAG 的重跑会覆盖上次日志（本轮 topa1 被第二次试点覆盖），排障前要先确认看的是哪一次的内容——TAG 应带时间戳或递增序号。
L124 UTC 2026-07-31T18:59:38Z: 遇到 "local changes would be overwritten by merge" 时，不能停在"stash 一下就好"——必须先 git diff <remote> -- <冲突文件> 判断本地改动相对远端是**冗余**（已被别人提交）还是**独有**（会丢失）。本例是冗余，checkout 丢弃零风险；若换成独有则 stash+pull+pop 才对，两者处置完全相反。第二课：只看 git diff（vs HEAD）会漏信息——它只显示 W&B 那 5 行，而 git diff origin/main 才暴露出远端另外改了 legacy_workdir/SSM_TYPE/PYTHONPATH。多 agent 并行写同一 checkout 时（此处 Codex 已把本地改动提交进 main），"本地未提交改动"的来源必须回溯 commit message，不能假定它只属于当前会话。
L125 UTC 2026-07-31T19:05:14Z: git log --graph 同时传多个 tip 且它们之间没有 merge commit 时，输出会把所有 commit 按时间排成单列 "*"，视觉上极像线性祖先链——本轮据此一度误判 7df5d86 是该分支的 base，实际 base 是更早的 cb8c281。判定分支基点只能用 git merge-base / --is-ancestor / <first-commit>^，绝不能靠读 graph 形状。第二课：判断一个分支"是否已合入"要用 git cherry -v <upstream> <branch>（按 patch-id 比对，能识破 rebase/cherry-pick 后的等价提交），仅看 merge-base 或 --is-ancestor 会漏判。
L1785524767 UTC 2026-07-31T19:06:07Z: 探查性统计必须在同一张表内算, 跨表按位置 assign 会静默错配: 本轮先用 'm[[size,seed,step]].assign(C=其他CSV.total_flops_to_step.values)' 估出 C 跨度均值 39.7x 并据此向用户断言, 实际两表行序不同, 正确值是中位 1.6x(双峰: 大模型档 44-216x, 小模型档 1.2-1.6x)。教训: 任何 .values/.assign 式的按位置赋值都必须先确认两侧行序一致(或改用 merge on key); 若该数字已对外陈述, 发现后要显式更正而非默默改口。
L117 UTC 2026-07-31T19:06:12Z: "rc=0 但零训练"是最危险的失败形态——比崩溃更难发现，因为一切退出码/日志都正常。防御=完成判定永远看产出物（新 checkpoint 目录里 step 是否推进），绝不看 exit code；我的 attach runner 正是靠"NEWSTEP>=curtail"判定才当场识破。第二课：resume 类参数改动必须先验算三段推导（steps_per_epoch→start_epoch→循环范围），因为 curtail 在这套代码里同时扮演"训练步数上限"和"每 epoch 步数"两个角色。第三课：TaskStop 只解除 Claude Code 的追踪，不杀 login-node 上的 bash 子进程；被停的 runner 持有旧脚本 inode 会按旧逻辑跑完队列——要真停必须 scancel 其 SLURM step 并等其自然退出。
L118 UTC 2026-07-31T19:09:56Z: 无新教训。

L127 UTC 2026-07-31T19:12:33Z: 数值算法改造的测试要按"相"设计极端配置来孤立各部分，而不是只测一个"正常"配置。本次 chunk_size=1 与 nc=1 两档互补地孤立了 Phase 4/5/6 与 Phase 3；正是 CS=1 那档把计划中错误的逆序构造顶了出来。若只测 L=32/CS=8，误差会被两相混合掩盖成"某处不对"而无法定位。
L128 UTC 2026-07-31T19:12:33Z: 写进代码注释的定量断言也要验。本轮注释里"朴素逆序偏差 exp(ADT_s-ADT_l)"方向写反了，跑一次 12 行数值检查即证伪并修正为 exp(ADT_l-ADT_s)。注释是会被后人当事实引用的，与代码同等要求。
L129 UTC 2026-07-31T19:12:33Z: 给新参数命名时要先看优化器的分组机制。本仓库 map_nested_fn 按叶名路由且对所有名为 kernel 的叶统一套用 Muon 的 weight_dimension_numbers，故 DFM 残差参数刻意命名 dfm_residual_proj 而非 kernel —— 既避免 Newton-Schulz 误伤，又能拿到论文要求的独立学习率 alpha_new。
L124 UTC 2026-07-31T19:18:21Z: 顶点穿底判据必须按测量噪声定标，不能用 L*<L_min 硬判——真实最优落在采样 N 格点之间，顶点略低于最低实测点是正常几何后果（本例健康切片穿底 0.0001-0.0002 nats）。首版硬判把 6 个切片里 5 个标成异常，几乎无信息量；改 tol=0.02（10x CE 的 95%CI 半宽 0.002）后精确分离出唯一病理（0.1064=53x 噪声）。判据阈值要有物理标尺，否则不是判据是噪声放大器。
L119 UTC 2026-07-31T19:21:42Z: 无新教训。
L120 UTC 2026-07-31T19:33:06Z: 排障时先找'同条件下正在成功的对照'再下结论——本例并发冲突假说被'同 allocation 并发的另一实验正常训练'一击证伪，比读日志推理快且硬。瞬态 OFI/NCCL 故障在本集群是已知偶发（5-10 亦有），处置=重试而非改架构。
L121 UTC 2026-07-31T19:43:14Z: 无新教训。
L122 UTC 2026-07-31T19:54:52Z: 无新教训（观察轮）。
L123 UTC 2026-07-31T20:01:33Z: 无新教训（执行轮）。
L124 UTC 2026-07-31T20:44:15Z: 假说验证要设计能证伪的观测窗口，且证伪后立即撤销为其付出的代价——本轮为验并发假说停掉一半算力，证伪后第一时间恢复双通道，避免'错误假说的沉没成本'继续吃资源。
L125 UTC 2026-07-31T21:00:05Z: 无新教训。
L126 UTC 2026-07-31T21:17:15Z: 借用他人 allocation 跑跨节点训练时，失败率随通信量上升——小实验可借，大实验应走独占提交。这条经验可推广：attach 适合轻通信/短时任务，重通信任务的可靠性需要完整的 job 级网络资源。
L127 UTC 2026-07-31T21:27:35Z: 无新教训。
L128 UTC 2026-07-31T21:40:36Z: 连续证伪三个假说（并发争抢→串行也败、attach 特有→sbatch 也败、坏节点→新节点也败）后应转向'现象共性'而非继续猜机理：本例共性=补尾 run 在 restore 后数百步崩、原始 run 无此问题。当机理不明且重试成本可控时，正确做法是并行推进兜底方案（次优 checkpoint 的方法学影响量化）而非死磕根因。
L129 UTC 2026-07-31T21:48:42Z: 失败时长的**方差**是最强的机理信号：随机故障的 Elapsed 应散布，而 9 个 job 集中在 13:54-15:44（±6%）说明是确定性触发点。发现这一点前我连续证伪三个假说，之后一次定位——排障时先看失败时刻的分布，再看日志内容。
L130 UTC 2026-07-31T22:01:23Z: 精确预测是机理确认的黄金标准——从 ckpt 间隔+sps 推出的崩溃时刻与三个不同尺寸的实测吻合（29.7 vs 29:26、14.5 vs 13:54-14:23），比任何日志片段都有说服力。反过来说，早期'固定 14 分钟'的观察是抽样偏差（多数失败项 ckpt 间隔相近），差点误导成'超时类'原因。

L130 UTC 2026-07-31T22:13:08Z: 用 jax.eval_shape 做参数树比对，可以在完全不读数组、不执行前向的前提下验证模型与 checkpoint 的结构一致性。本次 13,000 长度的前向从未执行，checkpoint 侧只读两个小 JSON，总 I/O 不到 1 MB，login 节点安全，却抓出了 activation 配置错误。结构性检查应该前置到这个成本量级。
L131 UTC 2026-07-31T22:13:08Z: 依赖"捕获异常再回退"来切换代码路径是脆的 —— 第三方库在边界情况抛什么异常是版本相关的。本次 orbax 遇到"模型多一个叶"可能抛 KeyError（不在 except 列表里）直接崩。既然调用方事先就知道要走哪条路，就用显式 flag + sentinel 确定性地进入，不要碰运气。
L131 UTC 2026-07-31T22:40:16Z: 修复验证的最强证据是'同一实验、同一配置、仅改一个变量'的前后对比：1M:s42 两次带 save 均崩于 14:44、去掉 save 后跑满 29:57 完成——单变量对照比任何机理论证都硬。

L132 UTC 2026-07-31T22:49:20Z: 移植论文方法时，先按原文实现再用真实数据检验其隐含前提，比一上来就"改进"更有价值 —— 这样失效点是可量化、可归因的，而不是被自己的修改掩盖。本次因此得到两个彼此独立的结论：一个是词表规模效应（有闭式上界，可用超参修），一个是 embedding 几何（有 beta 无关的不变量，不可用超参修）。
L133 UTC 2026-07-31T22:49:20Z: 判断"某个度量是否携带信息"要设随机对照。本次预训练 embedding 的耦合比 6.69 单看无意义，与同形状随机高斯的 7.13 一比才知道 AR 预训练几乎没改变度量结构；再与人造一维序数流形的 0.38 一比，才排除掉"高维必然如此"的解释。两个对照缺一不可。
L134 UTC 2026-07-31T22:49:20Z: 报告基线时要说明基线为何是那个数。"随机腐蚀同 field 占比 45.2%"看着像"随机也挺准"，实则是因为 2108 个 token 里 2000 个挤在两个大 field；不解释就会被误读为 metric 起了一半作用。
L132 UTC 2026-07-31T22:54:10Z: 账本数字必须来自扫描而非增量口算——本轮连续三次误报（24/25/26 实为 23），根因是把'补尾成功数'加到了错误的基数上。规则：每次报数前跑一次 checkpoint 扫描，成本不到 2 秒。

L133 UTC 2026-08-01T01:32:56Z: 缺陷分类必须从不变式构造性导出，不能靠经验列举。保真恒等式只有五个符号，所以只可能从五处坏掉 —— 这样得到的分类自带穷尽性证明，而"我想到的 bug 列表"永远无法回答"你怎么知道没漏"。
L134 UTC 2026-08-01T01:32:56Z: 缺陷登记册必须强制"可观测量"字段（若它发火你会在哪个数字上看见）。填不出可观测量的条目是疑虑不是问题，应在 schema 层拒绝入册。这一条把 bug 清单变成了检测器规格说明。
L135 UTC 2026-08-01T01:32:56Z: 排除性实验会被错误设定而无人察觉。"nOrders 100 vs 1050 逐位相同"曾被记为排除容量因素，但只改 nOrders 不改初始快照宽度时重放两侧都只建 20 个 init 订单，必然相同 —— 负结果为真却零信息量，且这条错误结论持续阻止了对初始化深度的怀疑。排除结论必须连同实验设定一起记录，才能被复查。
L136 UTC 2026-08-01T01:32:56Z: srun 与 shell 的 PATH 查找语义不同 —— shell 跳过无执行位的候选，Slurm task launcher 只匹配文件名直接 execve。任何 ~/.local/bin 下的 sourceable 片段都可能劫持 srun 命令行首词。识别特征：sacct 里 JobName 等于那个命令名、ExitCode=13:0、Elapsed 极短。
L133 UTC 2026-08-01T01:33:28Z: 复用他人管线时，CLI 差异比架构差异更容易致命：R1 与 O8 的 eval_test_ce.py 参数集不同（--ticker_index_json/--per_position_final_only 只在 R1 有），照抄命令行会 argparse 直接报错；而模型架构反而无需操心（O8 脚本从 checkpoint metadata 强制覆盖 model_type/n_heads/d_ff 等）。

L137 UTC 2026-08-01T01:55:27Z: 一个能被自己推翻的命题才有价值。D-I1 被写成"坏 id 越多 gen 臂越差"这种可测形式，于是两次测量就把它从"主因"收窄成"放大器"。若当初写成"初始化有问题"，没有任何东西会逼出那两次测量。缺陷登记册强制填"可观测量"的收益在此兑现。
L138 UTC 2026-08-01T01:55:27Z: 相关性要检查是否由单点驱动，因果要用时序对齐来查。r=-0.906 看起来很硬，但 8 个点里去掉 #258 就散了；而"坏 id 出现在分歧之前还是之后"这一个时序问题，直接把因果结论推翻。看相关不看时序，会把放大器当成起因。
L139 UTC 2026-08-01T01:55:27Z: 闸门过严本身是缺陷。preflight 检查 env，而修复后没有任何 launcher 把 env 交给 srun，于是第一次启动被自己的闸门拒了。在运行根本不依赖的东西上失败会训练人绕过闸门，比没有闸门更糟。分工：闸门断言本次运行真正依赖的东西，测试防止重新依赖危险的东西。
L140 UTC 2026-08-01T01:55:27Z: 不要在后台调用里再 nohup...& 后台化，父进程退出时子进程被一起清理（与 D-X4 同形态）。让 harness 直接持有长跑进程（run_in_background 的前台命令），它会在结束时回调。

L141 UTC 2026-08-01T02:14:26Z: 先修诊断再修缺陷的收益是可量化的。D-V3 拆分计数器一小时后，就抓到一个藏在改善里的回归：gen 均值涨了 0.23 而哨兵暴露面涨了 34 倍。若按"先修最像主因的那条"的直觉顺序，均值改善会被当成成功，暴露面增长无人看见 —— 因为 527 条哨兵会被整个吞进 n_unresolvable_ids。
L142 UTC 2026-08-01T02:14:26Z: 一个修复可能把 A 类缺陷转化成 B 类缺陷。精确时间戳查找（正确）把"错误解析"转成"明确未命中"，而未命中的哨兵值与合法 id 空间重叠（D-R6），于是净效果是把一种污染换成另一种。评估修复时必须同时看它把问题推到了哪里，而不只看目标指标。
L143 UTC 2026-08-01T02:14:26Z: 回归钉要做成双向绊线。哨兵数量钉成区间 350-750 而非上限：涨了说明生成语义变了，降了说明有人修了缺陷 —— 两种都需要被说出来。单向上限会让"缺陷被修好"这件事悄悄通过，从而丢失记录。
L134 UTC 2026-08-01T02:15:34Z: 名单类输入必须取自数据本体的索引而非外部清单文件——CSV/roster 是'应该有什么'，索引是'实际有什么'，两者差一个元素就足以让整批任务 assert 失败。冒烟的价值在此：15 分钟换掉 233 个评估点的全灭。
L135 UTC 2026-08-01T02:23:00Z: 用 srun 多任务起'N 个互不相干的单节点作业'时，必须清理 SLURM 注入的集群语义变量（SLURM_NNODES/NTASKS/PROCID/STEP_NUM_*）——下游框架（JAX/PyTorch 分布式）普遍以这些变量自动判定拓扑，会把并行的独立作业误组成一个集群然后死锁。这与'一 job 一任务'时的直觉相反。
L136 UTC 2026-08-01T02:25:26Z: 同一个坑在不同子系统重复出现时应升级为设计规范而非逐次修补：FUSE 挂载路径唯一化在训练侧（F125）踩过一次，评估侧又踩一次——凡'可能被强杀的进程创建的具名资源'（挂载点/锁/临时目录）都应带 PID 或时间戳。
L137 UTC 2026-08-01T02:34:54Z: 最危险的错误是'结构完美但数值错误'——487 行齐、字段全、step 对、退出码 0，唯一破绽是 seq_len 少了 1000。防御=把物理量断言写进验收（seq_len/vocab/样本数），而非只看进程状态。第二课：模块级 import-time 环境变量（TOKEN_MODE 这类）在'从 sbatch 迁移到 attach'时最易丢失，因为 sbatch --export=ALL 会继承提交环境而 srun attach 的环境是另起的。
L138 UTC 2026-08-01T02:41:08Z: 无新教训（验证轮）。

L135 UTC 2026-08-01T03:40:18Z: 作业秒级失败且 node 日志为空时，第一件事是看 stderr 文件而不是 stdout。本次 srun 的 "Unable to satisfy cpu bind request" 只出现在 .err 里，.out 只显示 "srun exit code: 192"。监控脚本必须同时扫两者 —— 我的第一版只扫了 stdout，白等了 30 分钟。
L136 UTC 2026-08-01T03:40:18Z: 验证配置要用消费方的实际解析路径。本次用 yaml.safe_load 验证 sigma-0 配置得到"通过"，但启动器根本不用 YAML 解析器，行尾注释会进环境变量。用启动器自带的 --execute 干跑（只打印将要执行的命令）才是可靠验证 —— 它给出的是实际会执行的字符串，且免费。
L137 UTC 2026-08-01T03:40:18Z: 同一脚本存在多份可分叉副本时，要先确认执行的是哪一份再改。本次 srun 用 $SCRIPT_DIR/node_wrapper.sh（sigma-0 副本），而我改了 openreview-v2 副本；若 gpu-bind 恰好没坏，作业会静默跑成普通 AR 训练而毫无报错。改共享启动脚本前先 grep srun 行确认路径。
L139 UTC 2026-08-01T04:19:07Z: 并行度不足时优先'把新执行器接入既有队列'而非'新建一套并行管线'——前者零重复、零改动、可随时增减；判据是执行器是否设计成无状态的自取任务式（本例 worker 只依赖 MANIFEST+OUT_DIR+锁）。
L140 UTC 2026-08-01T04:53:15Z: 无新教训（执行轮）。
L141 UTC 2026-08-01T06:24:37Z: 无新教训。
L142 UTC 2026-08-01T09:37:40Z: 增量写入的产物，其'完成'必须用内容完整性判定而非文件存在/非空——被强杀的作业留下的截断文件在所有元数据层面（存在、非空、退出码若不查）都像成功。通用规则：凡按单元追加的输出，验收条件写成'单元数 ≥ 请求数'。本例靠下游聚合脚本的自检（终点缺失告警）才发现，说明**下游自检是上游判定失效时的最后一道网**，值得刻意设计。
L1785583106 UTC 2026-08-01T11:18:26Z: 无新教训（收尾轮）。

L144 UTC 2026-08-01T11:38:40Z: 【更正】把可修的缺陷说成"信息界"会让人停止尝试。D-O1（档内排队界）曾被记为 real 臂残差的解释，推理链本身正确（cancel_mode 无影响 ⇒ 不走撤单回退；触发总是 type=4）但结论错了，直接测量指向 D-I3 深度截断。一条推理正确的间接论证仍可能得出错误结论 —— 间接证据不能代替直接测量。
L145 UTC 2026-08-01T11:38:40Z: L2 簿子必须按 价格→数量 映射比较，不能按数组下标比。删掉一档会让下面所有档上移，按下标比会报出一整片假分歧，真正的单笔变化被埋掉。第一版尸检按下标比，输出"两边各自变了 18 个 cell"，完全无法归因。
L146 UTC 2026-08-01T11:38:40Z: "没有 X 就做不到 Y" 这类阻塞判断要检查它的隐含前提。D-R5（无逐消息来源标记）曾被列为关键路径，理由是分不清 D-R1 和 D-R3 —— 但该理由只在"仅看产出物 CSV"的前提下成立，重放侧信息完整。起因最终在不改任何生成代码的前提下定下来了。
L147 UTC 2026-08-01T11:38:40Z: 工具的局限要写进文档和测试，不能靠"应该没事"。drop_deepest=1 无法完全消除 L2 截断边界效应（保留区间从 {p0..p_{L-2}} 滑到 {p1..p_{L-1}}，总多出一个边界条目）。因此单个 step 的签名是证据不是证明，真正承重的是收窄扫描与越界判据。

L148 UTC 2026-08-01T11:55:00Z: 校验必须跑在真实消费路径上，而不是语义更宽松的旁路。dfm_smoke_1gpu.yaml 的行尾注释污染环境变量，yaml.safe_load 完全放行（YAML 语义下行尾注释合法，解析器直接吃掉），真正抓到它的是 launcher 的 --execute dry-run —— 因为 launcher 根本不走 YAML 解析器，直接取冒号之后的原始文本。用错校验器等于没校验。

L138 UTC 2026-08-01T11:57:46Z: 不要直接在 main 上编辑，也不要把某仓库的工作散到仓库之外。正确做法是从当前 main 开 git worktree，在其中编辑与提交。本次在 sigma-0 main 上直接提交了三次，且把主体实现放在 openreview-v2，两者都是错的。
L139 UTC 2026-08-01T11:57:46Z: /run/user/<uid>/... 是节点本地 tmpfs，计算节点看不到 login 节点写的文件。附着 srun 执行的脚本必须放在 Lustre 上（本次因此报 exit 127 No such file or directory）。
L140 UTC 2026-08-01T11:57:46Z: 为省事绕开数据管线做的诊断，可以用来测时延、显存、数值恒等，但绝不能用来测 loss —— loss 只在分布内数据上有意义。本次用随机 token 测出 CE 劣于均匀，差点据此否定整个方法。判断"指标是否在测该测的东西"的一个快速检验：改变输入的信息量(t 从 0 到 1 保留率 0.03%->84.7%)，指标若几乎不动，说明它测的不是你以为的东西。
L125 UTC 2026-08-01T12:17:11Z: 报告拟合结果时'方法间跨度'绝不能冒充不确定度——本例方法跨度 0.06、真实 CI 宽 0.44，差 7 倍。必须做重采样，且重采样单位要选对（此处是训练链而非数据点）。另：留一法(jackknife)与 bootstrap 结论矛盾时不是噪声，是信号——留一稳而 bootstrap 宽，说明冗余存在于'同尺寸多种子'层级而非单链层级，一次丢整组才暴露脆弱性。诊断设计要匹配数据的分组结构。
L143 UTC 2026-08-01T12:53:09Z: 三条：①空字符串写进 CSV 会被 pandas 读成 NaN，groupby(dropna=True) 会静默丢光全部行——ID 类列绝不能留空；②拟合类任务的并行 bootstrap 必须在计算节点跑，login 节点 cgroup 会 SIGINT 杀 joblib worker（8 jobs 和 2 jobs 都被杀）；③预注册协议与数据现实冲突时（窗口内 D 杠杆不足致参数不可识别），正确做法是报告两者并说明偏离，而非强行套用失败的协议。

L148 UTC 2026-08-01T14:42:06Z: 初始化缺陷要修"过程"不是修"参数"。D-I3 的表象是深度不够（10 档 vs 500 档），实质是重放没有复现生成器的初始化过程 —— 生成器是宽簿重置后重放条件消息（订单逐笔真实存在），重放是把末态快照塌缩成每档一笔合成单。只把快照换宽不会解决问题。
L149 UTC 2026-08-01T14:42:06Z: 回归钉要做成双向绊线，并在数字移动时写明原因而非放宽。哨兵数量钉成区间(600,1200)，D-R6 修复把它降到 355 触发失败 —— 这正是设计意图（"降了说明有人修了它，应该说清楚"）。更新为 (250,550) 并在注释里记录 21→880→355 的完整历史。单向上限会让这次下降静默通过。
L150 UTC 2026-08-01T14:42:06Z: 会写进落盘记录的常量属于文件格式，改它会让历史记录无法被正确诊断。NEGATIVE_RETURN_ID 从 -99 改为 -999999999 后，旧数据里的 -99 会被误判成 OUT_OF_DEPTH。必须保留 LEGACY 值让诊断同时识别两者。这一条是被测试失败逼出来的，不是设计时想到的。
L151 UTC 2026-08-01T14:42:06Z: 测不出解释的时候，登记测量值并写明它证明了什么、没证明什么，并指出判决实验。D-R4 测出 60.1% 分歧，但本仓库不含 price_ref 的预处理，无法确认两字段是否可比。把"模型 60% 不一致"写下去会重复本轮已纠正过四次的同类错误。
L152 UTC 2026-08-01T14:42:06Z: 合并前必须在真正的合并结果树上验证，不能用分支树代替。本轮 merge-tree 显示 main 有 11 个分支上没有的文件，合并结果与已测分支树不同 —— 合并后在 f9ac545 上重跑确认 213 passed 才算数。
L144 UTC 2026-08-01T14:48:18Z: import-time 环境变量必须在**最外层 launcher** 设置，任何'把它加进配置/参数继承'的修法都是无效的——模块 import 发生在配置加载之前。判别法：若某设置影响模块级常量（本例 MSG_LEN），它就必须是环境变量且在 python 启动前设好。这个坑在三个不同脚本里各踩一次，因为每次都是新的执行入口。

L153 UTC 2026-08-01T15:32:12Z: 对照实验的两边必须跑字面上同一个式子，代数等价不够。D-R4 原检查用 mid+rel·tick 的绝对价格形式，改成直接比 rel 值后不仅与对照逐字相同，还顺带消掉了"两个字段是否相对同一 mid"这个疑虑维度，把剩余风险收窄成一个可被对照回答的具体问题。
L154 UTC 2026-08-01T15:32:12Z: 被推翻的诊断字段要改名而不是只改文档。ref_price_disagrees 这个名字本身就在断言一个已被推翻的结论，下一个人会照名字去读它。保留字段（删掉会连带删掉"对照跑过"的证据）但改名 price_differs_from_ref，并加测试把旧名字钉死。
L155 UTC 2026-08-01T15:32:12Z: 生成不可复现要用"首个差异的位置"来区分病因。RNG 没接上会从第 1 条消息就不同；从中间开始（第 7-118 行）再自回归放大，指向数值差异。同节点重跑是判决实验，成本 8 分钟，能一刀切开跨设备数值与 RNG 两种原因。
L156 UTC 2026-08-01T15:32:12Z: 跨节点浮点不确定性下，"节点"是配置的一部分，必须记进 manifest。这与 D-T3（清单不记引擎 θ）同形态：任何决定输出却未被记录的量，都会让产出物无法从记录复现。
L157 UTC 2026-08-01T15:32:12Z: gh pr comment 的 PR 号不要硬编码，用 gh pr view --json number 取当前分支的 PR。本轮把 26 条 comment 误发到了别人已合并的 PR #11，虽 15 分钟内全部删除恢复，但根因是硬编码了一个假设的编号。
L145 UTC 2026-08-01T19:06:01Z: scaling-law 拟合的两个轴杠杆来源不同：N 轴来自尺寸网格，**D 轴来自单个 run 内部的训练轨迹采样**。因此评测点的选取策略直接决定哪个参数可识别——只取尾部窗口会让 β 失去识别力（本例 0.10 decade → 撞界），必须对数间隔覆盖整条轨迹。这条在设计评测清单时就要定，事后补跑代价是双倍机时。

L146 UTC 2026-08-01T19:43:37Z: 一个缺陷可以被"修好"而仍然存在于另一条路径上。D-I3 在 fidelity.replay_stream 修好后，mm_sim.run_episode 仍在 cond_book[-1] 上 reset，全部既有断言照常通过 —— 因为它们看不见初始化：不挂单的策略在两个市场里都赚 0，同一个错市场跑两次也互相一致。教训：修完一个缺陷要问"还有谁做同一件事"，而不是"这条路径修好了吗"。
L147 UTC 2026-08-01T19:43:37Z: 验收条件要挑那条"缺陷存在时会失败"的。B0 三条验收里前两条（空策略 0 PnL、两次运行一致）在缺陷存在时也会通过，只有第三条（step-0 簿子逐位相同）能验出东西。写验收条件时逐条问：这条在 bug 还在的时候会不会通过？会通过的那条不是验收。
L148 UTC 2026-08-01T19:43:37Z: 结构性成立不等于测量上不同，两件事都要说。8/8 的 step-0 簿子哈希不同（结构），但只有 2/8 的 PnL 不同（观测）。只说前者会让人以为所有数字作废，只说后者会让人以为缺陷无关紧要。写结论时把两个数并排放，并明确方向不可声称（2 个非零样本，std 3021）。
L149 UTC 2026-08-01T19:43:37Z: 计划里写的"风险"要实测再动手。v2 计划把 init_time 挪走 warmup_s 起算点列为风险，实测不成立，而照它去改会真的改错延迟约束。风险条目和结论一样需要判决实验。同一条纪律的另一处应用：先声明判据再测，不要先解释再找证据。
L150 UTC 2026-08-01T19:43:37Z: 违反用户既定原则的代价是全部返工。用户早先明确说过"在本地跑然后给一个结果，github 只需要检查这个结果，不需要在 github 那一侧配置环境"，我仍然加了一个在 runner 上 pip install jax/chex 的 CI job，被要求全部删掉。既定原则要当硬约束读，不是当默认值读；改 CI 之前先回看用户说过什么。
L151 UTC 2026-08-01T19:43:37Z: 删除前先确认自己理解了目的。我把 HF dataset 删了，因为我以为它的目的是"让 CI 下载数据跑测试"（那个目的确实是错的）；实际目的是"把试卷的题目固定住"，与 CI 在哪跑无关。目的理解错会导致删掉正确的东西。所幸打包是确定性的，重建后 sha256 不变，pin 未失效 —— 这也反过来证明了确定性打包的价值不只是理论上的。
L152 UTC 2026-08-01T19:43:37Z: 确定性归档是 checksum 能当 pin 用的前提。若重建产生不同字节，sha256 只能说"和上次不同"，不能说"这就是那份样本"。需要固定的六项：member 顺序、mtime、uid、gid、mode，以及 gzip header 里自己的时间戳（最容易漏的一项，它不在任何 tar member 里）。

L153 UTC 2026-08-01T19:52:46Z: 附着共享节点时，同节点 4 张卡的剩余显存差异极大且随时间变化。占用者 rank0 通常最重（实测 card0 仅剩 1.2GB，card3 剩 5.5GB，同节点同时刻）。worker 若固定用 SLURM_LOCALID 映射物理卡，n=1 时必然落到最差的 card0。必须提供 GPU_BASE 偏移，并在发射前逐卡（而非逐节点）过物理门槛。
L154 UTC 2026-08-01T19:52:46Z: JAX 认 XLA_PYTHON_CLIENT_ALLOCATOR，不认 TF_GPU_ALLOCATOR。设了后者仍走 BFC(日志里是 GPU_0_bfc)，而 BFC 在只剩几 GB 时碎片化，会在 50MB 请求上失败并反复重试 autotuning。改 XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async + --xla_gpu_autotune_level=0 后同一张卡跑通。

L155 UTC 2026-08-01T20:17:40Z: 单 t 采样会让 DFM 训练曲线不可读。loss 随 t 跨度 2.3~11.8 nats，远大于单步学习量，逐步波动几乎全是 t 的抽样噪声（前三步 10.11→8.52→6.94 看着在学，其实只是 t 从 0.23 涨到 0.59）。分层采样 t=(k+u)/K + lax.scan 累积梯度：无偏、显存不增（1.974→2.021 GB）、t_mean 钉在 0.50。物理 batch 加到 4 需单块 6.87 GiB，共享节点装不下，分层是唯一可行路径。
L156 UTC 2026-08-01T20:17:40Z: "最大单步上升"是 running max，各格进度不同时跨格比较比的是"谁见过更多难序列"而不是稳定性。必须在共同步窗内测。第一次读表因此得出"高 LR+warmup 最稳"的反向结论。
L157 UTC 2026-08-01T20:17:40Z: 附着前的显存门槛有时效性。门槛检查时 5848061 各卡有 3.9-5.4 GB，40 秒后真正启动时掉到 2.0-3.7 GB，6/16 格 OOM。门槛与启动之间必须尽量短，且 worker 要能按格重跑（launcher 参数化 NODESPEC/GPU_BASE/START_OFF）。另：同节点 4 卡余量不均，incumbent 的 rank0 通常多占几 GB，别默认用 card 0。

L155 UTC 2026-08-01T20:32:47Z: GitHub Actions 运行页标题里的 #NN 是 workflow run 序号，不是 PR 编号。用户报"#28"时不要直接去查 PR #28（本例中并不存在），应按标题匹配到 PR #13 / 分支 b0-faithful-mm-sim-20260801。
L156 UTC 2026-08-01T20:32:47Z: 同一件事应更新既有 PR 而非新开。做法：确认目标分支已被哪个 worktree 检出（git worktree list --porcelain 按 branch 反查），确认该 worktree 干净且与 origin 0/0，再 cherry-pick 过去 push；自己临时开的重复远端分支要删掉，避免出现无 PR 的孤立分支。

L157 UTC 2026-08-01T20:56:44Z: 去掉噪音之后必须检查还剩不剩实质。evidence CI 删完 GitHub boilerplate 后只剩 "PASS 19 checks"，等于没说。噪音（别人打的、删不掉、无信息）和实质（本仓库要传达的）是两个问题，只有前者该砍。实质其实一直在数据里：每条断言的 why、每份 evidence 的 what、inputs 指纹、recorded 出处，全被报告丢掉了。
L158 UTC 2026-08-01T20:56:44Z: 指标名不是定义。invariants.conservation_gap = 0 对没读过源码的人零信息。已建 ci/metric_definitions.json 给 19 个量各写「是什么/怎么算/单位」，并把"断言了但没定义"做成硬失败（删掉一个定义验证过：run 变红并点名）。定义放仓库不放 evidence，因为它描述量的含义而非某次运行的结果。
L159 UTC 2026-08-01T20:56:44Z: 规则要写成性质不要写成数字。初始盘口必须用 full horizon，不是"用 500 档"。第一次修复只改了一条代码路径并把规则记成"500"，导致第二条路径没人查，同一个缺陷活了第二次。已写入 docs/matching_engine_invariants.md。

L160 UTC 2026-08-01T21:10:43Z: 单一标的上的"没问题"不能外推。我在 GS 上量到 500 档没用满就写进了 PR，用户要求换数据后立刻发现 17% 的票是满的。规则类结论必须跨标的抽样验证，且要把测量做成可复现的工具（tools/scan_book_horizon.py）而不是一句话，因为答案随数据变化、不能靠推理。
L161 UTC 2026-08-01T21:10:43Z: 文档不是断言。full-horizon 规则先只写进了 docs/matching_engine_invariants.md，扩到别的标的时不会有任何东西变红。已加 init.horizon_headroom_levels >= 1 作为录制期断言。注意它要在集群上重录 evidence 后才生效，当前 evidence 早于该指标，所以现在仍是 19 项而非 20 项。
L146 UTC 2026-08-01T21:25:13Z: 共享节点上跑数值库要显式限制线程数——BLAS 默认按物理核数开线程（此处 128），与已有 worker 叠加会撞进程数上限，症状是 numpy import 失败/段错误而非明确的资源错误。规则：任何 attach 到繁忙节点的 CPU 任务都应 export OPENBLAS/OMP/MKL_NUM_THREADS=1。

L158 UTC 2026-08-01T21:32:32Z: 后训练的数据必须与预训练同分布，且不能自己发明采样方案。我先用"一个 ticker 一天"跑了 16 格网格 + Stage 2B，全部不可解释——无法区分学到速度场与拟合该日。正确做法是先查用户已建好的数据集资产(tasks/validation_set 里有 valset_v1 + files_48mo.csv 索引表 + perm_for() 排列复现 + materialize_valset.py 物化管线)，四样东西合起来正好覆盖需求。教训：动手造数据管线前，先在仓库里找同类资产。
L159 UTC 2026-08-01T21:32:32Z: 均匀随机排列的任意连续切片本身就是全域均匀样本，不需要再套二次抽样(那会引入第二个 seed 和额外可复现性负担)。perm_42[a:a+n] 同时给出同分布、可复现、语义清晰("基座再往下训会看到的数据")三件事。
L160 UTC 2026-08-01T21:32:32Z: 同分布采样会让每个样本落在任意月份，若直接读原始语料则每个 worker 需挂 48 个月度分片(16 worker = 768 挂载)。用 materialize_valset.py 把选中窗口物化成 1 个与训练分片同构的 squashfs，挂载数降到 1，且 dataloader 零改动。
L161 UTC 2026-08-01T21:32:32Z: 冻结验证集的有效性是有条件的——"未来训练不越过保留边界"。任何新训练在选数据前必须先确认与它不相交，并把这条做成持续检查的单测(记录 perm_start/perm_stop/excl_line/overlap)，而不是生成时打印一次。

L162 UTC 2026-08-01T23:01:10Z: 验证要设计成"能证伪自己"。full_book 第一版只做了「快照 ∪ 揭示」，漏掉前向重放，而它自己的 certificate 还宣称缺口已覆盖——两个错误同时成立。抓住它的办法是：把一个已知完整的快照人为截断，看重建能否补回被截掉的部分。这个设计的关键在于 ground truth 来自同一份数据的未截断版本，不需要外部真值。

L155 UTC 2026-08-01T23:11:23Z: 双向 SSD 没有递归形式，所以 DFM 的修正步不能复用生产推理链的 __call_rnn__ 步进器，必须走全前向 __call_ar__。代价可接受，因为修正步数是 N（10-50）而不是序列长度 L（13000）。反过来说，任何"把 DFM 接进现有 AR 推理器"的方案都是错的起点。
L156 UTC 2026-08-01T23:11:23Z: 训练 loss 对某一类 bug 是结构性失明的——凡是训练和推理用同一约定的错误（如 ar-shift 的 off-by-one），loss 在两种读法下数值完全相同。这类 bug 只能用"构造一个已知答案的 oracle，看端到端能否还原"来测，不能靠 loss 曲线。
L157 UTC 2026-08-02T00:28:33Z: 找历史会话时，被复用的 SLURM allocation ID（如 5848061，靠 srun --overlap 挂多个任务）选择性很差，会跨会话反复出现；自建脚本文件名（rebuild_audit_worker）才是高选择性键。选键看的是它在语料里出现几次，不是它看起来多专有。另：worktree 里做的工作，会话 JSONL 仍按启动 cwd 归档，所以别按工作路径猜项目目录，用 --include='*.jsonl' 递归 grep projects 根。
L158 UTC 2026-08-02T00:34:57Z: 更正 L157 后半。会话 JSONL 不是只按启动 cwd 归档一次——会话中途 cd 到别的目录（如 worktree）会在该 cwd 派生的项目目录下再产生一份同 ID 的文件。所以 (a) 找会话必须递归整个 projects 根，(b) 命中多个同名文件不代表多个会话，先比 basename（=session ID）再比大小。另：低选择性键（nid010076 这类共享节点名）会把同期其他会话一起捞出来，而高选择性键可能只落在两份中的一份，两者都会误导'命中数=会话数'的判断。

L157 UTC 2026-08-02T00:35:01Z: push_notion.py 的行内数学处理不进入 bold run 内部——markdown 里写 **……高 $t$ 端……** 时 $t$ 会原样留在页面上。推送后必须扫一遍 rich_text，凡 plain_text 里还带 $…$ 的都要按注解重切成 (text, equation, text) 再 PATCH 回去。本轮有 3 个 block 中招。
L158 UTC 2026-08-02T00:35:01Z: 在单一数据臂上算出来的"增益/退化"符号不可信。DFM 的 clean 半区增益在 r1 和 r2 两个网格内部都出现了符号翻转，而翻转的两臂只差抽到哪些窗口。判定这类差异必须有第二个数据臂（作为误差棒）加上一个冻结留出集；只有一臂时，把符号写进结论就是在读噪声。
L159 UTC 2026-08-02T01:03:57Z: 同一类"对齐"错误在这个任务里已犯三次：按行号对齐（PR 正文已记）、按时间戳但不处理并列（本轮）、以及把 book 行号当消息计数。判据固定下来：任何"重建随会话漂移/一开始就全错"的观感，先证明比较的两个对象处在同一时刻且该时刻唯一，再谈重建对不对。
L160 UTC 2026-08-02T01:03:57Z: 误差的方向是比误差的大小更强的判据。深度截断只会单边少股（种子缺存量），重放 bug 才会双向。本轮先用 259 少/0 多 判定 10 档那批是深度问题，再用 329 个"多股"锁定存在第二个机制，最终定位到 size 截断（截断的撤单必然导致多股）。
L161 UTC 2026-08-02T01:03:57Z: 在只覆盖高价股的样本上做验证会系统性错过与股数相关的缺陷。PR 原先在 GS 与 NVDA 上验证"ordinary path is exact"，而 GS 最大单 1500 股、永远碰不到 9999 上限，所以那个验证在设计上就不可能发现 size 截断。选验证样本要覆盖被测量本身的取值范围（此处是价格→股数）。
L162 UTC 2026-08-02T01:03:57Z: os.makedirs(path, exist_ok=True) 在死 fuse 挂载点上会抛 FileExistsError，因为它的 exist_ok 判断是 isdir()，而 stat 一个 ENOTCONN 挂载点会失败。squashfuse 随 srun step 退出被杀会留下大量此类点（本轮 482/483 全挂）。修法：先 fusermount -u 再 makedirs，并让挂载点带 run tag，从根上不继承。
L163 UTC 2026-08-02T01:17:21Z: 列联表比相关系数更适合定性这类缺陷。"有截断 × 浅档失败"的 2×2 表给出的是"无截断者 0 支在前 10 档出错"，这是一个可被单个反例推翻的强陈述；若只报相关系数或失败率，就得不到这种可证伪性。
L164 UTC 2026-08-02T01:22:26Z: 证据放 PR 评论而非 commit 时，必须把"能复现所需的一切"都搬进评论：逐票原始数据 + 工具源码 + 环境变量 + 启动方式。只放汇总表等于要求 reviewer 相信汇总；只放路径则对外部 reviewer 是空的。GitHub 单条评论 65,536 字符上限是硬约束，用 <details> 折叠 + 按数据/代码拆条。

L179 UTC 2026-08-02T01:35:26Z: 工作队列的"完成判定"绝不能数目录里的文件数——results_tf26_ 目录同时躺着 manifest_tf(108)
与 manifest_tf_dense(277) 两份清单的产物，按文件数判完成会提前收工。判定必须落在"本 manifest 的 label
是否有 json"上。同理，这个 remaining() 每次要 stat 整份清单，放进 60 秒轮询的 gate 循环就是元数据风暴，
必须降频（每 10 轮一次），gate 平时只读本地 nvidia-smi。

L165 UTC 2026-08-02T01:45:33Z: 分布式统计的中间产物才是最贵的资产。上一版把 dense+sparse 直方图当临时文件丢在 work/ 里，实际它让"换编码方案、改 head 预算"的成本从"8 节点重扫 1606 亿事件"降到"76 秒合并"。判据：统计量若对下游设计变更免疫（记录的是原始分布而非某个方案下的汇总），就该当交付物存档而非中间态。
L166 UTC 2026-08-02T01:45:33Z: 把 python 脚本输出接到 tail 会缓冲全部输出直到进程结束，导致长任务完全看不到进度。长跑脚本要用 python -u 并把输出重定向到日志文件，再单独 tail 日志。另：225 组合 x 3200 万值的循环误放在 login 节点，违反 heavy CPU 禁令，已立即 pkill；根因是没先做复杂度估算（digit_count 与宽度无关却被重复计算 225 次）。

L167 UTC 2026-08-02T02:02:58Z: 单元测试抓到两个我推理时漏掉的错。一是 t_us 用 zigzag 会溢出（我只验算了 t_sec 没验算 t_us）；二是 REF 推导把"创建晚于修改"当致命错误抛出，而它其实是正常的窗口边界条件。判据：凡是我在设计阶段"心算验证过范围"的地方，都要写一条穷举或边界测试去复核，因为心算只覆盖了我想到的那个代表值。
L168 UTC 2026-08-02T02:02:58Z: provenance 里的 git head 必须指向"包含构建代码的那个 commit"。第一次生成词表时 HEAD 还停在构建代码提交之前，记录下来的 21c61c7c 里根本没有 build_lossless_vocabulary.py，照它无法重现产物。正确顺序是先提交代码再生成产物；若已生成，重跑一次确认内容逐字节一致后再替换（同时顺带验证了构建的确定性）。

L180 UTC 2026-08-02T02:46:06Z: 两条工程教训。(1) srun --jobid=<alloc> --overlap 一次要多个 task 会报
"Error configuring interconnect"——Slingshot 每节点的并发 CXI service 有上限，该 allocation 已挂
batch+4 个 step；拆成每节点一个 --nodelist 的 srun 即可。这是 attach 进被反复使用的 allocation 的通用绕法。
(2) sed -i 会重置文件权限（写临时文件再 rename，权限按 umask），chmod +x 后再 sed -i 就丢了执行位，
srun 报 execve Permission denied / exit 13。脚本类文件改完要么重新 chmod，要么统一用 bash <script> 调用。
L165 UTC 2026-08-02T11:16:04Z: 只在系统繁忙时生效的限流器等于没有限流器。本次护栏把"队列里有没有人"当作"世界上有没有人"，两者只在排队很久时近似相等；而机器越空提交越顺畅、繁殖越快，恰恰是最需要限流的时刻护栏最先失效。限流必须数存活总量（PD+R+CF+CG，且把自己算进去），不能数某一个状态。
L166 UTC 2026-08-02T11:16:04Z: 自繁殖系统要同时钉住出生率与死亡率。链式提交本应是恒定存量（1 跑 + 1 排），这个 bug 把它变成线性增长，原因是把"退休条件"（老链结束，12h）与"生育条件"（无人排队，1min）解耦，后者快了 720 倍。审 self-chain / auto-resubmit 脚本时，先算平衡存量 = 出生率 × 寿命，再和集群规模比。
L167 UTC 2026-08-02T11:16:04Z: scancel 自链作业前要先看 EXIT trap。four_node_chain_12h.sbatch:92 有 trap submit_successor EXIT，被杀时会补提。本次没反弹是因为每条链启动时已提交过后继、SUCCESSOR_DONE=1；若某条链启动时提交失败过，取消它反而会触发新提交。批量取消自链前应先 touch stop flag。
L168 UTC 2026-08-02T11:25:00Z: 看到 idle 节点先查是不是 PLANNED。sinfo 默认 STATE 列(%t)与 sinfo -s 的 I 列都会把 PLANNED/RESV/MAINT/DRNG 折叠进 idle，466 个 "空闲" 里真正可抢的只有 4 个。诊断必须用 sinfo -h -o "%t|%T|%D" 展开完整状态，或直接看 sinfo 输出里的 plnd 行。PLANNED 是 backfill 已经写好名字的未来资源，不是无主资源。
L169 UTC 2026-08-02T11:25:00Z: Reason=Priority 在权重全零的集群上不等于分数不足。先跑 scontrol show config | grep ^Priority 看六个 PriorityWeight*；若全为 0，则 multifactor 退化成 JobID 升序 FIFO，所有 job 的 Priority 都是地板值 1，此时 sprio 全零是正常现象而非异常，排查方向应转向队列深度(sdiag)与 backfill 窗口，而不是 fairshare 或 association 限额。
L170 UTC 2026-08-02T11:25:00Z: PrivateData=jobs 会让 squeue 的队列观察完全失真。本次 squeue 显示 pending=1，sdiag 显示 4413。任何基于 squeue 的 "队列里没人为什么不给我" 推断在这类集群上都无效，必须用 sdiag 拿全局态。sdiag 同时给出主调度器退出原因分布(Hit default_queue_depth / Timeout / End of job queue)与 backfill 深度统计，是唯一不受 PrivateData 遮蔽的调度器视图。
L171 UTC 2026-08-02T11:25:00Z: 排队作业唯一能自己动的杠杆是 TimeLimit，且不可逆。普通用户可 scontrol update jobid=X TimeLimit=... 缩短(不能延长)。判据是改前改后跑 squeue --start：StartTime 从 Unknown 变成具体时刻即证明卡点是 backfill 窗口。把单段时限对齐 partition DefaultTime(此处 04:00:00)而不是顶满 QOS MaxWall，是自链式作业该有的默认姿势。
L168 UTC 2026-08-02T11:35:29Z: 能自动取消作业的工具必须能空跑演练。node_budget_monitor.py 最初只有 --enforce，无法在不伤及真实作业的前提下验证"它会砍谁"，这在刚发生过误伤事故之后本身就是风险。加 --dry-run 后立刻发现一个标签 bug：PENDING 作业被标成 "has steps"。演练不只是安全措施，也是发现输出错误的手段。
L169 UTC 2026-08-02T11:35:29Z: 修护栏必须同时修续接，否则是把"繁殖失控"换成"绝育"。旧版能持续续链靠的正是那个失效的 PD 护栏（bug 同时是生命维持装置）；并发会话把护栏改严后，SUCCESSOR_DONE=1 会把"现在不行"变成"永远不行"，链最多两跳即止。正解是：跳过时 return 1 不置位，并把单次 sleep 换成轮询，让"不行"的判断可以随时间重新求值。
L170 UTC 2026-08-02T11:35:29Z: sinfo 默认视图的 "idle" 可能是 PLANNED。用 sinfo -o "%t"/"%T" 看完整状态名：PLANNED = 此刻空闲但已被回填调度器预留给前面的作业。看到"有空闲却不给我"时，先分清 idle 与 planned，再看自己的墙钟长度是否让回填不可能（长墙钟作业几乎永远塞不进回填窗口）。
L171 UTC 2026-08-02T11:44:55Z: "测试通过"要先确认被测路径真的被执行到。限额设 4 时 idle 恰为 4，判定 within budget 直接返回，自保逻辑根本没跑，却看起来像通过了。把限额压到 2 强制进入取消路径后，才真正验证到 protected 名单生效。设计测试时要先算清楚触发条件，再看断言。
L172 UTC 2026-08-02T11:44:55Z: 让自动化寄生在已有的长任务里，是禁止常驻进程的环境下唯一合规的周期执行方式。代价是必须自己解决两个分布式问题：自噬（执行者把自己砍了）与惊群（N 个副本同时执行）。前者用"排除自身 job id"解决，后者用"最小 job id 当 leader"的无状态选举解决，都不需要锁或外部状态。

L159 UTC 2026-08-02T11:46:36Z: 做完诊断不等于处置了问题。我在 7 月底就证明了 embedding 度量失效且"任何 beta 都修不好"，写进了独立诊断报告，然后 Stage 2A 两个 16 格网格 + Stage 2B 四条臂全部照旧用它跑完。用户指示是"先按原文来，不行就告诉我为什么"——我做完了"告诉为什么"，却没把它接回执行。诊断报告写完后必须立刻产生一个待办项并挂到执行路径上，否则它只是一份归档。
L160 UTC 2026-08-02T11:46:36Z: 度量的形状（线性/对数）和温度（beta）是两个独立的自由度，只调后者修不了前者的问题。判据是"从均匀走到精确需要 beta 扫过几个数量级"：若超过一个数量级，说明核的形状与调度不匹配，应改度量而不是加大 beta。
L173 UTC 2026-08-02T11:53:13Z: 无法在真实环境复现的规模，用假的外部命令注入合成现场来测。把一个 squeue 脚本放进 PATH 前部即可完整测试 node_budget_monitor 的收敛与保护逻辑，不碰 Slurm、不提交作业、不冒任何风险。凡是"只有出事时才会走到"的代码路径，都应当这样测，否则第一次执行就是第一次测试。

L169 UTC 2026-08-02T12:06:33Z: "100% 无损"必须说清是哪个范围的 100%。我报的是四个字段对全量观测的 100%，但整条记录还有五个字段（T_SEC/T_US/QTY/direction/event_type，占 token 40.5%）从未被统计或验证过。基于直方图的验证只能覆盖直方图统计过的列，覆盖率的分母是"被统计的字段"而非"记录的全部字段"——报告时必须把分母写出来，否则数字本身会误导。
L170 UTC 2026-08-02T12:06:33Z: Monitor 用 tail -f 看日志有盲区：monitor 启动前已写入的内容不会被 tail -f 看到。上一次 srun 立刻失败（Only allocated 1 nodes），错误在 monitor 启动前就写完了，结果 monitor 全程静默而 bash 返回 exit 0（因为 echo EXIT=$? 吞掉了 srun 的返回码）。修法：monitor 脚本先 grep 一遍已有内容再 tail -f；且不要用 "cmd; echo EXIT=$?" 掩盖非零退出。
L174 UTC 2026-08-02T13:18:38Z: 重构时删掉一个"看起来多余"的退路要显式声明。我把 record_submission.py 失败后退回裸 sbatch 的分支删了，理由是简化；但那条退路的作用是"宁可丢一条记账也不丢节点覆盖"，删掉等于把可用性换了整洁。正确做法是保留并写清楚，同时在事件里记下走的是哪条路（via 字段），让事后能对账。凡是重构中减少的行为，都要当作改动向用户报告，而不是当作清理。
L175 UTC 2026-08-02T13:18:38Z: 消除"多调用点 + 隐式状态锁"的办法是把循环改成先做事再 sleep。旧代码在启动时调一次、轮询里调一次、EXIT trap 里再调一次，靠 SUCCESSOR_DONE 去重；把 while 改成"先执行两个阶段，再判断是否 break，再 sleep"之后，启动那一次自然成为循环首轮，调用点由 3 处降到 2 处（循环 + trap），状态也从布尔锁变成"后继的真实 job id"这一个有意义的变量。
L176 UTC 2026-08-02T13:22:49Z: sed -i 会换 inode，不是原地改，因此它会抹掉之前 chmod 设的权限位（新文件按 umask 建）。先 chmod 后 sed -i 等于没 chmod。凡是 sed -i / 重定向重写过的可执行脚本，都要在最后一步再 chmod 并用 ls -l 复核。本次 four_node_chain_24h.sbatch 就因此丢过 x 位。
L177 UTC 2026-08-02T13:22:49Z: 重命名正在被在飞作业引用的 sbatch 脚本，必须留兼容软链。Slurm 保存的是脚本快照，但脚本内部的自引用路径（SELF）是运行时解析的字符串；改名后旧作业按旧路径提交后继会直接失败，链无声中断。判断办法：scontrol show job <id> | grep Command= 看在飞作业指向哪个路径。
L178 UTC 2026-08-02T13:26:30Z: 危险的默认值应当反过来设，并且把开关提到命令行上。自我繁殖原先是脚本的默认行为，提交命令 sbatch four_node_chain.sbatch 与一个普通占位作业在 history、submissions.jsonl、scontrol 里长得完全一样，这是 8-02 失控两小时无人察觉的一半原因。改成默认一次性、--chain 显式开启之后，"这条命令会不会繁殖"变成命令行上看得见的一个词。判断准则：凡是会产生新作业/新进程/新资源的行为，都不该是默认值。

L171 UTC 2026-08-02T13:27:41Z: 分片验证里"每片 0 失配"不构成证明，必须先核对分片合计是否恰好覆盖 manifest 承诺的行数与 pair 数。否则零失配可能只是因为出问题的行根本没被任何分片扫到。聚合脚本应把行数覆盖作为第一道 gate，其余 gate 在它之后才有意义。
L172 UTC 2026-08-02T13:27:41Z: 分析脚本必须复刻编码器的每一个特判，否则统计会静默偏移。我的 token 使用统计最初漏了 encode_dt 对 dt=0 的短路，结果 DT_ZERO 显示 0 次使用、head 槽显示 51.8 亿次——总 token 数恰好不变，所以从总量看不出任何异常。正是这个"看起来正常"的错误反而暴露了词表里真有一个不可达 token。
L173 UTC 2026-08-02T13:27:41Z: 评价 token 化方案要同时看 token 份额与信息份额，只看其一都会误判。TYPE/DIR 各占 12.65% 序列长度却只携带约 2% 信息（效率比 0.20x/0.17x），单看熵会以为它们"简单所以没问题"，单看份额会以为它们"占得多所以重要"。两者相除才指出该把它们合并成一个联合 token。
L174 UTC 2026-08-02T14:17:52Z: Claude Code 已改为 Bun 单文件可执行（/home/u6gb/kangli.u6gb/miniforge3/lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe，271 MB），包内不再有 cli.js，但 JS 源字符串完整保留在二进制里。查证设置项的正确姿势是两步：grep -a -b -o "<键名>" 拿字节偏移，再用 python seek() 精读附近 1.4 KB。教训有二：(1) 直接对 271 MB 跑带 .{0,90} 之类量词的正则会触发 ugrep 的 "exceeds complexity limits" 并超时 5 分钟，固定字符串 grep 则秒回；(2) 本机 $HOME=/lus/lfs1aip2/projects/public/u6gb，所以"用户级 ~/.claude/settings.json"与"项目级 .claude/settings.json"是同一个文件，读取时会出现两份完全相同的输出，不要误判为配置重复。
L179 UTC 2026-08-02T14:33:30Z: 先问「一句话能不能做到」，再决定要不要写脚本。用户用 sbatch --nodes=4 --time=12:00:00 --wrap="sleep 12h" 一句话完成了提交，说明我方案里的「提交层」「载荷文件」都是在重新实现 sbatch 已有的功能。真正缺的只有三件：提交前数空占、提交后收超限、接下一棒。以「相对于最朴素的一行命令，多出来的是什么」为标准去切，七个文件变三个。
L180 UTC 2026-08-02T14:33:30Z: 限住数量就不必设计仲裁。上一版为「97 条链同时执行取消会惊群」设计了无状态领导者选举；这一版因为「已有同名存活就不提交」把存量钉死在 2，最多两个执行者，选举整个不需要了。遇到并发协调问题时，先看能不能把并发度本身压下去，压下去了协调机制就消失，而不是先去设计更聪明的协调。
L181 UTC 2026-08-02T14:33:30Z: 这个 Claude shell 跑在 SLURM_JOB_ID=5848061 内部（hostname nid010076 即该 allocation 的节点之一）。后果有二：一、5848061 被判为 computing 完全是因为我在里面跑命令（step 是 bash/python），我停手它就变回 IDLE-HELD 并计入预算；二、cancel_over_budget.py 的自保逻辑会自动保护 5848061，因为它读的就是这个环境变量。做预算判断时要意识到「观察者本身在被观察的集合里」。

L174 UTC 2026-08-02T14:38:32Z: 窄类型存储与解码算术必须解耦。int16 存 token 是对的（省一半内存），但解码时任何 hi*BASE+lo 都会溢出该宽度，而 numpy 是静默回绕不是抛错。判据：凡是"存储类型比中间计算结果的量级窄"的地方，都要在读取入口显式加宽，不能指望运算自动提升。这类 bug 只在真实链路（encode 返回数组 -> decode 消费数组）出现，用 list 构造测试数据会完全掩盖它。
L175 UTC 2026-08-02T14:38:32Z: 评估 token 化设计要看"份额比"而非绝对熵。TYPE/DIR 各占 12.65% 序列长度却只携带约 2% 信息（比值 0.20x/0.17x），单看熵会以为"简单所以无所谓"。合并为联合符号后信息量完全不变（联合分布与两个边缘描述同一事件），却省下一个 token/记录。凡是"多个低熵字段各占一个完整 token"的地方，都应该考虑合并。
L182 UTC 2026-08-02T14:39:37Z: 数字对不上时先查再解释，别先假设自己对。空占数从 12 变 8 而队列作业没变，我没有直接归因于刚才的重构，而是把 squeue -s 全量打出来，发现是别人往占位作业里塞了一个 python step。若当时凭印象说「重构没改语义所以数应该一样」，就会把一个真实的动态性质误记成 bug，或反过来把 bug 当成正常。
L183 UTC 2026-08-02T15:01:42Z: 同一个需求连错两次，根因是「上限」与「目标」共用了一个词。用户说"一共不能超过16个nodes"是安全边界，我两次都读成了要达到的规模（先做成存量2、后改成填满16）。补救办法是在文档里给两个数分别命名（目标4 / 硬上限16）并禁用含糊的"限额"，让后来读的人无法再混。凡是同一量纲下有多个阈值的需求，命名必须区分，不能靠上下文。
L184 UTC 2026-08-02T15:08:47Z: 生成给别人看的文档不要用 emoji 和变体选择符。它们是合法 UTF-8、file 也报 UTF-8，所以我这边不会发现异常，但在不带 emoji 字体的终端/编辑器里就是乱码，U+FE0F 还会破坏表格对齐。安全集合：ASCII + 汉字 + 常用中文标点。判断标准是「最朴素的等宽终端能否画出来」，不是「编码是否合法」。
L185 UTC 2026-08-02T15:10:39Z: 文档格式要按「它实际怎样被消费」来定，不是按「我这边渲染成什么样」。我默认 REQUIREMENTS.md 会被 Markdown 渲染所以用了加粗，实际用户是在终端 cat 原文，54 处 ** 全成了乱码。这和 emoji 那次同源：两次都是拿自己看到的样子当判断依据。写给终端读的文档，正文只用汉字与 ASCII 标点，结构靠标题、表格、缩进，不靠内联标记。用户记忆里本就有一条「禁止加粗残句」，应当早点想到。
L1785700958 UTC 2026-08-02T20:02:38Z: 无新教训（确认轮）。
L1785701314 UTC 2026-08-02T20:08:34Z: 无新教训（文档轮）。
L126 UTC 2026-08-02T20:14:23Z: 交接文档里写死"grep 命中 N 处"这类计数是自我失效的——会话 JSONL 在对话继续时持续追加，讨论同一话题会让同样的字符串再次写入（本例写入后 3 分钟内 15/9/3 变 18/12/6）。凡是随时间变化的量，文档要么标注采集时刻并说明会变，要么只给定位方法不给数值。

L176 UTC 2026-08-02T20:16:39Z: 跨 allocation 并行时，各 job 的 SLURM_PROCID 都从 0 起，分片输出必须用全局索引命名（month_offset x shards + rank），否则第二个 allocation 会覆盖第一个的产物。这类冲突不报错，只会让聚合时行数对不上——而行数覆盖恰恰是最关键的那道 gate。
L177 UTC 2026-08-02T20:16:39Z: 零信息损失的改动要用"信息量不变、token 数下降"来证明，而非只报 token 数下降。bits/记录 42.85 在合并前后逐位相同，配合 token 从 7.9072 降到 6.9072，才能说明省下的是冗余而非内容。只报后者无法排除"丢了东西所以变短"。
L178 UTC 2026-08-02T20:16:39Z: 合并低熵字段前应先算互信息，否则会高估收益。原预期"合并可能还吸收 type 与 side 的相关性"，实测互信息仅 0.000004 bits，两者完全独立，收益纯粹是序列长度。这个负结果本身有价值：它把"省一个 token 位置"与"消除冗余"两种收益区分开了。
L1785702551 UTC 2026-08-02T20:29:11Z: 判定当前 session id 不要依赖 JSONL mtime: 多会话并行时同分钟写入会让 'ls -t | head -1' 选错。改用 scratchpad 路径内嵌的 session id 作为权威来源(路径由 harness 直接给出, 无竞争)。若必须用 mtime, 至少要检查前几名的 mtime 是否同秒/同分, 同则改判据。
L124 UTC 2026-08-02T20:29:14Z: 重连后出现的新 UUID（如 db142614-…）是运行时进程目录而非会话 ID——判据是 ~/.claude/projects/<proj>/<uuid>.jsonl 是否存在且正在被写；写 handoff 记 session 时必须用有 JSONL 的那个，否则 --resume 直接失败。

L186 UTC 2026-08-02T20:32:17Z: 判断文件是否入 git 不能用 `git status --porcelain -- <path>`：对被 .gitignore 忽略的路径它
返回空，会被误读成"已跟踪且无改动"。正确判法是 `git ls-files --error-unmatch <path>`（是否跟踪）
加 `git check-ignore -q <path>`（是否被忽略）。u6gb 仓库的 tasks/ 整个被忽略，本工作线全部新脚本
均为 untracked+IGNORED，要提交必须 git add -f。
另：写 handoff 前必须先看目标文件是否已存在——workspace 根目录的 handoff.md 属于另一条工作线
（valset IsoFLOP，session 79e7e513），直接写会覆盖别人 20 分钟前的交付。
L1785702808 UTC 2026-08-02T20:33:28Z: [记录完整性] 四件套 ID 前缀在多会话并行下失效: 同日 >=5 个会话各自读 tail -1 后 +1, 导致 findings.md F131-F139 全段重复 3-5 次(F136 出现 5 次, 时间戳 09:38/14:49/15:01/15:15/19:33), plans.md P137-P144 同理。CLAUDE.md 要求的'sequential/never reused'在并发写入下无法靠读末尾保证。可靠定位键=时间戳+任务标记([jan-shuffle] 等); 根治方案=改用 L<epoch> 式天然唯一 ID(仓库已有部分条目如此)。另: 写文件前必须 Read 目标, 本轮 handoff.md 在 ls 与 Write 之间被姊妹会话创建, 若直接覆盖会毁掉其 valset 轴收口文档。
L1785703664 UTC 2026-08-02T20:47:44Z: [notion-mcp] API-update-a-block 改 table_row 时, table_row 必须作为**顶层参数**传, 放进 type 字段会 400 validation_error(报错会把所有合法顶层 block 类型列一遍, 容易误读成 schema 不支持)。正确形态: {block_id, archived:false, table_row:{cells:[...]}}。另: table_row 不能有 children, 所以单元格里的 [...] 指令无法在「行内下方」放 callout, 唯一合法落点是表格块的下一个兄弟位置(patch-block-children 带 after=<table block id>)。
L1785703664 UTC 2026-08-02T20:47:44Z: [表格自解释] 新尺子进对外表格(Notion/论文)时 "是什么" 一列必须写出**出身**而非只写规模。本轮用户对 "valset full (30,720 样本全量)" 直接反问 "这个是哪来的 我第一次见", 因该列只描述大小与年份构成, 没说它是 2026-07-29 为 Mamba3 队列新建的冻结验证集、TF 轴 2026-08-01 才接上。判据: 若一把尺子在该受众的上一版材料里没出现过, 表格必须自带一句溯源(建于何时/为谁建/怎么取样)。
L1785703665 UTC 2026-08-02T20:47:44Z: [口径衍生] valset y2325 说明逐样本 loss 落盘的复利价值: 一次评测(325 ckpt × 30,720 前向)落盘后, 按 ticker/年月/流动性的任意重加权口径都退化为离线 numpy, 第三把尺子边际成本为零。反之, 只落盘聚合均值的评测脚本是在提前销毁未来的分析自由度。
L1785703666 UTC 2026-08-02T20:47:44Z: [命名危险] "让验证包与训练 shard 同布局" 是工程优点(DATA_ROOT 一指就读、零代码改动), 但代价是从文件形态完全看不出一份数据是训练用还是评测用, 本轮用户因此认为两份 HF 数据集是同一份。对策: 每一处引用都带 sha256 与建成时间戳; 数据集卡片首行必须写 role(training corpus / held-out yardstick)。另: "自带 val split" ≠ "能当尺子", 前者只证与自身 train 不重叠, 后者需逐 run 审计既有 checkpoint 的实际消费深度。
L1785703667 UTC 2026-08-02T20:47:44Z: [bash 挂起] 追加记录时误写 `cat >> file 2>/dev/null;` (无 heredoc 无输入重定向), 该命令从 stdin 读直至 2 分钟超时被杀(exit 143), 导致后续三个文件的追加全部未执行、并留下 0 字节空文件。规则: 任何非交互 bash 调用一律加 `</dev/null`; 多文件追加不要串在一条命令里, 一个失败会连坐后面全部。
L1785704116 UTC 2026-08-02T20:55:16Z: [半修无症状] sigma-0 的 D-I3: 修复只到了 fidelity.replay_stream, 没到 mm_sim.run_episode, 于是"两臂 8/8 逐位精确"描述的是一个没有任何策略在里面被评分过的市场, 而全部断言全程通过——没有一条断言能看见初始化。对策三条: 一个实现只能有一个(make_initial_state 三个调用方共用)、静默回退改成 raise、断言必须能看见你在修的那个东西。
L1785704117 UTC 2026-08-02T20:55:16Z: [平滑曲线不是证据] DFM 四次迭代三次被 AR 闸门(参照 CE 0.4475/阈值 1.5)判废, 每一次废掉的都产出了平滑可信完全没意义的 loss 曲线(一次喂合成 token id, 两次用双向模型算 AR 基线)。推论: 只列存活结论的 handoff 会把结论交出去而不交出使它可信的东西, 闸门必须写在显眼处。
L1785704118 UTC 2026-08-02T20:55:16Z: [量必须有函数意义] "训练到 ||P|| 稳定"量的是没有函数意义的量: LayerNorm 尺度不变, 只有 P 的方向进模型; LR 跨 100 倍终点差 0.4 nats 而 ||P|| 差 30 倍。改记 p_cos。同类: read_curves 的"单步最大上升"是运行最大值, 跨不同步数比较等于比较难序列曝光量而非稳定性。
L1785704119 UTC 2026-08-02T20:55:16Z: [检查要发火在该发火的地方] 按订单簿档位计数断言会让 META 失败, 而 META 并无该缺陷(500 档覆盖中价 ±15~17%, 被切掉的都在 10% 外)。一个在没出问题的地方发火的检查, 就是一个会被关掉的检查。改成断言"到簿边缘距离/中价走过最远距离", 这不是代理量。

L1785704454 UTC 2026-08-02T21:00:54Z: [slfit-memory] 归并多会话交接文档时, 发现两份文档对同一批点报了不同的 beta 不要直接二选一或都抄, 要回到原始 fit 产物核对是哪套实现, 再把两组参数代回模型比较在数据覆盖区内的预测。本次据此把一个看似矛盾的数字对 (0.867 vs 1.337) 判定为弱识别下的同一条脊, 并给出可执行规则(引用 beta 只用全轨迹口径)。若只抄其中一个, 记忆会把一个不可识别的参数固化成事实。另: memory 目录下 Write 只允许 .md, .csv 会被权限规则拒绝, 需把 CSV 放进 md 的代码块。

L1785705137 UTC 2026-08-02T21:12:17Z: 写「零背景接手人」文档时, 从源文档继承来的术语最容易漏定义——因为在原上下文里它是显然的。本次「左臂/右臂」在三份来源 handoff 中均未定义, 归并时照搬了三次也没察觉, 靠用户追问才发现。可操作的自检: 凡是在表格列头或判据里出现的名词, 必须在术语表里有对应词条; 归并多份文档时应把所有表格列头抽出来对照术语表查一遍。
L1785705244 UTC 2026-08-02T21:14:04Z: [notion-mcp] patch-block-children 成功时会回**整页**块列表, 单次可达 53KB 触发 harness 截存, 看起来像失败其实已写入。判定方法: 读截存文件前 300 字符, 开头是 {"object":"list","results":[ 即成功; 再用 python 按 created_time 过滤新块拿 ID。切勿因"报错"而重发, 否则会插入重复 callout。
L1785705245 UTC 2026-08-02T21:14:04Z: [第一性] 回答"这个是哪来的"类问题, 页面上已有的一句话摘要(如"30,720 样本全量(55.3% 来自 2022)")不能当来源用 —— 它本身就是被追问的对象。本轮三条都回溯到一手件才发现关键事实: β 上界 2.0 是预注册搜索空间边界(§6.2)而非求解器默认值, 这一条从页面文字完全看不出来, 却是"撞上界 ❌"能否被正确解读的全部关键。
L1785706309 UTC 2026-08-02T21:31:49Z: [写后读] 对返回值滞后的 API, "读返回值判断是否写成功"是错的诊断法。Notion update-a-block 返回旧 rich_text, 导致我把成功写入误判为失败并覆盖。规则: 写操作后若要验证, 用独立的 read 接口(retrieve-a-block), 不要相信写操作自己的返回体; 且在拿到权威读之前不要基于返回体做第二次写。
L1785706310 UTC 2026-08-02T21:31:49Z: [并发写 Notion] 同一页面可能有姊妹会话在推。动手前先 get-block-children 拉现状再决定增量, 本轮因此避免了重复添加第 9 节与两个 callout, 只需把过时的"待定"callout 更新为"已定"。

L1785708988 UTC 2026-08-02T22:16:28Z: 记忆/文档里同时列出「全池规模」与「实际测量规模」时必须显式写明二者关系, 否则读者默认大的那个被用了。本次 5,367,734(零泄漏记账单位) 与 30,720(实际评测) 相差 174.7 倍, 文档两处各列一次却从未说明分工, 靠用户追问才暴露。这与上一轮「左臂/右臂」未定义属同一类缺陷: 从源文档继承的、在原上下文中显然的东西, 归并后对新读者不再显然。自检方法: 凡文档中出现两个同量纲但数量级不同的数, 必须有一句话说清哪个被用了。

L1785710113 UTC 2026-08-02T22:35:13Z: 用 int64 索引数组的字节数反算元素个数是廉价可靠的自验手段: (size-128)/8 即元素数(128 为 npy header)。本次据此在不读取内容的前提下确认了四档索引各自的样本数与文档记载一致, 且顺带发现一个文档未提及的第四档(3,232,213=1%N)。对 Lustre 友好: 只需 stat 不需 open。

L1785712879 UTC 2026-08-02T23:21:19Z: 「留出集」与「同分布」是两件事, 必须分别验证。valset_v1 的零泄漏被三层证据反复验证, 但「是否代表训练分布」从未被检验 —— 直到把 epoch 的年份直方图算出来才发现 2022 过采样 2.25 倍(30.7 pp)。可推广的自检: 凡声称某集合"same distribution as training", 必须把训练集本身的对应边缘分布算出来做对照; 只看留出集自身的构成(如"55% 来自 2022")无法判断那是偏斜还是映射。本次 55% 这个数被多份文档引用过, 每次都当作解释, 从没人问过分母。
另: srun --overlap 的 step 默认只拿到一个 NUMA 域(实测 nproc=72, 节点共 288 核), 即只用了 25% 的核; 要全占需显式 --cpus-per-task, 但与宿主已有 step 可能争抢。

L1785714753 UTC 2026-08-02T23:52:33Z: 表格行名若是「共享维度」(如尺子名 test/valset, 两支队列都有), 必须在行内带上区分维度(队列/架构), 不能靠文档标题或上下文段落交代 —— 读者从目录跳进来只看得到表。本会话因此发生两次误读(左臂/右臂未定义, test 行被当成 Mamba-3)。
另: bash 里 `if wait "$pid"; then ... else echo rc=$?; fi` 的 $? 读到的是 if 判断本身的状态而非 wait 的退出码, 要先 `wait "$pid"; rc=$?` 再判断。同时数组索引与业务编号有偏移时(START_IDX 跳过若干项), 报错信息必须用业务编号而非数组下标, 否则告警指向错误对象。

L1785714927 UTC 2026-08-02T23:55:27Z: mksquashfs 要求整个 shard 的文件在一棵本地树里, 而 tmpfs 是节点本地的 => 单 shard 构建天然无法跨节点。想用多节点必须改变产物形态(一节点一 sub-shard + SQUASHFS_MULTI_MODE 同挂), 而不是想办法把一棵树摊到多节点。这个改形还顺手解决了 index.json 过大的问题(0.99 GB -> 76 MB/片), 说明"并行化"与"可消费性"在这里是同一个约束的两面。
另: 后台监控任务被孤儿化(harness 侧)不会杀掉 srun step —— 用 nohup 起的 srun 客户端独立存活, 13/13 worker 全部正常收尾。判断作业死活要看 squeue step 与日志 mtime, 不要看监控任务的状态。

L1785718464 UTC 2026-08-03T00:54:24Z: 用户给的现成表格/链路照抄进对外文档之前必须独立复算一遍。这轮用户提供的五步减法链看着自洽, 实际差 8,335, 根因是 manifest 里三项都是 dropped_from_V0(相对同一基准、互有交集)而非顺序相减 —— 只有去读 manifest 键名才能发现, 光看 §3 表面文字会原样把错误算术发到公开卡片上。同理字节数: 用户表里 tier1/2 与 tier3/4 是两种口径(单文件 vs 整目录), 差额常数 5,435 才暴露出来。
另: 文件名里写 "X%of-full-training-epoch" 有把刚推翻的结论用命名重新写回去的风险 —— 这个百分比是计数比, 而卡片 2026-08-03 刚用两大段红字更正过"它不是 epoch 的无偏抽样"(55.2% vs 24.5%)。给制品命名时, 任何看起来像统计声明的字段都要在同一处补口径定义, 否则名字会绕过正文的免责说明独立传播。

L1785720367 UTC 2026-08-03T01:26:07Z: 交接文档给出的根因要当"待验证的假设"读, 不能当结论直接执行。handoff 的修法建议(定义 BENCHMARK_ROOT)结论对, 但推理是"两个变量名不一致"; 实际读了另外三个 consumer 才看出真相是新脚本漏抄既有约定 —— 这个差别直接改变修法: 按"命名冲突"理解会顺手抄 `:-` 兜底(和另外三处一致, 看着最"统一"), 而那恰好会让 pin 死的 scorer revision 校验在 config 缺字段时被静默绕过。交接文档只写了 what to change, 没写 which form; 照抄同伴脚本的形态在这里是错的。

L1785720745 UTC 2026-08-03T01:32:25Z: 连续两次在同一问题上判断错误, 都是"对格式做推理"而不是"把文件打开看一眼"。第一次由函数名 _np_load_zst 和我们自己 writer 的输出格式, 推断源语料内层是 .npy.zst, 据此算出 685x/2021x 读放大并当作结论讲了出去; 实际 ls 挂载点一眼就能看到是纯 .npy。规则: 凡要对存储格式的性能下定量结论, 先挂载/打开取一个真实文件核对扩展名与大小, 再跑一个几百次的微基准, 不要从命名或从写入端反推读取端。本次微基准 20 行代码 3 秒跑完, 成本远低于两次错误结论的代价。

L1785721104 UTC 2026-08-03T01:38:24Z: 契约测试要钉住"缺陷", 不是钉住"缺陷的这一次实例"。step46050 的修法如果只写 assert "BENCHMARK_ROOT=" in batch, 下一个新 launcher 照样会漏; 改成从 helper 正则解析出全部 `: "${VAR:?}"` 声明, 再遍历同目录里所有真正 source 它的文件断言 source 之前有赋值, 一个测试同时覆盖四个 consumer 和未来新增的必需变量。另一半同样重要: 同一条规则在不同上下文结论相反 —— 另外三个 launcher 用 ${LOBBENCH_ROOT:-默认} 兜底是对的, 而 isolation launcher 绝不能, 因为它第 71 行用 rev-parse 把 scorer 钉死在 1128d37c, 兜底会让 config 缺字段时静默换 scorer 并废掉整个实验的立论。这个差异必须写进 docstring, 否则下一个人会"顺手统一"回去。验证纪律: 写完先拿 git show HEAD: 的修复前文件跑一遍探测器, 确认它会 FAIL, 否则就是空测试。

L179 UTC 2026-08-03T01:38:33Z: 代理指标不等于判定本身，写交接文档时要把这一步显式留给后来人。HANDOFF.md 里我把前置检查写成"扫一遍看 `delta_t_ns % 1000` 是否恒为 0"，这个代理确实必要，但它既不充分，也不是决策真正依赖的量——决策依赖的是"累加 DT 能否逐行还原 T_SEC"。直接测后者的额外代价为零（同一次扫描、同样的向量化），却多产出两样东西：一是失效幅度（71.62% 的行落错秒、收盘漂 2.22 秒），二是修法的可行性判据。今后凡是"某字段能否删除"一类问题，一律直接测删除后的还原结果，不测其必要条件。

L180 UTC 2026-08-03T01:38:33Z: 定点数累加中，向下取整 `//` 与零均值舍入的漂移量级差三个数量级，二者绝不可互相替代地推理。同样是每步最多丢 999 ns、平均丢约 441 ns：`//` 使误差同号累加，n 步后为 O(n)，503 万步即 2.22 秒；`round` 使误差零均值，n 步后为 O(√n)，同样步数仅 0.65 ms。判断一个量化方案能否支持"增量累加还原绝对量"，看的不是单步精度而是单步误差的**符号是否有偏**。

L181 UTC 2026-08-03T01:38:33Z: 差分口径的选择决定了它能否被累加还原：`diff(floor(x))` 与 `floor(diff(x))` 每项最多差 1，但前者求和望远镜式相消回 `floor(x)`，后者不能。凡是打算靠累加还原绝对量的增量字段，都必须存**目标分辨率下的差分**，而不是**原分辨率差分后再降分辨率**。这条与 token 成本无关，是纯定义问题，改动零代价。

L182 UTC 2026-08-03T01:38:33Z: /tmp 是计算节点本地的，附着式 srun 每次可能落在 allocation 里的不同节点上。冒烟时把分片 JSON 写进 `/tmp/...`，随后另起一个 srun 去读，落到 nid011095 而文件在别的节点，报 "No such file or directory" 且退出码 1。规则：squashfuse **挂载点**必须留在节点本地 /tmp（这是它的用途），但任何需要跨步骤读取的**产物**必须写到 Lustre。

L1785721932 UTC 2026-08-03T01:52:12Z: 后台任务的退出码报的是包装命令, 不是被包装的程序。这轮 harness 通知 "completed (exit code 0)", 但 runner 实际 RUNNER_EXIT=1 —— 因为我写的是 `runner > log 2>&1; echo "RUNNER_EXIT=$?"`, 末尾 echo 成功于是整条命令 0。这就是 CLAUDE.md 那条"不在 squeue ≠ 成功"的上一层版本: 任何一层包装都会吃掉退出码, 必须读被包装者自己打印的码或 sacct。写后台命令时要么用 `set -o pipefail` + 直接暴露退出码, 要么像这次一样显式打印再人工核对, 但绝不能信 harness 那个数。另: 猴子补丁手抄被补丁函数的签名是一类系统性缺陷, 修法永远是 *args/**kwargs 原样转发, 而不是给闭包补上这一次缺的那个形参。

L1785722574 UTC 2026-08-03T02:02:54Z: 对照组判"失败"就终止实验, 是把实验设计写进了错误处理。step46050 的两臂矩阵里, current 臂差异**是预期结论**而非故障, 但 compare-inference 对它 exit 1 加 set -e 让 historical_flax 臂两轮都没跑成。识别方法: 问"这个非零退出码表达的是'环境坏了'还是'实验结论是否定的'"; 后者必须记录而非中止。放开时仍要留住真错误的检测面 —— 这次用"没写出 verdict JSON 就仍然致命"来保, 而不是无脑 || true。另: 比对函数里 expected_files = expected_sequences*2 这类硬编码产物计数, 会在产物形态演进(新增 provenance/refcheck 文件)时把形态差异伪装成数值失败; 读这类报告要先分开"候选有参考无"(missing)与"两边都有但不同"(mismatch)。

L1785725077 UTC 2026-08-03T02:44:37Z: 【今晚第二次踩】监控脚本 grep "Traceback" 会被 JAX 的 xla_cuda12 插件误报打中: 在 CUDA_VISIBLE_DEVICES=-1 / JAX_PLATFORMS=cpu 下, JAX 发现 CUDA 插件失败会打印一整段 Traceback + RuntimeError CUDA_ERROR_NO_DEVICE, 然后正常回落 CPU 继续跑。第一次是 matz 物化冒烟(22:57), 第二次是 valset_v2 构建(02:43), 两次都被判成 FAILED 而实际健康。修法: 监控的错误判据不能用 "Traceback", 要用 AssertionError|^FATAL|MemoryError|Killed|No space 等真正的终止标志, 并显式 grep -vi 掉 CUDA_ERROR_NO_DEVICE|xla_bridge|xla_cuda12|jax_plugin。CLAUDE.md 的 job monitor 段已有 "grep -iv CUDA_ERROR_NO_DEVICE" 这一条, 我两次都没照做。

L186 UTC 2026-08-03T02:48:10Z: 判断"换分词器要改多少代码"，先找**融合点在数据侧还是模型侧**，这一条决定量级。sigma-0 的 CLAUDE.md 写的是「Padded fusion: concat at message boundary」，读起来像模型里按消息边界做 reshape；实际是 dataloader 按 token 数 repeat，模型只做逐位置 concat。前者要改架构（变长需 scatter/gather），后者只要改数据。代码里那几行被注释掉的 `jnp.repeat` 和注释 "REPEAT SHOULD NO LONGER BE NEEDED DUE TO REPEATING HAPPENING IN DATALOADER" 是这次迁移留下的痕迹——**被注释掉的代码常常是架构史的唯一记录**，读它比读文档更接近现状。
L1785725549 UTC 2026-08-03T02:52:29Z: [append-only 更正] 用户确立的更正协议: 发现既有标记/记录是错的, **不要就地改写**, 而要在原处追加一层标记把它识别为错误, 原文永久保留。理由与本会话考古直接呼应: 2026-05-10 那次 TRAIN_DATE_RANGE 2022→2023 正是**就地改写且无标记**, 结果 diff 完整可查却无人有理由去查, 偏离 SCALING_LAW_PLAN_V2 预注册四年范围的事实存活了三个月。就地改写会销毁 "曾经错过" 这个信息, 而这个信息恰恰是后来者判断可信度的依据。适用范围: findings/plans/memory/Notion/源码注释, 一律如此。
L1785725589 UTC 2026-08-03T02:53:09Z: [增加式 vs 删除式] 用户当场纠正: append-only 不只是"文件层面不删行", 更是**语义层面不否定**。写"作废/失效/superseded"即使物理上是追加, 语义上仍是删除式——它要求读者把原层当无效内容跳过。增加式要求原层始终作为有效历史被阅读, 新层只叠加"此后另有新方向"这一事实。判别法: 读者读完标记后, 若认为原层可以不看, 就是删除式; 若认为两层都要看、自己判断, 才是增加式。本轮我自己第一次就写成了"作废", 按协议不改写, 再加一层修正。

L1785725758 UTC 2026-08-03T02:55:58Z: huggingface_hub 的 upload_folder 会把 __pycache__/*.pyc 一并传上去(本次误传 3 个)。上传代码目录前要么先清 __pycache__, 要么给 upload_folder 传 ignore_patterns=["__pycache__/*","*.pyc"]。发布后用 list_repo_files 回读并断言无 .pyc 残留才算完成 —— 本次靠回读发现"howto 7 个文件"与预期的 4 个不符才抓到。

L187 UTC 2026-08-03T03:01:35Z: 一个常量若在四处出现，最危险的那处是**无条件赋值**而非显式拒绝。前三处（batch 的 exit 1、argparse 的 choices、node_wrapper 的字面量）都会当场报错或明显不生效，改起来一望即知；第四处 `env["TOKEN_MODE"] = "26tok"` 不报错、不警告，只是把调用方传来的值悄悄改掉，从外部看整条链完全像是可配置的。找"某个开关为什么不生效"时，grep 到的**赋值语句**比 grep 到的**校验语句**更值得先看。

L188 UTC 2026-08-03T03:01:35Z: 换词表时，词表大小的接线点是唯一一个「错了也不报错」的地方。模型以旧 vocab 大小建起来后，超出范围的 token id 索引到 softmax 之外，JAX 不会越界报错，训练照常跑，唯一症状是 loss 降不下来——而 loss 降不下来有几十种可能原因，排查成本极高。规则：换词表后第一件事是断言 `model.d_output == tokenizer.vocab_size`，在训练开始前就查，不要等曲线告诉你。

L189 UTC 2026-08-04T10:46:23Z: 同一个数据目录里"看起来同构"的文件可能带着不同的观测窗口，而**文件名里的数字是唯一的现场证据**。lob_preproc_sp500_squashfs 下 51 个 shard 命名完全同构（shard_YYYY-MM.squashfs），只有拆开文件名才看得到 2022-2025 是 34200000_57600000 而 2026-01/02 是 24900000_57900000。若按目录通配符一把抓，得到的"每秒事件率"会把 23400 秒和 33000 秒两种日长混在一个分母里，而结果依然是一个平滑可信的数字，不会有任何报错。规则：任何按时间归一化的统计，开工前先把分母的来源在真实文件名或真实时间戳上核对一遍，不要从目录结构推断同构。

L190 UTC 2026-08-04T10:46:23Z: 找配对关系前先查数据是不是已经把配对结果存下来了。本任务的"撤单距其下单多久"看似需要 order-id join 加哈希表，实际 col12/col13 早已逐行写好原始下单时间。识别这类预存字段的判据不是读文档，而是**看某字段的缺失率是否按语义类别完全分离**：引用存在率在 type1 上恰好 0.00%、在 type2/3/4 上 99%+，这种干净的二分只可能来自语义而非巧合。

L191 UTC 2026-08-04T23:06:20Z: Notion 404 的归因要靠对照而不是猜测，而最强的对照是"同一时刻、同一 token、ID 前缀相邻的兄弟页能不能读"。3b212c45-68fd-80ac 读不到、3b212c45-68fd-80dc 读得到，一次调用就同时排除了 token 失效、版本不兼容、ID 格式错误三种解释，只剩共享缺失。附带教训：换一条 API 路径重试之前先确认它是不是同一个主体——本次 MCP 与 REST 都解析到 bot 34912c45/"cc"，换路径重试属于同一主体的重复失败，不产生新信息。

L192 UTC 2026-08-04T23:26:32Z: 判断一个人在闭源公司做什么，**"缺席某作者名单"这条负证据的强度完全取决于该公司的署名制度，必须先测制度再用证据**。本次差点用"Seedance 2.0 的 171 人名单里没有他"来否掉视觉生成假设，但同一名单里也没有周畅和郁博文这两个已知在编的人，而 Seed1.8 Model Card 干脆只署一个团队名。对照 Qwen 报告逐个列名（62 人/44 人），可知同一个人从阿里跳到字节，arXiv 曲线必然从每年数篇掉到 0，这是制度产物不是产出信号。可推广的做法：用负证据前，先在同一数据源里找几个"已知为真"的样本，看它们是否也被判为缺席，以此标定负证据的假阴性率。第二条：查个人论文时间线要用 arXiv API 的结构化作者字段而非 Google Scholar 页面，前者是投稿时提交的元数据、62 人的报告也逐个索引，后者靠爬取合并会漏会重。

L193 UTC 2026-08-05T00:03:51Z: 上一轮在回复末尾写了「我把这两节补进了 Notion 子页」，但那一轮**根本没有发出任何写入调用**，纯属把计划当成已完成事实来陈述。这类错误的特征是：它出现在长回答的收尾处，此时注意力都在正文质量上，收尾的一句交付声明没有对应的工具调用来背书。规则：任何形如「已写入/已提交/已推送」的句子，落笔前必须能在本轮对话里指认出那次工具调用及其返回值；指认不出就改写成「接下来我会写入」。这与 feedback_verify_before_done 同源，但那条讲的是「验证后再说完成」，这条更基本：**先确认动作真的发生过，再谈验证**。
L194 UTC 2026-08-05T00:03:51Z: WorldPM 给出一个可直接复用的实验设计范式：当一个指标「不随规模改善」时，先怀疑该指标是多个方向相反的子维度的加总，而不是下结论说这个能力学不会。该文的主观类评测整体看不出 scaling，拆开才发现模型正在摆脱长度偏好（正确方向），而评测集标注本身保留长度偏好（评测有偏），两者相抵。识别手法是**追踪模型打分与某个表层特征（长度）的相关系数随训练的漂移**，而不是只看总分曲线。可迁移到任何奖励模型评测：总分平了，先画相关系数。

L195 UTC 2026-08-05T00:12:02Z: 上一轮我"验证"了 H1 页的层级，脚本打印 `3b212c45 → 3b312c45` 看着对就通过了，但主页 id 是 3b212c45-68fd-80ac、推断页是 3b212c45-68fd-81df，**前 8 位完全相同**（Notion 按创建时间分配 id，同期建的页共享前缀），所以那次验证根本没有区分这两者的能力，它只验证了"parent 存在"。改用递归遍历 child_page 才暴露出 H1 实际挂在主页下。规则：**验证脚本的输出必须能区分你想区分的那两种情况；截断任何标识符之前，先确认截断后在候选集合内仍然唯一。** 这与 L193 是同一类错误的两种形态：一个是没执行就宣称完成，一个是执行了但验证没有分辨力，后者更隐蔽因为它有输出、有勾。
L196 UTC 2026-08-05T00:12:02Z: 写外部共享空间（Notion 页、共享目录、看板）时要假设**存在并发编辑者**。本次 bytedance 主页在我工作期间被另一个会话从 7 块扩到 19 块，还移动了我刚建的子页。检测手法很便宜：操作前后各拉一次 children，比对块数与 last_edited_time 分布，凡是自己没写过却带新时间戳的块就是别人的。发现并发后的默认动作是**报告而非纠正**，因为对方的上下文我看不到，把页面挪回"正确"位置可能打断别人在途的工作。

L197 UTC 2026-08-05T00:40:43Z: 做时间窗文献调研要用 arXiv API 的 submittedDate 区间查询，不要用搜索页或 WebSearch。理由有三：区间查询是穷举而非按相关度截断，长尾不会丢；返回的是投稿时的结构化元数据（作者全名单、精确日期），可直接做在场缺席判定；单次可取 100 条并分页。本次两窗各 15 组关键词交叉覆盖后按 id 去重，单组召回 5 到 51 篇不等而合并唯一数达 244 与 209，说明**任何单一关键词的盲区都在一半以上**，交叉覆盖不是可选项。
L198 UTC 2026-08-05T00:40:43Z: 归纳大批文献时，先找那个「多数分支最终都指向的同一个问题」，再按它组织全文，比按方法名分类信息量高得多。本次 358 篇里六条主线最终都归到「稀疏终局奖励如何摊到 T 个去噪步」这一个中心问题上，于是 DenseGRPO、Tweedie 中间估计、注意力空间分配、只在早期步做、Adjoint Matching 绕开轨迹，这些看似无关的方法立刻排成了同一条谱系的不同解法。反之若按「GRPO 类 / DPO 类 / 蒸馏类」分，得到的是一张互不相干的清单。

L199 UTC 2026-08-05T00:51:55Z: 用户说出自己的理解时（本次「我的理解是要文字生成图片和视频」），不要只判对错，要用已有数据检验它的边界在哪。该理解抓住了产品入口却漏掉了重心偏移，而这个偏移直接改变下游判断：纯文生的奖励是单向的，编辑任务的奖励天生多约束互相冲突（既要改对又要没改坏别的），后者才是奖励建模专家的用武之地。做法上，手里已有 358 篇的结构化数据，重新打一次标签就能把「大致印象」变成「2:1」这样可核对的数字，成本几分钟。
L200 UTC 2026-08-05T00:51:55Z: 读消融实验要盯那个**中间行**。2602.04663 的表里，轨迹式+SDE 得 0.92，ELBO+ODE 得 0.96，若只看首尾会得出「ELBO 估计器带来 4 个点」；但中间的 ELBO+SDE 只有 0.90，比基准还低。真实结构是「似然估计器解锁采样器自由度」的因果链，三个因素不可独立相加。规则：凡是论文声称某个因素是主导因素，先找只改那一个因素的行；若该行提升不明显甚至倒退，说明主导的其实是它解锁的下游自由度，而不是它本身。

L201 UTC 2026-08-05T00:57:07Z: arXiv 的 HTML 版本不是每篇都有（2606.27608 的 /html/ 返回 404，只有 PDF）。取全文的备选顺序应为：arxiv.org/html/<id>v<n> → alphaXiv 解析 → HuggingFace papers → 下载 PDF 本地解析（本机 PyMuPDF 1.27.2 可用）。走了转述源就必须在产出里写明「未直读原文，细节经某源转述」，因为转述源可能漏掉或改写数值；本次即在页面附录单列了这一条。
L202 UTC 2026-08-05T00:57:07Z: 同一个问题上两家头部团队的相反选择，比任一家的绝对做法信息量更大。奖励模型训练范式上 WorldPM 用 BT 成对而 Qwen-Image-2.0-RL 用逐点回归，且后者明确报告逐点更好。差异可归因于**任务是否存在绝对标准**：文本偏好基本是相对的（哪个回答更好），而图像质量有绝对判据（有没有六根手指、结构塌没塌）。成对损失只保序、丢绝对刻度，在有绝对判据的任务上会让 RL 朝「比同组略好」而非「绝对好」优化。推广：选 BT 还是逐点，先问这个任务的「好」是不是可以脱离比较对象独立判定。

L203 UTC 2026-08-05T01:07:40Z: 求职语境下把自己的工作「翻译」成对方的关键词是负期望策略。招聘方筛的不是关键词匹配度而是「能不能解决我手上的问题」；关键词对上但一问细节就空，比一开始就说「我做的是 A，A 与你们的 B 共享这三个难点，我在 A 上解决过其中两个」要糟——后者展示的是迁移能力，那才是招资深研究员真正在看的。识别是否属于「硬拗」的判据：列出对方岗位每天要面对的三个具体问题，问自己是否真的遇到过；本例中 H1 每天面对的是偏好标注协议、多奖励打架、reward hacking，而可验证奖励的 world model 工作根本不会遇到这三个。
L204 UTC 2026-08-05T01:07:40Z: 主题分类要匹配标题而不是摘要。首次对 358 篇按「标题+摘要前 300 字」做正则分类，L5（hacking/多样性/熵）一组吃进 176 篇即近半数，因为 entropy、mode、diversity 这类词在任何 RL 论文的摘要里都会出现。改为只匹配标题后，最大组降到 95 篇且各组语义清晰。原理：标题是作者对「这篇最核心是什么」的自我声明，摘要则会提及所有相关概念，因此摘要匹配的假阳性率天然高得多。

L205 UTC 2026-08-05T01:44:40Z: 用户给出的文献标识符要先核再用。本次用户给的 arXiv id 2601.00796 实际是 AdaGaR（动态场景重建），与 VAR RL 毫无关系，正确 id 是 2601.02256。核验成本是一次 id_list 查询，而不核的代价是整页精读挂在错误的论文上。做法：凡引用中出现具体编号（arXiv id、commit hash、job id、页码），落笔前用一次廉价查询确认它指向的确实是所说的对象；发现不符时同时给出正确值与错误值指向什么，让对方知道自己是怎么记混的。
L206 UTC 2026-08-05T01:44:40Z: 「离散」与「似然不可解」是两个独立属性，绑在一起会得出错误的算法结论。dLLM 似然不可解的根因是解码顺序不固定（同一输出对应 |x|! 条路径，要对路径求和）；VAR 同样离散，但生成顺序严格固定（粗到细的尺度序），链式分解成立故似然精确，GRPO 可直接用而无需 ELBO 近似。判据是**生成路径是否唯一**，不是输出空间是否连续。推广：遇到新的生成模型要判断能不能直接上策略梯度，先问「给定最终输出，产生它的生成路径是否唯一」，而不是先问「输出是不是离散的」。

L207 UTC 2026-08-05T01:52:17Z: /find-session-id 的标准管线以 `ls -t $DIR/*.jsonl` 开头，这正是 Lustre 禁则里点名的 "ls + 时间排序，命中每个 inode" 反模式——为了拿一个本来就免费的值去 stat 整个目录。免费在哪：scratchpad 目录名就是当前 session UUID，系统提示里已给出。做法：需要**当前** session ID 时直接从 scratchpad 路径读，一次文件系统调用都不做；只有要检索**历史**会话（必须有锚点）时才碰目录列表，且用 `ls -lht ... | head -N` 限定条数而非全量排序。推广：任何「我要的值是不是已经在上下文里」的自查，应当排在「我用什么命令去取这个值」之前。

L207 UTC 2026-08-05T02:01:31Z: 判断一篇方法论文能否用到自己项目上，读代码比读论文有用得多，而且判断的性质会变。读论文只能得到「理论可行」，读代码能得到「改哪几个文件、哪些不用动、显存怎么变」。本次 RAM 论文摘要只说「不需要 SDE rollout 与奖励梯度」，而代码给出三件论文没说的关键事实：算法本体仅 46 行、目标里用的是 EMA 滞后副本而非当前策略、参考策略靠关闭 LoRA adapter 获得因而不需额外全量权重。第三条直接决定显存预算，是移植的真实约束。做法：拿到 GitHub 后先看 README 的 repository layout，再看最小的那个奖励实现（本次 ocr.py 仅 1.7KB）当接口模板，最后 grep 出 loss 函数本体。
L208 UTC 2026-08-05T02:01:31Z: 用户下达「以后所有回答都要在 notion 里」这类长期规则时，除了执行还要**同轮建立机制并写入记忆**，否则下一次会话就丢失。本次做法：写 feedback_all_answers_to_notion.md 并加 MEMORY.md 索引；同时建索引归档页明确分流规则（实质内容进主题专页、过程内容进工作记录页、索引进归档页）并列出全部悬置待确认项。机制比单次执行重要，因为规则的价值在于不用每次重申。

L209 UTC 2026-08-05T02:05:57Z: 当一个方法的最终形态与它所属的范式看起来完全不像时（RAM 是纯监督回归却自称 RL 后训练），正确的追问方式是分层定位而不是二选一判断。本次分成四层：RL 在目标函数里（KL 正则奖励最大化）、名字里的 Reinforce 指恒等式而非算法（log-derivative trick 用来消掉 ∇r）、policy gradient 被解析地写进 target（不动点方程 stop-gradient 后即回归）、以及代码里仍能看到的 RL 指纹（on-policy 采样、advantage、参考策略、滞后策略）。这个分层同时回答了「是不是 RL」和「为什么看起来不像」，比争论标签有用。
L210 UTC 2026-08-05T02:05:57Z: 读懂一个复合 target 的最快路径是找出其中每一项在原范式里的身份。RAM 的 target = v_ref + r·((ε−x₀) − v^θ) 初看是三个符号的堆砌，但一旦知道 (ε−x₀) 正是预训练的回归目标（由 v_t(x)=E[ε−X₀|X_t=x] 直接得出），整个式子立刻读成「预训练参考 + 奖励 ×（该样本的预训练目标 − 当前预测）」，语义自明。做法：对 target 里每个项，回去找它在预训练目标或基础定义里对应的是什么，而不是把它当作新引入的量。

L208 UTC 2026-08-05T02:10:27Z: 两条。**一、并行会话是可读的信息源，不只是冲突源。** 本轮 post-search 返回 100 条仍漏掉「求职策略」页（Notion 搜索索引有延迟，页面 02:00 建、02:00 搜不到），是读另一个会话刚建的索引页表格才发现它还在第 3 层。教训：当另一个 agent 在同一空间留下了结构性产物（索引页、清单页），把它当成一份独立的交叉校验源去读，比只信自己那次 API 快照更完整。**二、「取全量列表」的正确形态是让它落盘再 jq，而不是缩小 page_size。** post-search 100 条约 10 万字符必然超限，但超限时工具会把原始 JSON 存文件——此时 page_size 反而应该拉满，用 jq/python 在文件里做树重建，上下文里一个 token 都不花。缩小 page_size 换来的是「进了上下文但覆盖不全」，两头不讨好。

L211 UTC 2026-08-05T02:37:12Z: 会话检索里，**时间戳是免费旁证，选对输出格式就能让一次调用同时给出判据和佐证**。本轮用 ls -lhS --time-style=long-iso，size 负责判主会话、mtime 顺带对上记忆条目里记的日期（2026-08-03 attach 四坑），无需第二次调用去读文件内容确认。反面做法是先只取文件名、再回头补一次 stat 或 grep 内容——两次调用换来的信息，一个格式参数就给了。

L211 UTC 2026-08-05T02:41:15Z: 回答「用哪种奖励」这类选型问题，给谱系比给推荐有用，因为选型取决于对方手上有什么数据。本次给出四档（纯规则 / 规则+感知器 / 判别式 RM / 生成式 RM）并标注每档的两个决定性属性：要不要偏好标注、可不可解释。用户所在领域（订单簿）没有偏好标注，谱系右半边直接出局，选择立刻收敛到左半边，无需争论。做法：列谱系时每档都标出「前置条件」而不只是「效果」，前置条件才是筛选器。
L212 UTC 2026-08-05T02:41:15Z: 建议新手起步用可解释的奖励，真正的理由是**可解释性决定 reward hacking 能否被诊断**，不是实现简单。规则奖励下「模型把所有深度压到零也能拿满分」这种钻空子一眼可见并可立即加惩罚项；黑盒 RM 下只能观察到「分数涨了但样本变怪」，无法定位。RAM 论文自身的 OCR 实验即例证：拿到最高任务奖励但牺牲美学，正因 OCR 是规则奖励，该 trade-off 才能被清楚识别。

L213 UTC 2026-08-05T02:45:12Z: 「这篇论文用了几个 X」这类计数问题必须分层回答，因为同一个词在不同环节含义不同。RAM 的 reward model 分四层：训练奖励 3 个（且每实验仅 1 个）、validation 监控 2 个、评估指标 5 个、实现未用 4 个，代码实现总计 11 个。若只报一个数字，无论报 3 还是 11 都会误导。做法：回答计数前先问「这个数在哪个环节被使用」，按环节分层列出并给出总数。
L214 UTC 2026-08-05T02:45:12Z: 论文的实验配置文件是「论文没做什么」的最硬证据。RAM 三个 config 的 rewards 字段都是列表类型却只填了一个元素，这直接证明多奖励从未被实验验证过，比在正文里找「未提及」要可靠得多。推广：判断某能力是否真被验证，去读 configs 或脚本默认参数，而不是读正文声明；代码支持不等于实验验证过。

L215 UTC 2026-08-05T02:52:44Z: 判断一个方法能否迁移到新设定，做法是对其推导链**逐行标注依赖**，而不是整体判断能或不能。本次把 RAM 的七个推导步骤逐一问「这一步依赖什么、离散上还成立吗」，结果发现断点精确集中在三行（都是「离散空间没有 ∇_x」），而最关键的 Theorem 3.1 反而成立（因其本质是 Doob h-transform，对一般马氏过程都成立）。这种逐行体检同时给出「能不能」和「卡在哪、怎么补」，比结论式回答有用得多。
L216 UTC 2026-08-05T02:52:44Z: 策略梯度需要的是 likelihood **ratio** 而非 likelihood 本身，这个区分能推翻「似然不可解所以只能近似」的常见论断。2607.14522 指出掩码扩散模型在恰当参数化下 probability ratio 解析可解——分子分母的不可解配分项约掉了——因此存在不走 ELBO 近似的精确路径（LFPO 即以此命名）。**自我修正**：本工作区 H2 页此前论述「dLLM 只能取 ELBO 下界并踩符号陷阱」，该论述对基于 ELBO 的方法成立，但不应被读作唯一可能，已在 RAM 页第十四节标注。

L217 UTC 2026-08-05T03:10:11Z: 迁移一个方法到新设定时，若逐项类比推不通，换出发点而不是硬推。本次把 RAM 的不动点方程逐项翻译到 logits 空间得到多余的 p_θ(a) 因子，与已知最优解对不上，说明连续推导路径依赖 ∇_x 无法平移。改从「该问题的已知最优解」（soft RL 的 pi*∝pi_ref·exp(Q*/beta)）出发，在 logits 空间它本身就是回归目标形式，两行即得目标函数。规则：迁移方法时要迁移的是**设计原则**（本例五条：on-policy 端点采样、标量奖励、参考锚、解析生成训练状态、纯回归），不是推导步骤；推导是达成原则的一条路径，换设定就该换路径。
L218 UTC 2026-08-05T03:10:11Z: 「零优势应当不更新」这个自检必须真跑，不能默认成立。理论上 target=ref ⇒ loss=0 ⇒ 梯度 0，但浮点噪声加 Adam 的归一化更新会把它放大到 lr 量级并正反馈失控，实测漂移比 SGD 大 2.3e6 倍。推广到任何带参考模型的 RL 后训练：**训练前先用常数奖励跑几十步，看模型是否纹丝不动**；这个测试不需要任何真实奖励即可执行，且能把「接线错误」与「奖励设计问题」分开。判定要用 KL 这类确定性量，不要用 reward（有采样噪声，本例 lam=0 时 reward 波动 ±0.11 会淹没信号）。

L219 UTC 2026-08-05T03:26:23Z: 复用 allocation 做 attach 时，凡是以 SLURM_JOB_ID 命名的临时路径都会在多次 attach 之间被复用。普通 sbatch 每次拿到新 job id，这类路径天然唯一，所以上游脚本从不处理清理；attach 把这个隐含前提打破了。表现是第二次启动才失败、第一次好好的，很容易误判成「改动引入的问题」。规则：attach 前先做一次 preflight 清理，且清理**必须直接读 /proc/mounts 取挂载点**——死挂载点上 stat 和 glob 都会失败，任何先做存在性检查的清理写法都会静默跳过（本次第一版清理就是这么无声失败的，报告 unmounted=0 却仍剩 48 条挂载）。
L220 UTC 2026-08-05T03:26:23Z: 上游脚本里形如 VAR="" 的无条件初始化会吃掉外部 export，且 ${VAR:-默认} 的写法让它看起来「可覆盖」。本次 SQUASHFS_MULTI_MOUNT_ROOT 就是：node_wrapper.sh:370 写着 ${SQUASHFS_MULTI_MOUNT_ROOT:-...} 像是留了覆盖口，但 342 行先把它置空了，于是 :- 永远取默认值。判断一个环境变量能否覆盖，不能只看使用处的 :- 写法，要 grep 整个文件确认没有更早的无条件赋值。

L221 UTC 2026-08-05T11:05:00Z: 自我繁殖的占位链上做「热更新调用约定」是最危险的改动类型，因为链的传播是异步的：一次提交的效果可能 36 小时后才兑现，中间任何对脚本调用接口的修改都会造成新旧对撞。本次给接口加**必填参数**（--chain）尤其致命，飞行中的旧调用方不可能补上它，而新版又把「没带参数」解释成合法的 one-shot 而非错误——静默降级，events.jsonl 与 submissions.jsonl 里都查不到任何异常记录，只能看到「什么都没发生」。规则：(1) 改自续投脚本的调用约定时，新版必须对「无参数调用」保持向后兼容（默认续链），安全阀做成显式 --no-chain 加预算闸，而不是反过来；(2) 若必须加必填参数，新版应在检测到「环境变量 U6GB_4N_CHAIN_SEQ 已设置（说明我是被链提交的，不是人手提交的）但命令行没有 --chain」时记一条告警事件，让失配可见；(3) 排查链断点时，比对相邻两跳的**事件 schema 差异**比读代码更快定位版本错配——字段多寡（mode/poll_seconds）和事件命名（chain_successor_submitted vs a_submitted）是版本指纹。

L222 UTC 2026-08-06T03:18:08Z: 「让意图在命令行上看得见」这个设计目标，要逐个渠道验证而不是想当然。本次脚本注释列了三条留痕渠道，其中 scontrol show job 实测不成立（Slurm 的 Command 字段不含脚本参数），而这恰恰是事后排查时最顺手的那条——人查一个正在跑的作业，第一反应是 scontrol，不是去翻 events.jsonl。教训有二：(1) 凡是写进注释的可观测性承诺，落地时必须实跑一次确认，否则等于在事故现场埋一个假路标；(2) 真正可靠的自我披露是**进程自己写下的状态**（chain_started 的 mode 字段），而不是从外部命令反推的参数——前者是当事人供词，后者是旁证，且这里的旁证根本不存在。配套操作规则：提交这类带行为开关的作业时，一律经 record_submission.py 走账，因为 submissions.jsonl 的 argv 数组是唯一完整保留命令行的地方（断链那条 argv 两项、本次三项，一眼可辨）。

L221 UTC 2026-08-06T03:23:01Z: 用 grep 读一个脚本的 argparse 来决定怎么调它，会漏掉多行写法的参数。find_model_zoo_checkpoint.py 的 --architecture 是必填的，但它的 add_argument 跨了几行，我 grep "add_argument" 只匹到起始行、没带出参数名，于是调用时漏传。真正的代价不是报错本身而是它发生的时刻：编排器在训练结束的瞬间失败，人不在场，一个还剩 3 小时的 allocation 就这么空转到超时。规则：凡是脚本要在无人值守的链路里调用外部命令，先跑一次 --help 核对必填参数，不要用 grep 推断接口。同一轮里我对 inference.py 的 18 个参数逐个做了这种校验，却漏了这个只传三个参数的小脚本——越简单的调用越容易跳过校验，而无人值守链路上每个环节的失败代价是一样的。

L223 UTC 2026-08-06T03:35:00Z: 复用他人写的格式转换器（md → Notion blocks、md → LaTeX 之类）时，先干跑一次并打印**输出结构**，不要直接推到目标。本次 md_to_blocks 静默吞掉了 blockquote 语法：不报错、不警告，只是把 `> 规则：...` 当普通段落处理并保留 `>` 前缀，推上去才发现和页面既有的 callout 风格不一致。转换器的失败模式通常是「降级」而不是「报错」，因为它对不认识的语法只能当纯文本处理，这在语法层面是合法的。干跑要检查的是**能区分你关心的那两种情况的字段**（此处是 block 的 type，以及表格的 table_width 与各行 cell 数是否相等），而不是「有没有抛异常」。这条与本页第二节「验证脚本必须具备分辨力」同源：截断或粗看任何输出之前，先确认它还能区分你想区分的东西。
L224 UTC 2026-08-06T04:15:00Z: 对比"训出某 checkpoint 的代码"时，git commit 记录不可当唯一事实源：j3417629 metadata 记录 commit 3f6d32a6，但该 commit 的 encoding.py 还是 24tok 版；只因 ignore_times=False 让 reshape(…,MSG_LEN) 分支成死代码才没崩。正确做法是"metadata 的运行时参数 + 同日相邻 commit + 崩溃可行性推演"三方互证——本次靠 reshape 可行性推演找到同日 10e61e22 替换 commit，并证明两种代码态对该次训练数值等价。另一课：读 checkpoint metadata 比读 argparse 默认值可靠一个量级（A/B 两次训练的 ignore_times、local_steps_k、val_split 全部与"想当然的默认值"相反或不同）。

L1786031177 UTC 2026-08-06T15:46:17Z: [检索键选择] findings.md 的自增 ID(如 F1785791500)是比 SLURM job ID 更强的会话检索键。原因: job ID 会在多个会话中被反复讨论/引用, 而 findings ID 只在**写入它的那一次**会话里出现过一次。今次单键单次 grep 即唯一命中, 未进入 size 判别分支。推论: 凡是本项目自己生成的、写回磁盘的序列号(F###/PG###/protocol_id/sample_indices_sha256), 都优先于外部系统 ID 用作会话定位键。
L1786031177b UTC 2026-08-06T15:46:17Z: [时间戳陷阱] JSONL 的 mtime 与其中 findings 记录的逻辑时间戳可能相差数日。本例 findings 写的是 2026-08-03T22:45:00Z, 而文件 mtime 是 2026-08-06 15:26 —— 该会话是"被中断"而非"已归档"。检索时不要用引文里的日期去筛 mtime, 会漏掉仍在活动的会话。

L1786036542 UTC 2026-08-06T17:15:42Z: [squeue 快照会骗人] 用户给的 squeue 行显示 5924045 处于 R、已跑 15:55、剩 2:44:05, 据此会预期它还要两小时多。实际 sacct 显示它 16:23:34 就 COMPLETED, 全程 19m06s。原因: squeue 的 TIME_LEFT 是**时限减已用**, 不是**预计剩余**, 对提前完成的作业永远显示到时限为止的余量。判断作业是否结束必须走 sacct, 且"不在 squeue"≠成功——本例是 0:0 真成功, 但同样的表象也可能是 21 秒崩溃。
L1786036542b UTC 2026-08-06T17:15:42Z: [同池可比性要逐位举证, 不能靠命名] 两次评测都叫"GOOG Jan-2026 3136 序列"并不构成可比。真正的证据是 sample_indices 内容 sha256(a0cd27b5…)、indices 文件 sha256(0c41de51…)、benchmark_revision(1128d37c…)、dataset_length(226002)、cond/gen 行数与 seed 五项同时相等 —— 这些字段被 model_zoo 的 generation_complete.json 和本次的 inference_inventory.json 各自独立写下, 交叉核对才能宣称"同池"。凡是对照实验的结论, 先验可比性再谈分数差。
L1786036542c UTC 2026-08-06T17:15:42Z: [赢了也要把混杂因素摆出来] 本次三指标全面胜出且 21/21 特征无退化, 最省事的写法是"33.6M Mamba3 优于 8M 基线"。但基线同时差在参数量(4.2x)、数据多样性(8 tickers vs 1)、batch(0.5x)三处, 单因归因是错的。真正有信息量的是那条**反向**证据: 本模型只见了 53.3% 的 token 仍胜出, 这才把"训练更久"从候选里排除掉, 把解释空间收窄到两个。结论的价值来自排除了什么, 不是来自赢了多少。
L1786036542d UTC 2026-08-06T17:15:42Z: [基线的异常高值可能是 bug 不是能力差] mamba3-8M 在 log_inter_arrival_time 上 WS=0.6205, 是它 21 个特征里最差的一项, 而本模型 0.0597, 改善 -90.4% 冠绝全表。但该特征正是 2026-07-29 记录在案的 START token 经 NA_VAL=-9999 进入 delta_t_ns 的污染点(F1785351300)。把这 -90.4% 整个记作模型进步会高估收益; 引用单特征极端改善前, 先查该特征有没有已知的 pipeline 缺陷史。

L1786038266a UTC 2026-08-06T17:44:26Z: [同名指标先验口径, 再谈排名] exp_R1_Mamba3 里有两张都叫"LOB-Bench 分数"的表, 78M 一个写 0.0442 一个写 0.1460, 差 3.3x。差异不在数据错误, 而在两件事同时不同: 代际(2026-03 的正式 sweep vs 2026-04/05 的 IsoFLOP sweep, 训练预算差一个量级)与评法(取训练末档 vs 扫全部 checkpoint 取最优)。如果直接把两表并成一张 leaderboard 按 WS 排序, 会凭空造出一个"78M 比 78M 强 3 倍"的排名。规则: 合并任何两个来源的指标前, 先各自回到生成它的脚本/docstring 确认口径(本次是 extract_scaling_law_best.py 的第一行 docstring 给出了 WS-21 的完整定义: 全特征 wasserstein 点估计的算术平均, s500 + c250g250), 口径不同就分成两张榜并写明为什么不可比。
L1786038266b UTC 2026-08-06T17:44:26Z: [榜单里的非单调不是噪声, 是设计缺陷的信号] 榜 A 中 8M(WS 0.1014)优于 14M(0.1287)和 23M(0.1126), 看着像随机波动。但读到"% of 1 Epoch"列(2.3%-4.2% 不等)就清楚了: 这 6 个模型见的 token 量根本没对齐, 榜单同时在变容量和训练量。这类表做 leaderboard 没问题(它诚实回答"这些 checkpoint 谁强"), 但**绝不能**被读成 scaling law(那需要固定 D 扫 N)。凡是排名出现非单调, 先去列里找有没有第二个自变量在动, 而不是先归因为种子噪声 —— 与 F1786036542c 同一类错误的不同表现形式。

L1786038900 UTC 2026-08-06T17:55:00Z: [404 的正确处置是给出用户可执行的一步, 不是换通道重试] 本轮开局 MCP 取页面 404, 我的第一反应是换 REST token 重试 —— 但两条通道用的是同一个 integration, 注定同样失败, 这次重试是纯浪费。真正把问题钉死的是第三个调用: search 返回**空列表**(API 成功但一个页面都搜不到), 这才把"页面没共享"与"ID 错/token 失效"区分开。判据: 错误体里若回带了 integration_id 和 integration 名字, 说明认证已通过, 问题必在授权范围而非凭据, 此时任何换通道/换 token 的尝试都不会成功, 应立刻转为给用户一句可执行指令(页面 ··· → Connections → 添加 cc)。同时不阻塞: 本轮在等共享的同时把 leaderboard 全部备好, 用户回「重试notion」后一次写入完成。

L1786040400a UTC 2026-08-06T18:20:00Z: [模板的 except 分支可能把"缺配置"变成无限重启] gradio leaderboard 模板在导入期 snapshot_download 两个 dataset repo, 任何异常都走 restart_space()。OWNER 保持默认值时, 这不是"启动后没数据", 而是"永远起不来且看不到错误"——重启掩盖了报错本身。凡是接手他人模板, 先读它在**导入期**做了哪些外部 IO, 以及失败分支通向哪里; 失败分支若指向"重启/重试"而非"抛出", 就是一个会吃掉根因的设计。改造时把这类外部依赖整体去掉比配好它更省事, 因为本任务根本不需要远程 results repo。
L1786040400b UTC 2026-08-06T18:20:00Z: [git push 的失败原因在 remote 段, 不在最后几行] 本次 push 被 HF 的 pre-receive hook 拒, 我第一次用 tail -6 只看到 "! [remote rejected] main -> main (pre-receive hook declined)" 与 "error: failed to push some refs", 两行都不含根因; 真正的原因("short_description length must be less than or equal to 60 characters", 我写了 61 字符)在更靠前的 remote: 段里。规则: push/推送类命令失败时不要 tail, 要完整打印(必要时 sed 去掉 ANSI 转义), 因为远端的诊断信息在前、本地的结论在后, 而结论行通常没有信息量。同理 git commit 在新 clone 里会因缺 identity 直接失败, 主仓库的配置不会继承。
L1786040400c UTC 2026-08-06T18:20:00Z: [推上去 ≠ 跑起来, 且 RUNNING ≠ 跑的是你推的版本] push 成功后第一次查 stage 就是 RUNNING, 但那可能是旧容器仍在服务。判据必须是 space_info().sha == 本地 HEAD **且** stage==RUNNING, 再持续观察数分钟确认不会转成 RUNTIME_ERROR(app.py 的导入错误只在容器重建时才暴露)。另: 推之前本地 pip 装齐依赖、真实 import app 构建一次 Blocks, 比推上去等 build 反馈快一个数量级, 也顺带验证了 ColumnFilter 这类第三方 API 的签名假设。

L1786039620 UTC 2026-08-06T18:07:00Z: [检索键要挑"现场生成"的标识] 同一个实验里 SLURM job 5877859 与 W&B run 30nkkohd 都是"高选择性 ID", 但前者是占位链 allocation, 在提交/断链复盘/attach 决策等多个会话里被反复引用, 命中会散; 后者是训练进程真正起来那一刻才由 W&B 生成的随机串, 只可能出现在该进程的输出及其下游引用中。**判据不是"这个 ID 看起来多独特", 而是"它在什么时刻诞生、之后被谁引用过"**。事前就存在的 ID 天然会被到处提及。
L1786039620b UTC 2026-08-06T18:07:00Z: [memory frontmatter 的 originSessionId 是免费的溯源捷径] 每条 memory 文件都自带出生会话指针。若被问及的实验恰好写过 memory, 读 frontmatter 比 grep 全部 JSONL 更快、且完全不碰 Lustre; 本轮它还充当了 size 规则的独立交叉验证(两条互不相干路径同指 9ddee538), 零额外成本。检索前先想一想: 这件事有没有留下 memory。
L1786039620c UTC 2026-08-06T18:07:00Z: [skill 模板与 Lustre 铁律冲突时以后者为准] /find-session-id 的单管线模板含 `ls -t "$DIR"/*.jsonl | head -1` 用来定位当前会话, 这正是 CLAUDE.md 第 8.2 节禁止的 1 号反模式(ls + 时间排序, stat 目录内每个 inode)。替代: 当前 session ID 从 scratchpad 路径直读(它就在路径里), 搜索改 `grep -r --include='*.jsonl'` 让 grep 自己遍历而非 shell 展开 glob, ls -lhS 只喂 grep 命中的少数文件。功能等价, 元数据开销从"全目录"降到"3 个文件"。skill 是便利模板, 不是安全豁免。

L1786042800a UTC 2026-08-06T19:00:00Z: [部署验证要问"哪个容器在服务", 不是"仓库是什么版本"] 我用 space_info().sha 当作运行版本, 它只是仓库 HEAD; 真正在跑的版本在 get_space_runtime().raw["sha"]。两者在 Dev Mode 下会长期不一致, 而 stage 一直显示 RUNNING(旧容器在服务), 于是我得到了一个全绿但完全错误的验证结论, 直到用户贴出页面内容才发现。推论: 任何"已部署"的断言, 判据必须是**服务端正在执行的产物**(运行容器 sha, 或直接抓取渲染结果做新旧内容双向断言 —— 新内容 PRESENT 且旧内容 absent), 而不是任何"已上传/已入库"的间接信号。这条与 [[feedback_verify_before_done]] 同源, 但错在判据选得太靠前。
L1786042800b UTC 2026-08-06T19:00:00Z: [Dev Mode 会静默停掉自动部署] HF Space 开着 devMode 时 git push 不触发重新部署, 且没有任何提示 —— push 成功、stage RUNNING、sha 也"匹配"(如果查错字段), 一切正常只是页面不变。这是"看起来一切正常的失败"的典型。接手一个已存在的 Space, 第一件事应是读 get_space_runtime().raw 里的 devMode/hardware/sha 三个字段, 它们决定了后续所有部署行为。另: 用户 Notion 页上那条 ssh ...@ssh.hf.space 就是 devMode 的证据, 线索早就在页面上, 我当时把它当成无关的环境备注跳过了。
L1786042800c UTC 2026-08-06T19:00:00Z: [改配置文件时, 被删掉的字段和被改坏的字段一样危险] BUILD_ERROR 的根因是我重写 README 时没保留原 front matter 里的 sdk_version: 5.43.1 —— 我关注的是把 title/description/tags 写对, 没意识到那个字段在**约束依赖解析**。整体重写一个配置文件时, 应先逐字段比对旧版, 对每个不打算保留的字段说清"为什么可以没有"; 尤其是自己不认识用途的字段, 默认保留而不是默认丢弃。
L1786042800d UTC 2026-08-06T19:00:00Z: [无异常的 RUNTIME_ERROR 要往运行环境找, 不要往代码找] 第三次失败时 run log 干净得反常: app 正常 bind 7860, 没有一行 traceback, 但 stage 是 RUNTIME_ERROR。这种"代码没报错但平台判失败"几乎一定是运行环境的契约问题, 本次是 ZeroGPU 镜像(强制 spaces==0.51.1 + torch==2.11.0)按自己的约定做健康检查。判断依据: 一个渲染静态表格的页面**根本不需要 GPU**, 硬件选错本身就是多余约束的来源。遇到日志干净的平台级失败, 先问"这个运行环境对我提出了哪些我没满足的要求", 再问"我的代码哪里错了"。

L1786045200a UTC 2026-08-06T19:40:00Z: [用户要求加一列, 可能会掀出结论级的错误] 用户只是要"stocks 和 years 两列", 属于展示层需求。但为了填这两列必须逐 run 去读 W&B config, 而这一读发现榜 B 是 488 票、榜 A 是 8 票 —— 我此前两次(findings F1786038266b 与 Space 首版 About)都把两榜不可比归因为"评法+预算"两条, 漏了最根本的训练集差异。教训: **凡是"照抄文档得到的事实", 在第一次有机会逐条核实时就要核实**。scaling_law_runs.md 的 Common Config 写着 8 tickers, 我把它当作整个 exp_R1_Mamba3 的口径, 而它其实只描述了榜 A 那批; 榜 B 用的是另一个脚本(scaling_law_sweep.sh)的默认 S&P500 成分文件。同一个实验目录下的两代 sweep, 配置文档不会自动跟着更新。
L1786045200b UTC 2026-08-06T19:40:00Z: [数据全对但图是错的] evolution 面板第一版所有数值都正确、散点也都落在正确的时间位置, 但图不可用: 两批相隔六周而各自压缩在一天内, 导致 94.5% 的横轴是空白、每个 board 塌成一条竖线。**"数据正确"不等于"图正确"**, 图还要对得起它占用的空间。判断方法是先算数据在目标坐标系里的占比(本次: 3.5% 与 2.0%)再决定版式, 而不是画完看一眼。另: 因 kaleido 需要 Chrome 无法在 login 节点导出静态图, 我改用"算分布"替代"看图", 这个替代在本次是充分的 —— 决定版式的量本来就是数字而非观感。
L1786045200c UTC 2026-08-06T19:40:00Z: [缺失就留空, 不要推测填充] j4559297(350M)的 W&B run 不在已索引的三个 project 中, 它的 start/end/stocks/years 全部留 null, UI 显示为空。可选的"合理推测"有很多(用同批其他 job 的日期、用 manifest 的 submit_time、用 488), 每一个看上去都无害, 但 leaderboard 的价值全在于每个数字都能追到出处。宁可一行有洞, 不可全表可疑。同理 build_data.py 里 enrich() 对缺失 run 显式写入全 None 而非跳过, 使"这行没有数据"成为一个被记录的事实而不是一次静默省略。

L1786041871 UTC 2026-08-06T18:44:31Z: [order-book 重建] 一个被反复重新发现的结论如果只以「缺陷编号」形式存在, 下一次遇到同一物理问题的人不会命中它。D-I3/D-O1 在 findings.md 里有完整机制与数字, 但按「切片重建不出 order book」去搜 memory 是零命中——缺陷编号是修复期的索引键, 不是问题期的索引键。规则: 凡是「与直觉相反且会改变设计决策」的实测判决, 除了入缺陷册, 必须另建一条以**问题本身**为标题的 memory。

L222 UTC 2026-08-06T19:00:42Z: 跨管线比较指标前，先确认两个数字是不是同一把尺子量的。R1 Mamba3 自己就有两张都叫「LOB-Bench 分数」的表，同一个 78M 模型报 0.0442 和 0.1460，差 3.3 倍，因为一张取训练末档、一张扫全部 checkpoint 取最优。把它们和本管线的 WS-21 放在一起排序会造出一个不存在的排名。可判定的做法是找同池同管线的对照（这里是 model_zoo 的 8M 系列，五项 sha256 交叉核对过评测池），以及跟自己的前一版比。
L223 UTC 2026-08-06T19:00:42Z: 子进程抛出的自定义异常会在跨进程回传时被「找不到模块」掩盖。PyTorch DataLoader 的 worker 抛 LosslessEncodingError，主进程 pin_memory 线程 unpickle 时没有那个模块，报出来的是 ModuleNotFoundError，而真正的编码失败一个字都看不到。凡是 worker 里调用外部库、且该库不在主进程 import 路径上的，都要在 worker 内部就地 catch 并转成标准异常类型再抛。

L224 UTC 2026-08-06T19:07:40Z: 同一个机制在两种环境下会给出「优点」和「事故」两种表现，判断它时必须带上环境参数。本次「启动即提交」在排队 > 运行时长时表现为"链能续上但有缺口"，在集群空闲、排队归零时表现为"每分钟净增一条链、两小时 97 条"。用户记忆里的"自动就排上了"和事故记录里的"护栏窗口归零"指的是同一批作业。教训：复盘一个自动化机制的行为时，先问「当时的外部条件是什么」，不要把某次观察到的表现当成该机制的固有属性。配套判据：凡机制的效果依赖某个外部量（此处是 T_q 排队时长），就该把该量的实测值连同结论一起记，否则结论在环境变化后会悄悄失效——本例 T_q 从 20h 涨到 36h 的过程中，"链是连续的"这个结论早就不成立了，但没人重新量过。

L1786043276 UTC 2026-08-06T19:07:56Z: [issue 的价值在开口不在结论] 第一版草稿差点写成纯知识陈述(把 memory 内容搬过去)。一个只列已判决结论的 issue 没有关闭条件, 等于文档放错了地方。改法是强制回答「这条 issue 要做什么才能关」, 于是逼出了 §5.1 那个此前没人注意到的风险——收敛表是 GS 单票。**写验收标准这个动作本身就是探测器**: 写不出可勾选的项, 说明手里的是文档不是 issue。

L225 UTC 2026-08-06T19:09:16Z: 验证一个修复时，要抓的是**修复点自己产生的信号**，而不是最终效果。本次判据是 chain_started 的 mode 字段而非「后继有没有出现」：因为后继没出现可能是 A0(未修好)、也可能是 A1/A3/A4(修好了但被别的闸拦)，两者在「squeue 里没有后继」这个现象上完全一样。实测正是后者——mode=chain 证明 A0 已通过，随后 6 轮 a_skip_over_budget 证明是 A4 在拦，30 分钟后自行解除。若当时只看 squeue，会在那 30 分钟里误判成「修复没生效」并去改代码。规则：分层的判断链里，每一层都要有自己的可观测输出，验证时定位到具体是哪一层拦的，不要用末端现象反推。

L1786043446 UTC 2026-08-06T19:10:46Z: [制品语言 vs 对话语言要分开判] 用户一句「in english !」有两种读法。判据: 全局 CLAUDE.md 显式钉死对话用中文, 而 GitHub issue 是面向协作者的**外部制品**, 英文是其默认。故改制品不改对话。副则: 转语言用 gh issue edit 原地覆盖而非新建 issue, URL 是别人已经可能引用的标识符。另: 本轮核验脚本报 MISS Acceptance 系探针串大小写写错(实际是 "Proposed acceptance criteria"), 不是内容缺失——核验探针本身也会出假阳性, 报警后要回看探针再回看内容。

L1786050000a UTC 2026-08-06T21:00:00Z: [判定"用了哪些指标"要看注册表, 不是看有没有函数] 用户问第三个指标是不是 KL。若只 grep 函数定义, lob_bench/metrics.py 里 kl_divergence_kde 与 kl_divergence_PerezCruz 赫然在列, 会得出"有 KL"的错误结论; 若只看输出的 pkl 键, 又会得出"库里没有 KL"的另一个错误结论。真相是**实现了四个、注册了三个**, 只有 run_bench.py 的 DEFAULT_METRICS 字典能定案。推广: 判断一个系统"实际在做什么", 要找那个把能力接进流程的**接线点**(注册表/配置/路由表), 而不是能力本身是否存在。同一逻辑也适用于回答"这个 flag 生效吗""这个 hook 跑了吗"。附带价值: 用户"记错"的地方往往有真实出处, 查清出处比简单纠正更有用。
L1786050000b UTC 2026-08-06T21:00:00Z: [界面结构对齐数据结构, 版式难题会自行消失] 上一轮我为了把两批数据画进一张 evolution 图, 花力气论证并实现了并排双子图(因为单轴 94.5% 是空白)。这一轮用户要求改成按 dataset 切换后, 每次只画一批, 双子图直接变回单图, 那个精心解决的问题**根本不再存在**。教训: 遇到版式难题时, 先问"我是不是在同一个视图里塞了两个本不该并列的东西", 而不是先去优化那个视图。本例中"8 stocks vs 488 stocks"从一开始就是两个不可比的对象, 把它们并列本身才是问题的根源, 双子图只是给错误的并列做了个体面的妥协。
