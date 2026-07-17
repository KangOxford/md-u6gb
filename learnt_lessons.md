# Learnt Lessons

## 2026-07-05 Notion table answer color correction

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
L060 UTC 2026-07-17T12:16:12Z: 教训——不要把 state.json.batch_script 字段(它只是 logger 每日读取的 'command provenance',见 plans.md 第 3 行)当成 job 实际执行的命令。判断一个 SLURM job 是真训练还是空占位,先查两个信号:(1) sacct step 名(python/run_train/node_wrapper vs 裸 bash/sleep);(2) StdOut 文件大小(真训练 MB 级 vs 空转 0 字节)。本轮我据 batch_script 字段误报 5678750 '正在跑 R1_Mamba3 训练、16 节点 x 4 GPU 全在算',被用户当场纠正。断言 job 活动状态前必须先看这两项。
