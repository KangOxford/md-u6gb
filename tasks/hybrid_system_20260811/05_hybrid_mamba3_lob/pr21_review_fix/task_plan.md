# PR#21 review 修复执行计划

> **目标一句话**：把 alexandre 的 review 指出的问题**真正修掉**（不只是回应），每完成一个
> 里程碑在 PR#21 下发英文 progress comment；9/10 前交出训练矩阵与最终报告。
> 依据：2026-08-26 三条回应评论（#issuecomment-5419896076 / -6949 / -7611）里的公开承诺。
> 计划文件本身：本目录 `task_plan.md` / `findings.md` / `progress.md`（中文）。

## 状态总览

| 阶段 | 内容 | 依赖 | 状态 |
|---|---|---|---|
| P0 | 计划文件 + 工作区准备 | — | ✅ complete |
| P1 | CI 系列落 main（A1） | P0 | ✅ **complete**：PR#35+#36 已合并，main 四关卡全真绿；两份过期 evidence 重录；稳定输入家建立 |
| P2 | PR#21 机械修复（A2/A3/A4 + C3/C4/C5 + minors + reshard 出列） | P0 | ✅ **complete**（ece14dc 已推送，合并树 295/0/8；PR checks 等确认；唯一已知红 = injection.json 待 GPU 重录） |
| P4 | PR body 重写（D1/D2 reword + token 表 + B3 张力句 + 23 files + 内存表行） | 与 P1/P2 并行 | ✅ **complete**（已 gh pr edit） |
| P3 | reshard 独立 PR（C1 集体协商修复 + C2 round-trip 证据 + 脚本入库 + 模式校验） | 代码部分随时；round-trip 等 GPU | pending |
| P5 | 八点逐格 SHA 表（B2，login CPU 从 bench 日志重建） | 与其余并行 | pending |
| P6 | 训练矩阵 6 组 + ask2/ask3 评测 | **等卡**（当前 32/32 忙，hyper-recipe ~18h 到期） | pending |
| P7 | 全绿后请求 re-review | P1–P6 | pending |

```
P0 ─┬─ P1 CI 系列（main，新 worktree）────────────┐
    ├─ P2 PR#21 机械修复（PR 分支 worktree）────────┤
    ├─ P4 PR body 重写（gh pr edit，无代码依赖）───┼── P7 请求 re-review
    ├─ P3 reshard 新 PR ── C2 round-trip（等 GPU）─┤
    ├─ P5 SHA 表（login CPU 日志考古）─────────────┤
    └─ P6 训练矩阵（等 GPU 档期，gtop-first）───────┘
```

**排序理由**：reviewer 原话 "Happy to re-review once A1 to A4 are done"，所以 P1+P2 最优先；
P6 被卡挡着，正好把机械修复做完。

## 阶段细节与验收标准

### P1 —— CI 系列（A1，落 main 的独立 PR）

| # | 步骤 | 说明 |
|---|---|---|
| 1 | 新 worktree | 从 `origin/main` 建分支 `ci/pipefail-per-workflow`，worktree 放 s5e 空间（u6gb inode 有打满前科）：`/lus/lfs1aip2/projects/s5e/quant_team/quant/sigma0_ci_wt` |
| 2 | **先**修 english-only 自扫 | 读 `ci/check_english_only.py` 弄清它为何扫到自己和 workflow 文件；优先「检测字符集改 `\uXXXX` 转义 + known-list 外移数据文件」，退路是自豁免。**顺序必须在 pipefail 之前**，否则一开全红 |
| 3 | 四个 workflow 加 pipefail | 统一 `defaults: run: shell: bash`（= `-eo pipefail`，PR#28 `lobbench_merge.yml` 已验证的写法）；english_only 1 处、evidence 1 处、injection 5 处、return_bench 6 处 `\| tee` |
| 4 | 补 `--self-test` | 用户既有令「每个 PR 自带 check」+ 验收三条证据（shell 行 / 打印 / caught）。english_only 与 evidence 两个 checker 必须有；injection/return_bench 已有的不重造 |
| 5 | 模拟 runner 验证 | 干净 checkout + `bash --noprofile --norc -eo pipefail` 逐步重放每个 workflow（memory：在非开发环境跑一次，上回抓出两个幽灵文件） |
| 6 | 既有漂移重录 | `pick_trade_anchors.py` / `mm_sim.py` 漂移（reviewer 确认是 main 既有）：login CPU 能重录就一并；要 GPU 则记到 P3 |
| 7 | 英文 commit + 开 PR | commit / PR 全英文；开完在 PR#21 发短评。**合并键在 Kang 手里**——合并前 P2/P3 不算被真正 gate |

**验收**：模拟 runner 下人为破坏（塞中文 tracked 文件 / 改坏 evidence json）→ 对应步骤退出码非零；
改过的 checker `--self-test` 全 caught；干净树上 english-only 报 0（自扫已消）。

### P2 —— PR#21 机械修复（A2/A3/A4 + C3/C4/C5 + minors）

| # | 步骤 | 说明 |
|---|---|---|
| 1 | worktree 检出 PR 分支 | `feat/hybrid-mamba3-nemotron-20260811`；先 `git worktree list` 查是否已有检出 |
| 2 | **reshard 出列（ask 5）** | `git revert d4d9ad9`（恰是 head 提交）。选 revert 不选 rebase-drop：保留 review 锚点与时间线，最终 diff 同样干净；reshard 在 P3 重新实现而非 cherry-pick（C1 本来就要重写降级链） |
| 3 | A3 + A2 | 两个测试文件搬 `tests/unit/`，删 `src/s5/tests/`；`test_model_registry` 六元组加第七项 `hybrid_mamba3`（保持精确断言） |
| 4 | A4 译英 | revert 后重数三文件剩余中文行（reshard 大注释随 revert 消失，17aa51c 的 lr 注释等仍在），全部译英，内容不减 |
| 5 | C3 stamp | `build_backbone` 把解析出的 attention 位置写回 args → 存进 checkpoint；load 优先读 stamp，只有 fresh build 才重推导 |
| 6 | C4 构造测试 | `tests/unit/test_hybrid_construction.py`：n_layers=12 → (3,9)；递归位 carry 是 mamba3 state tuple；注意力位是请求长度的 KV |
| 7 | C5 + minors | defect register 补四条（PE mismatch / KV cache / grad-norm crash / reshard deadlock）；`has_attention` 上 `BackboneDefinition` 替掉 `HYBRID_ARCHITECTURES` 字面集合；删死存储 `_attn_at`；`locals().get` 改显式 `None` 初始化（train_helpers.py:1282-1285）；`LOG_EVERY`/`HYBRID_ATTN_*` 写文档 |
| 8 | 全量 pytest + suite.json 重录 | login 节点 CPU 允许（<30min/<16G）；按 `ci/record_evidence.py` 机制重录；预期 290→298+ 全绿 |
| 9 | push + comment | 普通 push（revert 不改历史）；PR#21 发 "A2–A4, C3–C5 done at <sha>, suite N passed" 短评 |

**验收**：`pytest` 全绿且收集数 ≥298；`grep -P '[\x{4e00}-\x{9fff}]'` 三文件归零；
构造测试单独跑过；PR diff 里无 reshard 痕迹。

### P3 —— reshard 独立 PR（C1/C2）

| # | 步骤 | 说明 |
|---|---|---|
| 1 | main 新分支 `fix/ckpt-reshard-collective` | **重新实现**而非 cherry-pick：模式概念（none/host/broadcast）+ C1 修复 |
| 2 | C1 集体协商 | 每次 save 包一层：本地尝试 → **所有进程无条件进入**一次小 all-gather（状态旗）→ 共同决定降级与重试；异常过滤收窄回 `(OSError, ValueError)`，其余照常 raise；注释写明「挂死不 raise，本链只管本地写错误类」 |
| 3 | 模式校验 + 文档 | `CKPT_RESHARD_MODE` 非三值即 fail loud（现在拼错静默走 host 路径）；env 文档 |
| 4 | 脚本入库 | `check_replica_groups.py` 从 md-u6gb 拷入 `tools/diagnostics/`（登录节点纯 CPU 可复核的那半证据） |
| 5 | 决策逻辑单测 | 降级决策抽成纯函数（(local_ok_flags, cur_mode) → next_mode/action），不用 GPU 就能测穷举 |
| 6 | **C2 round-trip（等 GPU）** | 4 节点 30 分钟档：`none` 模式真实多主机 save → resume 成功一次，日志与步骤号写进 PR 正文。**没有这条证据 PR 不请求合并** |
| 7 | injection.json 集群重录 | PR#21 改了它的两个 source；跟 round-trip 同一次 GPU 档期做掉 |

**验收**：纯函数单测穷举全过；round-trip 日志显示 save+resume 步号衔接；PR 正文带全部证据。

### P4 —— PR body 重写（可立即做）

D1 reword（按 reviewer 建议句式 + registry.py:332 精度注）；D2 改 SSD intermediates；
"16 files, +728/−29" → "15 files, +728/−33"；内存表 3k/4k 行按 35.4 GB/1k 斜率重算（≈106/141）；
新增 token 恒等表（26.624B，逐组）；新增 B3 张力句（分布拟合上现证据跟参数走、与 recall 反向）。
方式：`gh pr edit 21 --body-file`，改前存旧 body 备份。验收：新 body 无 protocol 一词、全英文。

### P5 —— 八点逐格 SHA 表（B2）

从 bench job 日志（md-u6gb `05_hybrid_mamba3_lob` 的 results/logs + Slurm sacct）逐格重建
`(组, 步数点, bench job id, 日期, worktree SHA)`；查不到 SHA 的格标记待重测。
产物：results 新文件 + PR#21 comment 附表。已知边界：两修复 08-12 12:38/20:57 UTC 落地，
最早被 bench 的 ~4.5k 步检查点 08-13 才存在。

### P6 —— 训练矩阵与评测（等 GPU）

回应评论 3/3 的六组矩阵 + ask2/ask3 评测，另记 `run_matrix.md` 排期（起跑时建）。
规则：每次动手前 gtop+PENDING 两侧同查；attach-first；提交走 submit-job 技能；
**基准-500 vs 基准-2k 的 token 对齐点评测不用等新训练**（两个检查点都在），第一批卡空出来就先跑它。

### P7 —— 收尾

P1–P5 全绿 + P6 至少中期计分后，PR#21 发 comment 请求 re-review（对照他一条条给证据链接）。

## comment 节奏（用户令：解决完写 comment）

| 时机 | 内容 |
|---|---|
| P1 PR 开出 | 短评：A1 done + PR 链接 + self-test caught 证据 |
| P2 push 后 | 短评：A2–A4/C3–C5/minors + sha + pytest 计数 |
| P3 PR 开出（证据齐） | 短评：C1/C2 done + round-trip 日志摘录 |
| P4 body 更新 | 并入 P2 短评或单发一句 |
| P5 表完成 | 短评附表 |
| P6 各里程碑 | scoreboard 短评 |

原则：每条 comment 带可核验证据（SHA / 退出码 / 测试计数 / 链接），不发空话；全英文，无 protocol。

## 关键决策记录

| 决策 | 备选 | 选择与理由 |
|---|---|---|
| reshard 出列方式 | rebase-drop + force-push / revert | **revert**：保 review 锚点；diff 同样干净；C1 要重写，cherry-pick 无意义 |
| 工作区 | 动共用检出 / 新 worktree | **worktree**（s5e 空间）：共用检出在 main 落后 42 领先 1，不碰；u6gb inode 有前科 |
| C1 修法 | 砍掉降级链 fail-loud / 集体协商 | **集体协商**：观测到的失败类正是单主机 OSError（quota），链有真实价值；程序错误类退回 raise |
| P1 合并 | 我合 / Kang 合 | **Kang 合**：共享仓 main，外向动作留给用户按键 |
| 评测抢跑 | 全等新训练 / token 对齐点先跑 | **先跑 ask2 的 token 对齐点**：两检查点已存在，第一批空卡即可做 |

## Errors Encountered

| Error | Attempt | Resolution |
|---|---|---|
| md-u6gb 提交被拒：`tasks/` 在 .gitignore | 1 | 历史文件本就是强加的 → `git add -f`，成功 |
| Edit 工具把 `\uXXXX` 解码成真字符：写「转义版」正则实际写入字面 CJK，注释与实现互相矛盾 | 2 | 参数传输层解码 `\u`（`\n` 不受影响）。改用 Bash heredoc + `chr()` 程序化替换，替换后当场跑 checker 验证。**教训：涉及字节层的编辑，改完必须用检测器实测，不能信编辑成功** |
| u6gb conda 缺 `chex`（s5e env 无权限） | 1 | scratchpad 建 `--system-site-packages` venv 叠加层，不动用户环境 |
| simulator 重录 pthread EAGAIN：沙箱 cgroup 线程上限，XLA 按 288 核开池被拒；限线程 env 全无效（tsl 池不理会） | 2 | 转 claude-login43 tmux（沙箱外、正常用户限额、单次有界命令）执行 |
| tmux send-keys 忘带 `cd`，**连犯三次**（相对路径在错误 cwd 解析） | 3 | 第三次才用 `CMD="cd … && …"` 变量构造 + 发送后 capture-pane 验证。教训：重发失败命令前先 diff 自己到底改了什么 |
| return-bench 重录：记录的输入目录 `data/hp_rn` 已被 0817 prepurge 清除 | 1 | 同输入重录不可能；换基线 = 数据语义决策，升级给 Kang（PR 里给两个选项：指定替代目录 `--accept-new-inputs` 重录，或接受该检查在 main 上如实变红直到 P6 产出新推理目录）。**用户裁定选项 A + 稳定家**，已执行 |
| 套件收集期 `ModuleNotFoundError: torchvision` | 1 | u6gb env 缺件（录制环境是 s5e 那套）；overlay venv 补装 torchvision 0.28，不动用户环境 |
| 三个 mamba3 等价测试 setup 炸：`jnp.clip() got an unexpected keyword argument 'a_min'` | 2 | 初判 WIP 所致（stash 隔离后仍败，判断被推翻）；真因是 jax 0.11 移除 numpy 风格 kwargs，录制环境 jax0.9 掩盖了地雷。改位置参数写法（新旧 jax 语义相同），18/18 过。**教训：怀疑环境版本前先隔离掉最显眼的嫌疑人，但别停在第一个嫌疑人** |
| 合并 main 冲突：`ci/evidence/register.json` 两侧都重录过 | 1 | 生成物不做文本合并——合并树上 `--register` 重录即是解 |
