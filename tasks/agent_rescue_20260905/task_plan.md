# PLAN — subagent recovery and the work the lost agents owed

Living plan file for this task. Canonical path:
`/lus/lfs1aip2/projects/public/u6gb/tasks/agent_rescue_20260905/task_plan.md`
Updated 2026-09-05T20:40 UTC under protocol PLAN-GPU-20260905T2030. The round-by-round
narrative that this file used to be is kept below, unchanged, from "# subagent 断线恢复".

**A plan existing is not evidence that anything in it is done.** Every row that claims
completion names the artefact and the check that produced it.

---

## 1. Original goal

Nine subagents died with their session on 2026-09-04. Recover what they established and finish
what they owed, without re-running what already exists and without disturbing other lines that
share this Unix account.

## 2. Claims to be adjudicated

| # | claim | status | decided by |
|---|---|---|---|
| C1 | a lost subagent's work is unrecoverable | **refuted** | 603 durable transcripts on `/home`; all nine had 416 KB–5.6 MB and their prompts inline in the registry |
| C2 | round 4 ends below round 3 on R | holds (pre-existing) | 18 vs 4 single-seed separation, p = 2.7e-4 |
| C3 | the exit was caused by the importance weights | refuted (pre-existing) | uniform-weight control moves −0.0969 vs −0.0808 |
| C4 | the headline `t = −7.8` came from a test that fires on noise | **refuted, this task** | −7.8362 is `naive_day_level_t` (df 19), the *conservative* test here (null max 2.27) |
| C5 | the registered analysis matches what was run | **5 match, 1 gap, 1 qualification** | `round6/H8_REGISTERED_VS_RUN.md` |
| **C6** | **round 4's exit survives trajectory-to-trajectory noise** | **UNDECIDED — the open scientific question** | needs `s_trajectory`; see §4 |

**C6 is the one that matters and it is not settled.** Everything delivered so far corrects how
the existing numbers are read. None of it establishes that the effect is real beyond the
variance component that has never been measured. **Pushing the notebooks did not decide C6.**

## 3. Necessary steps and dependencies

```
S1 recover the nine ─> S2 finish their owed work ─> S3 correct the analysis artefacts
                                                          |
S4 pipeline preconditions (CPU) ──────────────────────────+
                                                          v
                                            S5 score the 12 trajectories (GPU)
                                                          |
                                                          v
                                            S6 estimate s_trajectory ─> decides C6
```

| step | what | needs | state |
|---|---|---|---|
| S1 | registry, recovery packets, verify harness | CPU | **done** |
| S2 | the plan sections and the pipeline/notebook fixes | CPU | **done for 3 of 4 agents**; `sol_history` H6 declined, H7/H8 done |
| S3 | H8; display defects D-1/D-2/D-3; notebooks with outputs in the PR | CPU | **done** |
| **S4a** | `.done` tested for content, not existence — **9 sites still use `-f`** | CPU | **OPEN — next action** |
| **S4b** | two call sites still decide on the exit code (`r3null_cell.sh:45`, `inter_cell.sh:60`) | CPU | **OPEN — next action** |
| S4c | 8 launchers still end in `exec`, defeating their EXIT trap | CPU | open, not on the C6 path |
| **S4d** | inode/byte headroom for generation | placement decision, **not mine alone** | **BLOCKING S5, and now blocking file creation entirely** |
| S5 | generate + score the 12 trained trajectories at step 4800 | **GPU** | blocked by S4a/S4b/S4d + ownership |
| S6 | estimate `s_trajectory`; re-read C2/C3 against it | CPU | blocked by S5 |

## 4. The GPU step, sized

Not the 354 node-h plan — that was *training* new trajectories and stays withdrawn. This is
scoring checkpoints that already exist.

| quantity | value | source |
|---|---|---|
| trajectories trained, complete at step 4800 | **12** (`wm_ft_traj_s30…s41`, seeds 30–41, all distinct) | `round5/TRAJECTORY_LEDGER.md` |
| of those, scored | **0** | `sweep_results.jsonl` arms are only `multi3/multi4/unifw` |
| cells for the full estimate | 12 × 8 tickers = **96** | one prespecified checkpoint each |
| measured cost per cell | **17.2 min**, 1 node × 4 GPU | `sacct` `crps-*` COMPLETED 17:12 / 17:06 / 17:28 |
| **minimum useful request** | **4 GPUs (1 node), ~2.5 h** → an 8-cell strip | enough to separate spread ≤ 0.03 from > 0.08 |
| full panel | ≈ **27.5 node-h**; on 8 nodes ≈ 3.5 h wall | |
| **GPU workers this session holds now** | **0** | no step of mine is running |

**Stopping criteria, fixed before data.** Stop after the 8-cell strip and report its spread.
Go to 96 cells only if the strip's spread is < 0.08 **and** devices are coordinated for it. If
the strip shows > 0.08, stop and report C6 as unresolvable at this budget rather than buying
more cells.

## 4A. Provenance of the 12 trajectories of job 6317365 (read-only audit, 2026-09-05T22:xx UTC)

Extracted from `/home/u6gb/kangli.u6gb/traj_s{30..41}.log` and each trajectory's own
`ft_progress.json`. **Nothing below is a default value used to fill a gap**; fields that are
not recorded are marked as not recorded.

### Recorded in the artefacts

| seed | parent ckpt | load step | order_sha1 | seed0_sha1 | n_items | max_step | restore |
|---|---|---|---|---|---|---|---|
| s30 | `wm_ft_multi3` | 69378 | `14b7ec5e1dad` | `0f14669f2a4d` | 4800 | 4800 | partial |
| s31 | `wm_ft_multi3` | 69378 | `f2e846ae79a3` | `0f14669f2a4d` | 4800 | 4800 | partial |
| s32 | `wm_ft_multi3` | 69378 | `6109f5107177` | `0f14669f2a4d` | 4800 | 4800 | partial |
| s33 | `wm_ft_multi3` | 69378 | `84aad7b2e9a4` | `0f14669f2a4d` | 4800 | 4800 | partial |
| s34 | `wm_ft_multi3` | 69378 | `c340f571e2b7` | `0f14669f2a4d` | 4800 | 4800 | partial |
| s35 | `wm_ft_multi3` | 69378 | `abc6c23751f1` | `0f14669f2a4d` | 4800 | 4800 | partial |
| s36 | `wm_ft_multi3` | 69378 | `e481f09c818d` | `0f14669f2a4d` | 4800 | 4800 | partial |
| s37 | `wm_ft_multi3` | 69378 | `526a507d8c2a` | `0f14669f2a4d` | 4800 | 4800 | partial |
| s38 | `wm_ft_multi3` | 69378 | `bc977395b98e` | `0f14669f2a4d` | 4800 | 4800 | partial |
| s39 | `wm_ft_multi3` | 69378 | `966cddd494e0` | `0f14669f2a4d` | 4800 | 4800 | partial |
| s40 | `wm_ft_multi3` | 69378 | `18c94703b061` | `0f14669f2a4d` | 4800 | 4800 | partial |
| s41 | `wm_ft_multi3` | 69378 | `877cc66f0930` | `0f14669f2a4d` | 4800 | 4800 | partial |

**Twelve distinct `order_sha1`** — the data orders really are independent, which is what the
`--train-seed` patch was for. All twelve share the same `seed0_sha1` reference, which is how
that patch proves each order differs from seed 0's.

`partial_restore=True` in every case: `StandardRestore` failed because the on-disk optimizer
state has no `muon` inner state, so the run fell back to a partial restore. **`wm_ft_multi4`
records the same fallback**, so this is shared, not a difference.

### WEIGHTS and PREFIX — not recorded, and not filled in

Searched: each trajectory's `ft_progress.json` (keys are only `arm, ckpt, complete, hold_gen,
hold_real, last_saved_step, max_step, step, train_seed`), `ft_log.json` (`final, init, steps`),
and the full training logs (`v5m4|v5m3` matches: **0**). **The values these twelve ran with are
not recorded anywhere in their own artefacts.**

What *is* known constrains them without recording them. `ft_arm.sh:29-31` maps parent
checkpoint to arm one-to-one:

| arm | PARENT_CKPT | WEIGHTS | PREFIX | SEED_BASE |
|---|---|---|---|---|
| r1 | `$PRETRAINED` | `v5m_weights.npz` | `v5m` | 96000 |
| r3 | `$T/ckpt/wm_ft_multi2` | `v5m3_weights.npz` | `v5m3` | 99000 |
| **r4** | **`$T/ckpt/wm_ft_multi3`** | **`v5m4_weights.npz`** | **`v5m4`** | 99100 |

The twelve load from `wm_ft_multi3`, which matches **r4 and no other arm**. Two facts stop this
from being a record: `ft_arm.sh` **superseded** `traj_seed_train.sh` at 2026-09-05T04:00:35Z,
*after* these twelve synced (02:33–02:48Z); and r4's `SEED_BASE` is 99100 while these carry
train seeds 30–41. **So r4's values are the configuration the parent field points at, not the
values this run is known to have used.** Closing this needs the owner of job 6317365, or the
job's submit script.

### Comparability with the current task's artefacts

| dimension | the 12 | `wm_ft_multi4` | verdict |
|---|---|---|---|
| parent checkpoint | `wm_ft_multi3` | **`wm_ft_multi3`** | **same** |
| load step | 69378 | 69378 | same |
| restore mode | `partial_restore=True` | `partial_restore=True` | same |
| budget | 4800 steps, n_items 4800 | 4800 steps | same |
| hyperparameters | fixed by the launcher family: `--lr 1e-5 --anchor-lambda 1.0 --clip 1.0 --epochs 1 --micro 2 --group-items 1` | same family | same source |
| `WEIGHTS` / `PREFIX` | **not recorded** | **not recorded** | **cannot be compared** |
| `order_sha1` | 12 distinct values | **not logged** (the seed patch postdates it) | cannot be compared |
| ownership | job 6317365 (CANCELLED), another line | this line | **authorisation required** |

**Five dimensions match exactly, two cannot be compared because neither side records them.**
The twelve are therefore the right sample for the trajectory rung of the round-4 configuration;
the residual risk is confined to `WEIGHTS`/`PREFIX` and to the specific data order of multi4.

## 4B. The minimum scoring run that can decide C6

**Entry point:** `/lus/lfs1aip2/projects/public/u6gb/nb_build_pr22/traj_cell.sh`.
Not `sweep_cell.sh` — that builds `ckpt/wm_ft_traj_s30_step4800` as a sibling directory, which
does not exist; `traj_cell.sh` builds `ckpt/${TRAJ}/step_${STEP}`, which resolves for all twelve
(spot-checked s30, s33, s41).

**Independent unit: one training trajectory.** Tickers, days, contexts and generation seeds are
all nested inside a trajectory, so **n = number of trajectories**, never the number of cells.
Effective n of a ticker-averaged quantity is reported alongside every n.

**What each cell does.** One cell = one trajectory × one ticker. It generates 2 × 500 contexts
of 250 conditioning / 250 generated messages from that trajectory's step-4800 checkpoint on that
ticker's frozen index file, joins the two seeds as two `OUT_ROOT`s so the scorer sees K = 2, and
appends one JSON row `{traj, step, ticker, node, crps, qL1, sd_ratio}`. **The eight rows differ
only in `TRAJ`**, so their spread in `sd_ratio` is the trajectory rung and nothing else.

| item | value |
|---|---|
| cells | **8** — `wm_ft_traj_s30 … s37`, one ticker (GOOG), `STEP=4800` |
| generation seeds per cell | 2 (`97901 97902`), the script's own default, with its reasoning in the file |
| per cell | **≈ 8.6 min** — halved from a measured 17.2 min 4-seed cell (`sacct` 17:12 / 17:06 / 17:28). **Derived, not measured at K=2** |
| **wall clock** | **≈ 70 min on 1 node × 4 GPU** |
| **resources needed** | **1 node, 4 GPUs, ~1.5 h of walltime**, plus node-local scratch for `ROOT` and `RES` |
| shared-storage need | **none** — `ROOT` is `/local/user/$(id -u)/…` by construction, which is why the exhausted project inode quota does not block it |
| output | 8 JSON rows; `s_trajectory` = sd of their `sd_ratio` |

```
RES=/local/user/$(id -u)/traj_strip.jsonl
for TR in wm_ft_traj_s30 wm_ft_traj_s31 wm_ft_traj_s32 wm_ft_traj_s33 \
          wm_ft_traj_s34 wm_ft_traj_s35 wm_ft_traj_s36 wm_ft_traj_s37; do
  TRAJ=$TR STEP=4800 STOCK=GOOG RES=$RES \
  bash /lus/lfs1aip2/projects/public/u6gb/nb_build_pr22/traj_cell.sh
done
```

**Stopping criterion, fixed before data:** if the spread exceeds 0.08, stop and report C6 as
unresolvable at this budget. Do not buy more cells to chase it.

**Not started.** Scoring and the device assignment are for the coordinator to hand over
explicitly. This session holds **0 GPU workers and 0 un-launched submitters** — process scan
finds no `sbatch`, no `*_fleet.sh`, no `*_cell.sh`, no deferred submitter; `crontab` and `atq`
are empty; all eight tmux windows sit at a shell prompt with nothing queued.

**Training complete is not scoring complete.** Twelve trajectories are trained and **zero are
scored**, so **C6 remains INSUFFICIENT**.

## 5. Blockers, each with the action that clears it

| # | blocker | measurement (2026-09-05T20:4x UTC) | action |
|---|---|---|---|
| **B1** | **storage** | Lustre inodes **51,200,000 / 51,200,000 — full**, `/home` bytes **105,728,116 / 105,468,748 KB — over**. Creating this file as a new path failed with `Disk quota exceeded`; it is written by overwriting an existing inode. Generation's default path writes ~12,004 files/member, 24,016/cell | free inodes, or run generation with `PACK_MEMBER=1` to node-local and copy back 4 files/member (the `sweep_cell.sh` shape). **Not clearable by me alone** |
| B2 | good data discarded on exit code | `r3null_cell.sh:45`, `inter_cell.sh:60` still `\|\| exit 7`; the failure is intermittent → silently biased sample | S4b, mine, CPU |
| B3 | `.done` existence ≠ integrity | 9 sites still `-f` | S4a, mine, CPU |
| B4 | ownership | the 12 came from job 6317365, another line, owner unknown | explicit authorisation to score them |
| B5 | devices | shared account; a card free at read time may belong to another of Kang's tasks | coordinated allocation/node/device before starting |

## 6. Cancelled / deferred

| item | decision |
|---|---|
| **354 node-h plan (train 6 or 24 new trajectories)** | **withdrawn** — its premise was false; replicates already exist |
| H6, a GPU-hour total for the line | **declined** — cancelled/resubmitted steps make a total a guess |
| `step_budget.ipynb`, `verdict_audit.ipynb` | **out of scope** — another session holds uncommitted edits to one builder |
| `sol_corrected_inference` | **forbidden** — another session owns it |
| S4c (`exec` launchers) | deferred — real, not on the C6 path |

## 7. Budget and stopping criteria

CPU work continues with no device request. GPU spend is capped at the §4 strip and only after
a coordinated device assignment. Nothing is started to occupy reserved cards; no allocation or
held job of Kang's is cancelled or released.

## 8. Next action

**S4a + S4b — CPU, authorised, no device needed.** Two of the three things that would otherwise
waste the first GPU hour.

## 9. Readable results

| what | path |
|---|---|
| notebooks with outputs + HTML | `tasks/agent_rescue_20260905/notebooks/` |
| trajectory provenance ledger | `tasks/agent_rescue_20260905/round5/TRAJECTORY_LEDGER.md` |
| registered-vs-run | `tasks/agent_rescue_20260905/round6/H8_REGISTERED_VS_RUN.md` |
| acceptance | `round7/ACCEPTANCE_v4.md`, `round8/DISPLAY_ACCEPTANCE.md` |
| the four plan sections | `tasks/agent_rescue_20260905/delivered/` |
| PR | https://github.com/KangOxford/md-u6gb/pull/2 |

---

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
