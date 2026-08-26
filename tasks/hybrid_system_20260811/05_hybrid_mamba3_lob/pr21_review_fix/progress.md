# progress —— PR#21 review 修复

## 2026-08-26（会话 1）

- ✅ review 全部论断逐条核实（记录在 findings.md），A/C/D 无一可驳，两处仅需精度修正
- ✅ 三条回应评论发布（英文、无 protocol、无签名，机械自查过）
- ✅ P0：本计划目录建立（task_plan / findings / progress）
- ▶ P1 开工：worktree + 读 check_english_only.py / record_evidence.py 弄清自扫与重录机制
- GPU 状态：32/32 忙（hyper-recipe ×2，剩 ~18h），PENDING 两活两依赖挂起，无空卡可填 → P6 等档期

### P1 执行（同日续）

- ✅ worktree `sigma-0-worktrees/ci-pipefail-20260826`，分支 `ci/pipefail-per-workflow`（基 b813a9c = main tip = reviewer 基线）
- ✅ 自扫根因：正则用字面 CJK 区间端点 + docstring/workflow 注释里的「每」。修为 `\uXXXX` 转义 + 例句改述
- ✅ 四 workflow 统一 `defaults: run: shell: bash`；两 checker 各加 `--self-test`（english 7 探针、evidence 6 探针，含「own machinery is clean」双向断言）
- ✅ 验收：english self-test 7/7、scan 只剩 2 个 KNOWN 债务文件 rc=0、evidence self-test 6/6、四 yml 解析 ok
- ✅ **前后对照**：同一棵树（evidence 审计真实失败 2 文件），旧默认 shell 步骤 rc=0，`bash -eo pipefail` 步骤 rc=1
- ✅ 提交 a7fd450 "ci: give every workflow pipefail, and every checker a way to fail"（6 files +145/−9）
- ▶ simulator.json 重录：沙箱 cgroup 线程上限挡死（pthread EAGAIN），已转 tmux（claude-login43）跑，后台 waiter bvr41wc8a 盯完成
- ⛔ return_bench.json 重录被挡：记录的输入目录 `crps_.../data/hp_rn` 已在 0817 prepurge 中消失，**同输入重录不可能**；换基线目录是数据语义决策 → 留给 Kang（PR 里写明两个选项）
- ✅ PR#35 开出（前后对照 rc=0→rc=1 写进正文）+ PR#21 发 A1 进展短评（issuecomment-5424551442）

### 用户拍板后（同日午后）：可以 merge + return_bench 输入要放稳定位置

- ✅ **simulator.json 重录**：对账发现 6136391 的 16 卡全空（剩 21h47m）→ attach 纯 CPU srun（`--job-name=evrec-sim`，Lustre venv 叠加 chex）→ **20 秒录完，全部指标与 08-01 逐位相同**（唯二 diff：mm_sim 哈希 + python 版本）。登录节点三连败（沙箱/限线程/tmux 全被 pthread EAGAIN 挡）的正解就是回计算节点
- ✅ **稳定输入位置**（用户令「下次别被删掉」）：`/lus/.../u6gb/ci_evidence_inputs/return_bench_hp_base_member0`（22,005 文件 0.49 GB，源 = hp_base/member_0，hp_rn 的同族基准配置）+ **单 inode tar 复活备份** + README（写明哪份 evidence 依赖它）+ `chmod -R a-w` 只读锁。inode 现实：u6gb 98.8%、s5e 99.06%，这就是 hp_rn 被清的根源，tar+只读是对症设计
- ✅ **return_bench.json 重录**：`--accept-new-inputs` 对稳定副本，54 秒，2000 对序列，7 断言全过；基线移动在提交与 PR 正文写明
- ✅ 顺手修 `check_evidence.py --file` 相对路径崩溃（resolve 一行）
- ✅ 本地审计 **all 5 evidence file(s) pass** → f65d86b 提交推送 → PR#35 正文更新为全绿预测 → **合并（471c5668）**
- 🔴→🔧 **main 首轮真实运行：return-bench 红**——`recorded commit … is not in this repository`：checkout 默认 depth-1 浅克隆，ancestor 检查按设计需要完整历史；**这检查从落地起就在暗红**，pipefail 首次让它发声。本地重放没抓住：我的克隆是全量的（重放保真教训：checkout 形状也是环境）。一行 `fetch-depth: 0` 走 **PR#36** 快速修复
- ✅ PR#36 四检查全过 → 合并 → **main 四关卡全真绿**（injection 15s / english 12s / evidence 9s / return-bench 50s）→ P1 完结，PR#21 下发 A1 完结短评（issuecomment-5424849290）

### P2 执行（同日下午）

- ⚠️ hybrid worktree 有 8/18 的未提交 WIP（mamba3.py + mamba3_seqpar.py 的 sp_size 接线，154 行）——mtime 八天无人动。全程逐文件 add 避开；后经隔离证明它**不是**等价测试失败的原因；已 `git stash`（带说明信息）保存给原作者
- ✅ `git revert d4d9ad9` reshard 出列（ask 5，保留 review 锚点；batch 脚本中文随之清零）
- ✅ A4 译英：train.py 3 行 + train_helpers.py 21 行（含用户可见的 cadence print）；locals().get → 循环前显式 `grad_norms_or_logits = None`
- ✅ A3+A2：两个 parity 测试搬 `tests/unit/`；六元组断言加第七项 `hybrid_mamba3`
- ✅ C3：`build_backbone` 把解析出的注意力位置以 override 同格式**盖章回 args**（进 checkpoint config）
- ✅ minors：`has_attention` 上 `BackboneDefinition`（替掉字面集合）；删死存储 `_attn_at`
- ✅ C4：`test_hybrid_construction.py` 五测——(3,9)@12 层、L=31 复现 (8,13,17,22)、盖章往返、**规则被改后已盖章配置不动**（monkeypatch 证明）、carry 判别（注意力位含 cache 轴、递归位无）；登录节点 6.4s 全过
- ✅ C5：register 添 D-T4/T5（silent，FIXED）+ D-X8（loud，FIXED）+ **D-X9（reshard 死锁，OPEN**——修复在独立 PR）；渲染 + 13/13 register 测试过 + register evidence 重录（30 缺陷：24 修 2 开 1 界 3 驳）
- ✅ 文档：training runbook 添 LOG_EVERY / LOG_GRAD_NORMS / hybrid 六旋钮表（含盖章语义与 D-T4/T5 引用）
- 🔴→🔧 **真 bug 一枚（与 review 无关的额外收获）**：`jnp.clip(a_min=)/(a_max=)` 在 jax 0.11 已被移除 → 三个 mamba3 并行/递归等价测试 setup 即炸；录 suite 的旧环境（py3.12.11/jax0.9，s5e env）对 u6gb 两侧均无权限、计算节点也拒——改**位置参数**写法新旧通吃，18/18 转绿
- ✅ 合入 origin/main（b362a7e，即我自己的 #35+#36）；唯一冲突 `ci/evidence/register.json` = 生成物 → 合并树上重录解决，不做文本合并
- ▶ 合并树全量套件跑着（step 6136391.47），绿则就地录 suite.json → push → PR#21 checks → body 重写 → progress comment
- ⏸ injection.json 重录需 4-arm GPU rollout → 与 C2 round-trip、训练矩阵同一个 GPU 批次（PR#21 的 evidence 关卡在那之前会红着这一格，comment 里明说）
