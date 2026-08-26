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
- ⏭ 待办：simulator 结果落地 → 提交 evidence → push + 开 P1 PR（英文，带前后对照与 self-test 记录）→ PR#21 发 progress comment
