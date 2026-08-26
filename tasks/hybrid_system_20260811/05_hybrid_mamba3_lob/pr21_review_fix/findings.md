# findings —— PR#21 review 修复（已核实事实池）

> 全部在 2026-08-26 会话里逐条核实过；执行时直接引用，不重查。

## 仓库与分支

- sigma-0 共用检出：`/lus/lfs1aip2/projects/public/u6gb/sigma-0`，在 main、落后 origin 42 前 1，**不碰**；worktree 放 `/lus/lfs1aip2/projects/s5e/quant_team/quant/`（PR#28 先例 `sigma0_lobbench_modes_wt`）
- PR#21 分支 `feat/hybrid-mamba3-nemotron-20260811`，head `d4d9ad9a`，merge-base `5c168eda`，共 14 提交
- **reshard 是 head 提交** `d4d9ad9 fix(ckpt): the reshard broadcast was itself the CXI deadlock it claimed to avoid` → 出列 = revert 一个提交
- 其余 13 个提交：c0bcd15 hybrid 起点 → … → facf656 PE 修复 → 32e09d3 KV cache 修复 → 5297958 token 预算 → a69bf05+c39b52f seqpar → d2b3853 decode parity → 17aa51c 日志解耦 → 35a88ad grad-norm 修复
- 真实 diffstat（三点号）：15 files, +728/−33（body 误写 16/−29）

## CI（A1）

- main 上四个 workflow 全部 `| tee` 无 pipefail：english_only 1 处、evidence 1 处（:57）、injection 5 处、return_bench 6 处，共 13
- ci/ 内容：check_english_only.py、check_evidence.py、mutation_check.py、record_evidence.py、evidence/{injection,register,return_bench,simulator,suite}.json
- english-only 检查器**自扫**（"检测 X 的代码必然含 X"），修复顺序必须先于 pipefail；#26 有豁免先例
- 正确写法已有先例：PR#28 `lobbench_merge.yml` 用 `defaults: run: shell: bash` + checker `--self-test` 打印 caught
- suite.json 录于 2026-08-16，断言 pytest.failed==0；merged 树现状 1 failed（test_model_registry 六元组收到 7）281 passed 8 skipped

## 代码坐标（PR head d4d9ad9a）

- `pyproject.toml:40` testpaths=["tests"]；新测试在 `src/s5/tests/` 收不到；reviewer 已验证搬 `tests/unit/` 后 XLA_FLAGS 戏法存活，290→298 全绿（除六元组）
- `tests/unit/test_model_registry.py:18` 精确六元组断言（mamba3,s5,gdn,kda,transformer,nsa）
- C1：`src/lob/train.py:666-691` mid-epoch save `except Exception` 逐进程走 `_chain={'none':'host','host':'broadcast'}` 并在 except 里重试 save_checkpoint（集合操作）→ 单主机 OSError = 部分进入 = 挂死；epoch 末 `train.py:812` 仍是 `(OSError, ValueError)` + fatal exit，两路径不一致
- `_reshard_for_ckpt`（train.py:583-）拼错的 mode 静默落到 host 路径 → 需校验三值
- `src/lob/train_helpers.py:1282-1285` `locals().get("grad_norms_or_logits")`
- `src/s5/seq_model.py:69` `self._attn_at` 唯一出现 = 死存储
- `src/s5/registry.py:~100` `HYBRID_ARCHITECTURES = frozenset({"hybrid_mamba3"})` 字面集合，紧邻的注释在反对字面名单；修法 = `BackboneDefinition` 加 `has_attention` 字段
- **registry.py:332 在 hybrid 工厂把 `remat=` 传给注意力层工厂**（回应里的精度注）；443/463 是 transformer/NSA 工厂；mamba3 层无 remat 参数
- `associative_scan` 只在 `src/s5/ssm.py`（4 处）；mamba3 SSD 路径是 segsum+einsum → body 50.66 GiB 归因要改
- 中文行数（d4d9ad9a）：run/base_model/train_full_autoreg.batch 7、src/lob/train.py 38、src/lob/train_helpers.py 21；**revert reshard 后 train.py 会大减，需重数**

## 科学事实（回应已用，P4/P6 要写进 body/报告）

- **token 恒等**：500-ctx 64×13,000×32,000 = 26.624B；2k 80×52,000×6,400 = 26.624B（launcher `curtail=32000 (=6400×5)` 是痕迹）；两钟并进不可辨识 → 32k 步 2k 同时关两钟（~133B/组）
- 修复时间界：facf656d 2026-08-12 12:38 UTC；32e09d34 2026-08-12 20:57 UTC；campaign 日志 08-13 开（baseline 4462 / hybrid 2536 起）→ 最早 bench 的 ~4.5k 步检查点 08-13 才存在
- +60% 推理开销测在 2k（52k tok）本线 rollout（CTX2K_FINAL.md:140）
- 2k 吞吐 ~1.8 micro-steps/s @4N → 32k 步（160k micro）≈ 25h/组
- 等参数组做法：缩窄 hybrid 注意力 FFN（PMATCH_ARM.md），基线复用不重训；单 init-seed 是其威胁表已列的开放限制
- `check_replica_groups.py` 在 md-u6gb `tasks/hybrid_system_20260811/05_hybrid_mamba3_lob/code/` 有跟踪（32 个 tracked 文件之一）；sigma-0 里无 → P3 拷入 tools/diagnostics/

## 已发评论（承诺原文所在）

- 1/3 机械 A/C：issuecomment-5419896076
- 2/3 科学 B/D：issuecomment-5419896949
- 3/3 计划矩阵：issuecomment-5419897611
- PR body 备份：scratchpad/pr21_body.md（本会话）——P4 改前另存到本目录
