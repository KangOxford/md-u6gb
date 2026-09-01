# LDM-RL 在 Isambard-AI 上的落地：实测结果

记录**测出来的数**，不记录计划。计划在 `../PLAN.md`。
起于 2026-09-01。分支 `isambard-port`（基于 `rl` 分支 `5ca5efe9`）。

---

## 1. 已完成的关卡

| 阶段 | 内容 | 结果 |
|---|---|---|
| **E0** | 评测栈环境 | ✅ rdkit 2025.09.6 / gpytorch 1.15.2 / gauche 0.1.6 / lightgbm 4.7.0 / meeko 0.8.0 / gemmi 0.7.5 全部 import 通过 |
| | vina 二进制 | ✅ `AutoDock Vina f458505-mod`（conda-forge aarch64），装进 `envs/ldm-rl/bin/vina` |
| **E1** | `ldm_rl` 单元测试 | ✅ **61 passed in 1.37s**（交接文档说 53，rl 分支后续又加了 8） |
| **E2** | 真 reward 全链路 | ✅ 见下表 |
| — | 四组 episodes 数据 | ✅ 60 + 16×3 行，站点路径已烤入，零残留作者路径 |
| — | 三个模型下载 | ✅ 1.5B 2.9G / Qwen3.5-9B 19G / LDM-CoT-SFT 18G |
| — | 启动脚本参数化 | ✅ 10 个脚本，`/mnt/data0` 与 `/root` 清零 |

### E2：真 vina + 真活性模型 + 真 reward（纯 CPU，2 轮，确定性提议器）

三种 reward 跑在**完全相同的分子**上（vina 与 activity 逐轮一致），所以差别只来自 reward 本身：

| reward | round 0 | round 1 | 合计 |
|---|---:|---:|---:|
| `hypervolume` | 0.000000 | 0.000001 | 1e-06 |
| `acquisition` | 0.000000 | 0.000000 | 0.0 |
| `improvement` | 9.365461 | 0.503000 | 9.868461 |

共同的测量值：`vina = -3.395 / -3.898 kcal·mol⁻¹`，`activity(pIC50) = 5.9705 / 5.9003`。

---

## 2. acquisition reward 的 0.0 是真值，不是静默回退

**为什么要查**：`env.py:647-651` 在 `scores` 为空时 `return 0.0`，与「真的算出 0」在数值上不可分。
若是前者，reward 恒为零 ⇒ GRPO 优势全零 ⇒ **没有梯度信号**，而 acquisition 正是 R1/R2 的 reward，
R2 是整个设计的枢轴。

**判定依据落在结构上而不是数值上**：看 `components['scores']` 是不是空列表。

```
round=0 reward=0.000000 scores=[0.0]                  agg=max
round=1 reward=0.000000 scores=[0.0]                  agg=max
round=2 reward=0.063888 scores=[0.06388776076585678]  agg=max
```

**结论：非空 ⇒ 真值。** 前两轮为 0 是 **GP 冷启动**——`reservoir_size=2` 时 UCB 的 `mean + β·std`
还没有信息，到第 3 轮升到 0.0639。

**推论（影响起跑顺序）**：暖机那一步（`run_warmup_real_slime.sh`，dock 60 个分子）不是可选的性能优化，
而是**让 acquisition reward 在训练第一轮就有信号**的前提。跳过它，GRPO 的前若干轮拿到恒零 reward。

---

## 3. `vina_max_workers=32` 没有依据，实测饱和点在 8

PLAN §5 第 4 条要求「先量一个分子的真实耗时再定并发」。量了：

| 每轮评估数 | workers=1 | workers=8 | workers=32 |
|---|---:|---:|---:|
| 1（生产配置） | 5.4 s/轮 | 3.4 s/轮（**1.60×**） | 3.6 s/轮（1.52×） |
| 8 | 7.6 s/轮 | 5.3 s/轮（**1.43×**） | 5.4 s/轮（1.40×） |

**两点结论**：

1. **8 之后没有可测收益**，32 与 8 的差别（5.3 vs 5.4）落在噪声里。
   commit `5ca5efe9` 把它从 1 提到 32 的理由是「GH200 有 288 核」——
   **核数不是约束，每轮 dock 的分子数才是**。32 个 worker 里大部分没活干。

2. **总加速只有 1.4–1.6×，说明 docking 不是单轮耗时的主导项**，这与 PLAN §5
   「docking 是真瓶颈」相反。原因是 `vina_exhaustiveness=1`——**生产配置里也是 1**，
   所以 docking 本身就很便宜，其余时间花在 GP 拟合、活性预测与候选生成上。

**可操作的含义**：日后嫌慢时，调 `vina_max_workers` 不是那个旋钮。要先量出剩下那 60–70%
的时间花在哪里，再决定动谁。

---

## 3b. 训练栈 → 评测栈的进程边界已端到端验通（并补上一个会让 P0 白跑的缺件）

训练时 reward **不在训练进程里算**：`bridge.py` 起一个子进程
`task_python -m ldm_rl.real_env_worker`，走 JSON-lines stdio，并显式剥掉
`LD_LIBRARY_PATH` / `CUDA_HOME` / `CUDA_VISIBLE_DEVICES`。这个边界是代码写死的。

**缺件**：`bridge.py:87-89` 是

```python
task_python = str(spec.real.get("task_python")
                  or "/mnt/data0/ys/LDM/tasks/small_molecule/.venv/bin/python")
```

而 `gen_episodes_runs.sh` 把 `config.real_kwargs` 整块烤进 episode，**站点 config 里
原本没有 `task_python`**。于是四个 run 的 episode 全都缺这一项，训练时会回退到作者机器
的路径。已补进 `config_real.isambard.json` 并重新生成四个 episodes 文件。

**危害形态**：那个回退值在别的机器上依然是一个"合法"的字符串，所以它不报「配置缺失」，
而是在 reward 那一步报下游错误。加上 `remote_env.py:41` 把 worker 的 `stderr` 设成
`DEVNULL`，父进程只能看到一句 `task-venv worker exited unexpectedly`——**真实死因被丢掉了**。
（我自己第一次跑探针就撞上这个：忘了 export `PYTHONPATH`，worker 起来就
`ModuleNotFoundError` 退出，而父进程什么都不说。手动跑一次 worker 才看到原因。）

**端到端验证结果**——子进程起得来、协议往返通、reward 真的在另一个解释器里算出来：

| | round 0 | round 1 |
|---|---|---|
| 进程内（§1 的 E2） | vina −3.395 / act 5.9705 | vina −3.898 / act 5.9003 |
| **经子进程边界** | **vina −3.395 / act 5.9705** | **vina −3.898 / act 5.9003** |

两条路径**逐位一致**，所以远程路径不只是"能跑"，而是"算的是同一件事"。

**建议回馈上游**：`remote_env.py` 的 `stderr=subprocess.DEVNULL` 改成捕获并在
`RuntimeError` 里带出最后若干行。训练跑几小时后 worker 挂掉时，这是唯一的线索。

---

## 4. 两个文件系统的约束是互补的（决定了环境该建在哪）

| | 空间 | inode |
|---|---|---|
| `/home`（VAST） | **101G 硬配额，曾撞顶** | 893k / 10000k（8.9%，宽裕） |
| Lustre | 116.9T / 200T（宽裕） | **49.96M / 51.2M（97.6%，只剩 1.24M）** |

上一次安装失败的直接原因是 `/home` 撞顶，但**根因不是空间不够**：

`install_train_env.sh` 第 24 行把 conda 包缓存指向 `$TMPDIR/conda_pkgs_$USER` 以避开配额，
**却没有 `mkdir -p`**。conda 的 `pkgs_dirs` 是一条**回退链而不是强制指定**：首选目录建不出来时
它不报错，静默滑到下一个可写目录 `~/.conda/pkgs`——正好在配额里面。于是「避开配额」这个配置
一次都没生效。报错路径 `/home/.../.conda/pkgs/cmake-4.4.3...` 就是证据。

已修：建目录 + 验可写 + 打印实际落点，建不出来就停。

**腾空间的做法也随之改了**：上一次是把冷环境逐文件 rsync 到 Lustre——那是**把 7.5 万个 inode
从富余的一侧搬到紧缺的一侧**，正好搞反。改成 `tar --remove-files`（每个文件写进归档之后才解除
源端链接，与 rsync 同级的安全性），目标端是一个文件：

| 冷环境 | 文件数 | 归档后 Lustre inode |
|---|---:|---:|
| glm53-vllm | 77,247 | **1** |
| glm53-vllm_cu130_deprecated（Lustre 侧） | 45,421 | **1** |
| glm53-vllm_cu130_deprecated（/home 侧） | 59,456 | **1** |

冷的依据是逐一查证过的：在跑的 `6220558/6220580` 用的是 apptainer SIF
（`images/vllm-glm53-arm64-cu129.sif`），不是这几个 conda 环境；而 `ldm-nanogpt`
正被 `run_repro_arm.sh:54` 使用，**没有动**。

---

## 5. 启动脚本的配置到不了它要控制的路径

`slime_launch/*.sh` 里 `REPO_ROOT` / `MEGATRON_ROOT` / `CONDA_PREFIX` / `MODEL_HF` / `CONFIG`
全是**直接赋值**，外部 export 会被无声覆盖。改成 `${VAR:-<作者原值>}` 之后：作者机器上不设环境变量
行为完全不变，本机 `source site_env.sh` 才生效。命令行参数里的 `--hf-checkpoint` / `--ref-load` /
`--prompt-data` 也一并由 `REPO_ROOT` 与新增的 `HF_MODELS` 推出。

改完逐脚本求值验证过，五个脚本的全部变量都解析到本机路径（不是只看改没改）。

**订正 PLAN §4 的一处依赖**：E7（暖机 GP）在 PLAN 里写成只依赖 E2，实际上
`run_warmup_real_slime.sh` 需要 slime + 1.5B 的 torch_dist 检查点——它是 LLM 驱动的 rollout
去提分子再 docking。所以 **E7 依赖 E3 与 E5**，不能提前做。

---

## 6. 还没做的

| 阶段 | 内容 | 卡在哪 |
|---|---|---|
| E3 | 训练栈安装 | 进行中（sglang aarch64 轮子已确认存在） |
| E4 | **P0**：1.5B GRPO 冒烟，验 TE backward 不 SIGSEGV | 等 E3 |
| E5 | HF → Megatron torch_dist 转换 | 等 E3 |
| E6 | **P1**：9B real 极小 count 冒烟 | 等 E4+E5 |
| E7 | 暖共享 GP（60 个分子） | 等 E3+E5（依赖已订正） |
| E8 | **P2**：R1–R4 × 3–5 seed 真训练 | 等全部 |
| E9 | 离线评测 G12C + G12D | 等 E8；`best_g12c_model.joblib` 可并行训 |
