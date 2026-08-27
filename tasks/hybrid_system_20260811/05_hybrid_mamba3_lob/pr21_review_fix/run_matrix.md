# P6 训练矩阵排期（起跑记录，2026-08-26 起）

对外承诺（PR#21 comment 5419897611，Reply 3/3）：六组，全部 **fresh 32,000 步 cosine**
（不是在已退火完的 6,400 步 schedule 上续跑——原 2k 三组由 chain_manager 传
`COSINE_STEPS=6400`，退火已完整结束），每组 ≈133 B token，9/10 前完成。

## 六组定义与状态

| # | 组 | 配置差异 | 关键 env | 状态 |
|---|---|---|---|---|
| 1 | baseline-2k-32k | — | `ARCHITECTURE=mamba3`，EXPECTED_PARAMS=33610439 | **2026-08-26 起跑**（nid010252, 1N K=20） |
| 2 | hybrid-2k-32k | 两层 attention（Nemotron 规则 L=6→layer3；忠实档 d_ff=2560） | EXPECTED_PARAMS=35435423 | 等节点 |
| 3 | matched-2k-32k | hybrid 收窄到基准参数量（d_ff=1135） | EXPECTED_PARAMS=33609998 | 等节点 |
| 4 | mamba3-2k d_state=256 | 状态容量 ×2 | `MAMBA3_D_STATE=256` | 等节点 |
| 5 | nsa-2k | attention 家族参照 | `ARCHITECTURE=nsa` | 等节点 |
| 6 | 第二 init 种子的 1–3 三组 | 换 init 种子 | — | 排在 1–5 之后，让位时明说 |

## 预算数学（实测 4N ~1.8 micro-step/s，2k 上下文）

有效批量 80 = 1 micro × 4 GPU × 节点数 × K，K 由起跑脚本推导并硬校验。

| 节点数 | K | 秒/更新 | 32k 更新耗时 | 更新/小时 |
|---:|---:|---:|---:|---:|
| 4 | 5 | ≈2.8 | ≈25 h | ≈1290 |
| 2 | 10 | ≈5.6 | ≈50 h | ≈645 |
| 1 | 20 | ≈11.1 | ≈99 h | ≈323 |

单节点起跑不浪费 GPU 时（通信开销反而更小）；节点腾出后停下换更多节点续跑，
EFFECTIVE_BSZ 恒等式保证换节点数不改实验定义（checkpoint auto 每 15 分钟）。

## 当前落位（2026-08-26 晚）

| 资源 | 剩余 | 用途 |
|---|---|---|
| 6141106 / nid010252（残留 0.2–1.2G，零 compute PID） | ~14h | **#1 baseline-2k-32k**，1N K=20，MAX_JOB_HOURS=13.7，预计银行 ~4.4k 更新 |
| 6136391 / nid010329（全空） | ~9.5h | injection 重录（GPU rollout 四组）+ ask2 token 对齐点评测 |
| 其余 6 节点 | — | Hyper-XVLA 线的 6 个 ft step 占用；结束后依次上 #2 #3 并给 #1 扩节点 |

## ask2 评测的两个 checkpoint（都已存在）

| 侧 | checkpoint | 步号 |
|---|---|---|
| baseline-500 | `sigma-0/checkpoints/j5877859_30nkkohd_5877859` | 32001（update-matched 点） |
| baseline-2k | `sigma-0-worktrees/hybrid-mamba3-nemotron-20260811/checkpoints/j6000409_gjnf0e03_6000409` | 6509（token-matched 点，设计 6400，实际步号如实报） |

## 运行约定

- wandb project 全预算一代单独用 `sp500-mamba3-35m-ctx2k32k`；日志目录 `logs_lobs5/ctx2k32k_<arm>`，与 6.4k 那代分开。
- 每次动手前 gtop+PENDING 两侧同查；attach-first；step 起真名（`base-m3-2k32k` 等）。
- 观测：CHECKPOINT_EVERY=auto（15 分钟 ckpt / 1 分钟 wandb），LOG_GRAD_NORMS=1，LOG_EVERY=250。

## 2026-08-27 07:2x 执行形态修订：自有 sbatch 链为主，attach 为机会性补充

共享占位分配上三线互踩实录：04:56 我按物理空拿下 nid010817/010413（X-VLA 段间
空隙），07:00 X-VLA 链回收 010817 → base32k rank 被杀（barrier 遗言，无 Python
级根因 = 外部 kill）；010413 上其 ft-van 与我方 hyb32k 以 9.2G+83G 共存（合计
92.6G < 95.6G，侥幸活着）。**物理空探测对别家链的段间空隙无解**，故训练矩阵
改走独占 sbatch：

| 组 | 段（24h × 2N, afterany 链） | 作业号 |
|---|---|---|
| base | s1→s2→s3 | 6153336 → 6153357 → 6153359 |
| hyb | s1→…→s4 | 6153363 → 6153364 → 6153365 → 6153366 |
| pm | s1→…→s4 | 6153369 → 6153371 → 6153373 → 6153374 |

段脚本 `pr21_review_fix/code/sbatch_2k32k_segment.batch`：起跑时收掉本组 attach
副本（step 名匹配 + worktree 路径匹配 kill）、从共享 NODE_LOG_DIR 追最新
checkpoint 续、达标段直接退出。attach 侧（候位/链控制器）保留：能捡到的空隙
照捡，sbatch 段起跑时自动接管。
