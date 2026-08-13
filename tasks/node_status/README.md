# Node Status —— 「这个节点留给谁」的声明与核对

功能作为文件夹。这里放的是**节点归属声明**这一件事的全部：状态文件、读写模块、CLI。
`gtop`（终端 TUI）和 `sgpur`（网页）都是它的**消费者**，各自住在别处。

## 它解决的问题

`idle` 只说明这张卡**此刻没在算**，不说明它**没有主人**。
Slurm 知道谁**持有**一个节点，不知道它**本来是留给谁**的 —— 这个缺口只能靠声明来补。

多个 agent 会话共用同一批 chain 分配时，把 idle 读成「可抢」会出事：
2026-08-12 `wm_ft_b` 首启就是在 init 与第一个 backward 之间，被邻居的 88 GiB 预分配挤爆的。

## 用法

```bash
nodelock lock nid010547 --who claude-dfm -j 5992007 --note "专供 dfm 会话"
nodelock lock 5992007 --who crps-wm_ft         # jobid 展开成它持有的全部节点
nodelock unlock nid010547                       # 整节点
nodelock unlock nid010367:1,2                   # 只放两张卡，整节点锁自动收窄成 0,3
nodelock ls [--json] [--prune]
nodelock status                                 # 声明 vs squeue 现实
```

节点名支持 SLURM 方括号（`nid[011165-011167]`），冒号后跟卡号选具体的卡，纯数字当 jobid。

`nodelock status` 报三种不一致，每一种都值得动手：

| 判定 | 含义 |
|---|---|
| `STALE 锁（job 已结束）` | 声明还在，依托的 job 没了 → `--prune` |
| `锁着但我没持有这个节点` | 声明指向一个已经不属于我的节点 |
| `持有但无人认领` | 拿着节点却没人声明要用 → 要么用起来，要么还回去 |

## 四个状态

| 状态 | 判据 | 颜色 |
|---|---|---|
| `run` | util > 0 | 绿 |
| `held` | util == 0 但显存 > 64 MiB | 黄 —— **真浪费** |
| `idle` | 显存也空，且无人声明 | 红 —— 可抢 |
| `lock` | 本该 idle，但有人声明了 | 品红 —— 空得有意 |

### 三条设计约束（改之前必须知道）

**① `lock` 只覆盖 `idle`，绝不覆盖 `run` / `held`。**
这套工具的全部价值是「占着但没在算」这个判据。如果 lock 能盖住 `held`，
就把真浪费藏起来了。锁的语义只是「这张**空**卡的空是有意的」。实测：

```
nid010272 ▪ crps-wm_ft
  [0] util 0%  mem 83.7/95.6G   held               ← 占着显存不算，照报 held
nid010367 ▪ crps-wm_ft
  [1] util 0%  mem  0.0/95.6G   ▪lock:crps-wm_ft   ← 空卡，锁生效
nid010547（未锁时）
  [0] util 0%  mem  0.0/95.6G   idle               ← 真正无主的空卡
```

**② 挂在 jobid 上的锁随 job 结束变 `stale`，且 stale 按 `idle` 计。**
一把过期的锁若继续显示 lock，会永久藏起一张真空闲的卡 —— 比没有锁更危险。
不带 `-j` 的手动锁永久有效，要自己 `unlock`。
`running_ids()` 查不到时（squeue 失败）一律当 live：把有主的卡误判成无主，
方向比反过来更危险。

**③ 锁定的空卡不计入「空转功耗」**，否则那个数字就不再是「该去救哪张卡」。
`gtop --log` 的 JSONL 里 `gpu.lock` 单独一栏，事后想算回浪费也算得回来。

## 状态文件：为什么是两份，以及谁是真相

两个消费者各自先落地了自己的格式，而且都有道理：

| 文件 | 谁读 | schema |
|---|---|---|
| `tasks/node_status/gpu_locks.json` | `gtop`、`sgpur` | `{"locks":[{node,gpus,jobid,who,note,at}]}` —— **真相**，有逐卡粒度与 jobid 过期 |
| `~/.config/nodelocks.json`（`$NODE_LOCKS_PATH`） | `sgpur` | `{host: session}` —— **投影**，只保留 who |

扁平格式不能塞额外的 key：`sgpur` 的读法是把**每个顶层 key 当成节点名**，
放个 `"locks": [...]` 进去，网页上就会多出一个叫 `locks` 的幽灵节点。

所以 `nodelock.py` 做桥：**写的时候两份都落**（先富后扁——中途失败宁可留下没有投影的真相，
也不要留下没有依据的投影，因为投影才是别人拿去做闸门判断的那份），
**读的时候以富格式为准**，富格式缺失才回退读扁平（当作整节点、无 jobid 的手动锁）。
两个消费者一行都不用改。

**锁是协作性的**：它改变显示、也改变各会话自律闸门的判据，但并不真的阻止 CUDA 分配。

## 文件

| 文件 | 作用 |
|---|---|
| `nodelock.py` | 规范模块：路径、schema、读写、stale 判定、目标解析 |
| `nodelock` | CLI（已 symlink 到 `/projects/public/u6gb/.local/bin/nodelock`） |
| `gpu_locks.json` | 状态文件（真相） |
| `../gtop_20260810T182343Z/gtop` | 消费者：终端 TUI，四态渲染 + 自带 `lock/unlock/locks` |
| `/projects/public/u6gb/.local/bin/sgpur` | 消费者：网页版 |

## 协作注记（2026-08-13）

这个功能由三个并行会话同时在做，各写各的：
`gtop` 的注册表原语、`gtop` 的四态渲染与 CLI、`sgpur` 的网页侧。
一度出现**两套语义相反的注册表**（显式声明 vs 默认全锁）与**两个不同的状态文件路径**。

收敛的判据不是「谁的 schema 赢」，而是**让每个消费者都不用改**。
最终：显式声明胜出（默认全锁表达不了 `stale`），路径统一到本目录，格式差异由本模块桥接。
旧文件保留为 `../gtop_20260810T182343Z/gpu_locks.json.superseded`，没有删。
