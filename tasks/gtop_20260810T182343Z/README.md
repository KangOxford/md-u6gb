# gtop —— 看我自己手上这些节点在干什么

`htop` 看的是**一台机器**的进程，`gpustat` 看的是**一台机器**的 GPU。
`gtop` 看的是**我当前持有的整个 SLURM 分配**：所有节点、所有卡、谁在用、
以及最关键的一件事 —— **有多少卡正占着但没在算**。

```
  gtop  1 alloc · 4 nodes · 16 GPU   run 1/16 (6%)  held 3  idle 12   ·  空转功耗 1.52 kW
═══════════════════════════════════════════════════════════════════════════════════════════
 ▸ job 5975573  u6gb-4-node-chain  4N 16GPU  used 29m14s  left 23h29m
     steps: 5975573.16 bash(0:59)
   nid010195   288c cpu   1.1% [░░░░░░░░░░]  mem   209/856G [██░░░░░░░░]  ld 1.17  up 1h49m
     [0] GH200 120GB   36°C  140W util   1% [░░░░░░░░░░] ▁▁▂▁▁▃▂▁ mem  86.8/95.6G [█████████░]  kangli.u6gb:python/219342
     [1] GH200 120GB   37°C  134W util   0% [░░░░░░░░░░] ▁▁▁▁▁▁▁▁ mem   1.3/95.6G [░░░░░░░░░░]  kangli.u6gb:python/219342  held
     [2] GH200 120GB   37°C  127W util   0% [░░░░░░░░░░] ▁▁▁▁▁▁▁▁ mem   1.3/95.6G [░░░░░░░░░░]  kangli.u6gb:python/219342  held
     [3] GH200 120GB   37°C  131W util   0% [░░░░░░░░░░] ▁▁▁▁▁▁▁▁ mem   1.3/95.6G [░░░░░░░░░░]  kangli.u6gb:python/219342  held
   ...
  pending 2: 5980745(4N)   5980502(4N)
───────────────────────────────────────────────────────────────────────────────────────────
  sample 18:34:38 (1.1s)   next 1m47s   interval 120s   [q]uit [r]efresh [+/-]interval [p]ause [c]md
```

上面这张真实截图一眼给出的结论是：**4 张卡里只有 1 张在算，另外 3 张挂着 CUDA
context 但利用率恒为 0，剩下 3 个节点 12 张卡完全空转，合计 1.52 kW 白烧。**


## 1. 快速开始

```bash
cd /projects/public/u6gb/tasks/gtop_20260810T182343Z

./gtop                 # TUI，默认每 120 秒采样一次
./gtop -i 30           # 改成 30 秒
./gtop -1              # 采一次、打印、退出（适合塞进脚本或日志）
./gtop -j 5975573      # 只看某个作业
./gtop -c              # 展开每个 GPU 进程的完整命令行
```

已经建了软链，任何目录下直接敲 `gtop` 即可：

```
/projects/public/u6gb/.local/bin/gtop -> .../tasks/gtop_20260810T182343Z/gtop
```


## 2. 它是怎么拿到数据的

先说清楚约束：**登录节点上没有 GPU，也看不到计算节点的 GPU**。在 login40 上敲
`nvidia-smi` 报告的是 login40 自己（无卡）。要知道 `nid010195` 上四张卡的状态，
命令必须在那个节点上执行。合规的通道只有一条：用 `srun --overlap` 附着到
**我自己已经持有的分配**上去跑一条只读命令。

```
   登录节点 (login40)                          计算节点 (nid010195 …×4)
  ┌───────────────────────────┐               ┌────────────────────────────┐
  │ gtop (curses TUI)         │               │                            │
  │  │                        │               │  bash -s                   │
  │  ├─ squeue ───────────────┼──────────────>│   ├ /proc/stat  ×2 取差分  │
  │  │   作业 / 节点 / step   │   probe.sh    │   ├ /proc/meminfo          │
  │  │                        │   经 stdin    │   ├ /proc/<pid>/{status,   │
  │  └─ srun --overlap ───────┼───灌进去─────>│   │        cmdline}        │
  │        ▲                  │               │   └ nvidia-smi --query-*   │
  │        │  单行 JSON ×N    │<──────────────┤        │                   │
  │        └── 每 N 秒一次 ───┘               │   跑 0.4 秒然后退出        │
  └───────────────────────────┘               └────────────────────────────┘
        不留常驻进程                                  全程不碰 Lustre
```

三个设计选择，每个都有具体理由：

| 选择 | 为什么 |
|------|--------|
| **拉取式**（到点打一枪 srun，采完就退），而不是在节点上常驻采样循环 | 采样间隔是分钟级，常驻进程没有收益，却会踩 BriCS「禁止常驻 agent」的红线，还要多一套中间文件 |
| **probe.sh 经 stdin 灌进 `bash -s`**，而不是让节点各自去 Lustre 打开脚本 | 4 节点 × 每 2 分钟 = 周期性元数据脉冲。登录节点读一次文件、把内容当字符串发过去，计算节点一次 Lustre 访问都没有 |
| **`--exact --cpus-per-task=1`** | 不加 `--exact` 时 srun 默认索要节点的全部 CPU，会跟正在跑的训练抢核。加上之后只借 1 个核，0.4 秒还回去 |

`--overlap` 这个词的含义：SLURM 默认不允许两个 step 使用同一份已分配资源，
`--overlap` 显式声明「我要和已有 step 共享」。没有它，采集会一直排队等到训练结束。

### 2.1 一个必须处理的坑：从 attach 的 shell 里启动

如果 gtop 是在一个**已经 attach 到计算节点的 shell**（`sbash` / `srun --pty`）里
启动的，那个 shell 带着外层 step 的整套 `SLURM_*` 环境变量：

```
SLURM_NNODES=1            SLURMD_NODENAME=nid010195       SLURM_STEP_ID=30
SLURM_TASKS_PER_NODE=1    SLURM_STEP_GPUS=0,1,2,3         ...
```

SLURM 会把这些变量当作新 `srun` 的**默认参数**，而且在这一项上它们**压过命令行**：
`SLURM_NNODES=1` 直接否掉 `--nodes=4`，报

```
srun: error: Only allocated 1 nodes asked for 4
```

也就是说「命令行参数总是赢」在 SLURM 嵌套调用里不成立。所以 gtop 在发出任何
`squeue` / `srun` 之前，会先剥掉环境里所有 `SLURM` 开头的变量（保留 `SLURM_CONF`，
它指向 slurm.conf）。注意 `SLURMD_NODENAME` 是 `SLURMD_` 前缀，只匹配 `SLURM_`
会漏掉它。

这个坑值得单独写出来，因为**它恰好在最需要 gtop 的时候才会触发** —— 你 attach
上去干活了，才想看看卡跑得怎么样。


## 3. 界面怎么读

### 3.1 顶栏

```
  gtop  1 alloc · 4 nodes · 16 GPU   run 1/16 (6%)  held 3  idle 12   ·  空转功耗 1.52 kW
```

| 字段 | 含义 | 怎么算的 |
|------|------|----------|
| `1 alloc` | 我持有的 RUNNING 作业数（allocation = 一次 sbatch/salloc 拿到的整块资源） | `squeue -u $USER -t RUNNING` 计数 |
| `4 nodes` / `16 GPU` | 采样成功回话的节点数与卡数 | JSON 回话计数，不是 squeue 声称的数（回不来的节点不计入） |
| `run 1/16 (6%)` | 真正在算的卡数占比 | `utilization.gpu > 0` |
| `held 3` | **占着卡但没在算**的卡数 | `util == 0` 且 `memory.used > 64 MiB` |
| `idle 12` | 纯空转的卡数 | `util == 0` 且显存也是空的 |
| `空转功耗` | 非 run 状态的卡的瞬时功耗之和 | `Σ power.draw`（GH200 空载单卡约 82–103 W） |

### 3.2 GPU 的三种状态（这是 gtop 与 gpustat 最主要的差别）

`gpustat` 只区分「有没有进程」。在集群上这不够用，因为最常见的浪费形态恰好落在中间：

| 状态 | 判据 | 典型成因 | 颜色 |
|------|------|----------|------|
| `run` | `util > 0` | 真的在算 | 绿（≥90% 亮绿） |
| `held` | `util == 0` 且 `显存 > 64 MiB` | JAX/PyTorch 已建 CUDA context 但没喂到数据；卡在 dataloader；进程 hang；只在做 CPU 侧工作 | 黄 |
| `idle` | `util == 0` 且 `显存 ≤ 64 MiB` | 分配拿到了但没人用 | 红 |

64 MiB 这个门槛的来历：实测空闲 GH200 的 `memory.used` 是 1–3 MiB（驱动本底），
留一个量级余量即可可靠区分「真有进程占着」和「本底噪声」。

上面那张截图里的 `run 1 / held 3` 就是 JAX 多卡初始化的典型指纹：主卡吃满 86.8 G
开始算，其余三张各挂 1.3 G 的 context 空等。两态工具会把这四张卡一律报成 busy。

### 3.3 节点行

```
   nid010195   288c cpu   1.1% [░░░░░░░░░░]  mem   209/856G [██░░░░░░░░]  ld 1.17  up 1h49m
```

| 字段 | 定义 | 单位 / 来源 |
|------|------|-------------|
| `288c` | 节点物理核数 | `/proc/cpuinfo` 里 `processor` 行数 |
| `cpu 1.1%` | 全节点 CPU 利用率 | `/proc/stat` 相隔 0.35 秒两次快照的差分：`(Δtotal − Δidle) / Δtotal` |
| `mem 209/856G` | 已用 / 总内存 | `(MemTotal − MemAvailable) / MemTotal` |
| `ld 1.17` | 1 分钟平均负载 | `/proc/loadavg` 第一列 |
| `up 1h49m` | 节点开机时长 | `/proc/uptime` |

> **为什么不用 `nproc`**：采集 step 带 `--cpus-per-task=1`，cgroup（Linux 的资源
> 隔离机制）会把 `nproc` 限制成 1。而 `/proc/stat`、`/proc/meminfo` 是全节点视角、
> 不受 cgroup 约束。这是在受限容器里测宿主机负载的通用正确姿势。

### 3.4 GPU 行

```
     [0] GH200 120GB   36°C  140W util   1% [░░░░░░░░░░] ▁▁▂▁▁▃▂▁ mem  86.8/95.6G [█████████░]  kangli.u6gb:python/219342
      │       │          │     │       │         │           │          │              │                  │
      │       │          │     │       │         │           │          │              │                  └ 用户:进程名/PID
      │       │          │     │       │         │           │          └ 显存进度条    └ 已用/总显存
      │       │          │     │       │         └ util 进度条 └ sparkline：最近 8 次采样的利用率曲线
      │       │          │     │       └ 瞬时利用率
      │       │          │     └ 瞬时功耗（没在算却还烧几十瓦时标红）
      │       │          └ 核心温度
      │       └ 型号
      └ 节点内 GPU 序号
```

**sparkline**（迷你趋势曲线，用 `▁▂▃▄▅▆▇█` 八级方块画在一行内）是对低频采样的
补偿。2 分钟采一次的话，单个瞬时值分不清「这卡一直满载」和「刚刚才起来」，
但一条 8 点的曲线可以。需要 8 次采样（默认 16 分钟）才会填满。

### 3.5 steps 行

```
     steps: 5975573.16 bash(0:59)
     steps: 只有 .batch   · 没有计算 step 在跑
```

**step（作业步）** 是 SLURM 里比 job 更细的执行单位：一次 `srun` 就是一个 step。
这一行回答的是「这个分配上到底有没有人在跑东西」，与 GPU 状态互为印证：

- 有 GPU 在 run，但 steps 只有 `.batch` → 进程是从 batch 脚本直接起的，正常
- 没有任何计算 step，且卡全 idle → 分配纯粹空占着
- 有 step 但卡全 held → step 起来了，但卡在初始化或数据加载

自动排除三类噪音：gtop 自己的采集 step（命名为 `gtop-probe`）、`.extern`
（SLURM 给每个作业挂的容器 step）、以及单独标注的 `.batch`（批处理脚本本身）。

### 3.6 底栏与 pending

```
  sample 18:34:38 (1.1s)   next 1m47s   interval 120s   [q]uit ...
  pending 2: 5980745(4N)   5980502(4N, Priority, est 19:40)
```

`(1.1s)` 是本次采集的实际耗时。这个数波动很大（实测 1.1 s 到 10.9 s），主要取决于
提交时 slurmctld 的负载，不是固定成本 —— 这也是采样间隔不该做到秒级的原因之一。

pending 行显示排队中的作业：节点数、阻塞原因（`Priority` / `Resources` 等）、
以及 SLURM 回填调度器给出的预计启动时刻（`est HH:MM`，常为 `N/A`，因为控制器
只有在做回填计算后才会填这个值）。


## 4. 参数与按键

| 参数 | 默认 | 说明 |
|------|------|------|
| `-i, --interval N` | `120` | 采样间隔（秒）。TUI 里可用 `+` / `-` 实时增减 30 秒 |
| `-1, --once` | 关 | 采一次、打印、退出。有数据且无错误时退出码 0，否则 1 |
| `-j, --jobid ID` | 全部 | 只看指定作业 |
| `-u, --user NAME` | `$USER` | 换用户（只能看自己有权限的作业） |
| `-c, --show-cmd` | 关 | 每个 GPU 进程多展一行完整命令行 + 主机侧 RSS |
| `--ascii` | 关 | 用纯 ASCII 替代方块字符，终端不支持 Unicode 时用 |
| `--no-color` | 关 | 关颜色（仅 `--once`；重定向到文件时自动关） |
| `--timeout N` | `90` | 单次 srun 的超时秒数，超时该作业标记为采集失败但不影响其他作业 |
| `--log PATH` | 关 | 每次采样追加一行 JSONL 摘要 |

| 按键 | 作用 |
|------|------|
| `q` / `Esc` | 退出 |
| `r` | 立刻重采（不等倒计时） |
| `p` / 空格 | 暂停/恢复自动采样 |
| `+` / `-` | 采样间隔 ±30 秒（下限 10 秒，上限 3600 秒） |
| `c` | 展开/收起 GPU 进程命令行 |
| `↑` `↓` `PgUp` `PgDn` | 滚动（节点多时用） |

界面每 250 ms 重绘一次，所以即使 2 分钟才采一次数据，倒计时和「`◉` sampling…」
提示每秒都在动，不会让人怀疑程序是不是卡死了。终端宽度分三档自动降级：
≥132 列显示全部；104–131 列去掉型号与 uptime；<104 列再去掉功耗与 sparkline。


## 4.5 `lock` —— 把「空闲」和「无主」分开

`idle` 只说明这张卡此刻没在算，**不说明它没有主人**。多个会话共用同一批 chain
分配时，把 idle 读成「可抢」会出事：2026-08-12 `wm_ft_b` 首启就是在 init 与第一个
backward 之间，被邻居的 88 GiB 预分配挤爆的。`lock` 就是用来记「这张卡是谁的地盘」。

```bash
gtop lock nid010272 nid010367 nid010384 -j 5992007 --who crps-wm_ft --note "默认地盘"
gtop unlock nid010547          # 整节点撤销
gtop unlock nid010367:1,2      # 只放掉两张卡，整节点锁自动收窄成剩下的
gtop locks                     # 列出
gtop locks --prune             # 清掉 job 已结束的 stale 锁
```

节点写法支持 SLURM 的方括号（`nid[011165-011167]`），冒号后跟卡号选具体的卡。
注册表是一个共享 JSON（默认与 gtop 同目录的 `gpu_locks.json`，`$GTOP_LOCKS` 可改），
写入走 tmp + `os.replace` 原子替换。

### 三条设计约束（改它之前必须知道）

**① `lock` 只覆盖 `idle`，绝不覆盖 `run` / `held`。**
本工具的全部价值在于「占着但没在算」这个判据。如果 lock 能盖住 `held`，
就把真浪费藏起来了。锁定的语义只是「这张**空**卡的空是有意的」。实测：

```
nid010272 ▪ crps-wm_ft          ← 整节点锁
  [0] util 0%  mem 83.7/95.6G   held          ← 占着显存不算，照报 held
nid010367 ▪ crps-wm_ft
  [0] util 0%  mem 41.5/95.6G   held          ← 同上
  [1] util 0%  mem  0.0/95.6G   ▪lock:crps-wm_ft   ← 空卡，锁生效
nid010547                                      ← 已 unlock
  [0] util 0%  mem  0.0/95.6G   idle          ← 真正无主的空卡
```

**② 锁可以挂在 jobid 上，job 一结束就变 `stale`。**
`-j <jobid>` 挂载后，该 job 不在 RUNNING 时锁自动失效：**按 `idle` 计数**，
但在卡旁标 `(stale lock)`、在摘要行标 `stale N`。
一把过期的锁如果继续显示 lock，会永久藏起一张真空闲的卡 —— 那比没有锁更危险。
不带 `-j` 的手动锁永久有效，要自己 `unlock`。

**③ 锁定的空卡不计入「空转功耗」。**
它空着是有意的，算成浪费会让那个数字失去「该去救哪张卡」的含义。
`--log` 的 JSONL 里 `gpu.lock` 是单独一栏，事后想把它算回浪费也算得回来。

摘要行相应多两栏：

```
run 16/32 (50%)  held 5  idle 4  lock 7   ·  空转功耗 1.06 kW
                                 ↑ 锁之前这里是 idle 11、1.66 kW
```

**锁是协作性的**：它改变显示、也改变各会话自律闸门的判据，但并不真的阻止 CUDA 分配。


## 5. 用 `--log` 算清楚空转了多久

```bash
gtop -i 120 --log ~/gtop_5975573.jsonl
```

每行一条摘要（几百字节，2 分钟一行，对 Lustre 的压力可忽略）：

```json
{"t": 1786386993.9, "dur": 1.11, "jobs": {"5975573": {"name": "u6gb-4-node-chain",
 "nodes": 4, "gpu": {"run": 1, "held": 3, "idle": 12}, "watt_wasted": 1519.8,
 "cpu_mean": 0.38, "steps": 1}}}
```

事后统计（纯本地小文件读取，登录节点上跑没问题）：

```bash
python3 - <<'EOF'
import json, collections
tot = collections.Counter(); n = 0; wh = 0.0; span = [None, None]
for line in open('/home/u6gb/kangli.u6gb/gtop_5975573.jsonl'):
    r = json.loads(line); n += 1
    span = [span[0] or r['t'], r['t']]
    for j in r['jobs'].values():
        if 'gpu' not in j: continue
        tot.update(j['gpu'])
        wh += j['watt_wasted'] * 120 / 3600.0      # 采样间隔 120 秒 -> Wh
hours = (span[1] - span[0]) / 3600.0
gpu_tot = sum(tot.values())
print(f"观察窗口 {hours:.1f} h，{n} 次采样")
print(f"GPU-采样点: run {tot['run']} / held {tot['held']} / idle {tot['idle']}")
print(f"真正在算的比例: {100*tot['run']/gpu_tot:.1f}%")
print(f"非计算状态累计电耗: {wh/1000:.2f} kWh")
EOF
```

「有多少 GPU-hours 其实在空转」这个问题，靠印象是答不准的；这条链路能给出数字。


## 6. 为什么这不违反 BriCS 的规则

BriCS 在 2026-05-08 因为元数据风暴停过全组的作业，`CLAUDE.md` 里列了 9 条
anti-pattern。逐条对照：

| BriCS 禁止的模式 | gtop 的做法 |
|------------------|-------------|
| 递归 `ls` / `find` / `du -sh` / 大目录 glob | 一次都没有。节点端只读固定路径的 `/proc/*` |
| 训练中间产物写 Lustre | 不写任何东西。`--log` 默认关闭，开了也是几百字节/2 分钟，且路径由用户指定 |
| 计算节点从 Lustre 读脚本/权重 | 采集脚本经 stdin 传入，计算节点零 Lustre 访问 |
| 登录/计算节点上常驻 agent、自动重启 wrapper | 无常驻进程。采集 step 跑 0.4 秒即退出；关掉 gtop 就彻底没了 |
| 高频轰炸 slurmctld | 默认 2 分钟一次 `squeue` + 一次 `srun`。作为对比，`watch -n2 squeue` 的频率是它的 60 倍 |
| `scancel` | 代码里根本没有这个词。gtop 只读，不改变任何作业状态 |

另外 gtop **只能看自己的作业**：`squeue -u $USER` 只列自己的，`srun --jobid` 也只能
附着到自己有权限的分配上。别人的节点它既看不见也进不去。


## 7. 已知限制

| 限制 | 说明 |
|------|------|
| 只覆盖 RUNNING 的分配 | PENDING 作业还没有节点，只能在 pending 行显示排队信息 |
| 采样有 1–11 秒延迟 | srun step 的创建开销，取决于 slurmctld 当时的负载 |
| sparkline 需要 8 次采样才填满 | 默认间隔下是 16 分钟。急着看趋势就 `-i 30` |
| 训练把 CPU 占满时采集可能变慢 | `--overlap` 会共享资源，极端情况下靠 `--timeout` 兜底，该作业标记「采集失败」，其余作业不受影响 |
| 多卡负载不均不会主动告警 | 目前只呈现，不诊断。要判断 DDP 是否掉了一路，看四张卡的 util 是否同步 |
| GPU 进程的用户名依赖 `/proc` 可读 | 同节点同 uid 一定可读；跨 uid 场景（本集群整节点独占，不会出现）会退化成显示 PID |


## 8. 文件

| 路径 | 作用 |
|------|------|
| `/projects/public/u6gb/tasks/gtop_20260810T182343Z/gtop` | 主程序（Python 3，只用标准库，无第三方依赖） |
| `/projects/public/u6gb/tasks/gtop_20260810T182343Z/probe.sh` | 节点端采集脚本，被主程序读入后经 stdin 送到计算节点执行 |
| `/projects/public/u6gb/tasks/gtop_20260810T182343Z/selftest.py` | 纯函数自测，30 项，不需要 allocation 也能跑 |
| `/projects/public/u6gb/tasks/gtop_20260810T182343Z/README.md` | 本文件 |
| `/projects/public/u6gb/.local/bin/gtop` | 软链，使 `gtop` 可在任意目录直接调用 |

`probe.sh` 是采集逻辑的唯一真相源，`gtop` 启动时读它。改采集只改 `probe.sh`，
不需要动主程序。

改完跑一遍自测（一秒钟出结果，覆盖 Slurm 时间解析、时长渲染、GPU 三态判定、
进度条、sparkline、单位换算）：

```bash
python3 /projects/public/u6gb/tasks/gtop_20260810T182343Z/selftest.py
# ✓ 全部通过（Slurm 时间解析 / 时长渲染 / GPU 三态 / 进度条 / sparkline / 单位换算）
```

自测只覆盖不依赖集群的纯函数。采集链路本身（srun 附着、JSON 解析、多节点汇聚）
需要有 RUNNING 的分配才能验证，用 `gtop -1` 打一枪看有没有数据即可。
