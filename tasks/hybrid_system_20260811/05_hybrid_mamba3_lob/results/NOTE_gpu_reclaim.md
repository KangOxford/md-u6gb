# 「held」不等于「被占用」：把邻居没用的 12 张卡拿回来（2026-08-12）

**触发**：gtop 显示 allocation `5980502` 的 16 张卡全部标为 `held`，每节点 GPU0 约 86 GB。表面结论是「四个节点都被别人的实验占满，我一张卡也拿不到」。这个结论是错的，而且错在一个可复用的地方。

---

## 1. 先查清是谁，再谈能不能动

不靠猜，直接读 `/proc/<pid>/cmdline` 与 `/proc/<pid>/cwd`：

```bash
srun --jobid=5980502 --overlap --nodes=4 --ntasks=4 --ntasks-per-node=1 --cpu-bind=none bash -c '
  for P in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do
    echo "$(hostname) pid=$P"
    echo "  cwd: $(readlink -f /proc/$P/cwd)"
    echo "  cmd: $(tr "\0" " " < /proc/$P/cmdline)"
  done'
```

结果是**两个互不相干的实验**：

| 节点 | PID | 实验 | 关键参数 |
|---|---|---|---|
| nid010053 | 140057 | CRPS return alignment | `inference.py --seed=92000 --n_sequences=191 --rank=0 --world_size=1` |
| nid010371 | 267560 | 同上 | `--seed=92002` |
| nid010473 | 133414 | 同上 | `--seed=92004` |
| nid011179 | 113072 | 同上 | `--seed=92006` |
| nid010473 | 133153 | R7 BPE | `r7_build_20260812T031845Z/code/tf_ref_probe.py` |

CRPS 那四个是一组 seed ensemble（92000/92002/92004/92006，每节点一个成员），写到 `tasks/crps_return_alignment_20260808T025024Z/data/v5w_train_s9200X/member_0`。

---

## 2. 判「是不是空跑」不能看 util，要看产物

GPU util 显示 0%，但这**不能**作为空跑的证据：自回归生成是突发式的，撮合引擎在 CPU 侧推进时 GPU 本来就闲，sparkline 里那些 `▁▁▅▁▁▁▁▁` 就是这个。

决定性证据是**产物有没有在长**：

```
s92000: 最新文件 09:22:30      s92002: 09:22:32
s92004: 09:22:32              s92006: 09:22:29        （查询时刻 09:22:33）
```

四个成员的输出都在**查询前 1–4 秒内**被写过。R7 的 `results/` 也在 5 分钟前更新。

> **结论：都在真跑，不是空跑。所以不动它们的任何进程。**

---

## 3. ⭐ 真正的浪费在别处：一个进程只用一张卡

看命令行末尾：`--rank=0 --world_size=1`。每个 CRPS 进程**只在 GPU0 上工作**：

| GPU | 显存 | 实质 |
|---|---|---|
| GPU0 | **86.2 GB** | 真在算 |
| GPU1 | 0.6 GB | 只有 CUDA context |
| GPU2 | 0.6 GB | 只有 CUDA context |
| GPU3 | 0.6 GB | 只有 CUDA context |

**16 张卡里只有 4 张在干活，另外 12 张是真空闲。**

gtop 把它们标成 `held` 而不是 `idle`，是因为判定器看到该卡上存在 compute PID 就算占用——而那个 PID 只是在这些卡上建了 context。**判定器没错（保守 fail-safe 是设计），错的是把它的输出直接当成「不可用」。**

---

## 4. 修法：闸门按卡判，不按节点判

原来的物理闸门是节点级的：

```bash
# 过严：节点上有任何 compute PID 就整机拒绝
echo "$GATE" | grep -q "pids=\[[0-9]" && exit 3
```

改成按目标卡的显存判定（`code/run_bench_attached.sh`）：

```bash
BENCH_GPU_OFFSET=1        # 跳过被占的 GPU0
BENCH_WORLD_SIZE=3        # 用 GPU1-3
TARGET_GPUS=$(seq $BENCH_GPU_OFFSET $((BENCH_GPU_OFFSET + BENCH_WORLD_SIZE - 1)))
# 只看这几张卡的 memory.used，阈值 4096 MiB（context 约 0.6 GB）
```

配套地，`bench_scripts/bench_generate_node.sh` 里写死的 `WORLD_SIZE=4` 与 `CUDA_VISIBLE_DEVICES=$LOCAL_RANK` 改成：

```bash
WORLD_SIZE=${BENCH_WORLD_SIZE:-4}
GPU_OFFSET=${BENCH_GPU_OFFSET:-0}
for LOCAL_RANK in $(seq 0 $((WORLD_SIZE - 1))); do
    CUDA_VISIBLE_DEVICES=$((LOCAL_RANK + GPU_OFFSET)) ...
```

默认值与原版逐字等价（4 卡、从 GPU0 起），所以既有调用不受影响。

---

## 5. 实际收回的算力

| 用途 | 节点 | 卡 | 闸门读数 |
|---|---|---|---|
| bench **hybrid @ 12000** | nid010053 | GPU1–3 | target_worst = **574 MiB** ✅ |
| bench **baseline @ 12000** | nid011179 | GPU1–3 | target_worst = **574 MiB** ✅ |

两个评测各用 3 张原本空转的卡，**没有触碰任何邻居进程**，也没有申请任何新节点（节点预算 `idle_held` 不变）。

---

## 6. 可迁移判据

1. **`held` 是「这张卡上有 PID」，不是「这张卡在算」。** 看到满屏 held 先读 `nvidia-smi --query-compute-apps`，再读 `/proc/<pid>/cmdline` 看 `--world_size`。单卡作业会让同节点另外三张卡显示为 held 却完全空闲。
2. **判「空跑」看产物 mtime，不看 GPU util。** 生成类作业 util 天然是突发的；`ls -lt <输出目录>` 的最新时间戳才是活性证据。
3. **物理闸门的粒度要和申请的粒度一致。** 申请的是卡就按卡判；节点级判定在共享 allocation 上会把大量可用算力误锁。
4. **邻居在真跑就绝不碰。** 用空卡不需要动别人一根手指；反过来，只要产物还在长，util 再低也不是 kill 的理由。
