# hybrid 反复挂死的根因追查（2026-08-13，进行中）

> **证据强度标注**：§1–§3 是**已证实**的（代码事实 + 时间相关 + 现场特征）；
> §4 的最小复现**尚未触发死锁**，所以「`broadcast_one_to_all` 就是那个触发点」
> 目前是**强旁证，不是定论**。§5 的修复理由不依赖 §4 是否复现。

---

## 1. 现象

2026-08-13 hybrid 臂在 2k 上下文训练中三次挂死，可定位的两次：

| 运行 | 最后 checkpoint | 下一次该在 | 挂死 | 判定者 |
|---|---|---|---|---|
| 1 | 2571 @ 10:21:26 | **10:36:26** | 10:36（batch 3760） | 训练自带 StepWatchdog |
| 2 | 3457 @ 12:21:27 | **12:36:27** | 12:36–12:48 | 同上（阈值已收到 600s） |

看门狗自己的判词：

```
FATAL: Step watchdog timeout (1800s) at epoch 0 batch 3760. Likely NCCL deadlock.
```

**两次都落在 900 秒存点周期的边界上。** 若挂死时刻在时间上均匀分布，两次都撞进
±30 秒窗口的概率约 `(60/900)² ≈ 0.4%`。

## 2. 现场特征（挂死中途实测）

| 观测 | 排除了什么 |
|---|---|
| 主线程 `wchan = futex_wait_queue` | 不是在算，是在等锁 |
| 1300 线程 `S`，**恰好 4 个 `R`** | 4 = 每 GPU 一个 CUDA 自旋线程 |
| **0 个 orbax / checkpoint fd** | **不是写盘挂死** —— 卡在开文件之前 |
| `nvidia-smi util = 100%` | **这一项是误导**，见下 |
| 四个节点日志同一秒停写 | 同步停滞，不是单点故障 |

> **GPU util 100% 不等于在计算。** NCCL 集合通信在等对端时是忙等（spin-wait），
> 驱动照报满载。我最初正是据此判断 hybrid 健康，判错了。可靠判据只有**进度在不在变**。

## 3. 代码事实：workaround 被它自己调用的函数抵消

`src/lob/train.py:_reshard_for_ckpt`，**每次存点都走**：

```python
host_s = jax.device_get(s)
host_s = broadcast_one_to_all(host_s)          # ← 这一行
return jax.device_put(host_s, state_shardings)
```

它上面的注释写着：

> Reshard via host roundtrip — **avoids creating a new NCCL clique that can
> deadlock on CXI (Slingshot)** when memory registrations go stale.

但 JAX 的 `broadcast_one_to_all`（`multihost_utils.py:68-86`）内部干的正是这件事：

```python
devices = np.array(jax.devices()).reshape(process_count, local_device_count)
global_mesh = jax.sharding.Mesh(devices, ('processes', 'local_devices'))   # 新建 mesh
with jax.set_mesh(global_mesh):
    out = jax.jit(_psum, out_shardings=P())(in_tree)                       # 全设备 AllReduce
```

训练用的是 **2D 分层 mesh** `('nodes','gpus')`（日志：`2D hierarchical mesh: (4 nodes, 4 gpus)`），
psum 分别沿两个轴做，replica group 是「同节点 4 卡」与「跨节点同序号卡」。
而这里是**全 16 设备的一次 AllReduce** —— **replica group 不同 = 不同的 NCCL
communicator**。

**结论：注释里要避免的事，正是它调用的函数在做，而且每 900 秒做一次。**

仓库里还有一处旁证：`train.py:557` 有个 `CXI warm` 补丁，恢复时先做一次 dummy
save「趁 CXI 端点还热」，注释说否则「第一次真实存点会撞上 stale CXI
registrations → SIGABRT (RC:265)」。**前人撞过同一堵墙，只补了 resume 那一次。**

## 4. 最小复现：目前尚未触发（诚实记录）

`code/repro_ckpt_deadlock.py` + `run_repro.sh`，2 节点 × 4 GPU，同一套 conda /
NCCL 2.29.3 / `NCCL_BUFFSIZE=2MB`，构造同形状的分层 mesh，交替「训练几步 → 走一次
reshard」。

| 配置 | 结果 |
|---|---|
| 8 叶子 / 67 MB / MEM_FRACTION 0.12 / 80 次 | 无挂死，**180 ms** 一次 |
| 300 叶子 / 1.5 GB / MEM_FRACTION 0.55 / 加 6 GB 分配器搅动 | 无挂死，**8 秒**一次 |

**没能复现。** 可能缺的成分：4 节点（跨节点组从 2 变 4）、真实 87 GB/卡的显存压力、
Orbax 真的在并发写 Lustre、以及真实 TrainState 的树结构（params + Adam m,v）。

但这一轮量出了一个**独立于死锁的问题**：

> `broadcast_one_to_all` 的开销随叶子数强烈增长：8 叶子 180 ms → 300 叶子 **8 秒**。
> 它把整棵树 jit 成一次融合 psum。真实 TrainState 有数百个叶子，
> **每次存点白烧数秒，纯属多余**。

### 两个把复现器带偏的坑（都已修，值得记）

1. **静默退回 CPU**：conda 路径用了 CLAUDE.md 里那条 `/projects/s5e/quant/miniforge3`
   （对 s5e 账号成立，本会话是 u6gb），`Permission denied` 后退到系统 python，
   跑完 60 次报「无挂死」—— 测的是 CPU 上的 Gloo。
   **复现器最危险的失败模式不是崩，是安静地测了别的东西。** 已加硬断言。
2. **每进程只看到 1 张 GPU**：`jax.distributed.initialize` 不传 `local_device_ids`
   时按 `SLURM_LOCALID` 推断，`--ntasks-per-node=1` 布局下每进程 1 卡。
   训练侧 `runtime/train.py:373` 正是显式传的。

## 5. 修法（理由不依赖 §4 是否复现）

三条路径正在同一批节点上对比（`run_repro_sweep.sh`）：

| 模式 | reshard 实现 | 动机 |
|---|---|---|
| A | `device_get → broadcast_one_to_all → device_put` | 现状 |
| B | `device_get → device_put` | 去掉那次全设备 AllReduce；验证 bit-level 差异是否真是问题 |
| C | 完全不 reshard，直接交给 Orbax | Orbax 本就支持多主机分片数组 |

`broadcast_one_to_all` 存在的理由是注释里那句：`device_get` 后各主机可能有
bit-level 差异（Muon Newton-Schulz、NCCL 归约非确定性），广播 rank0 的副本好让
`device_put` 的 assert_equal 通过。**若 C 可行，这个理由整个消失**——不做 host
roundtrip 就没有 assert_equal 要满足。

## 6. 顺带修掉的可观测性缺口

三次挂死只有一次拿到了死因，另外两次被下一轮启动覆盖：日志名是
`training_<allocid>_node<N>.log`，同一 allocation 上续训会 `exec >` 截断它。

> **自愈链路越好用，事后可诊断性越差** —— 它默默重启，把现场一起清了。

已在两个 launcher 里加了起飞前归档到 `logs_lobs5/ctx2k_*/archive/`。
