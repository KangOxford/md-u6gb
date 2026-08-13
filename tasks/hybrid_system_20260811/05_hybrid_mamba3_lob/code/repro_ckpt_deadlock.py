#!/usr/bin/env python3
"""checkpoint 存点路径的 NCCL 死锁：最小复现 + 三种修法对比

背景
----
2026-08-13 hybrid 臂在 2k 上下文训练中三次挂死，两次可定位的挂死都落在
checkpoint 存点边界上（900s 周期的 ±30s 内，若挂死时间均匀分布，两次都撞进去
的概率约 (60/900)^2 ≈ 0.4%）。现场特征：

    主线程 wchan = futex_wait_queue
    1300 线程 S，恰好 4 个 R（每 GPU 一个 CUDA 自旋线程）
    0 个 orbax/checkpoint fd            ← 卡在写盘**之前**
    nvidia-smi util 100%                ← NCCL 集合是忙等，这一项是误导

嫌疑代码 train.py:_reshard_for_ckpt

    host_s = jax.device_get(s)
    host_s = broadcast_one_to_all(host_s)      # ← 这一行
    return jax.device_put(host_s, state_shardings)

它的注释写着「Reshard via host roundtrip —— avoids creating a new NCCL clique
that can deadlock on CXI (Slingshot)」。但 JAX 的 broadcast_one_to_all 内部
（multihost_utils.py:68-86）干的正是这件事：

    devices = np.array(jax.devices()).reshape(process_count, local_device_count)
    global_mesh = jax.sharding.Mesh(devices, ('processes', 'local_devices'))
    with jax.set_mesh(global_mesh):
        out = jax.jit(_psum, out_shardings=P())(in_tree)

**这个 workaround 被它自己调用的函数抵消了。** 训练用的是 2D 分层 mesh
('nodes','gpus')，psum 分别沿两个轴做，replica group 是「同节点 4 卡」与
「跨节点同序号卡」；而这里是全 N 设备的一次 AllReduce —— **不同的 replica
group = 不同的 NCCL communicator**，每次存点重建一次。

本脚本要回答三个问题
------------------
Q1  把存点频率压到几秒一次，死锁能不能在几分钟内复现？（能 → 根因坐实，且
    从此可按需触发、可验证修复）
Q2  去掉 broadcast 之后还能不能正常 device_put？（bit-level 差异是不是真问题）
Q3  完全不 reshard、直接把已分片的 state 交给 Orbax，行不行？

用法
----
  srun --jobid=X --overlap --nodes=2 --ntasks=2 --ntasks-per-node=1 \
       python repro_ckpt_deadlock.py --mode A --iters 200

  --mode A  现状：device_get -> broadcast_one_to_all -> device_put
  --mode B  去掉 broadcast：device_get -> device_put
  --mode C  完全不 reshard：直接用原 state
"""
import argparse
import os
import sys
import time

import numpy as np


def log(msg):
    """带 rank 与时刻的行缓冲输出。挂死时最后一行就是现场。"""
    import jax
    try:
        r = jax.process_index()
    except Exception:
        r = "?"
    print(f"[repro r{r} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["A", "B", "C"], default="A")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--mb", type=int, default=64,
                    help="被搬运的 pytree 大小（MB）。真实 train state 约 400MB，"
                         "但死锁在 communicator 建立上，与大小无关；小一点跑得快。")
    ap.add_argument("--train-steps", type=int, default=3,
                    help="每次存点之间跑几步「训练」，用来在分层 mesh 上建立"
                         "正常的 communicator —— 复现必须包含这一步，"
                         "因为要触发的是「两套 communicator 交替使用」。")
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="单次存点超过这个秒数就判为挂死并退出")
    ap.add_argument("--leaves", type=int, default=8,
                    help="pytree 叶子数。真实 train state 是 params+adam(m,v) 共数百个，"
                         "而 broadcast_one_to_all 会把整棵树 jit 成一次 psum —— "
                         "叶子数直接决定这个融合集合的复杂度。")
    ap.add_argument("--churn-mb", type=int, default=0,
                    help="每次存点之间额外分配/释放多少 MB，制造分配器压力。"
                         "原始注释说死锁发生在 memory registrations go stale，"
                         "而注册失效需要 BFC 分配器回收复用 —— 不承压就不会发生。")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    coord = os.environ.get("JAX_COORDINATOR_ADDRESS")
    pid = int(os.environ.get("SLURM_PROCID", "0"))
    nproc = int(os.environ.get("SLURM_NNODES", "1"))
    if nproc > 1:
        # local_device_ids 必须显式给。不给的话 JAX 的 Slurm 自动推断按
        # SLURM_LOCALID 判「每个 task 一张卡」，于是 --ntasks-per-node=1 的布局下
        # 每个进程只看到 1 张 GPU —— 训练侧 runtime/train.py:373 正是这么传的。
        ngpu = int(os.environ.get("GPUS_PER_NODE", "4"))
        jax.distributed.initialize(coordinator_address=coord,
                                   num_processes=nproc, process_id=pid,
                                   local_device_ids=list(range(ngpu)))
    log(f"mode={args.mode} processes={jax.process_count()} "
        f"local_devices={jax.local_device_count()} total={len(jax.devices())}")

    # 硬断言：拿不到 GPU 就失败，绝不退回 CPU。
    # 2026-08-13 首跑因为 conda 路径没权限静默退到系统 python，跑完 60 次报「无
    # 挂死」—— 测的是 CPU 上的 Gloo，与要复现的 NCCL/CXI 死锁毫无关系。
    # **复现器最危险的失败模式不是崩，是安静地测了别的东西。**
    plats = {d.platform for d in jax.devices()}
    if plats != {"gpu"} or jax.local_device_count() < 2:
        log(f"!! 设备平台={plats} local={jax.local_device_count()} —— 不是多 GPU，"
            f"这个测试无效。中止。")
        return 2

    if jax.process_count() < 2:
        log("!! process_count < 2 —— broadcast_one_to_all 会直接返回，复现不了。"
            "必须用 --nodes>=2。")
        return 2

    # ── 训练侧的 2D 分层 mesh，与 sharding_utils.py 一致 ──────────────────
    devs = np.array(jax.devices()).reshape(jax.process_count(),
                                           jax.local_device_count())
    train_mesh = Mesh(devs, axis_names=("nodes", "gpus"))
    repl = NamedSharding(train_mesh, P())          # 参数复制
    data_sh = NamedSharding(train_mesh, P(("nodes", "gpus")))
    log(f"训练 mesh: {train_mesh.shape}  axis={train_mesh.axis_names}")

    # 一个「参数树」：几个大数组，总量 ~args.mb MB
    nleaf = args.leaves
    per = max(1, (args.mb * 1024 * 1024 // 4) // nleaf)
    side = int(np.sqrt(per)) or 1
    tree = {f"w{i}": jax.device_put(
                np.full((side, side), 0.01 * (i + 1), dtype=np.float32), repl)
            for i in range(nleaf)}
    nbytes = sum(int(np.prod(v.shape)) * 4 for v in tree.values())
    log(f"pytree: {nleaf} 个 {side}x{side} fp32，共 {nbytes/1e6:.1f} MB")

    # 一个「训练步」：在分层 mesh 上做数据并行 + AllReduce，建立正常 communicator
    @jax.jit
    def train_step(t, x):
        g = {k: (v * 1.0000001 + jnp.mean(x)) for k, v in t.items()}
        return g

    xb = jax.device_put(np.ones((jax.process_count() * jax.local_device_count(),
                                 256), dtype=np.float32), data_sh)

    from jax.experimental.multihost_utils import broadcast_one_to_all

    def reshard_A(t):     # 现状
        h = jax.device_get(t)
        h = broadcast_one_to_all(h)
        return jax.tree.map(lambda a: jax.device_put(a, repl), h)

    def reshard_B(t):     # 去掉 broadcast
        h = jax.device_get(t)
        return jax.tree.map(lambda a: jax.device_put(a, repl), h)

    def reshard_C(t):     # 完全不 reshard
        return t

    reshard = {"A": reshard_A, "B": reshard_B, "C": reshard_C}[args.mode]

    # ── 主循环：训练几步 → 存点一次，重复 ────────────────────────────────
    # 「训练几步 → 存点」这个交替是复现的关键：单独反复调 broadcast 不会重建
    # communicator（缓存命中），要在两套 replica group 之间来回切才会。
    # 自带看门狗。真死锁时 block_until_ready 永不返回，下面那句
    # `dt > args.timeout` 的检查**根本执行不到** —— 检查写在被卡住的那条路径上
    # 是没有意义的。必须用独立的守护线程，这也正是训练侧 StepWatchdog 的道理。
    # 在共租节点上尤其不能省：挂死的进程会一直占着别人的卡。
    import threading
    _wd = {"t": time.time(), "it": -1}

    def _watch():
        while True:
            time.sleep(5)
            idle = time.time() - _wd["t"]
            if idle > args.timeout:
                sys.stderr.write(
                    f"\n[repro] WATCHDOG: 第 {_wd['it']} 次存点已卡 {idle:.0f}s "
                    f"> {args.timeout}s —— 判为死锁，强制退出\n")
                sys.stderr.flush()
                os._exit(3)

    threading.Thread(target=_watch, daemon=True).start()

    slow = 0
    t_all = time.time()
    for it in range(args.iters):
        _wd["t"], _wd["it"] = time.time(), it
        for _ in range(args.train_steps):
            tree = train_step(tree, xb)
        jax.block_until_ready(jax.tree.leaves(tree)[0])

        # 分配器搅动：申请一块大的再丢掉，逼 BFC 回收复用，让已注册的显存区域
        # 失效。这是复现 "memory registrations go stale" 的直接手段。
        if args.churn_mb:
            n = args.churn_mb * 1024 * 1024 // 4
            side2 = int(np.sqrt(n)) or 1
            junk = jax.device_put(np.zeros((side2, side2), dtype=np.float32), repl)
            junk = (junk + 1.0).block_until_ready()
            del junk

        t0 = time.time()
        out = reshard(tree)
        jax.block_until_ready(jax.tree.leaves(out)[0]
                              if args.mode != "B" else
                              jax.tree.leaves(out)[0])
        dt = time.time() - t0

        if dt > args.timeout:
            log(f"!! 第 {it} 次存点耗时 {dt:.1f}s > {args.timeout}s —— 判为挂死")
            return 3
        if dt > 5.0:
            slow += 1
            log(f"   慢：第 {it} 次 {dt:.2f}s")
        if it % 20 == 0:
            log(f"iter {it}/{args.iters}  reshard={dt*1000:.0f}ms  "
                f"累计 {time.time()-t_all:.0f}s  慢次数={slow}")

    log(f"完成 {args.iters} 次，无挂死。总 {time.time()-t_all:.0f}s，慢 {slow} 次")
    return 0


if __name__ == "__main__":
    sys.exit(main())
