#!/usr/bin/env python3
"""训练路径与存点路径分别生成什么 NCCL 集合？直接读 HLO 里的 replica_groups。

为什么这个检验优于「跑起来看会不会挂」
------------------------------------
replica group 是**编译期**决定的，写在 HLO 里。问「它生成了什么集合」不需要
真 GPU、不需要多节点、不需要等死锁发生 —— 用 --xla_force_host_platform_device_count
造 16 个假设备，在登录节点上几秒就能读出来。

而「跑起来看会不会挂」既慢又只能给出概率性的否定证据：2 节点跑 80 次没挂，
既不能证明它不会挂，也说不清为什么。

要回答的问题
-----------
train.py:_reshard_for_ckpt 每次存点调 broadcast_one_to_all。它的注释说这个
host roundtrip 是为了「avoid creating a new NCCL clique」。那么：

  存点路径的 replica_groups 与训练路径的一样吗？
  一样 -> XLA 复用同一个 communicator，「新建 clique」的说法不成立
  不一样 -> 存点确实在用训练从不使用的通信域，注释里要避免的事正在发生

用法（登录节点即可，纯 CPU）：
    python check_replica_groups.py --devices 16 --nodes 4
"""
import argparse
import os
import re
import sys


def groups_of(hlo: str):
    """从 HLO 文本里抽出所有集合算子的 replica_groups。"""
    out = []
    # XLA 有两种写法，都要认：
    #   老式   replica_groups={{0,1,2,3},{4,5,6,7}}
    #   紧凑式 replica_groups=[4,4]<=[4,4]T(1,0)      <- 现在用的是这种
    # 首版只认前者，于是两边都落进「未解析」的兜底字符串，
    # 判定成「两侧相同」—— **一个由解析失败伪造出来的结论**。
    for m in re.finditer(
            r"(all-reduce|all-gather|reduce-scatter|collective-permute|all-to-all)"
            r"[^\n]*?replica_groups=(\[[^\]]*\](?:<=\[[^\]]*\](?:T\([0-9,]*\))?)?"
            r"|\{[^\n]*?\}\})", hlo):
        out.append((m.group(1), m.group(2)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", type=int, default=16)
    ap.add_argument("--nodes", type=int, default=4)
    args = ap.parse_args()

    # 必须在 import jax 之前设
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") +
                               f" --xla_force_host_platform_device_count={args.devices}")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    devs = np.array(jax.devices())
    n_dev = len(devs)
    n_nodes = args.nodes
    n_local = n_dev // n_nodes
    assert n_local * n_nodes == n_dev
    print(f"假设备 {n_dev} 个，按 {n_nodes} 节点 x {n_local} 卡布局\n")

    # ── 1. 训练路径：2D 分层 mesh，数据沿两轴分片，梯度隐式 AllReduce ──────
    train_mesh = Mesh(devs.reshape(n_nodes, n_local), axis_names=("nodes", "gpus"))
    repl_t = NamedSharding(train_mesh, P())
    data_t = NamedSharding(train_mesh, P(("nodes", "gpus")))

    w = jax.device_put(np.ones((512, 512), np.float32), repl_t)
    x = jax.device_put(np.ones((n_dev * 4, 512), np.float32), data_t)

    def loss_fn(w, x):
        return jnp.sum((x @ w) ** 2)

    with jax.set_mesh(train_mesh):
        grad_jit = jax.jit(jax.grad(loss_fn),
                           in_shardings=(repl_t, data_t), out_shardings=repl_t)
        hlo_train = grad_jit.lower(w, x).compile().as_text()

    print("=== 训练路径（梯度 AllReduce）")
    gt = groups_of(hlo_train)
    for op, g in gt[:6]:
        print(f"  {op:16s} {g[:110]}")
    if not gt:
        print("  (没有集合算子)")

    # ── 2. 存点路径：broadcast_one_to_all 内部那个 mesh 与 psum ────────────
    # 照抄 jax/experimental/multihost_utils.py:68-86
    bcast_mesh = Mesh(devs.reshape(n_nodes, n_local),
                      axis_names=("processes", "local_devices"))
    inp_sh = NamedSharding(bcast_mesh, P("processes"))
    y = jax.device_put(np.ones((n_nodes, 512, 512), np.float32), inp_sh)

    def _psum(x):
        return jax.lax.psum(x, axis_name=None) if False else jnp.sum(x, axis=0)

    with jax.set_mesh(bcast_mesh):
        b_jit = jax.jit(lambda a: jnp.sum(a, axis=0),
                        in_shardings=inp_sh,
                        out_shardings=NamedSharding(bcast_mesh, P()))
        hlo_bcast = b_jit.lower(y).compile().as_text()

    print("\n=== 存点路径（broadcast_one_to_all 的 psum）")
    gb = groups_of(hlo_bcast)
    for op, g in gb[:6]:
        print(f"  {op:16s} {g[:110]}")
    if not gb:
        print("  (没有集合算子)")

    # ── 3. 判定 ───────────────────────────────────────────────────────────
    st = {g for _, g in gt}
    sb = {g for _, g in gb}
    print("\n=== 判定")
    if not st or not sb:
        print("  无法判定：有一侧没有集合算子")
        return 1
    common = st & sb
    only_b = sb - st
    print(f"  训练侧 replica_groups 种类: {len(st)}")
    print(f"  存点侧 replica_groups 种类: {len(sb)}")
    print(f"  共用: {len(common)}   存点独有: {len(only_b)}")
    if only_b:
        print("\n  >>> 存点路径使用了训练路径从不使用的通信域：")
        for g in list(only_b)[:4]:
            print(f"      {g[:150]}")
        print("\n  即：注释里要避免的『新建 NCCL clique』确实在每次存点时发生。")
    else:
        print("\n  >>> 存点路径的通信域是训练路径的子集 —— XLA 会复用同一个"
              " communicator，\n      『新建 clique』的说法**不成立**，机制要另找。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
