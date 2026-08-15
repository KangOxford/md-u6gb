#!/usr/bin/env python3
"""`predict_x1(draft=..., keep_tau=...)` 的单元测试。纯 CPU，合成 logits，秒级。

在花 GPU 时间之前把这十行逻辑验完。五条，其中两条带牙：

    K1  tau=0（关闭分支）与 tau=1.0（阈值到顶）几乎一致 —— 空操作对照（H4）
    K2  tau 很小 -> 输出几乎就是 draft
    K3  保留比例随 tau **单调下降**
    K4★ 句法非法的 draft token **一定**被替换（掩码把它的概率压成 0）
    K5★ 方向没写反：tau 小保留得多。写反的实现会让 K2 和 K5 同时失败
"""
import sys

import numpy as onp
import jax
import jax.numpy as jnp

sys.path.insert(0, "/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/"
                   "dfm-post-training-20260801/post_training/dfm/tools")
from dfm_sampler import predict_x1                                # noqa: E402

OK = []
B, L, V = 2, 260, 64
MSG = 26


def check(name, cond, detail=""):
    OK.append(bool(cond))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")


def setup(seed=0):
    rng = onp.random.default_rng(seed)
    # 每个 offset 只有一部分 token 合法，模拟句法掩码
    mask = onp.zeros((L, V), bool)
    for p in range(L):
        legal = rng.choice(V, size=8, replace=False)
        mask[p, legal] = True
    logits = jnp.asarray(rng.normal(0, 3.0, (B, L, V)).astype(onp.float32))
    # draft 全部取合法 token
    draft = onp.zeros((B, L), onp.int32)
    for p in range(L):
        legal = onp.nonzero(mask[p])[0]
        draft[:, p] = rng.choice(legal, size=B)
    return logits, jnp.asarray(mask), jnp.asarray(draft), rng


def run(logits, mask, draft, tau, key=0):
    return predict_x1(logits, draft, jnp, rng=jax.random.key(key), jax=jax,
                      mask=mask, draft=draft if tau > 0 else None,
                      keep_tau=tau)


def frac_kept(out, draft):
    o = onp.asarray(out)[:, 1:]
    d = onp.asarray(draft)[:, 1:]
    return float((o == d).mean())


def k1_canary():
    lg, mk, dr, _ = setup(1)
    a = run(lg, mk, dr, 0.0)
    b = run(lg, mk, dr, 1.0)
    same = float((onp.asarray(a) == onp.asarray(b)).mean())
    check("K1 tau=0 与 tau=1.0 一致（空操作对照）", same > 0.999,
          f"一致比例 {same:.5f}")


def k2_small_tau_is_draft():
    lg, mk, dr, _ = setup(2)
    out = run(lg, mk, dr, 1e-12)
    f = frac_kept(out, dr)
    check("K2 tau 极小 -> 输出几乎就是 draft", f > 0.99, f"保留 {f:.4f}")


def k3_monotone():
    lg, mk, dr, _ = setup(3)
    taus = [1e-12, 1e-6, 1e-3, 1e-2, 0.1, 0.5, 1.0]
    fs = [frac_kept(run(lg, mk, dr, t), dr) for t in taus]
    mono = all(fs[i] >= fs[i + 1] - 1e-9 for i in range(len(fs) - 1))
    check("K3 保留比例随 tau 单调下降", mono,
          "  ".join(f"{t:g}:{f:.3f}" for t, f in zip(taus, fs)))
    return taus, fs


def k4_illegal_draft_replaced():
    """★ 带牙 ★ 把 draft 的某些位置改成非法 token，它们必须 100% 被替换。"""
    lg, mk, dr, rng = setup(4)
    d = onp.asarray(dr).copy()
    bad_pos = onp.arange(5, L, 7)
    mk_np = onp.asarray(mk)
    for p in bad_pos:
        illegal = onp.nonzero(~mk_np[p])[0]
        d[:, p] = illegal[0]
    d = jnp.asarray(d)
    out = predict_x1(lg, d, jnp, rng=jax.random.key(9), jax=jax, mask=mk,
                     draft=d, keep_tau=1e-12)     # 最保守：几乎全保留
    o = onp.asarray(out)
    kept_illegal = int(sum((o[:, p] == onp.asarray(d)[:, p]).sum()
                           for p in bad_pos if p >= 1))
    # 对照：不给掩码时，非法 token 会被保留（证明这道闸是掩码在起作用）
    out2 = predict_x1(lg, d, jnp, rng=jax.random.key(9), jax=jax, mask=None,
                      draft=d, keep_tau=1e-12)
    o2 = onp.asarray(out2)
    kept2 = int(sum((o2[:, p] == onp.asarray(d)[:, p]).sum()
                    for p in bad_pos if p >= 1))
    check("K4★ 非法 draft token 一定被替换", kept_illegal == 0,
          f"保留了 {kept_illegal} 个")
    check("K4★ 去掉掩码后它们会被保留（证明测试有牙）", kept2 > 0,
          f"保留了 {kept2} 个")


def k5_direction(taus, fs):
    """★ 带牙 ★ 方向不能反：tau 小 -> 保留多。写反的实现这里必然失败。"""
    check("K5★ 方向正确（tau 小保留多）", fs[0] > fs[-1] + 0.5,
          f"tau=1e-12 保留 {fs[0]:.3f}, tau=1.0 保留 {fs[-1]:.3f}")


def k6_guard_runs_last():
    """★ 带牙 ★ keep 闸门不能撤销零 size 闸门的修复。

    实测证据：闸门在 keep 之后时，非法率 0.000%（关闭 keep）-> 0.289%（tau 0.01）
    -> 0.538%（tau 0.1）—— 保留 draft 的两位 size 数字，就把刚被修掉的 size=0
    放了回来。闸门执行的是对**输出**的硬约束，所以任何能改输出的东西都必须在它
    之前跑。
    """
    from dfm_sampler import digit_zero_guard
    rng = onp.random.default_rng(6)
    HI, LO, ZERO = 5, 6, 3
    mask = onp.zeros((L, V), bool)
    for p in range(L):
        legal = rng.choice(V, size=8, replace=False)
        mask[p, legal] = True
        if p % MSG in (HI, LO):
            mask[p, ZERO] = True     # 零在两个数位上都合法（真实编码就是这样）
    logits = jnp.asarray(rng.normal(0, 3.0, (B, L, V)).astype(onp.float32))
    draft = onp.zeros((B, L), onp.int32)
    for p in range(L):
        legal = onp.nonzero(mask[p])[0]
        draft[:, p] = rng.choice(legal, size=B)
    for p in range(L):                      # 在 draft 里埋 size=0
        if p % MSG in (HI, LO):
            draft[:, p] = ZERO
    draft = jnp.asarray(draft)
    g = digit_zero_guard(L, MSG, HI, LO, ZERO, jnp)

    def n_zero_size(out):
        o = onp.asarray(out)
        hi, lo = o[:, HI::MSG], o[:, LO::MSG]
        n = min(hi.shape[1], lo.shape[1])
        return int(((hi[:, :n] == ZERO) & (lo[:, :n] == ZERO)).sum())

    kw = dict(rng=jax.random.key(3), jax=jax, mask=jnp.asarray(mask),
              draft=draft, keep_tau=1e-12)          # 最保守：几乎全保留 draft
    a = n_zero_size(predict_x1(logits, draft, jnp, guard=g, **kw))
    b = n_zero_size(predict_x1(logits, draft, jnp, guard=None, **kw))
    check("K6* keep 闸门撤销不了零 size 闸门", a == 0, f"剩余 size=0 共 {a} 个")
    check("K6* 去掉闸门后它们会留下（证明测试有牙）", b > 0, f"剩余 {b} 个")


if __name__ == "__main__":
    print("keep_tau 单元测试（合成 logits, CPU）\n")
    k1_canary()
    k2_small_tau_is_draft()
    taus, fs = k3_monotone()
    k4_illegal_draft_replaced()
    k5_direction(taus, fs)
    k6_guard_runs_last()
    print(f"\n{sum(OK)}/{len(OK)} 通过")
    sys.exit(0 if all(OK) else 1)
