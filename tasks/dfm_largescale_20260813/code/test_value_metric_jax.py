#!/usr/bin/env python3
"""B3 —— `ValueMetric` 的 JAX 端测试。纯 CPU，登录节点可跑。

七个测试，其中四个**带牙**：它们必须在当前的逐位 `(V,V)` 度量上失败，否则我们
不知道这道闸有没有牙（PR1）。带牙的项在同一个测试里同时跑新旧两条路径，把缺陷
记录下来，而不是绕开它。

    T1  往返：每个字段 decode(encode(v)) == v，全值域
    T2  t=0 端点：每个字段在值域上均匀（不是在 token 上均匀）
    T3★ size=0 恒不出现            旧路径实测 6.24%
    T4★ size 的 99/101 概率相当     旧路径 1.5e-26
    T5★ time_ns 低位在 t=0 均匀     去掉 dither 会落在 gcd(64,1000) 的格子上
    T6  specials 原样穿过（START/MASK/HIDDEN/NA 既不被改也不被生成）
    T7★ 类别字段的腐蚀没有消失      直接用 beta_t 会让 t=0.1 的保留率就到 0.993
    T8  beta_max 错配必须 raise（B6 fail-closed）

跑法：
    cd <OR2 worktree> && JAX_PLATFORMS=cpu taskset -c 0-7 python3 <this>
`taskset` 是必须的：登录节点 pids.max=500，而 JAX 的 CPU client 会按核数
(288) 开线程池，直接 pthread_create 失败。
"""
import sys

import numpy as onp
import jax
import jax.numpy as jnp

from lob.encode.encoding import Message_Tokenizer, Vocab   # 作业实际走的模块
from lob.train.dfm_value import ValueMetric, sample_value
from lob.train.dfm import corrupt_sequence, beta_schedule

OK = []
MT = Message_Tokenizer
V = Vocab()
MET = ValueMetric(MT, V, a_target=12.0)
MSG = MT.MSG_LEN
BY_NAME = {f["name"]: f for f in MET.numeric}


def check(name, cond, detail=""):
    OK.append(bool(cond))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")


def make_clean(n_msg, rng):
    """合法的干净消息：每个字段在自己的值域里随机取值，再编码成 token。"""
    msg = onp.zeros((1, n_msg, MSG), dtype=onp.int32)
    truth = {}
    for f in MET.numeric:
        v = rng.integers(f["lo"], f["hi"] + 1, size=n_msg)
        truth[f["name"]] = v
        mag = onp.abs(v)
        if f["sign_off"] is not None:
            msg[0, :, f["sign_off"]] = onp.where(v >= 0, 1, 0) + f["sign_base"]
        for i, o in enumerate(f["digits"]):
            p = f["base"] ** (len(f["digits"]) - 1 - i)
            msg[0, :, o] = (mag // p) % f["base"] + f["block_base"]
    for name, off, blk, base, n in MET.categorical:
        msg[0, :, off] = rng.integers(0, n, size=n_msg) + base
    return jnp.asarray(msg.reshape(1, -1)), truth


def decode_all(x):
    """(1, L) token -> {field: (n_msg,) value}，用度量自己的 _decode。"""
    m = x.reshape(1, -1, MSG)
    out = {}
    for f in MET.numeric:
        v, ok = MET._decode(m, f)
        out[f["name"]] = (onp.asarray(v)[0], onp.asarray(ok)[0])
    for name, off, blk, base, n in MET.categorical:
        out[name] = (onp.asarray(m[..., off])[0] - base,
                     onp.ones(m.shape[1], bool))
    return out


def t1_roundtrip():
    rng = onp.random.default_rng(1)
    x, truth = make_clean(4000, rng)
    got = decode_all(x)
    worst, who = 0, None
    for f in MET.numeric:
        d = int(onp.abs(got[f["name"]][0] - truth[f["name"]]).max())
        if d > worst:
            worst, who = d, f["name"]
        assert got[f["name"]][1].all(), f["name"]
    check("T1 每个字段 decode(encode(v)) == v", worst == 0,
          f"最大偏差 {worst} @ {who}")


def t2_uniform_endpoint():
    """t=0 -> beta=0 -> a=0 -> 在**值域**上均匀。这是路径依赖的端点。"""
    rng = onp.random.default_rng(2)
    x, _ = make_clean(20000, rng)
    xt = corrupt_sequence(jax.random.key(0), x, MET,
                          jnp.array([0.0]), beta_max=None)
    got = decode_all(xt)
    worst, who = 0.0, None
    for f in MET.numeric:
        v = got[f["name"]][0].astype(float)
        u = (v - f["lo"]) / (f["hi"] - f["lo"])           # -> [0,1]
        cnt = onp.histogram(u, bins=20, range=(0, 1))[0].astype(float)
        tv = 0.5 * onp.abs(cnt / cnt.sum() - 1 / 20).sum()
        if tv > worst:
            worst, who = tv, f["name"]
    check("T2 t=0 每个字段在值域上均匀", worst < 0.05,
          f"最差 TV={worst:.4f} @ {who}")


def t3_size_zero():
    """★ 带牙 ★ size 的值域是 [1, 9999]，0 按构造不可达。"""
    rng = onp.random.default_rng(3)
    x, _ = make_clean(20000, rng)
    bad = 0
    for i, t in enumerate([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]):
        xt = corrupt_sequence(jax.random.key(100 + i), x, MET,
                              jnp.array([t]), beta_max=None)
        bad += int((decode_all(xt)["size"][0] < 1).sum())
    check("T3★ size=0 恒不出现", bad == 0, f"出现 {bad} 次（旧路径实测 6.24%）")


def t4_neighbours():
    """★ 带牙 ★ size=100 时 99 与 101 概率应当相当。旧度量下 1.5e-26。"""
    a = MET.beta_max / MET.log_spans["size"]              # t=1 的指数
    a_mid = beta_schedule(jnp.array(0.35), MET.beta_max) / MET.log_spans["size"]
    v = onp.arange(1, 10000)
    w = (1.0 + onp.abs(v - 100)) ** (-float(a_mid))
    p = w / w.sum()
    new = float(p[v == 99][0] / p[v == 101][0])
    # 旧度量：只有低位不同，0->99 距离 1.0，0->1 距离 log1p(1)/log1p(99)
    beta = float(beta_schedule(jnp.array(0.35), 70.0))
    old = float(onp.exp(-beta * (1.0 - onp.log1p(1) / onp.log1p(99))))
    check("T4★ 新度量下 99/101 概率相当", 0.5 <= new <= 2.0, f"比值 {new:.4f}")
    check("T4★ 旧度量下同一比值塌陷（证明测试有牙）", old < 1e-3,
          f"比值 {old:.3e}  (a_t={float(a_mid):.2f}, a_1={float(a):.2f})")


def t5_ns_low_digit():
    """★ 带牙 ★ time_ns 低位在 t=0 必须遍历 0..999。

    float32 在 1e9 附近的间距是 64，gcd(64,1000)=8 -> 去掉 dither 后低位只能
    落在有限个残数上。这里同时跑「有 dither」和「人为量化到 64」两条，后者
    必须失败，否则这条测试没有牙。
    """
    rng = onp.random.default_rng(5)
    x, _ = make_clean(30000, rng)
    xt = corrupt_sequence(jax.random.key(7), x, MET,
                          jnp.array([0.0]), beta_max=None)
    v = decode_all(xt)["time_ns"][0]
    lo = v % 1000
    cnt = onp.bincount(lo, minlength=1000).astype(float)
    tv = 0.5 * onp.abs(cnt / cnt.sum() - 1 / 1000).sum()
    hit = int((cnt > 0).sum())
    # 人为量化：模拟没有 dither 的情形
    vq = (v // 64) * 64
    hitq = int((onp.bincount(vq % 1000, minlength=1000) > 0).sum())
    check("T5★ time_ns 低位遍历 0..999 且近似均匀", hit >= 995 and tv < 0.12,
          f"命中 {hit}/1000, TV={tv:.4f}")
    check("T5★ 量化到 64 后低位塌陷（证明测试有牙）", hitq < 200,
          f"命中 {hitq}/1000")


def t6_specials():
    """specials 既不被改，也不被生成。"""
    rng = onp.random.default_rng(6)
    x, _ = make_clean(5000, rng)
    xm = onp.asarray(x).copy().reshape(1, -1, MSG)
    spec = [V.MASK_TOK, V.HIDDEN_TOK, V.NA_TOK, V.START_TOK]
    # 在若干位置埋 special
    for j, s in enumerate(spec):
        xm[0, j::17, (j * 5) % MSG] = s
    xin = jnp.asarray(xm.reshape(1, -1))
    xt = corrupt_sequence(jax.random.key(9), xin, MET,
                          jnp.array([0.4]), beta_max=None)
    a = onp.asarray(xin)[0]
    b = onp.asarray(xt)[0]
    was = onp.isin(a, spec)
    kept = int((b[was] == a[was]).sum())
    made = int(onp.isin(b[~was], spec).sum())
    check("T6 special 位置原样保留", kept == int(was.sum()),
          f"{kept}/{int(was.sum())}")
    check("T6 special 从不被生成", made == 0, f"新生成 {made} 个")


def t7_categorical_alive():
    """★ 带牙 ★ 类别字段在中段必须真的被腐蚀。

    直接把 beta_t 当成归一化距离的系数（旧写法），在 beta_max=248.68 下
    t=0.1 的保留率就到 0.993 —— 腐蚀实质上消失。
    """
    rng = onp.random.default_rng(7)
    x, truth0 = make_clean(20000, rng)
    keep = {}
    for i, t in enumerate([0.1, 0.3, 0.5]):
        xt = corrupt_sequence(jax.random.key(200 + i), x, MET,
                              jnp.array([t]), beta_max=None)
        a = onp.asarray(x).reshape(-1, MSG)[:, 0]
        b = onp.asarray(xt).reshape(-1, MSG)[:, 0]
        keep[t] = float((a == b).mean())
    # 旧写法下的保留率：softmax(-beta * 1) over 4 值
    beta01 = float(beta_schedule(jnp.array(0.1), MET.beta_max))
    old01 = 1.0 / (1.0 + 3 * onp.exp(-beta01))
    check("T7★ event_type 在 t=0.1 仍被大量腐蚀", keep[0.1] < 0.45,
          f"保留率 {keep[0.1]:.3f} (t=0.3: {keep[0.3]:.3f}, t=0.5: {keep[0.5]:.3f})")
    check("T7★ 旧写法下腐蚀已消失（证明测试有牙）", old01 > 0.98,
          f"保留率 {old01:.4f} @ beta={beta01:.1f}")


def t8_beta_fail_closed():
    """B6：传错 beta_max 必须 raise，而不是静默换一条路径。"""
    rng = onp.random.default_rng(8)
    x, _ = make_clean(100, rng)
    raised = False
    try:
        corrupt_sequence(jax.random.key(1), x, MET, jnp.array([0.5]))  # 默认 10.0
    except ValueError as e:
        raised = "beta_max" in str(e)
    check("T8 beta_max 错配 raise（fail-closed）", raised)


if __name__ == "__main__":
    print(f"ValueMetric: a_target={MET.a_target} beta_max={MET.beta_max:.4f} "
          f"numeric={len(MET.numeric)} categorical={len(MET.categorical)}\n")
    for f in (t1_roundtrip, t2_uniform_endpoint, t3_size_zero, t4_neighbours,
              t5_ns_low_digit, t6_specials, t7_categorical_alive,
              t8_beta_fail_closed):
        f()
    print(f"\n{sum(OK)}/{len(OK)} 通过")
    sys.exit(0 if all(OK) else 1)
