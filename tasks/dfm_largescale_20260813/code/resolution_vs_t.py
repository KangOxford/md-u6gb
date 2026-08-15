#!/usr/bin/env python3
"""值度量的分辨率-vs-t 表（E11 的前置证据）。纯 CPU。

一条概率路径有用的前提是它把信息**铺开在 t 上**。当前的 field 度量不是：实测
信息压在 t ∈ [0.15, 0.30] 这一小段里，两端各有一大片「已经收敛」和「还是纯噪声」
的区域，模型在那里学不到东西。dFlowGRPO 用 beta_t = 3(t/(1-t))^0.9 正是为此。

这里报每个字段在每个 t 上的**中位相对位移** median|v_t - v_1| / span，以及保留率。
判据（跑之前锁死）：
    好的路径 = 中位位移随 t **近似线性**下降，且没有任何一段 t 占掉 >50% 的下降。
    坏的路径 = 一个台阶：某两个相邻 t 之间掉了大半。
"""
import sys

import numpy as onp
import jax
import jax.numpy as jnp

sys.path.insert(0, "/lus/lfs1aip2/projects/public/u6gb/"
                   "openreview-v2-worktrees/dfm-post-training-20260801")
from lob.encode.encoding import Message_Tokenizer, Vocab           # noqa: E402
from lob.train.dfm_value import ValueMetric                        # noqa: E402
from lob.train.dfm import (corrupt_sequence, beta_schedule,        # noqa: E402
                           build_field_distance_matrix)

TS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_MSG = 8000
SHOW = ["size", "price", "delta_t_ns", "time_ns", "delta_t_s"]


def build_clean(met, n, rng):
    msg = onp.zeros((1, n, met.msg_len), dtype=onp.int32)
    truth = {}
    for f in met.numeric:
        v = rng.integers(f["lo"], f["hi"] + 1, size=n)
        truth[f["name"]] = v
        mag = onp.abs(v)
        if f["sign_off"] is not None:
            msg[0, :, f["sign_off"]] = onp.where(v >= 0, 1, 0) + f["sign_base"]
        for i, o in enumerate(f["digits"]):
            p = f["base"] ** (len(f["digits"]) - 1 - i)
            msg[0, :, o] = (mag // p) % f["base"] + f["block_base"]
    for name, off, blk, base, nv in met.categorical:
        msg[0, :, off] = rng.integers(0, nv, size=n) + base
    return jnp.asarray(msg.reshape(1, -1)), truth


def run_value(met, x, truth):
    rows = {}
    for i, t in enumerate(TS):
        xt = corrupt_sequence(jax.random.key(1000 + i), x, met,
                              jnp.array([t]), beta_max=None)
        m = xt.reshape(1, -1, met.msg_len)
        for f in met.numeric:
            v, _ = met._decode(m, f)
            v = onp.asarray(v)[0].astype(onp.float64)
            span = float(f["hi"] - f["lo"])
            d = onp.abs(v - truth[f["name"]]) / span
            rows.setdefault(f["name"], []).append(
                (float(onp.median(d)), float((d == 0).mean())))
    return rows


def run_field(D, met, x, truth, beta_max=70.0):
    """同样的量，但走当前的 (V,V) 逐位路径 —— 这是对照，不是新东西。"""
    rows = {}
    for i, t in enumerate(TS):
        xt = corrupt_sequence(jax.random.key(2000 + i), x, D,
                              jnp.array([t]), beta_max=beta_max)
        m = xt.reshape(1, -1, met.msg_len)
        for f in met.numeric:
            v, ok = met._decode(m, f)
            v = onp.asarray(v)[0].astype(onp.float64)
            span = float(f["hi"] - f["lo"])
            d = onp.abs(v - truth[f["name"]]) / span
            rows.setdefault(f["name"], []).append(
                (float(onp.median(d)), float((d == 0).mean())))
    return rows


def show(rows, tag):
    print(f"\n=== {tag}：中位相对位移 median|v_t - v_1| / span ===")
    print("t      " + "".join(f"{t:>7.2f}" for t in TS))
    for name in SHOW:
        if name not in rows:
            continue
        print(f"{name:<13}" + "".join(f"{d:>7.3f}" for d, _ in rows[name]))
    print(f"\n=== {tag}：保留率 P(v_t == v_1) ===")
    print("t      " + "".join(f"{t:>7.2f}" for t in TS))
    for name in SHOW:
        if name not in rows:
            continue
        print(f"{name:<13}" + "".join(f"{k:>7.3f}" for _, k in rows[name]))


def concentration(rows):
    """下降最集中的那一段 t 占掉了总下降的多少。"""
    out = {}
    for name, r in rows.items():
        d = onp.array([x[0] for x in r])
        drop = d[0] - d[-1]
        if drop <= 1e-9:
            out[name] = (float("nan"), None)
            continue
        step = -onp.diff(d)
        j = int(onp.argmax(step))
        out[name] = (float(step[j] / drop), (TS[j], TS[j + 1]))
    return out


def scan_schedules(rng):
    """按预注册判据挑 schedule，不靠眼看。

    判据（与主表同一条）：没有任何一段 t 占掉 >50% 的中位位移下降。
    报每个候选的最差字段集中度；最差值最小的那个胜出。
    """
    x, truth = build_clean(
        ValueMetric(Message_Tokenizer, Vocab()), N_MSG // 2, rng)
    print(f"\n=== schedule 扫描（判据：最差集中度 <= 0.50）===")
    print(f"{'schedule':<18}{'最差集中度':>12}{'最差字段':>14}  {'各字段':<40}")
    best = None
    for sched, ps in (("cosine", [1.0]), ("apow", [2.0, 3.0, 4.0, 6.0, 8.0]),
                      ("logit", [0.5, 0.9, 1.5])):
        for pp in ps:
            m = ValueMetric(Message_Tokenizer, Vocab(), a_target=12.0,
                            schedule=sched, sched_p=pp)
            r = run_value(m, x, truth)
            c = concentration(r)
            vals = {k: c[k][0] for k in SHOW if k in c}
            worst_k = max(vals, key=vals.get)
            w = vals[worst_k]
            tag = f"{sched}" + ("" if sched == "cosine" else f" p={pp:g}")
            det = " ".join(f"{k.split('_')[0][:4]}={v:.2f}" for k, v in vals.items())
            mark = " <-" if best is None or w < best[0] else ""
            print(f"{tag:<18}{w:>12.3f}{worst_k:>14}  {det:<40}{mark}")
            if best is None or w < best[0]:
                best = (w, sched, pp)
    print(f"\n胜出: schedule={best[1]} p={best[2]:g}  最差集中度 {best[0]:.3f}"
          f"  -> {'通过' if best[0] <= 0.5 else '仍未过'}")
    return best


if __name__ == "__main__":
    rng = onp.random.default_rng(20260815)
    if "--scan" in sys.argv:
        scan_schedules(rng)
        sys.exit(0)
    met = ValueMetric(Message_Tokenizer, Vocab(), a_target=12.0)
    x, truth = build_clean(met, N_MSG, rng)
    rv = run_value(met, x, truth)
    show(rv, "value 度量")
    D = build_field_distance_matrix(Vocab(), scale='log')
    rf = run_field(D, met, x, truth)
    show(rf, "field 度量（当前，对照）")

    print("\n=== 信息集中度：单个 t 区间占掉的下降比例（越小越好，理想 ~1/13=0.077）===")
    print(f"{'field':<13}{'value':>10}{'区间':>14}{'field':>10}{'区间':>14}")
    cv, cf = concentration(rv), concentration(rf)
    for name in SHOW:
        a, ai = cv.get(name, (float('nan'), None))
        b, bi = cf.get(name, (float('nan'), None))
        print(f"{name:<13}{a:>10.3f}{str(ai):>14}{b:>10.3f}{str(bi):>14}")
    worst_v = max(v[0] for k, v in cv.items() if onp.isfinite(v[0]))
    worst_f = max(v[0] for k, v in cf.items() if onp.isfinite(v[0]))
    print(f"\n判据（预注册）：没有任何一段 t 占掉 >50% 的下降")
    print(f"  value 最差 {worst_v:.3f}  ->  {'通过' if worst_v <= 0.5 else '未过'}")
    print(f"  field 最差 {worst_f:.3f}  ->  {'通过' if worst_f <= 0.5 else '未过'}")
