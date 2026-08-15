#!/usr/bin/env python3
"""B1 的参考测试。纯 numpy，纯 CPU，几秒跑完，**不需要任何数据或 GPU**。

五个测试，最后两个是**带牙的** —— 它们必须在旧的逐位度量上失败，否则我们不知道
这道闸有没有牙（PR1）。带牙的断言在同一个测试里同时跑新旧两套，把缺陷记录下来
而不只是绕开它。
"""
import sys
import numpy as onp

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from value_metric_ref import G, G_brute, sample, pmf_exact, beta_to_a

A_GRID = [0.0, 0.05, 0.2, 0.5, 0.9, 0.999, 1.0, 1.001, 1.1, 2.0, 3.0, 5.0, 10.0, 15.0]
OK = []


def check(name, cond, detail=""):
    OK.append(bool(cond))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")


def t1_euler_maclaurin():
    """G 对暴力求和的精度。a 扫过 1（可去奇点）。"""
    worst = 0.0
    for a in A_GRID:
        for n in [1, 2, 5, 15, 16, 17, 20, 100, 1000, 9999, 100000]:
            e = abs(float(G(n, a)) - G_brute(n, a))
            worst = max(worst, e)
    check("T1 G 对 brute-force", worst < 1e-8, f"最大绝对误差 {worst:.2e}")


def t2_float32_monotone():
    """float32 下 G 必须单调 —— 这是二分法可用的前提。"""
    bad = 0
    for a in A_GRID:
        n = onp.unique(onp.round(onp.logspace(0, 6, 4000)).astype(onp.int64))
        g = G(n.astype(onp.float64), a).astype(onp.float32)
        bad += int((onp.diff(g) < 0).sum())
    check("T2 float32 下 G 单调", bad == 0, f"非单调步数 {bad}")


def t3_sampler_matches_pmf():
    """采样器对枚举 pmf 的卡方检验。span 取小值以便全支撑枚举。"""
    rng = onp.random.default_rng(20260815)
    worst_p, worst = 1.0, None
    for span in (99, 999):
        lo, hi = 0, span
        for a in (0.0, 0.5, 1.0, 2.0, 5.0, 12.0):
            for v1 in (lo, span // 2, hi):
                n = 200000
                u = rng.random(n)
                s = sample(onp.full(n, v1), lo, hi, a, u)
                vals, p = pmf_exact(v1, lo, hi, a)
                obs = onp.bincount(s - lo, minlength=span + 1)
                exp = p * n
                m = exp > 5                      # 卡方要求期望频数不能太小
                chi = float(((obs[m] - exp[m]) ** 2 / exp[m]).sum())
                dof = int(m.sum()) - 1
                # 正态近似的尾概率
                z = (chi - dof) / onp.sqrt(2 * dof)
                if z > 0 and (0.5 * onp.exp(-z * z / 2)) < worst_p:
                    worst_p = float(0.5 * onp.exp(-z * z / 2))
                    worst = (span, a, v1, chi, dof, z)
    check("T3 采样器 == 枚举 pmf（卡方）", worst_p > 1e-4,
          f"最差 p≈{worst_p:.1e} @ {worst}")


def t4_uniform_endpoint():
    """a=0 时必须在整个值域上均匀 —— build_field_distance_matrix 的 docstring
    用一整段论证过这个端点，而候选网格方案在这里 TV 高达 0.974。"""
    rng = onp.random.default_rng(7)
    lo, hi, n = 1, 9999, 400000
    s = sample(onp.full(n, 5000), lo, hi, 0.0, rng.random(n))
    cnt = onp.histogram(s, bins=32, range=(lo, hi + 1))[0].astype(float)
    tv = 0.5 * onp.abs(cnt / cnt.sum() - 1.0 / 32).sum()
    check("T4 a=0 在值域上均匀", tv < 0.01,
          f"TV={tv:.4f}, min={s.min()}, max={s.max()}")


def t5_neighbours_are_neighbours():
    """★ 带牙 ★ size=100 的邻居 99 与 101 概率应当相当。

    新度量：比值 ∈ [0.5, 2]。
    旧度量（逐位）：size 100 = digits (1,0)，99 = (0,99)，101 = (1,1)。
      低位从 0 到 99 的距离是 log1p(99)/log1p(99) = 1.0（最大），
      而 0 到 1 只有 log1p(1)/log1p(99) = 0.151。所以旧度量下
      P(99)/P(101) = exp(-a*(1.0-0.151))，随 a 指数塌陷。
    """
    a = beta_to_a(70.0, 9999)                       # field 度量在 t=1 的指数
    v, p = pmf_exact(100, 1, 9999, a)
    new = float(p[v == 99][0] / p[v == 101][0])
    # 旧度量：只有低位不同，距离用同一条 log1p 归一
    d99 = onp.log1p(99) / onp.log1p(99)             # 低位 0 -> 99
    d101 = onp.log1p(1) / onp.log1p(99)             # 低位 0 -> 1
    old = float(onp.exp(-70.0 * (d99 - d101)))
    check("T5★ 新度量下 99/101 概率相当", 0.5 <= new <= 2.0, f"比值 {new:.3f}")
    check("T5★ 旧度量下同一比值塌陷（证明测试有牙）", old < 0.05,
          f"比值 {old:.3e}")


def t6_size_zero_impossible():
    """★ 带牙 ★ size 的值域是 [1, 9999]，所以 size=0 按构造不可能。

    实测：旧路径下修正臂 6.24% 的消息 size=0，而 draft 只有 0.028%；
    非法 size 的取值 100% 恰好是 0（n=29584）。
    """
    rng = onp.random.default_rng(11)
    lo, hi = 1, 9999
    worst = 0
    for a in (0.0, 0.5, 2.0, 12.0):
        for v1 in (1, 2, 100, 9999):
            s = sample(onp.full(50000, v1), lo, hi, a, rng.random(50000))
            worst = max(worst, int((s < 1).sum()))
    check("T6★ size=0 恒不出现", worst == 0, f"出现次数 {worst}（旧路径实测 6.24%）")


if __name__ == "__main__":
    print("值空间度量参考实现测试（纯 numpy / CPU）\n")
    for f in (t1_euler_maclaurin, t2_float32_monotone, t3_sampler_matches_pmf,
              t4_uniform_endpoint, t5_neighbours_are_neighbours,
              t6_size_zero_impossible):
        f()
    print(f"\n{sum(OK)}/{len(OK)} 通过")
    sys.exit(0 if all(OK) else 1)
