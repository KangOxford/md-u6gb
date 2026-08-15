#!/usr/bin/env python3
"""B1 —— 值空间腐蚀律的 numpy 参考实现（纯 CPU，无 jax，无数据）。

为什么需要它。当前度量是 `(V,V)` 矩阵，按 **token** 索引，而数值字段是**位置制多
token**（`size` = 2 位 base-100，`price` = 符号 + 2 位 base-1000）。于是：

    size 99  -> digits (0, 99)     数值相邻
    size 100 -> digits (1,  0)     低位 |99-0| = 99 -> log1p 距离 = 1.0（最大）
    size 0   -> digits (0,  0)     数值差 99
    size 99  -> digits (0, 99)     低位同样 = 1.0（最大）

**度量给「相邻」和「最远」同一个数**，而 field 度量存在的全部理由就是
「相近的 token 语义相近」。它在每个进位边界上失效。

全量实测（487 ticker × 2 月）：`size` 的水平是 **+0.958 / +0.968**，是五个字段里
最差的，且四个异常全落在它身上（唯一被推坏的序数字段、唯一随机方向也有效、
`size=0` 的来源、分位分箱退化最重）。

本文件只做一件事：把这条律写对，并证明它写对了。**不碰生产代码，不上 GPU。**

律：

    p(v | v1) ∝ (1 + |v - v1|)^(-a),   v ∈ [lo, hi] ∩ Z,   a = beta_t / log1p(span)
    Z(v1) = 1 + G(v1 - lo) + G(hi - v1)
    G(n)  = sum_{j=2..n+1} j^(-a)

`G` 不能用 Hurwitz zeta：它只在 a > 1 收敛，而 a = beta/log1p(span) 在整个 beta
扫描里大部分时间 **小于 1**（size 的临界 beta 是 9.21，price 是 14.51，ns 是 20.72）。
用「精确头部 + Euler-Maclaurin 尾部」代替，a=1 的可去奇点用 exprel 形式处理。
"""
import numpy as onp

M_EXACT = 16          # 头部精确求和的项数


def _exprel(x):
    """(e^x - 1) / x，在 x -> 0 处数值稳定。a=1 的可去奇点靠它。"""
    x = onp.asarray(x, onp.float64)
    small = onp.abs(x) < 1e-8
    return onp.where(small, 1.0 + x / 2.0 + x * x / 6.0,
                     onp.expm1(onp.where(small, 1.0, x)) / onp.where(small, 1.0, x))


def _pow_diff(m, N, c):
    """(N^c - m^c) / c，c -> 0 时退化为 ln(N/m)。

    直接写 (N**c - m**c)/c 在 a=1（c=0）处是 0/0，而 a 正好会扫过 1。
    改写成 m^c * Δ * exprel(cΔ)，Δ = ln(N/m)：c=0 时 exprel(0)=1，结果就是 Δ。
    """
    m = onp.asarray(m, onp.float64); N = onp.asarray(N, onp.float64)
    d = onp.log(N / m)
    return (m ** c) * d * _exprel(c * d)


def G(n, a):
    """sum_{j=2}^{n+1} j^(-a)，精确头部 + Euler-Maclaurin 尾部。

    EM 对 sum_{j=m}^{N} f(j)：∫ + (f(m)+f(N))/2 + (f'(N)-f'(m))/12 - (f'''(N)-f'''(m))/720
    """
    n = onp.asarray(n, onp.float64)
    out = onp.zeros_like(n)
    hi_exact = onp.minimum(n + 1.0, M_EXACT)
    # 头部：j = 2 .. hi_exact
    for j in range(2, M_EXACT + 1):
        out = out + onp.where(hi_exact >= j, float(j) ** (-a), 0.0)
    # 尾部：j = M_EXACT+1 .. n+1（仅当 n+1 > M_EXACT）
    m0, N0 = float(M_EXACT + 1), onp.maximum(n + 1.0, M_EXACT + 1.0)
    tail = (_pow_diff(m0, N0, 1.0 - a)
            + (m0 ** (-a) + N0 ** (-a)) / 2.0
            + a * (m0 ** (-a - 1.0) - N0 ** (-a - 1.0)) / 12.0
            - a * (a + 1.0) * (a + 2.0)
            * (m0 ** (-a - 3.0) - N0 ** (-a - 3.0)) / 720.0)
    return onp.where(n + 1.0 > M_EXACT, out + tail, out)


def G_brute(n, a):
    """暴力求和，只用于测试。"""
    n = int(n)
    if n < 1:
        return 0.0
    j = onp.arange(2, n + 2, dtype=onp.float64)
    return float((j ** (-a)).sum())


def sample(v1, lo, hi, a, u):
    """从 p(v|v1) ∝ (1+|v-v1|)^(-a) 抽样。`u` 是 [0,1) 上的均匀数。

    三分支 + 一次整数二分：
        u*Z < 1                -> v = v1              （留在原地）
        1 <= u*Z < 1 + G(Lft)  -> 向左 k 步
        否则                    -> 向右 k 步
    k = min{k in [1,n] : G(k) >= m}，二分 ceil(log2(span+2))+1 步。
    """
    v1 = onp.asarray(v1, onp.float64)
    L = v1 - lo
    R = hi - v1
    GL, GR = G(L, a), G(R, a)
    Z = 1.0 + GL + GR
    m = onp.asarray(u, onp.float64) * Z
    stay = m < 1.0
    left = (~stay) & (m < 1.0 + GL)
    mm = onp.where(left, m - 1.0, m - 1.0 - GL)
    nn = onp.where(left, L, R)
    # 整数二分：找最小的 k 使 G(k) >= mm
    klo = onp.ones_like(nn)
    khi = onp.maximum(nn, 1.0)
    steps = int(onp.ceil(onp.log2(max(float(onp.max(khi)), 2.0)))) + 2
    for _ in range(steps):
        mid = onp.floor((klo + khi) / 2.0)
        go_hi = G(mid, a) < mm
        klo = onp.where(go_hi, mid + 1.0, klo)
        khi = onp.where(go_hi, khi, mid)
    k = onp.minimum(klo, onp.maximum(nn, 1.0))
    v = onp.where(stay, v1, onp.where(left, v1 - k, v1 + k))
    return onp.clip(v, lo, hi).astype(onp.int64)


def pmf_exact(v1, lo, hi, a):
    """枚举出的精确 pmf，只用于测试（值域小的时候才可用）。"""
    v = onp.arange(lo, hi + 1, dtype=onp.float64)
    w = (1.0 + onp.abs(v - v1)) ** (-a)
    return v.astype(onp.int64), w / w.sum()


def beta_to_a(beta, span):
    """把路径温度换算成指数。span 是该字段的值域宽度。"""
    return beta / onp.log1p(span)
