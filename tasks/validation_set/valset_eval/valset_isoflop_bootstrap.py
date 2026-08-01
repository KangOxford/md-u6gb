#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""标度指数的不确定度：对训练链做 bootstrap 重采样，重跑整条估计流程。

目前报告只有点估计（0.46）和方法间跨度（0.43-0.49），缺统计不确定度。链是自然
的重采样单位：每条链是一次独立的训练实验，切片内的 (N, L) 点由链贡献，因此对链
有放回重采样、逐次重跑"插值→窗口拟合顶点→回归斜率"，即可得到指数的抽样分布。

注意重采样保持 C 切片位置固定（用全样本确定的 C_targets），否则每次重采样的切片
位置漂移会把"切片选择"的方差混进来，那不是我们要度量的量。

用法:
  python valset_isoflop_bootstrap.py --fit-ready <csv> --out-json <path> [--n-boot 2000]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def interp_L_at_C(g, C_target):
    g = g.sort_values("C")
    logC = np.log(g.C.to_numpy())
    L = g.test_ce.to_numpy()
    logCt = np.log(C_target)
    if logCt < logC.min() or logCt > logC.max():
        return None
    return float(np.interp(logCt, logC, L))


def vertex_window(N, L, delta):
    """窗口法顶点：只用谷底邻域 L <= L_min + delta 的点做二次拟合。"""
    L_min = L.min()
    m = L <= L_min + delta
    logN, Lw = np.log(N[m]), L[m]
    if len(logN) < 3 or len(np.unique(logN)) < 3:
        return None
    a, b, c = np.polyfit(logN, Lw, 2)
    if a <= 0:
        return None
    logN_star = -b / (2 * a)
    L_star = c - b**2 / (4 * a)
    if (L_min - L_star) > 0.02:      # 顶点显著穿底 → 该切片不可用
        return None
    return float(np.exp(logN_star))


def slope_from_chains(chain_list, C_targets, delta):
    """给定一组链（可含重复），返回 log N* ~ log C 的斜率；不足两个切片返回 None。"""
    pts_C, pts_N = [], []
    for C_t in C_targets:
        Ns, Ls = [], []
        for ch in chain_list:
            if not (ch["C_min"] <= C_t <= ch["C_max"]):
                continue
            L = interp_L_at_C(ch["g"], C_t)
            if L is not None:
                Ns.append(ch["N"])
                Ls.append(L)
        if len(Ns) < 3 or len(set(Ns)) < 3:
            continue
        Nstar = vertex_window(np.asarray(Ns), np.asarray(Ls), delta)
        if Nstar:
            pts_C.append(C_t)
            pts_N.append(Nstar)
    if len(pts_C) < 2:
        return None
    return float(np.polyfit(np.log(pts_C), np.log(pts_N), 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-ready", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-slices", type=int, default=6)
    ap.add_argument("--pct-lo", type=float, default=25.0)
    ap.add_argument("--pct-hi", type=float, default=75.0)
    ap.add_argument("--delta", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20260801)
    args = ap.parse_args()

    df = pd.read_csv(args.fit_ready, low_memory=False)
    df = df[df.model_type.str.lower() == "mamba3"]
    df["C"] = df.total_flops_to_step.astype(float)

    chains = []
    for (sz, sd), g in df.groupby(["size_label", "seed"]):
        g = g.sort_values("C")
        if len(g) < 2:
            continue
        chains.append({"label": f"{sz}-s{sd}", "N": float(g.num_params.iloc[0]),
                       "C_min": float(g.C.min()), "C_max": float(g.C.max()), "g": g})

    # C 切片位置由全样本一次确定，bootstrap 中固定不变
    cmins = np.array([c["C_min"] for c in chains])
    cmaxs = np.array([c["C_max"] for c in chains])
    C_targets = np.geomspace(np.percentile(cmins, args.pct_lo),
                             np.percentile(cmaxs, args.pct_hi), args.n_slices)

    point = slope_from_chains(chains, C_targets, args.delta)

    rng = np.random.default_rng(args.seed)
    boots, n_fail = [], 0
    for _ in range(args.n_boot):
        idx = rng.integers(0, len(chains), len(chains))
        s = slope_from_chains([chains[i] for i in idx], C_targets, args.delta)
        if s is None:
            n_fail += 1
        else:
            boots.append(s)
    boots = np.asarray(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])

    out = {"input": str(args.fit_ready.resolve()), "method": "window", "delta": args.delta,
           "n_chains": len(chains), "n_boot": args.n_boot,
           "n_boot_usable": int(len(boots)), "n_boot_failed": n_fail,
           "C_targets": [float(c) for c in C_targets],
           "slope_point": point, "slope_mean": float(boots.mean()),
           "slope_sd": float(boots.std(ddof=1)),
           "slope_ci95": [float(lo), float(hi)]}
    args.out_json.write_text(json.dumps(out, indent=1))

    print(f"chains={len(chains)}  n_boot={args.n_boot} (usable {len(boots)}, failed {n_fail})")
    print(f"slope point estimate : {point:.4f}")
    print(f"bootstrap mean +- sd : {boots.mean():.4f} +- {boots.std(ddof=1):.4f}")
    print(f"bootstrap 95% CI     : [{lo:.4f}, {hi:.4f}]")
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
