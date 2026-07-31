#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IsoFLOP 顶点估计的稳健化对照：全点 / 窗口 / 加权三种拟合。

动机（2026-07-31 实测）：全点二次拟合在低算力切片被严重欠训的大模型右臂主导，
曲率被拉到邻近切片的 10 倍，顶点外推到所有实测点之下（C=2.21e18 切片 L*=0.5384，
而该切片最低实测 0.64），并使 L*(C) 违反单调下降。bracketed 判据（左右各有点）
不足以捕获这种失败。

三种估计：
  full   全部点二次拟合（现状口径）
  window 只用 L <= L_min + delta 的谷底邻域点（delta 默认 0.15 nats）
  weight 全部点但权重 w = exp(-(L - L_min)/tau)（tau 默认 0.10 nats）

验收判据（任一不过即标记 vertex_undershoot / curvature_jump）：
  1. 顶点不得显著穿底：L_min_observed - L* <= tol。顶点略低于最低实测点是正常的
     （真实最优落在采样 N 格点之间），故 tol 按测量噪声定标：单点 CE 的 95% CI
     半宽约 0.002 nats，取 tol=0.02（10x 噪声）。C=2.21e18 全点拟合穿底 0.106
     nats（50x 噪声）才是病理。
  2. 曲率 a 与相邻切片同量级（相邻比值 < 5x）

用法:
  python valset_isoflop_robust.py --fit-ready <csv> --out-json <path> [--out-png <path>]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def interp_L_at_C(g, C_target):
    """链内 log-C 线性插值（与 valset_isoflop_interp.py 同口径）。"""
    g = g.sort_values("C")
    logC = np.log(g.C.to_numpy())
    L = g.test_ce.to_numpy()
    logCt = np.log(C_target)
    if logCt < logC.min() or logCt > logC.max():
        return None
    return float(np.interp(logCt, logC, L))


def fit_parabola(logN, L, w=None):
    """L = a*logN^2 + b*logN + c 的（加权）最小二乘；返回顶点。a<=0 视为无效。"""
    if len(logN) < 3 or len(np.unique(logN)) < 3:
        return None
    a, b, c = np.polyfit(logN, L, 2, w=w)
    if a <= 0:
        return None
    logN_star = -b / (2 * a)
    return dict(a=float(a), N_star=float(np.exp(logN_star)),
                L_star=float(c - b**2 / (4 * a)), n_points=int(len(logN)))


def estimate(pts_df, mode, delta, tau):
    """按 mode 从 (N, L) 点集估计顶点。"""
    logN = np.log(pts_df.N.to_numpy())
    L = pts_df.L.to_numpy()
    L_min = float(L.min())
    if mode == "full":
        return fit_parabola(logN, L)
    if mode == "window":
        m = L <= L_min + delta
        return fit_parabola(logN[m], L[m])
    if mode == "weight":
        return fit_parabola(logN, L, w=np.exp(-(L - L_min) / tau))
    raise ValueError(mode)


def slope_of(rows):
    """log N* ~ slope * log C 的最小二乘斜率。"""
    rows = [r for r in rows if r and r.get("N_star")]
    if len(rows) < 2:
        return None
    C = np.log([r["C"] for r in rows])
    N = np.log([r["N_star"] for r in rows])
    return float(np.polyfit(C, N, 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-ready", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-png", type=Path)
    ap.add_argument("--n-slices", type=int, default=6)
    ap.add_argument("--pct-lo", type=float, default=25.0)
    ap.add_argument("--pct-hi", type=float, default=75.0)
    ap.add_argument("--delta", type=float, default=0.15,
                    help="window 模式的谷底邻域宽度（nats）")
    ap.add_argument("--tau", type=float, default=0.10,
                    help="weight 模式的权重衰减尺度（nats）")
    ap.add_argument("--tol", type=float, default=0.02,
                    help="允许的顶点穿底幅度（nats）；默认 10x 单点 CE 噪声")
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
    chains = pd.DataFrame(chains)
    C_lo = np.percentile(chains.C_min, args.pct_lo)
    C_hi = np.percentile(chains.C_max, args.pct_hi)
    C_targets = np.geomspace(C_lo, C_hi, args.n_slices)

    modes = ["full", "window", "weight"]
    slices = []
    for C_t in C_targets:
        pts = []
        for _, ch in chains.iterrows():
            if not (ch.C_min <= C_t <= ch.C_max):
                continue
            L = interp_L_at_C(ch.g, C_t)
            if L is not None:
                pts.append((ch.N, L, ch.label))
        if len(pts) < 3:
            continue
        pts_df = pd.DataFrame(pts, columns=["N", "L", "label"])
        L_min_obs = float(pts_df.L.min())
        rec = {"C": float(C_t), "n_points": len(pts_df),
               "n_unique_N": int(pts_df.N.nunique()), "L_min_observed": L_min_obs}
        for mode in modes:
            f = estimate(pts_df, mode, args.delta, args.tau)
            if f:
                f["undershoot_nats"] = float(L_min_obs - f["L_star"])
                f["vertex_undershoot"] = bool(f["undershoot_nats"] > args.tol)
                # 左右臂计数（相对该模式的顶点）
                f["n_left"] = int((pts_df.N < f["N_star"]).sum())
                f["n_right"] = int((pts_df.N > f["N_star"]).sum())
            rec[mode] = f
        slices.append(rec)

    # 曲率跳变检查（相邻切片同一 mode 的 a 比值）
    for mode in modes:
        a = [(s["C"], s[mode]["a"]) for s in slices if s.get(mode)]
        for i, s in enumerate(slices):
            if not s.get(mode):
                continue
            neigh = [x for c, x in a if c != s["C"]]
            s[mode]["curvature_jump"] = bool(
                neigh and s[mode]["a"] / np.median(neigh) > 5.0)

    out = {"input": str(args.fit_ready.resolve()), "delta": args.delta, "tau": args.tau,
           "n_chains": int(len(chains)), "slices": slices, "slopes": {}}
    for mode in modes:
        rows = [dict(C=s["C"], **s[mode]) for s in slices if s.get(mode)]
        clean = [r for r in rows if not r["vertex_undershoot"] and not r.get("curvature_jump")]
        out["slopes"][mode] = {
            "all_valid": slope_of(rows), "n_all": len(rows),
            "clean_only": slope_of(clean), "n_clean": len(clean),
            "L_star_monotone": bool(all(
                x["L_star"] >= y["L_star"] for x, y in zip(clean, clean[1:]))) if len(clean) > 1 else None,
        }
    args.out_json.write_text(json.dumps(out, indent=1))

    print(f"chains={len(chains)}  slices={len(slices)}")
    print(f"\n{'C':>10} {'Lmin_obs':>9} | " + " | ".join(
        f"{m:>6}: {'N*':>7} {'L*':>7} {'under':>7} {'flag':>5}" for m in modes))
    for s in slices:
        row = f"{s['C']:10.2e} {s['L_min_observed']:9.4f} | "
        cells = []
        for m in modes:
            f = s.get(m)
            if not f:
                cells.append(f"{m:>6}: {'--':>7} {'--':>7} {'--':>7} {'no-min':>5}")
            else:
                flag = ("UNDER" if f["vertex_undershoot"] else
                        ("CURV" if f.get("curvature_jump") else "ok"))
                cells.append(f"{m:>6}: {f['N_star']/1e6:6.1f}M {f['L_star']:7.4f} "
                             f"{f['undershoot_nats']:+7.4f} {flag:>5}")
        print(row + " | ".join(cells))
    print("\nslope(logN* ~ logC):")
    for m in modes:
        s = out["slopes"][m]
        cl = f"{s['clean_only']:.4f}" if s["clean_only"] is not None else "n/a"
        al = f"{s['all_valid']:.4f}" if s["all_valid"] is not None else "n/a"
        print(f"  {m:>6}: all={al} (n={s['n_all']})   clean={cl} (n={s['n_clean']})"
              f"   L*_monotone={s['L_star_monotone']}")
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
