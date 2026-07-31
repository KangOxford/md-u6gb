#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""valset 轴 Approach 2（IsoFLOP）：isoflop_test_ce.py 的参数化版本。

方法与 `scaling_law_plots/isoflop_test_ce.py`（Jan-2026 test CE 版，Kang 偏好的
within-run interpolation 变体）逐行一致：
  1. chain = (size_label, seed)，链内对 (C, L) 做 log-C 线性插值（C = dmon 实测
     total_flops_to_step）；单点 chain 无插值区间，按原代码 `len(g) < 2` 跳过。
  2. C targets = geomspace(P25(C_min), P75(C_max), n_slices)——落在多数链的重叠区。
  3. 每个 C_k 从覆盖它的 chain 收 (N, L_interp)；≥3 个不同 N 才拟抛物线
     L = a·(logN − logN*)² + L*（L 线性、logN 空间，同原实现）。
  4. ≥3 个有效切片才对 log N* ~ log C 回归 slope（不足则如实打印，不报数）。
  5. 图 1：每 C_k 一个 subplot（散点 + chain 标注 + 黑虚抛物线 + 红点线 N*）；
     图 2：N*_C vs C（+幂律线）与 L*_C vs C 双联。

与原文件的全部差异：输入/输出/标题参数化；删去 Jan 分析特有的 293M/D=6.5T
外推标注；n_valid_para=0 时的 subplot 网格保护。

用法: python valset_isoflop_interp.py --fit-ready <csv> --out-prefix <path> --title "valset CE"
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter


def fmt_num(x):
    if x >= 1e9:  return f"{x/1e9:.1f}B"
    if x >= 1e6:  return f"{x/1e6:.0f}M"
    if x >= 1e3:  return f"{x/1e3:.0f}k"
    return f"{x:.0f}"


def interp_L_at_C(g, C_target):
    """Linear interp of L in log(C) for a single chain g."""
    g = g.sort_values("C")
    logC = np.log(g.C.to_numpy())
    L = g.test_ce.to_numpy()
    logCt = np.log(C_target)
    if logCt < logC.min() or logCt > logC.max():
        return None
    return float(np.interp(logCt, logC, L))


def fit_parabola(logN, L):
    """Fit L = a*(logN - logN_star)^2 + L_star.  Solve via least-squares on
    L = a*logN^2 + b*logN + c, then N_star = exp(-b/(2a)), L_star = c - b^2/(4a).
    """
    a, b, c = np.polyfit(logN, L, 2)
    if a <= 0:
        return None  # invalid (parabola opens downward — no minimum)
    logN_star = -b / (2 * a)
    L_star = c - b**2 / (4 * a)
    return dict(a=a, b=b, c=c, logN_star=logN_star, N_star=float(np.exp(logN_star)),
                L_star=float(L_star))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-ready", type=Path, required=True)
    ap.add_argument("--out-prefix", type=Path, required=True,
                    help="输出前缀：<prefix>_parabolas.{png,pdf}、_summary.{png,pdf}、_results.json")
    ap.add_argument("--title", default="valset CE")
    ap.add_argument("--c-label", default="measured dmon FLOPs",
                    help="C 轴口径标注；喂解析式 6ND 的输入必须改标，否则图注与数据不符")
    ap.add_argument("--n-slices", type=int, default=6)
    ap.add_argument("--pct-lo", type=float, default=25.0,
                    help="C target 下界 = P<pct-lo>(chain C_min)；原代码启发式=25")
    ap.add_argument("--pct-hi", type=float, default=75.0,
                    help="C target 上界 = P<pct-hi>(chain C_max)；原代码启发式=75")
    args = ap.parse_args()

    df = pd.read_csv(args.fit_ready, low_memory=False)
    df = df[df.model_type.str.lower() == "mamba3"]
    df["C"] = df.total_flops_to_step.astype(float)

    # Each chain's (C_min, C_max) and (N, interp-function)
    chains = []
    n_single = 0
    for (sz, sd), g in df.groupby(["size_label", "seed"]):
        g = g.sort_values("C")
        if len(g) < 2:
            n_single += 1
            continue
        chains.append({"label": f"{sz}-s{sd}", "N": float(g.num_params.iloc[0]),
                       "C_min": float(g.C.min()), "C_max": float(g.C.max()),
                       "g": g})
    chains = pd.DataFrame(chains)
    if chains.empty:
        raise SystemExit(f"0 usable chains (>=2 points each); {n_single} single-point "
                         "chains dropped — within-run interpolation undefined on this input")
    print(f"chains: {len(chains)} (dropped {n_single} single-point), "
          f"N range [{chains.N.min()/1e6:.2f}M, {chains.N.max()/1e6:.0f}M]")
    print(f"C range global: [{chains.C_min.min():.2e}, {chains.C_max.max():.2e}]")

    # Pick C targets log-spaced across overlap-friendly region
    C_lo = np.percentile(chains.C_min, args.pct_lo)
    C_hi = np.percentile(chains.C_max, args.pct_hi)
    n_slices = args.n_slices
    C_targets = np.geomspace(C_lo, C_hi, n_slices)
    print(f"\nC target slices ({n_slices}): {[f'{c:.2e}' for c in C_targets]}")

    # Per-target: gather (N, L) from chains where C_target is in [C_min, C_max]
    slice_rows = []
    parabolas = []
    for C_t in C_targets:
        pts = []
        for _, ch in chains.iterrows():
            if not (ch.C_min <= C_t <= ch.C_max):
                continue
            L = interp_L_at_C(ch.g, C_t)
            if L is None:
                continue
            pts.append((ch.N, L, ch.label))
        if not pts:
            continue
        pts_df = pd.DataFrame(pts, columns=["N", "L", "label"])
        n_unique_N = pts_df.N.nunique()
        para = None
        if n_unique_N >= 3:
            para = fit_parabola(np.log(pts_df.N.to_numpy()), pts_df.L.to_numpy())
        parabolas.append({
            "C": float(C_t),
            "n_chains": int(len(pts_df)),
            "n_unique_N": int(n_unique_N),
            "para": para,
            "points": pts_df.to_dict("records"),
        })
        # bracketed：谷底须落在该切片实测 N 范围内（paper/FIT_PROTOCOL 可靠性判据）；
        # 未 bracket 的谷底是抛物线外推，L* 常低于不可约项 E，不得进入 slope 回归。
        bracketed = bool(para is not None
                         and pts_df.N.min() <= para["N_star"] <= pts_df.N.max())
        left = int((pts_df.N < para["N_star"]).sum()) if para else 0
        right = int((pts_df.N > para["N_star"]).sum()) if para else 0
        slice_rows.append({
            "C": float(C_t),
            "n_chains": int(len(pts_df)),
            "n_unique_N": int(n_unique_N),
            "N_star": para["N_star"] if para else None,
            "L_star": para["L_star"] if para else None,
            "para_a": para["a"] if para else None,
            "bracketed": bracketed, "n_left": left, "n_right": right,
            "submission_bracketing_rule": bool(left >= 2 and right >= 2),
        })
    slice_df = pd.DataFrame(slice_rows)
    print("\n=== IsoFLOP slice results ===")
    print(slice_df.to_string(index=False))

    # Power-law regress log(N*) on log(C)：只用 bracketed 谷底（外推谷底不可信）
    valid = slice_df.dropna(subset=["N_star", "L_star"]) if len(slice_df) else slice_df
    if len(valid):
        n_pos = int((valid.para_a > 0).sum())
        valid = valid[(valid.para_a > 0) & valid.bracketed]
        print(f"\n[reliability] 正曲率 {n_pos}/{len(slice_df)}，其中 bracketed "
              f"{len(valid)}（仅 bracketed 进入 slope 回归；"
              f"其中满足投稿 left>=2&right>=2 的有 "
              f"{int(valid.submission_bracketing_rule.sum()) if len(valid) else 0}）")
    if len(valid) >= 3:
        slope_N, intercept_N = np.polyfit(np.log(valid.C), np.log(valid.N_star), 1)
        print(f"\nlog(N*) ~ slope·log(C) + b: slope = {slope_N:.4f}")
    else:
        slope_N = intercept_N = None
        print(f"\nNot enough valid slices ({len(valid)}) for power-law")

    # ── Plot 1: parabolas ──
    # 画所有有点的切片：拟出抛物线的画曲线+谷底；点数不足 3 个 N 的面板保留散点，
    # 标题注明缺口（否则 0 抛物线时整张图空白——2026-07-30 修复）。
    n_panels = len(parabolas)
    n_cols = max(1, min(3, n_panels))
    n_rows = max(1, (n_panels + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5*n_cols, 4*n_rows), squeeze=False,
                             constrained_layout=True)
    idx = 0
    for slot in parabolas:
        ax = axes[idx//n_cols, idx % n_cols]; idx += 1
        pts_df = pd.DataFrame(slot["points"])
        para = slot["para"]
        ax.scatter(pts_df.N, pts_df.L, c="C0", s=50, zorder=3)
        for _, p in pts_df.iterrows():
            ax.annotate(p.label, (p.N, p.L), fontsize=7, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")
        if para is not None:
            N_grid = np.geomspace(pts_df.N.min()*0.5, pts_df.N.max()*2, 200)
            logN_grid = np.log(N_grid)
            L_para = para["a"]*(logN_grid - para["logN_star"])**2 + para["L_star"]
            ax.plot(N_grid, L_para, "k--", lw=1.2)
            ax.axvline(para["N_star"], color="red", ls=":", alpha=0.7)
            title2 = f"N* = {fmt_num(para['N_star'])}, L* = {para['L_star']:.4f}"
            in_range = pts_df.N.min() <= para["N_star"] <= pts_df.N.max()
            if not in_range:
                title2 += "  [extrapolated]"
        elif slot["n_unique_N"] < 3:
            title2 = f"only {slot['n_unique_N']} distinct N (<3) — no parabola"
        else:
            title2 = "parabola opens downward — no minimum"
        ax.set_xscale("log")
        ax.set_xlim(pts_df.N.min()*0.5, pts_df.N.max()*2)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f"C = {slot['C']:.2e}  ({slot['n_chains']} chains, "
                     f"{slot['n_unique_N']} N)\n{title2}", fontsize=10)
        ax.set_xlabel("N"); ax.set_ylabel(args.title)
        ax.grid(True, alpha=0.3)
    for i in range(idx, n_rows*n_cols):
        axes[i//n_cols, i % n_cols].axis("off")
    fig.suptitle(f"{args.title} — IsoFLOP parabola fits (within-run interpolation, "
                 f"{args.c_label})\n"
                 "Each panel = a target compute C_k.  Vertical red dotted = N*_C.",
                 fontsize=11)
    out1 = args.out_prefix.parent / (args.out_prefix.name + "_parabolas.png")
    fig.savefig(out1, dpi=140, bbox_inches="tight")
    fig.savefig(out1.with_suffix(".pdf"), bbox_inches="tight")
    print(f"\nwrote {out1}")

    # ── Plot 2: summary N* vs C ──
    # 底层永远画每个切片的横截面点（竖线戳中的 (C, N) 与插值 L），0 谷底时图也
    # 有内容并直接可视化覆盖几何；谷底点线叠加其上（2026-07-30 修复空白图）。
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    xs_C, xs_N, xs_L = [], [], []
    for slot in parabolas:
        for p in slot["points"]:
            xs_C.append(slot["C"]); xs_N.append(p["N"]); xs_L.append(p["L"])
    if xs_C:
        ax1.scatter(xs_C, xs_N, s=28, c="0.65", edgecolor="k", linewidth=0.3,
                    zorder=2, label="cross-section points (chains hit)")
    if len(valid) > 0:
        ax1.plot(valid.C, valid.N_star, "o-", ms=9, color="C0", zorder=4,
                 label="IsoFLOP valley N*_C")
        for _, r in valid.iterrows():
            ax1.annotate(f"{fmt_num(r.N_star)}", (r.C, r.N_star), fontsize=8,
                         xytext=(5, 5), textcoords="offset points")
    if slope_N is not None:
        C_grid = np.geomspace(valid.C.min()*0.5, valid.C.max()*2, 200)
        Nstar_grid = np.exp(intercept_N + slope_N*np.log(C_grid))
        ax1.plot(C_grid, Nstar_grid, "k--", lw=1.5,
                 label=f"power-law fit: slope={slope_N:.3f}")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel(f"C = {args.c_label}"); ax1.set_ylabel("N (params)")
    ax1.set_title(f"IsoFLOP cross-sections & valleys "
                  f"({len(valid)} valid valley{'s' if len(valid) != 1 else ''})")
    ax1.legend(fontsize=8, loc="best")
    ax1.grid(True, alpha=0.3, which="both")

    if xs_C:
        sc = ax2.scatter(xs_C, xs_L, s=34, c=np.log10(xs_N), cmap="viridis",
                         edgecolor="k", linewidth=0.3, zorder=2)
        fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.03, label="$\\log_{10} N$")
    if len(valid) > 0:
        ax2.plot(valid.C, valid.L_star, "X-", ms=11, color="C3", zorder=4,
                 label="valley minimum L*_C")
        for _, r in valid.iterrows():
            ax2.annotate(f"{r.L_star:.4f}", (r.C, r.L_star), fontsize=8,
                         xytext=(5, 5), textcoords="offset points")
        ax2.legend(fontsize=8, loc="best")
    ax2.set_xscale("log")
    ax2.set_xlabel(f"C = {args.c_label}"); ax2.set_ylabel(f"{args.title} (interpolated)")
    ax2.set_title("Interpolated L at each C target (colour = model size)")
    ax2.grid(True, alpha=0.3, which="both")
    fig.suptitle(f"{args.title} — IsoFLOP summary · {len(slice_df)} slices with points, "
                 f"{int((slice_df['n_unique_N'] >= 3).sum()) if len(slice_df) else 0} with >=3 distinct N, "
                 f"{len(valid)} valid valleys", fontsize=11)
    out2 = args.out_prefix.parent / (args.out_prefix.name + "_summary.png")
    fig.savefig(out2, dpi=140, bbox_inches="tight")
    fig.savefig(out2.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out2}")

    results = {
        "method": "within-run log-C interpolation (isoflop_test_ce.py lineage)",
        "input": str(args.fit_ready.resolve()),
        "n_chains_used": int(len(chains)),
        "n_single_point_chains_dropped": int(n_single),
        "C_targets": [float(c) for c in C_targets],
        "slices": slice_rows,
        "slope_logNstar_vs_logC": (float(slope_N) if slope_N is not None else None),
        "n_valid_slices": int(len(valid)),
    }
    out3 = args.out_prefix.parent / (args.out_prefix.name + "_results.json")
    out3.write_text(json.dumps(results, indent=2))
    print(f"wrote {out3}")


if __name__ == "__main__":
    main()
