#!/usr/bin/env python3
"""任务 0: LOB 数据的属性分析 —— 论证「为什么噪声协方差在这份数据上是个真问题」。

产出 data_analysis.json + 四张图。核心是量化数据偏离各向同性的程度, 因为
iid 噪声 diffusion 的隐含假设正是"各方向等价"。
"""
from __future__ import annotations

import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_dataset, standardize, noise_covariance, FEATURE_NAMES

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DS = "/lus/lfs1aip2/projects/public/u6gb/datasets/lob_flat43_example_20260807T135612Z"
T8 = ["GOOG", "AAPL", "NVDA", "AMZN", "META", "TSLA", "MSFT", "AMD"]
OUT = "/projects/public/u6gb/tasks/noise_cov_diffusion_20260807T213317Z"


def main():
    T = 16
    X = load_dataset(DS, T8, T=T, stride=T, max_rows_per_ticker=200_000, seed=0)
    Xz, mu, sd = standardize(X)
    D = T * 43
    Xf = Xz.reshape(len(Xz), D)
    R = {}

    # ---- 1. 单行 43 通道的相关结构 ----
    A = Xz.reshape(-1, 43)
    C43 = np.corrcoef(A, rowvar=False)
    off = C43[~np.eye(43, dtype=bool)]
    w43 = np.linalg.eigvalsh(C43)[::-1]
    R["cross_section"] = {
        "n_rows": int(len(A)),
        "mean_abs_offdiag_corr": float(np.abs(off).mean()),
        "frac_offdiag_abs_gt_0.9": float((np.abs(off) > 0.9).mean()),
        "frac_offdiag_abs_gt_0.5": float((np.abs(off) > 0.5).mean()),
        "cond_number": float(w43[0] / max(w43[-1], 1e-12)),
        "pc1_var_explained": float(w43[0] / w43.sum()),
        "n_pc_for_95pct": int(np.searchsorted(np.cumsum(w43) / w43.sum(), 0.95) + 1),
    }

    # ---- 2. 窗口(D=688)的协方差谱 ----
    S = noise_covariance(Xf)
    w = np.clip(np.linalg.eigvalsh(S), 1e-12, None)[::-1]
    cw = np.cumsum(w) / w.sum()
    R["window_cov"] = {
        "D": D,
        "cond_number": float(w[0] / w[-1]),
        "pc1_var_explained": float(w[0] / w.sum()),
        "n_pc_for_50pct": int(np.searchsorted(cw, 0.50) + 1),
        "n_pc_for_90pct": int(np.searchsorted(cw, 0.90) + 1),
        "n_pc_for_99pct": int(np.searchsorted(cw, 0.99) + 1),
        "effective_rank_participation": float(w.sum() ** 2 / (w ** 2).sum()),
        "effective_rank_entropy": float(np.exp(-(w / w.sum() * np.log(w / w.sum())).sum())),
    }

    # ---- 3. 时间自相关(沿窗口内时间轴) ----
    ac = {}
    for ch, nm in [(0, "log_dt"), (1, "ask_p1_tick"), (11, "ask_v1_log"), (41, "trade_p_tick")]:
        s = Xz[:, :, ch]
        s = s - s.mean()
        a = [float((s[:, :T - k] * s[:, k:]).mean() / max(s.var(), 1e-12)) for k in range(min(8, T))]
        ac[nm] = a
    R["temporal_autocorr"] = ac

    # ---- 4. 边际分布的非高斯性 ----
    from scipy import stats as st
    marg = {}
    for j, nm in enumerate(FEATURE_NAMES):
        v = A[:, j]
        marg[nm] = {"skew": float(st.skew(v)), "kurtosis": float(st.kurtosis(v)),
                    "jb_reject_5pct": bool(st.jarque_bera(v[:20000]).pvalue < 0.05)}
    R["marginals"] = marg
    R["marginal_summary"] = {
        "n_channels": 43,
        "n_reject_normality": int(sum(m["jb_reject_5pct"] for m in marg.values())),
        "median_abs_skew": float(np.median([abs(m["skew"]) for m in marg.values()])),
        "median_excess_kurtosis": float(np.median([m["kurtosis"] for m in marg.values()])),
    }

    # ---------------- 图 ----------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    im = ax[0].imshow(C43, cmap="RdBu_r", vmin=-1, vmax=1)
    ax[0].set_title("43 channel correlation (single row)")
    for b, lab in [(0.5, "dt"), (10.5, "ask_p"), (20.5, "ask_v"), (30.5, "bid_p"), (40.5, "bid_v")]:
        ax[0].axhline(b, c="k", lw=.6); ax[0].axvline(b, c="k", lw=.6)
    plt.colorbar(im, ax=ax[0], fraction=.046)
    ax[1].loglog(np.arange(1, D + 1), w, lw=2, label="LOB window covariance")
    ax[1].axhline(1.0, c="r", ls="--", lw=2, label="iid noise (all eigenvalues = 1)")
    ax[1].set_xlabel("eigenvalue index"); ax[1].set_ylabel("eigenvalue")
    ax[1].set_title(f"Spectrum, D={D}, cond={w[0]/w[-1]:.2e}"); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/figs/fig0_structure.png", dpi=130); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(np.arange(1, D + 1), cw, lw=2, label="LOB")
    ax[0].plot(np.arange(1, D + 1), np.arange(1, D + 1) / D, "r--", lw=2, label="iid")
    ax[0].set_xscale("log"); ax[0].set_xlabel("# principal components")
    ax[0].set_ylabel("cumulative variance explained")
    ax[0].set_title("Variance concentration"); ax[0].legend(); ax[0].grid(alpha=.3)
    for nm, a in ac.items():
        ax[1].plot(a, marker="o", label=nm)
    ax[1].axhline(0, c="k", lw=.8); ax[1].set_xlabel("lag (events)")
    ax[1].set_ylabel("autocorrelation"); ax[1].set_title("Temporal autocorrelation")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/figs/fig0b_concentration.png", dpi=130); plt.close()

    with open(f"{OUT}/data_analysis.json", "w") as f:
        json.dump(R, f, indent=2, default=float)

    cs, wc, ms = R["cross_section"], R["window_cov"], R["marginal_summary"]
    print(f"截面(43 通道): 平均|非对角相关|={cs['mean_abs_offdiag_corr']:.3f}, "
          f"|r|>0.9 占 {100*cs['frac_offdiag_abs_gt_0.9']:.1f}%, 条件数={cs['cond_number']:.2e}")
    print(f"           PC1 解释 {100*cs['pc1_var_explained']:.1f}%, 95% 方差只需 {cs['n_pc_for_95pct']}/43 个主成分")
    print(f"窗口(D={wc['D']}): 条件数={wc['cond_number']:.3e}, PC1={100*wc['pc1_var_explained']:.1f}%")
    print(f"           50%/90%/99% 方差分别只需 {wc['n_pc_for_50pct']}/{wc['n_pc_for_90pct']}/{wc['n_pc_for_99pct']} 个主成分")
    print(f"           有效秩(participation)={wc['effective_rank_participation']:.1f}, "
          f"(entropy)={wc['effective_rank_entropy']:.1f}   [iid 时两者都 = {wc['D']}]")
    print(f"边际: {ms['n_reject_normality']}/43 个通道拒绝正态(JB, 5%), "
          f"中位|偏度|={ms['median_abs_skew']:.2f}, 中位超额峰度={ms['median_excess_kurtosis']:.2f}")
    print(f"时间自相关 lag1: " + ", ".join(f"{k}={v[1]:+.3f}" for k, v in ac.items()))


if __name__ == "__main__":
    main()
