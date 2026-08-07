#!/usr/bin/env python3
"""把 43 列 LOB 宽表转成 diffusion 可训练的标准化窗口张量, 并估计噪声协方差。

设计要点
--------
原始 43 列量纲差 6 个数量级(价格 ~2.8e6, 量 ~1e2, Δt ~1e4..1e9), 直接做 diffusion
会被价格列主导。这里做一次**可逆**的特征化再 z-score:

    col 0        log1p(Δt_ns)
    价格 20 列    (p - mid) / tick      -> 相对最优中间价的 tick 偏移
    量   20 列    log1p(size)
    trade_price  (p - mid) / tick, 无成交为 0
    trade_qty    log1p(qty)

用相对 mid 而非绝对价, 是因为绝对价的协方差会被 mid 的日内漂移这一个主成分吃掉
(所有价格列相关系数 >0.9999), 那样的"强相关"只反映非平稳性, 不反映盘口结构。
相对化之后剩下的相关才是真正的截面结构: 档位间距、两侧联动、量价关系。

窗口: x ∈ R^{T×43} 展平成 D=T*43 维。协方差在这个 D 维上估计, 因此同时覆盖
截面相关(列与列)与时间相关(行与行)。
"""
from __future__ import annotations

import json
import os

import numpy as np

TICK = 100                      # LOBSTER 最小变动 = 100 (即 $0.01)
N_LEVELS = 10
COL_DT = 0
COL_TRADE_P, COL_TRADE_Q = 41, 42
# 43 列里价格/量的列号: 每档 4 列 [ask_p, ask_v, bid_p, bid_v] 从 col1 起
ASK_P = [1 + 4 * i for i in range(N_LEVELS)]
ASK_V = [2 + 4 * i for i in range(N_LEVELS)]
BID_P = [3 + 4 * i for i in range(N_LEVELS)]
BID_V = [4 + 4 * i for i in range(N_LEVELS)]

FEATURE_NAMES = (
    ["log_dt"]
    + [f"ask_p{i+1}_tick" for i in range(N_LEVELS)]
    + [f"ask_v{i+1}_log" for i in range(N_LEVELS)]
    + [f"bid_p{i+1}_tick" for i in range(N_LEVELS)]
    + [f"bid_v{i+1}_log" for i in range(N_LEVELS)]
    + ["trade_p_tick", "trade_q_log"]
)
assert len(FEATURE_NAMES) == 43


def featurize(raw):
    """(N,43) 原始整数 -> (N,43) float 特征。返回 (feat, mid) ; mid 用于反变换。"""
    raw = raw.astype(np.float64)
    mid = 0.5 * (raw[:, ASK_P[0]] + raw[:, BID_P[0]])
    f = np.empty((len(raw), 43), dtype=np.float64)
    f[:, 0] = np.log1p(np.maximum(raw[:, COL_DT], 0))
    f[:, 1:11] = (raw[:, ASK_P] - mid[:, None]) / TICK
    f[:, 11:21] = np.log1p(raw[:, ASK_V])
    f[:, 21:31] = (raw[:, BID_P] - mid[:, None]) / TICK
    f[:, 31:41] = np.log1p(raw[:, BID_V])
    has_tr = raw[:, COL_TRADE_P] > 0
    f[:, 41] = np.where(has_tr, (raw[:, COL_TRADE_P] - mid) / TICK, 0.0)
    f[:, 42] = np.log1p(np.maximum(raw[:, COL_TRADE_Q], 0))
    return f, mid


def make_windows(f, T, stride):
    """(N,43) -> (M, T, 43) 滑窗。"""
    n = (len(f) - T) // stride + 1
    idx = np.arange(T)[None, :] + (np.arange(n) * stride)[:, None]
    return f[idx]


def load_dataset(ds_dir, tickers, T=16, stride=16, max_rows_per_ticker=200_000, seed=0):
    xs = []
    for t in tickers:
        p = os.path.join(ds_dir, f"{t}_2025-12-01_flat43.npy")
        raw = np.load(p, mmap_mode="r")[:max_rows_per_ticker]
        f, _ = featurize(np.asarray(raw))
        xs.append(make_windows(f, T, stride))
    X = np.concatenate(xs, 0)
    rng = np.random.default_rng(seed)
    rng.shuffle(X)
    return X


def standardize(X):
    """按 (T,43) 里的 43 个通道做 z-score, 时间轴共享统计量(平稳性假设)。"""
    mu = X.reshape(-1, X.shape[-1]).mean(0)
    sd = X.reshape(-1, X.shape[-1]).std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (X - mu) / sd, mu, sd


def noise_covariance(Xflat, shrink=1e-3):
    """在展平的 D 维上估计噪声协方差 Σ, 并归一化到 trace(Σ)=D。

    归一化是**公平对照的关键**: 两个 arm 注入的总噪声能量必须相同, 否则比较的是
    噪声强度而不是噪声的形状。shrink 是 Ledoit-Wolf 式的对角收缩, 保证正定可 Cholesky。
    """
    D = Xflat.shape[1]
    S = np.cov(Xflat, rowvar=False)
    S = (1 - shrink) * S + shrink * np.eye(D) * np.trace(S) / D
    S *= D / np.trace(S)
    return S


if __name__ == "__main__":
    DS = "/lus/lfs1aip2/projects/public/u6gb/datasets/lob_flat43_example_20260807T135612Z"
    T8 = ["GOOG", "AAPL", "NVDA", "AMZN", "META", "TSLA", "MSFT", "AMD"]
    T, STRIDE = 16, 16
    X = load_dataset(DS, T8, T=T, stride=STRIDE, max_rows_per_ticker=200_000)
    Xz, mu, sd = standardize(X)
    D = T * 43
    Xf = Xz.reshape(len(Xz), D)
    S = noise_covariance(Xf)
    w = np.linalg.eigvalsh(S)
    w = np.clip(w, 1e-12, None)[::-1]
    print(f"窗口张量 {X.shape}  展平维度 D={D}")
    print(f"协方差 Σ: trace={np.trace(S):.1f} (=D), 条件数={w[0]/w[-1]:.3e}")
    print(f"特征值 top5 = {np.round(w[:5],2)}   bottom5 = {np.round(w[-5:],6)}")
    for k in (1, 5, 10, 50, 100):
        print(f"  前 {k:3d} 个主成分解释方差 {100*w[:k].sum()/w.sum():5.2f}%  ({k}/{D} = {100*k/D:.1f}% 的维度)")
    # 各向同性基线: iid 噪声的协方差就是 I, 条件数 1
    print(f"\n对照: iid 噪声 Σ=I, 条件数 1.0, 前 {D//10} 个主成分解释 10.0%")
