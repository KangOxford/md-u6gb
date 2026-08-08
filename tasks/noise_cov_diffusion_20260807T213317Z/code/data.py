#!/usr/bin/env python3
"""把 43 列 LOB 宽表转成 diffusion 可训练的标准化窗口张量, 并估计噪声协方差。

设计要点
--------
原始 43 列量纲差 6 个数量级(价格 ~2.8e6, 量 ~1e2, Δt ~1e4..1e9), 直接做 diffusion
会被价格列主导。这里只做**可逆的特征化**(归一化在 normalize.py, 与本文件分离,
因为归一化口径是可切换的实验变量, 而特征化是固定的数据契约):

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


DS = "/lus/lfs1aip2/projects/public/u6gb/datasets/lob_flat43_example_20260807T135612Z"
T8 = ["GOOG", "AAPL", "NVDA", "AMZN", "META", "TSLA", "MSFT", "AMD"]

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


def noise_covariance_toeplitz(S, T, C, blend=1.0, verbose=True):
    """把 Σ 投影到**块 Toeplitz**(时间平稳)子空间, 再与原 Σ 按 blend 混合。

    动机: Σ 是 (T·C)×(T·C), 要从 9 万个窗口估 T²C²/2 ≈ 23.7 万个自由参数。
    但如果时间轴平稳, Σ[i,j] 只依赖 i-j, 自由参数降到 T·C² ≈ 2.96 万(8 倍)。
    沿每条块对角线平均就是到该子空间的正交投影 —— 它**恰好去噪的是滞后结构**,
    而滞后结构正是 ACF 距离测的东西。这是唯一一个"改进方向"与"评估指标"对齐的候选
    (见 REPORT 13H.5: 收缩这条路已被留出集似然否掉, 估计精度不是瓶颈)。

    blend<1 时做部分投影, 保留一部分非平稳成分(开盘/收盘效应可能是真的)。
    结果重新归一化到 trace=D, 与其它臂等噪声能量。
    """
    D = T * C
    assert S.shape == (D, D)
    B = S.reshape(T, C, T, C)
    P = np.zeros_like(B)
    for k in range(-(T - 1), T):
        idx = [(i, i - k) for i in range(T) if 0 <= i - k < T]
        m = np.mean([B[i, :, j, :] for i, j in idx], axis=0)
        for i, j in idx:
            P[i, :, j, :] = m
    P = P.reshape(D, D)
    P = 0.5 * (P + P.T)                       # 数值对称化
    out = blend * P + (1 - blend) * S
    # 投影可能把最小特征值压到 0 以下(平均是收缩操作), 抬回正定
    w = np.linalg.eigvalsh(out)
    if w.min() <= 1e-8:
        out = out + (1e-8 - w.min()) * np.eye(D)
    out *= D / np.trace(out)
    if verbose:
        er = lambda M: (lambda v: (v.sum() ** 2) / (v ** 2).sum())(
            np.clip(np.linalg.eigvalsh(M), 0, None))
        print(f"[toeplitz] blend={blend} 自由参数 {T*T*C*C//2:,} -> {T*C*C:,} "
              f"({T*T*C*C//2/(T*C*C):.1f}x), 有效秩 {er(S):.2f} -> {er(out):.2f}, "
              f"‖ΔΣ‖/‖Σ‖={np.linalg.norm(out-S)/np.linalg.norm(S):.4f}")
    return out


def noise_covariance_tilt(S, gamma, verbose=True):
    """谱倾斜: Σ(γ) ∝ Q diag(λ^γ) Qᵀ, 归一化到 trace=D。γ=1 即原 Σ。

    γ<1 压平谱(向各向同性), γ>1 磨尖。一个参数的外层搜索族 ——
    由**生成质量**(外生判据)选 γ, 不由去噪损失选(那个结构上选不出, 见 13H.4c)。
    """
    D = S.shape[0]
    lam, Q = np.linalg.eigh(S)
    lam = np.clip(lam, 1e-12, None) ** gamma
    lam = lam * D / lam.sum()
    out = (Q * lam) @ Q.T
    if verbose:
        print(f"[tilt] γ={gamma} 有效秩 {(np.clip(np.linalg.eigvalsh(S),0,None).sum()**2)/(np.clip(np.linalg.eigvalsh(S),0,None)**2).sum():.2f}"
              f" -> {(lam.sum()**2)/(lam**2).sum():.2f}")
    return out


def noise_covariance_shrunk(Xtr, Xval, grid=None, verbose=True):
    """用**留出集高斯似然**选收缩强度, 而不是硬编码 shrink=1e-3。

    动机(见 REPORT 13H.4c): 纯去噪损失结构上选不出好的 Σ —— Σ 同时是被学的参数和
    定义任务的东西, 学习只会把它推向"让任务变简单"。要让"学 Σ"有意义, 信号必须外生。
    数据本身就是那个外生信号: D=688 维、9 万个训练窗口, **大特征值估得准,
    小特征值几乎全是采样噪声**(Marchenko-Pastur)。收缩强度该由留出集决定, 不该拍脑袋。

    评分用留出集上的高斯对数似然(常数项省略, 只比较相对大小):
        ll(a) = -[ log det S(a) + tr(S(a)^-1 C_val) ] / D
    注意分解只做一次: S(a) = (1-a)S_tr + a*(tr/D)I 在 S_tr 的特征基里是对角的,
    所以整条 a 网格只需要一次 eigh, 不是每个 a 一次求逆。
    """
    D = Xtr.shape[1]
    S_tr = np.cov(Xtr, rowvar=False)
    S_tr *= D / np.trace(S_tr)
    C_val = np.cov(Xval, rowvar=False)
    C_val *= D / np.trace(C_val)
    lam, Q = np.linalg.eigh(S_tr)
    lam = np.clip(lam, 0, None)
    # C_val 在 S_tr 特征基下的对角元 —— tr(S(a)^-1 C_val) 只需要这些
    cv = np.einsum("ij,jk,ki->i", Q.T, C_val, Q)
    if grid is None:
        grid = np.concatenate([[0.0], np.logspace(-4, 0, 33)])
    best, best_ll = None, -np.inf
    for a in grid:
        d = (1 - a) * lam + a * 1.0          # trace(S_tr)/D = 1, 故各向同性部分就是 1
        d = d * D / d.sum()
        if d.min() <= 0:
            continue
        ll = -(np.log(d).sum() + (cv / d).sum()) / D
        if ll > best_ll:
            best, best_ll = float(a), float(ll)
    d = (1 - best) * lam + best
    d = d * D / d.sum()
    S = (Q * d) @ Q.T
    if verbose:
        er0 = (lam.sum() ** 2) / (lam ** 2).sum()
        er1 = (d.sum() ** 2) / (d ** 2).sum()
        print(f"[shrink] 留出集选出 a={best:.5g} (对照硬编码 1e-3), "
              f"留出 loglik={best_ll:.4f}, 有效秩 {er0:.2f} -> {er1:.2f}")
    return S, best, best_ll


