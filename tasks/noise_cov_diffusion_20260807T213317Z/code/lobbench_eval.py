#!/usr/bin/env python3
"""从生成的 (M,T,43) 特征窗口算 LOB-Bench feature, 并用权威指标打分。

指标实现照抄 lob_pipeline/lob_bench/metrics.py 的定义(2026-08-07 核对):
  wasserstein : real+gen 合并后 z-score, 再 scipy.stats.wasserstein_distance
  ks          : 同样先合并 z-score, 再 scipy.stats.ks_2samp().statistic
  l1          : 分箱计数各自归一化成概率后 |p-q|.sum()/2  (= 总变差距离, 上限 1)

覆盖 WS-21 中可从 43 列计算的 12 个 feature(见 COVERED)。另外 9 个(见 NOT_COVERED)
需要 order-id 追踪或 message type, 43 列宽表刻意不含这些字段, 故不在范围内。
这个划分不是任意的: 覆盖的全是盘口状态的函数(截面/跨时间统计量), 不覆盖的全需要
订单身份追踪。
"""
from __future__ import annotations

import numpy as np
from scipy import stats

# 特征空间的列索引 (见 data.py FEATURE_NAMES)
C_LOGDT = 0
ASK_P = slice(1, 11)
ASK_V = slice(11, 21)
BID_P = slice(21, 31)
BID_V = slice(31, 41)
C_TRADE_P, C_TRADE_Q = 41, 42

COVERED = ("spread", "orderbook_imbalance", "log_inter_arrival_time",
           "ask_volume_touch", "bid_volume_touch", "ask_volume", "bid_volume",
           "vol_per_min", "ofi", "ofi_up", "ofi_stay", "ofi_down")
NOT_COVERED = ("log_time_to_cancel", "limit_ask_order_depth", "limit_bid_order_depth",
               "ask_cancellation_depth", "bid_cancellation_depth",
               "limit_ask_order_ticks", "limit_bid_order_ticks",
               "ask_cancellation_ticks", "bid_cancellation_ticks")


def features_from_windows(F):
    """F: (M,T,43) 特征空间数组 -> dict[feature_name] = 1D 样本数组。

    量用 expm1 反 log1p 回到手数; 价格本来就是相对 BBO 的 tick, 无需绝对 mid。
    """
    ask_p, bid_p = F[..., ASK_P], F[..., BID_P]
    ask_v = np.expm1(np.clip(F[..., ASK_V], 0, 20))
    bid_v = np.expm1(np.clip(F[..., BID_V], 0, 20))
    dt = np.expm1(np.clip(F[..., C_LOGDT], 0, 25))
    tq = np.expm1(np.clip(F[..., C_TRADE_Q], 0, 20))

    spread = (ask_p[..., 0] - bid_p[..., 0]).ravel()
    av_t, bv_t = ask_v[..., 0], bid_v[..., 0]
    av_s, bv_s = ask_v.sum(-1), bid_v.sum(-1)
    imb = ((bv_s - av_s) / np.maximum(bv_s + av_s, 1e-9)).ravel()

    # OFI: 逐窗口沿时间轴的标准定义
    dbp = np.diff(bid_p[..., 0], axis=1)
    dap = np.diff(ask_p[..., 0], axis=1)
    bv0, av0 = bid_v[..., 0], ask_v[..., 0]
    dB = np.where(dbp > 0, bv0[:, 1:], np.where(dbp < 0, -bv0[:, :-1], bv0[:, 1:] - bv0[:, :-1]))
    dA = np.where(dap < 0, av0[:, 1:], np.where(dap > 0, -av0[:, :-1], av0[:, 1:] - av0[:, :-1]))
    ofi = dB - dA
    mid = 0.5 * (ask_p[..., 0] + bid_p[..., 0])
    dmid = np.diff(mid, axis=1)

    total_t = np.maximum(dt.sum(1), 1e-9)                    # 每窗口总时长(ns)
    vpm = (tq.sum(1) / total_t) * 60e9                       # 每分钟成交量

    return {
        "spread": spread,
        "orderbook_imbalance": imb,
        "log_inter_arrival_time": np.log1p(np.maximum(dt, 0)).ravel(),
        "ask_volume_touch": av_t.ravel(),
        "bid_volume_touch": bv_t.ravel(),
        "ask_volume": av_s.ravel(),
        "bid_volume": bv_s.ravel(),
        "vol_per_min": vpm,
        "ofi": ofi.ravel(),
        "ofi_up": ofi[dmid > 0],
        "ofi_stay": ofi[dmid == 0],
        "ofi_down": ofi[dmid < 0],
    }


# ------------------------------------------------- 权威指标(照抄定义)
def _joint_z(p, q):
    a = np.concatenate([p, q])
    m, s = a.mean(), a.std()
    s = s if s > 1e-12 else 1.0
    return (p - m) / s, (q - m) / s


def wasserstein(p, q):
    p, q = _joint_z(p, q)
    return float(stats.wasserstein_distance(p, q))


def ks(p, q):
    p, q = _joint_z(p, q)
    return float(stats.ks_2samp(p, q).statistic)


def l1(p, q, n_bins=50):
    """分箱后概率的 |p-q|/2 = 总变差。分箱边界由 real 的分位数定义。"""
    lo, hi = np.quantile(p, [0.001, 0.999])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 1.0
    edges = np.linspace(lo, hi, n_bins + 1)
    edges = np.concatenate([[-np.inf], edges, [np.inf]])
    hp, _ = np.histogram(p, edges)
    hq, _ = np.histogram(q, edges)
    hp = hp / max(hp.sum(), 1)
    hq = hq / max(hq.sum(), 1)
    return float(np.abs(hp - hq).sum() / 2.0)


def score_all(real_F, gen_F):
    """返回 {feature: {l1, wasserstein, ks}} 与三项均值。"""
    R, G = features_from_windows(real_F), features_from_windows(gen_F)
    out = {}
    for k in COVERED:
        p, q = R[k], G[k]
        p = p[np.isfinite(p)]
        q = q[np.isfinite(q)]
        if len(p) < 10 or len(q) < 10:
            out[k] = {"l1": 1.0, "wasserstein": np.nan, "ks": 1.0, "n_gen": int(len(q))}
            continue
        out[k] = {"l1": l1(p, q), "wasserstein": wasserstein(p, q), "ks": ks(p, q),
                  "n_gen": int(len(q))}
    means = {m: float(np.nanmean([out[k][m] for k in COVERED])) for m in ("l1", "wasserstein", "ks")}
    return out, means
