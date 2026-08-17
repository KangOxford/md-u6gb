#!/usr/bin/env python3
"""三方比较（base / hybpm / hyb）—— 机制 vs 容量的判决脚本。

自动发现盘上所有已落地的 (arm, step, seed) 产物，按「近配对三元组」分组，
对每组给出：
  1. 逐引用距离的 hybpm−base / hyb−base，以及 **等参数组复刻比** (hybpm−base)/(hyb−base)
  2. 跨 seed 的均值、极差、是否全同号、均值/极差（证据等级的标准）
  3. LOB-Bench 三方 + 逐特征分解（聚合的 WS-21 反复被一两个特征吃掉，必须同时给）

标准（§6.3 预注册，写在结果之前）：看 501–1000 档的 hybpm−base
  ≥ +15 pp        → 机制性
  ~0 或负          → 容量
  中间             → 两者都有，不下二元结论

用法：  python3 code/threeway.py            # 全部三元组
        python3 code/threeway.py 4500       # 只看某一档
"""
import json, glob, os, sys, itertools
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGES = ["1-10", "11-25", "26-50", "51-100", "101-250", "251-500", "501-1000", "before-window"]
# 近配对三元组：(标签, base_step, hybpm_step, hyb_step)。步号跨度写在标签里。
TRIPLES = [
    ("~4,500  跨度 2.08%", 4462, 4479, 4555),
    ("~4,850  跨度 2.51%", 4789, 4854, 4911),   # 中间点：判断复刻比是趋势还是涨落
    ("~5,063  跨度 0.81%", 5062, 5063, 5103),
    ("~5,300  跨度 1.11%", 5330, 5271, 5297),
    # ~6,020：hybpm 停在 5,994（训练被争用的分配挡住），但 base/hyb 在 6059/6057 各有
    # checkpoint —— 跨度 1.08%，比 ~5,300 还紧，且是**离已发表的 6265/6258 最近**的一个
    # 匹配点。所以不必把 hybpm 训到 6,400：5,994 本身就是一个可用的测量点。
    ("~6,020  跨度 1.08%", 6059, 5994, 6057),
    ("~6,260  跨度 0.11%", 6265, None, 6258),   # 已发表的两个配置；hybpm 未训到
]
ARM_LABELS = {  # 产物标签的前缀在历史上有两种写法
    "base": ["base_{s}_s{d}", "base2k_{s}_s{d}"],
    "hybpm": ["hybpm_{s}_s{d}"],
    "hyb": ["hyb_{s}_s{d}", "hyb2k_{s}_s{d}"],
}
SEEDS = [2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035]


def _load(label):
    """返回 (by_age, summary) 或 None。只认同时有 refer_success 与 summary 的目录。"""
    for f in glob.glob(f"{ROOT}/bench2k_*/refer_success_{label}.json"):
        s = os.path.join(os.path.dirname(f), "summary.json")
        if os.path.exists(s):
            return json.load(open(f))["by_age"], json.load(open(s))
    return None


def get(arm, step, seed):
    if step is None:
        return None
    for pat in ARM_LABELS[arm]:
        r = _load(pat.format(s=step, d=seed))
        if r:
            return r
    return None


def rate(by_age, a):
    d = by_age[a]
    return 100.0 * d["exact"] / d["n"]


def agg(by_age):
    n = sum(by_age[a]["n"] for a in AGES)
    return 100.0 * sum(by_age[a]["exact"] for a in AGES) / n


def stats(vals):
    m = sum(vals) / len(vals)
    sp = max(vals) - min(vals)
    same = all(v > 0 for v in vals) or all(v < 0 for v in vals)
    return m, sp, same, (abs(m) / sp if sp > 1e-12 else float("inf"))


def grade(ratio, n_seeds):
    if n_seeds < 2:
        return "B 单seed"
    if ratio >= 5:
        return "A"
    if ratio >= 2:
        return "B+"
    return "B 极差吞掉均值"


def report_triple(tag, bs, ps, hs):
    have = {}
    for d in SEEDS:
        b, p, h = get("base", bs, d), get("hybpm", ps, d), get("hyb", hs, d)
        if b and h:
            have[d] = (b[0], p[0] if p else None, h[0], b[1], p[1] if p else None, h[1])
    if not have:
        return False
    seeds = sorted(have)
    n_pm = sum(1 for d in seeds if have[d][1] is not None)
    print(f"\n{'='*104}")
    print(f"三元组 {tag}   base@{bs} / hybpm@{ps} / hyb@{hs}   "
          f"seed: {seeds}  （其中有 hybpm 的 {n_pm} 个）")
    print("=" * 104)

    # ---- 逐档 ----
    print(f"\n{'引用距离':>14} | {'hybpm−base 均值':>15} {'极差':>7} {'同号':>4} | "
          f"{'hyb−base 均值':>13} {'极差':>7} | {'等参数组复刻':>9}")
    print("-" * 104)
    for a in AGES:
        dp = [rate(have[d][1], a) - rate(have[d][0], a) for d in seeds if have[d][1]]
        dh = [rate(have[d][2], a) - rate(have[d][0], a) for d in seeds]
        if dp:
            mp, sp, sg, _ = stats(dp)
        mh, sh, _, _ = stats(dh)
        frac = 100 * mp / mh if dp and abs(mh) > 1e-9 else float("nan")
        pm = f"{mp:+15.4f} {sp:7.4f} {'yes' if sg else 'NO':>4}" if dp else f"{'—':>15} {'—':>7} {'—':>4}"
        print(f"{a:>14} | {pm} | {mh:+13.4f} {sh:7.4f} | {frac:8.1f}%")
    dp = [agg(have[d][1]) - agg(have[d][0]) for d in seeds if have[d][1]]
    dh = [agg(have[d][2]) - agg(have[d][0]) for d in seeds]
    mh, sh, _, rh = stats(dh)
    if dp:
        mp, sp, sg, rp = stats(dp)
        print("-" * 104)
        print(f"{'整体':>14} | {mp:+15.4f} {sp:7.4f} {'yes' if sg else 'NO':>4} | "
              f"{mh:+13.4f} {sh:7.4f} | {100*mp/mh:8.1f}%")
        print(f"{'':>14}   证据等级 hybpm−base: {grade(rp, len(dp))} (均值/极差={rp:.1f})   "
              f"hyb−base: {grade(rh, len(dh))} ({rh:.1f})")

        # ---- 预注册标准 ----
        far = [rate(have[d][1], "501-1000") - rate(have[d][0], "501-1000") for d in seeds if have[d][1]]
        mf, sf, sgf, _ = stats(far)
        verdict = ("机制性 (architectural)" if mf >= 15 else
                   "容量 (capacity)" if mf <= 2 else "两者兼有，不下二元结论")
        print(f"\n  ▶ 预注册标准（501–1000 档 hybpm−base）：{mf:+.4f} pp "
              f"[极差 {sf:.4f}, {len(far)} seed] → **{verdict}**")
    else:
        print("-" * 104)
        print(f"{'整体':>14} | {'—':>15} {'—':>7} {'—':>4} | {mh:+13.4f} {sh:7.4f} |")

    # ---- LOB-Bench ----
    print(f"\n  LOB-Bench 三方（越小越好）")
    print(f"  {'指标':>8} | {'hybpm−base 均值':>15} {'极差':>9} | {'hyb−base 均值':>13} {'极差':>9} | "
          f"{'等参数组占':>8} | {'噪声关卡':>10}")
    print("  " + "-" * 96)
    for m in ("ws21", "ks21", "l1_21"):
        base_lvl = sum(have[d][3][m] for d in seeds) / len(seeds)
        dpm = [have[d][4][m] - have[d][3][m] for d in seeds if have[d][4]]
        dhy = [have[d][5][m] - have[d][3][m] for d in seeds]
        mh2, sh2, _, _ = stats(dhy)
        if dpm:
            mp2, sp2, _, _ = stats(dpm)
            rel = 100 * abs(mp2) / base_lvl
            gate = "不可分辨" if rel < 16.6 else "可分辨"
            print(f"  {m:>8} | {mp2:+15.6f} {sp2:9.6f} | {mh2:+13.6f} {sh2:9.6f} | "
                  f"{100*mp2/mh2 if abs(mh2)>1e-12 else float('nan'):7.1f}% | {rel:5.2f}% {gate}")
        else:
            print(f"  {m:>8} | {'—':>15} {'—':>9} | {mh2:+13.6f} {sh2:9.6f} |")
    print("  （噪声关卡：WS-21 跨 seed 噪声底 16.6%，出处 BASELINE.md；相对差小于它就不可分辨）")

    # ---- 逐特征（只在有 hybpm 时，用第一个 seed）----
    d0 = next((d for d in seeds if have[d][4]), None)
    if d0:
        fb, fp, fh = have[d0][3]["feature_scores"], have[d0][4]["feature_scores"], have[d0][5]["feature_scores"]
        W = lambda x, f: x[f]["wasserstein"]
        N = len(fb)
        tot = sum(W(fp, f) - W(fb, f) for f in fb) / N
        worst = max(fb, key=lambda f: W(fp, f) - W(fb, f))
        c = (W(fp, worst) - W(fb, worst)) / N
        print(f"\n  逐特征（s{d0}）：hybpm−base 的 WS-21 总差 {tot:+.6f}；"
              f"最大单项 `{worst}` 贡献 {c:+.6f} = {100*c/tot if abs(tot)>1e-12 else 0:.1f}%；"
              f"其余 {N-1} 项净 {tot - c:+.6f}")
        g = {"成交量/价差": ["ask_volume","bid_volume","ask_volume_touch","bid_volume_touch","spread","orderbook_imbalance","vol_per_min"],
             "订单流不平衡": ["ofi","ofi_up","ofi_down","ofi_stay"],
             "时间": ["log_inter_arrival_time","log_time_to_cancel"],
             "撤单位置": ["ask_cancellation_depth","ask_cancellation_ticks","bid_cancellation_depth","bid_cancellation_ticks"],
             "挂单位置": ["limit_ask_order_depth","limit_ask_order_ticks","limit_bid_order_depth","limit_bid_order_ticks"]}
        print(f"  {'组':>12} {'项':>3} | {'hybpm−base':>11} {'hyb−base':>10}")
        for k, v in g.items():
            v = [f for f in v if f in fb]
            if v:
                print(f"  {k:>12} {len(v):>3} | {sum(W(fp,f)-W(fb,f) for f in v):+11.5f} "
                      f"{sum(W(fh,f)-W(fb,f) for f in v):+10.5f}")
    return True


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    shown = 0
    for tag, bs, ps, hs in TRIPLES:
        if only and only not in tag.replace(",", ""):
            continue
        if report_triple(tag, bs, ps, hs):
            shown += 1
    if not shown:
        print("没有任何三元组的产物齐备。")
    print(f"\n{'='*104}\n读法提醒：聚合 WS-21 已经两次被单个特征吃掉全部差额"
          f"（hybrid 的优势 72% 来自 bid_volume；等参数组的劣势 100.4% 来自 log_time_to_cancel）。"
          f"\n引用聚合值时必须同时给分解。")


if __name__ == "__main__":
    main()
