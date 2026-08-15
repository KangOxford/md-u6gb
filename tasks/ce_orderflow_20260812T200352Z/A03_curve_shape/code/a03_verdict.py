#!/usr/bin/env python3
"""M4 —— 全量 487 ticker × 2 月的最终裁决：水平与斜率，配对自举。

为什么要重算而不是读现成的 CI。npz 里存的是**每条臂各自的**自举 CI（边缘 CI）。
用「两条边缘 CI 是否重叠」判断两臂差异，在配对数据上保守到错：draft 与 corr
测在**同一批 487 个 ticker** 上，而 ticker 之间的异质性（`price_rel` 跨数量级）
同时抬高两条边缘 CI —— 那部分方差在**差值**里会抵消掉。

所以这里重采样的单位仍是 ticker（误差在 ticker 内相关，这是 A02 定的口径），
但被平均的量换成逐 ticker 的差 `corr_excess - draft_excess`。

三个臂的分工：

    draft   预训练模型的原始输出（要打败的对象）
    corr    加了学到的残差 P
    rand    加了**同样 Frobenius 范数**的随机方向残差 —— 归因对照。
            rand 也做到的事情，不能算 P 学到的东西。

判据（E7，跑之前就写在 plans 里）：
    >= 3/5 字段同时满足「水平更低」且「斜率非正」，且两个月都复现。
"""
import json
import sys

import numpy as np

FIELDS = ["event_type", "direction", "price_rel", "size", "log10_dt"]
ARMS = ["draft", "corr", "rand"]
NBOOT = 2000
SEED = 20260815


def per_ticker_stats(cur, floor, grid):
    """(T, M) 曲线 -> 每个 ticker 的 (水平@末点, 斜率×100)。

    超出基底 = 臂 - 基底。斜率用 OLS，只拟合有限点；有限点少于 5 个的 ticker
    在该字段上记 NaN，而不是用少数点外推出一个斜率。
    """
    ex = cur - floor
    lvl = ex[:, -1]
    slp = np.full(ex.shape[0], np.nan)
    for i in range(ex.shape[0]):
        m = np.isfinite(ex[i])
        if m.sum() >= 5:
            slp[i] = np.polyfit(grid[m], ex[i][m], 1)[0] * 100.0
    return lvl, slp


def paired_boot(a, b, rng, nboot=NBOOT):
    """对 a - b 做 ticker-block 自举，返回 (均值, lo, hi, P(diff<0))。"""
    d = a - b
    ok = np.isfinite(d)
    d = d[ok]
    if d.size < 20:
        return np.nan, np.nan, np.nan, np.nan
    idx = rng.integers(0, d.size, size=(nboot, d.size))
    bs = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), \
        float(np.percentile(bs, 97.5)), float((bs < 0).mean())


def one(path, tag):
    z = np.load(path, allow_pickle=True)
    grid = z["grid"].astype(float)
    rng = np.random.default_rng(SEED)
    out = {}
    for f in FIELDS:
        floor = z[f"{f}__floor"]
        st = {a: per_ticker_stats(z[f"{f}__{a}"], floor, grid) for a in ARMS}
        row = {}
        for a in ARMS:
            row[f"{a}_level"] = float(np.nanmean(st[a][0]))
            row[f"{a}_slope"] = float(np.nanmean(st[a][1]))
        for a in ("corr", "rand"):
            for k, i in (("level", 0), ("slope", 1)):
                m, lo, hi, p = paired_boot(st[a][i], st["draft"][i], rng)
                row[f"{a}_d{k}"] = m
                row[f"{a}_d{k}_lo"] = lo
                row[f"{a}_d{k}_hi"] = hi
                row[f"{a}_d{k}_pneg"] = p
        out[f] = row
    return out


def fmt(v, w=7, p=3):
    return f"{v:{w}.{p}f}" if np.isfinite(v) else " " * (w - 3) + "n/a"


def show(res, tag):
    print(f"\n{'=' * 108}\n{tag}\n{'=' * 108}")
    print(f"{'field':<11} {'draft_lv':>9} {'corr_lv':>8} "
          f"{'Δ水平 [95% CI]':>26} {'draft_sl':>9} {'corr_sl':>8} "
          f"{'Δ斜率 [95% CI]':>26}")
    print("-" * 108)
    for f in FIELDS:
        r = res[f]
        dl = f"{r['corr_dlevel']:+.3f} [{r['corr_dlevel_lo']:+.3f},{r['corr_dlevel_hi']:+.3f}]"
        ds = f"{r['corr_dslope']:+.3f} [{r['corr_dslope_lo']:+.3f},{r['corr_dslope_hi']:+.3f}]"
        lv_win = "✓" if r["corr_dlevel_hi"] < 0 else ("~" if r["corr_dlevel"] < 0 else "✗")
        sl_win = "✓" if r["corr_dslope_hi"] < 0 else ("~" if r["corr_dslope"] < 0 else "✗")
        print(f"{f:<11} {fmt(r['draft_level'],9)} {fmt(r['corr_level'],8)} "
              f"{dl:>24} {lv_win} {fmt(r['draft_slope'],9)} {fmt(r['corr_slope'],8)} "
              f"{ds:>24} {sl_win}")
    print("-" * 108)
    print("✓ = 95% CI 整个落在 0 以下（显著更好）   ~ = 均值更好但 CI 跨 0   ✗ = 更差")


def show_rand(res, tag):
    """归因对照：随机方向也做到的，不算 P 学到的。"""
    print(f"\n[归因对照 · {tag}] 随机方向残差（同 ||P||）相对 draft")
    print(f"{'field':<11} {'Δ水平(corr)':>12} {'Δ水平(rand)':>12} "
          f"{'Δ斜率(corr)':>12} {'Δ斜率(rand)':>12}  裁决")
    for f in FIELDS:
        r = res[f]
        # 学到的必须显著好于随机，否则「优势」只是加了个残差本身
        v = "学到的更好" if (r["corr_dslope"] < r["rand_dslope"] - 1e-9) else "随机也做到"
        print(f"{f:<11} {r['corr_dlevel']:>+12.3f} {r['rand_dlevel']:>+12.3f} "
              f"{r['corr_dslope']:>+12.3f} {r['rand_dslope']:>+12.3f}  {v}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "kl"
    allres = {}
    for mo in ("2026-01", "2026-02"):
        p = f"out/fx488_{mo}_{which}.npz"
        res = one(p, mo)
        allres[mo] = res
        show(res, f"fx488 全量 487 ticker · {mo} · {which}"
                  f"{'（已剔除非法箱）' if 'legal' in which else ''}")
        show_rand(res, mo)

    # E7 判据：两个月都要满足
    print(f"\n{'=' * 108}\nE7 判据（跑之前锁死）：>=3/5 字段同时「水平更低」且「斜率非正」，两个月都复现\n{'=' * 108}")
    both = []
    for f in FIELDS:
        ok = []
        for mo in ("2026-01", "2026-02"):
            r = allres[mo][f]
            ok.append(r["corr_dlevel"] < 0 and r["corr_slope"] <= 0)
        mark = "✓✓" if all(ok) else ("✓ " if any(ok) else "  ")
        print(f"  {f:<11} 2026-01 {'通过' if ok[0] else '未过'}   "
              f"2026-02 {'通过' if ok[1] else '未过'}   {mark}")
        if all(ok):
            both.append(f)
    print(f"\n跨月同时满足两条的字段: {len(both)}/5  {both}")
    print("裁决: " + ("**达成**" if len(both) >= 3 else f"**未达成**（还差 {3 - len(both)} 个字段）"))

    json.dump(allres, open(f"out/verdict_fx488_{which}.json", "w"), indent=1)
    print(f"\n写出 out/verdict_fx488_{which}.json")
