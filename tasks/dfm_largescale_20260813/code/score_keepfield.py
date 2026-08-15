#!/usr/bin/env python3
"""逐字段 vs 逐 token 的 keep 闸门：水平、累积、**以及字段内依赖**一起报。

第三列才是这次改动的机制指标，前两列是它的后果。判据（跑之前锁死）：

  M1  `size` 的 MI(高位,低位) 必须从逐 token 的 ~0.12 明显回升（>= 0.30）
      —— 若没回升，字段内取 AND 没起作用，后面两列就不必看
  M2  整百占比必须从 ~4% 回升（>= 15%）
  G   目标区：水平与累积**同时**低于 draft 的字段数，逐字段 >= 逐 token

同一个 checkpoint（阶梯 s14000）、同一批 ticker、同一 tau —— 唯一变量是分组。
"""
import glob
import sys

import numpy as np

sys.path.insert(0, '/lus/lfs1aip2/projects/public/u6gb/tasks/'
                   'ce_orderflow_20260812T200352Z/A01_does_ce_exist/code')
from a01_ce_existence import (FIELDS, action_fields, build_counts,     # noqa: E402
                              codes, make_edges, D_curve)

R = '/lus/lfs1aip2/projects/public/u6gb/tasks/dfm_largescale_20260813/rollouts_kf'


def mi_digits(msgs, cols, col='size', base=100):
    v = msgs[:, :, cols.index(col)].ravel()
    v = v[(v >= 1)].astype(np.int64)
    if v.size < 1000:
        return float('nan'), float('nan')
    hi, lo = v // base, v % base
    n = int(max(hi.max(), lo.max())) + 1
    J = np.zeros((n, n))
    np.add.at(J, (hi, lo), 1.0)
    J /= J.sum()
    ph, pl = J.sum(1, keepdims=True), J.sum(0, keepdims=True)
    m = J > 0
    return (float((J[m] * np.log(J[m] / (ph @ pl)[m])).sum()),
            float((v % base == 0).mean()))


def load(pfx, tag, month='2026-01'):
    fs = sorted(glob.glob(f'{R}/{pfx}_{tag}_*_{month}_learned.npz'))
    cur, raw = {}, {'real': [], 'draft': [], 'corr': []}
    tks = []
    cols = None
    for p in fs:
        try:
            z = np.load(p, allow_pickle=True)
            F = action_fields(z, 100)
        except Exception:
            continue
        N, Rr = z['real_msgs'].shape[0], z['real_msgs'].shape[1]
        if N < 4 or Rr < 160:
            continue
        cols = [str(v) for v in z['msg_cols']]
        tks.append(p.split('/')[-1].replace('.npz', '').split('_')[-3])
        for k, nm in (('real_msgs', 'real'), ('draft_msgs', 'draft'),
                      ('corr_msgs', 'corr')):
            raw[nm].append(z[k])
        g = np.arange(40, Rr - 40, 20)
        perm = np.random.default_rng(20260813).permutation(N)
        G = [perm[:N // 2], perm[N // 2:]]
        for nm in FIELDS:
            try:
                e = make_edges(F['true'][nm], F['true']['_ok'], nm, 40)
                cT, K = codes(F['true'][nm], F['true']['_ok'], nm, e)
                CT = build_counts(cT, G, g, 40, Rr, K)
                fl = D_curve(CT[0], CT[1], CT[0], CT[1], K)
                for arm in ('draft', 'corr'):
                    cG, _ = codes(F[arm][nm], F[arm]['_ok'], nm, e)
                    CG = build_counts(cG, G, g, 40, Rr, K)
                    cur.setdefault((nm, arm), []).append(
                        D_curve(CT[0], CT[1], CG[0], CG[1], K) - fl)
            except Exception:
                pass
    if not tks:
        return None
    return dict(tks=tks, cols=cols,
                raw={k: np.concatenate(v) for k, v in raw.items()},
                cur={k: np.nanmean(np.vstack(v), 0) for k, v in cur.items()})


if __name__ == "__main__":
    tau = sys.argv[1] if len(sys.argv) > 1 else "t001"
    A = load('kt14k', tau)          # 逐 token
    B = load('kf14k', tau)          # 逐字段
    if A is None or B is None:
        print("产物不足"); sys.exit(0)
    print(f"tau={tau}  逐token {len(A['tks'])} ticker · 逐字段 {len(B['tks'])} ticker\n")

    print("=== 机制指标：size 两位数字之间的依赖 ===")
    print(f"{'arm':<22}{'MI(hi,lo)':>12}{'整百占比':>12}")
    for lab, D, k in (('real', A, 'real'), ('draft', A, 'draft'),
                      ('corr 逐token', A, 'corr'), ('corr 逐字段', B, 'corr')):
        mi, rs = mi_digits(D['raw'][k], D['cols'])
        print(f"{lab:<22}{mi:>12.4f}{rs:>12.4f}")
    mi_t, rs_t = mi_digits(A['raw']['corr'], A['cols'])
    mi_f, rs_f = mi_digits(B['raw']['corr'], B['cols'])
    print(f"\nM1 MI 回升到 >=0.30 : {'通过' if mi_f >= 0.30 else '未过'}"
          f"  ({mi_t:.4f} -> {mi_f:.4f})")
    print(f"M2 整百 >=15%       : {'通过' if rs_f >= 0.15 else '未过'}"
          f"  ({rs_t:.4f} -> {rs_f:.4f})")

    print("\n=== 机制指标 2：log10_dt（由 4 个 token 拼成，同一个病更重）===")
    print(f"{'arm':<22}{'中位 dt':>12}{'p99 dt':>12}{'独特值':>10}")
    def dtstat(msgs, cols):
        t = msgs[:, :, cols.index('time')].astype(np.float64)
        d = np.diff(t, axis=1).ravel(); d = d[np.isfinite(d)]; d = d[d > 0]
        return (float(np.median(d)), float(np.quantile(d, 0.99)),
                int(len(np.unique(np.round(np.log10(d), 3)))))
    for lab, D, k in (('real', A, 'real'), ('draft', A, 'draft'),
                      ('corr 逐token', A, 'corr'), ('corr 逐字段', B, 'corr')):
        m, q, u = dtstat(D['raw'][k], D['cols'])
        print(f"{lab:<22}{m:>12.3e}{q:>12.3e}{u:>10}")
    md, _, _ = dtstat(A['raw']['draft'], A['cols'])
    mt, _, _ = dtstat(A['raw']['corr'], A['cols'])
    mf, _, _ = dtstat(B['raw']['corr'], B['cols'])
    print(f"\nM3 中位 dt 向 draft 回归: {'通过' if mf >= 3 * mt else '未过'}"
          f"  ({mt:.2e} -> {mf:.2e}, draft {md:.2e})")

    print(f"\n=== 后果：水平与累积 ===")
    print(f"{'field':<12}{'':>9}{'水平':>10}{'累积':>10}   目标区")
    n = {'tok': 0, 'fld': 0}
    for f in FIELDS:
        dl = float(A['cur'][(f, 'draft')][-1])
        da = float(A['cur'][(f, 'draft')][-1] - A['cur'][(f, 'draft')][0])
        print(f"{f:<12}{'draft':>9}{dl:>+10.4f}{da:>+10.4f}")
        for lab, D, key in (('逐token', A, 'tok'), ('逐字段', B, 'fld')):
            c = D['cur'][(f, 'corr')]
            l, a = float(c[-1]), float(c[-1] - c[0])
            win = (l < dl) and (a < da)
            n[key] += win
            print(f"{'':<12}{lab:>9}{l:>+10.4f}{a:>+10.4f}   "
                  + ("**在目标区**" if win else
                     ("水平赢" if l < dl else ("累积赢" if a < da else "都输"))))
    print(f"\nG 目标区字段数: 逐token {n['tok']}/5  逐字段 {n['fld']}/5"
          f"  -> {'通过' if n['fld'] >= n['tok'] else '未过'}")
