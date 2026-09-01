#!/usr/bin/env python3
"""tau/keepfield 扫描的收尾评分：每个配置的 5 字段 level/slope vs 各自 draft。

锁死标准（run_tau_sweep.sh 头部,2026-08-15 定）：
  存在某个 tau，使 >=3/5 字段水平低于 draft 且这些字段累积 <= 0。

口径注意：这批 rollout 是 --n-seq 8 的旧口径（每月约 1 个交易日），
比较只在本表内部做（同口径同月同 ticker 池），数值不与 n_seq=32 的表混。
两套 state 分开看：tau_*（lg488b 2a@3500）与 tau2_*/kf/kt（s14000 阶梯）。
"""
import os
import sys

import numpy as np

sys.path.insert(0, '/lus/lfs1aip2/projects/public/u6gb/tasks/'
                   'ce_orderflow_20260812T200352Z/A01_does_ce_exist/code')
from a01_ce_existence import (FIELDS, action_fields, build_counts,      # noqa: E402
                              codes, make_edges, D_curve)

R = '/lus/lfs1aip2/projects/public/u6gb/tasks/dfm_largescale_20260813'
MO = os.environ.get('SCORE_MONTH', '2026-01')
CFGS = [('rollouts_anc', 'p0bidir_s42'), ('rollouts_anc', 'p0bidir_s43'),
        ('rollouts_anc', 'r2d1_s42'), ('rollouts_anc', 'r2d1_s43'),
        ('rollouts_anc', 'r2d0_s42'), ('rollouts_anc', 'r2d0_s43'),
        ('rollouts_tau', 'tau_t0'), ('rollouts_tau', 'tau_t001'),
        ('rollouts_tau', 'tau_t01'),
        ('rollouts_tau2', 'tau2_t01'), ('rollouts_tau2', 'tau2_t03'),
        ('rollouts_kf', 'kt14k_t001'), ('rollouts_kf', 'kt14k_t01'),
        ('rollouts_kf', 'kf14k_t001'), ('rollouts_kf', 'kf14k_t01')]


def curves(path):
    z = np.load(path, allow_pickle=True)
    F = action_fields(z, 100)
    N, Rr = z['real_msgs'].shape[0], z['real_msgs'].shape[1]
    if N < 4 or Rr < 160:
        return None
    g = np.arange(40, Rr - 40, 20)
    perm = np.random.default_rng(20260813).permutation(N)
    G = [perm[:N // 2], perm[N // 2:]]
    out = {}
    for nm in FIELDS:
        try:
            e = make_edges(F['true'][nm], F['true']['_ok'], nm, 40)
            cT, K = codes(F['true'][nm], F['true']['_ok'], nm, e)
            CT = build_counts(cT, G, g, 40, Rr, K)
            fl = D_curve(CT[0], CT[1], CT[0], CT[1], K)
            for arm in ('draft', 'corr'):
                cG, _ = codes(F[arm][nm], F[arm]['_ok'], nm, e)
                CG = build_counts(cG, G, g, 40, Rr, K)
                out[(nm, arm)] = D_curve(CT[0], CT[1], CG[0], CG[1], K) - fl
        except Exception:
            pass
    return out


def main():
    import glob
    for d, tag in CFGS:
        per = {}
        n_tk = 0
        for p in sorted(glob.glob(f'{R}/{d}/{tag}_*_{MO}_learned.npz')):
            c = curves(p)
            if c:
                n_tk += 1
                for k, v in c.items():
                    per.setdefault(k, []).append(v)
        if not per:
            print(f'{tag}: 无产物')
            continue
        acc = {k: np.nanmean(np.vstack(v), 0) for k, v in per.items()}
        good = []
        rows = []
        for f in FIELDS:
            if (f, 'corr') not in acc:
                continue
            c, dr = acc[(f, 'corr')], acc[(f, 'draft')]
            cl, dl = float(c[-1]), float(dr[-1])
            ca = float(c[-1] - c[0])
            da = float(dr[-1] - dr[0])
            win = (cl < dl) and (ca <= 0)
            good.append(win)
            rows.append(f"    {f:<12} level {cl:>+8.4f} (draft {dl:>+8.4f})"
                        f"  slope {ca:>+8.4f} (draft {da:>+8.4f})"
                        f"  {'✓水平低且不爬' if win else ''}")
        n_win = sum(good)
        verdict = 'PASS' if n_win >= 3 else 'fail'
        print(f'== {tag:<12} ({n_tk} tk)  {n_win}/5 字段[水平<draft 且 累积<=0]'
              f'  -> {verdict}')
        for r in rows:
            print(r)
        print()


if __name__ == '__main__':
    main()
