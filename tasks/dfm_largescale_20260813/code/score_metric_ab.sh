#!/bin/bash
# 度量 A/B 的打分。判据在 `run_metric_ab.sh` 里跑之前就锁死了：
#   值度量必须**特异地**消掉 `size` 的水平异常，而其余四个字段不显著变差。
#   五个字段一起动 => 不是这个机制。
set -u
cd "$(dirname "$0")/.."
python3 - <<'PY'
import glob, sys
import numpy as np
sys.path.insert(0, '/lus/lfs1aip2/projects/public/u6gb/tasks/'
                   'ce_orderflow_20260812T200352Z/A01_does_ce_exist/code')
from a01_ce_existence import (FIELDS, action_fields, build_counts, codes,
                              make_edges, D_curve)
R = 'rollouts_metric'

def curves(tag, month='2026-01'):
    fs = sorted(glob.glob(f'{R}/met_{tag}_*_{month}_learned.npz'))
    acc = {f: {'draft': [], 'corr': []} for f in FIELDS}
    tks = []
    for p in fs:
        try:
            z = np.load(p, allow_pickle=True); F = action_fields(z, 100)
        except Exception:
            continue
        N, Rr = z['real_msgs'].shape[0], z['real_msgs'].shape[1]
        if N < 4 or Rr < 160:
            continue
        tks.append(p.split('/')[-1].split('_')[2])
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
                    acc[nm][arm].append(D_curve(CT[0], CT[1], CG[0], CG[1], K) - fl)
            except Exception:
                pass
    return tks, {f: {a: np.nanmean(np.vstack(v), 0)
                     for a, v in d.items() if v} for f, d in acc.items()}

def illegal(tag, month='2026-01'):
    tot = bad = 0
    for p in sorted(glob.glob(f'{R}/met_{tag}_*_{month}_learned.npz')):
        z = np.load(p, allow_pickle=True); cols = [str(v) for v in z['msg_cols']]
        m = z['corr_msgs']
        ev = m[:, :, cols.index('event_type')]; dr = m[:, :, cols.index('direction')]
        sz = m[:, :, cols.index('size')]; px = m[:, :, cols.index('price')]
        b = (~np.isin(ev, [1,2,3,4])) | (~np.isin(dr, [-1,1])) | (~(sz>=1)) | (~(px>0))
        bad += int(b.sum()); tot += ev.size
    return 100.0 * bad / max(tot, 1)

tk_v, cv = curves('value'); tk_f, cf = curves('field')
common = sorted(set(tk_v) & set(tk_f))
print(f"value {len(tk_v)} ticker · field {len(tk_f)} ticker · 共同 {len(common)}")
print(f"非法率  value {illegal('value'):.3f}%   field {illegal('field'):.3f}%\n")
print(f"{'field':<12}{'draft':>9}{'field度量':>11}{'value度量':>11}"
      f"{'value-field':>13}{'判定':>8}")
print('-'*64)
verdict = {}
for f in FIELDS:
    if f not in cv or 'corr' not in cv[f] or f not in cf or 'corr' not in cf[f]:
        print(f"{f:<12}  (缺产物)"); continue
    d = float(cf[f]['draft'][-1]); a = float(cf[f]['corr'][-1]); b = float(cv[f]['corr'][-1])
    dd = b - a
    verdict[f] = dd
    mark = '改善' if dd < -0.02 else ('恶化' if dd > 0.02 else '持平')
    print(f"{f:<12}{d:>+9.4f}{a:>+11.4f}{b:>+11.4f}{dd:>+13.4f}{mark:>8}")
print('-'*64)
print("\n预注册判据：值度量必须**特异地**改善 `size`，其余四个字段不显著变差")
if 'size' in verdict:
    others = [k for k in verdict if k != 'size']
    sz_better = verdict['size'] < -0.02
    others_ok = all(verdict[k] <= 0.02 for k in others)
    all_move = all(abs(verdict[k]) > 0.02 for k in verdict)
    print(f"  size 改善      {'是' if sz_better else '否'}  ({verdict['size']:+.4f})")
    print(f"  其余不变差     {'是' if others_ok else '否'}  "
          + " ".join(f"{k}={verdict[k]:+.3f}" for k in others))
    print(f"  五个一起动     {'是（机制被证伪）' if all_move else '否'}")
    print("\n裁决: " + ("**机制确认**" if (sz_better and others_ok) else
                       ("**机制被证伪**" if all_move else "**部分/不确定**")))
PY
