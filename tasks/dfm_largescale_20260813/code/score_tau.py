#!/usr/bin/env python3
"""keep-tau 扫描的评分：水平与斜率**同时**报（PR2）。

本文件由 `score_tau.py` 复制而来，只改了目录、tag 表与判据文字。

tau 那条假设已被它自己的数据否证：`event_type` 的水平代价随 tau **变小而
单调上升**（+0.094 -> +0.754，8 倍），方向与预测相反。根因是 `predict_x1` 对
**每个**可编辑位置都重新抽样，所以修正臂是 DFM 模型自己的一次完整生成，而不是
draft 的扰动；缩残差只是把那个模型改坏，步间加噪却照旧。

keep-tau 是真的旋钮：只在模型给 draft token 的概率**低于** tau 时才覆盖它。
    tau 小 -> 保留多 -> 接近 draft；tau=0 -> 关闭分支 = 纯修正器。

判据（跑之前锁死，与 E7 同一条）：
  存在某个 tau，使 **>=3/5 字段的水平低于 draft**，且这些字段的**累积涨幅
  <= draft 涨幅的一半**。——「水平更低且不再累积」才算真正降低了复合误差。

同时必须报：tau=0 这一臂与已有 fx488 learned 臂在同一批 ticker 上应当一致，
否则这次运行本身有问题（可重复性检查）。
"""
import glob, sys, os
import numpy as np
sys.path.insert(0, '/lus/lfs1aip2/projects/public/u6gb/tasks/ce_orderflow_20260812T200352Z/A01_does_ce_exist/code')
from a01_ce_existence import (FIELDS, action_fields, build_counts, codes,
                              make_edges, D_curve)

R = '/lus/lfs1aip2/projects/public/u6gb/tasks/dfm_largescale_20260813/rollouts_tau'

def curves(prefix, month='2026-01'):
    fs = sorted(glob.glob(f'{R}/{prefix}*_{month}_learned.npz'))
    acc = {f: {'draft': [], 'corr': []} for f in FIELDS}; grid = None
    for p in fs:
        try: z = np.load(p, allow_pickle=True); F = action_fields(z, 100)
        except Exception: continue
        N, Rr = z['real_msgs'].shape[0], z['real_msgs'].shape[1]
        if N < 4 or Rr < 160: continue
        g = np.arange(40, Rr - 40, 20); grid = g
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
            except Exception: pass
    n = len(acc[FIELDS[0]]['corr'])
    return grid, n, {f: {a: np.nanmean(np.vstack(v), 0)
                         for a, v in d.items() if v} for f, d in acc.items()}

def illegal(prefix, month='2026-01'):
    fs = sorted(glob.glob(f'{R}/{prefix}*_{month}_learned.npz')); tot = bad = 0
    for p in fs:
        z = np.load(p, allow_pickle=True); cols = [str(v) for v in z['msg_cols']]
        m = z['corr_msgs']
        ev = m[:, :, cols.index('event_type')]; dr = m[:, :, cols.index('direction')]
        sz = m[:, :, cols.index('size')]; px = m[:, :, cols.index('price')]
        b = (~np.isin(ev, [1, 2, 3, 4])) | (~np.isin(dr, [-1, 1])) | (~(sz >= 1)) | (~(px > 0))
        bad += int(b.sum()); tot += ev.size
    return 100.0 * bad / max(tot, 1)

TAUS = [('t0', 0.0), ('t001', 0.01), ('t01', 0.1)]
print(f"{'tau':>8} {'n_tk':>5} {'illegal%':>9}  " + "  ".join(f"{f[:9]:>9}" for f in FIELDS))
base = None
rows = {}
for tag, a in TAUS:
    g, n, c = curves(f'tau_{tag}_')
    if n == 0: print(f"{a:>6.2f} {0:>5}  (无产物)"); continue
    if base is None:
        base = {f: c[f]['draft'] for f in FIELDS}
        print(f"{'draft':>6} {n:>5} {'—':>9}  " +
              "  ".join(f"{base[f][-1]:>+9.4f}" for f in FIELDS))
    il = illegal(f'tau_{tag}_')
    rows[a] = c
    print(f"{a:>6.2f} {n:>5} {il:>8.3f}%  " +
          "  ".join(f"{c[f]['corr'][-1]:>+9.4f}" for f in FIELDS))

print(f"\n{'tau':>8}  优于 draft 的字段（水平 m=440）")
for a, c in rows.items():
    win = [f for f in FIELDS if c[f]['corr'][-1] < base[f][-1]]
    print(f"{a:>6.2f}  {len(win)}/5  {' '.join(win) if win else '(无)'}")

print(f"\n{'tau':>8}  累积涨幅（corr / draft，越小越好）")
for a, c in rows.items():
    s = []
    for f in FIELDS:
        dr = base[f][-1] - base[f][0]; cr = c[f]['corr'][-1] - c[f]['corr'][0]
        s.append(f"{f[:9]}={cr/dr:+.2f}" if abs(dr) > 1e-9 else f"{f[:9]}=—")
    print(f"{a:>6.2f}  " + "  ".join(s))
