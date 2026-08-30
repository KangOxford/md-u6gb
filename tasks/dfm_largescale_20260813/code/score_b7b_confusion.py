#!/usr/bin/env python3
"""Criterion A：生成侧 delta_t_ns 高位入低位混淆（置换 z），B7b 开 vs 关。

原判据（place_probe.py，teacher forcing）：模型在位 j 出错时，吐的值落在位 j2
真值 5% 半径内的比例，对 200 次置换零假设的 z。实测 1→0（千位吐百万位值）
z=+33.32、2→1（个位吐千位值）z=+6.15，其余四对 |z|<1。

生成侧移植：自由生成没有逐位真值可条件化，改测**消息内跨位接近度**——
生成的 delta_t_ns 里，位 a 与位 b 的数字值在同一消息内 |v_a−v_b|≤tol 的比例，
零假设置换消息配对（保边缘分布、破配对结构）。真实消息（real）是天然参照：
真实数据里高低位的物理相关（如小 dt 的共零）同样出现在 real 的 z 里，
corr 相对 real/draft 的**超额**才是混淆签名。

数位重构：dt = diff(time) 秒；ns = round((dt−floor(dt))·1e9)；
(hi, mid, lo) = (ns//1e6, ns//1e3 %1000, ns %1000)。float64 在当天秒数
（≤~6e4 s）下绝对精度 ~1e-11 s ≈ 13 ns，远小于 tol≈50，不影响判定。

用法：SCORE_MONTH=2026-01 python code/score_b7b_confusion.py
输出同时写 JSON（figs/confusion_<月>.json）供汇总。
"""
import json
import os

import numpy as np

R = '/lus/lfs1aip2/projects/public/u6gb/tasks/dfm_largescale_20260813'
MO = os.environ.get('SCORE_MONTH', '2026-01')
NPERM = 200
TOL = 0.05 * 999.0          # 与原版一致：位值域 span 的 5%
PAIRS = [('s42', 'b7bON_s42', 'b7bOFF_s42'),
         ('s43', 'b7bON_s43', 'b7bOFF_s43')]
DIGIT_PAIRS = [('hi', 'mid'), ('mid', 'lo'), ('hi', 'lo')]   # (hi,mid)≙原1→0，(mid,lo)≙原2→1


def digits_of(path, arm_key):
    """一个 npz 一个组的 (hi, mid, lo) 有效消息数位，展平成一维。"""
    z = np.load(path, allow_pickle=True)
    cols = [str(v) for v in z['msg_cols']]
    m = z[arm_key]
    ev = m[:, :, cols.index('event_type')]
    dr = m[:, :, cols.index('direction')]
    sz = m[:, :, cols.index('size')]
    px = m[:, :, cols.index('price')]
    t = m[:, :, cols.index('time')]
    dt = np.full_like(t, np.nan)
    dt[:, 1:] = np.diff(t, axis=1)
    ok = (np.isin(ev, [1, 2, 3, 4]) & np.isin(dr, [-1, 1]) & (sz >= 1)
          & (px > 0) & np.isfinite(dt) & (dt > 0))
    d = dt[ok]
    ns = np.rint((d - np.floor(d)) * 1e9).astype(np.int64)
    ns = np.clip(ns, 0, 999_999_999)      # rint 可能进到 1e9，收回值域
    return {'hi': ns // 1_000_000,
            'mid': (ns // 1_000) % 1_000,
            'lo': ns % 1_000}


def perm_z(va, vb, rng):
    """|va−vb|≤TOL 的比例对置换零假设的 z。"""
    obs = float((np.abs(va - vb) <= TOL).mean())
    nulls = np.empty(NPERM)
    for i in range(NPERM):
        nulls[i] = (np.abs(va - rng.permutation(vb)) <= TOL).mean()
    mu, sd = float(nulls.mean()), float(nulls.std() + 1e-12)
    return obs, mu, (obs - mu) / sd


def main():
    out = {}
    for seed, on, off in PAIRS:
        tks = sorted(t for t in open(f'{R}/logs/g2x_tk60.txt').read().split()
                     if all(os.path.exists(
                         f'{R}/rollouts_anc/{a}_{t}_{MO}_learned.npz')
                         for a in (on, off)))
        if len(tks) < 8:
            print(f'{seed}: 共有 ticker 只有 {len(tks)} 个，跳过')
            continue
        print(f'=== {MO}  seed {seed} ===  共有 ticker {len(tks)}')
        # arm → 位名 → 合并全部 ticker 的一维数组
        D = {}
        for lab, tag, arm_key in (('real', on, 'real_msgs'),
                                  ('draft', on, 'draft_msgs'),
                                  ('corr_off', off, 'corr_msgs'),
                                  ('corr_on', on, 'corr_msgs')):
            acc = {'hi': [], 'mid': [], 'lo': []}
            for t in tks:
                d = digits_of(f'{R}/rollouts_anc/{tag}_{t}_{MO}_learned.npz',
                              arm_key)
                for k in acc:
                    acc[k].append(d[k])
            D[lab] = {k: np.concatenate(v).astype(float) for k, v in acc.items()}
            n = len(D[lab]['hi'])
            print(f'  {lab:<9} n={n}')
        if os.environ.get('EQN', '0') == '1':
            # 等 n 子采样：z 随 √n 膨胀，四组合法消息数不同（on 的 malformed
            # 率更低 → n 多 ~46%）会把「更多合法消息」误记成「更强相关」。
            nmin = min(len(dd['hi']) for dd in D.values())
            rng0 = np.random.default_rng(20260830)
            for lab, dd in D.items():
                idx = rng0.choice(len(dd['hi']), nmin, replace=False)
                D[lab] = {k: v[idx] for k, v in dd.items()}
            print(f'  [eqn] 全组子采样到 n={nmin}')
        rows = []
        hdr = (f"{'pair':<10}{'variant':<10}" +
               ''.join(f'{lab:>12}' for lab in D) + f"{'z_on−z_off':>12}")
        print(hdr)
        for a, b in DIGIT_PAIRS:
            for variant in ('all', 'nz'):        # nz = 排除双零共现
                zs = {}
                for lab, dd in D.items():
                    va, vb = dd[a], dd[b]
                    if variant == 'nz':
                        m = (va != 0) | (vb != 0)
                        va, vb = va[m], vb[m]
                    rng = np.random.default_rng(20260830)
                    obs, mu, zv = perm_z(va, vb, rng)
                    zs[lab] = dict(obs=obs, perm=mu, z=zv, n=int(len(va)))
                dz = zs['corr_on']['z'] - zs['corr_off']['z']
                rows.append(dict(pair=f'{a}->{b}', variant=variant,
                                 month=MO, seed=seed, cells=zs, dz=dz))
                print(f"{a+'->'+b:<10}{variant:<10}" +
                      ''.join(f"{zs[lab]['z']:>+12.2f}" for lab in D) +
                      f"{dz:>+12.2f}")
        out[seed] = rows
        print()
    os.makedirs(f'{R}/figs', exist_ok=True)
    sfx = '_eqn' if os.environ.get('EQN', '0') == '1' else ''
    with open(f'{R}/figs/confusion_{MO}{sfx}.json', 'w') as f:
        json.dump(out, f, indent=1)
    print(f"wrote {R}/figs/confusion_{MO}{sfx}.json")


if __name__ == '__main__':
    main()
