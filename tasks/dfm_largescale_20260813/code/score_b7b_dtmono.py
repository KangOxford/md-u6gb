#!/usr/bin/env python3
"""B7b 生成时间戳单调率：dt>0 的相邻消息比例，四格 × {real, draft, corr}。

真实 LOB 有同纳秒批量事件，real 的 dt>0 只有 ~72%——那是市场结构不是病理，
所以读法是 |corr − real| 的偏差，不是「越高越好」。
用法：SCORE_MONTH=2026-01 python code/score_b7b_dtmono.py
"""
import os

import numpy as np

R = '/lus/lfs1aip2/projects/public/u6gb/tasks/dfm_largescale_20260813'
MO = os.environ.get('SCORE_MONTH', '2026-01')
PAIRS = [('s42', 'b7bON_s42', 'b7bOFF_s42'),
         ('s43', 'b7bON_s43', 'b7bOFF_s43')]


def mono(path, key):
    z = np.load(path, allow_pickle=True)
    cols = [str(v) for v in z['msg_cols']]
    t = z[key][:, :, cols.index('time')]
    dt = np.diff(t, axis=1)
    return (dt > 0).sum(), dt.size


def main():
    for seed, on, off in PAIRS:
        tks = sorted(t for t in open(f'{R}/logs/g2x_tk60.txt').read().split()
                     if all(os.path.exists(
                         f'{R}/rollouts_anc/{a}_{t}_{MO}_learned.npz')
                         for a in (on, off)))
        print(f'=== {MO} seed {seed}  共有 ticker {len(tks)}')
        for lab, tag, key in (('real', on, 'real_msgs'),
                              ('draft', on, 'draft_msgs'),
                              ('corr_off', off, 'corr_msgs'),
                              ('corr_on', on, 'corr_msgs')):
            num = den = 0
            for t in tks:
                a, b = mono(f'{R}/rollouts_anc/{tag}_{t}_{MO}_learned.npz', key)
                num += a
                den += b
            print(f'  {lab:<9} dt>0 = {num/den*100:.2f}%')


if __name__ == '__main__':
    main()
