#!/usr/bin/env python3
"""Extract every number the compound-error notebook plots into one small JSON.

The rollout .npz files live on Lustre and total tens of GB; the notebook must be
readable on GitHub without them.  So all reduction happens here, once, and the
notebook only ever loads the resulting cache (a few hundred kB).

Emits, per (cell, field):
  - the full D(depth) curve, corrector and draft arms, averaged over tickers
  - the depth grid those curves live on
Plus the per-cell ticker count, so coverage differences stay visible.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, '/lus/lfs1aip2/projects/public/u6gb/tasks/'
                   'ce_orderflow_20260812T200352Z/A01_does_ce_exist/code')
from a01_ce_existence import (FIELDS, action_fields, build_counts,      # noqa: E402
                              codes, make_edges, D_curve)

R = '/lus/lfs1aip2/projects/public/u6gb/tasks/dfm_largescale_20260813'
MO = os.environ.get('SCORE_MONTH', '2026-01')
CELLS = [('p0bidir_s42', 'P=0 (untrained bidirectional)', 42),
         ('p0bidir_s43', 'P=0 (untrained bidirectional)', 43),
         ('r2d0_s42', 'R2 no offset table', 42),
         ('r2d0_s43', 'R2 no offset table', 43),
         ('r2d1_s42', 'R2 ungated', 42),
         ('r2d1_s43', 'R2 ungated', 43),
         ('r2g1_s42', 'R2 gated residual', 42),
         ('r2g1_s43', 'R2 gated residual', 43)]


def curves(path):
    z = np.load(path, allow_pickle=True)
    F = action_fields(z, 100)
    N, Rr = z['real_msgs'].shape[0], z['real_msgs'].shape[1]
    if N < 4 or Rr < 160:
        return None, None
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
                out[f'{nm}|{arm}'] = (D_curve(CT[0], CT[1], CG[0], CG[1], K)
                                      - fl).tolist()
        except Exception:
            pass
    return out, g.tolist()


def main():
    cache = {'month': MO, 'fields': list(FIELDS), 'cells': {}}
    for tag, label, seed in CELLS:
        per, grid = {}, None
        n_tk = 0
        for p in sorted(glob.glob(f'{R}/rollouts_anc/{tag}_*_{MO}_learned.npz')):
            c, g = curves(p)
            if c:
                n_tk += 1
                grid = g
                for k, v in c.items():
                    per.setdefault(k, []).append(v)
        if not per:
            print(f'  {tag}: no rollouts for {MO}', file=sys.stderr)
            continue
        cache['cells'][tag] = {
            'label': label, 'seed': seed, 'n_tickers': n_tk, 'grid': grid,
            'curves': {k: np.nanmean(np.vstack(v), 0).tolist()
                       for k, v in per.items()}}
        print(f'  {tag}: {n_tk} tickers', file=sys.stderr)
    out = f'{R}/figs/notebook_data_{MO}.json'
    os.makedirs(f'{R}/figs', exist_ok=True)
    with open(out, 'w') as f:
        json.dump(cache, f)
    print(f'wrote {out} ({os.path.getsize(out)/1024:.0f} kB)', file=sys.stderr)


if __name__ == '__main__':
    main()
