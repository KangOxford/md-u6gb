#!/usr/bin/env python3
import json, glob, itertools, pathlib, collections
import numpy as np
P = pathlib.Path('/lus/lfs1aip2/projects/public/u6gb/nb_build_pr22')
TK = "AMD AMZN GOOG INTC JPM META MSFT NFLX".split()
rows=[]
for ln in open(P/'sweep_results.jsonl'):
    ln=ln.strip()
    if ln:
        try: rows.append(json.loads(ln))
        except Exception: pass
print('jsonl rows', len(rows))
for f in (P/'sweep').glob('s*.json'):
    d=json.loads(f.read_text())
    rows.append({'arm':'multi4','step':str(d['step']),'ticker':d['ticker'],'seed':d['seed'],
                 'sd_ratio':d['sd_ratio'],'qL1':d['qL1']})
print('total rows (jsonl + sweep dir)', len(rows))
# sweep_results.jsonl holds TWO record shapes and they are different estimands:
#   per-seed rows  {arm,step,ticker,seed,sd_ratio,qL1}          -- one member
#   pooled rows    {arm,step,ticker,node,crps,sd_ratio,qL1}     -- K>=2, CRPS-scored, NO seed
# Keying both by (ticker,seed) raised KeyError: 'seed'.  The crash was correct; the file
# really does contain two things.  Dropping the seedless rows would silently discard
# exactly the 48 rows that carry `crps`, so they are partitioned and both counts printed.
per_seed = [r for r in rows if 'seed' in r]
pooled   = [r for r in rows if 'seed' not in r]
print(f'  per-seed rows {len(per_seed)}   pooled (K>=2, no seed) rows {len(pooled)}'
      f'   -- kept separate, NOT merged')
if pooled:
    ps=collections.Counter((r['arm'], str(r['step'])) for r in pooled)
    print('  pooled cells by (arm,step):', dict(sorted(ps.items())))
cells=collections.defaultdict(dict)
for r in per_seed:
    cells[(r['arm'],int(r['step']))][(r['ticker'],r['seed'])]=(r['sd_ratio'],r['qL1'])
NL=json.loads((P/'fix_nulls.json').read_text())
R3=np.array([NL['arm_levels']['r3_september_repro']['ticker_pooled'][t] for t in TK])
NSD=NL['null_ladder']['rung3_disjoint_seeds_same_epoch']['exact_randomisation']['ticker_level_mean']['sd_signed_symmetrised']
def sf(d):
    n=len(d); o=abs(d.mean())
    return sum(1 for s in itertools.product([-1,1],repeat=n) if abs(np.mean(np.array(s)*d))>=o-1e-18)/2**n
CV=json.loads((P/'sweep_curve.json').read_text())
pub={(a,r['step']):r for a in CV['arms'] for r in CV['arms'][a]}
print()
print(f"{'arm':7} {'step':>5} {'seeds':>6} {'nseed':>5} {'R':>8} {'|R-1|':>7} {'qL1':>7} {'8/8':>4} {'p':>7}   published R")
for arm in ('multi4','unifw','multi3'):
    for st in sorted(s for a,s in cells if a==arm):
        c=cells[(arm,st)]
        seeds=sorted({s for _,s in c})
        # the published rule: use the maximal seed set for which all 8 tickers are complete
        best=None
        for k in range(len(seeds),0,-1):
            for sub in itertools.combinations(seeds,k):
                if all((t,s) in c for t in TK for s in sub):
                    best=sub; break
            if best: break
        if not best: 
            print(f"{arm:7} {st:5d}  incomplete, seeds present {seeds}")
            continue
        v=np.array([np.mean([c[(t,s)][0] for s in best]) for t in TK])
        q=np.array([np.mean([c[(t,s)][1] for s in best]) for t in TK])
        d=v-R3
        pr=pub.get((arm,st))
        print(f"{arm:7} {st:5d} {str(list(best))[-14:]:>6} {len(best):5d} {v.mean():8.4f} {abs(v.mean()-1):7.4f} {q.mean():7.4f} {int((d>0).sum()):3d}/8 {sf(d):7.4f}   "
              f"{pr['R']:.4f} (n={pr['n_seeds']})" if pr else
              f"{arm:7} {st:5d} {str(list(best))[-14:]:>6} {len(best):5d} {v.mean():8.4f} {abs(v.mean()-1):7.4f} {q.mean():7.4f} {int((d>0).sum()):3d}/8 {sf(d):7.4f}   -- not published --")
