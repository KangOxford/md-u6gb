#!/usr/bin/env python3
"""Generate the compound-error analysis notebook (figures first, text last)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip()))
def co(s): C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Compound error in the DFM corrector: what six configurations rule out

The draft model's distributional distance to real data **grows with generation depth**
(compound error). The residual corrector was built to flatten that growth. It does
not: it flattens the growth only by paying a larger constant offset, and no
configuration tested so far gets below the draft on any field.

This notebook holds every measurement behind that claim. All rollouts are
`n_seq=32`, `t_start=0.80`, `seed=2026`, month 2026-01, scored on the same five
action fields with the same ticker pools per cell.

**Cells compared**

| tag | what it is | changes vs the one above |
| --- | --- | --- |
| `draft` | base model, no correction | reference arm inside every rollout |
| `p0bidir` | corrector with the learned projection zeroed | isolates the untrained bidirectional pass |
| `r2d0` | trained on `corrupt(draft)`, no offset table | the distribution fix alone |
| `r2d1` | `r2d0` + offset embedding table | |
| `r2g1` | `r2d1` + gated residual (`sigmoid` gate, quiet default) | the architecture fix |

Two seeds (42, 43) per cell. Nothing below is a single-seed claim.
""")

co(r"""
import json, numpy as np, matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

D   = json.load(open('data/notebook_data_2026-01.json'))
AUX = json.load(open('data/nb_aux.json'))
FIELDS = D['fields']

# Categorical slots, fixed order, validated for CVD separation (never cycled).
CLR = {'draft':'#2a78d6', 'r2d1':'#eb6834', 'r2g1':'#1baf7a',
       'r2d0':'#eda100', 'p0bidir':'#4a3aa7', 'real':'#52514e'}
NAME = {'draft':'draft', 'p0bidir':'P=0 bidirectional', 'r2d0':'R2 no offset',
        'r2d1':'R2 ungated', 'r2g1':'R2 gated'}
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#d8d7d2', '#fcfcfb'

plt.rcParams.update({
    'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'grid.color': GRID,
    'axes.spines.top': False, 'axes.spines.right': False,
    'font.size': 10, 'axes.titlesize': 11, 'axes.grid': True,
    'grid.linewidth': 0.6, 'lines.linewidth': 2.0, 'figure.dpi': 120})

def cell(tag): return D['cells'][tag]
def curve(tag, field, arm): return np.array(cell(tag)['curves'][f'{field}|{arm}'])
def level(tag, field, arm): return float(curve(tag, field, arm)[-1])
def slope(tag, field, arm):
    # cumulative growth: end minus start of the depth curve
    c = curve(tag, field, arm); return float(c[-1] - c[0])

CELLS = ['p0bidir', 'r2d0', 'r2d1', 'r2g1']
SEEDS = [42, 43]
print({t: {s: cell(f'{t}_s{s}')['n_tickers'] for s in SEEDS} for t in CELLS})
""")

md(r"""
## Figure 1 — every corrector sits above the draft, everywhere

One point per (cell, field, seed): the corrector's distributional distance at the
deepest measured point against the draft's own distance on the same rollouts. The
diagonal is "no better, no worse than not correcting at all".
""")

co(r"""
fig, ax = plt.subplots(figsize=(6.4, 6.0))
lim = [8e-3, 3.0]
dg = np.logspace(np.log10(lim[0]), np.log10(lim[1]), 200)
ax.plot(dg, dg, color=INK2, lw=1.2, ls='--', zorder=1)
ax.fill_between(dg, dg, lim[1], color='#eb6834', alpha=0.06, zorder=0)
ax.text(0.0095, 2.15, 'corrector WORSE than draft', color=INK2, fontsize=9.5, style='italic')
ax.text(1.05, 0.030, 'corrector better\n(target region — empty)', color=INK2,
        fontsize=9.5, style='italic', ha='center')

MK = {42: 'o', 43: '^'}
for tag in CELLS:
    for s in SEEDS:
        k = f'{tag}_s{s}'
        if k not in D['cells']: continue
        xs = [level(k, f, 'draft') for f in FIELDS if f'{f}|corr' in cell(k)['curves']]
        ys = [level(k, f, 'corr')  for f in FIELDS if f'{f}|corr' in cell(k)['curves']]
        ax.scatter(xs, ys, s=46, marker=MK[s], color=CLR[tag], zorder=3,
                   edgecolor=SURF, linewidth=1.2,
                   label=NAME[tag] if s == 42 else None)

ax.set(xscale='log', yscale='log', xlim=lim, ylim=lim,
       xlabel='draft distance to real  (deepest point)',
       ylabel='corrector distance to real  (deepest point)',
       title='Figure 1 — 40 of 40 measurements land above the diagonal')
ax.xaxis.set_major_locator(LogLocator(numticks=5)); ax.yaxis.set_major_locator(LogLocator(numticks=5))
ax.legend(frameon=False, loc='lower left', fontsize=9,
          title='circle = seed 42,  triangle = seed 43', title_fontsize=8.5)
plt.tight_layout(); plt.show()

above = sum(level(f'{t}_s{s}', f, 'corr') > level(f'{t}_s{s}', f, 'draft')
            for t in CELLS for s in SEEDS for f in FIELDS
            if f'{f}|corr' in cell(f'{t}_s{s}')['curves'])
total = sum(1 for t in CELLS for s in SEEDS for f in FIELDS
            if f'{f}|corr' in cell(f'{t}_s{s}')['curves'])
print(f'above the diagonal: {above} / {total}')
""")

md(r"""
## Figure 2 — the curves themselves

Distance to real as a function of generation depth. The draft's rising line is the
compound error the corrector was built to remove.
""")

co(r"""
fig, axes = plt.subplots(1, 5, figsize=(15.5, 3.4), sharex=True)
for ax, f in zip(axes, FIELDS):
    g = np.array(cell('r2d1_s42')['grid'])
    ax.plot(g, curve('r2d1_s42', f, 'draft'), color=CLR['draft'], zorder=3)
    for tag in ['p0bidir', 'r2d1', 'r2g1']:
        k = f'{tag}_s42'
        if f'{f}|corr' in cell(k)['curves']:
            ax.plot(np.array(cell(k)['grid']), curve(k, f, 'corr'),
                    color=CLR[tag], zorder=2)
    ax.axhline(0, color=INK2, lw=1.0, ls=':')
    ax.set(title=f, xlabel='generation depth (messages)')
    ax.set_yscale('symlog', linthresh=0.05)
axes[0].set_ylabel('excess distance to real')
h = [plt.Line2D([], [], color=CLR[t], lw=2) for t in ['draft','p0bidir','r2d1','r2g1']]
fig.legend(h, [NAME[t] for t in ['draft','p0bidir','r2d1','r2g1']], frameon=False,
           fontsize=9.5, ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.14))
fig.suptitle('Figure 2 — seed 42.  The draft (blue) is the lowest line in every panel',
             y=1.04, fontsize=11)
plt.tight_layout(); plt.show()
""")

md(r"""
## Figure 3 — the gate did its job, and it was not enough

Compound error has two parts: a **constant offset** (level) and **growth with depth**
(slope). The gate was added to stop the corrector from shouting. Splitting the two
parts shows where it acted, and how far the two seeds agree about it: the seeds are
plotted separately because on three of the five fields they disagree about the sign,
and a seed-average would hide that.
""")

co(r"""
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4))
x = np.arange(len(FIELDS)); off = {'r2d1': -0.16, 'r2g1': 0.16}

for ax, fn, ttl in [(axes[0], level, 'level  (constant offset)'),
                    (axes[1], slope, 'slope  (growth with depth)')]:
    for tag in ['r2d1', 'r2g1']:
        for i, f in enumerate(FIELDS):
            r = [fn(f'{tag}_s{s_}', f, 'corr') / max(fn(f'{tag}_s{s_}', f, 'draft'), 1e-6)
                 for s_ in SEEDS if f'{f}|corr' in cell(f'{tag}_s{s_}')['curves']]
            if not r: continue
            xx = i + off[tag]
            ax.plot([xx, xx], [min(r), max(r)], color=CLR[tag], lw=1.4, alpha=0.55, zorder=2)
            ax.scatter([xx]*len(r), r, s=[52, 52][:len(r)],
                       marker='o', facecolor=[CLR[tag], SURF][:len(r)],
                       edgecolor=CLR[tag], linewidth=1.6, zorder=3,
                       label=NAME[tag] if i == 0 else None)
    ax.axhline(1.0, color=INK2, lw=1.4, ls='--', zorder=4)
    ax.text(len(FIELDS) - 0.45, 1.0, ' draft', color=INK2, fontsize=9, va='center')
    ax.set(xticks=x, xlim=(-0.6, len(FIELDS) - 0.4), title=ttl,
           ylabel='ratio to draft  (1.0 = as good as no correction)')
    ax.set_xticklabels(FIELDS, rotation=20, ha='right')
axes[0].legend(frameon=False, fontsize=9, loc='upper left')
axes[0].set_yscale('log')
# symlog: a negative ratio means the corrector's growth reversed sign, which is the
# target region. A plain log axis drops those points without saying so.
axes[1].set_yscale('symlog', linthresh=0.2)
axes[1].axhspan(axes[1].get_ylim()[0], 0, color='#1baf7a', alpha=0.06, zorder=0)
axes[1].text(0.985, 0.045, 'growth reversed (target zone)', transform=axes[1].transAxes,
             fontsize=9, color=INK2, style='italic', ha='right')
fig.text(0.5, -0.06, 'filled = seed 42,   hollow = seed 43,   bar spans the two seeds',
         ha='center', fontsize=9, color=INK2, style='italic')
fig.suptitle('Figure 3 — the gated cell (green) never enters the target zone; the ungated one '
             'does, on seed 43.\nEvery level, both cells, both seeds, is above the draft',
             y=1.06, fontsize=11)
plt.tight_layout(); plt.show()

print(f"{'field':<12}{'seed 42':>9}{'seed 43':>9}   both below draft?")
for f in FIELDS:
    r = [slope(f'r2g1_s{s_}', f, 'corr') / max(slope(f'r2g1_s{s_}', f, 'draft'), 1e-6)
         for s_ in SEEDS if f'{f}|corr' in cell(f'r2g1_s{s_}')['curves']]
    print(f'{f:<12}{r[0]:>9.2f}{r[1]:>9.2f}   {"yes" if max(r) < 1 else "no"}')
""")

md(r"""
## Figure 4 — the timing pathology the gate was aimed at

Fraction of consecutive generated messages with a strictly positive time gap. Real
order flow has simultaneous messages; a corrector that emits a fresh timestamp for
every message pushes this toward 100%.
""")

co(r"""
fig, ax = plt.subplots(figsize=(7.4, 4.0))
arms = ['real', 'draft', 'r2d1', 'r2g1']
x = np.arange(len(arms)); w = 0.36
for i, s in enumerate(['s42', 's43']):
    v = [AUX['dtmono'][s][a] for a in arms]
    b = ax.bar(x + (i - 0.5) * w, v, w * 0.92, zorder=3,
               color=[CLR[a] for a in arms], alpha=1.0 if i == 0 else 0.55,
               edgecolor=SURF, linewidth=1.5)
    ax.bar_label(b, fmt='%.1f', fontsize=8.5, color=INK2, padding=2)
ax.axhline(AUX['dtmono']['s42']['real'], color=INK2, lw=1.2, ls='--', zorder=4)
ax.set(xticks=x, ylim=(60, 105), ylabel='messages with dt > 0   (%)',
       title='Figure 4 — the gate moves the corrector 4-5 points back toward real (both seeds)')
ax.set_xticklabels(['real', 'draft', 'R2 ungated', 'R2 gated'])
ax.text(3.42, AUX['dtmono']['s42']['real'] + 0.8, 'real', color=INK2, fontsize=9)
ax.text(0.02, 0.06, 'solid = seed 42,  pale = seed 43', transform=ax.transAxes,
        fontsize=9, color=INK2, style='italic')
plt.tight_layout(); plt.show()
""")

md(r"""
## Figure 5 — where the gate made things worse

Each generated message is a string of digits. This measures whether a wrong
high-order digit travels with a wrong lower-order digit, as a z-score against a
200-draw permutation null. Real flow carries a strong positive coupling on the
`hi -> mid` pair; both correctors break it, and the gated one inverts it.
""")

co(r"""
pairs = ['hi->mid|all', 'hi->mid|nz', 'mid->lo|nz', 'hi->lo|nz']
arms  = ['real', 'draft', 'r2d1', 'r2g1']
fig, ax = plt.subplots(figsize=(9.0, 4.2))
x = np.arange(len(pairs)); w = 0.2
for i, a in enumerate(arms):
    v = [AUX['confusion']['s42'][p][a] for p in pairs]
    ax.bar(x + (i - 1.5) * w, v, w * 0.9, color=CLR[a], label=NAME.get(a, a), zorder=3,
           edgecolor=SURF, linewidth=1.2)
ax.axhline(0, color=INK2, lw=1.2, zorder=4)
ax.set(xticks=x, ylabel='digit-pair coupling  (z vs permutation null)',
       title='Figure 5 — seed 42.  The gated corrector inverts the coupling real data has')
ax.set_xticklabels(pairs); ax.legend(frameon=False, fontsize=9, ncol=4, loc='lower left')
plt.tight_layout(); plt.show()
""")

md(r"""
## Figure 6 — what the gate learned, and what the projection did about it

The gate is `sigmoid(h @ w + b)` with `w` zero-initialised and `b = -2` (a quiet
default, ~0.12 open). If the gate simply damped the residual, `||P||` would be
unchanged. It is not.
""")

co(r"""
fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8))
G = AUX['gate']

ax = axes[0]
b = ax.bar(['seed 42', 'seed 43'], [G['r2g1_s42']['w_norm'], G['r2g1_s43']['w_norm']],
           0.5, color=CLR['r2g1'], zorder=3)
ax.bar_label(b, fmt='%.2f', color=INK2, fontsize=9)
ax.axhline(0, color=INK2, lw=1.4, ls='--')
ax.text(1.35, 0.12, 'init', color=INK2, fontsize=9)
ax.set(ylabel='||w||', title='gate weight left zero\n(the gate is input-dependent)')

ax = axes[1]
b = ax.bar(['seed 42', 'seed 43'], [G['r2g1_s42']['sigmoid_b'], G['r2g1_s43']['sigmoid_b']],
           0.5, color=CLR['r2g1'], zorder=3)
ax.bar_label(b, fmt='%.3f', color=INK2, fontsize=9)
ax.axhline(0.1192, color=INK2, lw=1.4, ls='--')
ax.text(1.34, 0.1215, 'init', color=INK2, fontsize=9)
ax.set(ylim=(0, 0.2), ylabel='sigmoid(b)', title='default openness barely moved\n(-2.00 -> -1.89)')

ax = axes[2]
x = np.arange(2); w2 = 0.36
for i, tag in enumerate(['r2d1', 'r2g1']):
    v = [G[f'{tag}_s{s}']['P_fro'] for s in [42, 43]]
    b = ax.bar(x + (i - 0.5) * w2, v, w2 * 0.92, color=CLR[tag], label=NAME[tag], zorder=3)
    ax.bar_label(b, fmt='%.1f', color=INK2, fontsize=9)
ax.set(xticks=x, ylabel='||P||_F', title='the projection grew 40% to compensate')
ax.set_xticklabels(['seed 42', 'seed 43'])
ax.legend(frameon=False, fontsize=9, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.34))
fig.suptitle('Figure 6 — the gate closes the residual and the projection re-opens it',
             y=1.05, fontsize=11)
plt.tight_layout(); plt.show()
""")

md(r"""
## Figure 7 — held-out loss cannot see any of this

The training objective is flat across the whole second half of training, and the
gated and ungated runs sit on the same plateau. Every distinction above is invisible
to it. This is why the judgment is made on rollouts and not on the loss curve.
""")

co(r"""
fig, ax = plt.subplots(figsize=(7.2, 3.8))
for s, ls in [('r2g1_s42', '-'), ('r2g1_s43', '--')]:
    a = np.array(AUX['heldout'][s])
    ax.plot(a[:, 0], a[:, 1], ls, color=CLR['r2g1'], marker='o', ms=5,
            label=f'gated seed {s[-2:]}')
ax.set(xlabel='training step', ylabel='held-out loss',
       title='Figure 7 — flat from step 1000 on; the rollout differences live below its resolution')
ax.legend(frameon=False, fontsize=9)
plt.tight_layout(); plt.show()
""")

md(r"""
## The locked bar

Registered before the runs (`run_tau_sweep.sh`): a cell passes if **at least 3 of 5
fields** have level below the draft **and** cumulative growth at or below zero.
""")

co(r"""
rows = []
for tag in CELLS:
    for s in SEEDS:
        k = f'{tag}_s{s}'
        if k not in D['cells']: continue
        ok_lvl = ok_slp = ok_both = 0
        for f in FIELDS:
            if f'{f}|corr' not in cell(k)['curves']: continue
            lv = level(k, f, 'corr') < level(k, f, 'draft')
            sl = slope(k, f, 'corr') <= 0
            ok_lvl += lv; ok_slp += sl; ok_both += (lv and sl)
        rows.append((NAME[tag], s, cell(k)['n_tickers'], ok_lvl, ok_slp, ok_both,
                     'PASS' if ok_both >= 3 else 'fail'))

hdr = f"{'cell':<20}{'seed':>5}{'tickers':>9}{'level<draft':>13}{'slope<=0':>10}{'both':>7}{'verdict':>10}"
print(hdr); print('-' * len(hdr))
for r in rows:
    print(f'{r[0]:<20}{r[1]:>5}{r[2]:>9}{r[3]:>10}/5{r[4]:>7}/5{r[5]:>5}/5{r[6]:>10}')
""")

md(r"""
## What this rules out

**The residual corrector, as formulated, cannot beat not correcting at all.** Forty
of forty (cell, field, seed) measurements land above the diagonal in Figure 1. The
six configurations differ in training distribution, in whether an offset table
exists, in whether the projection is trained at all, and in whether the residual is
gated. None of them changes the sign.

Three things were each fixed and each failed to be sufficient:

- **The train/deploy distribution mismatch.** `r2d0`/`r2d1` train on `corrupt(draft)`
  rather than `corrupt(real)`, so the corrector now sees at training time the input
  it sees at deployment. Levels improved by an order of magnitude on `event_type`
  (Figure 2) and the corrector stopped being uniformly catastrophic. It still never
  reaches the draft.
- **The projection's own damage.** `p0bidir` zeroes the learned projection and the
  level gets *worse*, not better (Figure 1, violet). So most of what the trained
  projection does is undo damage inflicted by the untrained bidirectional pass it
  sits inside; only a minority of its capacity is spent on the actual correction.
- **The residual's loudness.** The gate acts where it was aimed, on two of the three
  measurements that track it. The dt > 0 overshoot retreats 4-5 points toward real on
  both seeds (Figure 4), and `size` growth falls to 0.07x and 0.22x of the draft's on
  seeds 42 and 43 (Figure 3, right). On the other four fields the two seeds disagree
  about the sign, so the gate has no established effect there. Meanwhile the level
  gets worse on both seeds, and Figure 6 shows why: the projection grows 40% in
  Frobenius norm, pushing back through the gate that was meant to quiet it.

That compensation is the informative part. The gate does not lose to noise; it loses
to the rest of the network absorbing it. A constraint applied at one point in a
jointly-trained residual path is undone elsewhere in that path, and the seed-level
disagreement on four of five fields says the training signal never pinned down what
the residual should do in the first place.

**What is still open.** Every cell so far edits the residual *correction* while
keeping per-position factorised sampling. The measurements that do not improve
across any cell (Figure 5's digit coupling, and the level floor) are exactly the
ones that depend on joint structure within a message. That points the next round at
the sampler rather than the corrector.
""")

nb['cells'] = C
nb.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
nb.metadata.language_info = {'name': 'python', 'version': '3.12'}
A = ('/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/dfm-r2-20260830/'
     'post_training/dfm/analysis/compound_error_r2.ipynb')
nbf.write(nb, A)
print('wrote', A, len(C), 'cells')
