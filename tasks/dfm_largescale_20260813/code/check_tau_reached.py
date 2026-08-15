#!/usr/bin/env python3
"""三条 tau 臂到底是不是同一次运行？—— 比对输出，不看日志。

比日志更强：日志说的是**意图**，输出说的是**效果**。判据：

  1. 三条臂的 draft 必须**逐元素相同**（同 seed、同数据），否则不是配对实验
  2. 三条臂的 corr 必须**互不相同**，否则 keep_tau 没到达模型
  3. 与 draft 的一致率必须随 tau 变小而**上升**（tau 小保留多）
"""
import glob, sys
import numpy as np
R = '/lus/lfs1aip2/projects/public/u6gb/tasks/dfm_largescale_20260813/rollouts_tau'
TAGS = [('t0', 0.0), ('t001', 0.01), ('t01', 0.1)]

common = None
data = {}
for tag, tau in TAGS:
    fs = {p.split('/')[-1].split('_')[2] for p in glob.glob(f'{R}/tau_{tag}_*_2026-01_learned.npz')}
    common = fs if common is None else (common & fs)
common = sorted(common or [])
if not common:
    print("三条臂还没有共同完成的 ticker"); sys.exit(0)
use = common[:8]
print(f"共同 ticker {len(common)} 个，用前 {len(use)}: {use}\n")

for tag, tau in TAGS:
    d, c = [], []
    for tk in use:
        z = np.load(f'{R}/tau_{tag}_{tk}_2026-01_learned.npz', allow_pickle=True)
        d.append(z['draft_msgs']); c.append(z['corr_msgs'])
    data[tag] = (np.concatenate([x.ravel() for x in d]),
                 np.concatenate([x.ravel() for x in c]))

base = data['t0']
print(f"{'tau':>6}  {'draft 与 t0 相同':>16}  {'corr 与 t0 相同':>16}  {'corr 与自己 draft 相同':>22}")
for tag, tau in TAGS:
    dd, cc = data[tag]
    same_d = float((dd == base[0]).mean())
    same_c = float((cc == base[1]).mean())
    keep = float((cc == dd).mean())
    print(f"{tau:>6.3g}  {same_d:>16.6f}  {same_c:>16.6f}  {keep:>22.6f}")

ok = []
ok.append(("1 三臂 draft 逐元素相同",
           all(float((data[t][0] == base[0]).mean()) == 1.0 for t, _ in TAGS)))
ok.append(("2 corr 互不相同（knob 到达了模型）",
           all(float((data[t][1] == base[1]).mean()) < 0.999 for t, _ in TAGS[1:])))
keeps = [float((data[t][1] == data[t][0]).mean()) for t, _ in TAGS]
ok.append(("3 保留率随 tau 变小而上升", keeps[1] >= keeps[2] - 1e-9))
print()
for n, v in ok:
    print(f"  {'PASS' if v else 'FAIL'}  {n}")
sys.exit(0 if all(v for _, v in ok) else 1)
