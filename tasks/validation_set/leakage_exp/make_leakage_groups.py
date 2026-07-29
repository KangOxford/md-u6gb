#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成泄漏实验三组样本索引（各 30,720），存 npy。

SEEN : perm5[0 : 10,590×128] 内均匀抽样 —— 350M-seed5 (j4499580) 实际消费前缀，
       对 78M-seed5 (j4499538, 消费 28,710×128) 同样是已见数据。
MID  : perm5 位置 ∈ [20%N, 98%N) 段内抽样，且不属于任何 seed 的 last-2% 尾段
       （通过 42/137 逆排列位置 < 98% 验证）——确定未见、且与 val 无交。
VAL  : val_subset_30720（被检验对象）。
三组同为全域均匀随机子集，仅训练暴露状态不同（规避 MIA 文献的分布偏移假信号）。
"""
import numpy as np, torch, json
from pathlib import Path

BASE = Path("/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set")
OUT = BASE / "leakage_exp/groups"
OUT.mkdir(parents=True, exist_ok=True)
N = 323_221_385
SEEN_PREFIX = 10_590 * 128          # 350M seed5 已确认消费前缀
GROUP = 30_720
RNG_SEED = 20260730
TAIL = int(0.02 * N)
X20 = -(-N // 5)  # ceil(0.2N)

def perm(seed):
    g = torch.Generator(); g.manual_seed(seed)
    return torch.randperm(N, generator=g).numpy()

rng = np.random.default_rng(RNG_SEED)
p5 = perm(5)

seen_pool = p5[:SEEN_PREFIX]
seen = np.sort(seen_pool[rng.choice(len(seen_pool), GROUP, replace=False)])

mid_pool = p5[X20 : N - TAIL]                      # seed5 中段（未见、非 seed5 尾）
cand = mid_pool[rng.choice(len(mid_pool), GROUP * 3, replace=False)]
del p5, mid_pool, seen_pool
keep = np.ones(len(cand), dtype=bool)
for s in (42, 137):                                # 排除其它 seed 的尾段成员
    ps = perm(s)
    inv = np.empty(N, dtype=np.int64); inv[ps] = np.arange(N)
    keep &= inv[cand] < (N - TAIL)
    del ps, inv
mid = np.sort(cand[keep][:GROUP])
assert len(mid) == GROUP

val = np.load(BASE / "artifacts_valset_v1_j5790795/val_subset_30720.npy")
assert len(val) == GROUP
assert not np.intersect1d(seen, val).size and not np.intersect1d(mid, val).size
assert not np.intersect1d(seen, mid).size

np.save(OUT / "group_seen.npy", seen)
np.save(OUT / "group_mid.npy", mid)
np.save(OUT / "group_val.npy", np.sort(val))
json.dump(dict(N=N, seen_prefix=SEEN_PREFIX, group=GROUP, rng_seed=RNG_SEED,
               note="SEEN⊆perm5[:10590*128]; MID⊆perm5[20%:98%] minus all tails; VAL=val_subset_30720"),
          open(OUT / "groups_manifest.json", "w"), indent=1)
print("groups saved:", len(seen), len(mid), len(val))
