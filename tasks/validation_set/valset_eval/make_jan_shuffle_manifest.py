#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Jan-2026 shuffle 评测的样本清单：全月窗口池（每文件 floor(rows/500) 个不重叠
500-msg 窗口，与训练 dataloader 同枚举）→ 固定 seed 完全 shuffle → 前 30,720 个。
输出：jan_shuffle_30720_indices.npy（dataset 全局窗口索引，配 rand_offset=False 的
create_lobster_prediction_dataset(2026-01) 使用）+ per-sample ticker 映射 + 统计。

窗口枚举顺序必须与 lobster_dataloader 一致：ticker 字母序 → 文件名序 → 文件内窗口序。
（与 valset 的 L2/对齐验证同一性质，运行时以 dataset N 断言 + ANOVA 复核。）
"""
import json, collections
import numpy as np

IDX = "/lus/lfs1aip2/projects/public/s5e/quant_team/lob_preproc_sp500_squashfs/index_2026-01.json"
OUT_DIR = "/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/valset_eval"
SEED = 20260131
N_SAMPLE = 30720
MSG_PER_WIN = 500

idx = json.load(open(IDX))
# message 文件 → 窗口数；按 (ticker, filename) 排序 = dataloader 枚举序
files = []
for path, meta in idx["shapes"].items():
    if "message" not in path:
        continue
    tk, fname = path.split("/", 1)
    files.append((tk, fname, meta["shape"][0] // MSG_PER_WIN))
files.sort(key=lambda x: (x[0], x[1]))

win_ticker, win_file, win_local = [], [], []
for tk, fname, n_win in files:
    win_ticker.extend([tk] * n_win)
    win_file.extend([f"{tk}/{fname}"] * n_win)
    win_local.extend(range(n_win))
N_pool = len(win_ticker)
win_ticker = np.array(win_ticker)

rng = np.random.default_rng(SEED)
perm = rng.permutation(N_pool)
sel = np.sort(perm[:N_SAMPLE])  # 排序后交给顺序 sampler（选中集合不变）

np.save(f"{OUT_DIR}/jan_shuffle_30720_indices.npy", sel.astype(np.int64))
np.save(f"{OUT_DIR}/jan_ticker_per_sample_30720.npy", win_ticker[sel])
np.save(f"{OUT_DIR}/jan_pool_ticker_all.npy", win_ticker)  # 运行时对齐校验用

cnt = collections.Counter(win_ticker[sel])
pool_cnt = collections.Counter(win_ticker)
print(f"pool windows = {N_pool:,} (487 tickers, 全月)")
print(f"sampled = {len(sel):,}  seed={SEED}  tickers covered = {len(cnt)}/487")
top = cnt.most_common(5)
print("sampled top5:", [(t, n, f"pool {pool_cnt[t]:,}") for t, n in top])
import math
exp_top = [(t, N_SAMPLE * pool_cnt[t] / N_pool) for t, _ in top]
print("expected top5 (natural share):", [(t, f"{e:.0f}") for t, e in exp_top])
meta = dict(seed=SEED, n_pool=int(N_pool), n_sample=N_SAMPLE,
            source_index=IDX, months=["2026-01"],
            enumeration="ticker-alpha -> filename -> window-in-file, floor(rows/500), no offset")
json.dump(meta, open(f"{OUT_DIR}/jan_shuffle_manifest_meta.json", "w"), indent=1)
print("wrote indices/ticker-map/meta")
