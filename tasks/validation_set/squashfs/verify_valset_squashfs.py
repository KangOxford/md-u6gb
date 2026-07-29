#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""物化 shard 双层校验：
L1 逐字节：抽样 n_check 个样本，物化文件全行 vs 原 shard 文件对应行区间 np.array_equal。
L2 格式兼容：用训练 dataloader（rand_offset=False）挂物化 shard 构建数据集，
   断言样本数并 smoke 读首尾样本。
"""
import os, sys, json, argparse, re
import numpy as np
from pathlib import Path

EXP_DIR = "/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_R1_Mamba3"
CONSTITUENTS = "/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/scaling_law_plots/snp500_constituents_20260131.csv"
sys.path.insert(0, EXP_DIR)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

ap = argparse.ArgumentParser()
ap.add_argument("--mount", required=True, help="物化 shard 挂载点")
ap.add_argument("--provenance", required=True)
ap.add_argument("--n_samples", type=int, required=True)
ap.add_argument("--n_check", type=int, default=2048)
A = ap.parse_args()

from lob.lobster_dataloader import LOBSTER_Dataset, _np_load_zst
prov = np.load(A.provenance)
g_all = prov["global_idx"]
rng = np.random.default_rng(20260731)
pick = np.sort(rng.choice(len(g_all), min(A.n_check, len(g_all)), replace=False))

# L1: 逐字节
fail = 0
for k, i in enumerate(pick):
    g = int(g_all[i]); s = int(prov["seq_start"][i])
    mp, bp = str(prov["msg_paths"][i]), str(prov["book_paths"][i])
    tk = Path(mp).parent.name
    date = re.search(r"([0-9]{4}-[0-9]{2}-[0-9]{2})", Path(mp).name).group(1)
    m_new = np.asarray(_np_load_zst(f"{A.mount}/{tk}/{tk}_{date}_message_val{g:08d}.npy.zst"))
    b_new = np.asarray(_np_load_zst(f"{A.mount}/{tk}/{tk}_{date}_orderbook_val{g:08d}.npy.zst"))
    m_src = np.asarray(_np_load_zst(mp, mmap_mode='r'))[s:s + 500]
    b_src = np.asarray(_np_load_zst(bp, mmap_mode='r'))[s:s + 500]
    if not (np.array_equal(m_new, m_src) and np.array_equal(b_new, b_src)):
        fail += 1
        print(f"[L1 FAIL] g={g}", flush=True)
    if k % 500 == 0:
        print(f"[L1] {k}/{len(pick)} checked", flush=True)
assert fail == 0, f"L1 byte-compare failed on {fail} samples"
print(f"[L1] PASS: {len(pick)} samples byte-identical", flush=True)

# L2: dataloader 兼容性
def read_tickers(path):
    out = []
    for i, line in enumerate(open(path)):
        c = line.split(",")[0].strip()
        if not c: continue
        if i == 0 and c.lower().startswith("ticker"): continue
        out.append(c)
    return out

from lob.dataloading import create_lobster_prediction_dataset
ret = create_lobster_prediction_dataset(
    A.mount, seed=42, mask_fn=LOBSTER_Dataset.no_mask, msg_seq_len=500,
    micro_bsz=1, num_devices=1, use_book_data=True, use_simple_book=False,
    book_transform=True, book_depth=500, book_ablation="real", n_data_workers=0,
    shuffle_train=True, rand_offset=False, debug_overfit=False,
    val_split=0.0, test_split=0.0, pin_memory=False, prefetch_factor=None,
    persistent_workers=False, test_dir_name=None, use_distributed_sampler=False,
    process_rank=0, process_count=1, tickers=read_tickers(CONSTITUENTS),
    data_root=A.mount, train_date_range=["2022-01-01", "2025-12-31"],
    test_date_range=None, token_mode="26tok")
ds = ret[0].dataset_train
assert len(ds) == A.n_samples, f"len={len(ds)} != {A.n_samples}"
for probe in (0, len(ds) // 2, len(ds) - 1):
    sample = ds[probe]
    assert sample[0] is not None
print(f"[L2] PASS: dataloader sees {len(ds):,} samples (rand_offset=False), probes OK", flush=True)
print("VERIFY_OK", flush=True)
