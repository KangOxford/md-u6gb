#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 valset_v1 子集物化成与 training shard 同构的目录树（供 mksquashfs 打包）。

布局与月度 shard 完全一致，现有 dataloader 零改动可读（评测时传
--random_offsets_train False，每文件恰存一个 500-msg 窗口）：
  TICKER/TICKER_<date>_message_val<GLOBALIDX:08d>.npy.zst   # [500, W_msg] 原 dtype
  TICKER/TICKER_<date>_orderbook_val<GLOBALIDX:08d>.npy.zst # [500, W_book]
  index.json   # {"version":1,"shard":...,"n_files":...,"shapes":{rel:{shape,dtype}}}
  VALSET_MATERIALIZE_MANIFEST.json

切片定义与训练管线逐字节一致：行区间 = [seq_start, seq_start+500)，
seq_start = 恒定 offset[file] + 500*j（valset 构建时已导出于 decode.npz）。
"""
import os, sys, io, json, argparse
import numpy as np
from pathlib import Path
from multiprocessing import Pool

EXP_DIR = "/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_R1_Mamba3"
CONSTITUENTS = "/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/scaling_law_plots/snp500_constituents_20260131.csv"
ART = Path("/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/artifacts_valset_v1_j5790795")
N_EXPECT = 323_221_385
sys.path.insert(0, EXP_DIR)

ap = argparse.ArgumentParser()
ap.add_argument("--subset_npy", required=True, help="要物化的样本 global_idx 列表")
ap.add_argument("--shard_name", required=True, help="如 valset_v1_30720")
ap.add_argument("--data_root", required=True, help="48 个月挂载点逗号串")
ap.add_argument("--out_tree", required=True, help="输出目录树根（TMPDIR 下）")
ap.add_argument("--nproc", type=int, default=48)
A = ap.parse_args()

def read_tickers(path):
    out = []
    for i, line in enumerate(open(path)):
        c = line.split(",")[0].strip()
        if not c: continue
        if i == 0 and c.lower().startswith("ticker"): continue
        out.append(c)
    return out

# ── 用训练同款工厂重建 file_idx → (msg_path, book_path, offset) 映射 ──
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from lob.lobster_dataloader import LOBSTER_Dataset, _np_load_zst
from lob.dataloading import create_lobster_prediction_dataset
ret = create_lobster_prediction_dataset(
    "/lus/lfs1aip2/projects/public/s5e/quant_team/lob_preproc_sp500", seed=42,
    mask_fn=LOBSTER_Dataset.no_mask, msg_seq_len=500, micro_bsz=1, num_devices=1,
    use_book_data=True, use_simple_book=False, book_transform=True, book_depth=500,
    book_ablation="real", n_data_workers=0, shuffle_train=True, rand_offset=True,
    debug_overfit=False, val_split=0.0, test_split=0.0, pin_memory=False,
    prefetch_factor=None, persistent_workers=False, test_dir_name=None,
    use_distributed_sampler=False, process_rank=0, process_count=1,
    tickers=read_tickers(CONSTITUENTS), data_root=A.data_root,
    train_date_range=["2022-01-01", "2025-12-31"], test_date_range=None,
    token_mode="26tok")
train = ret[0].dataset_train
assert len(train) == N_EXPECT
msg_files = [str(p) for p in train.message_files]
book_files = [str(p) for p in train.book_files]
cumsum = np.asarray(train._seqs_cumsum)
offsets = train.seq_offsets.numpy().astype(np.int64)
del ret, train
print(f"[map] files={len(msg_files):,}", flush=True)

sub = np.sort(np.load(A.subset_npy))
dec = np.load(ART / "val_pool_decode.npz")
pos = np.searchsorted(dec["global_idx"], sub)
assert np.array_equal(dec["global_idx"][pos], sub), "子集必须 ⊆ val pool"
fidx = dec["file_idx"][pos]
sstart = dec["seq_start_msg"][pos]
# 交叉校验 decode 的 seq_start 与本次重建的 offset 一致
j = sub - cumsum[fidx]
assert np.array_equal(sstart, offsets[fidx] + 500 * j), "offset 漂移：数据集状态与构建时不一致!"
print(f"[subset] {len(sub):,} samples in {len(np.unique(fidx)):,} source files", flush=True)

OUT = Path(A.out_tree)
import re as _re
DATE_RE = _re.compile(r"([0-9]{4}-[0-9]{2}-[0-9]{2})")
try:
    import zstandard as zstd
    CCTX = zstd.ZstdCompressor(level=10)
except ImportError:
    raise SystemExit("zstandard required")

def npy_zst_bytes(arr):
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(arr))
    return CCTX.compress(buf.getvalue())

def work(file_group):
    fi, items = file_group          # items: [(global_idx, seq_start), ...]
    X = np.asarray(_np_load_zst(msg_files[fi], mmap_mode='r'))
    B = np.asarray(_np_load_zst(book_files[fi], mmap_mode='r'))
    ticker = Path(msg_files[fi]).parent.name
    date = DATE_RE.search(Path(msg_files[fi]).name).group(1)
    tdir = OUT / ticker
    tdir.mkdir(parents=True, exist_ok=True)
    shapes = {}
    for g, s in items:
        xm = X[s:s + 500]
        xb = B[s:s + 500]
        assert xm.shape[0] == 500 and xb.shape[0] == 500, f"short slice g={g}"
        rm = f"{ticker}/{ticker}_{date}_message_val{g:08d}.npy.zst"
        rb = f"{ticker}/{ticker}_{date}_orderbook_val{g:08d}.npy.zst"
        (OUT / rm).write_bytes(npy_zst_bytes(xm))
        (OUT / rb).write_bytes(npy_zst_bytes(xb))
        shapes[rm] = {"shape": list(xm.shape), "dtype": str(xm.dtype)}
        shapes[rb] = {"shape": list(xb.shape), "dtype": str(xb.dtype)}
    return shapes

groups = {}
for g, fi, s in zip(sub.tolist(), fidx.tolist(), sstart.tolist()):
    groups.setdefault(fi, []).append((g, s))
work_items = sorted(groups.items())
print(f"[extract] {len(work_items):,} file groups, nproc={A.nproc}", flush=True)

OUT.mkdir(parents=True, exist_ok=True)
all_shapes = {}
with Pool(A.nproc) as pool:
    for i, shapes in enumerate(pool.imap_unordered(work, work_items, chunksize=8)):
        all_shapes.update(shapes)
        if i % 2000 == 0:
            print(f"  [extract] {i}/{len(work_items)} files done", flush=True)

assert len(all_shapes) == 2 * len(sub), f"{len(all_shapes)} != {2*len(sub)}"
json.dump({"version": 1, "shard": A.shard_name, "n_files": len(all_shapes),
           "shapes": all_shapes}, open(OUT / "index.json", "w"))
# provenance（校验用：物化文件 ↔ 原文件行区间；不进 squashfs）
np.savez_compressed(Path(A.out_tree).parent / f"provenance_{A.shard_name}.npz",
                    global_idx=sub, file_idx=fidx, seq_start=sstart,
                    msg_paths=np.array([msg_files[i] for i in fidx.tolist()]),
                    book_paths=np.array([book_files[i] for i in fidx.tolist()]))
json.dump(dict(shard=A.shard_name, n_samples=int(len(sub)),
               source="valset_v1 (artifacts_valset_v1_j5790795)",
               subset_npy=os.path.basename(A.subset_npy),
               slice_rule="rows [seq_start, seq_start+500) of paired msg/book; "
                          "seq_start = constant per-file offset + 500*j (decode.npz)",
               eval_note="mount as a data_root; use --random_offsets_train False; "
                         "one 500-msg window per file, filename embeds global_idx",
               layout="identical to monthly training shards (message/orderbook pairing)"),
          open(OUT / "VALSET_MATERIALIZE_MANIFEST.json", "w"), indent=1)
print(f"[tree] {len(all_shapes):,} files + index.json written to {OUT}", flush=True)
print("MATERIALIZE_OK", flush=True)
