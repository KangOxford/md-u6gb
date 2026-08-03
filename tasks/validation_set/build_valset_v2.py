#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_valset_v2.py — 构建 SP500_2022_2025_Validation_Version_1。

与 valset_v1 的唯一配方差异：**删除 36 个月子域（Transformer）的 20% 排除项**。

  v1: V = (∪_s perm_s48[-2%:]) \\ (∪_s perm_s48[:20%])
          \\ map48(∪_s perm_s36[:20%], ±1 guard)      ← 只作用于 2023-2025，是偏斜的唯一成因
          \\ map48(perm42_466[:19200], guard) \\ (GOOG × 2025-12)

  v2: V = (∪_s perm_s48[-2%:]) \\ (∪_s perm_s48[:20%])
          \\ map48(perm42_466[:19200], guard) \\ (GOOG × 2025-12)

为什么这样就够：本集合只需保证 **Mamba-3 从未见过**。Mamba-3 的三个 seed 全部在 48 个月域上
训练，其消费区完全由 perm_s48[:20%] 覆盖（实测最深 seed 5 为 16.63% < 20%）。36 个月域的排除
是为 Transformer 加的额外保险，而该子域不含 2022，导致 2022 留存 3.744% 对 2023-25 的 0.984%
（比值 3.80），这正是 v1 里 2022 占 55.2%（全域 24.51%）的来源。删掉它，留存率对全部 48 个月
一致，偏斜从根上消失。

**代价（必须写进文档）**：v2 对 Transformer 队列不保证 held-out。TF 在 36 个月域上训练，其消费
区未被排除。v2 只服务 Mamba-3。

规模：固定 0.5% × N48 = 1,616,107 个窗口。

零偏斜的保证方式不是「靠随机抽样期望上无偏」，而是**按 (ticker, month) 联合分层比例抽样**，
再用 IPF（迭代比例拟合）把两个边际同时对齐到全域比例，最后用最大余数法取整到精确总数。
(GOOG, 2025-12) 因 finetune 整片消费被强制置 0，IPF 会把这一格的配额在该行与该列内重新分配，
使 ticker 边际与 month 边际仍精确匹配。

嵌套子集用「分层交错序」构造：每个样本的排序键为 (格内序号 + 0.5) / 格内总数，
按此键全局排序后，**任意前缀都近似按比例覆盖所有格**，因此 30,720 ⊂ 307,200 ⊂ 全集
且每一档都无偏斜。
"""
import os, sys, json, gc, math, hashlib, re
from datetime import datetime, timezone
from pathlib import Path

EXP_DIR = "/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_R1_Mamba3"
CONSTITUENTS_CSV = "/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/scaling_law_plots/snp500_constituents_20260131.csv"
PILOT_TICKERS_JSON = "/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/pilot_8plu95a3_tickers.json"
RAW_ROOT_LABEL = "/lus/lfs1aip2/projects/public/s5e/quant_team/lob_preproc_sp500"

VERSION = "SP500_2022_2025_Validation_Version_1"
SEEDS = [5, 42, 137]
TAIL_FRAC = 0.02
EXCL_FRAC = 0.20
TARGET_FRAC = 0.005                  # ← 固定 0.5%
CONSUMED_CAPS = {5: 420_000 * 128, 42: max(106_909 * 128, 168_200 * 128), 137: 106_909 * 128}
C466_SAMPLES = 300 * 64
V1_TICKERS = {"GOOG", "AAPL", "NVDA", "AMZN", "META", "TSLA", "MSFT", "AMD"}
GOOG_EXCISE = ("GOOG", "2025-12")
SUBSET_SEED = 20260803
NESTED_SIZES = [30_720, 307_200]     # 主集合（0.5%）之下的两档嵌套子集
ASSERT_N48_DIV8 = 40_402_673
MONTHS48 = [f"{y}-{m:02d}" for y in range(2022, 2026) for m in range(1, 13)]

# 零偏斜验收阈值（百分点）。分层+IPF 后应远优于此。
SKEW_TOL_PP_MAIN = 0.02
SKEW_TOL_PP_NESTED = 0.60

OUTDIR = Path(os.environ["ARTIFACTS_DIR"])
MOUNT_ROOT = Path(os.environ["SQUASHFS_MULTI_MOUNT_ROOT"])
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, EXP_DIR)

import numpy as np
import torch

DATE_RE = re.compile(r"([0-9]{4}-[0-9]{2}-[0-9]{2})")

def log(*a):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}]", *a, flush=True)

def read_constituents(path):
    out = []
    for i, line in enumerate(open(path)):
        c = line.split(",")[0].strip()
        if not c: continue
        if i == 0 and c.lower().startswith("ticker"): continue
        out.append(c)
    return out

def build_domain(tag, months, tickers, date_range):
    from lob.dataloading import create_lobster_prediction_dataset
    from lob.lobster_dataloader import LOBSTER_Dataset
    data_root = ",".join(str(MOUNT_ROOT / m) for m in months)
    log(f"[{tag}] building dataset: {len(months)} months, {len(tickers)} tickers, range={date_range}")
    ret = create_lobster_prediction_dataset(
        cache_dir=RAW_ROOT_LABEL, seed=42, mask_fn=LOBSTER_Dataset.no_mask,
        msg_seq_len=500, micro_bsz=1, num_devices=1, use_book_data=True,
        use_simple_book=False, book_transform=True, book_ablation="real", book_depth=500,
        n_data_workers=0, shuffle_train=True, rand_offset=True, debug_overfit=False,
        val_split=0.0, test_split=0.0, pin_memory=False, prefetch_factor=None,
        persistent_workers=False, test_dir_name=None, use_distributed_sampler=False,
        process_rank=0, process_count=1, tickers=list(tickers), data_root=data_root,
        train_date_range=list(date_range), test_date_range=None, token_mode="26tok")
    train = ret[0].dataset_train
    N = len(train)
    files = [str(p) for p in train.message_files]
    tk = np.array([Path(p).parent.name for p in files])
    dt = np.array([(DATE_RE.search(Path(p).name).group(1) if DATE_RE.search(Path(p).name) else "?") for p in files])
    seqs = np.asarray(train._seqs_per_file, dtype=np.int64)
    cumsum = np.asarray(train._seqs_cumsum, dtype=np.int64)
    offsets = train.seq_offsets.numpy().astype(np.int64).copy()
    rows = np.asarray(train._num_rows_per_file, dtype=np.int64)
    log(f"[{tag}] N={N:,} files={len(files):,}")
    assert int(cumsum[-1]) == N
    del ret, train; gc.collect()
    return dict(tag=tag, N=N, tickers=tk, dates=dt, seqs=seqs, cumsum=cumsum,
                offsets=offsets, rows=rows, files=files)

def perm_for(N, seed):
    g = torch.Generator(); g.manual_seed(seed)
    return torch.randperm(N, generator=g).numpy().astype(np.int64)

def verify_sampler_equivalence(N, seed=42, R=8, k=1000):
    from torch.utils.data import DistributedSampler
    class Dummy:
        def __len__(self): return N
    s = DistributedSampler(Dummy(), num_replicas=R, rank=0, shuffle=True, seed=seed, drop_last=True)
    s.set_epoch(0)
    it = iter(s)
    got = np.array([next(it) for _ in range(k)], dtype=np.int64)
    assert np.array_equal(got, perm_for(N, seed)[0::R][:k]), "DistributedSampler ≠ randperm 复现，中止！"
    log(f"sampler equivalence OK (N={N:,}, R={R}, k={k})")

def foreign_domain_maps(dom_small, dom48, fidx_V0, j_V0):
    key48 = {(dom48["tickers"][f], dom48["dates"][f]): f for f in range(len(dom48["seqs"]))}
    nf_s = len(dom_small["seqs"])
    base = np.full(len(dom48["seqs"]), -1, dtype=np.int64)
    nseq = np.zeros(len(dom48["seqs"]), dtype=np.int64)
    ddir = np.zeros(len(dom48["seqs"]), dtype=np.int64)
    matched = mism = 0
    for fs in range(nf_s):
        f48 = key48.get((dom_small["tickers"][fs], dom_small["dates"][fs]))
        if f48 is None: continue
        matched += 1
        if dom_small["seqs"][fs] != dom48["seqs"][f48]: mism += 1
        base[f48] = dom_small["cumsum"][fs]; nseq[f48] = dom_small["seqs"][fs]
        d = int(dom_small["offsets"][fs]) - int(dom48["offsets"][f48])
        ddir[f48] = 0 if d == 0 else (1 if d > 0 else -1)
    log(f"[{dom_small['tag']}] file join: matched={matched:,}/{nf_s:,} seq-mismatch={mism}")
    assert mism == 0, "同一文件在两域的窗口数不一致——配对逻辑漂移!"
    b = base[fidx_V0]
    valid = (b >= 0) & (j_V0 < nseq[fidx_V0])
    m = np.where(valid, b + j_V0, 0)
    m2 = m - ddir[fidx_V0]
    has_m2 = valid & (ddir[fidx_V0] != 0) & (m2 >= b) & (m2 < b + nseq[fidx_V0])
    return valid, m, np.where(has_m2, m2, 0), has_m2

def drop_by_zone(tag, N_small, seeds_and_thresholds, valid, m, m2, has_m2):
    drop = np.zeros(len(valid), dtype=bool)
    for s, X in seeds_and_thresholds:
        p = perm_for(N_small, s)
        inv = np.empty(N_small, dtype=np.int64); inv[p] = np.arange(N_small, dtype=np.int64)
        del p; gc.collect()
        hit = valid & (inv[m] < X); hit |= has_m2 & (inv[m2] < X)
        drop |= hit
        log(f"[{tag}] seed {s} zone<{X:,}: +{int(hit.sum()):,} (cum {int(drop.sum()):,})")
        del inv; gc.collect()
    return drop

def largest_remainder(target_real, total):
    """按最大余数法把实数配额取整成整数且总和精确等于 total。"""
    fl = np.floor(target_real).astype(np.int64)
    rem = total - int(fl.sum())
    if rem > 0:
        frac = target_real - fl
        idx = np.argsort(-frac, kind="stable")[:rem]
        fl[idx] += 1
    elif rem < 0:
        frac = target_real - fl
        cand = np.flatnonzero(fl > 0)
        idx = cand[np.argsort(frac[cand], kind="stable")[:(-rem)]]
        fl[idx] -= 1
    assert int(fl.sum()) == total
    return fl

def stratified_allocate(dom_cnt, pool_cnt, total, forbidden, n_iter=40):
    """(ticker, month) 联合分层配额。

    dom_cnt / pool_cnt / forbidden: shape (n_ticker, n_month)。
    先按全域联合比例给初始配额，把 forbidden 格置 0，然后用 IPF 反复把
    ticker 边际与 month 边际拉回「全域边际 × TARGET_FRAC」，最后受 pool_cnt 上限截断。
    """
    row_t = dom_cnt.sum(1) * (total / dom_cnt.sum())      # 目标 ticker 边际
    col_t = dom_cnt.sum(0) * (total / dom_cnt.sum())      # 目标 month 边际
    a = dom_cnt.astype(np.float64) * (total / dom_cnt.sum())
    a[forbidden] = 0.0
    for _ in range(n_iter):
        rs = a.sum(1); rs[rs == 0] = 1.0
        a *= (row_t / rs)[:, None]; a[forbidden] = 0.0
        cs = a.sum(0); cs[cs == 0] = 1.0
        a *= (col_t / cs)[None, :]; a[forbidden] = 0.0
    a = np.minimum(a, pool_cnt)                            # 不能超过池里实际有的
    flat = largest_remainder(a.ravel() * (total / max(a.sum(), 1e-9)), total)
    alloc = flat.reshape(a.shape)
    over = alloc > pool_cnt
    if over.any():                                         # 溢出的配额重新摊到有余量的格
        excess = int((alloc[over] - pool_cnt[over]).sum())
        alloc[over] = pool_cnt[over]
        slack = (pool_cnt - alloc).ravel()
        room = np.flatnonzero(slack > 0)
        w = slack[room].astype(np.float64); w /= w.sum()
        add = largest_remainder(w * excess, excess)
        f = alloc.ravel(); f[room] += add; alloc = f.reshape(a.shape)
        log(f"[alloc] {excess:,} 个配额因格内池不足被重新分配")
    assert int(alloc.sum()) == total and bool((alloc <= pool_cnt).all())
    return alloc

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()

def marginal_report(name, sel_t, sel_m, dom_t, dom_m, tol_pp):
    """检查 ticker / month 两个边际与全域比例的最大偏差。"""
    out = {}
    for lbl, sel, dom in (("ticker", sel_t, dom_t), ("month", sel_m, dom_m)):
        s = sel / sel.sum() * 100.0
        d = dom / dom.sum() * 100.0
        mx = float(np.abs(s - d).max())
        out[lbl] = mx
        log(f"  [{name}] {lbl:6s} 最大偏差 {mx:.4f} pp  (阈值 {tol_pp} pp)")
    assert out["ticker"] <= tol_pp and out["month"] <= tol_pp, f"{name} 偏斜超阈值: {out}"
    return out

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now(timezone.utc)
    tickers488 = read_constituents(CONSTITUENTS_CSV)
    tickers466 = json.load(open(PILOT_TICKERS_JSON))
    log(f"tickers: 488-list={len(tickers488)}, pilot-list={len(tickers466)}")

    dom48 = build_domain("48mo", MONTHS48, tickers488, ("2022-01-01", "2025-12-31"))
    N48 = dom48["N"]
    assert N48 // 8 == ASSERT_N48_DIV8, f"N48={N48} 与 8N 日志不符!"
    dom466 = build_domain("466tk", MONTHS48, tickers466, ("2022-01-01", "2026-02-15"))
    verify_sampler_equivalence(N48)

    X48 = math.ceil(EXCL_FRAC * N48); T = math.floor(TAIL_FRAC * N48)
    TOTAL = round(TARGET_FRAC * N48)
    for s, cap in CONSUMED_CAPS.items():
        assert cap < X48, f"seed {s} 消费 {cap:,} ≥ 排除区 {X48:,}!"
    log(f"N48={N48:,} X48={X48:,} T={T:,} TARGET={TOTAL:,} ({TARGET_FRAC*100}% of N)")

    # ── 主配方：尾巴并集 − 48mo 排除区并集（v1 前两步，原样保留）──
    excluded = np.zeros(N48, dtype=bool); tails = []
    for s in SEEDS:
        p = perm_for(N48, s)
        excluded[p[:X48]] = True
        tails.append(p[N48 - T:].copy())
        del p; gc.collect()
        log(f"seed {s}: zone+tail done")
    tail_union = np.unique(np.concatenate(tails)); del tails; gc.collect()
    V0 = tail_union[~excluded[tail_union]]
    n_tail_union = int(len(tail_union)); del tail_union, excluded; gc.collect()
    log(f"tail union={n_tail_union:,}  V0={len(V0):,} ({len(V0)/N48*100:.3f}% of N)")

    fidx_V0 = np.searchsorted(dom48["cumsum"], V0, side="right") - 1
    j_V0 = V0 - dom48["cumsum"][fidx_V0]

    # ★★ v1 在此处有 36mo 排除；v2 **不做**。这是与 v1 的唯一配方差异。 ★★

    v466, m466, m466b, h466 = foreign_domain_maps(dom466, dom48, fidx_V0, j_V0)
    drop466 = drop_by_zone("466tk", dom466["N"], [(42, C466_SAMPLES)], v466, m466, m466b, h466)
    file_goog_dec = (dom48["tickers"] == GOOG_EXCISE[0]) & \
                    np.char.startswith(dom48["dates"].astype(str), GOOG_EXCISE[1])
    drop_goog = file_goog_dec[fidx_V0]
    keep = ~(drop466 | drop_goog)
    POOL = V0[keep]; fidx_P = fidx_V0[keep]; j_P = j_V0[keep]
    log(f"drops: 466tk={int(drop466.sum()):,} goog-dec={int(drop_goog.sum()):,} "
        f"→ POOL={len(POOL):,} ({len(POOL)/N48*100:.3f}% of N)")
    assert len(POOL) >= TOTAL, f"池 {len(POOL):,} < 目标 {TOTAL:,}"

    # ── 硬验证：池对每个 seed 的 first-20% 零交集 ──
    for s in SEEDS:
        p = perm_for(N48, s)
        inv = np.empty(N48, dtype=np.int64); inv[p] = np.arange(N48, dtype=np.int64)
        del p; gc.collect()
        pos = inv[POOL]
        assert int(pos.min()) >= X48, f"seed {s}: 池中有样本在 first-20% 区!"
        assert int(pos.min()) >= CONSUMED_CAPS[s], f"seed {s}: 池与实际消费区相交!"
        del inv, pos; gc.collect()
    log("hard verification: POOL ∩ (∪ 48mo first-20%) = ∅  ✓")

    # ── (ticker, month) 联合分层 ──
    months_of_file = np.char.mod("%s", dom48["dates"].astype(str)).astype("U7")
    tk_list = sorted(set(dom48["tickers"].tolist())); mo_list = MONTHS48
    tk_ix = {t: i for i, t in enumerate(tk_list)}; mo_ix = {m: i for i, m in enumerate(mo_list)}
    f_ti = np.array([tk_ix[t] for t in dom48["tickers"]], dtype=np.int64)
    f_mi = np.array([mo_ix[m] for m in months_of_file], dtype=np.int64)
    nT, nM = len(tk_list), len(mo_list)

    dom_cnt = np.zeros((nT, nM), dtype=np.int64)
    np.add.at(dom_cnt, (f_ti, f_mi), dom48["seqs"])
    p_ti = f_ti[fidx_P]; p_mi = f_mi[fidx_P]
    pool_cnt = np.zeros((nT, nM), dtype=np.int64)
    np.add.at(pool_cnt, (p_ti, p_mi), 1)
    forbidden = np.zeros((nT, nM), dtype=bool)
    forbidden[tk_ix[GOOG_EXCISE[0]], mo_ix[GOOG_EXCISE[1]]] = True
    log(f"cells: {nT} tickers × {nM} months = {nT*nM:,};  pool 非空格 {int((pool_cnt>0).sum()):,}")

    alloc = stratified_allocate(dom_cnt, pool_cnt, TOTAL, forbidden)

    # 每格内用固定 RNG 打乱后取前 alloc[c] 个；同时记录格内序号用于交错序
    rng = np.random.default_rng(SUBSET_SEED)
    cell = p_ti * nM + p_mi
    order = np.argsort(cell, kind="stable")
    cell_s = cell[order]
    bounds = np.searchsorted(cell_s, np.arange(nT * nM + 1))
    sel_parts, key_parts = [], []
    af = alloc.ravel()
    for c in range(nT * nM):
        lo, hi = bounds[c], bounds[c + 1]
        k = int(af[c])
        if k <= 0 or hi <= lo: continue
        idx = order[lo:hi]
        pick = rng.permutation(len(idx))[:k]
        sel_parts.append(idx[pick])
        key_parts.append((np.arange(k, dtype=np.float64) + 0.5) / k)   # 分层交错键
    sel = np.concatenate(sel_parts); keys = np.concatenate(key_parts)
    inter = np.argsort(keys, kind="stable")        # 任意前缀都近似按比例覆盖所有格
    sel_interleaved = sel[inter]
    V = np.sort(POOL[sel_interleaved])
    assert len(V) == TOTAL and len(np.unique(V)) == TOTAL
    log(f"V = {len(V):,} ({len(V)/N48*100:.4f}% of N48)")

    # ── 零偏斜验收 ──
    dom_t = dom_cnt.sum(1); dom_m = dom_cnt.sum(0)
    def marg(sub_local_idx):
        ti = p_ti[sub_local_idx]; mi = p_mi[sub_local_idx]
        st = np.bincount(ti, minlength=nT); sm = np.bincount(mi, minlength=nM)
        return st, sm
    log("零偏斜验收：")
    st, sm = marg(sel_interleaved)
    skew_main = marginal_report(f"{VERSION} (0.5%)", st, sm, dom_t, dom_m, SKEW_TOL_PP_MAIN)

    nested = {}
    skew_nested = {}
    for k in NESTED_SIZES:
        loc = sel_interleaved[:k]
        nested[k] = np.sort(POOL[loc])
        st, sm = marg(loc)
        skew_nested[str(k)] = marginal_report(f"nested {k:,}", st, sm, dom_t, dom_m, SKEW_TOL_PP_NESTED)
    for k in NESTED_SIZES:
        assert np.isin(nested[k], V).all(), f"nested {k} 不是 V 的子集!"
    if len(NESTED_SIZES) == 2:
        assert np.isin(nested[NESTED_SIZES[0]], nested[NESTED_SIZES[1]]).all(), "嵌套关系破坏!"
    log("嵌套关系与偏斜阈值全部通过 ✓")

    # ── 落盘 ──
    loc_all = sel_interleaved
    fidx_V = fidx_P[loc_all]; j_V = j_P[loc_all]
    o = np.argsort(POOL[loc_all], kind="stable")
    fidx_V, j_V = fidx_V[o], j_V[o]
    seq_start = dom48["offsets"][fidx_V] + 500 * j_V
    flag_v1 = np.isin(dom48["tickers"], sorted(V1_TICKERS))[fidx_V]

    np.save(OUTDIR / "val_pool_indices.npy", V)
    np.savez_compressed(OUTDIR / "val_pool_decode.npz",
                        global_idx=V, file_idx=fidx_V.astype(np.int64),
                        seq_idx=j_V.astype(np.int64),
                        seq_start_msg=seq_start.astype(np.int64), flag_v1_8ticker=flag_v1)
    for k, arr in nested.items():
        np.save(OUTDIR / f"val_subset_{k}.npy", arr)
    json.dump([int(x) for x in nested[NESTED_SIZES[0]]],
              open(OUTDIR / f"val_subset_{NESTED_SIZES[0]}.json", "w"))
    with open(OUTDIR / "files_48mo.csv", "w") as f:
        f.write("file_idx,ticker,date,msg_rows,seqs,cum_start,offset\n")
        for i in range(len(dom48["seqs"])):
            f.write(f"{i},{dom48['tickers'][i]},{dom48['dates'][i]},{dom48['rows'][i]},"
                    f"{dom48['seqs'][i]},{dom48['cumsum'][i]},{dom48['offsets'][i]}\n")

    yr = np.array([d[:4] for d in months_of_file])[fidx_V]
    year_share = {y: float((yr == y).mean() * 100) for y in ["2022", "2023", "2024", "2025"]}
    dom_year = {}
    for y in ["2022", "2023", "2024", "2025"]:
        msk = np.array([m.startswith(y) for m in months_of_file])
        dom_year[y] = float(dom48["seqs"][msk].sum() / dom48["seqs"].sum() * 100)

    manifest = dict(
        version=VERSION, built_utc=t0.isoformat(),
        supersedes="valset_v1 (kangoxford/sp500-lob-valset-v1)",
        purpose="Frozen in-distribution validation set for the SP500 **Mamba-3** scaling-law "
                "cohort. Built so that no Mamba-3 run has ever touched any sample. Unlike "
                "valset_v1 it does NOT exclude the 36-month Transformer sub-domain, which is what "
                "made valset_v1 55% weighted on 2022; v2 is unskewed in both month and ticker.",
        scope_warning="This set is held-out for Mamba-3 ONLY. The Transformer cohort trains on the "
                      "36-month (2023-01..2025-12) sub-domain and its consumed region is NOT "
                      "excluded here. Do not use v2 as held-out for Transformer runs.",
        domain=dict(corpus="lob_preproc_sp500 via monthly SquashFS shards", months=MONTHS48,
                    n_tickers=len(tickers488), constituents_csv=CONSTITUENTS_CSV,
                    train_date_range=["2022-01-01", "2025-12-31"], token_mode="26tok",
                    msg_seq_len=500, use_book_data=True, book_transform=True,
                    N48=int(N48), N466=int(dom466["N"]), files48=len(dom48["seqs"])),
        recipe=dict(seeds=SEEDS, tail_frac=TAIL_FRAC, excl_frac=EXCL_FRAC,
                    target_frac=TARGET_FRAC, tail_len=int(T), excl_len_48mo=int(X48),
                    definition="POOL = (∪_s perm_s48[-2%:]) \\ (∪_s perm_s48[:20%]) \\ "
                               "map48(perm42_466[:19200], directional guard) \\ (GOOG × 2025-12); "
                               "V = stratified 0.5% of N48 drawn from POOL by (ticker, month) "
                               "proportional allocation + IPF marginal matching",
                    diff_vs_v1="v1 additionally removed map48(∪_s perm_s36[:20%]); v2 does NOT. "
                               "That term touched only 2023-2025 and caused v1's 2022 skew.",
                    stratification="joint (ticker, month) proportional + 40 IPF iterations to match "
                                   "both marginals to the full-domain proportions; largest-remainder "
                                   "rounding to the exact total; (GOOG, 2025-12) forced to 0 and its "
                                   "quota redistributed within its row and column by IPF",
                    subset_seed=SUBSET_SEED,
                    nesting="stratified interleaved order: key = (rank_in_cell + 0.5)/cell_size, so "
                            "every prefix is proportional across all cells",
                    torch_version=torch.__version__, numpy_version=np.__version__,
                    python=sys.version.split()[0]),
        consumed_caps_samples_48mo={str(k): int(v) for k, v in CONSUMED_CAPS.items()},
        counts=dict(tail_union=n_tail_union, POOL=int(len(POOL)), V=int(len(V)),
                    V_pct_of_N=float(len(V) / N48 * 100),
                    messages=int(len(V)) * 500, tokens=int(len(V)) * 13000,
                    nested={str(k): int(len(v)) for k, v in nested.items()}),
        skew_audit=dict(main_max_dev_pp=skew_main, nested_max_dev_pp=skew_nested,
                        tolerance_pp=dict(main=SKEW_TOL_PP_MAIN, nested=SKEW_TOL_PP_NESTED),
                        year_share_val=year_share, year_share_domain=dom_year),
        cross_domain_treatment=dict(
            tk466=dict(runs=["oxford-lob/mamba3-squashfs-multi/8plu95a3"],
                       samples=int(C466_SAMPLES), dropped_from_V0=int(drop466.sum())),
            goog_dec=dict(runs=["oxford-lob/neurips-mamba3-finetune/arq1lyt0 (epochs=2)",
                                 "oxford-lob/neurips-mamba3-finetune/hxcamslh"],
                          rule="excise entire (GOOG, 2025-12) slice",
                          dropped_from_V0=int(drop_goog.sum()),
                          note="the only structural hole; IPF redistributes its quota so both "
                               "marginals stay exact")),
        flags=dict(v1_8ticker=dict(tickers=sorted(V1_TICKERS), n_flagged=int(flag_v1.sum()))),
        future_budget=dict(rule="Mamba-3 training with seeds {5,42,137} on the 48-month domain must "
                                "keep per-seed total steps ≤ budget_48mo_steps.",
                           budget_48mo_steps=int(X48 // 128),
                           used_48mo={"5": 420_000, "42": 168_200, "137": 106_909}),
    )
    json.dump(manifest, open(OUTDIR / "manifest.json", "w"), indent=2, ensure_ascii=False)
    with open(OUTDIR / "SHA256SUMS.txt", "w") as f:
        for p in sorted(OUTDIR.iterdir()):
            if p.name != "SHA256SUMS.txt":
                f.write(f"{sha256(p)}  {p.name}\n")

    log("==== SUMMARY ====")
    log(f"{VERSION}: V={len(V):,} ({len(V)/N48*100:.4f}% of N48) from POOL={len(POOL):,}")
    log(f"year share  val: {year_share}")
    log(f"year share  dom: {dom_year}")
    log(f"skew main (pp): {skew_main}")
    print("BUILD_VALSET_V2_OK", flush=True)

if __name__ == "__main__":
    main()
