#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_valset.py — 构建永久固定的 SP500 scaling-law validation dataset（valset_v1）。

配方（用户指定，Notion validation-set 页）+ 双域对称扩展：
  【48mo 主域】对 3 个 shuffle seed s ∈ {5,42,137}（= Mamba3 v3 主矩阵 JAX_SEED，
  经代码链验证 JAX_SEED 即 torch DistributedSampler 的 shuffle seed，epoch=0）：
    perm_s = torch.randperm(N48, manual_seed(s))            # 与训练逐位一致
    tail_s = perm_s[-floor(0.02*N48):]                       # last 2%（该 seed 从未消费）
    zone_s = perm_s[:ceil(0.20*N48)]                         # first 20% 排除区
  V0 = (∪ tail_s) − (∪ zone_s)
  【36mo 对称排除】TF 矩阵正在 36 个月域（2023-2025 × 488tk）重跑（squeue 证据：
  tf-0p2M-s{5,42,137} pending, TF sweep 默认 36mo，curtail ≤65,664 步）。为使 val set
  对整个 TF 计划永久免疫，同样排除 perm_s36[:20%*N36] (s∈{5,42,137}) 的 48mo 映射像，
  并加方向性邻窗守卫（两域逐文件 offset 均为确定性常量但取值不同，错位方向逐文件已知，
  每个 36mo 窗口只与 48mo 侧 2 个窗口有消息重叠）。历史 TF/O2d 消费（≤49,590 步）⊂ 该区。
  【466tk 精确减除】squashfs-pilot 8plu95a3（466tk × 48mo 域，seed 42，300 步 × gBSZ 64）。
  【GOOG-2025-12 整片切除】finetune arq1lyt0/hxcamslh（epochs=2 消费整片）。
  【v1 8-ticker 旗标】GOOG/AAPL/NVDA/AMZN/META/TSLA/MSFT/AMD 标记不剔除（8-ticker 旧语料
  时代实验在消息层面接触过这些 ticker 的原始数据）。

产物（ARTIFACTS_DIR）：val_pool_indices.npy、嵌套子集（30,720 / 307,200 / 1%N48）、
decode 表、files 表、manifest.json、SHA256SUMS.txt。
"""
import os, sys, json, gc, math, hashlib, re
from datetime import datetime, timezone
from pathlib import Path

EXP_DIR = "/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_R1_Mamba3"
CONSTITUENTS_CSV = "/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/scaling_law_plots/snp500_constituents_20260131.csv"
PILOT_TICKERS_JSON = "/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/pilot_8plu95a3_tickers.json"
RAW_ROOT_LABEL = "/lus/lfs1aip2/projects/public/s5e/quant_team/lob_preproc_sp500"

SEEDS = [5, 42, 137]
TAIL_FRAC = 0.02
EXCL_FRAC = 0.20
# 已核实的 per-seed 实际消费上限（W&B 全项目扫描 2026-07-29，样本数 = 步数 × gBSZ128）
CONSUMED_CAPS = {
    5:   420_000 * 128,                          # 6M-700B 链 curtail（实际 419,930 步）
    42:  max(106_909 * 128, 168_200 * 128),      # 主矩阵 curtail vs full-d 链实停 168,200 步
    137: 106_909 * 128,                          # 主矩阵 curtail（实际 106,740 步）
}
C36_HIST_SAMPLES = 49_590 * 128     # 历史 36mo 消费上限（O2d curtail ⊇ TF 7438*128），须 ⊂ 20% 区
C466_SAMPLES = 300 * 64             # pilot 8plu95a3：300 步 × gBSZ 16*4*1=64，seed 42
V1_TICKERS = {"GOOG", "AAPL", "NVDA", "AMZN", "META", "TSLA", "MSFT", "AMD"}
GOOG_EXCISE = ("GOOG", "2025-12")   # finetune 整片切除
SUBSET_SEED = 20260729
SUBSET_SIZES = [30_720, 307_200]    # 另加 floor(0.01*N48)，运行时补
# 独立日志证据（硬断言）
ASSERT_N48_DIV8 = 40_402_673        # j4531958: rank=0/8 samples_per_node
ASSERT_N36_DIV2 = 122_000_461       # O2d pilot j4567612: rank=0/2 samples_per_node

MONTHS48 = [f"{y}-{m:02d}" for y in range(2022, 2026) for m in range(1, 13)]
MONTHS36 = [f"{y}-{m:02d}" for y in range(2023, 2026) for m in range(1, 13)]

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
    tickers = []
    with open(path) as f:
        for i, line in enumerate(f):
            first = line.split(",")[0].strip()
            if not first:
                continue
            if i == 0 and first.lower().startswith("ticker"):
                continue
            tickers.append(first)
    return tickers

def build_domain(tag, months, tickers, date_range):
    """用与生产 run 完全相同的工厂代码路径重建数据集，抽出元数据。"""
    from lob.dataloading import create_lobster_prediction_dataset
    from lob.lobster_dataloader import LOBSTER_Dataset
    data_root = ",".join(str(MOUNT_ROOT / m) for m in months)
    log(f"[{tag}] building dataset: {len(months)} months, {len(tickers)} tickers, range={date_range}")
    ret = create_lobster_prediction_dataset(
        cache_dir=RAW_ROOT_LABEL,
        seed=42,
        mask_fn=LOBSTER_Dataset.no_mask,
        msg_seq_len=500,
        micro_bsz=1,
        num_devices=1,
        use_book_data=True,
        use_simple_book=False,
        book_transform=True,
        book_ablation="real",
        book_depth=500,
        n_data_workers=0,
        shuffle_train=True,
        rand_offset=True,
        debug_overfit=False,
        val_split=0.0,
        test_split=0.0,
        pin_memory=False,
        prefetch_factor=None,
        persistent_workers=False,
        test_dir_name=None,
        use_distributed_sampler=False,
        process_rank=0,
        process_count=1,
        tickers=list(tickers),
        data_root=data_root,
        train_date_range=list(date_range),
        test_date_range=None,
        token_mode="26tok",
    )
    train = ret[0].dataset_train
    N = len(train)
    files = [str(p) for p in train.message_files]
    tk = np.array([Path(p).parent.name for p in files])
    dt = np.array([(DATE_RE.search(Path(p).name).group(1) if DATE_RE.search(Path(p).name) else "?") for p in files])
    seqs = np.asarray(train._seqs_per_file, dtype=np.int64)
    cumsum = np.asarray(train._seqs_cumsum, dtype=np.int64)
    offsets = train.seq_offsets.numpy().astype(np.int64).copy()
    rows = np.asarray(train._num_rows_per_file, dtype=np.int64)
    log(f"[{tag}] N={N:,} files={len(files):,} seqs.sum={int(seqs.sum()):,}")
    assert int(cumsum[-1]) == N
    del ret, train
    gc.collect()
    return dict(tag=tag, N=N, tickers=tk, dates=dt, seqs=seqs, cumsum=cumsum,
                offsets=offsets, rows=rows, files=files)

def perm_for(N, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randperm(N, generator=g).numpy().astype(np.int64)

def verify_sampler_equivalence(N, seed=42, R=8, k=1000):
    """torch DistributedSampler(shuffle=True, seed, drop_last=True, epoch0) 的 rank-0 流
    必须等于 perm[0::R] 前缀——证明本脚本的 perm 与训练消费顺序同源。"""
    from torch.utils.data import DistributedSampler
    class Dummy:
        def __len__(self):
            return N
    s = DistributedSampler(Dummy(), num_replicas=R, rank=0, shuffle=True, seed=seed, drop_last=True)
    s.set_epoch(0)
    it = iter(s)
    got = np.array([next(it) for _ in range(k)], dtype=np.int64)
    p = perm_for(N, seed)
    ref = p[0::R][:k]
    assert np.array_equal(got, ref), "DistributedSampler ≠ randperm 复现，中止！"
    log(f"sampler equivalence OK (N={N:,}, R={R}, k={k})")

def foreign_domain_maps(dom_small, dom48, fidx_V0, j_V0):
    """把 V0 每个样本 (48mo file, j) 映射到小域的全局索引 m 与方向性邻窗 m2。

    返回 (valid, m, m2, has_m2)。两域逐文件 offset 均为确定性常量：
    off_small > off_48 时小域窗口 j 覆盖的消息与 48mo 侧窗口 {j, j+1} 相交，反向则 {j, j-1}，
    相等则只 {j}。因此 48mo 样本 j 只可能与小域窗口 {j, j-dir} 重叠（dir 按文件恒定）。
    """
    key48 = {}
    for f in range(len(dom48["seqs"])):
        key48[(dom48["tickers"][f], dom48["dates"][f])] = f
    nf_s = len(dom_small["seqs"])
    base = np.full(len(dom48["seqs"]), -1, dtype=np.int64)   # 48file → small 域 cum start
    nseq = np.zeros(len(dom48["seqs"]), dtype=np.int64)
    ddir = np.zeros(len(dom48["seqs"]), dtype=np.int64)
    matched, mismatched_seqs = 0, 0
    for fs in range(nf_s):
        key = (dom_small["tickers"][fs], dom_small["dates"][fs])
        f48 = key48.get(key)
        if f48 is None:
            continue
        matched += 1
        if dom_small["seqs"][fs] != dom48["seqs"][f48]:
            mismatched_seqs += 1
        base[f48] = dom_small["cumsum"][fs]
        nseq[f48] = dom_small["seqs"][fs]
        d = int(dom_small["offsets"][fs]) - int(dom48["offsets"][f48])
        ddir[f48] = 0 if d == 0 else (1 if d > 0 else -1)
    log(f"[{dom_small['tag']}] file join: matched={matched:,}/{nf_s:,} seq-mismatch={mismatched_seqs}")
    assert mismatched_seqs == 0, "同一文件在两域的窗口数不一致——书页配对逻辑漂移!"
    b = base[fidx_V0]
    valid = (b >= 0) & (j_V0 < nseq[fidx_V0])
    m = np.where(valid, b + j_V0, 0)
    # 小域侧与 48mo 窗口 j 有消息重叠的第二个窗口: j - dir（在小域索引里 = m - dir）
    d2 = ddir[fidx_V0]
    m2 = m - d2
    has_m2 = valid & (d2 != 0) & (m2 >= b) & (m2 < b + nseq[fidx_V0])
    return valid, m, np.where(has_m2, m2, 0), has_m2

def drop_by_zone(tag, N_small, seeds_and_thresholds, valid, m, m2, has_m2):
    """逐 seed 建逆排列，检查 m / m2 是否落在该 seed 的排除前缀内。O(1)/样本。"""
    drop = np.zeros(len(valid), dtype=bool)
    for s, X in seeds_and_thresholds:
        p = perm_for(N_small, s)
        inv = np.empty(N_small, dtype=np.int64)
        inv[p] = np.arange(N_small, dtype=np.int64)
        del p; gc.collect()
        hit = valid & (inv[m] < X)
        hit |= has_m2 & (inv[m2] < X)
        drop |= hit
        log(f"[{tag}] seed {s} zone<{X:,}: +{int(hit.sum()):,} dropped (cum {int(drop.sum()):,})")
        del inv; gc.collect()
    return drop

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now(timezone.utc)
    tickers488 = read_constituents(CONSTITUENTS_CSV)
    tickers466 = json.load(open(PILOT_TICKERS_JSON))
    log(f"tickers: 488-list={len(tickers488)}, pilot-list={len(tickers466)}")

    dom48 = build_domain("48mo", MONTHS48, tickers488, ("2022-01-01", "2025-12-31"))
    N48 = dom48["N"]
    assert N48 // 8 == ASSERT_N48_DIV8, f"N48={N48} 与 8N 日志 samples_per_node 不符!"
    dom36 = build_domain("36mo", MONTHS36, tickers488, ("2023-01-01", "2025-12-31"))
    N36 = dom36["N"]
    assert N36 // 2 == ASSERT_N36_DIV2, f"N36={N36} 与 O2d 2N 日志不符!"
    dom466 = build_domain("466tk", MONTHS48, tickers466, ("2022-01-01", "2026-02-15"))

    verify_sampler_equivalence(N48)

    X48 = math.ceil(EXCL_FRAC * N48)
    X36 = math.ceil(EXCL_FRAC * N36)
    T = math.floor(TAIL_FRAC * N48)
    for s, cap in CONSUMED_CAPS.items():
        assert cap < X48, f"seed {s} 消费 {cap:,} ≥ 48mo 排除区 {X48:,}!"
    assert C36_HIST_SAMPLES < X36, "历史 36mo 消费超出 20% 区!"
    log(f"N48={N48:,} N36={N36:,} X48={X48:,} X36={X36:,} T={T:,}")

    # ── 主配方（48mo 域）：尾巴并集 − 排除区并集 ──
    excluded = np.zeros(N48, dtype=bool)
    tails = []
    for s in SEEDS:
        p = perm_for(N48, s)
        excluded[p[:X48]] = True
        tails.append(p[N48 - T:].copy())
        del p; gc.collect()
        log(f"seed {s}: 48mo zone+tail done")
    tail_union = np.unique(np.concatenate(tails))
    del tails; gc.collect()
    V0 = tail_union[~excluded[tail_union]]
    n_tail_union = int(len(tail_union))
    del tail_union, excluded; gc.collect()
    log(f"tail union={n_tail_union:,}  V0 (recipe pool)={len(V0):,} ({len(V0)/N48*100:.3f}% of N)")

    fidx_V0 = np.searchsorted(dom48["cumsum"], V0, side="right") - 1
    j_V0 = V0 - dom48["cumsum"][fidx_V0]

    # ── 36mo 域对称 20% 排除（TF 重跑免疫）──
    v36, m36, m36b, h36 = foreign_domain_maps(dom36, dom48, fidx_V0, j_V0)
    drop36 = drop_by_zone("36mo", N36, [(s, X36) for s in SEEDS], v36, m36, m36b, h36)
    # ── 466tk 域精确前缀减除 ──
    v466, m466, m466b, h466 = foreign_domain_maps(dom466, dom48, fidx_V0, j_V0)
    drop466 = drop_by_zone("466tk", dom466["N"], [(42, C466_SAMPLES)], v466, m466, m466b, h466)

    # ── GOOG × 2025-12 整片切除 + v1 8-ticker 旗标 ──
    file_goog_dec = (dom48["tickers"] == GOOG_EXCISE[0]) & \
                    np.char.startswith(dom48["dates"].astype(str), GOOG_EXCISE[1])
    drop_goog = file_goog_dec[fidx_V0]
    file_v1 = np.isin(dom48["tickers"], sorted(V1_TICKERS))

    keep = ~(drop36 | drop466 | drop_goog)
    V = V0[keep]
    fidx_V = fidx_V0[keep]
    j_V = j_V0[keep]
    flag_v1 = file_v1[fidx_V]
    log(f"drops: 36mo-20%zone={int(drop36.sum()):,} 466tk={int(drop466.sum()):,} "
        f"goog-dec={int(drop_goog.sum()):,} → V={len(V):,} ({len(V)/N48*100:.3f}% of N)")

    # ── 硬验证：V 对每个 seed 的 48mo first-20% 区零交集，且每成员属于 ≥1 个 tail ──
    in_any_tail = np.zeros(len(V), dtype=bool)
    for s in SEEDS:
        p = perm_for(N48, s)
        inv = np.empty(N48, dtype=np.int64)
        inv[p] = np.arange(N48, dtype=np.int64)
        del p; gc.collect()
        pos = inv[V]
        assert int(pos.min()) >= X48, f"seed {s}: V 中有样本位于 first-20% 区!"
        assert int(pos.min()) >= CONSUMED_CAPS[s], f"seed {s}: V 与实际消费区相交!"
        in_any_tail |= (pos >= N48 - T)
        del inv, pos; gc.collect()
    assert bool(in_any_tail.all()), "V 中有样本不属于任何 seed 的 last-2%!"
    log("hard verification passed: V ∩ (∪ 48mo first-20%) = ∅, V ⊆ ∪ last-2%")

    # ── 嵌套固定子集 ──
    rng = np.random.default_rng(SUBSET_SEED)
    order = rng.permutation(len(V))
    sizes = SUBSET_SIZES + [int(0.01 * N48)]
    subsets = {}
    for sz in sizes:
        assert sz <= len(V), f"子集 {sz:,} > 池 {len(V):,}"
        subsets[sz] = np.sort(V[order[:sz]])

    # ── 落盘 ──
    seq_start = dom48["offsets"][fidx_V] + 500 * j_V
    np.save(OUTDIR / "val_pool_indices.npy", V)
    np.savez_compressed(OUTDIR / "val_pool_decode.npz",
                        global_idx=V, file_idx=fidx_V.astype(np.int64),
                        seq_idx=j_V.astype(np.int64),
                        seq_start_msg=seq_start.astype(np.int64),
                        flag_v1_8ticker=flag_v1)
    for sz, arr in subsets.items():
        np.save(OUTDIR / f"val_subset_{sz}.npy", arr)
    json.dump([int(x) for x in subsets[30_720]], open(OUTDIR / "val_subset_30720.json", "w"))
    with open(OUTDIR / "files_48mo.csv", "w") as f:
        f.write("file_idx,ticker,date,msg_rows,seqs,cum_start,offset\n")
        for i in range(len(dom48["seqs"])):
            f.write(f"{i},{dom48['tickers'][i]},{dom48['dates'][i]},{dom48['rows'][i]},"
                    f"{dom48['seqs'][i]},{dom48['cumsum'][i]},{dom48['offsets'][i]}\n")

    manifest = dict(
        version="valset_v1",
        built_utc=t0.isoformat(),
        purpose="Frozen in-distribution validation set for the SP500 Mamba3/Transformer "
                "scaling-law cohort (NeurIPS 2026 rebuttal). No training run has touched any "
                "sample; immune by construction to the in-flight TF matrix rerun on the 36mo "
                "domain (per-seed budget 20% of N36).",
        domain=dict(
            corpus="lob_preproc_sp500 via monthly SquashFS shards",
            months=MONTHS48, n_tickers=len(tickers488),
            constituents_csv=CONSTITUENTS_CSV,
            train_date_range=["2022-01-01", "2025-12-31"],
            token_mode="26tok", msg_seq_len=500, use_book_data=True, book_transform=True,
            N48=int(N48), N36=int(N36), N466=int(dom466["N"]),
            files48=len(dom48["seqs"]),
        ),
        recipe=dict(
            seeds=SEEDS, tail_frac=TAIL_FRAC, excl_frac=EXCL_FRAC,
            tail_len=int(T), excl_len_48mo=int(X48), excl_len_36mo=int(X36),
            definition="V = (∪_s perm_s48[-2%:]) \\ (∪_s perm_s48[:20%]) \\ "
                       "map48(∪_s perm_s36[:20%], directional ±1 guard) \\ "
                       "map48(perm42_466[:19200], directional guard) \\ (GOOG × 2025-12)",
            torch_version=torch.__version__, numpy_version=np.__version__,
            python=sys.version.split()[0],
            env="/lus/lfs1aip2/projects/public/s5e/quant_team/quant/miniforge3 (training env)",
        ),
        consumed_caps_samples_48mo={str(k): int(v) for k, v in CONSUMED_CAPS.items()},
        consumed_caps_note="seed5=6M-700B chain curtail 420000*128 (16.6%N); seed42=full-d chain "
                           "observed 168200*128 (6.7%N); seed137=primary curtail 106909*128 "
                           "(4.2%N). All < 20%*N48. 36mo historical (TF xbuaya9r + O2d cqo9qoit, "
                           f"both seed5) = {C36_HIST_SAMPLES:,} ⊂ 20%*N36 zone.",
        cross_domain_treatment=dict(
            mo36=dict(rule="exclude map48(perm_s36[:ceil(0.20*N36)]) for s in {5,42,137} with "
                           "directional neighbor guard",
                      reason="TF matrix rerun in-flight on 36mo domain (squeue 2026-07-29: "
                             "tf-0p2M-s{5,42,137} pending; TF sweep default months 2023-01..2025-12, "
                             "max curtail 65,664 steps = 3.4% N36 << 20%)",
                      dropped_from_V0=int(drop36.sum())),
            tk466=dict(runs=["oxford-lob/mamba3-squashfs-multi/8plu95a3"],
                       samples=int(C466_SAMPLES), dropped_from_V0=int(drop466.sum())),
            goog_dec=dict(runs=["oxford-lob/neurips-mamba3-finetune/arq1lyt0 (epochs=2)",
                                 "oxford-lob/neurips-mamba3-finetune/hxcamslh"],
                          rule="excise entire (GOOG, 2025-12) slice",
                          dropped_from_V0=int(drop_goog.sum())),
        ),
        flags=dict(v1_8ticker=dict(tickers=sorted(V1_TICKERS),
                                    n_flagged=int(flag_v1.sum()),
                                    note="8-ticker-corpus era experiments (v1 sweep, phase-b-*, "
                                         "R1/O2 lines, GOOG projects) touched these tickers' raw "
                                         "messages under a different preproc. Flag only; do NOT "
                                         "evaluate 8-ticker-trained models on flagged samples.")),
        counts=dict(tail_union=n_tail_union, V0_recipe=int(len(V0)), V_final=int(len(V)),
                    V_pct_of_N=float(len(V) / N48 * 100),
                    messages=int(len(V)) * 500, tokens=int(len(V)) * 13000,
                    subsets={str(k): int(len(v)) for k, v in subsets.items()},
                    subset_seed=SUBSET_SEED,
                    subset_note="nested: subset_30720 ⊂ subset_307200 ⊂ subset_1pct (prefixes of "
                                "one fixed shuffle of V)"),
        guarantees=[
            "Index-exact (48mo): V ∩ perm_s48[:20%] = ∅ for all s ∈ {5,42,137}; every production "
            "run consumed a strict prefix ≤ curtail*128 ≤ 20%*N48 (verified against W&B step "
            "records of all runs incl. crashed, live scan 2026-07-29).",
            "Message-exact within domain: per-file random offsets come from a fixed-seed RNG "
            "independent of JAX_SEED (lobster_dataloader init_defaults seed=42), so (file,j) maps "
            "to identical message ranges in every run and future eval.",
            "36mo-immune: V excludes the 48mo images (with directional neighbor guard) of all "
            "three 36mo 20% zones, covering the historical TF/O2d prefix (≤49,590 steps) and any "
            "future 36mo run up to 381,251 steps per seed.",
            "(GOOG, 2025-12) excised entirely (finetune consumed the slice over 2 epochs).",
        ],
        residual_risks=[
            "April raw-tree era short runs (TF/M3 config-iteration batches ~618 steps, dmon FLOPs "
            "profile jobs ~300 steps, pre-SquashFS smokes) ran on unreconstructable file orderings; "
            "expected ≤0.3% of V could share messages. All were discarded scaffolding models.",
            "squashfs-pilot half-month run p1qqvt0i (54 steps, 473tk half-month domain) and 4 "
            "squashfs-multi 4-month runs (lastStep=1): expected <1e-4 of V.",
            "O2d pilot runs (4x310 steps, 36mo domain, W&B records deleted): covered by the 36mo "
            "20% zone if seeds ∈ {5,42,137}; an unknown other seed would contribute ~0.05% of V.",
        ],
        future_budget=dict(
            rule="Future training with seeds {5,42,137} must keep per-seed total steps ≤ budget; "
                 "training on any NEW index domain over this corpus requires a new purity audit.",
            budget_48mo_steps=int(X48 // 128), used_48mo={"5": 420_000, "42": 168_200, "137": 106_909},
            budget_36mo_steps=int(X36 // 128), used_36mo={"5": 49_590, "42": "pending TF rerun",
                                                           "137": "pending TF rerun"},
        ),
        evidence=dict(
            wandb_scans=["scratchpad wandb_post_may23_scan.txt", "wandb_scan2.txt (2026-07-29)"],
            log_anchors={"j4531958_8N_samples_per_node": ASSERT_N48_DIV8,
                          "j4567612_2N_samples_per_node_36mo": ASSERT_N36_DIV2},
            squeue_snapshot="2026-07-29: tf-0p2M-s{5,42,137} PENDING (jobs 5824382/85/88)",
        ),
    )
    json.dump(manifest, open(OUTDIR / "manifest.json", "w"), indent=2, ensure_ascii=False)

    with open(OUTDIR / "SHA256SUMS.txt", "w") as f:
        for p in sorted(OUTDIR.iterdir()):
            if p.name == "SHA256SUMS.txt":
                continue
            f.write(f"{sha256(p)}  {p.name}\n")

    log("==== SUMMARY ====")
    log(f"N48={N48:,}  N36={N36:,}  N466={dom466['N']:,}")
    log(f"V0={len(V0):,}  V={len(V):,} ({len(V)/N48*100:.3f}% of N48) "
        f"= {len(V)*500/1e6:.1f}M msgs = {len(V)*13000/1e9:.2f}B tokens")
    log(f"subsets: {[len(v) for v in subsets.values()]}  flagged_v1_8tk={int(flag_v1.sum()):,}")
    log(f"artifacts → {OUTDIR}")
    print("BUILD_VALSET_OK", flush=True)

if __name__ == "__main__":
    main()
