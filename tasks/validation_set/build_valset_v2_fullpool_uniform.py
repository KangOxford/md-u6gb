#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_valset_v2_fullpool_uniform.py — 构建 valset_v2（uniform full pool）。

与 build_valset.py（v1）的**唯一配方差异**：关闭 36 个月子域的 20% 排除项（APPLY_36MO_EXCLUSION=False）。
其余每一行——排列、尾巴、48mo 排除区、466tk 前缀减除、GOOG 整片切除、硬验证——逐字不变。

为什么这一步是 v1 年份偏斜的唯一成因：36mo 子域按定义只覆盖 2023-01..2025-12，所以这一刀
只能砍 2023 年以后，2022 原封不动。v1 实测留存率 2022 = 3.744% 对 2023-25 = 0.984%，
比值 3.80 = 1/0.8^6（3 seed × 2 邻窗守卫 = 6 次独立 20% 裁剪），于是 v1 里 2022 占 55.25%
而全域只有 24.51%，偏差 30.7 个百分点。删掉这一步，48 个月留存率一致，偏斜从根上消失。

**关键性质**：步骤 1（尾巴并集）与步骤 2（48mo 排除区）都是均匀随机排列的前缀/后缀操作，
年份中性。因此 V2 是全域的**均匀随机子集**，天然在**所有**边缘分布上与训练分布一致
（年份、月份、ticker、星期、波动率区制……），而不只是被显式配平的那几维。
这比事后按年份配平严格更强：配平只能保证你想到要配的维度。

**代价（必须写进文档）**：v2 对 Transformer 队列**不保证** held-out。TF 在 36 个月域上训练，
其消费区未被排除。v2 只服务 Mamba-3（三个 seed 全部在 48mo 域上训练，最深消费 16.63% < 20%）。
需要对两支队列都安全的尺子时，继续用 valset_v1。

规模：不做任何二次抽样，保留步骤 2 之后的**全部**样本（预计约 1.21e7，占 N48 约 3.74%）。

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
SUBSET_SEED = 20260803              # v2 独立种子，避免与 v1 的 20260729 产生相同抽样
SUBSET_SIZES = [30_720, 307_200]    # 另加 floor(0.01*N48)，运行时补

# ── v2 唯一的配方开关 ──
# False = 不排除 36mo 子域的 20% 区 = 不引入年份偏斜 = 对 TF 不再免疫。
APPLY_36MO_EXCLUSION = False
VERSION = "valset_v2_uniform_fullpool"
FILE_PREFIX = "valset_v2_uniform"
# 验收阈值
TOL_YEAR_RETENTION_RATIO = 1.02     # max(留存率)/min(留存率)，v1 此处是 3.80
TOL_YEAR_SHARE_PP = 0.05            # 逐年份额与 epoch 的绝对偏差上限（百分点）
TOL_MONTH_RATIO = 1.05              # 48 个月 pool/epoch ratio 的 max/min
TOL_KISH_REL = 0.05                 # ticker Kish n_eff 与 epoch 的相对偏差上限
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

    # ── 36mo 域对称 20% 排除（TF 重跑免疫）── v2: 关闭 ──
    if APPLY_36MO_EXCLUSION:
        v36, m36, m36b, h36 = foreign_domain_maps(dom36, dom48, fidx_V0, j_V0)
        drop36 = drop_by_zone("36mo", N36, [(s, X36) for s in SEEDS], v36, m36, m36b, h36)
    else:
        drop36 = np.zeros(len(V0), dtype=bool)
        log("36mo exclusion DISABLED (v2): validation set is NOT held-out for the "
            "Transformer queue; it is uniform over the 48mo domain and serves Mamba-3 only.")
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
    log(f"drops: 36mo-20%zone={int(drop36.sum()):,} (0 = disabled in v2) "
        f"466tk={int(drop466.sum()):,} "
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

    # ── v2 验收闸门：均匀性必须实测通过，否则中止（不写盘）──
    def _kish(counts):
        w = np.asarray(counts, dtype=float); w = w[w > 0]
        return float((w.sum() ** 2) / (w ** 2).sum())

    years_f = np.array([d[:4] for d in dom48["dates"].astype(str)])
    months_f = np.array([d[:7] for d in dom48["dates"].astype(str)])
    # epoch 参照必须扣掉 (GOOG, 2025-12)：该整片按配方被强制切除，v2 在构造上不可能含有它，
    # 拿含它的 epoch 作分母是苹果对橘子。首轮构建即因此在 GATE2 上失败（实测 0.0580 pp，
    # 其中 10,377/87,969,827 = 0.0118 pp 的 2025 留存率缺口完全由该切除解释）。
    # 注意这与「放宽阈值」不同：放宽会掩盖结构，修正参照才让闸门保持精确。
    seqs_f = np.where(file_goog_dec, 0, np.asarray(dom48["seqs"], dtype=np.int64))
    seqs_raw = np.asarray(dom48["seqs"], dtype=np.int64)

    def _by(keys_f, idx_f):
        ks = sorted(set(keys_f.tolist()))
        ep = {k: int(seqs_f[keys_f == k].sum()) for k in ks}
        po = {k: int((keys_f[idx_f] == k).sum()) for k in ks}
        return ks, ep, po

    yrs, ep_y, po_y = _by(years_f, fidx_V)
    ret = {y: po_y[y] / ep_y[y] for y in yrs}
    ratio_y = max(ret.values()) / min(ret.values())
    ep_yN = sum(ep_y.values()); po_yN = sum(po_y.values())
    dev_pp = {y: abs(100 * po_y[y] / po_yN - 100 * ep_y[y] / ep_yN) for y in yrs}

    mos, ep_m, po_m = _by(months_f, fidx_V)
    ret_m = {m: po_m[m] / ep_m[m] for m in mos if ep_m[m] > 0}
    ratio_m = max(ret_m.values()) / min(ret_m.values())

    ep_t = np.array([int(seqs_f[dom48["tickers"] == t].sum())
                     for t in sorted(set(dom48["tickers"].tolist()))])
    po_t = np.bincount(np.unique(dom48["tickers"], return_inverse=True)[1][fidx_V])
    kish_ep, kish_po = _kish(ep_t), _kish(po_t)
    kish_rel = abs(kish_po - kish_ep) / kish_ep

    log("==== v2 acceptance gates ====")
    log(f"  epoch reference excludes (GOOG, 2025-12): "
        f"{int(seqs_raw.sum()) - int(seqs_f.sum()):,} windows removed from the denominator")
    log(f"  per-year retention: " + "  ".join(f"{y}={100*ret[y]:.4f}%" for y in yrs))
    log(f"  GATE1 retention max/min = {ratio_y:.4f}  (need < {TOL_YEAR_RETENTION_RATIO})")
    log(f"  GATE2 max |year share - epoch| = {max(dev_pp.values()):.4f} pp "
        f"(need < {TOL_YEAR_SHARE_PP})")
    log(f"  GATE3 month ratio max/min = {ratio_m:.4f}  (need < {TOL_MONTH_RATIO})")
    log(f"  GATE4 ticker Kish n_eff pool={kish_po:.2f} epoch={kish_ep:.2f} "
        f"rel={kish_rel:.4f} (need < {TOL_KISH_REL})")
    assert ratio_y < TOL_YEAR_RETENTION_RATIO, f"GATE1 FAIL: 逐年留存率不齐 {ratio_y:.4f}"
    assert max(dev_pp.values()) < TOL_YEAR_SHARE_PP, f"GATE2 FAIL: 年份份额偏离 epoch"
    assert ratio_m < TOL_MONTH_RATIO, f"GATE3 FAIL: 逐月留存率不齐 {ratio_m:.4f}"
    assert kish_rel < TOL_KISH_REL, f"GATE4 FAIL: ticker 集中度偏离 epoch"
    for sz, arr in subsets.items():
        fi = np.searchsorted(dom48["cumsum"], arr, side="right") - 1
        d = max(abs(100 * (years_f[fi] == y).sum() / len(arr) - 100 * ep_y[y] / ep_yN)
                for y in yrs)
        nt = len(set(dom48["tickers"][fi].tolist()))
        log(f"  subset {sz:>9,}: max |year share - epoch| = {d:.4f} pp, tickers = {nt}")
    log("  ALL GATES PASSED")

    distribution = dict(
        epoch_year_share={y: 100 * ep_y[y] / ep_yN for y in yrs},
        pool_year_share={y: 100 * po_y[y] / po_yN for y in yrs},
        per_year_retention_pct={y: 100 * ret[y] for y in yrs},
        year_retention_ratio_max_over_min=float(ratio_y),
        max_year_share_dev_pp=float(max(dev_pp.values())),
        epoch_month_share={m: 100 * ep_m[m] / ep_yN for m in mos},
        pool_month_share={m: 100 * po_m[m] / po_yN for m in mos},
        month_retention_ratio_max_over_min=float(ratio_m),
        ticker_kish_n_eff_pool=kish_po, ticker_kish_n_eff_epoch=kish_ep,
        n_tickers_pool=int((po_t > 0).sum()),
        epoch_reference="48mo domain MINUS the (GOOG, 2025-12) slice, which the recipe "
                        "excises and v2 therefore cannot contain",
        epoch_reference_windows_removed=int(seqs_raw.sum()) - int(seqs_f.sum()),
        note="v1 for comparison: year shares 55.25/13.51/15.10/16.13, retention ratio 3.80, "
             "ticker Kish n_eff 163.19 vs epoch 127.56.",
    )

    # ── 落盘 ──
    seq_start = dom48["offsets"][fidx_V] + 500 * j_V
    np.save(OUTDIR / f"{FILE_PREFIX}_pool_indices.npy", V)
    np.savez_compressed(OUTDIR / f"{FILE_PREFIX}_pool_decode.npz",
                        global_idx=V, file_idx=fidx_V.astype(np.int64),
                        seq_idx=j_V.astype(np.int64),
                        seq_start_msg=seq_start.astype(np.int64),
                        flag_v1_8ticker=flag_v1)
    for sz, arr in subsets.items():
        np.save(OUTDIR / f"{FILE_PREFIX}_subset_{sz}.npy", arr)
    json.dump([int(x) for x in subsets[30_720]],
              open(OUTDIR / f"{FILE_PREFIX}_subset_30720.json", "w"))
    with open(OUTDIR / "files_48mo.csv", "w") as f:
        f.write("file_idx,ticker,date,msg_rows,seqs,cum_start,offset\n")
        for i in range(len(dom48["seqs"])):
            f.write(f"{i},{dom48['tickers'][i]},{dom48['dates'][i]},{dom48['rows'][i]},"
                    f"{dom48['seqs'][i]},{dom48['cumsum'][i]},{dom48['offsets'][i]}\n")

    manifest = dict(
        version=VERSION,
        built_utc=t0.isoformat(),
        purpose="Frozen UNIFORM validation set for the SP500 Mamba-3 48-month scaling-law "
                "cohort. Identical to valset_v1 except the 36-month sub-domain exclusion is "
                "DISABLED, which removes the year skew at its source. No Mamba-3 training run "
                "has touched any sample. NOT held-out for the Transformer queue: use valset_v1 "
                "for that.",
        applicability=dict(
            valid_for=["Mamba-3 48-month cohort (seeds 5/42/137)"],
            NOT_valid_for=["Transformer / any run on the 36-month (2023-2025) domain"],
            reason="The 36mo 20% exclusion zones are not subtracted in v2; a TF run's consumed "
                   "prefix may intersect this set.",
            use_instead_for_transformer="valset_v1",
        ),
        lineage=dict(
            derived_from="build_valset.py (valset_v1 recipe)",
            only_difference="APPLY_36MO_EXCLUSION = False",
            expected_superset_of="valset_v1 (v1 = v2 minus the 36mo exclusion)",
        ),
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
                       "map48(perm42_466[:19200], directional guard) \\ (GOOG × 2025-12)"
                       "   [v2: the 36mo term of the v1 recipe is REMOVED]",
            apply_36mo_exclusion=bool(APPLY_36MO_EXCLUSION),
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
                                "one fixed shuffle of V). No stratification is applied or needed: "
                                "V is already a uniform random subset of the 48mo domain, so any "
                                "uniform prefix of it is too."),
        guarantees=[
            "Index-exact (48mo): V ∩ perm_s48[:20%] = ∅ for all s ∈ {5,42,137}; every production "
            "run consumed a strict prefix ≤ curtail*128 ≤ 20%*N48 (verified against W&B step "
            "records of all runs incl. crashed, live scan 2026-07-29).",
            "Message-exact within domain: per-file random offsets come from a fixed-seed RNG "
            "independent of JAX_SEED (lobster_dataloader init_defaults seed=42), so (file,j) maps "
            "to identical message ranges in every run and future eval.",
            "NOT 36mo-immune (v2): the 36mo exclusion is disabled, so this set is NOT held-out "
            "for the Transformer queue or any run on the 2023-2025 domain. Use valset_v1 there.",
            "Uniform by construction: steps 1-2 are prefix/suffix operations on uniform random "
            "permutations and are year-neutral, so V is a uniform random subset of the 48mo "
            "domain and matches the training distribution in every marginal, not only the ones "
            "explicitly checked.",
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
            budget_36mo_steps="not applicable (v2 makes no 36mo guarantee)",
        ),
        distribution=distribution,
        evidence=dict(
            wandb_scans=["scratchpad wandb_post_may23_scan.txt", "wandb_scan2.txt (2026-07-29)"],
            log_anchors={"j4531958_8N_samples_per_node": ASSERT_N48_DIV8,
                          "j4567612_2N_samples_per_node_36mo": ASSERT_N36_DIV2},
            squeue_snapshot="2026-07-29: tf-0p2M-s{5,42,137} PENDING (jobs 5824382/85/88)",
        ),
    )
    json.dump(manifest, open(OUTDIR / f"manifest_{FILE_PREFIX}.json", "w"),
              indent=2, ensure_ascii=False)

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
