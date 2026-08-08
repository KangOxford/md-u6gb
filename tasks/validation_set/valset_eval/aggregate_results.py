#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""汇总 valce_*.json + per-sample loss → summary CSV / markdown 表。
micro = 30,720 样本等权均值（与训练 loss 同口径）；
macro = 先按 ticker 求均值再对 487 ticker 等权（与 Jan-2026 test CE 同口径）。
用法: python aggregate_results.py <results_dir> [ticker_per_sample.npy]
第二参数省略时用 valset 的 ticker 映射；Jan-shuffle 轴须显式传 jan_ticker_per_sample_30720.npy
（macro 的分组向量是数据的属性，不是脚本的属性）。"""
import csv, glob, json, os, sys
import numpy as np

TICKER_NPY = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ticker_per_sample_30720.npy")

rd = sys.argv[1]
tickers = np.load(TICKER_NPY)
rows = []
for f in sorted(glob.glob(os.path.join(rd, "valce_*.json"))):
    r = json.load(open(f))
    sl = f.replace(".json", "_sampleloss.npy")
    if os.path.exists(sl):
        s = np.load(sl)
        assert len(s) == len(tickers) == 30720, f"{sl}: {len(s)}"
        # macro：ticker 内先平均，再对 ticker 等权平均
        order = np.argsort(tickers, kind="stable")
        st, ss = tickers[order], s[order]
        cuts = np.flatnonzero(np.r_[True, st[1:] != st[:-1]])
        per_ticker = np.array([ss[a:b].mean() for a, b in zip(cuts, np.r_[cuts[1:], len(ss)])])
        r["val_ce_macro"] = float(per_ticker.mean())
        r["n_tickers"] = int(len(per_ticker))
    rows.append(r)
rows.sort(key=lambda r: (r["num_params"], r["seed"]))
print(f"{len(rows)} checkpoints aggregated")

csv_path = os.path.join(rd, "valset_ce_summary.csv")
cols = ["label", "size", "seed", "jid", "step", "num_params", "D_tokens",
        "val_ce_micro", "val_ce_lo", "val_ce_hi", "val_ce_macro",
        "val_acc", "jan2026_ce_macro", "delta_macro_val_minus_jan",
        "eval_bsz_per_gpu", "wall_sec"]
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f); w.writerow(cols)
    for r in rows:
        w.writerow([r["label"], r["size"], r["seed"], r["jid"], r["step"],
                    r["num_params"], r["D_tokens"],
                    f"{r['val_ce_mean']:.6f}", f"{r['val_ce_ci95'][0]:.6f}",
                    f"{r['val_ce_ci95'][1]:.6f}",
                    f"{r.get('val_ce_macro', float('nan')):.6f}",
                    f"{r['val_acc_mean']:.6f}", f"{r['jan2026_ce']:.6f}",
                    f"{r.get('val_ce_macro', float('nan'))-r['jan2026_ce']:+.6f}",
                    r["eval_bsz_per_gpu"], r["wall_sec"]])
print("wrote", csv_path)

import statistics as st_
md = ["| Size | N params | Seeds | Val CE micro per seed | micro mean ± sd | macro mean | Jan-2026 CE (macro) | Δ(val−Jan, macro) |",
      "|---|---:|---|---|---|---|---|---|"]
by = {}
for r in rows: by.setdefault(r["size"], []).append(r)
for size, rs in sorted(by.items(), key=lambda kv: kv[1][0]["num_params"]):
    vs = [r["val_ce_mean"] for r in rs]
    ms = [r.get("val_ce_macro", float("nan")) for r in rs]
    js = [r["jan2026_ce"] for r in rs]
    per = "; ".join(f"s{r['seed']}:{r['val_ce_mean']:.6f}" for r in rs)
    sd = st_.stdev(vs) if len(vs) > 1 else 0.0
    md.append(f"| {size} | {rs[0]['num_params']:,} | {len(rs)} | {per} | "
              f"{st_.mean(vs):.6f} ± {sd:.6f} | {st_.mean(ms):.6f} | "
              f"{st_.mean(js):.6f} | {st_.mean(ms)-st_.mean(js):+.6f} |")
open(os.path.join(rd, "valset_ce_summary.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
