#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""拼接 256 点 valset CE 全轨迹表：132 尾窗（已有 master CSV）+ 124 早期补评（本次 JSON）。
统一口径输出 plot-ready 列（micro CE、CI、val_acc、jan2026_ce、delta=micro−jan、source）。
写到新时间戳文件，绝不覆盖已有 132 表。
用法: python build_master_table_256.py <backfill_results_dir> <out_csv>
"""
import csv, glob, json, os, sys

BACKFILL_DIR = sys.argv[1]
OUT_CSV = sys.argv[2]
M132 = "/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/scaling_law_plots/valset_ce_eval_20260730/valset_ce_132_master_table.csv"

COLS = ["label", "size", "seed", "step", "num_params", "D_tokens",
        "val_ce_micro", "val_ce_ci95_lo", "val_ce_ci95_hi", "val_acc",
        "jan2026_ce", "delta_micro_minus_jan", "is_terminal", "source"]

rows = []

# 132 尾窗：直接读 master CSV 的对应列
with open(M132) as f:
    for r in csv.DictReader(f):
        vce = float(r["val_ce_micro"])
        jan = float(r["jan2026_ce_macro"]) if r.get("jan2026_ce_macro") else float("nan")
        rows.append(dict(
            label=r["label"], size=r["size"], seed=r["seed"], step=r["step"],
            num_params=r["num_params"], D_tokens=r["D_tokens"],
            val_ce_micro=vce, val_ce_ci95_lo=r["val_ce_ci95_lo"], val_ce_ci95_hi=r["val_ce_ci95_hi"],
            val_acc=r["val_acc"], jan2026_ce=jan,
            delta_micro_minus_jan=vce - jan, is_terminal=r["is_terminal"], source="terminal132"))

# 124 早期：读本次 JSON
for f in sorted(glob.glob(os.path.join(BACKFILL_DIR, "valce_*.json"))):
    if "sampleloss" in f:
        continue
    d = json.load(open(f))
    vce = float(d["val_ce_mean"])
    jan = float(d["jan2026_ce"]) if d.get("jan2026_ce") is not None else float("nan")
    ci = d.get("val_ce_ci95", [float("nan"), float("nan")])
    rows.append(dict(
        label=d["label"], size=d["size"], seed=d["seed"], step=d["step"],
        num_params=d["num_params"], D_tokens=d["D_tokens"],
        val_ce_micro=vce, val_ce_ci95_lo=ci[0], val_ce_ci95_hi=ci[1],
        val_acc=d.get("val_acc_mean", ""), jan2026_ce=jan,
        delta_micro_minus_jan=vce - jan, is_terminal=False, source="backfill124"))

# 排序：size(按 num_params) → seed → step，便于逐链读轨迹
rows.sort(key=lambda x: (int(x["num_params"]), str(x["seed"]), int(x["step"])))
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader()
    w.writerows(rows)

n132 = sum(1 for r in rows if r["source"] == "terminal132")
n124 = sum(1 for r in rows if r["source"] == "backfill124")
print(f"wrote {OUT_CSV}: {len(rows)} rows ({n132} terminal132 + {n124} backfill124)")
sizes = {}
for r in rows:
    sizes.setdefault(r["size"], [0, 0])
    sizes[r["size"]][0 if r["source"] == "terminal132" else 1] += 1
print("per-size (terminal/backfill):")
for s in sorted(sizes, key=lambda s: [r for r in rows if r["size"] == s][0]["num_params"] and int([r for r in rows if r["size"] == s][0]["num_params"])):
    print(f"  {s:>6}: {sizes[s][0]:>3} + {sizes[s][1]:>3} = {sum(sizes[s])}")
ce = [r["val_ce_micro"] for r in rows]
print(f"val_ce_micro range: {min(ce):.4f} .. {max(ce):.4f}")
