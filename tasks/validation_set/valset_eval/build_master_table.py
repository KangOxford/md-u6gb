#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""构建 132 行 master 大表：manifest + 结果 json + per-sample 分位数 + target_step。
用法: python build_master_table.py <results_dir> <out_prefix>"""
import csv, glob, json, os, sys
import numpy as np

rd, out_prefix = sys.argv[1], sys.argv[2]
VE = os.path.dirname(os.path.abspath(__file__))
LAST25 = "/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/scaling_law_plots/aramis/results/selected_test_last25.csv"

target_step = {}
for r in csv.DictReader(open(LAST25)):
    target_step[(r["run_id"], int(r["step"]))] = int(r["target_step"])

man = {m["label"]: m for m in json.load(open(os.path.join(VE, "manifest_132ckpt.json")))}

rows = []
for f in sorted(glob.glob(os.path.join(rd, "valce_*.json"))):
    r = json.load(open(f))
    m = man[r["label"]]
    s = np.load(f.replace(".json", "_sampleloss.npy"))
    run_id = r["label"].split("@")[0]
    tstep = target_step.get((run_id, r["step"]))
    rows.append(dict(
        label=r["label"], run_id=run_id, size=r["size"], seed=r["seed"],
        jid=r["jid"], wandb_id=m["wandb_id"], step=r["step"], target_step=tstep,
        pct_of_target=round(r["step"] / tstep, 4) if tstep else None,
        is_terminal=m["is_terminal"],
        num_params=r["num_params"], D_tokens=r["D_tokens"],
        tokens_per_param=round(r["D_tokens"] / r["num_params"], 1),
        n_samples=r["n_samples"], eval_bsz_per_gpu=r["eval_bsz_per_gpu"],
        val_ce_micro=round(r["val_ce_mean"], 6),
        val_ce_ci95_lo=round(r["val_ce_ci95"][0], 6),
        val_ce_ci95_hi=round(r["val_ce_ci95"][1], 6),
        val_ce_macro=None,  # 填在下方
        val_acc=round(r["val_acc_mean"], 6),
        jan2026_ce_macro=round(r["jan2026_ce"], 6),
        delta_val_macro_minus_jan=None,
        sample_loss_p10=round(float(np.percentile(s, 10)), 6),
        sample_loss_p50=round(float(np.percentile(s, 50)), 6),
        sample_loss_p90=round(float(np.percentile(s, 90)), 6),
        wall_sec=r["wall_sec"], ckpt_dir=r["ckpt_dir"],
        sampleloss_file=os.path.basename(f).replace(".json", "_sampleloss.npy"),
    ))
    # macro
    tk = np.load(os.path.join(VE, "ticker_per_sample_30720.npy"))
    order = np.argsort(tk, kind="stable"); st, ss = tk[order], s[order]
    cuts = np.flatnonzero(np.r_[True, st[1:] != st[:-1]])
    per_t = [ss[a:b].mean() for a, b in zip(cuts, np.r_[cuts[1:], len(ss)])]
    rows[-1]["val_ce_macro"] = round(float(np.mean(per_t)), 6)
    rows[-1]["delta_val_macro_minus_jan"] = round(rows[-1]["val_ce_macro"] - rows[-1]["jan2026_ce_macro"], 6)

rows.sort(key=lambda x: (x["num_params"], x["seed"], x["step"]))
cols = list(rows[0].keys())
with open(out_prefix + ".csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

with open(out_prefix + ".md", "w") as f:
    f.write("| " + " | ".join(cols) + " |\n")
    f.write("|" + "---|" * len(cols) + "\n")
    for r in rows:
        f.write("| " + " | ".join(str(r[c]) for c in cols) + " |\n")
print(f"master table: {len(rows)} rows x {len(cols)} cols -> {out_prefix}.csv/.md")
