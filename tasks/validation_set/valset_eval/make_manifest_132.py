#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 132-checkpoint（final-25% 窗口）manifest。
label 规则：terminal step 沿用 `{run_id}`（与 33 版 json/锁完全兼容 → 已完成自动跳过），
非 terminal 用 `{run_id}@{step}`。每行 D_tokens/jan2026_ce 取该 step 的行内值。"""
import csv, json, os

SL = "/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/scaling_law_plots"
EXP = "/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_R1_Mamba3"
OUT = "/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/valset_eval/manifest_132ckpt.json"

snap = {}
for r in csv.DictReader(open(f"{SL}/v3-mamba3-plan-and-results/wandb_mamba3_runs_snapshot.csv")):
    snap.setdefault(r["name"].lstrip("j"), []).append(r)

terminal_step = {r["run_id"]: int(r["step"])
                 for r in csv.DictReader(open(f"{SL}/aramis/results/selected_test_endpoint.csv"))}

rows, problems = [], []
for r in csv.DictReader(open(f"{SL}/aramis/results/selected_test_last25.csv")):
    jid, step = r["source_jid"], int(r["step"])
    best = max(snap[jid], key=lambda c: int(c["global_step"] or 0))
    ckpt_dir = f"{EXP}/checkpoints/j{jid}_{best['run_id']}_{jid}"
    if not os.path.isdir(f"{ckpt_dir}/{step}"):
        problems.append(f"{r['run_id']}@{step}: missing"); continue
    is_term = (step == terminal_step.get(r["run_id"]))
    label = r["run_id"] if is_term else f"{r['run_id']}@{step}"
    rows.append(dict(
        label=label, size=r["label"], seed=int(r["seed"]), jid=jid,
        wandb_id=best["run_id"], ckpt_dir=ckpt_dir, step=step,
        micro_bsz=int(best["micro_bsz"]), num_params=int(r["N"]),
        D_tokens=int(r["D"]), jan2026_ce=float(r["L"]),
        is_terminal=is_term, protocol_row="last25"))

rows.sort(key=lambda x: (x["num_params"], x["seed"], x["step"]))
json.dump(rows, open(OUT, "w"), indent=1)
n_term = sum(1 for x in rows if x["is_terminal"])
print(f"manifest_132: {len(rows)} rows ({n_term} terminal-labeled, "
      f"{len(rows)-n_term} @step), problems={len(problems)}")
for p in problems: print("PROBLEM:", p)
EOF_MARKER_NOT_USED = None
