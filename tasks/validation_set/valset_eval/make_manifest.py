#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 33-checkpoint 评测 manifest：selected_test_endpoint.csv (33 行终点)
join wandb_mamba3_runs_snapshot.csv (jid→wandb_id, micro_bsz)，并验证
checkpoint 目录与目标 step 子目录真实存在。"""
import csv, json, os, sys

SL = "/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/scaling_law_plots"
EXP = "/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_R1_Mamba3"
ENDPOINT = f"{SL}/aramis/results/selected_test_endpoint.csv"
SNAPSHOT = f"{SL}/v3-mamba3-plan-and-results/wandb_mamba3_runs_snapshot.csv"
OUT = "/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/valset_eval/manifest_33ckpt.json"

# jid → [(wandb_id, state, global_step, micro_bsz, num_params)]
snap = {}
for r in csv.DictReader(open(SNAPSHOT)):
    jid = r["name"].lstrip("j")
    snap.setdefault(jid, []).append(r)

rows, problems = [], []
for r in csv.DictReader(open(ENDPOINT)):
    jid, step = r["source_jid"], int(r["step"])
    cands = snap.get(jid, [])
    if not cands:
        problems.append(f"{r['run_id']}: jid {jid} not in snapshot"); continue
    # 同 jid 多 wandb run 时取 global_step 最大者（auto-resume 产生的最终 run）
    best = max(cands, key=lambda c: int(c["global_step"] or 0))
    ckpt_dir = f"{EXP}/checkpoints/j{jid}_{best['run_id']}_{jid}"
    step_dir = f"{ckpt_dir}/{step}"
    ok_dir, ok_step = os.path.isdir(ckpt_dir), os.path.isdir(step_dir)
    if not ok_dir: problems.append(f"{r['run_id']}: missing dir {ckpt_dir}")
    elif not ok_step: problems.append(f"{r['run_id']}: missing step {step_dir}")
    rows.append(dict(
        label=r["run_id"], size=r["label"], seed=int(r["seed"]), jid=jid,
        wandb_id=best["run_id"], ckpt_dir=ckpt_dir, step=step,
        micro_bsz=int(best["micro_bsz"]), num_params=int(r["N"]),
        D_tokens=int(r["D"]), jan2026_ce=float(r["L"]),
        wandb_state=best["state"], dir_ok=ok_dir, step_ok=ok_step))

# 小模型在前：编译/排错快，且先出的点多
rows.sort(key=lambda x: (x["num_params"], x["seed"]))
json.dump(rows, open(OUT, "w"), indent=1)
print(f"manifest rows: {len(rows)}  (expect 33)")
for p in problems: print("PROBLEM:", p)
print("all step dirs exist" if not problems else f"{len(problems)} problems")
for x in rows:
    print(f"  {x['label']:>10} jid={x['jid']} step={x['step']:>6} bsz={x['micro_bsz']} "
          f"params={x['num_params']:>11,} dir_ok={x['dir_ok']} step_ok={x['step_ok']}")
