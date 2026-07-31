#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 manifest_densify180.json：盘上存在但从未评过的 checkpoint。

目的：加密链内轨迹采样（现状 7-10 点跨 2 个数量级 C，log-C 线性插值误差偏大）。
注意：这些点全部落在各链已有 [C_min, C_max] 之内或之上，**不会**拉低任何链的
C_min，因此无法弥补低算力切片的左臂缺口（早期 checkpoint 已被 max_to_keep 删除，
数据不存在）。加密只提升插值精度，不改变 bracketing 结构。

schema 与 manifest_backfill124.json 一致（valset_ce_eval.py 消费的 10 个键）。
jan2026_ce 置 None：这些 step 当年未做 Jan 评测，无配对值。

用法: python make_manifest_densify.py   （login 节点纯 CPU，分钟级）
"""
import csv
import json
import os
import subprocess
from pathlib import Path

VE = Path(__file__).resolve().parent
M256 = VE / "valset_ce_256_master_table_20260731T161800Z.csv"
BACKFILL = VE / "manifest_backfill124.json"
OUT = VE / "manifest_densify180.json"

bf = json.loads(BACKFILL.read_text())
ckpt_dir = {}
micro_bsz = {}
num_params = {}
for x in bf:
    k = (x["size"], str(x["seed"]))
    ckpt_dir.setdefault(k, x["ckpt_dir"])
    micro_bsz.setdefault(x["size"], int(x["micro_bsz"]))
    num_params.setdefault(x["size"], int(x["num_params"]))

jid = {}
wandb_id = {}
for x in bf:
    k = (x["size"], str(x["seed"]))
    jid.setdefault(k, x["jid"])
    wandb_id.setdefault(k, x.get("wandb_id", ""))

# 已评 (size, seed, step) + 每链 tokens_per_step（由已评点的 D_tokens/step 反推）
done = set()
tps = {}
for r in csv.DictReader(open(M256)):
    k = (r["size"], str(r["seed"]))
    done.add((r["size"], str(r["seed"]), int(r["step"])))
    if int(r["step"]) > 0:
        tps.setdefault(k, float(r["D_tokens"]) / int(r["step"]))

items = []
for k, d in sorted(ckpt_dir.items()):
    sz, sd = k
    out = subprocess.run(["lfs", "find", d, "-maxdepth", "1", "-type", "d"],
                         capture_output=True, text=True, timeout=120).stdout
    steps = sorted(int(os.path.basename(l)) for l in out.split()
                   if os.path.basename(l).isdigit())
    for s in steps:
        if (sz, sd, s) in done:
            continue
        items.append(dict(
            label=f"{sz}-s{sd}@{s}", size=sz, seed=int(sd), jid=jid[k],
            wandb_id=wandb_id[k], ckpt_dir=d, step=s,
            micro_bsz=micro_bsz[sz], num_params=num_params[sz],
            D_tokens=int(round(tps[k] * s)), jan2026_ce=None,
            is_terminal=False, protocol_row="densify"))

# 大模型优先（长杆先起），同 size 内按 step 升序
items.sort(key=lambda x: (-x["num_params"], x["step"]))
OUT.write_text(json.dumps(items, indent=1))

print(f"wrote {OUT}: {len(items)} checkpoints")
per = {}
for x in items:
    per.setdefault(x["size"], []).append(x)
for sz in sorted(per, key=lambda s: -per[s][0]["num_params"]):
    v = per[sz]
    print(f"  {sz:>6}: {len(v):3d} ckpts, step {min(x['step'] for x in v)}..{max(x['step'] for x in v)}")
