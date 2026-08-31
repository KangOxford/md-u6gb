#!/usr/bin/env python3
"""复现结果分析:算论文 C1 的比值 Δ_A/Δ_B。

论文 C1:LDM 的验证 BPB 绝对下降幅度是 LLM-only 反思的 2.4 倍。
比的是「同起点下两组各自降了多少」之比,不是绝对 bpb。

数据来源优先级:summary.json(跑完才有)→ model_based.log(跑中也能读)。
"""
import json, re, sys
from pathlib import Path

RUNS = Path("/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/"
            "Large-Discovery-Models/tasks/nanogpt/runs")
BPB = re.compile(rb"val_bpb=([0-9.]+(?:e[+-]?\d+)?)")
SENTINEL = 1e9

# 共同基线:未修改的 resources/train/real_train.py 实测 val_bpb。
# 运行配置里 evaluate_root=False、warmup_include_root=False —— 原始代码不进评测,
# 两组的第一次评测已经是 LLM 改过的候选,起点各不相同(A ~0.994 vs B ~1.05)。
# 拿各自的第一次评测当起点会让起得高的那组天然有更大下降空间,
# 论文比的是「从共同起点的改善」,所以两组都以这个固定基线为准。
BASELINE = 1.019308

def pick_dir(arm, seed):
    """按数字后缀选最新的 run 目录(mtime 不可靠:旧目录可能被后写)。"""
    cands = list(RUNS.glob(f"repro_arm{arm}_s{seed}")) + \
            list(RUNS.glob(f"repro_arm{arm}_s{seed}_*"))
    if not cands:
        return None
    def key(p):
        m = re.search(r"_(\d+)$", p.name)
        return int(m.group(1)) if m else 0
    return max(cands, key=key)

def series(d):
    """有效 bpb 序列(按评测顺序),哨位值剔除。"""
    log = d / "model_based.log"
    vals = []
    if log.exists():
        raw = log.read_bytes().replace(b"\r", b"\n")
        for m in BPB.finditer(raw):
            v = float(m.group(1))
            if 0 < v < SENTINEL / 2:
                vals.append(v)
    return vals

def main():
    rows, deltas = {}, {}
    for arm in ("A", "B"):
        for seed in (1, 2, 3):
            d = pick_dir(arm, seed)
            if d is None:
                continue
            vals = series(d)
            done = (d / "summary.json").exists()
            rows[(arm, seed)] = (d.name, vals, done)

    print(f"共同基线(未修改 real_train.py 实测): val_bpb = {BASELINE:.6f}\n")
    print(f"{'组':<3} {'种子':<4} {'有效评测':>8} {'首评':>10} {'最好':>10} {'Δ下降':>10} {'状态':<6} 目录")
    print("-" * 84)
    for (arm, seed), (name, vals, done) in sorted(rows.items()):
        st = "完成" if done else "跑中"
        if not vals:
            print(f"{arm:<3} {seed:<4} {0:>8} {'—':>10} {'—':>10} {'—':>10} {st:<6} {name}")
            continue
        first, best = vals[0], min(vals)
        deltas[(arm, seed)] = BASELINE - best
        print(f"{arm:<3} {seed:<4} {len(vals):>8} {first:>10.6f} {best:>10.6f} "
              f"{BASELINE-best:>10.6f} {st:<6} {name}")

    print()
    ratios = []
    for seed in (1, 2, 3):
        a, b = deltas.get(("A", seed)), deltas.get(("B", seed))
        if a is None or b is None:
            continue
        if b <= 1e-9:
            print(f"seed {seed}:  Δ_A={a:.6f}  Δ_B={b:.6f}  → B 组无改善,比值无定义")
            continue
        ratios.append(a / b)
        print(f"seed {seed}:  Δ_A={a:.6f}  Δ_B={b:.6f}  →  Δ_A/Δ_B = {a/b:.2f}×")

    if ratios:
        mean = sum(ratios) / len(ratios)
        wins = sum(1 for r in ratios if r > 1.0)
        print(f"\n三种子均值 Δ_A/Δ_B = {mean:.2f}×      论文 C1 报的是 2.4×")
        print(f"方向一致性:{wins}/{len(ratios)} 个种子上 LDM 优于 LLM-only")
        print("\n注:单条轨迹不构成因果估计(论文自述),多种子只给方向一致性。")
    else:
        print("还没有可算的比值(需要 A/B 同种子都有有效数据)。")

if __name__ == "__main__":
    main()
