#!/usr/bin/env python3
"""汇总 A/B 两组的 val_bpb 轨迹,算论文 C1 的那个比值。

论文 C1:LDM 的验证 BPB 绝对下降幅度 = LLM-only 反思的 2.4 倍。
所以要算的是 Δ_A / Δ_B,其中 Δ = 起点 bpb − 该组最好 bpb。
「起点」取该条 campaign 预热阶段的最好值(两组的预热设置相同,
起点本身不该有系统差异,这一点会打印出来供核对)。
"""
import json, sys, pathlib, statistics as st

RUNS = pathlib.Path("/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model"
                    "/Large-Discovery-Models/tasks/nanogpt/runs")
FAIL = 1e8   # 大于此值的是失败哨兵(1e+09)

def load(run_dir):
    """返回 (预热最好值, 全程最好值, 成功评测数, 总评测数)。"""
    buf = run_dir / "model_based_buffer.jsonl"
    if not buf.exists():
        return None
    rows = []
    for line in buf.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        v = (r.get("metrics") or {}).get("val_bpb", r.get("val_bpb"))
        if v is None:
            continue
        rows.append((r.get("iteration"), float(v)))
    ok = [v for _, v in rows if v < FAIL]
    if not ok:
        return None
    # 预热行的 iteration 为 None 或 <=0;取前若干条成功值作起点参考
    warm = [v for it, v in rows if v < FAIL and (it is None or it <= 0)]
    start = min(warm) if warm else max(ok)
    return dict(start=start, best=min(ok), n_ok=len(ok), n_all=len(rows))

MIN_OK = 10   # 少于这么多次成功评测的运行不进统计

def main():
    arms = {"A": [], "B": []}
    for d in sorted(RUNS.glob("repro_arm*")):
        name = d.name                      # repro_armA_s1_03
        arm = "A" if "_armA_" in name else ("B" if "_armB_" in name else None)
        if arm is None:
            continue
        got = load(d)
        if got is None:
            print(f"  [跳过] {name}: 无成功评测")
            continue
        got["run"] = name
        got["delta"] = got["start"] - got["best"]
        arms[arm].append(got)

    # 只跑通预热就死掉的运行(生成全失败)不能进统计:它的 Δ 恒为 0,
    # 会把两组的均值一起拉向 0,看起来像「都没效果」。
    for arm in ("A", "B"):
        kept, dropped = [], []
        for g in arms[arm]:
            (kept if g["n_ok"] >= MIN_OK else dropped).append(g)
        arms[arm] = kept
        for g in dropped:
            print(f"  [排除] {g['run']}: 仅 {g['n_ok']} 次成功评测 (<{MIN_OK}),视为未跑起来")

    print(f"\n{'组':<3}{'run':<26}{'起点':>10}{'最好':>10}{'Δ下降':>10}{'成功/总':>10}")
    for arm in ("A", "B"):
        for g in arms[arm]:
            ratio = f"{g['n_ok']}/{g['n_all']}"
            print(f"{arm:<3}{g['run']:<26}{g['start']:>10.5f}{g['best']:>10.5f}"
                  f"{g['delta']:>10.5f}{ratio:>10}")

    if not arms["A"] or not arms["B"]:
        print("\n两组都要有数据才能算比值。"); return 1

    dA = [g["delta"] for g in arms["A"]]
    dB = [g["delta"] for g in arms["B"]]
    mA, mB = st.mean(dA), st.mean(dB)
    print(f"\nA 组(LDM,GP 引导开) Δ 均值 = {mA:.5f}  (n={len(dA)}"
          f"{', SD=%.5f' % st.stdev(dA) if len(dA) > 1 else ''})")
    print(f"B 组(LLM-only)      Δ 均值 = {mB:.5f}  (n={len(dB)}"
          f"{', SD=%.5f' % st.stdev(dB) if len(dB) > 1 else ''})")
    if mB > 0:
        print(f"\n比值 Δ_A/Δ_B = {mA/mB:.2f}×    论文 C1 报的是 2.4×")
    else:
        print("\nB 组没有下降,比值无意义(分母 ≤ 0)")

    # 成对比较:同种子的 A/B 落同一节点,配对能消掉节点与时段差异
    pairs, used = [], set()
    for a in arms["A"]:
        seed = a["run"].split("_s")[1].split("_")[0]
        for b in arms["B"]:
            if b["run"] in used:
                continue                      # 一条 B 只配一次,否则同一条会被重复计入
            if b["run"].split("_s")[1].split("_")[0] == seed:
                pairs.append((seed, a["delta"], b["delta"]))
                used.add(b["run"])
                break
    if pairs:
        print("\n成对(同种子):")
        for s, da, db in pairs:
            mark = "A 更好" if da > db else ("B 更好" if db > da else "平")
            print(f"  seed {s}: Δ_A={da:.5f}  Δ_B={db:.5f}  → {mark}")
        wins = sum(1 for _, da, db in pairs if da > db)
        print(f"  A 组在 {wins}/{len(pairs)} 对中下降更多")
    return 0

if __name__ == "__main__":
    sys.exit(main())
