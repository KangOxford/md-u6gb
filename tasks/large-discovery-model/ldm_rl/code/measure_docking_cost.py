"""量真实 docking 的单轮耗时,好定 vina_max_workers。

PLAN §5 第 4 条:"vina_max_workers=32 是按 288 核猜的;docking 是真瓶颈,
先量一个分子的真实耗时再定并发"。这里就量那个数。

要回答的是两件事:
  1. 一轮(含 docking + 活性预测 + GP 更新)实际多久
  2. 并发从 1 提到 N,单轮耗时降不降 —— 如果 evaluations_per_round=1,
     并发很可能一点用都没有,那 32 这个数就是假的

用法: python measure_docking_cost.py [rounds] [workers1,workers2,...]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_REPO = Path("/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/LDM-rl")
for p in (str(_REPO / "rl"), str(_REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from ldm_rl import EnvConfig  # noqa: E402
from ldm_rl.factories import build_env  # noqa: E402
from tasks.small_molecule.core.workflow import ExpandingMockCase2LLM  # noqa: E402

ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
WORKER_GRID = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["1", "8", "32"])]
EVALS = int(os.environ.get("EVALS_PER_ROUND", "1"))

print(f"[cost] rounds={ROUNDS} evaluations_per_round={EVALS} workers={WORKER_GRID}", flush=True)
print(f"{'workers':>8} {'总耗时(s)':>10} {'每轮(s)':>10}   逐轮", flush=True)

results = {}
for w in WORKER_GRID:
    env = build_env(
        "small_molecule",
        mode="real",
        config=EnvConfig(
            iterations=ROUNDS,
            reservoir_size=2,
            evaluations_per_round=EVALS,
            reward="acquisition",
        ),
        vina_bin=os.environ["VINA_BIN"],
        nn_model_path=os.environ["NN_MODEL"],
        vina_pdb_id="8UN5",
        vina_chain_id="A",
        gp_device="cpu",
        vina_exhaustiveness=1,
        vina_n_poses=1,
        vina_max_workers=w,
    )
    llm = ExpandingMockCase2LLM()
    obs = env.reset()
    per_round = []
    t0 = time.perf_counter()
    for _ in range(ROUNDS):
        t = time.perf_counter()
        step = env.step(llm.chat("system", obs, json_mode=True))
        per_round.append(time.perf_counter() - t)
        obs = step.observation
        if step.done:
            break
    total = time.perf_counter() - t0
    results[w] = total / max(len(per_round), 1)
    print(
        f"{w:>8} {total:>10.1f} {results[w]:>10.1f}   "
        + " ".join(f"{x:.1f}" for x in per_round),
        flush=True,
    )

print()
base = results[WORKER_GRID[0]]
print(f"以 workers={WORKER_GRID[0]} 为基准:")
for w in WORKER_GRID:
    print(f"  workers={w:>3}: 加速 {base / results[w]:.2f}×")
if len(WORKER_GRID) > 1 and max(base / results[w] for w in WORKER_GRID) < 1.2:
    print()
    print("并发几乎没有收益 —— 说明每轮只 dock 极少分子(evaluations_per_round=1),")
    print("vina_max_workers 这个旋钮在当前设置下碰不到瓶颈。要提吞吐得先提每轮分子数。")
