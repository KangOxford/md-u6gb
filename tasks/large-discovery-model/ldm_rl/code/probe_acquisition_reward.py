"""区分 acquisition reward 的 0.0 是「真的算出 0」还是「一个分数都没取到」。

env.py:647-651 在 scores 为空时静默 return 0.0。两种情形的后果完全不同:
  真 0      -> reward 有定义,只是这一步的期望效用恰好为 0
  空列表 0  -> reward 恒为 0 ⇒ GRPO 优势全零 ⇒ 没有梯度信号,R1/R2 会空训

判断依据是 components['scores']:空列表就是后者。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve()
while _REPO.name != "LDM-rl" and _REPO.parent != _REPO:
    _REPO = _REPO.parent
_REPO = Path("/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/LDM-rl")
for p in (str(_REPO / "rl"), str(_REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from ldm_rl import EnvConfig  # noqa: E402
from ldm_rl.factories import build_env  # noqa: E402
from tasks.small_molecule.core.workflow import ExpandingMockCase2LLM  # noqa: E402

ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
RESERVOIR = int(sys.argv[2]) if len(sys.argv) > 2 else 2
EVALS = int(sys.argv[3]) if len(sys.argv) > 3 else 1

env = build_env(
    "small_molecule",
    mode="real",
    config=EnvConfig(
        iterations=ROUNDS,
        reservoir_size=RESERVOIR,
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
    vina_max_workers=int(os.environ.get("VINA_WORKERS", "8")),
)

llm = ExpandingMockCase2LLM()
obs = env.reset()
print(f"[probe] rounds={ROUNDS} reservoir={RESERVOIR} evals/round={EVALS}", flush=True)
empty_rounds = 0
for r in range(ROUNDS):
    action = llm.chat("system", obs, json_mode=True)
    step = env.step(action)
    comp = step.info.get("reward_components", step.info) if hasattr(step, "info") else {}
    scores = comp.get("scores", "<无 scores 键>")
    is_empty = isinstance(scores, list) and len(scores) == 0
    empty_rounds += is_empty
    print(
        f"round={r} reward={step.reward:.6f} scores={scores} agg={comp.get('agg')}"
        f"  {'<<< 空列表:走了静默回退' if is_empty else ''}",
        flush=True,
    )
    obs = step.observation
    if step.done:
        break

print()
if empty_rounds:
    print(f"结论: {empty_rounds}/{ROUNDS} 轮的 scores 是空列表 —— acquisition reward 恒 0,")
    print("      GRPO 优势全零,R1/R2 会空训。必须先修才能起跑。")
else:
    print("结论: 每一轮都取到了真实的 acquisition 分数,0.0 是真实数值而非回退。")
