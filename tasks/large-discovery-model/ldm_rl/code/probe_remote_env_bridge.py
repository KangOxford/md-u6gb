"""端到端验 RemoteLDMEnv 这条进程边界(训练栈 -> 评测栈)。

为什么单独验:训练时 reward 不是在训练进程里算的。bridge.py 起一个**子进程**
(`task_python -m ldm_rl.real_env_worker`),走 JSON-lines stdio,并显式剥掉
LD_LIBRARY_PATH / CUDA_HOME / CUDA_VISIBLE_DEVICES —— 这个边界是代码写死的。

而 bridge.py:87-89 读 task_python 的方式是
    spec.real.get("task_python") or "/mnt/data0/ys/LDM/tasks/small_molecule/.venv/bin/python"
一个在别的机器上依然"合法"的硬编码回退。缺这一项时它不报"配置缺失",
而是在 reward 那步报下游错误,把诊断成本推到几个环节之外。

这里直接用真 episode 文件里的 spec 走一遍 RemoteLDMEnv,确认:
  1. 子进程起得来(task_python 指对了)
  2. reset/step 的 JSON-lines 往返通
  3. 真的算出了 reward 与 vina/activity 指标
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/LDM-rl")
for p in (str(REPO / "rl"), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

# 关键:子进程靠 **PYTHONPATH** 找到 ldm_rl,不是靠父进程的 sys.path。
# RemoteLDMEnv 用 os.environ.copy() 传环境,所以这里必须真的 export 出去,
# 否则 worker 起来就 ModuleNotFoundError 退出 —— 而它的 stderr 是 DEVNULL,
# 父进程只会看到一句 "task-venv worker exited unexpectedly"。
# 真实训练里 slime 通过 RUNTIME_ENV_JSON 设了 PYTHONPATH,这里复刻同样的形状。
import os  # noqa: E402

os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(REPO / "rl"), str(REPO)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])
)

EPISODES = REPO / (sys.argv[1] if len(sys.argv) > 1 else "rl_episodes_sm_acqmax.jsonl")
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 2

from ldm_rl.episodes import EpisodeSpec  # noqa: E402
from ldm_rl.remote_env import RemoteLDMEnv  # noqa: E402
from tasks.small_molecule.core.workflow import ExpandingMockCase2LLM  # noqa: E402

row = json.loads(EPISODES.read_text().splitlines()[0])
spec = EpisodeSpec.from_json(str(row["prompt"]))
tp = spec.real.get("task_python")
print(f"[bridge] episodes = {EPISODES.name}")
print(f"[bridge] mode={spec.mode}  reward={spec.reward}/{spec.acquisition_agg}  iterations={spec.iterations}")
print(f"[bridge] task_python = {tp}")
if not tp:
    raise SystemExit("FATAL: episode 里没有 task_python，bridge 会回退到作者机器的路径")
if not Path(tp).exists():
    raise SystemExit(f"FATAL: task_python 指向的解释器不存在: {tp}")

# 把轮数压到 ROUNDS，冒烟不需要跑满 20 轮。
# EpisodeSpec 是 frozen dataclass，只能 replace 不能赋值。
import dataclasses  # noqa: E402

spec = dataclasses.replace(spec, iterations=min(spec.iterations, ROUNDS))

print("[bridge] 起子进程 ...", flush=True)
env = RemoteLDMEnv(spec, tp)
try:
    obs = env.reset()
    print(f"[bridge] reset 成功，观测 {len(str(obs))} 字符", flush=True)
    llm = ExpandingMockCase2LLM()
    total = 0.0
    for r in range(spec.iterations):
        step = env.step(llm.chat("system", obs, json_mode=True))
        total += step.reward
        ev = (step.info.get("evaluated") or [{}])[0]
        met = ev.get("evaluation", {}).get("metrics", {})
        comp = step.info.get("reward_components", {})
        print(
            f"[bridge] round={r} reward={step.reward:.6f} kind={comp.get('kind')} "
            f"scores={comp.get('scores')} metrics={ {k: round(v,4) for k,v in met.items()} }",
            flush=True,
        )
        obs = step.observation
        if step.done:
            break
finally:
    try:
        env.close()
    except Exception:
        pass

print()
print(f"BRIDGE_OK  总 reward={total:.6f}")
print("训练栈 -> 评测栈的子进程边界通了；reward 是在另一个解释器里真算出来的。")
