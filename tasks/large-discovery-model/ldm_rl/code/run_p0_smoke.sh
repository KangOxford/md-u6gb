#!/bin/bash
# P0 关卡:1.5B GRPO 冒烟。
#
# 这一步要回答的**唯一**问题:aarch64 上 TE 的 backward 会不会 SIGSEGV
# (HANDOFF §6/§7 把它列为头号风险)。所以起跑前必须把每个前提都验掉 ——
# 若因为缺一个文件而挂,这次运行什么都没回答,却看起来"跑过了"。
#
# 用法: bash run_p0_smoke.sh [节点]     节点默认自动挑一个 4/4 全空的
set -u
T=/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/ldm_rl
source "$T/code/site_env.sh" >/dev/null 2>&1

JOB=${JOB:-6217606}
NODE=${1:-}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$T/logs/p0_smoke_${STAMP}.log

say() { echo "[$(date -u +%H:%M:%S)] $*"; }
fail=0
chk() { # chk <说明> <路径>
    if [ -e "$2" ]; then printf "  ✓ %-34s %s\n" "$1" "$2"
    else printf "  ✗ %-34s %s  <<< 缺\n" "$1" "$2"; fail=1; fi
}

say "=== P0 起跑前检查 ==="
chk "训练栈 python"          "$CONDA_PREFIX/bin/python"
chk "Megatron-LM"            "$MEGATRON_ROOT/megatron"
chk "slime"                  "$SLIME_ROOT/train.py"
chk "1.5B HF 权重"           "$HF_MODELS/Qwen2.5-1.5B-Instruct/config.json"
chk "1.5B torch_dist(ref)"   "$REPO_ROOT/rl/qwen2.5-1.5B_torch_dist_te"
chk "episodes(acq-max)"      "$REPO_ROOT/rl_episodes_sm_acqmax.jsonl"
chk "vina 二进制"            "$VINA_BIN"
chk "G12D 活性模型"          "$NN_MODEL"
chk "评测栈 python"          "$TASK_PYTHON"

# 训练栈里的关键 import(这几个装不上,后面全是白跑)。
#
# 判断依据不能只是「import 不报错」:PYTHONPATH 里有 $REPO_ROOT/rl,于是
# `import slime` 会命中 rl/slime/ 这个源码目录(命名空间包)而成功,哪怕它
# 根本没被 pip install -e 装过。所以对 slime 与 megatron 额外要求能查到
# 安装记录(importlib.metadata),而不是只看 import。
say "--- 训练栈自检 ---"
PYTHONPATH= "$CONDA_PREFIX/bin/python" - <<'PY' || fail=1
import importlib, sys
from importlib.metadata import version, PackageNotFoundError

# (模块名, 分发包名或 None)。给了分发包名的,还要能查到安装记录。
CHECKS = [("torch", "torch"), ("transformer_engine.pytorch", "transformer_engine"),
          ("sglang", "sglang"), ("sglang_router", "sglang-router"),
          ("megatron.core", "megatron-core"), ("slime", "slime"),
          ("ray", "ray"), ("flash_attn", "flash-attn"),
          ("torch_memory_saver", "torch-memory-saver")]
bad = []
for mod_name, dist in CHECKS:
    try:
        importlib.import_module(mod_name)
    except Exception as e:
        print(f"  ✗ {mod_name:30s} import 失败: {type(e).__name__}: {str(e)[:60]}")
        bad.append(mod_name); continue
    try:
        v = version(dist)
        print(f"  ✓ {mod_name:30s} {v}")
    except PackageNotFoundError:
        print(f"  ✗ {mod_name:30s} import 成功但查不到安装记录(可能只是命中了源码目录)")
        bad.append(mod_name)

try:
    import torch
    print(f"  torch {torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("  !! torch 看不见卡 —— 多半是装成了 cu13(驱动 565 只到 CUDA 12.7)")
        bad.append("torch.cuda")
except Exception as e:
    print(f"  ✗ torch 不可用: {e}")
sys.exit(1 if bad else 0)
PY

# TE 的 backward 探针:装得上 != ABI 对得上。HANDOFF §7 的头号风险是
# 「aarch64 上 TE 的 backward 会 SIGSEGV」,而那个问题**不需要**先转检查点、
# 起 ray、起 sglang 才能回答。几秒钟的最小实验先给出答案,失败时的栈也干净。
# 注意它要在计算节点上跑(要卡),所以走 srun,不在这里直接执行。
if [ "$fail" = 0 ]; then
    say "--- TE backward 探针(在计算节点上跑) ---"
    _probe_node=${NODE:-}
    if [ -z "$_probe_node" ]; then
        _probe_node=$(squeue -h -j "$JOB" -o "%N" | sed 's/[][]//g' | cut -d, -f1 | sed 's/-.*//')
    fi
    srun --overlap --jobid="$JOB" --nodes=1 --ntasks=1 -w "$_probe_node" \
         --gres=gpu:1 --cpus-per-task=8 --cpu-bind=none --job-name=ldmrl-teprobe \
         "$CONDA_PREFIX/bin/python" "$T/code/probe_te_backward.py" 2>&1 | sed 's/^/    /'
    _te_rc=${PIPESTATUS[0]}
    [ "$_te_rc" = 0 ] || { say "TE backward 探针失败(rc=$_te_rc) —— 这正是 P0 要验的那件事,先修它"; fail=1; }
fi

[ "$fail" = 0 ] || { say "前提不全,不起跑(空跑无意义)"; exit 3; }

# 挑一个 4/4 全空的节点:4 卡作业落在部分空的节点上会在启动窗口之后 SIGABRT,
# 而 `Training on 4 device` 这行挡不住 —— Slurm 给的是配额,JAX/torch 看见的是
# 裸进程已经占掉的显存。
if [ -z "$NODE" ]; then
    say "--- 找 4/4 全空的节点 ---"
    NODE=$(gtop --once 2>/dev/null | awk -v job="$JOB" '
        $0 ~ "job "job {inj=1} inj && /^ ▸ job/ && $0 !~ "job "job {inj=0}
        inj && /^   nid/ {n=$1; c=0}
        inj && /mem   0\.0\/95\.6G/ {c++; if(c==4) {print n; exit}}')
    [ -n "$NODE" ] || { say "FATAL: 分配 $JOB 里没有 4/4 全空的节点"; exit 4; }
fi
say "起跑于 $NODE,日志 $LOG"
echo "$LOG" > "$T/logs/LATEST_P0_LOG"

setsid nohup srun --overlap --jobid="$JOB" --nodes=1 --ntasks=1 -w "$NODE" \
    --gres=gpu:4 --cpus-per-task=128 --cpu-bind=none --job-name=ldmrl-p0 \
    bash -c "
      source $T/code/site_env.sh >/dev/null 2>&1
      export EPISODES=\$REPO_ROOT/rl_episodes_sm_acqmax.jsonl
      export SAVE=\$REPO_ROOT/rl/qwen2.5-1.5B_slime_p0_smoke
      echo '[p0] 节点 \$(hostname)  CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-<未设>}'
      nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
      bash \$LAUNCH/run_train_real_slime.sh
    " > "$LOG" 2>&1 &
say "已提交(pid $!)。用 tail -f $LOG 看"
