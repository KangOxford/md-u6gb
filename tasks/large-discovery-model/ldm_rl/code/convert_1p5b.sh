#!/bin/bash
# Qwen2.5-1.5B-Instruct 的 HF -> Megatron torch_dist 转换。
#
# 为什么要单独写:slime_launch 只有 convert_9b.sh。而 run_train.sh /
# run_warmup_real_slime.sh / run_train_real_slime.sh 都引用
# $REPO_ROOT/rl/qwen2.5-1.5B_torch_dist_te,当作它已经存在 —— 交接里缺这一件。
# P0 关卡(1.5B GRPO 冒烟,验 TE backward 在 ARM 上不 SIGSEGV)直接卡在这里。
#
# 为什么不 source slime 的 scripts/models/qwen2.5-1.5B.sh:
#   那个文件写 --rotary-base 10000,而 Qwen2.5-1.5B-Instruct 的 config.json 里
#   rope_theta = 1000000.0(实测)。启动脚本内联的 MODEL_ARGS 用的正是 1000000,
#   是对的;slime 那个文件是陈旧的。RoPE base 没有可学参数,错了不会在转换时报错,
#   只会让参照模型用错的频率基底 —— KL 与 logprob 全偏,而训练照跑。
#   所以这里照抄启动脚本的内联版本,与训练侧逐字一致。
set -eu
: "${REPO_ROOT:?先 source site_env.sh}"
SLIME_ROOT=${SLIME_ROOT:-$REPO_ROOT/rl/slime}
MODEL_HF=${MODEL_HF_1P5B:-${HF_MODELS:?}/Qwen2.5-1.5B-Instruct}
SAVE=${SAVE_1P5B:-$REPO_ROOT/rl/qwen2.5-1.5B_torch_dist_te}

export PATH=$CONDA_PREFIX/bin:$PATH
export PYTHONPATH=$MEGATRON_ROOT:$REPO_ROOT/rl:$REPO_ROOT:${PYTHONPATH:-}
export CUDA_HOME=$CONDA_PREFIX

# 与 run_warmup_real_slime.sh / run_train_real_slime.sh 里的 MODEL_ARGS 逐字一致。
MODEL_ARGS=(
   --swiglu --num-layers 28 --hidden-size 1536 --ffn-hidden-size 8960
   --num-attention-heads 12 --use-rotary-position-embeddings --disable-bias-linear
   --add-qkv-bias --normalization "RMSNorm" --norm-epsilon 1e-6 --rotary-base 1000000
   --group-query-attention --num-query-groups 2 --vocab-size 151936
)

echo "[convert-1.5B] HF   = $MODEL_HF"
echo "[convert-1.5B] SAVE = $SAVE"
[ -f "$MODEL_HF/config.json" ] || { echo "FATAL: $MODEL_HF 下没有 config.json"; exit 2; }

# 起跑前对账:HF config 里的 rope_theta 必须等于我们要传的 --rotary-base。
# 这是上面那段注释的机械化版本 —— 注释会过期,检查不会。
python - "$MODEL_HF" 1000000 <<'PY'
import json, sys, pathlib
cfg = json.loads((pathlib.Path(sys.argv[1]) / "config.json").read_text())
want = float(sys.argv[2]); got = float(cfg.get("rope_theta", -1))
print(f"[convert-1.5B] config.rope_theta={got:g}  要传的 --rotary-base={want:g}")
if got != want:
    raise SystemExit(f"FATAL: 不一致。转换与训练两侧的 RoPE base 必须相同,否则参照模型全偏。")
for k, v in [("num_hidden_layers",28),("hidden_size",1536),("intermediate_size",8960),
             ("num_attention_heads",12),("num_key_value_heads",2),("vocab_size",151936)]:
    if cfg.get(k) != v:
        raise SystemExit(f"FATAL: config.{k}={cfg.get(k)} 与 MODEL_ARGS 里的 {v} 不符")
print("[convert-1.5B] 结构参数与 MODEL_ARGS 逐项一致")
PY

# 转换器按 `WORLD_SIZE or SLURM_NTASKS or 1` 定并行度
# (tools/convert_hf_to_torch_dist.py:92)。SLURM_NTASKS 会从外层分配/step 继承下来,
# 如果它是 4 而实际只起 1 个进程,分布式初始化会挂住等不到的 rank。显式钉成 1。
export WORLD_SIZE=1
export LOCAL_RANK=0
unset SLURM_NTASKS SLURM_NTASKS_PER_NODE SLURM_STEP_NUM_TASKS 2>/dev/null || true
echo "[convert-1.5B] WORLD_SIZE=$WORLD_SIZE (单进程单卡)"

cd "$SLIME_ROOT"
# --no-gradient-accumulation-fusion:APEX 按 PLAN §2 跳过了(源码编,且训练脚本
# 本来就带这个标志)。但**转换脚本没带** —— Megatron 的 ColumnParallelLinear 默认
# gradient_accumulation_fusion=True,找不到 fused_weight_gradient_mlp_cuda 就直接
#   RuntimeError: ... you must install APEX with --cpp_ext and --cuda_ext
# 训练那边有 APEX_ARGS 挡着,转换这边没有,所以要在这里补。convert_9b.sh 同样缺。
python tools/convert_hf_to_torch_dist.py \
   "${MODEL_ARGS[@]}" \
   --no-gradient-accumulation-fusion \
   --hf-checkpoint "$MODEL_HF" \
   --save "$SAVE"
echo "[convert-1.5B] 完成 -> $SAVE/"
