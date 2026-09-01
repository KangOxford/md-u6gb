#!/bin/bash
# Qwen3.5-9B(base 或 SFT)的 HF -> Megatron torch_dist 转换。
#
# 与上游 slime_launch/convert_9b.sh 的差别只有两处,都是 1.5B 那次实测出来的:
#   1. --no-gradient-accumulation-fusion —— APEX 按 PLAN §2 跳过了,训练脚本
#      本来就带这个标志,但转换脚本没带。Megatron 的 ColumnParallelLinear 默认
#      gradient_accumulation_fusion=True,找不到 fused_weight_gradient_mlp_cuda
#      就直接 RuntimeError 要求装 APEX。
#   2. WORLD_SIZE 钉成 1 —— 转换器按 `WORLD_SIZE or SLURM_NTASKS or 1` 定并行度
#      (convert_hf_to_torch_dist.py:92),而 SLURM_NTASKS 会从外层分配继承;
#      若它是 4 而实际只起 1 个进程,分布式初始化会挂住等不到的 rank。
#
# 起跑前对账(与 convert_1p5b.sh 同样的做法):这两个模型是**多模态**的
# (architectures=Qwen3_5ForConditionalGeneration,有 vision_config),文本参数在
# config["text_config"] 下,rope 在 text_config["rope_parameters"]["rope_theta"]。
# 实测两个模型的七项结构参数与 slime 的 qwen3.5-9B.sh 逐项相符。
set -eu
: "${REPO_ROOT:?先 source site_env.sh}"
SLIME_ROOT=${SLIME_ROOT:-$REPO_ROOT/rl/slime}
WHICH=${1:?用法: convert_9b.sh base|sft}

case "$WHICH" in
  base) MODEL_HF="${HF_MODELS:?}/Qwen3.5-9B";  SAVE="$REPO_ROOT/rl/qwen3.5-9B_torch_dist" ;;
  sft)  MODEL_HF="${HF_MODELS:?}/LDM-CoT-SFT"; SAVE="$REPO_ROOT/rl/qwen3.5-9B-sft_torch_dist" ;;
  *) echo "FATAL: 第一个参数只能是 base 或 sft"; exit 2 ;;
esac

export PATH=$CONDA_PREFIX/bin:$PATH
export PYTHONPATH=$MEGATRON_ROOT:$REPO_ROOT/rl:$REPO_ROOT:${PYTHONPATH:-}
export CUDA_HOME=$CONDA_PREFIX
export WORLD_SIZE=1
export LOCAL_RANK=0
unset SLURM_NTASKS SLURM_NTASKS_PER_NODE SLURM_STEP_NUM_TASKS 2>/dev/null || true

echo "[convert-9B:$WHICH] HF   = $MODEL_HF"
echo "[convert-9B:$WHICH] SAVE = $SAVE"
[ -f "$MODEL_HF/config.json" ] || { echo "FATAL: $MODEL_HF 下没有 config.json"; exit 2; }
# 跳过判断要看**完成标记**,不能只看目录存在:一次失败会留下几 KB 的残目录
# (实测 sft 那次失败后留了 8.5K),只看存在就会把重跑误判成"已完成"。
# 转换器成功时会写 latest_checkpointed_iteration.txt。
if [ -f "$SAVE/latest_checkpointed_iteration.txt" ]; then
    echo "[convert-9B:$WHICH] $SAVE 已完成(有 latest_checkpointed_iteration.txt),跳过"; exit 0
fi
if [ -d "$SAVE" ]; then
    _old="${SAVE}_partial_$(date -u +%Y%m%dT%H%M%SZ)"
    echo "[convert-9B:$WHICH] $SAVE 存在但没有完成标记 —— 是上次失败的残留,改名到 $(basename $_old)"
    mv "$SAVE" "$_old"
fi

# torch.distributed 的 TCPStore 端口:转换器用 os.environ.setdefault("MASTER_PORT","12355")
# (convert_hf_to_torch_dist.py:101)。同一节点上两个转换并行、或上一次被打断留下的
# 进程还占着,就会 EADDRINUSE。按 base/sft 各给一个端口,并允许外部覆盖。
case "$WHICH" in
  base) export MASTER_PORT=${MASTER_PORT:-12361} ;;
  sft)  export MASTER_PORT=${MASTER_PORT:-12362} ;;
esac
export MASTER_ADDR=${MASTER_ADDR:-localhost}
echo "[convert-9B:$WHICH] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

# 对账:注释会过期,检查不会
python - "$MODEL_HF" <<'PY'
import json, sys, pathlib
c = json.loads((pathlib.Path(sys.argv[1]) / "config.json").read_text())
t = c["text_config"]
rope = t.get("rope_parameters", {})
SPEC = {"num_attention_heads":16, "num_key_value_heads":4, "head_dim":256,
        "num_hidden_layers":32, "hidden_size":4096, "intermediate_size":12288,
        "vocab_size":248320}
bad = [f"{k}: config={t.get(k)} spec={v}" for k, v in SPEC.items() if t.get(k) != v]
if rope.get("rope_theta") != 10000000:
    bad.append(f"rope_theta: config={rope.get('rope_theta')} spec=10000000")
if rope.get("partial_rotary_factor") != 0.25:
    bad.append(f"partial_rotary_factor: config={rope.get('partial_rotary_factor')} spec=0.25")
if c.get("tie_word_embeddings") is not False:
    bad.append(f"tie_word_embeddings={c.get('tie_word_embeddings')},而 spec 用 --untie-embeddings-and-output-weights")
if bad:
    print("FATAL: config 与 slime 的 qwen3.5-9B.sh 对不上:")
    for b in bad: print("   ", b)
    raise SystemExit(1)
print("[convert-9B] 结构参数与 slime spec 逐项一致(含 rope_theta 与 partial_rotary_factor)")
PY

cd "$SLIME_ROOT"
source "$SLIME_ROOT/scripts/models/qwen3.5-9B.sh"   # 设 MODEL_ARGS=(...)
python tools/convert_hf_to_torch_dist.py \
   "${MODEL_ARGS[@]}" \
   --no-gradient-accumulation-fusion \
   --hf-checkpoint "$MODEL_HF" \
   --save "$SAVE"
echo "[convert-9B:$WHICH] 完成 -> $SAVE/"
