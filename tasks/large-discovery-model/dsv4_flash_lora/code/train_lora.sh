#!/bin/bash
# DeepSeek-V4-Flash LoRA 训练 —— 每节点 1 个任务,由 srun 启动(attach 或 sbatch 皆可)
# 必需外部变量: MASTER_ADDR
# 可调外部变量: EP_SIZE GBS MBS TAG TRAIN_ITERS DATASET_CAP
set -u

source /lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/dsv4_flash_lora/code/env_dsv4.sh
MODEL_DIR=${MODEL_DIR_OVERRIDE:-$MODEL_DIR}   # smoke 可指到 mini 模型
MTP_LAYERS=${MTP_LAYERS:-1}

# —— 分布式几何(全部显式,不靠继承)——
export NNODES=${SLURM_NNODES:?需要在 srun 下运行}
export NODE_RANK=${SLURM_NODEID:?}
export NPROC_PER_NODE=4                      # GH200 每节点 4 卡
export MASTER_ADDR=${MASTER_ADDR:?外层必须给 MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT:-29511}

WORLD=$(( NNODES * NPROC_PER_NODE ))
EP_SIZE=${EP_SIZE:-16}
MBS=${MBS:-1}
GBS=${GBS:-$WORLD}                           # 缺省: DP 张数一人一条,零累积
TAG=${TAG:-run}
DATASET_CAP=${DATASET_CAP:-600}              # 只决定 cached 数据目录名,训练时不再触碰原始数据集
TRAIN_ITERS=${TRAIN_ITERS:-}                 # 空 = 按 1 epoch 跑

# 数据一律走离线 cached dataset(export_cached_ds.sh 单进程建好):训练时 64 rank
# 只 mmap 只读 arrow,无 map、无 fingerprint、无缓存竞态(6144379/6146475 两次死因)
CACHED_DS=${CACHED_DS:-$DSV4_TASK/data/cached_ds_cap${DATASET_CAP}}
if [ ! -d "$CACHED_DS/train" ]; then
    echo "FATAL: cached dataset 不存在: $CACHED_DS/train —— 先跑 code/export_cached_ds.sh" >&2; exit 6
fi

# 有效批量自查: GBS 必须能被 DP×MBS 整除(TP=1,PP=1 ⇒ DP=WORLD)
if [ $(( GBS % (WORLD * MBS) )) -ne 0 ]; then
    echo "FATAL: GBS=$GBS 不能被 WORLD($WORLD)×MBS($MBS) 整除" >&2; exit 5
fi
[ "$NODE_RANK" = "0" ] && echo "[bsz] 有效批量 $GBS = MBS$MBS × DP$WORLD × 累积$(( GBS / (WORLD*MBS) ))  | EP$EP_SIZE  节点数 $NNODES"

EXTRA_ARGS=()
[ -n "$TRAIN_ITERS" ] && EXTRA_ARGS+=(--train_iters "$TRAIN_ITERS")

exec megatron sft \
    --model $MODEL_DIR \
    --save_safetensors true \
    --cached_dataset "$CACHED_DS/train" \
    --cached_val_dataset "$CACHED_DS/val" \
    --merge_lora false \
    --load_from_cache_file true \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --tuner_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --tensor_model_parallel_size 1 \
    --expert_model_parallel_size $EP_SIZE \
    --micro_batch_size $MBS \
    --global_batch_size $GBS \
    --padding_free false \
    --group_by_length true \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --num_train_epochs 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --output_dir $DSV4_TASK/results/megatron_output/$TAG \
    --eval_steps 200 \
    --save_steps 200 \
    --max_length 4096 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --mtp_num_layers $MTP_LAYERS \
    --attention_backend flash \
    "${EXTRA_ARGS[@]}"
