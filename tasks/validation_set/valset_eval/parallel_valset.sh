#!/bin/bash
# 4-GPU 弹性工作队列 valset eval：挂载一次 squashfs，每张 GPU 一个单卡进程，
# 从共享 manifest 用 mkdir 锁抢 checkpoint（大模型优先）。GPU0 等常驻进程退出后
# 自动加入；所有 33 个 json 齐了各 worker 自动退出。绝不 kill 任何现有进程。
# 用法: parallel_valset.sh <OUT_DIR>
set -uo pipefail
OUT_DIR="$1"
MANIFEST="${MANIFEST:-/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/valset_eval/manifest_33ckpt.json}"
TOTAL="${TOTAL:-33}"
GATE_MB=2000
QUANT_ROOT=/lus/lfs1aip2/projects/public/s5e/quant_team/quant
export PATH="$QUANT_ROOT/miniforge3/bin:$PATH"
EXP_DIR=$QUANT_ROOT/AlphaTrade/experiments/exp_R1_Mamba3
VE_DIR=/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/valset_eval
VALSET_SQFS="${SQFS:-/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/squashfs/output/shard_valset_v1_30720.squashfs}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
# MODE=jan：Jan-2026 shuffle 评测（参数在脚本内组装——srun --export 会在含逗号的
# 值处截断，绝不能经 env 传含逗号/空格的复合字符串，见 2026-07-30 事故）
if [ "${MODE:-valset}" = "jan" ]; then
    VE_ABS=/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/valset_eval
    VALSET_SQFS=/lus/lfs1aip2/projects/public/s5e/quant_team/lob_preproc_sp500_squashfs/shard_2026-01.squashfs
    EXTRA_ARGS="--sampler_indices $VE_ABS/jan_shuffle_30720_indices.npy --n_expect 7507307 --date_range 2026-01-02,2026-01-31 --provenance $VE_ABS/jan_pool_ticker_all.npy"
fi
export PYTHONPATH="$EXP_DIR"
# r3 教训：大 GEMM 的 Triton autotune 失败 → 回落 cuBLAS
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false ${XLA_FLAGS:-}"

TMP_BASE="${TMPDIR:-/tmp}"
MOUNT="$TMP_BASE/valset30720_par_$$"
cleanup() { mountpoint -q "$MOUNT" 2>/dev/null && fusermount -u "$MOUNT" 2>/dev/null || true; }
trap cleanup EXIT
mkdir -p "$MOUNT" "$OUT_DIR"
squashfuse "$VALSET_SQFS" "$MOUNT"
echo "[squashfs] mounted at $MOUNT"

# 清残锁（此刻无 worker 在跑；有 json 的锁无所谓，无 json 的锁会卡死队列）
for L in "$OUT_DIR"/lock_*; do [ -d "$L" ] && rmdir "$L" 2>/dev/null || true; done

done_count() { ls "$OUT_DIR"/valce_*.json 2>/dev/null | wc -l; }

gpu_worker() {
    local GPU=$1
    local WLOG="$VE_DIR/../logs/valce_par_gpu${GPU}_j${SLURM_JOB_ID:-5790795}.out"
    {
    echo "[gpu${GPU}] gate: waiting used_mem < ${GATE_MB} MiB"
    local pass=0
    while :; do
        if (( $(done_count) >= TOTAL )); then echo "[gpu${GPU}] queue drained before gate"; return 0; fi
        local used
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -dc 0-9)
        if [ -n "$used" ] && (( used < GATE_MB )); then
            pass=$((pass+1)); (( pass >= 2 )) && break
        else
            pass=0
        fi
        sleep 60
    done
    echo "[gpu${GPU}] gate OPEN (used=${used}MiB) — starting worker"
    cd "$EXP_DIR"
    for attempt in 1 2; do
        CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_MEM_FRACTION=0.80 \
        python -u "$VE_DIR/valset_ce_eval.py" \
            --manifest "$MANIFEST" \
            --data_root "$MOUNT" --out_dir "$OUT_DIR" \
            --num_devices 1 --n_data_workers 12 $EXTRA_ARGS
        local rc=$?
        echo "[gpu${GPU}] worker attempt ${attempt} exited rc=${rc}"
        (( rc == 0 )) && break
        (( $(done_count) >= TOTAL )) && break
        # 残锁属于本卡刚死的进程：本 worker 死亡后其锁必然无 json，清掉自己无法完成的抢占
        for L in "$OUT_DIR"/lock_*; do
            [ -d "$L" ] && [ ! -f "${L/lock_/valce_}.json" ] && rmdir "$L" 2>/dev/null || true
        done
        sleep 30
    done
    } >> "$WLOG" 2>&1
}

for G in 0 1 2 3; do gpu_worker "$G" & done
wait
echo "[all] workers done: $(done_count)/$TOTAL jsons"
echo "PARALLEL_VALSET_OK"
