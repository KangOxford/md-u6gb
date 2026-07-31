#!/bin/bash
# 追加空闲节点到运行中的 backfill124 弹性队列（每节点 4 worker/GPU）。
# 由 srun --jobid=<free-chain> --overlap --nodes=N --ntasks-per-node=1 调起，
# 每 task 落在一个节点，本脚本挂载 squashfs 一次后拉起 4 个单卡 worker。
#
# 协调：所有 worker 指向同一 OUT_DIR，valset_ce_eval.py 内部 json-exists→skip、
# mkdir lock→skip claimed 跨节点靠 Lustre 天然仲裁。
# 铁律：绝不清锁——nid010937 上有活 worker 持在跑 checkpoint 的 in-progress 锁，
# 清锁会导致同一 350M 被两卡重算。孤儿锁恢复交给原 4 worker（parallel_valset.sh 有 retry）。
set -uo pipefail
OUT_DIR="$1"
MANIFEST="${MANIFEST:?MANIFEST required}"
TOTAL="${TOTAL:-124}"
QUANT_ROOT=/lus/lfs1aip2/projects/public/s5e/quant_team/quant
export PATH="$QUANT_ROOT/miniforge3/bin:$PATH"
EXP_DIR=$QUANT_ROOT/AlphaTrade/experiments/exp_R1_Mamba3
VE_DIR=/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/valset_eval
VALSET_SQFS=/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/squashfs/output/shard_valset_v1_30720.squashfs
export PYTHONPATH="$EXP_DIR"
# r3 教训：大 [B*13000,d] GEMM 令 Triton autotuner 失败 → 回落 cuBLAS
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false ${XLA_FLAGS:-}"
HOST=$(hostname -s)
LOGD=/lus/lfs1aip2/projects/public/u6gb/tasks/validation_set/logs
mkdir -p "$LOGD"

TMP_BASE="${TMPDIR:-/tmp}"
MOUNT="$TMP_BASE/valset30720_add_${HOST}_$$"
cleanup() { mountpoint -q "$MOUNT" 2>/dev/null && fusermount -u "$MOUNT" 2>/dev/null || true; }
trap cleanup EXIT
mkdir -p "$MOUNT"
squashfuse "$VALSET_SQFS" "$MOUNT"
echo "[${HOST}] mounted valset30720 at $MOUNT"

done_count() { ls "$OUT_DIR"/valce_*.json 2>/dev/null | grep -cv sampleloss; }

worker() {
    local G=$1
    local WLOG="$LOGD/valce_add_${HOST}_gpu${G}.out"
    {
      echo "[${HOST} gpu${G}] worker start $(date -u +%H:%M:%SZ)"
      # 每 pass 走一遍 manifest（跳过 done/claimed）；最多 4 pass，全 claimed 即秒退。
      # 无锁清理：只做纯增量吞吐，孤儿锁留给原 worker 的 retry 回收。
      for pass in 1 2 3 4; do
          (( $(done_count) >= TOTAL )) && { echo "[${HOST} gpu${G}] queue drained"; break; }
          CUDA_VISIBLE_DEVICES=$G XLA_PYTHON_CLIENT_MEM_FRACTION=0.80 \
          python -u "$VE_DIR/valset_ce_eval.py" \
              --manifest "$MANIFEST" --data_root "$MOUNT" --out_dir "$OUT_DIR" \
              --num_devices 1 --n_data_workers 12
          echo "[${HOST} gpu${G}] pass ${pass} rc=$? $(date -u +%H:%M:%SZ)"
          sleep 10
      done
      echo "[${HOST} gpu${G}] worker done"
    } >> "$WLOG" 2>&1
}

for G in 0 1 2 3; do worker "$G" & done
wait
echo "[${HOST}] all 4 workers exited; done=$(done_count)/$TOTAL"
