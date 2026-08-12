#!/bin/bash
# 只做评分，不重新生成。
#
# 用途：生成已经跑完（data_gen 里 3136 组齐全），但收尾校验挂了。重新生成要
# 半小时且结果逐位相同，没有理由再跑一次。
#
# 必需 env: OUTPUT_ROOT ARM_ID ARM_NAME  （OUTPUT_ROOT 下须已有 inference/data_gen）
set -uo pipefail
: "${OUTPUT_ROOT:?}" "${ARM_ID:?}" "${ARM_NAME:?}"
TASKDIR=/lus/lfs1aip2/projects/public/u6gb/tasks/hybrid_system_20260811/05_hybrid_mamba3_lob
export WORKDIR=/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/hybrid-mamba3-nemotron-20260811
export QUANT_ROOT=/lus/lfs1aip2/projects/public/s5e/quant_team/quant
export PYTHON="$QUANT_ROOT/miniforge3/bin/python"
export BENCHMARK_ROOT="$QUANT_ROOT/AlphaTrade/lob_pipeline"
export INFERENCE_DIR="$OUTPUT_ROOT/inference"
export LOBBENCH_OUTPUT="$OUTPUT_ROOT/lobbench"
export SUMMARY="$OUTPUT_ROOT/summary.json"
export N_SEQUENCES=${N_SEQUENCES:-3136}
export N_COND_MSGS=${N_COND_MSGS:-250}
export N_GEN_MSGS=${N_GEN_MSGS:-250}
export LOBBENCH_WORKERS=${LOBBENCH_WORKERS:-48}
export RUN_NAME="${ARM_NAME}-step${CHECKPOINT_STEP:-12000}"
export LOBBENCH_RUN_ID="${ARM_ID}_${CHECKPOINT_STEP:-12000}"

N=$(ls -1 "$INFERENCE_DIR/data_gen" 2>/dev/null | grep -c "_message_real_id_")
echo "[score] $OUTPUT_ROOT  已生成序列=$N  期望=$N_SEQUENCES"
[ "$N" -eq "$N_SEQUENCES" ] || { echo "FATAL: 生成数不符，拒绝打分" >&2; exit 2; }

echo "[score] === LOB-Bench (7.3) $(date -u +%H:%M:%SZ) ==="
bash "$TASKDIR/bench_scripts/bench_score_node.sh"; echo "[score] 7.3 rc=$?"

echo "[score] === return bench (7.2) $(date -u +%H:%M:%SZ) ==="
"$PYTHON" "$BENCHMARK_ROOT/return_bench/run_return_bench.py" \
    --infer_dir "$INFERENCE_DIR" --n_cond "$N_COND_MSGS" \
    --horizons 10,50,100,250 --workers 32 \
    --output_dir "$OUTPUT_ROOT" --name "$LOBBENCH_RUN_ID"; echo "[score] 7.2 rc=$?"

echo "[score] === refer success (7.4) $(date -u +%H:%M:%SZ) ==="
"$PYTHON" "$TASKDIR/code/refer_success.py" \
    --gen-dir "$INFERENCE_DIR/data_gen" \
    --out "$OUTPUT_ROOT" --label "$LOBBENCH_RUN_ID"; echo "[score] 7.4 rc=$?"
echo "[score] 完成 $(date -u +%H:%M:%SZ)"
