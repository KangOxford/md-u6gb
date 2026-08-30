#!/bin/bash
# B7b 2026-02 补生成：四格（on/off × s42/s43），配置与 1 月完全一致
# （SEED=2026、NSEQ=32、TSTART=0.80、ANC_MODE=off），只换月份。
# 三个 tag 走 6197253（与 LIBERO 评测共存，显存 4-9G + 29G 放得下），
# 第四个走 6184062 的 nid010731（该分配还剩 ~2h20m，生成预计 ~90 分钟）。
set -u
T=/lus/lfs1aip2/projects/public/u6gb/tasks/dfm_largescale_20260813
cd $T

# 清单：每个 seed 取 1 月 on/off 都存在的 ticker 交集，与评分口径对齐
for sd in s42 s43; do
  : > logs/tkb7b_feb_$sd.txt
  for tk in $(cat logs/g2x_tk60.txt); do
    if [ -f "rollouts_anc/b7bON_${sd}_${tk}_2026-01_learned.npz" ] && \
       [ -f "rollouts_anc/b7bOFF_${sd}_${tk}_2026-01_learned.npz" ]; then
      echo $tk >> logs/tkb7b_feb_$sd.txt
    fi
  done
  echo "[feb] $sd 清单: $(wc -l < logs/tkb7b_feb_$sd.txt) tickers"
done

go () {  # TAG ARM SEEDLIST NODE JOB
  local S=$T/artifacts_b7b/$2_state.msgpack
  [ -f "$S" ] || { echo "FATAL: 缺 state $S" >&2; return 1; }
  local _st
  _st=$(python3 -c "import json;print(json.load(open('$S.meta')).get('step','?'))" 2>/dev/null)
  [ "$_st" = "8000" ] || { echo "FATAL: $2 state step=$_st != 8000" >&2; return 1; }
  echo "[state] $2 step=$_st ok"
  TKL=$T/logs/tkb7b_feb_$3 STATE=$S RSCALE=1.0 SEED=2026 TSTART=0.80 NSEQ=32 \
    NCH=4 GPU0=0 MEMFRAC=0.30 TAG=$1 AXIS=digit ORDF='' ANC_MODE=off \
    MO=2026-02 NODE=$4 JOB=$5 bash $T/run_anc.sh
  sleep 15
}

go b7bON_s42  b7b_on_s42  s42 nid011109 6197253
go b7bOFF_s42 b7b_off_s42 s42 nid011131 6197253
go b7bON_s43  b7b_on_s43  s43 nid011132 6197253
go b7bOFF_s43 b7b_off_s43 s43 nid010731 6184062
echo "=== B7b 2026-02 四格 launched $(date -u +%H:%M:%SZ) ==="
