#!/bin/bash
# 给数据量阶梯的一个 rung 打分：推理 -> A03 曲线 -> 配对裁决。
#
# 为什么单独写：`pipeline.sh` 只跑到 A02（一个数字的斜率），而目标问的是
# **水平和斜率两件事**，那要 A03 的曲线 + `a03_verdict.py` 的配对自举。
# 把三步串成一条命令，是因为阶梯有五个 rung，手动串五遍必然串错一次。
#
# 用法：
#   STEP=14000 JOB=6014308 NODES="nid010723 nid010896" bash score_ladder.sh
#
# 对照口径：与 fx488_g2（当前基线，1:79）**逐字比对**同一套判据 ——
# 同样的 ticker 表、同样的 n-seq、同样的两个月、同样的估计量。唯一变的是
# 残差 checkpoint 的训练步数。
set -u
T=/lus/lfs1aip2/projects/public/u6gb/tasks/dfm_largescale_20260813
A3=/lus/lfs1aip2/projects/public/u6gb/tasks/ce_orderflow_20260812T200352Z/A03_curve_shape
STEP=${STEP:?need STEP}
ARM=${ARM:-dv25_lr1e4_s0u}
TAG=${TAG:-dvlad${STEP}}
STATE=$T/artifacts_dv/ladder/${ARM}_s${STEP}_state.msgpack
JOB=${JOB:?need JOB}
NODES=${NODES:?need NODES}
MONTHS=${MONTHS:-"2026-01 2026-02"}
NSEQ=${NSEQ:-8}

[ -f "$STATE" ] || { echo "FATAL: 没有这个 rung 的快照 $STATE" >&2; exit 3; }
echo "=== rung step=$STEP  实际 $(python3 -c "import json;print(json.load(open('$STATE.meta'))['step'])")  $(date -u +%H:%M:%SZ) ==="

TAG=$TAG STATE=$STATE JOB=$JOB NODES="$NODES" MONTHS="$MONTHS" NSEQ=$NSEQ \
  bash $T/pipeline.sh || { echo "FATAL: pipeline 失败" >&2; exit 4; }

cd $A3
for MO in $MONTHS; do
  for DIV in kl js; do
    python3 code/a03_curves.py --rollouts $T/rollouts --prefix "${TAG}_" \
      --month $MO --div $DIV --out out/${TAG}_${MO}_${DIV}.npz \
      > out/log_${TAG}_${MO}_${DIV}.txt 2>&1
  done
done
echo "=== 配对裁决 ==="
# a03_verdict 按 out/fx488_<月>_<div>.npz 取文件名，这里临时改成本 TAG
sed "s/fx488/${TAG}/g" code/a03_verdict.py > code/_verdict_${TAG}.py
python3 code/_verdict_${TAG}.py kl
echo "=== 与基线 fx488_g2 (1:79) 对比 ==="
python3 - <<PY
import json
a=json.load(open("out/verdict_${TAG}_kl.json"))
b=json.load(open("out/verdict_fx488_kl.json"))
print(f"{'field':<12}{'月':<9}{'Δ水平 rung':>12}{'Δ水平 1:79':>12}{'改善':>9}"
      f"{'Δ斜率 rung':>12}{'Δ斜率 1:79':>12}")
for f in ["event_type","direction","price_rel","size","log10_dt"]:
    for mo in ["2026-01","2026-02"]:
        x,y=a[mo][f],b[mo][f]
        d=y["corr_dlevel"]-x["corr_dlevel"]
        print(f"{f:<12}{mo:<9}{x['corr_dlevel']:>+12.3f}{y['corr_dlevel']:>+12.3f}"
              f"{d:>+9.3f}{x['corr_dslope']:>+12.3f}{y['corr_dslope']:>+12.3f}")
PY
