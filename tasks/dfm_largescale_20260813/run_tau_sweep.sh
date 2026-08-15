#!/bin/bash
# keep-tau 扫描：draft <-> 修正器的**真**旋钮。
#
# 本文件由 `run_alpha_sweep.sh` **逐字复制**而来，只换了被扫的那个标志。
# 上一版我凭印象重写启动行，把 `--stocks/--index-dir/--group-size/--t-start/
# --n-steps` 全丢了，还把 `DFM_GPU` 写成 `CUDA_VISIBLE_DEVICES` —— 同一个坑
# 今天第二次（见 feedback_read_the_call_site_not_the_name）。
#
# 为什么 alpha 不是旋钮，而这个是。修正器每步对**每个**可编辑位置都从模型
# logits 重新抽样（`predict_x1` 只保留位置 0），所以修正臂不是「draft + 修正」，
# 而是 **DFM 模型自己的一次完整生成**。实测印证：修正臂随深度**平**（不累积），
# 而它的**水平**是 DFM 模型边缘律的性质，5 个字段里 4 个比 AR draft 差。
#
# alpha 被实测否证：`event_type` 的水平代价随 alpha 变小**单调上升**，
# +0.094 (a=1.00) -> +0.754 (a=0.25)，8 倍，方向反了 —— 步间重新加噪与 alpha
# 无关，小 alpha 等于「加满噪 + 修不动」，极限是腐蚀而不是 draft。
#
# keep-tau：只在模型给 draft token 的概率**低于** tau 时才覆盖它。
#     tau 小  -> 保留得多 -> 接近 draft
#     tau = 1 -> 几乎不保留 -> 就是修正器
#     tau = 0 -> 关闭分支（**另一条代码路径**）
# 于是 **tau=1 与 tau=0 必须几乎一致** —— 免费的空操作对照（H4）。
#
# 判据（跑之前锁死，与 E7 同一条）：
#   存在某个 tau，使 >=3/5 字段**水平低于 draft** 且这些字段**绝对斜率 <=0**；
#   且 |tau=1 与 tau=0 之差| 必须显著小于 tau 之间的差，否则这道闸没接上。
set -u
T=/lus/lfs1aip2/projects/public/u6gb/tasks/dfm_largescale_20260813
S0=/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/dfm-post-training-20260801
A=/lus/lfs1aip2/projects/public/u6gb/tasks/ce_orderflow_20260812T200352Z/A02_scale
W=$S0/post_training/dfm/eval/run_eval_node.sh
STATE=$T/artifacts/lg488b_g2_state.msgpack
JOB=${JOB:-6014307}; NODE=${NODE:-nid010895}; MO=${MO:-2026-01}; NTK=${NTK:-60}
# GPU0：这一批臂从哪张卡起排。节点上通常只有**部分**卡空出来，写死 0
# 会把新臂压到还在训练的卡上（今天正好只有 g0/g1 空）。
GPU0=${GPU0:-0}
mkdir -p $T/${TAUDIR:-rollouts_tau} $T/logs
head -n $NTK $A/logs/tk_feb.txt > $T/logs/tau_tk.txt

i=0
# 四个臂对四张卡。tau=1.0 不进网格：CPU 单元测试已证 tau=0 与 tau=1.0 的输出
# **逐元素相同**（一致比例 1.00000），所以它只会浪费一张卡。tau=0 这一臂留着，
# 它同时是「纯修正器」参照，和对已有 fx488 learned 臂的可重复性检查。
#
# 合成 logits 上的保留比例（`code/test_keep_tau.py` K3，真实模型更自信、曲线会
# 右移，但形状给出了有用区间）：
#     tau  1e-6  1e-3  1e-2  0.1   0.5   1.0
#     保留 0.994 0.786 0.556 0.284 0.176 0.137
TAUS=${TAUS:-"0 0.001 0.01 0.1"}
_n=$(echo $TAUS | wc -w)
[ $((GPU0 + _n)) -le 4 ] || { echo "FATAL: GPU0=$GPU0 加 $_n 个 tau 超出 4 张卡" >&2; exit 5; }
for A_ in $TAUS; do
  TAG=t$(echo $A_ | tr -d .)
  env -u SLURM_NNODES -u SLURM_NTASKS -u SLURM_JOB_ID -u SLURMD_NODENAME \
  setsid nohup srun --jobid=$JOB --overlap --exact --cpus-per-task=8 -w $NODE -N1 -n1 \
    --cpu-bind=none --job-name=dfm-${PFX:-tau}-$TAG \
    --export=ALL,DFM_GPU=$((GPU0 + i)),DFM_SCRIPT=dfm_correct_runner.py,\
XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.30},XLA_FLAGS=--xla_gpu_enable_triton_gemm=false \
    bash $W --month $MO --n-cond 500 --n-gen 500 \
      --stocks $T/logs/tau_tk.txt --index-dir $A/idx --group-size 8 \
      --validate-first 8 --gate-batches 2 --state "$STATE" \
      --t-start 0.80 --n-steps 8 --n-seq 8 --batch-size 2 --corr-batch 2 \
      --keep-tau $A_ --skip-existing \
      --out-template "$T/${TAUDIR:-rollouts_tau}/${PFX:-tau}_${TAG}_{stock}_{month}_learned.npz" \
    > $T/logs/${PFX:-tau}_$TAG.log 2>&1 < /dev/null &
  echo "  tau=$A_ -> $NODE gpu$((GPU0 + i)) tag=$TAG"; i=$((i+1)); sleep 3
done
echo "=== keep-tau 扫描 launched $(date -u +%H:%M:%SZ) ==="
