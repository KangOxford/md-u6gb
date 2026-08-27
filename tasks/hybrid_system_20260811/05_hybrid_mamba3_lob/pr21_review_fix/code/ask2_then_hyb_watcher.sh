#!/bin/bash
# 候位脚本 v2（限时，默认 12 小时后自动退出）。
#
# v1→v2 的原因：单节点训练会静默丢掉梯度累积（train_helpers.py:1821 只在
# use_shard_map ∧ K>1 时构建累积路径，而 shard_map 只有多节点才开）——
# 2026-08-27 baseline-32k 在 1N 上跑了 1.8h，有效批量实为 4 而不是 80，
# cosine 以 20 倍速退火，产物已整体改名 _deprecated_bsz4。
# 所以训练组一律要求**同分配 ≥2 空节点**（允许 2 或 4，K=10 或 5），
# 单节点位只给 doq 重跑与 bench。
#
# 两条队列，按容量分派（多节点位优先喂训练）：
#   MULTI  (≥2N): base32k → hyb32k → pmatch32k     （首启；续段归 chain_manager32k）
#   SINGLE (1N):  doqrerun → bench2k → bench500
# 所有长任务 setsid 脱管；起载后按 step 名/日志验证，验证失败留在队列。
set -u

TD=/lus/lfs1aip2/projects/public/u6gb/tasks/hybrid_system_20260811/05_hybrid_mamba3_lob
WT=/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/hybrid-mamba3-nemotron-20260811
CK500=/lus/lfs1aip2/projects/public/u6gb/sigma-0/checkpoints/j5877859_30nkkohd_5877859
CK2K=$WT/checkpoints/j6000409_gjnf0e03_6000409
ALLOCS_HINT=""            # 空 = 每轮动态取账户全部 RUNNING 多节点分配
POLL=600
MAX_CYCLES=72
QUEUE_MULTI="base32k hyb32k pmatch32k"
QUEUE_SINGLE="doqrerun bench2k bench500"
WANDB_PROJECT_32K=sp500-mamba3-35m-ctx2k32k

log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }

left_min() {
    local L; L=$(squeue -j "$1" -h -o "%L" 2>/dev/null | tr -d ' ')
    [ -z "$L" ] && { echo 0; return; }
    python3 - "$L" <<'EOF'
import sys
s=sys.argv[1]; d=0
if '-' in s: d,s=s.split('-',1); d=int(d)
p=[int(x) for x in s.split(':')]
while len(p)<3: p.insert(0,0)
print(d*1440+p[0]*60+p[1])
EOF
}

free_nodes_in() {   # $1=jid → stdout 每行一个 4 卡全 <4096MiB 的节点
    local jid="$1" nodes n out ok m
    nodes=$(squeue -h -j "$jid" -o "%N" 2>/dev/null | tr -d ' ')
    [ -z "$nodes" ] && return 0
    for n in $(scontrol show hostnames "$nodes" 2>/dev/null); do
        out=$(timeout 90 srun --overlap --jobid="$jid" -w "$n" --ntasks=1 \
              --job-name=probe-ask2 --cpu-bind=none \
              nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
        [ "$(echo "$out" | grep -c .)" -ge 4 ] || continue
        ok=1
        while read -r m; do [ -n "$m" ] && [ "$m" -ge 4096 ] && ok=0; done <<< "$out"
        [ "$ok" = 1 ] && echo "$n"
    done
}

launch_train() {   # $1=item $2=alloc $3=nodelist(逗号) $4=nn → rc0 成功出队
    local item=$1 alloc=$2 nl=$3 nn=$4 lm mjh sn ld sc extra
    case "$item" in
      base32k)  sn=base-m3-2k32k; ld=$WT/logs_lobs5/ctx2k32k_base; sc=launch_2k_baseline.sh
                extra="EXPECTED_PARAMS=33610439" ;;
      hyb32k)   sn=hyb-m3-2k32k;  ld=$WT/logs_lobs5/ctx2k32k_hyb;  sc=launch_2k_hybrid.sh
                extra="EXPECTED_PARAMS=35435423" ;;
      pmatch32k) sn=pm-m3-2k32k;  ld=$WT/logs_lobs5/ctx2k32k_pm;   sc=launch_2k_hybrid.sh
                extra="EXPECTED_PARAMS=33609998 HYBRID_ATTN_D_FF=1135" ;;
      *) return 9 ;;
    esac
    lm=$(left_min "$alloc")
    if [ "$lm" -lt 240 ]; then log "$item：$alloc 只剩 ${lm}min（<240），不放训练"; return 7; fi
    mjh=$(python3 -c "print(max(1.0, round(($lm-25)/60.0, 1)))")
    log "起 $item on [$nl] (${nn}N, $alloc)，MAX_JOB_HOURS=$mjh（setsid 脱管）"
    setsid nohup env ATTACH_JOBID="$alloc" NNODES_ATTACH="$nn" NODELIST="$nl" \
        SMOKE=0 DEDICATED_ALLOC=0 GATE_OWNER=claude-ctx2k32k \
        STEP_NAME="$sn" COSINE_STEPS=32000 \
        CHECKPOINT_EVERY=auto LOG_GRAD_NORMS=1 LOG_EVERY=250 \
        NODE_LOG_DIR="$ld" WANDB_PROJECT="$WANDB_PROJECT_32K" \
        MAX_JOB_HOURS="$mjh" $extra \
        bash "$TD/code/$sc" \
        > "$TD/logs/launch_${item}_$(date -u +%m%dT%H%M%S).log" 2>&1 &
    sleep 300
    if squeue -s -j "$alloc" -h 2>/dev/null | grep -q "$sn"; then
        # 记下节点，chain_manager32k 达标收刀要用
        case "$item" in base32k) a=A;; hyb32k) a=B;; pmatch32k) a=P;; esac
        echo "$nl" > "$TD/results/.ctx2k32k_nodes_$a"
        echo "$alloc" > "$TD/results/.ctx2k32k_nodes_$a.alloc"
        log "$item step 已现身，放置成功"
        return 0
    fi
    log "$item 300s 未见 step，视为失败重排（看 launch_${item} 日志）"
    return 8
}

launch_single() {   # $1=item $2=alloc $3=node → rc0 成功出队
    local item=$1 alloc=$2 node=$3
    case "$item" in
      doqrerun)
        log "起 doqrerun on $node ($alloc)（同步，约 10min）"
        timeout 2400 srun --overlap --jobid="$alloc" -w "$node" --ntasks=1 \
            --job-name=doq-rerun --cpu-bind=none \
            env ARMS_SUBDIR=arms_v2 bash "$TD/pr21_review_fix/code/run_four_arms_pr21.sh" \
            > "$TD/logs/doq_pr21_rerun.log" 2>&1
        if ! grep -q DRIVER_DONE "$TD/logs/doq_pr21_rerun.log"; then
            log "DOQ_ROLLOUT_FAIL（driver 未完成，看 doq_pr21_rerun.log）"; return 0
        fi
        local A2=/home/u6gb/kangli.u6gb/pr21_doq_artifacts_20260826/arms_v2
        local CPX COIX
        CPX=$(git -C /lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/ci-pipefail-20260826 rev-parse HEAD)
        COIX=$(git -C "$WT" rev-parse HEAD)
        python3 "$WT/tools/diagnostics/verify_injection.py" \
            --dir-p "$A2/arm_P" --dir-p2 "$A2/arm_P2" --dir-o "$A2/arm_O" --dir-i "$A2/arm_I" \
            --inject-step 8 --inject-event-type 1 --inject-side 1 --inject-qty 777 \
            --inject-offset-ticks -1 --tick-size 100 \
            --n-cond-msgs 16 --batch-size 1 --seed 42 --node "$node" --gpu 0 \
            --ckpt-pin m3-goog-78m-u6gb \
            --ckpt-metadata-sha256 02d6f8b0d9258fd0b5375dd896fab57ad0321e8aead4ac994255c082d4099c86 \
            --checkpoint-step 28730 --data-day GOOG_2026-02-27 \
            --commit-p "$CPX" --commit-oi "$COIX" \
            --out "$TD/pr21_review_fix/injection_report_20260827_v2.json" \
            >> "$TD/logs/doq_pr21_rerun.log" 2>&1
        if [ $? -eq 0 ]; then log "DOQ_VERIFY_PASS（report v2 全绿，待录证据）"
        else log "DOQ_VERIFY_FAIL（clean 节点仍不过，需要人看 report v2）"; fi
        return 0
        ;;
      bench2k)
        log "起 bench2k：base2k@6509 on $node ($alloc)（脱管，约 2-3h）"
        setsid nohup env ATTACH_JOBID="$alloc" NODE="$node" ARCHITECTURE=mamba3 \
            ARM_ID=base2ktok ARM_NAME=base2ktok \
            CHECKPOINT_PATH="$CK2K" CHECKPOINT_STEP=6509 \
            BENCH_BATCH="$TD/bench_scripts/bench_2k.batch" MIN_WALLTIME_MIN=150 \
            bash "$TD/code/run_bench_attached.sh" \
            > "$TD/logs/ask2_bench2k_20260826.log" 2>&1 &
        sleep 180
        if grep -q "FATAL" "$TD/logs/ask2_bench2k_20260826.log" 2>/dev/null; then
            log "bench2k 关卡拒绝，留队（看 ask2_bench2k 日志）"; return 8
        fi
        log "bench2k 已放置（结果稍后从 bench2k_* 目录收）"; return 0
        ;;
      bench500)
        log "起 bench500：base500@32001 on $node ($alloc)（脱管，约 1-2h）"
        setsid nohup env ATTACH_JOBID="$alloc" NODE="$node" ARCHITECTURE=mamba3 \
            ARM_ID=base500tok ARM_NAME=base500tok \
            CHECKPOINT_PATH="$CK500" CHECKPOINT_STEP=32001 \
            BENCH_BATCH="$TD/bench_scripts/bench_hybrid.batch" MIN_WALLTIME_MIN=90 \
            bash "$TD/code/run_bench_attached.sh" \
            > "$TD/logs/ask2_bench500_20260826.log" 2>&1 &
        sleep 180
        if grep -q "FATAL" "$TD/logs/ask2_bench500_20260826.log" 2>/dev/null; then
            log "bench500 关卡拒绝，留队（看 ask2_bench500 日志）"; return 8
        fi
        log "bench500 已放置"; return 0
        ;;
    esac
}

log "候位 v2 开始：MULTI=[$QUEUE_MULTI]  SINGLE=[$QUEUE_SINGLE]  POLL=${POLL}s"
cycle=0
while { [ -n "$QUEUE_MULTI" ] || [ -n "$QUEUE_SINGLE" ]; } && [ "$cycle" -lt "$MAX_CYCLES" ]; do
    cycle=$((cycle+1))
    log "cycle $cycle  PENDING=$(squeue -u "$USER" -t PENDING -h 2>/dev/null | wc -l)  MULTI=[$QUEUE_MULTI] SINGLE=[$QUEUE_SINGLE]"
    placed=0
    for alloc in $(squeue -u "$USER" -h -t RUNNING -S "-L" -o "%i %D" | awk '$2>=2{print $1}'); do
        lm=$(left_min "$alloc")
        [ "$lm" -lt 90 ] && continue
        free=$(free_nodes_in "$alloc")
        cnt=$(echo "$free" | grep -c . || true)
        [ "${cnt:-0}" -lt 1 ] && continue
        log "  $alloc（剩${lm}min）空节点 $cnt 个：$(echo $free | tr '\n' ' ')"
        if [ "$cnt" -ge 2 ] && [ -n "$QUEUE_MULTI" ]; then
            take=2; [ "$cnt" -ge 4 ] && take=4
            nl=$(echo "$free" | head -"$take" | paste -sd,)
            item=${QUEUE_MULTI%% *}
            launch_train "$item" "$alloc" "$nl" "$take"; rc=$?
            [ "$rc" -eq 0 ] && QUEUE_MULTI=$(echo "$QUEUE_MULTI" | sed "s/^$item//;s/^ //")
            placed=1; break
        elif [ -n "$QUEUE_SINGLE" ]; then
            node=$(echo "$free" | head -1)
            item=${QUEUE_SINGLE%% *}
            launch_single "$item" "$alloc" "$node"; rc=$?
            [ "$rc" -eq 0 ] && QUEUE_SINGLE=$(echo "$QUEUE_SINGLE" | sed "s/^$item//;s/^ //")
            placed=1; break
        fi
    done
    [ -z "$QUEUE_MULTI" ] && [ -z "$QUEUE_SINGLE" ] && break
    [ "$placed" -eq 0 ] && sleep "$POLL" || sleep 60
done
log "候位 v2 退出：MULTI=[$QUEUE_MULTI] SINGLE=[$QUEUE_SINGLE]（都空=放完；未空=超时，需人接手）"
