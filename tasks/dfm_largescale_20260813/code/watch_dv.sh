#!/bin/bash
# 数据量网格监控：等 1:10 档（13880 步）跑完。
#
# 退出条件同样要能区分「没启动」和「跑完」：这里直接看**产物**
# （residual msgpack），而不是看 squeue 里有没有 step。产物存在是单调的、
# 不会因为重启而回退。
cd "$(dirname "$0")/.."
target=13880
while :; do
    done_n=0; line=""
    for f in logs/dv_dv10_*u.log logs/dv_dv10_*_s[01].log; do
        [ -f "$f" ] || continue
        t=$(basename "$f" .log | sed 's/^dv_//')
        s=$(grep -oE '^\[c0\][[:space:]]+step [0-9]+' "$f" | tail -1 | grep -oE '[0-9]+$')
        s=${s:-0}
        [ "$s" -ge "$target" ] && done_n=$((done_n+1))
        line="$line$(printf '%-20s %6s/%s\n' "$t" "$s" "$target")"
    done
    echo "--- $(date -u +%H:%M:%SZ)  1:10 档完成 $done_n ---"
    printf '%s' "$line"
    [ "$done_n" -ge 4 ] && { echo "=== 1:10 档至少 4 臂完成 ==="; break; }
    sleep 300
done
ls -la residuals/ 2>/dev/null | grep dv10 | head -20
