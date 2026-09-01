#!/bin/bash
# 把 slime_launch/*.sh 里写死的作者机器路径改成「可被环境覆盖的默认值」。
#
# 为什么不是直接改成本机路径:那样脚本就只能在本机跑,回馈不了上游,
# 作者那边也会坏。改成 X=${X:-<作者原值>} 之后:
#   作者机器上不设环境变量 -> 行为与现在完全一致
#   本机 source site_env.sh 先 export -> 覆盖生效
# 这也是「配置要真的到达它控制的那条路径」的最小形态 —— 现在是直接赋值,
# 外部 export 会被无声覆盖掉,配置根本到不了。
#
# 幂等:已经是 ${X:-...} 形态的行不再处理。
set -u
LAUNCH=${LAUNCH:-/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/LDM-rl/rl/slime_launch}
cd "$LAUNCH" || exit 1

# 要参数化的变量名(全部是脚本头部的直接赋值)
VARS="REPO_ROOT SLIME_ROOT MEGATRON_ROOT CONDA_PREFIX MODEL_HF SAVE CUDART_BLOCK"

changed=0
for f in *.sh; do
    before=$(md5sum "$f" | cut -d' ' -f1)
    for v in $VARS; do
        # X=/some/path  ->  X=${X:-/some/path}   (跳过已经是 ${X:- 形态的)
        sed -i -E "s|^([[:space:]]*)${v}=([^\$\{].*)$|\1${v}=\$\{${v}:-\2\}|" "$f"
    done
    # /root/cudart_block 出现在 mkdir/touch/LD_LIBRARY_PATH 三处,统一走变量
    sed -i "s|/root/cudart_block|\${CUDART_BLOCK:-/root/cudart_block}|g" "$f"
    # 上一条会把已经声明的 CUDART_BLOCK=${CUDART_BLOCK:-...} 也套一层,修回来
    sed -i -E 's|CUDART_BLOCK=\$\{CUDART_BLOCK:-\$\{CUDART_BLOCK:-([^}]*)\}\}|CUDART_BLOCK=${CUDART_BLOCK:-\1}|' "$f"
    after=$(md5sum "$f" | cut -d' ' -f1)
    if [ "$before" != "$after" ]; then echo "  改了 $f"; changed=$((changed+1)); fi
done
echo "共改动 $changed 个脚本"

echo; echo "=== 改后仍剩的写死路径(应只余注释与 --hf-checkpoint/--ref-load 等命令行参数) ==="
grep -nE "/mnt/data0|/root/" *.sh | grep -v '^\S*:[0-9]*:#' | head -20

echo; echo "=== 语法检查 ==="
for f in *.sh; do bash -n "$f" 2>&1 | head -2 && echo "  OK $f"; done
