#!/bin/bash
# 把确认冷掉的 conda 环境从 /home(101G 硬配额,已撞顶)打包进 Lustre。
#
# 为什么打包而不是 rsync 逐文件搬:两个文件系统的约束正好相反 ——
#   /home (VAST) : 空间 101G 满,inode 只用 8.9% (893k/10000k)
#   Lustre       : 空间 83T 余,inode 只剩 2.4% (1.24M/51.2M)
# 逐文件搬会把 7.5 万个 inode 从富余的一侧挪到紧缺的一侧。一个 tar = 1 个 inode。
#
# tar --remove-files 的语义:每个文件**写进归档之后**才解除源端链接。
# 与 rsync --remove-source-files 同级的安全性,数据完整保留在 tar 里(tar -xf 可还原)。
#
# 冷的依据(2026-09-01 逐一查证):
#   glm53-vllm / glm53-vllm_cu130_deprecated / cu13-test
#     -> 在跑的 6220558/6220580 用的是 apptainer SIF(images/vllm-glm53-arm64-cu129.sif),
#        不是这几个 conda 环境;源码构建那几轮已被 SIF 取代。
#   ldm-nanogpt -> run_repro_arm.sh:54 正在用,**不动**。
set -u
COLD=/lus/lfs1aip2/projects/public/u6gb/envs_cold
SRC=/home/u6gb/kangli.u6gb/envs
mkdir -p "$COLD"

say() { echo "[$(date -u +%H:%M:%S)] $*"; }

# glm53-vllm 上一次被裸 mv 打断,现在 /home 与 Lustre 各有一半。
# 先把 /home 的残余并回 Lustre 那棵树,凑齐之后再整棵打包。
if [ -d "$SRC/glm53-vllm" ] && [ -d "$COLD/glm53-vllm" ]; then
    say "glm53-vllm 两侧都有(上次搬迁被打断)-> 先并回 Lustre"
    rsync -a --remove-source-files "$SRC/glm53-vllm/" "$COLD/glm53-vllm/" 2>&1 | tail -3
    find "$SRC/glm53-vllm" -depth -type d -exec rmdir {} \; 2>/dev/null
    left=$(find "$SRC/glm53-vllm" 2>/dev/null | wc -l)
    say "  /home 残留 $left 项 $([ "$left" = 0 ] && echo '(已清空)' || echo '(还有东西,没动)')"
fi

# 把 Lustre 上已经摊开的那几棵树收成 tar,把 inode 还回去
for e in glm53-vllm glm53-vllm_cu130_deprecated_20260827T0000Z; do
    [ -d "$COLD/$e" ] || continue
    [ -f "$COLD/$e.tar" ] && { say "$COLD/$e.tar 已存在,跳过"; continue; }
    n=$(find "$COLD/$e" -type f 2>/dev/null | wc -l)
    say "打包 Lustre 上的 $e ($n 个文件) -> $e.tar"
    tar --remove-files -cf "$COLD/$e.tar" -C "$COLD" "$e" || { say "  !! tar 失败,源保留"; continue; }
    m=$(tar -tf "$COLD/$e.tar" 2>/dev/null | grep -vc '/$')
    say "  归档内 $m 个文件(源端 $n);大小 $(du -h "$COLD/$e.tar" | cut -f1)"
done

# /home 上还没搬过的冷环境:直接打成 tar 落 Lustre
for e in glm53-vllm_cu130_deprecated_20260827T0000Z cu13-test; do
    [ -d "$SRC/$e" ] || { say "$e: /home 上已不在,跳过"; continue; }
    TARGET="$COLD/${e}.home.tar"
    [ -f "$TARGET" ] && TARGET="$COLD/${e}.home2.tar"
    n=$(find "$SRC/$e" -type f 2>/dev/null | wc -l)
    say "打包 /home 上的 $e ($n 个文件,$(du -sh "$SRC/$e" | cut -f1)) -> $(basename $TARGET)"
    tar --remove-files -cf "$TARGET" -C "$SRC" "$e" || { say "  !! tar 失败,源保留"; continue; }
    find "$SRC/$e" -depth -type d -exec rmdir {} \; 2>/dev/null
    m=$(tar -tf "$TARGET" 2>/dev/null | grep -vc '/$')
    say "  归档内 $m 个文件(源端 $n);大小 $(du -h "$TARGET" | cut -f1)"
    say "  /home 配额: $(quota -s 2>/dev/null | tail -1 | awk '{print $1}')"
done

say "=== 完成 ==="
quota -s 2>/dev/null | tail -2
lfs quota -h -p 1483804535 /lus/lfs1aip2 2>/dev/null | tail -1
