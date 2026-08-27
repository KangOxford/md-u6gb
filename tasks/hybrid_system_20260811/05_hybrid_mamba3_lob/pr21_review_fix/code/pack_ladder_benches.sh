#!/bin/bash
# inode 紧急释放（2026-08-27，配额 51.2M 打满、训练起跑被挡）：
# 把 8 月 12–18 日 ladder 时代的 bench2k 输出目录逐个打包成单文件 tar，
# 核对文件数一致后释放原树。内容零丢失（tar 可随时抽取，P5 的日志考古用
# `tar -xOf <tar> <member>` 照常读）。今晚（0827）的 ask2 目录不在匹配范围。
# 出处与授权先例：[[project_u6gb_inode_pack_dont_delete]]（0816 用户裁定打包释放）。
set -u
TD=/lus/lfs1aip2/projects/public/u6gb/tasks/hybrid_system_20260811/05_hybrid_mamba3_lob
packed=0; freed=0
verify() {   # $1=dir $2=tar → rc0 当且仅当 常规文件数与符号链接数分别一致
    local nf nl tf tl
    nf=$(lfs find "$1" -type f 2>/dev/null | wc -l)
    nl=$(lfs find "$1" -type l 2>/dev/null | wc -l)
    tf=$(tar -tvf "$2" 2>/dev/null | grep -c '^-')
    tl=$(tar -tvf "$2" 2>/dev/null | grep -c '^l')
    [ "$nf" = "$tf" ] && [ "$nl" = "$tl" ] && { echo "$nf"; return 0; }
    echo "src f=$nf l=$nl vs tar f=$tf l=$tl"; return 1
}
for d in "$TD"/bench2k_2026081[2-8]T*; do
    [ -d "$d" ] || continue
    case "$d" in *.tar) continue;; esac
    b=$(basename "$d")
    if [ ! -e "$d.tar" ]; then
        tar -cf "$d.tar" -C "$TD" "$b" 2>/dev/null || { echo "[STOP] $b tar 失败"; exit 3; }
    fi
    if out=$(verify "$d" "$d.tar"); then
        rm -rf "$d"
        packed=$((packed+1)); freed=$((freed+out))
        echo "[ok] $b  files=$out"
    else
        echo "[STOP] $b 核对不一致（$out），保留原树，人工检查"; exit 3
    fi
done
echo "[done] packed=$packed dirs, freed≈$freed inodes（每目录换回 1 个 tar inode）"
lfs quota -p 1483804535 /lus/lfs1aip2 2>/dev/null | sed -n '3p'
