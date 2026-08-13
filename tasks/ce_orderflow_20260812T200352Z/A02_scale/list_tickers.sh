#!/bin/bash
# 在计算节点上挂一次留出月的 shard，把 ticker 目录名列出来。
# 不在登录节点做：squashfuse 挂载 + 一次目录列举应当发生在分配内。
set -eu
SH=/lus/lfs1aip2/projects/public/s5e/quant_team/lob_preproc_sp500_squashfs/shard_${1:-2026-01}.squashfs
M=/tmp/$USER/dfmlist.$$
mkdir -p $M
squashfuse $SH $M
ls $M | sort
fusermount -u $M && rmdir $M
