# Lustre inode relief, 2026-09-04

## 速览（中文）

项目 `u6gb` 的 Lustre inode 配额打满（51,200,000/51,200,000），**空间没满**（129 TB / 200 TB），
满的是文件数。后果是**任何新建文件都失败**，覆盖已有文件不受影响。

释放 inode 必须真的删掉文件，而 `mv` 到同一文件系统内只是改名、**不释放 inode**。
出路是把目录移到**另一个文件系统**——`/home` 是 NFS（`data1.vastp2.isambard.ac.uk:/p2/home`，
15 PB、226 亿空闲 inode），跨文件系统的 `mv` 是「逐文件复制后删除」，因此确实释放 Lustre 这边的 inode。

本次我移走 1 个目录（54,159 个 inode），另一个会话同时在做同一件事、移走了另 1 个（56,814 个）。
**峰值 51,200,000 → 50,980,729**，空出约 219,000 个。**但到 2026-09-04T17:40Z 已回到 51,199,888——
释放是临时的**，其他活动很快把腾出来的重新占满。剩下三个归档共约 424,000 个 inode 仍在 Lustre 上，
最大的一块 `s5e_archive`（13,925,267 个，占配额 27%）没有动。

---

## Why this was needed

The project's Lustre quota is on **file count**, not bytes:

```
/lus/lfs1aip2   used 129.8 T of 200 T          <- space was never the problem
                files 51,200,000 of 51,200,000  <- this is what was full
```

A full inode quota fails **file creation** while leaving overwrites working, which is a
confusing signature: scripts that rewrite an existing output keep succeeding, and the
first thing that tries to create a new file dies with `OSError: [Errno 122] Disk quota
exceeded`. That is how it first surfaced here, from a run trying to write a new JSON.

Freeing inodes requires unlinking files. Renaming inside the same filesystem does not
help, because the inode is still allocated. The route that works is moving to a
**different filesystem**, where `mv` degrades to copy-then-unlink:

| mount | type | free inodes |
|---|---|---|
| `/lus/lfs1aip2` | lustre | project-capped at 51,200,000 |
| `/home` | nfs, `data1.vastp2.isambard.ac.uk:/p2/home` | 22,684,190,449 |

## What moved, and where

Every path below is written in full. `/projects/public/u6gb` is a symlink whose real
location is `/lus/lfs1aip2/projects/public/u6gb`.

### Moved by this session

| from | to | inodes | bytes |
|---|---|---|---|
| `/lus/lfs1aip2/projects/public/u6gb/.cargo/registry` | `/home/u6gb/kangli.u6gb/lustre_inode_relief_20260904/cargo_registry` | 54,159 | 1.2 G |

Chosen because it is a pure download cache for Rust crates and regenerates with
`cargo fetch`. Confirmed complete: the source path no longer exists on Lustre and the
destination is present on `/home`.

### Moved by a concurrent session

| from | to | inodes |
|---|---|---|
| `/lus/lfs1aip2/projects/public/u6gb/_s5e_inference_archive_20260904` | `/home/u6gb/kangli.u6gb/_lustre_offload_20260904/_s5e_inference_archive_20260904` | 56,814 |

That session also left two helper scripts in its home,
`/home/u6gb/kangli.u6gb/archive_scored_inference.sh` and
`/home/u6gb/kangli.u6gb/archive_superseded_bench.sh`.

### Started, interrupted, and left as partial copies

A session teardown killed a four-directory move mid-flight. Because a cross-filesystem
`mv` copies then unlinks per file, an interruption leaves part of the tree on both sides.
Two partial copies exist. **Neither cost any Lustre inodes** — they sit on `/home` — and
in both cases a complete copy of the data exists elsewhere, so nothing was lost:

| partial copy on `/home` | the complete copy |
|---|---|
| `/home/u6gb/kangli.u6gb/lustre_inode_relief_20260904/PARTIAL_of__s5e_inference_archive_20260902__source_still_on_lustre__20260904T155050Z` | the original, still on Lustre |
| `/home/u6gb/kangli.u6gb/_lustre_offload_20260904/PARTIAL_duplicate_of__s5e_inference_archive_20260904__complete_copy_is_sibling__20260904T155050Z` | its sibling directory, moved successfully |

Those two names were written by the concurrent session, which found the fragments and
labelled where the good copy was. They can be deleted once someone confirms them, which
this session cannot do.

### Not moved

| path | inodes | why it stayed |
|---|---|---|
| `/lus/lfs1aip2/projects/public/u6gb/s5e_archive` | 13,925,267 | 27% of the whole quota in one directory, untouched since 2026-08-15. Moving 13.9M files across filesystems is hours of sustained metadata traffic, which is the load pattern that had group jobs suspended on 2026-05-08. It needs batching and pacing, and a decision from someone who knows whether the archive is dead. |
| `/lus/lfs1aip2/projects/public/u6gb/_s5e_inference_archive_20260902` | 313,719 | move interrupted; source intact |
| `/lus/lfs1aip2/projects/public/u6gb/_s5e_deprecated_20260902` | 64,146 | move never reached it |
| `/lus/lfs1aip2/projects/public/u6gb/_s5e_scored_archive_20260904` | 46,269 | move never reached it |
| `/lus/lfs1aip2/projects/public/u6gb/.conda/envs` (11 environments) | 350,576 | in use. `latex_env` is the LaTeX toolchain documented in `CLAUDE.md`; the others belong to work this session cannot vouch for. Directory mtime is **not** evidence of disuse: running a program out of an environment does not update it. |

## What it achieved, and for how long

```
51,200,000   peak, quota exhausted, new files failing
50,980,729   after the two completed moves   (about 219,000 free)
51,199,888   2026-09-04T17:40Z               (about 112 free)
```

**The relief lasted a few hours.** Roughly 219,000 inodes were freed and other activity
consumed them again within the same afternoon. Moving one cache and one archive is not a
fix at this rate of consumption; either the large archive goes, or something is producing
inodes faster than incremental cleanup can recover them.

## The concurrent-session collision

Two sessions worked the same problem without knowing about each other, and both moved
`_s5e_inference_archive_20260902` and `_s5e_inference_archive_20260904` at once. The
result was one complete copy plus one redundant fragment per directory, needing a third
pass to work out which was which. The two `PARTIAL_*` / `REDUNDANT_*` names in the tables
above are that third pass.

The cheap prevention is a destination directory whose name carries the owning session, so
two movers cannot land in the same place, and a claim written down before the first `mv`
rather than after the first conflict.

## Reproducing the measurements

```bash
# quota, by project id (u6gb is 1483804535)
lfs quota -p 1483804535 /lus/lfs1aip2

# inode count of one directory, metadata-batched: do NOT use find or du -sh here
lfs find /lus/lfs1aip2/projects/public/u6gb/<dir> | wc -l

# confirm /home really is a different filesystem, or the move frees nothing
stat -f -c '%n type=%T' /home/u6gb/kangli.u6gb /lus/lfs1aip2
```
