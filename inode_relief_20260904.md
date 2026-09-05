# Lustre inode relief, 2026-09-04

## 速览（中文）

项目 `u6gb` 的 Lustre inode 配额打满（51,200,000/51,200,000），**空间没满**（129 TB / 200 TB），
满的是文件数。后果是**任何新建文件都失败**，覆盖已有文件不受影响。

释放 inode 必须真的删掉文件，而 `mv` 到同一文件系统内只是改名、**不释放 inode**。
出路是把目录移到**另一个文件系统**——`/home` 是 NFS（`data1.vastp2.isambard.ac.uk:/p2/home`，
15 PB、226 亿空闲 inode），跨文件系统的 `mv` 是「逐文件复制后删除」，因此确实释放 Lustre 这边的 inode。

本次我移走 1 个目录（54,159 个 inode），另一个会话同时在做同一件事、移走了另 1 个（56,814 个），
**但后者当天下午被移回了 Lustre**（见正文）。峰值 51,200,000 → 50,980,729，空出约 219,000 个，
到 17:40Z 又回到 51,199,888。

**最新一次实测：2026-09-05T03:46Z 为 50,461,324 / 51,200,000，空余约 739,000——配额当前并不告急。**
这个好转来自别处的清理，不是本文记录的这两次移动。别处若还写着「inode 只剩 0」，那是 09-04 的状态。

最大的一块 `s5e_archive`（13,925,267 个，占配额 27%）没有动，它仍是唯一能单独改变全局的一步。
另有一项**位置未知**：`_s5e_inference_archive_20260902` 在 09-05T03:46Z 的复核中，于原路径、
两个 relief 目录、以及 `/home/u6gb/kangli.u6gb` 前两层都没找到，只余一个 7 项的碎片副本（作为证据保留）。
**「这几处没找到」就是全部结论——不能据此认定它已被删除**；未做深度搜索（那是站点规则禁止的元数据负载），
所以未搜索到的位置仍然可能存在。

> **更新 2026-09-05T04:55Z（交叉引用，不改上面那段）：** 同目录的
> `/lus/lfs1aip2/projects/public/u6gb/deletion_audit_20260905.md`（生成日期 2026-09-05）
> 记述了这个目录的去向。**该文档称**：它由执行删除的那次会话以后台 `rm -rf` 删除、不可逆；
> 被删内容是 10 个 run 的 `evaluation/inference/` 中间产物（313,654 文件 / 7.2 GB）；
> 评测结论本身完整存活于现存 31 个源目录中。
> **该文档自述的依据是 grep 那次会话的 jsonl 记录**，外加只读的 `lfs find`/`stat`/`cat`/`md5sum`。
>
> 上面那段 03:46Z 的「位置未知」**按原样保留，它当时并没有判断错**：
> 「这几处没找到」确实不足以认定删除。**补上缺口的是那份 jsonl 记录，不是任何 `stat` 结果**——
> 路径不存在在任何时候都不能推出谁删的、为什么删。
> 该 audit 文档自身状态为**部分开放**（其第 6 节的 12 个子目录清单当时仍由后台 subagent 重建中），
> 故此处只做转述与指路，不替它下最终结论。

配额读数一律带日期：**09-05T03:46Z 实测 50,461,324 / 51,200,000（空余约 739,000），当前不告急**；
口径是 `lfs quota -p 1483804535 /lus/lfs1aip2` 的 files 列（项目级、按文件数，不是空间）。
`/home/u6gb/kangli.u6gb/FACTS.md`（09-04T18:03Z）里那句「inode 只剩 0」是**当时的快照**，
不是长期状态，照它决策前需重新测量。

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

### Moved by a concurrent session, then moved back

| from | to (2026-09-04T15:50Z) | then to (verified 2026-09-05T03:46Z) | inodes |
|---|---|---|---|
| `/lus/lfs1aip2/projects/public/u6gb/_s5e_inference_archive_20260904` | `/home/u6gb/kangli.u6gb/_lustre_offload_20260904/_s5e_inference_archive_20260904` | `/lus/lfs1aip2/projects/public/u6gb/_home_space_relief_20260904T163000Z/_lustre_offload_20260904/_s5e_inference_archive_20260904` | 56,814 |

The offload was **reversed** later the same day: the whole
`/home/u6gb/kangli.u6gb/_lustre_offload_20260904` directory was moved back onto Lustre and
now sits inside `_home_space_relief_20260904T163000Z`, a name suggesting the concurrent
session hit a space constraint on `/home` and undid the offload to relieve it. The inodes
that move had freed therefore returned to the Lustre project.

That session also left helper scripts in its home:
`/home/u6gb/kangli.u6gb/archive_scored_inference.sh`,
`/home/u6gb/kangli.u6gb/archive_superseded_bench.sh`,
`/home/u6gb/kangli.u6gb/archive_arm.sh` and
`/home/u6gb/kangli.u6gb/archive_arm_v2.sh`.

### Started, interrupted, and left as partial copies

A session teardown killed a four-directory move mid-flight. Because a cross-filesystem
`mv` copies then unlinks per file, an interruption leaves part of the tree on both sides.
Two partial copies exist. **Neither cost any Lustre inodes** — they sit on `/home` — and
in both cases a complete copy of the data exists elsewhere, so nothing was lost:

| partial copy, located 2026-09-05T03:46Z | the complete copy |
|---|---|
| `/home/u6gb/kangli.u6gb/lustre_inode_relief_20260904/PARTIAL_of__s5e_inference_archive_20260902__source_still_on_lustre__20260904T155050Z` (7 entries) | **could no longer be located, see below** |
| `/lus/lfs1aip2/projects/public/u6gb/_home_space_relief_20260904T163000Z/_lustre_offload_20260904/PARTIAL_duplicate_of__s5e_inference_archive_20260904__complete_copy_is_sibling__20260904T155050Z` | its sibling in the same directory, intact |

Those two names were written by the concurrent session, which found the fragments and
labelled where the good copy was. The second fragment travelled back to Lustre with the
rest of the offload. Neither can be deleted by this session.

> **Open item, status: location unknown, raised 2026-09-05T03:46Z.**
> `_s5e_inference_archive_20260902` was not found by any of the checks run at that time.
> What was checked, and only this: its original path
> `/lus/lfs1aip2/projects/public/u6gb/_s5e_inference_archive_20260902`; both relief
> directories; and the top two levels of `/home/u6gb/kangli.u6gb`. It was absent from all
> of them, apart from the 7-entry fragment in the row above.
>
> **"Not found by these checks" is the whole finding. It does not establish that the
> directory was deleted**, and no conclusion about its fate should be drawn from this
> note. A deep filesystem search was deliberately not run, because that is the metadata
> load pattern the site rules prohibit, so an unsearched location remains a live
> possibility. Recorded for someone who knows where it went.
>
> The 7-entry fragment is kept as evidence and should not be discarded while this item is
> open, since it is the only material this note can point to.

> ---
>
> **Update 2026-09-05T04:55Z, cross-reference. The note above is left exactly as written.**
> A sibling record in the same directory,
> `/lus/lfs1aip2/projects/public/u6gb/deletion_audit_20260905.md` (dated 2026-09-05),
> accounts for this directory. **That document states** it was removed by a background
> `rm -rf` issued by the session that wrote the audit, that the removal is irreversible,
> that what was removed was the `evaluation/inference/` payload of 10 runs
> (313,654 files, 7.2 GB), and that the benchmark conclusions themselves survive intact
> in 31 source directories that still exist. **The evidence that document cites for the
> removal is a grep of that session's own jsonl transcript**, alongside read-only
> `lfs find` / `stat` / `cat` / `md5sum`.
>
> The 03:46Z note above was not wrong: "not found by these checks" genuinely does not
> establish deletion, and **an absent path never identifies who removed something or
> why**. What closes the gap is the jsonl transcript, not any `stat` result. That audit
> is itself marked partially open (its §6 list of the 12 archived subdirectories was
> still being reconstructed when it was written), so this is a pointer and a summary of
> its claims, not a verdict adopted here.

### Not moved

| path | inodes | why it stayed |
|---|---|---|
| `/lus/lfs1aip2/projects/public/u6gb/s5e_archive` | 13,925,267 | 27% of the whole quota in one directory, untouched since 2026-08-15. Moving 13.9M files across filesystems is hours of sustained metadata traffic, which is the load pattern that had group jobs suspended on 2026-05-08. It needs batching and pacing, and a decision from someone who knows whether the archive is dead. |
| `/lus/lfs1aip2/projects/public/u6gb/_s5e_inference_archive_20260902` | 313,719 | move interrupted, source intact **at the time of writing**; by 2026-09-05T03:46Z the path no longer existed, see the open item above (and the cross-reference added there on 2026-09-05 to `deletion_audit_20260905.md`, which accounts for it) |
| `/lus/lfs1aip2/projects/public/u6gb/_s5e_deprecated_20260902` | 64,146 | move never reached it; still present 2026-09-05T03:46Z |
| `/lus/lfs1aip2/projects/public/u6gb/_s5e_scored_archive_20260904` | 46,269 | move never reached it; still present 2026-09-05T03:46Z |
| `/lus/lfs1aip2/projects/public/u6gb/.conda/envs` (11 environments) | 350,576 | in use. `latex_env` is the LaTeX toolchain documented in `CLAUDE.md`; the others belong to work this session cannot vouch for. Directory mtime is **not** evidence of disuse: running a program out of an environment does not update it. |

## What it achieved, and for how long

```
51,200,000   peak, quota exhausted, new files failing
50,980,729   after the two completed moves        (about 219,000 free)
51,199,888   2026-09-04T17:40Z                    (about 112 free)
50,461,324   2026-09-05T03:46Z                    (about 739,000 free)
```

**Every line above is a dated snapshot, not a standing fact.** The first three describe a
few hours on 2026-09-04, during which roughly 219,000 inodes were freed and consumed
again. By the next morning the project had about 739,000 free, from cleanup elsewhere
rather than from anything recorded here.

What the numbers are, so a later reader can reproduce or supersede them:

| | |
|---|---|
| command | `lfs quota -p 1483804535 /lus/lfs1aip2` |
| scope | the **project** quota for id 1483804535 (`/lus/lfs1aip2/projects/public/u6gb`), not a user or group quota |
| the number quoted | the **files** column, i.e. inodes, against a cap of 51,200,000 |
| space, separately | 137 TB of 200 TB used at the same reading, and never the binding limit here |

**As of 2026-09-05T03:46Z the quota is not exhausted.** Any note that still reads
"Lustre inode quota ~0 free" is quoting 2026-09-04 and needs re-measuring before it is
acted on. One such note exists and is easy to hit:
`/home/u6gb/kangli.u6gb/FACTS.md`, written 2026-09-04T18:03Z, tells readers to write to
`/home` because "Lustre project inode quota ~0 free". That was true when written. It is a
dated snapshot, and this note does not amend it, since it belongs to another session.

The structural point survives the recovery: two directories moved by hand, one of which
was later moved back, do not add up to a fix at the rate this project creates files. The
largest single object, `s5e_archive` at 13,925,267 inodes, is untouched and remains the
only move that would change the picture by itself.

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
