# md-u6gb：服务器 Markdown 笔记 → Obsidian 同步

把 Isambard 上 `/projects/public/u6gb` 工作区里**我自己写的 Markdown 笔记**单向同步到本地 Mac，用 Obsidian 查看。GitHub 仅作中转。

## 架构（为什么不是"服务器每分钟 cron"）

```
服务器(写 md 的那一刻)
  └─ PostToolUse hook → .git-md-sync.sh <该文件>
       └─ 只对刚改的那一个 md： git add -f → commit → push(后台)
          不走目录树、不需要 cron、不需要常驻进程
  │  push
  ▼
GitHub: KangOxford/md-u6gb (HTTPS, 中转)
  │  pull
  ▼
Mac: launchd/cron 每分钟 git pull → Obsidian 看到更新
```

服务器端推送是**事件驱动**(写 md 即推)，唯一的定时器在 Mac 端(Mac 有 launchd、是本地盘、随便轮询)。原因：这台 login 节点没有 `crontab`，HPC 禁止 login 节点常驻轮询 daemon，且工作区根是 ~300 条目的 Lustre 重树，每分钟 `git add -A` 会造成元数据风暴。

## 追踪范围(allowlist)

只追踪 u6gb 自己的笔记，**不含** 19 个嵌套 repo 里的 md：

| 范围 | 路径 |
|------|------|
| Depth-1 notes | `*.md`(plans / findings / progress / learnt_lessons / CLAUDE.md ...) |
| Auto-memory | `.claude/projects/*/memory/*.md` |

安全保证：`.gitignore = *`(默认忽略一切，md 由脚本 `git add -f` 显式加入) + `git config status.showUntrackedFiles no`(`git status` 代价只与被追踪文件数成正比，不遍历重树)。误跑 `git add -A` 是空操作。

## 服务器端：手动同步

```bash
bash /projects/public/u6gb/.git-md-sync.sh --all   # 重新同步整个 allowlist
# 已加别名: mdsync
```

hook 会在 Claude 写 md 时自动触发；手动改(vim 等)后用 `mdsync` 补推。

## Mac 端配置(在 Mac 上执行)

1. Clone 成 Obsidian vault 的一个子目录：

```bash
mkdir -p ~/ObsidianVault
git clone https://github.com/KangOxford/md-u6gb.git ~/ObsidianVault/u6gb
```

2. Obsidian 打开 `~/ObsidianVault`(或直接打开 `~/ObsidianVault/u6gb`)作为 vault。

3. 每分钟自动 pull。任选其一：

```bash
# 方式 A: cron(最简单)
( crontab -l 2>/dev/null; echo '* * * * * cd ~/ObsidianVault/u6gb && /usr/bin/git pull --rebase --autostash -q' ) | crontab -
```

```xml
<!-- 方式 B: launchd ~/Library/LaunchAgents/com.u6gb.mdpull.plist 然后 launchctl load -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.u6gb.mdpull</string>
  <key>ProgramArguments</key>
  <array><string>/bin/zsh</string><string>-lc</string>
    <string>cd ~/ObsidianVault/u6gb &amp;&amp; /usr/bin/git pull --rebase --autostash -q</string></array>
  <key>StartInterval</key><integer>60</integer>
  <key>RunAtLoad</key><true/>
</dict></plist>
```

只读使用最稳：只要不在 Mac 上改这些 md，就几乎不会有冲突；`--rebase --autostash` 也能兜住偶尔的本地小改动。

## 文件清单

| 文件 | 作用 |
|------|------|
| `.git-md-sync.sh` | 事件驱动同步脚本(单文件 / `--all`) |
| `.gitignore` | `*`，安全忽略一切 |
| `README.md` | 本说明 |
