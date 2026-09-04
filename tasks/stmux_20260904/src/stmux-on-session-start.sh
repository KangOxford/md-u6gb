#!/bin/bash
# SessionStart hook (2026-08-14, 用户要求"每次登录 Isambard 都敲 stmux")
#
# 保证当前节点上存在一张属于 Claude 的 tmux session,并把它的名字告诉 Claude。
#
# 为什么不是直接调 `stmux`:
#   1. stmux 是 ~/.bashrc 里的 shell function,不是可执行文件 —— hook 的非交互
#      子 shell 里这个名字根本不存在;它依赖的 `module` 也是 Lmod 注入的 function。
#   2. stmux 无参走 `tmux new -A`,即 attach 分支。Claude 的 Bash 工具没有 tty
#      (`tty` → not a tty),attach 必报 "open terminal failed: not a terminal"。
#   3. stmux 无参用 UTC 时间戳命名,每次都是新 session。SessionStart 在
#      startup/resume/clear/compact 都会触发,照搬会堆出一地孤儿 session。
#      这里改用按节点固定的名字,所以是"建或复用",重连能接回原来那张。
#
# 失败一律 exit 0:这个 hook 绝不允许拖住/挡住会话启动。

_STMUX_BC="${HOME}/.tmux-sessions.tsv"

# ── 1. 砍掉继承来的 $TMUX ────────────────────────────────────────────────
# $TMUX 是 socket 路径,不是"我在不在 tmux 里"的布尔量。Claude 经常跑在计算节点上
# (srun/sbash 进去的),而 $TMUX 是从登录节点 shell 继承来的,它指向的 socket 在
# /tmp 里 —— 而 /tmp 是节点本地的,在计算节点上不存在。留着它,tmux 会照那个路径
# 在计算节点上另起一台幽灵 server,`tmux ls` 默认 socket 完全看不见,现象酷似
# "session 没建起来"。2026-08-12 的 `fake` socket 事故就是这个。
unset TMUX TMUX_PANE

# ── 2. 找 tmux 二进制 ───────────────────────────────────────────────────
# 2026-09-03 改:原来这里是 `command -v tmux`,即"PATH 上的第一个"。那是个 bug。
#   tmux 的 socket 是 per-uid 的(/tmp/tmux-<uid>/default),这台机器上你所有的 tmux
#   共用它 —— 谁先建 server,谁就定死了这个 socket 的协议版本。而本环境里有两个
#   tmux(miniforge 3.6 / module brics 3.4),兼容是单向的:3.6 client 连 3.4 server
#   可以,3.4 client 连 3.6 server 报 "server exited unexpectedly"。那句话是谎话,
#   server 活着,只是握手对不上 —— 但它会把人骗去重建 session,连带杀掉里面在跑的东西。
#   看 PATH 就意味着:这个 hook 用哪个版本建 server,取决于启动 claude 时那个 shell
#   有没有 module load 过。同一个 socket 上,hook / stmux / 用户裸敲的 tmux 三条路径
#   各自独立决定版本,只钉死其中一条(2026-08-14 只钉了 stmux)修不好。
#   现在三条都问同一个脚本。
_STMUX_HELPER="${HOME}/.local/bin/stmux-tmux"
[ -x "$_STMUX_HELPER" ] || _STMUX_HELPER=/projects/public/u6gb/.local/bin/stmux-tmux
# 2026-09-03(下午,用户令) 先清掉二进制已被删除的 tmux server,再去找该用哪个。
#   顺序不能反:那种 server 占着 socket,任何客户端都连不上,下面的 new-session 必失败,
#   而 hook 是 exit 0 静默的 —— 会表现为"工作区莫名其妙没建起来"。
#   stderr 丢掉:hook 的输出会进 Claude 的上下文,清理是例行动作不值得每次刷屏。
[ -x "$_STMUX_HELPER" ] && "$_STMUX_HELPER" --sweep-deleted 2>/dev/null

_tmux=""
[ -x "$_STMUX_HELPER" ] && _tmux=$("$_STMUX_HELPER" 2>/dev/null)
if [ -z "$_tmux" ] || [ ! -x "$_tmux" ]; then
    # helper 不在(别的机器/别的账号)才退回原来的找法
    _tmux=$(command -v tmux 2>/dev/null)
    if [ -z "$_tmux" ]; then
        # module 是 Lmod 注入的 shell function,非交互子 shell 里没有,得先 source init
        [ -r /opt/cray/pe/lmod/lmod/init/bash ] && . /opt/cray/pe/lmod/lmod/init/bash >/dev/null 2>&1
        module load brics/tmux/3.4 >/dev/null 2>&1
        _tmux=$(command -v tmux 2>/dev/null)
    fi
fi
[ -n "$_tmux" ] || exit 0

# ── 3. 建或复用 ─────────────────────────────────────────────────────────
_node=$(hostname 2>/dev/null) || exit 0
_name="claude-${_node}"

# 不用 `new-session -A -d`:-A 命中已存在的 session 时会走 attach 分支,-d 拦不住
# (man 只用一句从句写"-A 时 -D 才 behave like -d"),无 tty 下第二次调用直接 rc=1。
# has-session 的 -t 必须加 "=" 前缀强制精确匹配,否则前缀/fnmatch 匹配会误命中。
_state=reused
if ! "$_tmux" has-session -t "=${_name}" 2>/dev/null; then
    "$_tmux" new-session -d -s "${_name}" -c "${CLAUDE_PROJECT_DIR:-$HOME}" >/dev/null 2>&1 || exit 0
    _state=created
fi

# 记一笔"这个 socket 上的 server 是谁建的" —— 见 ~/.tmux-servers/README。
# 这件事只存在于跑着的进程里,ls/cat/读脚本都看不到,所以建完就写下来。
[ -x "$_STMUX_HELPER" ] && "$_STMUX_HELPER" --record hook 2>/dev/null

# ── 4. 登记 breadcrumb,让用户的 `stmux -l` 看得见这张 session ───────────
# 格式与 .bashrc 里的 _stmux_bc_record 完全一致:<节点>TAB<session>TAB<UTC>,
# 同 (节点,session) 覆盖,写临时文件再 mv 做原子替换。
_t=$(date -u +%Y-%m-%dT%H:%M:%SZ)
_tmp="${_STMUX_BC}.$$"
{
    [ -f "$_STMUX_BC" ] && awk -F'\t' -v n="$_node" -v s="$_name" '!($1==n && $2==s)' "$_STMUX_BC"
    printf '%s\t%s\t%s\n' "$_node" "$_name" "$_t"
} > "$_tmp" 2>/dev/null
mv -f "$_tmp" "$_STMUX_BC" 2>/dev/null

# ── 5. 把结果交回 Claude 的上下文 ───────────────────────────────────────
cat <<EOF
[stmux] 本节点 ${_node} 上的 Claude tmux 工作区: ${_name} (${_state})
凡是"必须活过本会话"的进程都放进去跑,比 setsid nohup 多一样:输出还能回看。
  ${_tmux} send-keys   -t '=${_name}:' '<命令>' Enter
  ${_tmux} capture-pane -p -t '=${_name}:' | tail -40
注意那个尾冒号:send-keys/capture-pane 的 -t 收的是 target-**pane**,而
has-session 收的是 target-**session**,两者解析规则不同。裸的 '=${_name}'
对前者会报 "can't find pane"。'=name:' 才既精确匹配又落到活动 pane 上。
用户接回:先到 ${_node},再 stmux ${_name}(tmux server 是节点本地的,别的节点看不见)。
EOF
exit 0
