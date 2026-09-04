# >>> stmux: module load brics tmux + 新建/接回 session (2026-08-11) >>>
# stmux <name>  =  module load brics/tmux/3.4 && tmux new -A -s <name>
#   -A: session 已存在就 attach 回去,不存在才新建(否则第二次敲会 duplicate session 报错)
# stmux         =  自动命名 stmux-<UTC时间戳> 并进入(2026-08-12 改,原为列表)
# stmux -l|ls   =  跨节点 breadcrumb 表 + 本节点 tmux ls
#
# 2026-08-12 无参数改为"自动起名新建"。动机: 起名是打断,大多数时候只是想要一张新的
#   工作区。时间戳取 UTC 是因为登录节点时区本身就是 UTC(date 与 date -u 输出一致),
#   不存在本地/UTC 错位。副作用要知道: 时间戳每秒不同 => 无参数每次都是新 session,
#   永远不会接回旧的; 要接回必须显式给名字。原列表功能整块挪到 -l/--list/ls。
#
# 2026-08-12 加 breadcrumb。动机(实测于当日): tmux server 是登录节点本地的,socket 在
#   /tmp/tmux-<uid>/ 而 /tmp 不跨节点;`ssh isambard` 走负载均衡,每次随机落到
#   login01/02/03/42/44/45 之一;登录节点之间 port 22 不通(2026-06-24 实测),无法跨
#   节点代查进程或 tmux。后果:落错节点时 `tmux ls` 返回空,看起来 session 死了,其实
#   活在别的节点上 —— 2026-08-11 的 hybrid 就是这样,它建在 login42,重连落到 login44。
#   解法:HOME=/projects/public/u6gb 在 Lustre 上跨节点共享,把 session→节点映射写成
#   一个小文件。这是 CLAUDE.md breadcrumb 原则的直接应用:不靠探测,靠写下来的事实。
_STMUX_BC="$HOME/.tmux-sessions.tsv"

# 2026-08-14 钉死 tmux 二进制。动机(当日实测): 一个 socket 只能有一个 tmux 二进制。
#   SessionStart hook 用绝对路径的 miniforge tmux 3.6 在 /tmp/tmux-<uid>/default 上
#   建了 server,而下面 stmux() 第一行 `module load brics/tmux/3.4` 把 3.4 顶到 PATH
#   最前 —— 3.4 的 client 连 3.6 的 server 握手对不上,报 "server exited unexpectedly"。
#   这句话是骗人的: server 一直活着(用 3.6 的客户端 `tmux ls` 当场能看到 session)。
#   误导性在于它把"协议不兼容"说成了"对方死了",于是人会去重建 session,而重建会
#   把还在里面跑的东西(当天是 fx488 恢复驱动)一起带走。
#   解法: tmux 一律走一个绝对路径。选 miniforge 那个,因为它不需要 module 就在默认
#   PATH 上,也正是 hook 用的那个。module load 一行保留(它还负责别的环境变量),
#   只是不再决定"用哪个 tmux"。
#   逃生口 —— 要接回历史上由 3.4 建的 server(别的登录节点可能还有):
#       _STMUX_TMUX=$(module load brics/tmux/3.4; command -v tmux) stmux <name>
#
# 2026-09-03 上面那条修复只修了三条路径中的一条,所以同一个报错又出现了(login44)。
#   一个 socket 上有三条路径各自独立地决定用哪个 tmux:
#       stmux()            _STMUX_TMUX 钉死 3.6 —— 但底下那句 `|| _STMUX_TMUX=tmux`
#                          在钉死的路径取不到时**静默退回 PATH**,而 PATH 第一位正是
#                          上一行 module load 刚顶上去的 3.4。那不是安全网,是上膛。
#       SessionStart hook  `command -v tmux`,完全看 PATH,根本没钉
#       用户随手敲 tmux    module load 之后就是 3.4
#   实测方向是单向的: 3.6 client 连 3.4 server rc=0,反过来 rc=1 报那句谎话。
#   所以修法是把"用哪个 tmux"收拢成单一事实来源,三条路径都去问同一个脚本:
_STMUX_HELPER="$HOME/.local/bin/stmux-tmux"
# 2026-09-03(下午,用户令) tmux 二进制写死成一个绝对路径,不再从帮助脚本推导。
#   在此之前这个变量有**两个**赋值点,只有一个做了校验:
#       这里          _STMUX_TMUX=$(帮助脚本),后面跟 ${...:-固定路径} 兜底 —— 保证非空
#       stmux() 里    _STMUX_TMUX=$(帮助脚本) || return 1        —— 没有兜底
#   当日实测的后果:socket 是陈旧文件时帮助脚本静默 exit 1 且不输出任何东西,
#   函数里那句便把这个**全局**变量写成空(赋值先于 || 完成,|| 拦不住它;且函数里
#   没有 local)。此后本终端每次 stmux 都拿空串当命令跑 -> rc=127 -> 被读成
#   "连不上 server",而同一屏的诊断表显示 OK、kill-stale 说"连得上不是 stale"
#   —— 三句话自相矛盾,真正错的只有读了空变量的那一句,但正是它把流程带进死路。
#   固定路径没有这个失败模式:它不依赖任何运行时状态,也就没有"取不到"这一说。
#   /tools/brics/... 是 root 所有、全站 1000+ 用户共享的那份,不会被删或改名。
#   帮助脚本保留,但只做三件不决定"用哪个 tmux"的事:--diagnose / --record / --kill-stale。
_STMUX_TMUX=/tools/brics/apps/linux-sles15-neoverse_v2/gcc-12.3.0/tmux-3.4-5vcftkte724cekyuashr2ex65c5fpfxj/bin/tmux
# 取不到就报错,不退回 PATH 上的 `tmux`: 那正是当初撞版本的来路,而报错信息谎称
# server 死了,人会去重建 session,把里面在跑的东西一起带走。宁可这里不动。
if [ ! -x "$_STMUX_TMUX" ]; then
    echo "stmux: 找不到可用的 tmux ($_STMUX_TMUX)。诊断: $_STMUX_HELPER --diagnose" >&2
fi

# 让裸敲的 `tmux` 也拿到同一个 3.4。3.6 改名之后 PATH 上一个 tmux 都没有了,
# 而 3.4 只存在于 module 树里 —— 这一句是它进 PATH 的唯一途径。
# 注意这与"stmux 函数里删掉 module load"不矛盾: 当时删是因为它引入了**第二个**
# 版本;现在 3.6 已不在 PATH 上,这一句带进来的正是唯一的那个,裸敲不再有歧义。
module load brics/tmux/3.4 >/dev/null 2>&1

# 记一行 <节点>TAB<session>TAB<UTC>;同 (节点,session) 覆盖。写临时文件再 mv,原子替换。
_stmux_bc_record() {
    local _n _t _tmp
    _n=$(hostname); _t=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    _tmp=$(mktemp "${_STMUX_BC}.XXXXXX" 2>/dev/null) || return 0
    {
        [ -f "$_STMUX_BC" ] && awk -F'\t' -v n="$_n" -v s="$1" '!($1==n && $2==s)' "$_STMUX_BC"
        printf '%s\t%s\t%s\n' "$_n" "$1" "$_t"
    } > "$_tmp" 2>/dev/null
    mv -f "$_tmp" "$_STMUX_BC" 2>/dev/null || rm -f "$_tmp"
}

# 打印全表。本节点的行用 has-session 当场验证并顺手清理死记录(GC 只发生在验证得了的
# 地方);别节点的行一律标 "?" 保留 —— 无法验证的东西不能当作死掉处理。
_stmux_bc_show() {
    local _here _n _s _t _st _dead _tmp
    _here=$(hostname)
    if [ ! -s "$_STMUX_BC" ]; then
        echo "  (breadcrumb 空: $_STMUX_BC — 用 stmux <name> 建 session 后自动登记)"
        return 0
    fi
    printf '  %-9s %-24s %-21s %s\n' NODE SESSION RECORDED_UTC STATUS
    _dead=0
    while IFS=$'\t' read -r _n _s _t; do
        case "$_n" in ''|'#'*) continue ;; esac
        if [ "$_n" = "$_here" ]; then
            if "$_STMUX_TMUX" has-session -t "=$_s" 2>/dev/null; then _st="ALIVE  (本节点已验证)"
            else _st="DEAD   (本节点已验证,清理)"; _dead=1; fi
        else
            _st="?      ssh $_n 后 stmux 确认"
        fi
        printf '  %-9s %-24s %-21s %s\n' "$_n" "$_s" "$_t" "$_st"
    done < "$_STMUX_BC"
    if [ "$_dead" = "1" ]; then
        _tmp=$(mktemp "${_STMUX_BC}.XXXXXX" 2>/dev/null) || return 0
        while IFS=$'\t' read -r _n _s _t; do
            case "$_n" in ''|'#'*) continue ;; esac
            [ "$_n" = "$_here" ] && ! "$_STMUX_TMUX" has-session -t "=$_s" 2>/dev/null && continue
            printf '%s\t%s\t%s\n' "$_n" "$_s" "$_t"
        done < "$_STMUX_BC" > "$_tmp"
        mv -f "$_tmp" "$_STMUX_BC" 2>/dev/null || rm -f "$_tmp"
    fi
}

stmux() {
    # 2026-09-03 删掉原来的 `module load brics/tmux/3.4 || return 1`。
    #   查过 modulefile,它只设 PATH / MANPATH / CMAKE_PREFIX_PATH / TMUX_ROOT /
    #   TERMINFO_DIRS。本函数用的是绝对路径 $_STMUX_TMUX,从不读 PATH 上的 tmux,
    #   所以这一行对 stmux 自己毫无作用 —— 它唯一的实际效果是把 3.4 顶到 PATH 最前,
    #   于是用户此后随手敲的裸 `tmux` 全变成 3.4,替下一次版本冲突埋好雷。
    #   删掉之后 PATH 上唯一的 tmux 就是 miniforge 3.6(/usr/bin/tmux 不存在),
    #   裸 `tmux` 自动就是对的那个。要 3.4: module load brics/tmux/3.4 自己敲。
    # 列表/帮助走子命令。ls 与"名为 ls 的 session"理论上撞名,真要接那个用
    # `tmux new -A -s ls`; 为这种概率不加 `--` 转义,少一层规则更好记。
    case "${1-}" in
        -l|--list|ls)
            echo "login node: $(hostname)   # tmux server 节点本地,session 只在它被创建的那台机器上"
            echo "--- 跨节点 breadcrumb ($_STMUX_BC) ---"
            _stmux_bc_show
            echo "--- 本节点 tmux ls ---"
            "$_STMUX_TMUX" ls 2>/dev/null || echo "  (本节点无 tmux session)"
            return 0
            ;;
        -h|--help)
            echo "stmux                自动命名 stmux-<UTC时间戳> 新建并进入(每次都是新的)"
            echo "stmux <name> [cmd]   新建或接回名为 <name> 的 session"
            echo "stmux -l | --list | ls   跨节点 breadcrumb 表 + 本节点 tmux ls"
            return 0
            ;;
    esac
    # 2026-09-03(下午,用户令) 每次用 stmux 先清掉二进制已被删除的 tmux server。
    #   登录节点之间 port 22 不通,没法一次扫全部,所以只能"落到哪台清哪台"。
    #   规则很窄:只杀 exe 带 " (deleted)" 的 —— 那种 server 没有任何客户端能连,
    #   留着只是把 socket 占死;版本对不上但二进制还在的一律不动(换个客户端可无损接回)。
    [ -x "$_STMUX_HELPER" ] && "$_STMUX_HELPER" --sweep-deleted

    local _name
    if [ $# -eq 0 ]; then
        _name=$(date -u +stmux-%Y%m%d-%H%M%S)
        echo "stmux: 未给名字,自动命名 $_name  (接回请显式 stmux $_name)"
    else
        _name=$1; shift
    fi
    _stmux_bc_record "$_name"
    # 记一笔"这个 socket 上的 server 是谁建的"。start-server 是幂等的:已有 server
    # 就什么都不做,没有就先把它拉起来 —— 必须在这里拉,因为下面 attach 分支是前台
    # 阻塞的,等它返回时人已经 detach 了,那时再记就晚了(而且记不到"谁建的")。
    if [ -x "$_STMUX_HELPER" ]; then
        "$_STMUX_TMUX" start-server 2>/dev/null
        "$_STMUX_HELPER" --record stmux 2>/dev/null
    fi
    if [ -n "$TMUX" ]; then
        # 已在 tmux 内: 不能嵌套 attach,先确保 session 存在(不接管终端)再 switch 过去
        # 注意 -A -d 在此不管用: session 已存在时 -A 会走 attach 分支,-d 拦不住
        "$_STMUX_TMUX" has-session -t "=$_name" 2>/dev/null || "$_STMUX_TMUX" new-session -d -s "$_name" "$@" || return 1
        "$_STMUX_TMUX" switch-client -t "=$_name"
    else
        # 先探一下这个 socket 连不连得上。tmux 在版本不匹配时报的是
        # "server exited unexpectedly" —— server 其实活着,那句话会把人骗去重建
        # session。宁可自己先查清楚,把真实原因和逃生命令给出来。
        if [ -S "${TMUX_TMPDIR:-/tmp}/tmux-$(id -u)/default" ] \
           && ! "$_STMUX_TMUX" ls >/dev/null 2>&1; then
            echo "stmux: $_STMUX_TMUX 连不上本节点已有的 tmux server。" >&2
            echo "       多半是版本不匹配,不是 server 死了 —— tmux 那句" >&2
            echo "       'server exited unexpectedly' 会骗人。诊断:" >&2
            [ -x "$_STMUX_HELPER" ] && "$_STMUX_HELPER" --diagnose >&2
            # 2026-09-03 用户令: 撞上就地解决,别让人再去敲第二条命令。
            # --kill-stale 自己会做三件事,所以这里直接交给它:
            #   1. server 其实连得上 -> 拒绝(那不是 stale)
            #   2. 记录里的二进制还在且连得上 -> 拒绝,提示无损接回
            #   3. 真是孤儿 -> 列出进程、读 /dev/tty 问一次确认、再 kill
            if [ -x "$_STMUX_HELPER" ] && "$_STMUX_HELPER" --kill-stale; then
                echo "stmux: 已清理,继续建 $_name" >&2
            else
                return 1
            fi
        fi
        "$_STMUX_TMUX" new -A -s "$_name" "$@"
    fi
}
# <<< stmux <<<

# >>> cc: 已停用,整块注释保留备查 (2026-08-12,用户要求) >>>
# 撤回理由: cc 就是 claude,不该顺带做 tmux 编排。定义见文件上方 alias cc='claude'。
# 下面这版曾强制在 tmux 内启动 claude,原始注释一并保留:
#
# # 为什么必须强制。2026-08-12 爬进程链实测,裸 SSH 下的链条是:
# #     systemd(1) -> sshd listener -> sshd[priv] -> sshd@pts/N -> -bash -> claude
# #   断网 → sshd 关闭 pts/N → 内核向该终端的前台进程组发 SIGHUP → bash 和 claude 一起死。
# #   放进 tmux 后链条变成:
# #     systemd(1) -> tmux server(ppid=1) -> -bash -> claude
# #   tmux server 不是 sshd 的后代,SIGHUP 在进程树上摸不到它,所以 claude 断网后照活。
# #   判据只有一条: `echo $TMUX` 为空 = 断网必死。
# # 注意"会话活着"不等于"工作继续": 没有输入 claude 不会自己醒来推进实验(CLAUDE.md
# #   2026-08-07)。长时间无人值守的计算必须落在 SLURM 作业里,不能指望会话活着。
# # 想明确绕过(例如只跑一句 claude -p): CC_NO_TMUX=1 cc ... 或直接 command claude ...
# # 想换 session 名: CC_TMUX_SESSION=hybrid cc
# cc() {
#     if [ -n "$TMUX" ] || [ -n "$CC_NO_TMUX" ]; then command claude "$@"; return; fi
#     local _s="${CC_TMUX_SESSION:-cc}" _cmd="claude"
#     [ $# -gt 0 ] && _cmd="claude $(printf '%q ' "$@")"
#     module load brics/tmux/3.4 || { echo "cc: module load tmux 失败,拒绝裸跑 claude" >&2; return 1; }
#     _stmux_bc_record "$_s"
#     if tmux has-session -t "=$_s" 2>/dev/null; then
#         echo "cc: 接回 tmux session '$_s' @ $(hostname)(里面可能已有 claude 在跑)"
#         tmux attach -t "=$_s"
#     else
#         echo "cc: 裸 SSH 会话 → 新建 tmux session '$_s' @ $(hostname) 再启动 claude"
#         # claude 退出后用 exec bash -l 保住 session,否则 window 关掉就没法 --resume 了
#         tmux new -s "$_s" "$_cmd; echo; echo '[claude 已退出;session 保留,可 claude --resume <id> 续]'; exec bash -l"
#     fi
# }
# <<< cc <<<

# schain: 提交 4 节点 24h 自续占位链 (u6gb-4-node-chain, 每一棒自动安排下一棒)
alias schain='sbatch /lus/lfs1aip2/projects/public/u6gb/tasks/u6gb_16_nodes_daily_log/four_node_chain_24h.sbatch --chain'
