#!/usr/bin/env python3
"""nodelock —— 「这个节点是留给谁的」的唯一真相。

为什么要有它
------------
`idle` 只说明这张卡此刻没在算，**不说明它没有主人**。多个会话共用同一批 chain
分配时，把 idle 读成「可抢」会出事：2026-08-12 `wm_ft_b` 首启就是在 init 与第一个
backward 之间，被邻居的 88 GiB 预分配挤爆的。Slurm 知道谁**持有**一个节点，
不知道它**本来是留给谁**的 —— 这个缺口只能靠声明来补。

为什么是两个文件，以及为什么本模块是桥
--------------------------------------
两个已有的消费者各自先落地了自己的格式，而且都有道理：

  gtop   tasks/node_status/gpu_locks.json   {"locks":[{node,gpus,jobid,who,note,at}]}
         富格式：有逐卡粒度和 jobid 过期
  sgpur  ~/.config/nodelocks.json           {host: session}
         扁平：因为它的读法是把**每个顶层 key 当成节点名**，塞任何额外的 key
         （`"locks": [...]` 之类）都会在网页上多出一个叫 `locks` 的幽灵节点

把任何一边改成另一边，都要动别人正在写的代码，还会来回翻。所以本模块**两边都写**：
富格式是真相，扁平文件是它的投影。两个消费者一行都不用改，看到的却是同一份声明。

  写：do_lock / do_unlock -> 同时落 gpu_locks.json 与 nodelocks.json
  读：load() 以富格式为准；富格式缺失时回退读扁平文件（当作整节点锁）

schema
------
gpu_locks.json    {"locks": [{"node":"nid010547", "gpus":[0,1]|null,
                              "jobid":"5992007"|null, "who":"claude-dfm",
                              "note":"...", "at":"...Z"}]}
nodelocks.json    {"nid010547": "claude-dfm"}          ← 投影，只保留 who

`gpus: null` = 整节点。`jobid` 非空时，该 job 不在 RUNNING 里则这条锁 **stale**：
按 idle 计，但要标出来。一把过期的锁若继续显示成 lock，会永久藏起一张真正空闲的
卡 —— 那比没有锁更危险。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.realpath(__file__))
GPUS_PER_NODE = 4          # GH200 固定 4 卡；逐卡解锁整节点锁时用来展开
DEFAULT_TTL = 600          # 秒。10 分钟没动静的锁自动打开（用户 2026-08-13 定）
MEM_FLOOR_MIB = 64         # 空卡本底 1-3 MiB；超过就算「有人占着」，与 gtop 一致


GTOP = os.path.join(os.path.dirname(HERE), "gtop_20260810T182343Z", "gtop")


def rich_path():
    """gtop 读的那份 —— **问 gtop 自己**，不要猜。

    这个功能同时有多个会话在改。路径被翻过一次（gtop 目录 <-> node_status/），
    而任何一侧硬编码另一侧的路径，翻的那一下就会让注册表分叉：一边写、
    另一边读到空。所以这里直接 import gtop 的 locks_path()，
    它翻到哪儿桥就跟到哪儿。导不进来才回退到本目录。
    """
    if os.environ.get("GTOP_LOCKS"):
        return os.environ["GTOP_LOCKS"]
    try:
        import importlib.machinery as m
        import importlib.util as u
        # 必须用 spec_from_file_location：spec_from_loader 不设 __file__，
        # 而 gtop 的 HERE = dirname(realpath(__file__))，取不到就整个导入失败，
        # 静默退到兜底路径 —— 那正是分叉本身。
        spec = u.spec_from_file_location(
            "_gtop", GTOP, loader=m.SourceFileLoader("_gtop", GTOP))
        mod = u.module_from_spec(spec)
        # exec 之前必须先登记进 sys.modules：gtop 里有 @dataclass，
        # 而 dataclasses 在装饰时要靠 sys.modules[cls.__module__] 反查命名空间，
        # 查不到就抛 AttributeError，整个导入静默失败退到兜底路径。
        sys.modules["_gtop"] = mod
        try:
            spec.loader.exec_module(mod)
            return mod.locks_path()
        finally:
            sys.modules.pop("_gtop", None)
    except Exception:
        return os.path.join(HERE, "gpu_locks.json")


def flat_path():
    """sgpur 读的那份。"""
    return os.environ.get("NODE_LOCKS_PATH") or os.path.join(
        os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config")),
        "nodelocks.json")


def path():                 # 给报错信息用；真相在富格式里
    return rich_path()


def _read(p, default):
    try:
        with open(p) as f:
            d = json.load(f)
        return d if isinstance(d, type(default)) else default
    except (OSError, ValueError):
        return default


def _write(p, d):
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    tmp = f"{p}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(d, f, ensure_ascii=False, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, p)                 # 原子替换，别让并发读到半个文件


def load():
    """-> {host: {"who":str, "gpus":list|None, "jobid":str|None, "note":str, "at":str}}

    富格式是真相。它缺失时回退读扁平文件（当作整节点、无 jobid 的手动锁），
    这样别人直接编辑 nodelocks.json 也不会凭空丢掉声明。
    """
    out = {}
    for e in _read(rich_path(), {}).get("locks", []):
        h = e.get("node")
        if not h:
            continue
        gs = e.get("gpus")
        out[h] = {"who": e.get("who") or e.get("jobid") or "?",
                  "gpus": sorted(gs) if isinstance(gs, list) else None,
                  "jobid": e.get("jobid") or None,
                  "note": e.get("note", ""), "at": e.get("at", ""),
                  "ttl": e.get("ttl", DEFAULT_TTL),
                  # 上锁那一刻算一次活动 —— 否则刚锁完还没来得及起任务就过期了
                  "last_activity": e.get("last_activity", e.get("ts", 0)) or 0,
                  "last_probe": e.get("last_probe", 0) or 0}
    for h, who in _read(flat_path(), {}).items():
        if isinstance(who, str) and h not in out:
            out[h] = {"who": who, "gpus": None, "jobid": None,
                      "note": "(来自 nodelocks.json)", "at": "",
                      "ttl": DEFAULT_TTL, "last_activity": 0, "last_probe": 0}
    return out


def save(locks):
    """两边都写：富格式给 gtop，扁平投影给 sgpur。

    先写富格式（真相），再写投影。反过来的话，中途失败会留下一份
    没有依据的投影，而那正是别人会拿去做闸门判断的文件。
    """
    _write(rich_path(), {"locks": [
        {"node": h, "gpus": e["gpus"], "jobid": e["jobid"],
         "who": e["who"], "note": e["note"], "at": e["at"],
         "ttl": e.get("ttl", DEFAULT_TTL),
         "last_activity": e.get("last_activity", 0),
         "last_probe": e.get("last_probe", 0)}
        for h, e in sorted(locks.items())]})
    _write(flat_path(), {h: e["who"] for h, e in locks.items()})


def running_ids(user=None):
    try:
        p = subprocess.run(
            ["squeue", "-h", "-u", user or os.environ.get("USER", ""),
             "-t", "RUNNING", "-o", "%i"],
            capture_output=True, text=True, timeout=30)
        return {x.strip() for x in p.stdout.split() if x.strip()}
    except (OSError, subprocess.TimeoutExpired):
        return None                    # None = 查不到，别把所有锁误判成 stale


def idle_expired(entry):
    """闲置过期：我**一直在看**，看了 ttl 秒都没动静。

    判据是两个时间戳的差 `last_probe - last_activity`，不是 `now - last_activity`。
    这个区别是承重的：**「没观测到活动」不等于「观测到空闲」。**
    用 now 的话，没人跑 refresh 的时候所有锁都会自动打开 —— 恰好在最需要锁的
    时候（谁都没盯着）把所有卡放出去。用 last_probe，没人看时它不前进，锁不过期。
    """
    ttl = entry.get("ttl", DEFAULT_TTL) or DEFAULT_TTL
    lp, la = entry.get("last_probe", 0) or 0, entry.get("last_activity", 0) or 0
    return bool(lp and la and (lp - la) > ttl)


def is_live(entry, ids):
    """三个条件同时成立才算 live：

      1. 挂在 jobid 上的锁，该 job 还在 RUNNING（无 jobid 的手动锁跳过此条）
      2. 没有闲置过期（见 idle_expired）

    ids 为 None（squeue 失败）时第 1 条一律通过 —— 把锁误判成 stale 会让别人
    去抢一张有主的卡，方向比误判成 live 更危险。注意这与第 2 条不矛盾：
    那一条要的是**正面证据**（确实看了、确实没动静），观测失败不构成证据。
    """
    if idle_expired(entry):
        return False
    jid = entry.get("jobid")
    if not jid or ids is None:
        return True
    return jid in ids


def why_dead(entry, ids):
    if idle_expired(entry):
        m = int((entry["last_probe"] - entry["last_activity"]) // 60)
        return f"闲置 {m}min"
    jid = entry.get("jobid")
    if jid and ids is not None and jid not in ids:
        return "job 已结束"
    return ""


# ---------------------------------------------------------------- 活动探测
def probe_activity(user=None):
    """-> ({host: busy_bool}, probed_ok)

    对每个 RUNNING 分配打一枪 srun，问每张卡的 util 与显存。
    与 gtop 同一条合规通道：--overlap --exact 附着到自己已持有的分配，
    只读、采完即退，计算节点上不留常驻进程，也不碰 Lustre。

    busy 的判据与 gtop 的 run/held 一致：util>0 或显存 > 64 MiB。
    「占着显存但没在算」也算活动 —— 那是有人在用这张卡（哪怕用得浪费），
    不是无主。闲置过期要处理的是「彻底没人碰」。
    """
    user = user or os.environ.get("USER", "")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("SLURM") or k == "SLURM_CONF"}
    try:
        p = subprocess.run(["squeue", "-h", "-u", user, "-t", "RUNNING",
                            "-o", "%i|%D|%N"], capture_output=True, text=True,
                           timeout=30, env=env)
    except (OSError, subprocess.TimeoutExpired):
        return {}, False
    busy, ok_any = {}, False
    for line in p.stdout.splitlines():
        parts = line.split("|")
        if len(parts) != 3:
            continue
        jid, nn, nl = parts
        hosts = expand_nodes(nl)
        for h in hosts:
            busy.setdefault(h, False)
        cmd = ["srun", f"--jobid={jid}", "--overlap", "--exact",
               "--cpus-per-task=1", f"--nodes={nn}", f"--ntasks={nn}",
               "--ntasks-per-node=1", "--cpu-bind=none",
               "--job-name=nodelock-probe", "bash", "-c",
               'echo "$(hostname)|$(nvidia-smi '
               '--query-gpu=utilization.gpu,memory.used '
               '--format=csv,noheader,nounits | tr "\\n" ";")"']
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=90, env=env)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if r.returncode != 0:
            continue
        ok_any = True
        for ln in r.stdout.splitlines():
            if "|" not in ln:
                continue
            h, gpus = ln.split("|", 1)
            h = h.strip()
            for g in gpus.split(";"):
                if "," not in g:
                    continue
                try:
                    u, mem = (int(x.strip()) for x in g.split(",", 1))
                except ValueError:
                    continue
                if u > 0 or mem > MEM_FLOOR_MIB:
                    busy[h] = True
    return busy, ok_any


def refresh(user=None):
    """探一次活动，把结果写回注册表。-> (探到几个 busy, 是否探测成功)

    探测失败时**只字不改**：last_probe 不前进，于是没有任何锁会因为
    这次失败而向过期靠近一步。
    """
    busy, ok = probe_activity(user)
    if not ok:
        return 0, False
    locks, now_s, n = load(), int(time.time()), 0
    for h, e in locks.items():
        if h not in busy:
            continue                       # 这个节点不在我手上，无从观测
        e["last_probe"] = now_s
        if busy[h]:
            e["last_activity"] = now_s
            n += 1
        elif not e.get("last_activity"):
            e["last_activity"] = now_s     # 首次观测：从现在开始计时，不追溯
    save(locks)
    return n, True


def touch(targets):
    """手动心跳：马上要用但还没起进程时，用它防止锁被闲置过期收走。"""
    locks, now_s, n = load(), int(time.time()), 0
    for t in targets:
        for host, _ in parse_target(t):
            if host in locks:
                locks[host]["last_activity"] = now_s
                n += 1
    save(locks)
    return n


def owner_of(locks, host, gpu_idx, ids=None):
    """(host, gpu) 上**仍然有效**的锁条目；无锁或已 stale 返回 None。"""
    e = locks.get(host)
    if not e:
        return None
    gs = e.get("gpus")
    if gs is not None and gpu_idx not in gs:
        return None
    return e if is_live(e, ids) else None


def stale_of(locks, host, gpu_idx, ids=None):
    e = locks.get(host)
    if not e:
        return None
    gs = e.get("gpus")
    if gs is not None and gpu_idx not in gs:
        return None
    return None if is_live(e, ids) else e


def expand_nodes(spec):
    """nid[011165-011167] -> [nid011165, ...]；解析失败原样返回。"""
    if not spec or spec.startswith("("):
        return []
    try:
        p = subprocess.run(["scontrol", "show", "hostnames", spec],
                           capture_output=True, text=True, timeout=15)
        return p.stdout.split() if p.returncode == 0 else [spec]
    except (OSError, subprocess.TimeoutExpired):
        return [spec]


def parse_target(s):
    """`nid010547` / `nid[..]` / `nid010547:0,2` / `5992007` -> [(host, gpus|None)]

    纯数字视为 jobid，展开成它持有的全部节点。方括号里不会有冒号，所以 rsplit 安全。
    """
    gs = None
    if ":" in s:
        s, idx = s.rsplit(":", 1)
        gs = sorted({int(x) for x in idx.split(",") if x.strip() != ""})
    if s.isdigit():
        try:
            p = subprocess.run(["squeue", "-h", "-j", s, "-o", "%N"],
                               capture_output=True, text=True, timeout=30)
            hosts = expand_nodes(p.stdout.strip())
        except (OSError, subprocess.TimeoutExpired):
            hosts = []
    else:
        hosts = expand_nodes(s) or [s]
    return [(h, gs) for h in hosts]


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def do_lock(targets, who, jobid=None, note="", ttl=DEFAULT_TTL):
    locks, n, now_s = load(), 0, int(time.time())
    for t in targets:
        for host, gs in parse_target(t):
            locks[host] = {"who": who, "gpus": gs, "jobid": jobid,
                           "note": note, "at": now(), "ttl": ttl,
                           # 上锁即算一次活动：给一整个 ttl 的宽限期把任务起来
                           "last_activity": now_s, "last_probe": 0}
            n += 1
    save(locks)
    return n


def do_unlock(targets):
    """逐卡解锁一条整节点锁时**收窄**成剩下的卡，而不是整条删掉 ——
    否则放掉一张卡会连带把另外三张一起放掉。"""
    locks, before = load(), None
    before = len(locks)
    for t in targets:
        for host, gs in parse_target(t):
            e = locks.get(host)
            if not e:
                continue
            if gs is None:
                locks.pop(host, None)
                continue
            cur = e.get("gpus")
            cur = list(range(GPUS_PER_NODE)) if cur is None else list(cur)
            keep = [i for i in cur if i not in gs]
            if keep:
                e["gpus"] = keep
            else:
                locks.pop(host, None)
    save(locks)
    return before - len(locks)
