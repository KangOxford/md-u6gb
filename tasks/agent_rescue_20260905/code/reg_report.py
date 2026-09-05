#!/usr/bin/env python3
"""`agent_reg.sh pending` -- what is unfinished and what can actually be recovered.

The first version of this report answered the wrong question.  It looked for exactly one
artefact, work/<slug>/progress.md, and printed "nothing was banked" when it was absent.
Measured 2026-09-05: all nine unfinished agents had NO progress.md, and all nine had a
complete 416 KB-5.6 MB transcript on VAST /home plus their full spawn prompt sitting
inline in the very registry row being printed.  The work was never lost; the report was.

So the report now names every source it checked and what each one holds, and it says
UNRECOVERABLE only when all of them are empty.
"""
import json, os, sys

REGISTRY = "/lus/lfs1aip2/projects/public/u6gb/.claude/agent_registry"
HOME_PROJECTS = os.path.expanduser("~/.claude/projects")


def index_durable():
    """agent_id -> (path, bytes).  One scandir per session dir, never a recursive walk."""
    out = {}
    if not os.path.isdir(HOME_PROJECTS):
        return out
    for proj in os.scandir(HOME_PROJECTS):
        if not proj.is_dir():
            continue
        for s in os.scandir(proj.path):
            if not s.is_dir():
                continue
            sd = os.path.join(s.path, "subagents")
            if not os.path.isdir(sd):
                continue
            for e in os.scandir(sd):
                if e.name.startswith("agent-") and e.name.endswith(".jsonl"):
                    aid = e.name[len("agent-"):-len(".jsonl")]
                    sz = e.stat().st_size
                    if aid not in out or sz > out[aid][1]:
                        out[aid] = (os.path.realpath(e.path), sz)
    return out


def main(reg_path):
    rows = []
    for line in open(reg_path):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            print(f"  (unparseable registry line skipped: {line[:60]}…)", file=sys.stderr)
    seen = {}
    for r in rows:
        seen.setdefault(r["agent_id"], {}).update(r)

    live = [r for r in seen.values() if r.get("status") == "running"]
    failed = [r for r in seen.values() if r.get("status") == "failed"]
    if failed:
        print(f"{len(failed)} agent(s) recorded FAILED:")
        for r in failed:
            print(f"  {r.get('slug','?'):<28} {r['agent_id']}  {r.get('reason','')}")
        print()
    if not live:
        print("no unfinished agents")
        return 0

    durable = index_durable()
    print(f"{len(live)} UNFINISHED agent(s). Sources checked per agent: banked progress.md, "
          f"prompt (file or inline), durable transcript.\n")
    for r in live:
        aid = r["agent_id"]
        slug = r.get("slug", "?")
        w = r.get("work") or ""
        print(f"  {slug:<28} {aid}")
        print(f"    {r.get('desc','')}")

        # --- layer 2: what the agent banked itself -------------------------------------
        # A missing work dir must READ as missing.  os.path.join("", "progress.md") yields
        # a RELATIVE path that resolves against the cwd, so a stray progress.md there gets
        # reported as this agent's findings.  Measured 2026-09-04: nine rows with work=None
        # all reported the same unrelated 1705-line file.
        # When `work` was never recorded, the conventional location <registry>/work/<slug>
        # is still deterministic, so check it -- but label it INFERRED.  Reporting an
        # inferred path as a recorded one is how the 1705-line file got attributed to nine
        # agents; reporting nothing at all is how 21 real lines of plan_measurement's
        # banked work got called "nothing was banked".  Both are wrong; the fix is to say
        # which of the two it is.
        prog = None
        cand, how = (w, "recorded") if os.path.isabs(w) else \
                    (os.path.join(REGISTRY, "work", slug), "INFERRED from slug")
        p = os.path.join(cand, "progress.md")
        if os.path.isabs(p) and os.path.exists(p):
            n = sum(1 for _ in open(p, errors="replace"))
            if n:
                prog = (p, n, how)
        print(f"    banked progress : {prog[1]} lines at {prog[0]}  [{prog[2]}]" if prog
              else "    banked progress : none (agent banked nothing before it died)")

        # --- the prompt: file first, inline second --------------------------------------
        pf = os.path.join(w, "prompt.txt") if os.path.isabs(w) else ""
        inline = r.get("prompt_inline") or ""
        if pf and os.path.exists(pf) and os.path.getsize(pf):
            print(f"    prompt          : {os.path.getsize(pf)} bytes at {pf}")
            has_prompt = True
        elif inline:
            print(f"    prompt          : {len(inline)} chars INLINE in the registry row "
                  f"(run `recover` to materialise it as prompt.txt)")
            has_prompt = True
        else:
            print("    prompt          : none")
            has_prompt = False

        # --- the transcript: durable copy first, rescued snapshot second ------------------
        d = durable.get(aid)
        res = os.path.join(REGISTRY, "transcripts", aid + ".output")
        rs = os.path.getsize(res) if os.path.exists(res) else 0
        if d:
            extra = f"  (rescued tmpfs copy: {rs} B, shorter)" if 0 < rs < d[1] else ""
            print(f"    transcript      : {d[1]} bytes durable on /home{extra}")
            print(f"                      {d[0]}")
        elif rs:
            print(f"    transcript      : {rs} bytes, rescued tmpfs snapshot only "
                  f"(may be truncated)")
            print(f"                      {os.path.realpath(res)}")
        else:
            print("    transcript      : none")

        recoverable = bool(prog or has_prompt or d or rs)
        print(f"    -> {'RECOVERABLE: agent_reg.sh recover ' + slug if recoverable else 'UNRECOVERABLE: nothing survived; re-specify by hand'}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(REGISTRY, "registry.jsonl")))
