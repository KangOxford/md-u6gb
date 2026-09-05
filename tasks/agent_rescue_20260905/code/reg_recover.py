#!/usr/bin/env python3
"""`agent_reg.sh recover <agent_id|slug>` -- turn a dead agent's remains into a packet
that can be handed straight to a replacement.

Re-entrancy is the point, not a nicety.  The reason to run this is that something was
disconnected, and the thing most likely to happen next is being disconnected again in the
middle of it.  So every run stamps what it consumed (transcript path, size, mtime) and a
second run over unchanged inputs rewrites nothing and says so.  Without that, the
recovery tool has the same failure mode as the thing it recovers from: redoing work that
was already done, and being unable to tell that it did.

Outputs, all under <registry>/work/<slug>/ :
    prompt.txt                 the spawn prompt, materialised from the inline copy
    banked_from_transcript.md  mechanical rebuild of the transcript (extract_progress.py)
    RESUME.md                  the packet: prompt + what is already established + failures
    .recover_stamp.json        what this run consumed, so the next run can skip
"""
import json, os, sys, subprocess, datetime, argparse

REGISTRY = "/lus/lfs1aip2/projects/public/u6gb/.claude/agent_registry"
REG = os.path.join(REGISTRY, "registry.jsonl")
EXTRACT = os.path.join(REGISTRY, "extract_progress.py")
sys.path.insert(0, REGISTRY)
from extract_progress import locate  # reuse, do not reimplement


def load():
    seen = {}
    for line in open(REG):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        seen.setdefault(r["agent_id"], {}).update(r)
    return seen


def resolve(seen, key):
    if key in seen:
        return seen[key]
    hits = [r for r in seen.values() if r.get("slug") == key]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        sys.exit(f"no agent with id or slug {key!r} in {REG}")
    sys.exit(f"slug {key!r} is ambiguous: {[h['agent_id'] for h in hits]}")


def now():
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def recover(rec, force=False):
    aid, slug = rec["agent_id"], rec.get("slug") or rec["agent_id"]
    wd = rec.get("work") if os.path.isabs(rec.get("work") or "") else os.path.join(REGISTRY, "work", slug)
    os.makedirs(wd, exist_ok=True)
    stamp_p = os.path.join(wd, ".recover_stamp.json")

    src, kind, _ = locate(aid)
    sig = None
    if src:
        st = os.stat(src)
        sig = {"src": os.path.realpath(src), "size": st.st_size, "mtime": int(st.st_mtime)}

    if not force and os.path.exists(stamp_p):
        try:
            old = json.load(open(stamp_p))
        except Exception:
            old = {}
        if old.get("source") == sig and all(os.path.exists(os.path.join(wd, f))
                                            for f in old.get("produced", [])):
            print(f"{slug}: already recovered at {old.get('ts')} from an unchanged "
                  f"transcript -- nothing re-done")
            print(f"  packet: {os.path.join(wd, 'RESUME.md')}")
            return "noop"
        if old.get("source") and sig and old["source"]["size"] < sig["size"]:
            print(f"{slug}: transcript grew {old['source']['size']} -> {sig['size']} bytes "
                  f"since {old.get('ts')}, rebuilding")

    produced = []

    # 1. the prompt.  A prompt that lives in only one place has a single point of failure,
    #    and that failure already happened once (full filesystem, 2026-09-04T18:30Z).
    pf = os.path.join(wd, "prompt.txt")
    inline = rec.get("prompt_inline") or ""
    if inline and (force or not os.path.exists(pf) or not os.path.getsize(pf)):
        with open(pf, "w", encoding="utf-8") as fh:
            fh.write(inline)
        print(f"  prompt.txt <- {len(inline)} chars from the registry row")
    if os.path.exists(pf) and os.path.getsize(pf):
        produced.append("prompt.txt")

    # 2. the mechanical rebuild
    banked = os.path.join(wd, "banked_from_transcript.md")
    if src:
        r = subprocess.run([sys.executable, EXTRACT, aid, "-o", banked],
                           capture_output=True, text=True)
        if r.returncode:
            print(f"  extract FAILED: {r.stderr.strip()[:300]}")
        else:
            produced.append("banked_from_transcript.md")
            print(f"  banked_from_transcript.md <- {kind}")
    else:
        print("  no transcript found; packet will carry the prompt only")

    # 3. the packet
    prog = os.path.join(wd, "progress.md")
    prog_n = sum(1 for _ in open(prog, errors="replace")) if os.path.exists(prog) else 0
    nfail = 0
    if os.path.exists(banked):
        for line in open(banked, errors="replace"):
            if line.startswith("| explicit tool failures |"):
                try:
                    nfail = int(line.split("|")[2].strip())
                except Exception:
                    pass
    resume = os.path.join(wd, "RESUME.md")
    with open(resume, "w", encoding="utf-8") as fh:
        fh.write(f"""# RESUME packet — `{slug}` (agent `{aid}`)

Built {now()} by `agent_reg.sh recover`. Hand this whole file to the replacement agent.

| what | where |
|---|---|
| spawn prompt | `{pf}` {'(present)' if 'prompt.txt' in produced else '(MISSING)'} |
| agent's own banked notes | `{prog}` ({prog_n} lines) |
| mechanical transcript rebuild | `{banked}` {'(present)' if os.path.exists(banked) else '(MISSING)'} |
| transcript source | `{sig['src'] if sig else 'none'}` |
| explicit tool failures recorded | {nfail} |

## Instructions for the replacement

1. Read `banked_from_transcript.md` **first**. It contains the previous agent's own words
   and its full tool trail. Everything listed there is already done; do not redo it.
2. Read `progress.md` if it is non-empty — that is what the previous agent chose to bank
   deliberately, so it outranks the transcript rebuild where they disagree.
3. Then carry out the original prompt below, starting from where the trail stops.
4. Bank as you go: `>>` append to `{prog}`. Assume you will be killed without warning.

## Original prompt

```
{inline or '(prompt not recorded; reconstruct from the first user message in the rebuild)'}
```
""")
    produced.append("RESUME.md")
    json.dump({"ts": now(), "agent_id": aid, "slug": slug, "source": sig,
               "produced": produced}, open(stamp_p, "w"), indent=1)
    print(f"  packet: {resume}")
    return "built"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", help="agent id or slug; omit with --all")
    ap.add_argument("--all", action="store_true", help="every agent still marked running")
    ap.add_argument("--force", action="store_true", help="rebuild even if inputs unchanged")
    a = ap.parse_args()
    seen = load()
    if a.all:
        targets = [r for r in seen.values() if r.get("status") == "running"]
        if not targets:
            print("no unfinished agents")
            return 0
    elif a.target:
        targets = [resolve(seen, a.target)]
    else:
        ap.error("give an agent id/slug or --all")
    tally = {"built": 0, "noop": 0}
    for rec in targets:
        print(f"{rec.get('slug', rec['agent_id'])}:")
        tally[recover(rec, a.force)] += 1
    print(f"\n{tally['built']} packet(s) built, {tally['noop']} already current "
          f"(re-entry skipped them).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
