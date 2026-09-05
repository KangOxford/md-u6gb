#!/usr/bin/env python3
"""Four-stage ledger: prepared -> submitted -> processed -> artifact.

A recovery packet being built says nothing about the task being done.  The old registry had
one axis (running/done/failed) and so could not express "the packet exists but nobody ran
it", which is precisely the state all nine agents were in after the previous round.  Four
stages, recorded independently:

    prepared   a RESUME packet exists for this agent
    submitted  the packet was handed to an executor (a replacement agent, or this session)
    processed  that executor ran to completion and said what it did
    artifact   the task's DECLARED output exists and is non-empty -- MEASURED, not asserted

`artifact` is the only stage that is not a claim: it takes a path and stats it.  The other
three are testimony, and the ledger keeps them apart from the measurement on purpose.
"""
import json, os, sys, datetime

REGISTRY = "/lus/lfs1aip2/projects/public/u6gb/.claude/agent_registry"
REG = os.path.join(REGISTRY, "registry.jsonl")
STAGES = ["prepared", "submitted", "processed", "artifact"]


def now():
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def load():
    seen, stages = {}, {}
    for line in open(REG):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "stage" in r:
            stages.setdefault(r["agent_id"], {})[r["stage"]] = r
        else:
            seen.setdefault(r["agent_id"], {}).update(r)
    return seen, stages


def resolve(seen, key):
    if key in seen:
        return key
    hits = [a for a, r in seen.items() if r.get("slug") == key]
    if len(hits) == 1:
        return hits[0]
    sys.exit(f"no agent (or ambiguous) for {key!r}")


def record(key, stage, note="", path=None):
    if stage not in STAGES:
        sys.exit(f"stage must be one of {STAGES}")
    seen, _ = load()
    aid = resolve(seen, key)
    row = {"ts": now(), "agent_id": aid, "slug": seen[aid].get("slug"), "stage": stage,
           "note": note}
    if stage == "artifact":
        if not path:
            sys.exit("stage artifact needs the declared output path -- it is measured, not asserted")
        ok = os.path.exists(path)
        sz = (sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path))
              if ok and os.path.isdir(path) else (os.path.getsize(path) if ok else 0))
        row["evidence"] = {"path": os.path.realpath(path) if ok else path,
                           "exists": ok, "bytes": sz,
                           "mtime": (datetime.datetime.fromtimestamp(os.path.getmtime(path),
                                     datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ") if ok else None)}
        if not ok or sz == 0:
            row["note"] = (note + " | MEASURED ABSENT OR EMPTY").strip(" |")
    with open(REG, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"{seen[aid].get('slug')}  stage={stage}" +
          (f"  evidence={row['evidence']}" if "evidence" in row else f"  note={note}"))


def table():
    seen, stages = load()
    live = {a: r for a, r in seen.items() if r.get("status") == "running"}
    if not live:
        print("no unfinished agents")
        return
    w = max(len(r.get("slug") or a) for a, r in live.items())
    print(f"{'slug':<{w}}  " + "  ".join(f"{s:<9}" for s in STAGES) + "  artifact evidence")
    print("-" * (w + 60))
    for a, r in live.items():
        st = stages.get(a, {})
        cells = []
        for s in STAGES:
            if s not in st:
                cells.append(f"{'-':<9}")
            elif s == "artifact":
                # A stage row recording "I measured it and it was absent" is NOT the same
                # as "the artifact is there".  Printing `yes` for both is the label saying
                # what the measurement did not.
                ev = st[s].get("evidence") or {}
                cells.append(f"{('PRESENT' if ev.get('exists') and ev.get('bytes') else 'ABSENT'):<9}")
            else:
                cells.append(f"{'yes':<9}")
        ev = st.get("artifact", {}).get("evidence")
        eb = ""
        if ev:
            eb = f"{ev['bytes']} B  {ev['path']}" if ev["exists"] else f"ABSENT {ev['path']}"
        print(f"{(r.get('slug') or a):<{w}}  " + "  ".join(cells) + f"  {eb}")


if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1] == "table":
        table()
    else:
        record(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "",
               sys.argv[4] if len(sys.argv) > 4 else None)
