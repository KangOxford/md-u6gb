#!/usr/bin/env python3
"""`agent_reg.sh verify` -- checks that can actually go red.

Each check below is a regression test for a defect that was measured on this registry, not
a restatement of the code.  A check that cannot fail is not a check: run
`reg_verify.py --self-test` to see each one deliberately broken and reported red.

Exit code is the number of failing checks, so it is usable as a gate.
"""
import json, os, sys, tempfile, shutil

REGISTRY = "/lus/lfs1aip2/projects/public/u6gb/.claude/agent_registry"
sys.path.insert(0, REGISTRY)


def load(reg):
    seen = {}
    n = 0
    for line in open(reg):
        line = line.strip()
        if not line:
            continue
        n += 1
        r = json.loads(line)          # a row that will not parse is itself the failure
        seen.setdefault(r["agent_id"], {}).update(r)
    return seen, n


def checks(registry=REGISTRY):
    reg = os.path.join(registry, "registry.jsonl")
    out = []                                     # (name, ok, detail)

    try:
        seen, nrows = load(reg)
        out.append(("registry parses", True, f"{nrows} rows, {len(seen)} agents"))
    except Exception as e:
        return [("registry parses", False, repr(e))]

    live = [r for r in seen.values() if r.get("status") == "running"]

    # C2 -- the defect that started this: a report that says "nothing was banked" while the
    # prompt sits inline in the same row and a 400 KB-5.6 MB transcript sits on /home.
    import reg_report
    durable = reg_report.index_durable()
    bad = [r["agent_id"] for r in live
           if not (r.get("prompt_inline") or durable.get(r["agent_id"]))
           and os.path.exists(os.path.join(registry, "transcripts", r["agent_id"] + ".output"))]
    out.append(("every running agent with a transcript is reported recoverable",
                not bad, f"{len(live)} running, unreported: {bad}"))

    # C3 -- the 1705-line bug: a relative progress path resolves against the cwd, so an
    # unrelated file gets attributed to nine agents.  No emitted path may be relative.
    rel = []
    for r in live:
        w = r.get("work") or ""
        if w and not os.path.isabs(w):
            rel.append(r["agent_id"])
    out.append(("no work path is relative", not rel, f"relative: {rel}"))

    # C4 -- re-entry: every recovered agent carries a stamp that matches its current
    # transcript, and every file the stamp claims exists and is non-empty.
    stale, empty = [], []
    for r in live:
        slug = r.get("slug") or r["agent_id"]
        wd = r.get("work") if os.path.isabs(r.get("work") or "") else os.path.join(registry, "work", slug)
        sp = os.path.join(wd, ".recover_stamp.json")
        if not os.path.exists(sp):
            continue
        st = json.load(open(sp))
        for f in st.get("produced", []):
            p = os.path.join(wd, f)
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                empty.append(f"{slug}/{f}")
        src = (st.get("source") or {}).get("src")
        if src and os.path.exists(src) and os.path.getsize(src) != st["source"]["size"]:
            stale.append(slug)
    out.append(("recovered artefacts exist and are non-empty", not empty, f"empty/missing: {empty}"))
    out.append(("stamps match their transcripts (re-run would be a no-op)",
                not stale, f"stale (transcript grew): {stale}"))

    # C5 -- the prompt actually round-trips: what got written is what was recorded.
    mism = []
    for r in live:
        inline = r.get("prompt_inline")
        if not inline:
            continue
        slug = r.get("slug") or r["agent_id"]
        wd = r.get("work") if os.path.isabs(r.get("work") or "") else os.path.join(registry, "work", slug)
        pf = os.path.join(wd, "prompt.txt")
        if os.path.exists(pf) and open(pf, encoding="utf-8").read() != inline:
            mism.append(slug)
    out.append(("materialised prompt.txt == prompt_inline", not mism, f"mismatched: {mism}"))
    return out


def run(registry=REGISTRY, quiet=False):
    res = checks(registry)
    fails = 0
    for name, ok, detail in res:
        if not ok:
            fails += 1
        if not quiet:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n         {detail}")
    if not quiet:
        print(f"\n{len(res) - fails}/{len(res)} checks pass")
    return fails


def self_test():
    """Break each invariant in a throwaway copy and confirm the check turns red.  Without
    this, a check that silently always passes is indistinguishable from a check that works."""
    print("self-test: each check is deliberately broken; every line must say RED\n")
    src_reg = os.path.join(REGISTRY, "registry.jsonl")
    rows = [json.loads(l) for l in open(src_reg) if l.strip()]
    merged, _ = load(src_reg)
    LIVE_IDS = {a for a, r in merged.items() if r.get("status") == "running"}

    def first_live(rs, need_prompt=False):
        """A row is only worth mutating if it survives the merge as `running`."""
        for r in rs:
            if r["agent_id"] in LIVE_IDS and r.get("slug") \
               and (not need_prompt or r.get("prompt_inline")):
                return r
        return None

    def trial(label, mutate, expect):
        d = tempfile.mkdtemp(prefix="regverify_")
        try:
            os.makedirs(os.path.join(d, "work"), exist_ok=True)
            os.makedirs(os.path.join(d, "transcripts"), exist_ok=True)
            rs = mutate([dict(r) for r in rows], d)
            with open(os.path.join(d, "registry.jsonl"), "w") as fh:
                for r in rs:
                    fh.write(json.dumps(r) + "\n")
            res = {n: ok for n, ok, _ in checks(d)}
            red = res.get(expect) is False
            print(f"  [{'RED ' if red else 'GREEN (BUG: check cannot fail)'}] {label}")
            return red
        finally:
            shutil.rmtree(d, ignore_errors=True)

    ok = []

    def m_rel(rs, d):
        for r in rs:
            if r.get("status") == "running":
                r["work"] = "work/relative_path"
        return rs
    ok.append(trial("give a running agent a RELATIVE work path", m_rel, "no work path is relative"))

    def m_empty(rs, d):
        r = first_live(rs)
        wd = os.path.join(d, "work", r["slug"])
        os.makedirs(wd, exist_ok=True)
        open(os.path.join(wd, "RESUME.md"), "w").close()              # zero bytes
        json.dump({"produced": ["RESUME.md"], "source": None},
                  open(os.path.join(wd, ".recover_stamp.json"), "w"))
        r.pop("work", None)
        return rs
    ok.append(trial("leave a zero-byte artefact behind", m_empty,
                    "recovered artefacts exist and are non-empty"))

    def m_prompt(rs, d):
        r = first_live(rs, need_prompt=True)
        wd = os.path.join(d, "work", r["slug"])
        os.makedirs(wd, exist_ok=True)
        open(os.path.join(wd, "prompt.txt"), "w").write("NOT THE RECORDED PROMPT")
        r.pop("work", None)
        return rs
    ok.append(trial("write a prompt.txt that differs from the recorded prompt", m_prompt,
                    "materialised prompt.txt == prompt_inline"))

    def m_stale(rs, d):
        r = first_live(rs)
        wd = os.path.join(d, "work", r["slug"])
        os.makedirs(wd, exist_ok=True)
        fake = os.path.join(d, "fake_transcript.jsonl")
        open(fake, "w").write("x" * 100)
        json.dump({"produced": [], "source": {"src": fake, "size": 5, "mtime": 0}},
                  open(os.path.join(wd, ".recover_stamp.json"), "w"))
        r.pop("work", None)
        return rs
    ok.append(trial("stamp a transcript size that no longer matches", m_stale,
                    "stamps match their transcripts (re-run would be a no-op)"))

    bad = ok.count(False)
    print(f"\n{len(ok) - bad}/{len(ok)} checks proved they can fail")
    return bad


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    sys.exit(run())
