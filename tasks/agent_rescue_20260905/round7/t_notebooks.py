#!/usr/bin/env python3
"""Acceptance for the RENDERED notebooks, not the builder sources.

Round 6 stated plainly that nothing checked the rendered notebooks. This does. It takes a
notebook path so the same checks run against the pre-fix backups (.bak_<ts>) and show red,
which is the only way to know the checks detect anything.

It never runs the producer and never masks a producer's exit code; the caller runs the
builder and reports rc itself.
"""
import base64, json, re, sys, os

npass = nfail = 0
def check(name, ok, detail=""):
    global npass, nfail
    if ok: print(f"  PASS  {name}"); npass += 1
    else:  print(f"  FAIL  {name}\n        {detail}"); nfail += 1

def load(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)

def texts(nb):
    out = []
    for c in nb.get("cells", []):
        src = c.get("source")
        out.append("".join(src) if isinstance(src, list) else (src or ""))
        for o in c.get("outputs", []) or []:
            for k in ("text", "text/plain"):
                v = (o.get("data") or {}).get(k) or o.get(k)
                if v: out.append("".join(v) if isinstance(v, list) else str(v))
    return "\n".join(out)

def images(nb):
    n, total = 0, 0
    for c in nb.get("cells", []):
        for o in c.get("outputs", []) or []:
            png = (o.get("data") or {}).get("image/png")
            if png:
                n += 1
                total += len(base64.b64decode(png if isinstance(png, str) else "".join(png)))
    return n, total

def errors(nb):
    return [o for c in nb.get("cells", []) for o in (c.get("outputs") or [])
            if o.get("output_type") == "error"]

path = sys.argv[1]
want = sys.argv[2] if len(sys.argv) > 2 else None      # a phrase the rendered notebook must carry
name = os.path.basename(path)
nb = load(path)
print(f"{name}  ({os.path.getsize(path)} bytes)")

check("NB1 parses as nbformat 4", nb.get("nbformat") == 4, f"nbformat={nb.get('nbformat')}")

cells = nb.get("cells", [])
code = [c for c in cells if c.get("cell_type") == "code"]
with_out = [c for c in code if c.get("outputs")]
check("NB2 every code cell carries output",
      bool(code) and len(with_out) == len(code),
      f"{len(with_out)} of {len(code)} code cells have outputs -- an empty output is an unrun cell")

nimg, nbytes = images(nb)
check("NB3 the notebook embeds figures",
      nimg > 0 and nbytes > 50_000,
      f"{nimg} embedded PNGs, {nbytes} bytes total")

check("NB4 no cell carries an error output", not errors(nb),
      f"{len(errors(nb))} error outputs")

if want:
    body = texts(nb)
    check(f"NB5 the rendered notebook carries the correction: {want[:52]!r}",
          want in body, "the corrected sentence is in the source but not in the rendered cells")

# NB6/NB7 need the builder that produced this notebook.
builder = sys.argv[3] if len(sys.argv) > 3 else None
if builder:
    # NB7 is the check whose absence left round 6's gap open: the notebook on disk was
    # four hours older than the builder that was supposed to have produced it, so every
    # source-level PASS said nothing about what a reader would actually open.
    nb_m, b_m = os.path.getmtime(path), os.path.getmtime(builder)
    check("NB7 the notebook is not older than its builder",
          nb_m >= b_m,
          f"notebook {nb_m:.0f} < builder {b_m:.0f} -- stale by "
          f"{(b_m - nb_m) / 60:.0f} min; the rendered file predates the fix")

figs = os.path.join(os.path.dirname(os.path.abspath(path)), "figs")
if os.path.isdir(figs):
    bad = []
    for e in os.scandir(figs):
        if not e.name.endswith(".png"):
            continue
        with open(e.path, "rb") as fh:
            head = fh.read(8)
        if e.stat().st_size == 0 or head != b"\x89PNG\r\n\x1a\n":
            bad.append(e.name)
    check("NB6 every figure on disk is a non-empty PNG",
          not bad, f"bad: {bad[:6]}")

print(f"\n{npass} passed, {nfail} failed")
sys.exit(nfail)
