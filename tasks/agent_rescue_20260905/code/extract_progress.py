#!/usr/bin/env python3
"""Rebuild an agent's banked work from its transcript, mechanically.

Why this exists.  Layer 2 of agent-rescue asks the agent itself to append findings to
work/<slug>/progress.md.  That layer failed for nine agents on 2026-09-04: the sibling
prompt.txt writes hit a full filesystem at 18:30Z, so nothing was banked.  Their
transcripts were never lost -- they sit on VAST /home, 416 KB to 5.6 MB each -- but a
transcript is not a checkpoint until something turns it back into instructions.

Nothing here calls a model.  Every line of output is copied or counted from the JSONL,
so re-running it on the same transcript gives byte-identical output and costs no tokens.
The agent pays nothing for this checkpoint: the harness wrote the transcript anyway.

Two transcript copies exist and they are NOT interchangeable:

  ~/.claude/projects/<project>/<session>/subagents/agent-<id>.jsonl   VAST /home, durable,
      written incrementally, survives node changes            <-- prefer this
  <registry>/transcripts/<id>.output                          point-in-time copy out of
      node-local tmpfs; truncated at whatever the rescue caught

Measured 2026-09-05 on agent a05ee6464a5a7fabe: 115 lines on /home vs 90 in the rescued
copy.  Taking the rescued one would have silently dropped the last 25 steps, which are
exactly the steps closest to where the agent stopped.
"""
import json, os, sys, argparse, datetime

HOME_PROJECTS = os.path.expanduser("~/.claude/projects")
REGISTRY = "/lus/lfs1aip2/projects/public/u6gb/.claude/agent_registry"


def index_durable(project_dir):
    """agent_id -> path, over one scandir per session dir.  No recursive walk: Lustre and
    VAST both charge per directory entry, and a recursive find here would touch every
    session ever created."""
    out = {}
    if not os.path.isdir(project_dir):
        return out
    for s in os.scandir(project_dir):
        if not s.is_dir():
            continue
        sd = os.path.join(s.path, "subagents")
        if not os.path.isdir(sd):
            continue
        for e in os.scandir(sd):
            if e.name.startswith("agent-") and e.name.endswith(".jsonl"):
                aid = e.name[len("agent-"):-len(".jsonl")]
                # Keep the largest copy if an id somehow appears twice.
                if aid not in out or e.stat().st_size > os.path.getsize(out[aid]):
                    out[aid] = e.path
    return out


def locate(agent_id):
    """Return (path, source_label, alternatives_considered)."""
    cands = []
    for proj in (os.scandir(HOME_PROJECTS) if os.path.isdir(HOME_PROJECTS) else []):
        if not proj.is_dir():
            continue
        p = index_durable(proj.path).get(agent_id)
        if p:
            cands.append((os.path.getsize(p), p, "durable /home jsonl"))
    rescued = os.path.join(REGISTRY, "transcripts", agent_id + ".output")
    if os.path.exists(rescued):
        cands.append((os.path.getsize(rescued), rescued, "rescued tmpfs .output"))
    if not cands:
        return None, None, []
    cands.sort(reverse=True)          # largest wins; durable is larger in every measured case
    return cands[0][1], cands[0][2], [(c[2], c[0]) for c in cands]


def text_of(content):
    """message.content is either a string or a list of typed blocks."""
    if isinstance(content, str):
        return content, [], []
    texts, tools, results = [], [], []
    for b in content if isinstance(content, list) else []:
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t == "text":
            texts.append(b.get("text", ""))
        elif t == "tool_use":
            tools.append((b.get("name", "?"), b.get("input") or {}))
        elif t == "tool_result":
            body = b.get("content")
            if isinstance(body, list):
                body = " ".join(x.get("text", "") for x in body if isinstance(x, dict))
            results.append((bool(b.get("is_error")), str(body or "")))
    return "\n".join(texts), tools, results


def one_line(s, n=220):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n - 1] + "…"


def tool_brief(name, inp):
    """A tool call is only useful for resumption if you can tell WHICH file / command."""
    for k in ("file_path", "path", "notebook_path", "command", "pattern", "prompt", "url", "query"):
        if k in inp:
            return f"{name}({k}={one_line(inp[k], 160)})"
    return f"{name}({one_line(json.dumps(inp, ensure_ascii=False), 120)})"


def render(agent_id, path, source, alts, max_steps=None):
    rows = []
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    spawn_prompt = None
    steps, failures, texts = [], [], []
    for r in rows:
        msg = r.get("message") or {}
        role = msg.get("role")
        txt, tools, results = text_of(msg.get("content"))
        ts = r.get("timestamp", "")
        if role == "user" and spawn_prompt is None and not results:
            spawn_prompt = txt
            continue
        if role == "assistant":
            if txt.strip():
                texts.append((ts, txt.strip()))
            for name, inp in tools:
                steps.append((ts, tool_brief(name, inp)))
        for is_err, body in results:
            if is_err:
                failures.append((ts, one_line(body, 300)))

    first_ts = rows[0].get("timestamp", "") if rows else ""
    last_ts = rows[-1].get("timestamp", "") if rows else ""
    out = []
    A = out.append
    A(f"# Banked work — agent `{agent_id}`\n")
    A("Rebuilt mechanically from the transcript by `extract_progress.py`. No model was\n"
      "involved, so this file is a function of the transcript alone.\n")
    A("| field | value |")
    A("|---|---|")
    A(f"| source | `{os.path.realpath(path)}` |")
    A(f"| source kind | {source} |")
    A(f"| transcript lines | {len(rows)} |")
    A(f"| tool calls | {len(steps)} |")
    A(f"| explicit tool failures | {len(failures)} |")
    A(f"| first message | {first_ts} |")
    A(f"| last message | {last_ts} |")
    A(f"| rebuilt at | {datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')} |")
    A("")
    if len(alts) > 1:
        A("Copies considered (largest wins; the rescued tmpfs copy is a point-in-time")
        A("snapshot and is routinely shorter than the durable one):\n")
        for label, size in alts:
            A(f"- {label}: {size} bytes")
        A("")
    A("## Spawn prompt (verbatim — this is what to re-hand the replacement agent)\n")
    A("```")
    A(spawn_prompt or "(no user-role opening message found in transcript)")
    A("```\n")
    A("## Explicit failures\n")
    if failures:
        A("Recorded so a replacement does not walk into the same wall:\n")
        for ts, f in failures:
            A(f"- `{ts}` {f}")
    else:
        A("None: no tool result in this transcript carried `is_error`.")
    A("")
    A("## What the agent already said (its own words, in order)\n")
    if texts:
        for ts, t in texts:
            A(f"### {ts}\n")
            A(t)
            A("")
    else:
        A("(the agent produced no free text; see the tool trail below)\n")
    A("## Tool trail\n")
    shown = steps if max_steps is None else steps[:max_steps]
    for i, (ts, s) in enumerate(shown, 1):
        A(f"{i:>3}. `{ts}` {s}")
    if max_steps is not None and len(steps) > max_steps:
        A(f"\n**{len(steps) - max_steps} further tool calls not listed** (--max-steps={max_steps}).")
    A("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agent_id")
    ap.add_argument("-o", "--out", help="write here instead of stdout")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="cap the tool trail; the cap is always printed, never silent")
    a = ap.parse_args()
    path, source, alts = locate(a.agent_id)
    if not path:
        print(f"no transcript found for {a.agent_id} "
              f"(looked in {HOME_PROJECTS}/*/*/subagents/ and {REGISTRY}/transcripts/)",
              file=sys.stderr)
        return 2
    md = render(a.agent_id, path, source, alts, a.max_steps)
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(md)
        print(f"{os.path.realpath(a.out)}  ({len(md)} bytes, from {source})")
    else:
        sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
