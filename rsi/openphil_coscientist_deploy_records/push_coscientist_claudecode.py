#!/usr/bin/env python3
"""Push the coscientist-vs-heuristic-learning subpage under openphil-quant.

Reuses md_to_blocks / api / HEADERS from the notion-push-via-rest skill so the
skill source stays untouched. Token comes from $NOTION_TOKEN_PATH.
"""
import sys
from pathlib import Path

SKILL_DIR = "/projects/public/u6gb/.claude/skills/notion-push-via-rest"
sys.path.insert(0, SKILL_DIR)
import push_notion as pn  # noqa: E402  (import runs only constant/token setup)

PARENT = "38712c45-68fd-8070-945c-d3e0173a45bb"  # openphil-quant
MD_PATH = "/projects/public/u6gb/openphil_coscientist_deploy_records/coscientist_vs_heuristic_learning_claudecode.md"
TITLE = "OpenPhil Coscientist 与 Heuristic Learning 的关系 · claudecode"
MARKER = "/projects/public/u6gb/openphil_coscientist_deploy_records/.claudecode_subpage_pushed"


def main():
    if Path(MARKER).exists():
        print("ALREADY_PUSHED:", Path(MARKER).read_text().strip())
        return 0
    md = Path(MD_PATH).read_text()
    blocks = pn.md_to_blocks(md)
    print(f"converted {len(blocks)} blocks")
    create_body = {
        "parent": {"page_id": PARENT},
        "properties": {"title": {"title": [{"type": "text", "text": {"content": TITLE}}]}},
        "children": blocks[:100],
    }
    created = pn.api("POST", "/pages", create_body)
    child_id = created["id"]
    url = created.get("url", "")
    print(f"created child page {child_id}")
    remaining = blocks[100:]
    n = 0
    while remaining:
        batch, remaining = remaining[:100], remaining[100:]
        pn.api("PATCH", f"/blocks/{child_id}/children", {"children": batch})
        n += 1
        print(f"appended batch {n} ({len(batch)} blocks)")
    Path(MARKER).write_text(f"{url}\n{child_id}\n")
    print("DONE_URL:", url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
