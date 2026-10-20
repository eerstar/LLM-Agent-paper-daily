"""Strict catalogue storage and atomic publication of the existing formats."""

import copy
import datetime as dt
import json
import os
from pathlib import Path
import re
import tempfile


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return copy.deepcopy(default)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def atomic_write_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", delete=False,
        ) as stream:
            temporary = stream.name
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_json(path, data):
    # Serialize first: serialization errors must not truncate existing data.
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def merge_catalog(existing, incoming):
    merged = copy.deepcopy(existing)
    for topic, papers in incoming.items():
        if not isinstance(papers, dict):
            raise ValueError(f"Invalid catalogue category {topic!r}")
        target = merged.setdefault(topic, {})
        for paper_id, row in papers.items():
            if not isinstance(row, str):
                raise ValueError(f"Invalid paper row {topic}/{paper_id}")
            previous = target.get(paper_id, "")
            old, new = previous.strip().split("|"), row.strip().split("|")
            for cells in (old, new):
                if len(cells) == 7 and not cells[0] and not cells[-1]:
                    cells.insert(-1, "null")
            if len(old) >= 8 and len(new) >= 8:
                for column in (-3, -2):
                    if new[column].strip().lower() in ("", "null"):
                        new[column] = old[column]
                    elif old[column].strip().lower() not in ("", "null"):
                        # Keep conflicting historical links visible, not silently
                        # overwrite one source's nonempty cell with another.
                        urls = list(dict.fromkeys(re.findall(
                            r"\]\((https?://[^)]+)\)", new[column] + " " + old[column],
                        )))
                        if urls:
                            new[column] = " ".join(f"**[link]({url})**" for url in urls)
                row = "|".join(new) + "\n"
            elif len(new) >= 8:
                row = "|".join(new) + "\n"
            target[paper_id] = row
    return merged


def load_catalog(config):
    canonical = config.get("json_readme_path", "docs/llm-agent-arxiv-daily.json")
    data = read_json(canonical, {})
    if not isinstance(data, dict):
        raise ValueError(f"Invalid catalogue root: {canonical}")
    web_path = config.get("json_gitpage_path")
    if web_path and web_path != canonical:
        web = read_json(web_path, {})
        if not isinstance(web, dict):
            raise ValueError(f"Invalid catalogue root: {web_path}")
        # Canonical metadata wins; historical web-only links fill empty cells.
        data = merge_catalog(web, data)
    return merge_catalog({}, data)


def render_catalog(data, web=False):
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    lines = ["---", "layout: default", "---", ""] if web else []
    lines += [
        "# LLM Agent Paper Daily", "",
        "[Pipeline operation and recovery](https://github.com/eerstar/LLM-Agent-paper-daily/blob/main/docs/operations.md)",
        "", f"## Updated on {today}", "",
    ]
    for topic, papers in sorted(data.items()):
        if not papers:
            continue
        lines += [f"## {topic}", "", "|Publish Date|Title|Authors|PDF|Code|Model|",
                  "|---|---|---|---|---|---|"]
        for paper_id in sorted(papers, reverse=True):
            lines.append(papers[paper_id].rstrip())
        lines.append("")
    return "\n".join(lines) + "\n"


def publish_catalog(config, data):
    canonical = config.get("json_readme_path", "docs/llm-agent-arxiv-daily.json")
    atomic_write_json(canonical, data)
    if config.get("publish_readme", True):
        path = config.get("md_readme_path", "README.md")
        panel = ""
        if Path(path).exists():
            match = re.search(
                r"<!-- STATS_PANEL_START -->.*?<!-- STATS_PANEL_END -->",
                Path(path).read_text(encoding="utf-8"), re.S,
            )
            if match:
                panel = match.group() + "\n\n"
        atomic_write_text(path, panel + render_catalog(data))
    if config.get("publish_gitpage", False):
        atomic_write_json(config["json_gitpage_path"], data)
        atomic_write_text(config["md_gitpage_path"], render_catalog(data, web=True))
    if config.get("publish_wechat", False):
        atomic_write_json(config["json_wechat_path"], data)
        atomic_write_text(config["md_wechat_path"], render_catalog(data))
