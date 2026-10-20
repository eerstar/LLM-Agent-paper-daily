import os
import re
import sys
import yaml
import logging
import datetime
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.paper_store import read_json

logging.basicConfig(
    format='[%(asctime)s %(levelname)s] %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)

CONFIG_PATH = "config.yaml"
HARVEST_STATE_PATH = ".cache/harvest.json"
AGENT_CATEGORY = "Agent"
AGENT_TOP_N = 5


def ensure_dirs() -> None:
    os.makedirs("reports", exist_ok=True)
    os.makedirs(os.path.join("docs", "weekly"), exist_ok=True)


def load_config(path: str = CONFIG_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid configuration in {path}: expected an object")
    return config


def load_json_data(path: str) -> Dict[str, Any]:
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Missing or invalid JSON object in {path}")
    return data


def parse_timestamp(value: Any) -> datetime.datetime:
    if not isinstance(value, str):
        raise ValueError("Harvest timestamp must be an ISO8601 string")
    timestamp = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("Harvest timestamp must include a UTC offset")
    return timestamp.astimezone(datetime.timezone.utc)


def validate_harvest(config: Dict[str, Any], now: datetime.datetime) -> None:
    state = load_json_data(config.get("harvest_state_path", HARVEST_STATE_PATH))
    expected_topics = config.get("keywords", config.get("kv"))
    if not isinstance(expected_topics, dict) or not expected_topics:
        raise ValueError("No harvest topics configured")
    max_days = config.get("freshness_max_days", 3)
    if type(max_days) not in (int, float) or not max_days > 0:
        raise ValueError("freshness_max_days must be positive")
    max_age = datetime.timedelta(days=max_days)

    def require_fresh(value: Any, label: str) -> None:
        age = now - parse_timestamp(value)
        if age < datetime.timedelta(0) or age > max_age:
            raise ValueError(f"Stale or future harvest timestamp: {label}")

    require_fresh(state.get("last_success_at"), "last_success_at")
    last_run = state.get("last_run")
    if (
        not isinstance(last_run, dict)
        or last_run.get("status") != "success"
        or last_run.get("failed_topics") != []
    ):
        raise ValueError("Latest harvest did not succeed for all configured topics")
    require_fresh(last_run.get("finished_at"), "last_run.finished_at")
    topics = state.get("topics")
    if not isinstance(topics, dict):
        raise ValueError("Harvest state has no topic success records")
    for topic in expected_topics:
        record = topics.get(topic)
        if not isinstance(record, dict):
            raise ValueError(f"Missing successful harvest for topic: {topic}")
        require_fresh(record.get("last_success_at"), topic)


def strip_markdown_bold(text: str) -> str:
    text = text.strip()
    if text.startswith("**") and text.endswith("**") and len(text) >= 4:
        return text[2:-2].strip()
    return text


def parse_paper_row(
    category: str,
    paper_key: str,
    row: str,
) -> Dict[str, Any]:
    parts = row.split("|")
    if len(parts) < 7 or parts[0].strip():
        raise ValueError(f"Invalid paper row: {category}/{paper_key}")

    raw_date = strip_markdown_bold(parts[1].strip())
    title = strip_markdown_bold(parts[2].strip())
    authors = parts[3].strip()
    pdf_col = parts[4].strip()
    code_col = parts[5].strip() if len(parts) > 5 else ""
    model_col = parts[6].strip() if len(parts) > 6 else ""

    try:
        date_obj = datetime.date.fromisoformat(raw_date)
    except ValueError as exc:
        raise ValueError(f"Invalid paper date: {category}/{paper_key}: {raw_date}") from exc

    # extract arxiv url from markdown link if present
    m = re.search(r"\((http[^)]+)\)", pdf_col)
    if m:
        arxiv_url = m.group(1)
    else:
        arxiv_url = pdf_col

    has_code = bool(code_col and code_col.lower() != "null")
    has_model = bool(model_col and model_col.lower() != "null")

    return {
        "category": category,
        "key": paper_key,
        "date": date_obj,
        "title": title,
        "authors": authors,
        "arxiv_url": arxiv_url,
        "has_code": has_code,
        "has_model": has_model,
    }


def build_papers(data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    if not isinstance(data, dict):
        raise ValueError("Catalogue must be an object")
    result: Dict[str, List[Dict[str, Any]]] = {}
    for category, papers in data.items():
        if not isinstance(papers, dict):
            raise ValueError(f"Invalid catalogue category: {category}")
        for paper_key, row in papers.items():
            if not isinstance(row, str):
                raise ValueError(f"Invalid paper row: {category}/{paper_key}")
            paper = parse_paper_row(category, paper_key, row)
            result.setdefault(category, []).append(paper)
    return result


def filter_by_date_range(
    papers_by_category: Dict[str, List[Dict[str, Any]]],
    start_date: datetime.date,
    end_date: datetime.date,
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for category, papers in papers_by_category.items():
        filtered = [
            p for p in papers if start_date <= p["date"] <= end_date
        ]
        if filtered:
            out[category] = filtered
    return out


def sort_papers(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        papers,
        key=lambda p: (p["date"], p["key"]),
        reverse=True,
    )


def format_paper_line(p: Dict[str, Any]) -> str:
    code_flag = "✅ Code" if p["has_code"] else "❌ Code"
    model_flag = "✅ Model" if p["has_model"] else "❌ Model"
    return (
        f"- [{p['date'].isoformat()}] **{p['title']}** — {p['authors']}. "
        f"[arXiv]({p['arxiv_url']}) · {code_flag} · {model_flag}"
    )


def generate_daily_markdown(
    target_date: datetime.date,
    papers_by_category: Dict[str, List[Dict[str, Any]]],
    agent_top_n: int,
) -> str:
    date_str = target_date.isoformat()
    lines: List[str] = []
    lines.append(f"# Daily Report {date_str}")
    lines.append("")
    lines.append(f"统计日期：{date_str}")
    lines.append("")

    if not papers_by_category:
        lines.append("_本日未找到新增论文。_")
        return "\n".join(lines)

    for category in sorted(papers_by_category.keys()):
        papers = sort_papers(papers_by_category[category])
        if category == AGENT_CATEGORY:
            papers = papers[:agent_top_n]
        lines.append(f"## {category}")
        lines.append("")
        for p in papers:
            lines.append(format_paper_line(p))
        lines.append("")

    return "\n".join(lines)


def generate_weekly_markdown(
    start_date: datetime.date,
    end_date: datetime.date,
    agent_papers: List[Dict[str, Any]],
    agent_top_n: int,
) -> str:
    end_str = end_date.isoformat()
    start_str = start_date.isoformat()
    lines: List[str] = []
    lines.append(f"# Weekly Report {end_str}")
    lines.append("")
    lines.append(f"统计区间：{start_str} ~ {end_str}")
    lines.append("")
    lines.append(f"## Agent Top {agent_top_n}")
    lines.append("")

    if not agent_papers:
        lines.append("_本周未找到 Agent 相关论文。_")
        return "\n".join(lines)

    papers_sorted = sort_papers(agent_papers)[:agent_top_n]
    for p in papers_sorted:
        lines.append(format_paper_line(p))

    return "\n".join(lines)


def run_daily(json_path: str, target_date: Optional[datetime.date] = None) -> None:
    today = target_date or datetime.datetime.now(datetime.timezone.utc).date()
    data = load_json_data(json_path)
    papers_by_category = build_papers(data)
    filtered = filter_by_date_range(papers_by_category, today, today)
    markdown = generate_daily_markdown(today, filtered, AGENT_TOP_N)

    ensure_dirs()
    report_path = os.path.join("reports", f"daily-{today.isoformat()}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    logging.info("daily report written to %s", report_path)


def run_weekly(
    json_path: str,
    end_date: Optional[datetime.date] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    config = load_config(CONFIG_PATH) if config is None else config
    now = datetime.datetime.now(datetime.timezone.utc)
    validate_harvest(config, now)
    categories = config.get("weekly_categories")
    expected_topics = config.get("keywords", config.get("kv", {}))
    if (
        not isinstance(categories, list)
        or not categories
        or any(not isinstance(c, str) or c not in expected_topics for c in categories)
    ):
        raise ValueError("weekly_categories must name configured harvest topics")
    end_date = end_date or now.date() - datetime.timedelta(days=1)
    start_date = end_date - datetime.timedelta(days=6)
    data = load_json_data(json_path)
    papers_by_category = build_papers(data)
    filtered = filter_by_date_range(papers_by_category, start_date, end_date)

    # A paper can match several topics; retain the newest row for each key.
    unique_papers: Dict[str, Dict[str, Any]] = {}
    for category in categories:
        for paper in filtered.get(category, []):
            previous = unique_papers.get(paper["key"])
            if previous is None or paper["date"] > previous["date"]:
                unique_papers[paper["key"]] = paper
    agent_papers = list(unique_papers.values())
    markdown = generate_weekly_markdown(start_date, end_date, agent_papers, AGENT_TOP_N)

    ensure_dirs()
    report_path = os.path.join("reports", f"weekly-{end_date.isoformat()}.md")
    weekly_index = os.path.join("docs", "weekly", "index.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    logging.info("weekly report written to %s", report_path)

    with open(weekly_index, "w", encoding="utf-8") as f:
        f.write(markdown)
    logging.info("weekly index written to %s", weekly_index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["daily", "weekly"],
        default="daily",
        help="report mode: daily or weekly",
    )
    parser.add_argument(
        "--date",
        type=datetime.date.fromisoformat,
        help="inclusive report end date (YYYY-MM-DD); weekly defaults to yesterday UTC",
    )
    args = parser.parse_args()

    config = load_config(CONFIG_PATH)
    json_path = config.get("json_readme_path", "./docs/llm-agent-arxiv-daily.json")

    if args.mode == "daily":
        run_daily(json_path, args.date)
    else:
        run_weekly(json_path, args.date, config)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        logging.error("unexpected error in weekly_report: %s", exc)
        raise SystemExit(1) from exc
