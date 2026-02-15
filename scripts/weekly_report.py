import os
import re
import json
import yaml
import logging
import datetime
import argparse
from typing import Dict, Any, List, Optional

logging.basicConfig(
    format='[%(asctime)s %(levelname)s] %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)

CONFIG_PATH = "config.yaml"
AGENT_CATEGORY = "Agent"
AGENT_TOP_N = 5


def ensure_dirs() -> None:
    os.makedirs("reports", exist_ok=True)
    os.makedirs(os.path.join("docs", "weekly"), exist_ok=True)


def load_config(path: str = CONFIG_PATH) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logging.warning("config file %s not found, use defaults", path)
        return {}
    except Exception as exc:  # noqa: BLE001
        logging.error("failed to load config %s: %s", path, exc)
        return {}


def load_json_data(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    except FileNotFoundError:
        logging.warning("json file %s not found, treat as empty", path)
        return {}
    except json.JSONDecodeError as exc:
        logging.error("failed to decode json %s: %s", path, exc)
        return {}
    except Exception as exc:  # noqa: BLE001
        logging.error("failed to load json %s: %s", path, exc)
        return {}


def strip_markdown_bold(text: str) -> str:
    text = text.strip()
    if text.startswith("**") and text.endswith("**") and len(text) >= 4:
        return text[2:-2].strip()
    return text


def parse_paper_row(
    category: str,
    paper_key: str,
    row: str,
) -> Optional[Dict[str, Any]]:
    parts = row.split("|")
    if len(parts) < 6:
        return None

    raw_date = strip_markdown_bold(parts[1].strip())
    title = strip_markdown_bold(parts[2].strip())
    authors = parts[3].strip()
    pdf_col = parts[4].strip()
    code_col = parts[5].strip() if len(parts) > 5 else ""
    model_col = parts[6].strip() if len(parts) > 6 else ""

    try:
        date_obj = datetime.date.fromisoformat(raw_date)
    except ValueError:
        logging.debug("invalid date for %s/%s: %s", category, paper_key, raw_date)
        return None

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
    result: Dict[str, List[Dict[str, Any]]] = {}
    for category, papers in data.items():
        if not isinstance(papers, dict):
            continue
        for paper_key, row in papers.items():
            if not isinstance(row, str):
                continue
            paper = parse_paper_row(category, paper_key, row)
            if paper is None:
                continue
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


def run_daily(json_path: str) -> None:
    today = datetime.date.today()
    data = load_json_data(json_path)
    papers_by_category = build_papers(data)
    filtered = filter_by_date_range(papers_by_category, today, today)
    markdown = generate_daily_markdown(today, filtered, AGENT_TOP_N)

    ensure_dirs()
    report_path = os.path.join("reports", f"daily-{today.isoformat()}.md")
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        logging.info("daily report written to %s", report_path)
    except Exception as exc:  # noqa: BLE001
        logging.error("failed to write daily report %s: %s", report_path, exc)


def run_weekly(json_path: str) -> None:
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=6)
    data = load_json_data(json_path)
    papers_by_category = build_papers(data)
    filtered = filter_by_date_range(papers_by_category, start_date, end_date)

    agent_papers = filtered.get(AGENT_CATEGORY, [])
    markdown = generate_weekly_markdown(start_date, end_date, agent_papers, AGENT_TOP_N)

    ensure_dirs()
    report_path = os.path.join("reports", f"weekly-{end_date.isoformat()}.md")
    weekly_index = os.path.join("docs", "weekly", "index.md")

    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        logging.info("weekly report written to %s", report_path)
    except Exception as exc:  # noqa: BLE001
        logging.error("failed to write weekly report %s: %s", report_path, exc)

    try:
        with open(weekly_index, "w", encoding="utf-8") as f:
            f.write(markdown)
        logging.info("weekly index written to %s", weekly_index)
    except Exception as exc:  # noqa: BLE001
        logging.error("failed to write weekly index %s: %s", weekly_index, exc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["daily", "weekly"],
        default="daily",
        help="report mode: daily or weekly",
    )
    args = parser.parse_args()

    config = load_config(CONFIG_PATH)
    json_path = config.get("json_readme_path", "./docs/llm-agent-arxiv-daily.json")

    if args.mode == "daily":
        run_daily(json_path)
    else:
        run_weekly(json_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        logging.error("unexpected error in weekly_report: %s", exc)
