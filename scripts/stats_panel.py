import os
import re
import sys
import yaml
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import requests

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.paper_store import atomic_write_json, read_json
from scripts.weekly_report import build_papers, parse_timestamp

logging.basicConfig(
    format='[%(asctime)s %(levelname)s] %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)

CONFIG_PATH = "config.yaml"
CACHE_DIR = ".cache"
CACHE_FILE = os.path.join(CACHE_DIR, "stats.json")
HARVEST_STATE_PATH = os.path.join(CACHE_DIR, "harvest.json")

STATS_PANEL_START = "<!-- STATS_PANEL_START -->"
STATS_PANEL_END = "<!-- STATS_PANEL_END -->"

GITHUB_API_BASE = "https://api.github.com"
USER_AGENT = "llm-agent-stats-panel/1.0 (+https://github.com/eerstar/LLM-Agent-paper-daily)"


def ensure_dirs() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)


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


def strip_markdown_bold(text: str) -> str:
    text = text.strip()
    # remove leading and trailing '**' if present
    if text.startswith("**") and text.endswith("**") and len(text) >= 4:
        return text[2:-2].strip()
    return text


def parse_date_from_row(row: str) -> Optional[datetime.date]:
    # row example: "|**2025-07-23**|**Title**|Author et.al.|[id](url)|..."
    parts = row.split("|")
    if len(parts) < 3:
        return None
    raw_date = strip_markdown_bold(parts[1].strip())
    try:
        return datetime.date.fromisoformat(raw_date)
    except ValueError:
        logging.debug("failed to parse date from %s", raw_date)
        return None


def compute_stats(
    data: Dict[str, Any],
    target_date: datetime.date,
    harvest_state: Optional[Dict[str, Any]] = None,
) -> Tuple[int, int]:
    papers_by_category = build_papers(data)
    total_keys = {
        paper["key"] for papers in papers_by_category.values() for paper in papers
    }
    if harvest_state is None:
        harvest_state = load_json_data(HARVEST_STATE_PATH)
    last_run = harvest_state.get("last_run")
    if not isinstance(last_run, dict):
        raise ValueError("Harvest state has no last_run")
    if last_run.get("status") not in ("success", "partial", "failed"):
        raise ValueError("Invalid harvest run status")
    new_papers = last_run.get("new_papers")
    if type(new_papers) is not int or new_papers < 0:
        raise ValueError("Harvest new_papers must be a nonnegative integer")
    finished_at = parse_timestamp(last_run.get("finished_at"))
    today_new = new_papers if finished_at.date() == target_date else 0
    return today_new, len(total_keys)


def get_repo_context(config: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    repo_env = os.getenv("GITHUB_REPOSITORY")
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")

    owner: Optional[str]
    repo: Optional[str]

    if repo_env:
        try:
            owner, repo = repo_env.split("/", 1)
        except ValueError:
            logging.error("invalid GITHUB_REPOSITORY: %s", repo_env)
            owner = None
            repo = None
    else:
        owner = config.get("user_name")
        repo = config.get("repo_name")

    return owner, repo, token


def fetch_stars(owner: Optional[str], repo: Optional[str], token: Optional[str]) -> str:
    if not owner or not repo:
        logging.warning("owner/repo not available, skip star fetch")
        return "-"

    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": USER_AGENT,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        resp = requests.get(url, headers=headers, timeout=8)
        if resp.status_code != 200:
            logging.warning(
                "GitHub repo API non-200 status %s: %s",
                resp.status_code,
                resp.text[:200],
            )
            return "-"
        data = resp.json()
        stars = data.get("stargazers_count")
        if type(stars) is int and stars >= 0:
            return str(stars)
        return "-"
    except Exception as exc:  # noqa: BLE001
        logging.error("failed to fetch GitHub stars: %s", exc)
        return "-"


def load_stats_cache() -> Dict[str, Any]:
    cache = read_json(CACHE_FILE, {})
    if not isinstance(cache, dict):
        raise ValueError(f"Invalid stats cache object in {CACHE_FILE}")
    return cache


def save_stats_cache(cache: Dict[str, Any]) -> None:
    atomic_write_json(CACHE_FILE, cache)


def create_milestone_issue(
    total_papers: int,
    today: datetime.date,
    owner: Optional[str],
    repo: Optional[str],
    token: Optional[str],
) -> bool:
    if os.getenv("ENABLE_MILESTONE_ISSUES", "").lower() != "true":
        return False
    if not (owner and repo and token):
        logging.info("GitHub context or token missing, skip milestone issue")
        return False

    milestone_base = (total_papers // 100) * 100
    if milestone_base < 100:
        return False

    title = f"Milestone reached: {milestone_base} papers on {today.isoformat()}"
    body = (
        f"This repository has reached **{milestone_base}** papers in total on {today.isoformat()}.\n\n"
        f"Total papers counted from docs/llm-agent-arxiv-daily.json."
    )

    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/issues"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": USER_AGENT,
        "Authorization": f"Bearer {token}",
    }
    payload = {"title": title, "body": body}

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=8)
        if resp.status_code not in (200, 201):
            logging.warning(
                "failed to create milestone issue (status %s): %s",
                resp.status_code,
                resp.text[:200],
            )
            return False
        issue_url = resp.json().get("html_url")
        logging.info("milestone issue created: %s", issue_url)
        return True
    except Exception as exc:  # noqa: BLE001
        logging.error("exception when creating milestone issue: %s", exc)
        return False


def update_readme_panel(
    readme_path: str,
    today_new: int,
    total_papers: int,
    stars: str,
    today: datetime.date,
) -> None:
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    panel_line = (
        f"📈 **Latest run added (today UTC)**: +{today_new} papers ｜ "
        f"**Total**: {total_papers} ｜ "
        f"**Stars**: {stars}"
    )
    panel_block = f"{STATS_PANEL_START}\n{panel_line}\n{STATS_PANEL_END}"

    if STATS_PANEL_START in content and STATS_PANEL_END in content:
        # replace existing panel block
        pattern = re.compile(
            re.escape(STATS_PANEL_START) + r".*?" + re.escape(STATS_PANEL_END),
            re.DOTALL,
        )
        new_content = pattern.sub(panel_block, content)
    else:
        # insert before the first "## Updated on" line if exists
        marker = "## Updated on"
        idx = content.find(marker)
        if idx != -1:
            new_content = content[:idx] + panel_block + "\n\n" + content[idx:]
        else:
            new_content = panel_block + "\n\n" + content

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    logging.info(
        "stats panel updated: today=%s, total=%s, stars=%s",
        today_new,
        total_papers,
        stars,
    )


def handle_milestone(
    total_papers: int,
    today: datetime.date,
    owner: Optional[str],
    repo: Optional[str],
    token: Optional[str],
) -> None:
    cache = load_stats_cache()
    last_milestone = int(cache.get("last_milestone_total", 0))

    if total_papers < 100:
        cache["last_total"] = total_papers
        cache["last_run_date"] = today.isoformat()
        save_stats_cache(cache)
        return

    current_milestone = (total_papers // 100) * 100

    if current_milestone <= last_milestone:
        cache["last_total"] = total_papers
        cache["last_run_date"] = today.isoformat()
        save_stats_cache(cache)
        logging.info(
            "no new milestone: total=%s, last_milestone=%s", total_papers, last_milestone
        )
        return

    created = False
    if os.getenv("ENABLE_MILESTONE_ISSUES", "").lower() == "true":
        created = create_milestone_issue(total_papers, today, owner, repo, token)
    if created:
        cache["last_milestone_total"] = current_milestone
    cache["last_total"] = total_papers
    cache["last_run_date"] = today.isoformat()
    save_stats_cache(cache)


def main() -> None:
    ensure_dirs()
    config = load_config(CONFIG_PATH)

    json_path = config.get("json_readme_path", "./docs/llm-agent-arxiv-daily.json")
    readme_path = config.get("md_readme_path", "README.md")

    data = load_json_data(json_path)
    today = datetime.datetime.now(datetime.timezone.utc).date()
    harvest_state = load_json_data(config.get("harvest_state_path", HARVEST_STATE_PATH))

    today_new, total_papers = compute_stats(data, today, harvest_state)

    cache = load_stats_cache()
    owner, repo, token = get_repo_context(config)
    stars = fetch_stars(owner, repo, token)
    if stars.isdigit():
        cache["last_star_count"] = int(stars)
    else:
        cached_stars = cache.get("last_star_count")
        if type(cached_stars) is int and cached_stars >= 0:
            stars = str(cached_stars)

    update_readme_panel(readme_path, today_new, total_papers, stars, today)
    save_stats_cache(cache)
    handle_milestone(total_papers, today, owner, repo, token)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        logging.error("unexpected error in stats_panel: %s", exc)
        raise SystemExit(1) from exc
