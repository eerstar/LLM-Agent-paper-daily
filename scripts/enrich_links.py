"""Bounded GitHub-only enrichment; checkpoint each ID and publish one catalogue.

Cache timestamps are Unix seconds. Successful results can be replayed after a
failed publication; negatives expire in 30 days and transient failures in 15
minutes. Persist .cache/link-enrichment.json between scheduled invocations.
"""

import argparse
from contextlib import contextmanager
from email.utils import parsedate_to_datetime
import html
import logging
import math
import os
import re
import signal
import threading
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Timeout

from daily_arxiv import (
    format_md_row_6cols, load_config, parse_md_row_6cols, salvage_md_row_6cols,
)
from scripts.paper_store import (
    atomic_write_json, load_catalog, publish_catalog, read_json,
)


LOG = logging.getLogger(__name__)
CACHE_PATH = ".cache/link-enrichment.json"
GITHUB_SEARCH_API = "https://api.github.com/search/repositories"
NEGATIVE_TTL = 30 * 86400
TRANSIENT_TTL = 15 * 60
SEARCH_INTERVAL = 2.1
REQUEST_TIMEOUT = 15
ID_PATTERN = re.compile(r"(\d{4}\.\d{4,5}|[a-z][a-z.-]+/\d{7})(?:v\d+)?", re.I)
REPO_PATTERN = re.compile(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/?")
COLLECTION_PATTERN = re.compile(
    r"(?:^|[-_\s])(?:awesome|papers|survey|surveys|list|collection|reading|bibliography)"
    r"(?:$|[-_\s])", re.I,
)
STOP_WORDS = {"a", "an", "the", "for", "of", "in", "on", "to", "and", "with", "via", "using"}


class DeadlineReached(Exception):
    pass


def make_session():
    """One HTTP attempt per ID, with neither adapter nor application retries."""
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=0))
    return session


def _words(text):
    return set(re.findall(r"[a-z0-9]+", html.unescape(text).lower())) - STOP_WORDS


def select_repository(items, title):
    """Require title evidence in ID-search results; never rank by stars.

    Searching an exact arXiv ID in README/description establishes a citation,
    not ownership. Also require a paper-specific repository name or substantial
    title agreement in its description. Exclude collections, forks and ambiguous
    matches. Conservative false negatives are preferable to unrelated links.
    """
    title_words = _words(title)
    prefix = title.split(":", 1)[0] if ":" in title else ""
    prefix_name = re.sub(r"[^a-z0-9]", "", prefix.lower())
    matches = set()
    for item in items:
        if not isinstance(item, dict) or item.get("fork") or item.get("archived"):
            continue
        url = item.get("html_url", "")
        if not isinstance(url, str) or not REPO_PATTERN.fullmatch(url):
            continue
        name = url.rstrip("/").rsplit("/", 1)[-1]
        description = item.get("description") or ""
        if not isinstance(description, str) or COLLECTION_PATTERN.search(name):
            continue
        if re.search(r"\b(?:reading list|paper collection|list of papers|literature survey)\b",
                     description, re.I):
            continue
        compact_name = re.sub(r"[^a-z0-9]", "", name.lower())
        prefix_match = (
            len(prefix_name) >= 4 and len(_words(prefix)) <= 3
            and compact_name in {prefix_name, prefix_name + "code", prefix_name + "official"}
        )
        overlap = title_words & _words(description)
        title_match = len(overlap) >= 3 and len(overlap) >= 0.8 * len(title_words)
        if prefix_match or title_match:
            matches.add(url.rstrip("/"))
    return next(iter(matches)) if len(matches) == 1 else None


def _timestamp(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _load_state(path):
    state = read_json(path, {"version": 1, "papers": {}, "retry_after": 0})
    if not isinstance(state, dict) or state.get("version") != 1 or not isinstance(state.get("papers"), dict):
        raise ValueError("Invalid link-enrichment cache; refusing to discard checkpoint")
    for field in ("retry_after", "next_request_at"):
        if not _timestamp(state.get(field, 0)):
            raise ValueError("Invalid link-enrichment cache timestamp")
    for entry in state["papers"].values():
        if not isinstance(entry, dict) or not _timestamp(entry.get("last_attempt")) or not _timestamp(entry.get("retry_after")):
            raise ValueError("Invalid link-enrichment paper checkpoint")
        if entry.get("status") not in {
            "found", "not_found", "transient_error", "rate_limited", "unauthorized", "in_progress",
        }:
            raise ValueError("Invalid link-enrichment paper status")
        result = entry.get("result")
        if result is not None and (not isinstance(result, str) or not REPO_PATTERN.fullmatch(result)):
            raise ValueError("Invalid cached repository URL")
        if entry["status"] == "found" and not result:
            raise ValueError("Missing cached repository URL")
        if entry["status"] != "found" and result is not None:
            raise ValueError("Unexpected result in negative/error checkpoint")
    return state


def _retry_after(headers, now):
    """Use the latest applicable server hint, with a one-minute safety floor."""
    dates = [now + 60]
    retry = headers.get("Retry-After")
    if retry:
        try:
            seconds = float(retry)
            if _timestamp(seconds):
                dates.append(now + seconds)
        except (TypeError, ValueError):
            try:
                dates.append(parsedate_to_datetime(retry).timestamp())
            except (TypeError, ValueError, OverflowError):
                pass
    try:
        reset = float(headers.get("X-RateLimit-Reset", 0))
        if _timestamp(reset):
            dates.append(reset)
    except (TypeError, ValueError):
        pass
    return max(dates)


def _search(session, paper_id, title, token, remaining, clock):
    response = None
    cooldown = 0
    try:
        response = session.get(
            GITHUB_SEARCH_API,
            params={"q": f'"{paper_id}" in:readme,description', "per_page": 10},
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "User-Agent": "llm-agent-link-enrichment/1.0",
            },
            timeout=Timeout(total=min(REQUEST_TIMEOUT, remaining),
                            connect=min(5, remaining), read=min(10, remaining)),
            allow_redirects=False,
        )
        now = clock.time()
        if response.status_code in (403, 429):
            return "rate_limited", None, _retry_after(response.headers, now)
        if response.status_code == 401:
            return "unauthorized", None, now + TRANSIENT_TTL
        if response.headers.get("X-RateLimit-Remaining") == "0":
            cooldown = _retry_after(response.headers, now)
        if response.status_code != 200:
            return "transient_error", None, cooldown
        payload = response.json()
        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("items"), list)
            or payload.get("incomplete_results", False)
            or not all(isinstance(item, dict) for item in payload["items"])
            or not isinstance(payload.get("total_count"), int)
            or payload["total_count"] != len(payload["items"])
        ):
            return "transient_error", None, cooldown
        result = select_repository(payload["items"], title)
        return ("found" if result else "not_found"), result, cooldown
    except (requests.RequestException, ValueError):
        # Exceptions and response bodies may contain credentials. Log only status.
        return "transient_error", None, cooldown
    finally:
        if response is not None:
            response.close()


def _missing(code):
    return not code or code.strip().lower() == "null"


def _parse_row(paper_id, row):
    parsed = parse_md_row_6cols(row)
    if parsed:
        return parsed
    parsed = salvage_md_row_6cols(paper_id, row)
    if not parsed:
        return None
    # Salvage guesses link order. A recognizable PDF/code/model tail is stronger
    # evidence, especially for legacy IDs or rows with only a Model link.
    tail = re.search(
        r"\[[^\]]+\]\((?P<pdf>[^)]+)\)\|(?P<code>[^|]*)\|(?P<model>[^|]*)\|\s*$",
        row,
    )
    if not tail:
        return None
    parsed.update({
        "pdf_url": tail["pdf"].strip(), "code": tail["code"].strip(),
        "model": tail["model"].strip(),
    })
    return parsed


def _fill_rows(data, rows, code):
    count = 0
    for category, key, parsed in rows:
        if not _missing(parsed["code"]):
            continue
        data[category][key] = format_md_row_6cols(
            date=parsed["date"], title=parsed["title"], authors=parsed["authors"],
            arxiv_id=parsed["arxiv_id"], pdf_url=parsed["pdf_url"],
            code_cell=code, model_cell=parsed["model"],
        )
        parsed["code"] = code
        count += 1
    return count


def _step_summary(text):
    LOG.info("%s", text)
    if path := os.getenv("GITHUB_STEP_SUMMARY"):
        try:
            with open(path, "a", encoding="utf-8") as stream:
                stream.write(text + "\n\n")
        except OSError:
            LOG.warning("Unable to append link-enrichment step summary")


@contextmanager
def _finish_uninterrupted():
    previous = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, signal.SIG_IGN)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def enrich_links(config, session=None, *, max_papers=40, max_seconds=600, clock=None):
    """Enrich one merged catalogue and return success for bounded/optional stops.

    Storage errors propagate: a corrupt checkpoint or failed publication is not
    an empty catalogue. Inject a requests-compatible session and time-compatible
    clock for offline testing. The caller must serialize catalogue writers.
    """
    if max_papers <= 0 or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("Enrichment budgets must be positive and finite")
    token = os.getenv("GITHUB_TOKEN", "").strip() or os.getenv("GH_TOKEN", "").strip()
    if not token:
        _step_summary("Link enrichment skipped: set GITHUB_TOKEN or GH_TOKEN; attempted=0.")
        return 0

    clock = clock or time
    started = clock.monotonic()
    deadline = started + max_seconds
    path = config.get("enrichment_state_path", CACHE_PATH)
    state = _load_state(path)
    data = load_catalog(config)
    stats = {
        "status": "running", "attempted": 0, "found": 0, "updated_rows": 0,
        "cached": 0, "deferred": 0, "invalid_rows": 0,
    }
    state["last_run"] = stats
    owned_session = session is None

    def check_deadline():
        if clock.monotonic() >= deadline:
            raise DeadlineReached

    try:
        groups = {}
        for category, papers in data.items():
            for key, row in papers.items():
                check_deadline()
                match = ID_PATTERN.fullmatch(key)
                if not match:
                    stats["invalid_rows"] += 1
                    continue
                paper_id = match.group(1).lower()
                parsed = _parse_row(paper_id, row)
                if not parsed:
                    stats["invalid_rows"] += 1
                    continue
                groups.setdefault(paper_id, []).append((category, key, parsed))

        pending = []
        for paper_id, rows in groups.items():
            check_deadline()
            existing = next((parsed["code"] for _, _, parsed in rows if not _missing(parsed["code"])), None)
            entry = state["papers"].get(paper_id, {})
            if existing:
                stats["updated_rows"] += _fill_rows(data, rows, existing)
            elif entry.get("result"):
                stats["updated_rows"] += _fill_rows(data, rows, f'**[link]({entry["result"]})**')
                stats["cached"] += 1
            elif entry.get("retry_after", 0) > clock.time():
                stats["deferred"] += 1
            else:
                pending.append(paper_id)

        # New IDs first, then least-recently attempted IDs; category order cannot
        # starve the rest of the catalogue when a transient result expires.
        pending.sort(key=lambda paper_id: (
            state["papers"].get(paper_id, {}).get("last_attempt", -1), paper_id,
        ))
        if state.get("retry_after", 0) > clock.time():
            stats["status"] = "rate_limited"
        else:
            for paper_id in pending:
                check_deadline()
                if stats["attempted"] >= max_papers:
                    stats["status"] = "max_papers"
                    break
                wait = max(0, state.get("next_request_at", 0) - clock.time())
                if wait >= deadline - clock.monotonic():
                    raise DeadlineReached
                if wait:
                    clock.sleep(wait)
                check_deadline()
                session = session if session is not None else make_session()
                attempted_at = clock.time()
                entry = {
                    "status": "in_progress", "result": None,
                    "last_attempt": attempted_at,
                    "retry_after": attempted_at + TRANSIENT_TTL,
                }
                state["papers"][paper_id] = entry
                state["next_request_at"] = attempted_at + SEARCH_INTERVAL
                stats["attempted"] += 1
                atomic_write_json(path, state)
                check_deadline()
                status, result, cooldown = _search(
                    session, paper_id, groups[paper_id][0][2]["title"],
                    token, deadline - clock.monotonic(), clock,
                )
                entry.update({
                    "status": status, "result": result,
                    "retry_after": 0 if result else clock.time() + (
                        NEGATIVE_TTL if status == "not_found" else TRANSIENT_TTL
                    ),
                })
                state["next_request_at"] = clock.time() + SEARCH_INTERVAL
                if cooldown:
                    state["retry_after"] = cooldown
                    if status in ("rate_limited", "unauthorized"):
                        entry["retry_after"] = cooldown
                if result:
                    stats["found"] += 1
                    stats["updated_rows"] += _fill_rows(data, groups[paper_id], f"**[link]({result})**")
                atomic_write_json(path, state)
                LOG.info("Link enrichment: id=%s status=%s attempted=%d/%d updated_rows=%d",
                         paper_id, status, stats["attempted"], max_papers, stats["updated_rows"])
                if cooldown:
                    stats["status"] = "unauthorized" if status == "unauthorized" else "rate_limited"
                    break
            else:
                stats["status"] = "complete"
    except DeadlineReached:
        stats["status"] = "deadline"
    except KeyboardInterrupt:
        stats["status"] = "interrupted"
    except Exception:
        stats["status"] = "error"
        raise
    finally:
        with _finish_uninterrupted():
            stats["elapsed_seconds"] = round(clock.monotonic() - started, 3)
            stats["finished_at"] = clock.time()
            try:
                # Save replayable positive results before publishing any view.
                atomic_write_json(path, state)
                try:
                    publish_catalog(config, data)
                except Exception:
                    stats["status"] = "publication_error"
                    atomic_write_json(path, state)
                    raise
            finally:
                if owned_session and session is not None:
                    session.close()
                _step_summary(
                    "Link enrichment: " + "; ".join(
                        f"{key}={stats[key]}" for key in (
                            "status", "attempted", "found", "updated_rows", "cached",
                            "deferred", "invalid_rows", "elapsed_seconds",
                        )
                    ) + "."
                )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", default="config.yaml")
    parser.add_argument("--max-papers", type=int, default=40)
    parser.add_argument("--max-seconds", type=float, default=600)
    args = parser.parse_args(argv)
    if args.max_papers <= 0 or not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error("budgets must be positive and finite")
    config = load_config(args.config_path)

    def interrupt(signum, frame):
        # A second SIGTERM must not interrupt the checkpoint/publication cleanup.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        return enrich_links(config, max_papers=args.max_papers, max_seconds=args.max_seconds)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
