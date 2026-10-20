"""Bounded, resumable arXiv harvesting without optional lookup dependencies."""

import argparse
import datetime as dt
import logging
import os
import re
import time
import xml.etree.ElementTree as ET

import arxiv
import requests

from scripts.paper_store import (
    atomic_write_json, load_catalog, merge_catalog, publish_catalog, read_json,
)

UTC = dt.timezone.utc


class BudgetExceeded(RuntimeError):
    pass


class TimeoutSession(requests.Session):
    deadline = None

    def request(self, method, url, **kwargs):
        response = None
        for attempt in range(3):
            remaining = 90 if self.deadline is None else self.deadline - time.monotonic()
            if remaining <= 0:
                raise BudgetExceeded("network time budget reached")
            kwargs["timeout"] = (min(10, remaining), min(60, remaining))
            try:
                response = super().request(method, url, **kwargs)
                if response.status_code not in (429, 500, 502, 503, 504):
                    break
                wait = 30 * (attempt + 1) if response.status_code == 429 else 5 * (attempt + 1)
                retry_after = response.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    wait = max(wait, int(retry_after))
                response.close()
            except (requests.Timeout, requests.ConnectionError):
                if attempt == 2:
                    raise
                wait = 5 * (attempt + 1)
            if attempt == 2:
                raise requests.HTTPError(f"arXiv transient response after 3 attempts: {response.status_code}")
            if self.deadline is not None and time.monotonic() + wait >= self.deadline:
                raise BudgetExceeded("source cooldown exceeds remaining budget")
            logging.warning("arXiv transient failure; retry %s/2 after %ss", attempt + 1, wait)
            time.sleep(wait)
        if response.status_code == 200:
            try:
                root = ET.fromstring(response.content)
                atom = "{http://www.w3.org/2005/Atom}"
                opensearch = "{http://a9.com/-/spec/opensearch/1.1/}"
                if root.tag != atom + "feed":
                    raise ValueError("arXiv response is not an Atom feed")
                total = int(root.findtext(opensearch + "totalResults", "-1"))
                offset = int(root.findtext(opensearch + "startIndex", "-1"))
                entries = root.findall(atom + "entry")
                if total < 0 or offset < 0 or (not entries and total > offset):
                    raise ValueError("arXiv response has missing or inconsistent pagination")
                for entry in entries:
                    for name in ("id", "title", "published", "updated"):
                        if not entry.findtext(atom + name):
                            raise ValueError(f"arXiv entry missing {name}")
                    if "/api/errors" in entry.findtext(atom + "id"):
                        raise ValueError("arXiv returned an error entry")
            except ET.ParseError as exc:
                raise ValueError("arXiv returned invalid XML") from exc
        return response


def make_client():
    client = arxiv.Client(page_size=500, delay_seconds=5, num_retries=0)
    # arxiv 4.0.1 has no public timeout option; keep this adapter version-tested.
    client._session.close()
    client._session = TimeoutSession()
    return client


def utc_now():
    return dt.datetime.now(UTC).isoformat()


def safe_cell(value):
    return re.sub(r"\s+", " ", str(value)).strip().replace("|", "&#124;")


def result_row(result):
    paper_id = re.sub(r"v\d+$", "", result.get_short_id())
    text = " ".join([
        result.summary or "", result.comment or "",
        *[link.href for link in result.links],
    ])
    code = re.search(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", text)
    model = re.search(r"https://huggingface\.co/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", text)
    code_cell = f"**[link]({code.group().rstrip('.')})**" if code else "null"
    model_cell = f"**[link]({model.group().rstrip('.')})**" if model else "null"
    author = safe_cell(result.authors[0]) if result.authors else "unknown"
    row = (
        f"|**{result.published.date()}**|**{safe_cell(result.title)}**|{author} et.al.|"
        f"[{paper_id}](https://arxiv.org/abs/{paper_id})|{code_cell}|{model_cell}|\n"
    )
    return paper_id, row


def date_query(query, start, end):
    return f"({query}) AND submittedDate:[{start:%Y%m%d}0000 TO {end:%Y%m%d}2359]"


def step_summary(text):
    logging.info(text)
    if path := os.getenv("GITHUB_STEP_SUMMARY"):
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(text + "\n\n")


def harvest_daily(config, client=None, max_seconds=900, max_per_topic=1500):
    client = client or make_client()
    deadline = time.monotonic() + max_seconds
    if isinstance(getattr(client, "_session", None), TimeoutSession):
        client._session.deadline = deadline
    data = load_catalog(config)
    before = {key for papers in data.values() for key in papers}
    state_path = config.get("harvest_state_path", ".cache/harvest.json")
    state = read_json(state_path, {})
    state.setdefault("topics", {})
    today = dt.datetime.now(UTC).date()
    default_start = today - dt.timedelta(days=config.get("lookback_days", 3))
    failed, succeeded = [], []
    recovering = state.get("last_run", {}).get("status") in ("partial", "failed", "running")
    for topic, query in config["kv"].items():
        batch = {}
        record = state["topics"].setdefault(topic, {})
        if recovering and not record.get("pending") and record.get("covered_through") == today.isoformat():
            succeeded.append(topic)
            continue
        previous = record.get("covered_through") or record.get("last_success_at")
        start = min(default_start, dt.date.fromisoformat(previous[:10])) if previous else default_start
        pending = record.setdefault("pending", {
            "start": start.isoformat(), "end": today.isoformat(), "offset": 0, "query": query,
        })
        if pending["query"] != query:
            raise ValueError(f"Unfinished harvest query changed: {topic}")
        # Persist even the first failed attempt's date boundary.
        atomic_write_json(state_path, state)
        try:
            if time.monotonic() >= deadline:
                raise BudgetExceeded("harvest time budget reached")
            start, end = dt.date.fromisoformat(pending["start"]), dt.date.fromisoformat(pending["end"])
            search = arxiv.Search(
                query=date_query(query, start, end), max_results=None,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Ascending,
            )
            for result in client.results(search, offset=pending["offset"]):
                if time.monotonic() >= deadline:
                    raise BudgetExceeded("harvest time budget reached")
                if len(batch) >= max_per_topic:
                    raise BudgetExceeded(f"{topic}: per-topic cap reached; next run resumes saved offset")
                paper_id, row = result_row(result)
                batch[paper_id] = row
                pending["offset"] += 1
            record["covered_through"] = end.isoformat()
            del record["pending"]
            if end < today:
                raise BudgetExceeded(f"{topic}: recovery window complete; next run fetches remaining recent days")
            succeeded.append(topic)
            record.update({"last_success_at": utc_now(), "papers": len(batch)})
            record.pop("last_error", None)
        except Exception as exc:
            failed.append(topic)
            logging.exception("Harvest failed for %s: %s", topic, exc)
            state["topics"].setdefault(topic, {})["last_error"] = str(exc)[:300]
        data = merge_catalog(data, {topic: batch})
        # A failed topic can still have yielded valid papers before the error.
        atomic_write_json(config.get("json_readme_path", "docs/llm-agent-arxiv-daily.json"), data)
        state["last_run"] = {
            "status": "running", "finished_at": utc_now(),
            "failed_topics": failed, "succeeded_topics": succeeded,
        }
        atomic_write_json(state_path, state)
    publish_catalog(config, data)
    added = len({key for papers in data.values() for key in papers} - before)
    state["last_run"] = {
        "status": "partial" if failed and succeeded else "failed" if failed else "success",
        "finished_at": utc_now(), "new_papers": added,
        "failed_topics": failed, "succeeded_topics": succeeded,
    }
    if not failed:
        state["last_success_at"] = utc_now()
    atomic_write_json(state_path, state)
    step_summary(f"Harvest: {len(succeeded)} topics succeeded; {len(failed)} failed; {added} new unique papers.")
    return 1 if failed else 0


def harvest_backfill(
    config, client=None, start_date="2026-05-15", end_date=None,
    max_papers=1000, max_seconds=600,
):
    client = client or make_client()
    state_path = config.get("backfill_state_path", ".cache/backfill.json")
    state = read_json(state_path, {})
    requested_end = end_date or (dt.datetime.now(UTC).date() - dt.timedelta(days=1)).isoformat()
    if not state:
        state = {
            "start_date": start_date, "end_date": requested_end, "day": start_date,
            "topic_index": 0, "offset": 0, "complete": False,
            "topics": list(config["kv"]), "queries": config["kv"],
        }
    if state["start_date"] != start_date or (end_date and state["end_date"] != end_date):
        raise ValueError("Backfill range differs from saved cursor; use a separate backfill_state_path.")
    if state["topics"] != list(config["kv"]) or state["queries"] != config["kv"]:
        raise ValueError("Backfill queries changed; use a separate cursor instead of silently skipping papers.")
    if state["complete"]:
        if not state.get("published"):
            publish_catalog(config, load_catalog(config))
            state["published"] = True
            atomic_write_json(state_path, state)
        step_summary("Backfill already complete.")
        return 0
    start, end = dt.date.fromisoformat(state["day"]), dt.date.fromisoformat(state["end_date"])
    if start > end:
        raise ValueError("Backfill start must not be after end.")
    data = load_catalog(config)
    deadline = time.monotonic() + max_seconds
    if isinstance(getattr(client, "_session", None), TimeoutSession):
        client._session.deadline = deadline
    processed, error = 0, None
    try:
        while start <= end:
            topic = state["topics"][state["topic_index"]]
            query = date_query(config["kv"][topic], start, start)
            search = arxiv.Search(
                query=query, max_results=None,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Ascending,
            )
            batch = {}
            try:
                for result in client.results(search, offset=state["offset"]):
                    if processed >= max_papers or time.monotonic() >= deadline:
                        raise BudgetExceeded("backfill batch complete")
                    paper_id, row = result_row(result)
                    batch[paper_id] = row
                    state["offset"] += 1
                    processed += 1
            finally:
                data = merge_catalog(data, {topic: batch})
                atomic_write_json(config.get("json_readme_path", "docs/llm-agent-arxiv-daily.json"), data)
                # Data reaches disk before advancing the cursor.
                atomic_write_json(state_path, state)
            state["offset"] = 0
            state["topic_index"] += 1
            if state["topic_index"] == len(state["topics"]):
                state["topic_index"] = 0
                start += dt.timedelta(days=1)
                state["day"] = start.isoformat()
            state["complete"] = start > end
            atomic_write_json(state_path, state)
            if processed >= max_papers or time.monotonic() >= deadline:
                break
    except BudgetExceeded:
        pass
    except Exception as exc:
        error = str(exc)
        logging.exception("Backfill stopped; saved cursor will resume")
    finally:
        publish_catalog(config, data)
        state["published"] = bool(state["complete"])
        state["updated_at"] = utc_now()
        state["last_error"] = error
        atomic_write_json(state_path, state)
    step_summary(
        f"Backfill: {processed} records; cursor={state['day']} topic={state['topic_index']} "
        f"offset={state['offset']}; complete={state['complete']}; error={error or 'none'}."
    )
    return 1 if error else 0


def main(argv=None):
    import daily_arxiv

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", default="config.yaml")
    parser.add_argument("--backfill", action="store_true")
    parser.add_argument("--start-date", default="2026-05-15")
    parser.add_argument("--end-date")
    parser.add_argument("--max-papers", type=int, default=1000)
    parser.add_argument("--max-seconds", type=int, default=900)
    args = parser.parse_args(argv)
    if args.max_papers <= 0 or args.max_seconds <= 0:
        parser.error("budgets must be positive")
    config = daily_arxiv.load_config(args.config_path)
    if args.backfill:
        return harvest_backfill(config, start_date=args.start_date, end_date=args.end_date,
                                max_papers=args.max_papers, max_seconds=args.max_seconds)
    return harvest_daily(config, max_seconds=args.max_seconds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
