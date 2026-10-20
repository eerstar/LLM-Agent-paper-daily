import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import daily_arxiv


class HarvestRegressionTests(unittest.TestCase):
    def test_legacy_entry_uses_client_and_does_not_enrich(self):
        result = SimpleNamespace(
            get_short_id=lambda: "2610.00001v1",
            title="A | paper",
            authors=["Author"],
            updated=dt.datetime(2026, 10, 20),
            published=dt.datetime(2026, 10, 19),
            summary="Code: https://github.com/example/paper",
            comment="",
            links=[],
        )
        with patch("arxiv.Client.results", return_value=iter([result])), patch(
            "daily_arxiv.fetch_pwc_official_repo",
            side_effect=AssertionError("optional lookup entered harvest"),
        ):
            data, _ = daily_arxiv.get_daily_papers("Agent", "agent", 1)
        self.assertIn("2610.00001", data["Agent"])
        self.assertIn("2026-10-19", data["Agent"]["2610.00001"])
        self.assertIn("A &#124; paper", data["Agent"]["2610.00001"])

    def test_merge_retains_historical_links(self):
        from scripts.paper_store import merge_catalog

        old = {"Agent": {"1": "|**2026-01-01**|**Old**|A|[1](https://arxiv.org/abs/1)|**[link](https://github.com/a/b)**|null|\n"}}
        fresh = {"Agent": {"1": "|**2026-01-02**|**New**|A|[1](https://arxiv.org/abs/1)|null|null|\n"}}
        merged = merge_catalog(old, fresh)
        self.assertIn("New", merged["Agent"]["1"])
        self.assertIn("github.com/a/b", merged["Agent"]["1"])

    def test_corrupt_existing_json_is_not_empty_fallback(self):
        from scripts.paper_store import read_json

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            path.write_text("{broken")
            with self.assertRaises(ValueError):
                read_json(path, {})

    def test_atomic_json_failure_keeps_original(self):
        from scripts.paper_store import atomic_write_json

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            path.write_text('{"saved":1}')
            with self.assertRaises(TypeError):
                atomic_write_json(path, {"bad": object()})
            self.assertEqual(json.loads(path.read_text()), {"saved": 1})

    def test_daily_partial_failure_keeps_success_and_reports_failure(self):
        from scripts.harvest import harvest_daily

        result = SimpleNamespace(
            get_short_id=lambda: "2610.00001v1", title="Paper", authors=["A"],
            published=dt.datetime(2026, 10, 19), updated=dt.datetime(2026, 10, 19),
            summary="", comment="", links=[],
        )

        class Client:
            def results(self, search, offset=0):
                if "bad" in search.query:
                    raise RuntimeError("source unavailable")
                return iter([result])

        config = {"kv": {"good": "good", "bad": "bad"}, "lookback_days": 3}
        with tempfile.TemporaryDirectory() as directory:
            config["harvest_state_path"] = str(Path(directory) / "state.json")
            config["json_readme_path"] = str(Path(directory) / "papers.json")
            with patch("scripts.harvest.publish_catalog") as publish:
                code = harvest_daily(config, Client(), max_seconds=60)
            self.assertEqual(code, 1)
            self.assertIn("2610.00001", publish.call_args.args[1]["good"])
            state = json.loads(Path(config["harvest_state_path"]).read_text())
            self.assertEqual(state["last_run"]["failed_topics"], ["bad"])
            self.assertNotIn("last_success_at", state)

    def test_backfill_resumes_offset_without_skipping_topic(self):
        from scripts.harvest import harvest_backfill

        def item(number):
            return SimpleNamespace(
                get_short_id=lambda: f"2605.{number:05d}v1", title="Paper",
                authors=["A"], published=dt.datetime(2026, 5, 15),
                updated=dt.datetime(2026, 5, 15), summary="", comment="", links=[],
            )

        class Client:
            def __init__(self):
                self.offsets = []

            def results(self, search, offset=0):
                self.offsets.append(offset)
                return iter([item(i) for i in range(offset, 3)])

        with tempfile.TemporaryDirectory() as directory:
            config = {
                "kv": {"Agent": "agent"},
                "json_readme_path": str(Path(directory) / "papers.json"),
                "backfill_state_path": str(Path(directory) / "backfill.json"),
            }
            client = Client()
            with patch("scripts.harvest.publish_catalog"):
                self.assertEqual(harvest_backfill(config, client, "2026-05-15", "2026-05-15", max_papers=2), 0)
                self.assertEqual(harvest_backfill(config, client, "2026-05-15", "2026-05-15", max_papers=2), 0)
            self.assertEqual(client.offsets, [0, 2])
            self.assertTrue(json.loads(Path(config["backfill_state_path"]).read_text())["complete"])

    def test_daily_uses_recovery_window_after_short_outage(self):
        from scripts.harvest import harvest_daily

        now = dt.datetime.now(dt.timezone.utc)
        queries = []

        class Client:
            def results(self, search, offset=0):
                queries.append(search.query)
                return iter([])

        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            previous = now - dt.timedelta(days=7)
            state_path.write_text(json.dumps({
                "topics": {"Agent": {"last_success_at": previous.isoformat()}},
            }))
            config = {
                "kv": {"Agent": "agent"}, "lookback_days": 3,
                "harvest_state_path": str(state_path),
                "json_readme_path": str(Path(directory) / "papers.json"),
            }
            with patch("scripts.harvest.publish_catalog"):
                self.assertEqual(harvest_daily(config, Client()), 0)
            self.assertIn(f"{previous.date():%Y%m%d}0000", queries[0])


class WorkflowTests(unittest.TestCase):
    def test_shared_runner_preserves_failures_and_artifacts(self):
        import yaml
        path = Path(__file__).resolve().parents[1] / ".github/workflows/paper-pipeline.yml"
        workflow = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
        job = workflow["jobs"]["update"]
        self.assertEqual(job.get("defaults", {}).get("run", {}).get("shell"), "bash")
        artifacts = next(step for step in job["steps"] if step.get("name") == "Preserve diagnostics")
        self.assertIn("docs/llm-agent-arxiv-daily.json", artifacts["with"]["path"])


class ReviewRegressionTests(unittest.TestCase):
    def test_invalid_success_feed_is_rejected(self):
        from scripts.harvest import make_client
        import arxiv
        import requests
        for payload in (
            b"<html>maintenance</html>", b"not xml",
            b'<feed xmlns="http://www.w3.org/2005/Atom" xmlns:os="http://a9.com/-/spec/opensearch/1.1/"><os:totalResults>20</os:totalResults><os:startIndex>0</os:startIndex></feed>',
        ):
            client = make_client()
            client.num_retries = 0
            response = requests.Response()
            response.status_code = 200
            response._content = payload
            with patch("requests.Session.request", return_value=response):
                with self.assertRaises(ValueError):
                    list(client.results(arxiv.Search(query="agent", max_results=1)))

    def test_legacy_code_and_conflicting_links_are_retained(self):
        from scripts.paper_store import merge_catalog
        old = {"Agent": {"1": "|**2026-01-01**|**Paper**|A|[1](https://arxiv.org/abs/1)|**[link](https://github.com/a/old)**|\n"}}
        fresh = {"Agent": {"1": "|**2026-01-02**|**Paper**|A|[1](https://arxiv.org/abs/1)|**[link](https://github.com/a/new)**|null|\n"}}
        row = merge_catalog(old, fresh)["Agent"]["1"]
        self.assertIn("github.com/a/old", row)
        self.assertIn("github.com/a/new", row)
        self.assertEqual(row.count("github.com/a/new"), 1)

    def test_daily_cap_saves_and_resumes_frozen_window(self):
        from scripts.harvest import harvest_daily
        offsets = []
        result = lambda i: SimpleNamespace(
            get_short_id=lambda: f"2610.{i:05d}v1", title="Paper", authors=["A"],
            published=dt.datetime(2026, 10, 19), updated=dt.datetime(2026, 10, 19),
            summary="", comment="", links=[],
        )
        class Client:
            def results(self, search, offset=0):
                offsets.append(offset)
                return iter(result(i) for i in range(offset, 3))
        with tempfile.TemporaryDirectory() as directory:
            config = {
                "kv": {"Agent": "agent"}, "harvest_state_path": str(Path(directory) / "state.json"),
                "json_readme_path": str(Path(directory) / "data.json"),
            }
            with patch("scripts.harvest.publish_catalog"):
                self.assertEqual(harvest_daily(config, Client(), max_per_topic=2), 1)
                pending = json.loads(Path(config["harvest_state_path"]).read_text())["topics"]["Agent"]["pending"]
                self.assertEqual(pending["offset"], 2)
                self.assertEqual(harvest_daily(config, Client(), max_per_topic=2), 0)
            self.assertEqual(offsets, [0, 2])
            self.assertEqual(len(json.loads(Path(config["json_readme_path"]).read_text())["Agent"]), 3)

    def test_first_failure_preserves_window_start(self):
        from scripts.harvest import harvest_daily
        class Client:
            def results(self, search, offset=0):
                raise RuntimeError("offline")
        with tempfile.TemporaryDirectory() as directory:
            config = {
                "kv": {"Agent": "agent"}, "harvest_state_path": str(Path(directory) / "state.json"),
                "json_readme_path": str(Path(directory) / "data.json"),
            }
            with patch("scripts.harvest.publish_catalog"):
                harvest_daily(config, Client())
            topic = json.loads(Path(config["harvest_state_path"]).read_text())["topics"]["Agent"]
            self.assertIn("start", topic["pending"])
            self.assertEqual(topic["pending"]["offset"], 0)

    def test_completed_backfill_retries_publication(self):
        from scripts.harvest import harvest_backfill
        class Client:
            def results(self, search, offset=0):
                return iter([])
        with tempfile.TemporaryDirectory() as directory:
            config = {
                "kv": {"Agent": "agent"}, "backfill_state_path": str(Path(directory) / "state.json"),
                "json_readme_path": str(Path(directory) / "data.json"),
            }
            with patch("scripts.harvest.publish_catalog", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    harvest_backfill(config, Client(), "2026-05-15", "2026-05-15")
            with patch("scripts.harvest.publish_catalog") as publish:
                harvest_backfill(config, Client(), "2026-05-15", "2026-05-15")
                publish.assert_called_once()


if __name__ == "__main__":
    unittest.main()
