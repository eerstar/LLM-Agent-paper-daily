from email.utils import formatdate
import importlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from io import StringIO
from unittest.mock import patch

import requests

from scripts.paper_store import atomic_write_json, publish_catalog, read_json


TOKEN = "test-token-never-log"
PAPER_ID = "2610.00001"
TITLE = "MemoryCraft: Persistent Episodic Planning"
REPO = "https://github.com/research/MemoryCraft"


def row(paper_id=PAPER_ID, code="null", model="null", title=TITLE):
    return (
        f"|**2026-10-19**|**{title}**|Author et.al.|"
        f"[{paper_id}](https://arxiv.org/abs/{paper_id})|{code}|{model}|\n"
    )


def repository(url=REPO, description=None, **extra):
    full_name = url.removeprefix("https://github.com/")
    return {
        "html_url": url, "full_name": full_name,
        "name": full_name.split("/")[-1],
        "description": description or f"Official code for {TITLE} ({PAPER_ID})",
        "fork": False, "archived": False, "stargazers_count": 0,
        **extra,
    }


class FakeClock:
    def __init__(self):
        self.epoch = 1_800_000_000.0
        self.elapsed = 0.0
        self.sleeps = []

    def time(self):
        return self.epoch + self.elapsed

    def monotonic(self):
        return self.elapsed

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.elapsed += seconds


class Response:
    def __init__(self, items=(), status=200, headers=None, **payload):
        self.status_code = status
        self.headers = requests.structures.CaseInsensitiveDict(headers or {})
        self.payload = {
            "items": list(items), "total_count": len(items),
            "incomplete_results": False, **payload,
        }

    def json(self):
        return self.payload

    def close(self):
        pass


class FakeSession:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if not self.responses:
            raise AssertionError("Unexpected additional network call")
        response = self.responses.pop(0)
        if callable(response):
            response = response()
        if isinstance(response, BaseException):
            raise response
        return response

    def close(self):
        pass


class EnrichmentTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            importlib.util.find_spec("scripts.enrich_links"),
            "The bounded enrichment module has not been implemented",
        )
        self.module = importlib.import_module("scripts.enrich_links")
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = {
            "json_readme_path": str(self.root / "papers.json"),
            "json_gitpage_path": str(self.root / "web.json"),
            "json_wechat_path": str(self.root / "wechat.json"),
            "md_readme_path": str(self.root / "README.md"),
            "md_gitpage_path": str(self.root / "index.md"),
            "md_wechat_path": str(self.root / "wechat.md"),
            "publish_readme": True, "publish_gitpage": True,
            "publish_wechat": True,
            "enrichment_state_path": str(self.root / ".cache/link-enrichment.json"),
        }
        self.clock = FakeClock()
        self.env = patch.dict(os.environ, {"GITHUB_TOKEN": TOKEN}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def seed(self, data=None):
        data = data if data is not None else {"Agent": {PAPER_ID: row()}}
        atomic_write_json(self.config["json_readme_path"], data)
        return data

    def state(self):
        return read_json(self.config["enrichment_state_path"], {})

    def data(self):
        return read_json(self.config["json_readme_path"], {})

    def run_enrichment(self, session, **kwargs):
        with patch.object(self.module, "publish_catalog", wraps=publish_catalog) as publish:
            result = self.module.enrich_links(
                self.config, session=session, clock=self.clock, **kwargs,
            )
        self.assertLessEqual(publish.call_count, 1)
        return result

    def test_deduplicates_ids_updates_all_views_and_preserves_model(self):
        model = "**[link](https://huggingface.co/research/memory)**"
        self.seed({"Agent": {PAPER_ID: row(model=model)},
                   "Memory": {PAPER_ID: row(model=model)}})
        session = FakeSession(Response([repository()]))
        self.assertEqual(self.run_enrichment(session), 0)
        self.assertEqual(len(session.calls), 1)
        data = self.data()
        for papers in data.values():
            self.assertIn(REPO, papers[PAPER_ID])
            self.assertIn(model, papers[PAPER_ID])
        for key in ("json_gitpage_path", "json_wechat_path"):
            self.assertEqual(read_json(self.config[key], {}), data)
        for key in ("md_readme_path", "md_gitpage_path", "md_wechat_path"):
            self.assertIn(REPO, Path(self.config[key]).read_text())
        entry = self.state()["papers"][PAPER_ID]
        self.assertEqual(entry["result"], REPO)
        self.assertEqual(entry["last_attempt"], self.clock.time())

    def test_any_existing_code_is_preserved_and_shared_without_lookup(self):
        original = "**[code](https://example.org/project)**"
        other = "**[link](https://github.com/other/project)**"
        before = self.seed({
            "Agent": {PAPER_ID: row(code=original)},
            "Memory": {PAPER_ID: row()},
            "Other": {PAPER_ID: row(code=other)},
        })
        self.run_enrichment(FakeSession())
        self.assertEqual(self.data()["Agent"], before["Agent"])
        self.assertEqual(self.data()["Other"], before["Other"])
        self.assertIn(original, self.data()["Memory"][PAPER_ID])

    def test_web_only_existing_link_is_reused(self):
        self.seed()
        atomic_write_json(self.config["json_gitpage_path"], {
            "Agent": {PAPER_ID: row(code=f"**[link]({REPO})**")},
        })
        self.run_enrichment(FakeSession())
        self.assertIn(REPO, self.data()["Agent"][PAPER_ID])

    def test_limit_counts_unique_attempts_and_next_run_does_not_restart(self):
        ids = ["2610.00001", "2610.00002", "2610.00003"]
        self.seed({"Agent": {key: row(key) for key in ids},
                   "Other": {key: row(key) for key in ids}})
        first, second = FakeSession(Response()), FakeSession(Response())
        self.run_enrichment(first, max_papers=1)
        self.run_enrichment(second, max_papers=1)
        first_query = first.calls[0][1]["params"]["q"]
        second_query = second.calls[0][1]["params"]["q"]
        self.assertNotEqual(first_query, second_query)
        self.assertEqual(len(self.state()["papers"]), 2)
        self.assertEqual(self.state()["last_run"]["attempted"], 1)

    def test_negative_results_expire_after_thirty_days(self):
        self.seed()
        self.run_enrichment(FakeSession(Response()))
        entry = self.state()["papers"][PAPER_ID]
        self.assertEqual(entry["status"], "not_found")
        self.assertEqual(entry["retry_after"] - entry["last_attempt"], 30 * 86400)
        self.clock.sleep(30 * 86400 - 1)
        self.run_enrichment(FakeSession())
        self.clock.sleep(2)
        again = FakeSession(Response())
        self.run_enrichment(again)
        self.assertEqual(len(again.calls), 1)

    def test_unattempted_ids_take_priority_over_expired_errors(self):
        self.seed()
        self.run_enrichment(FakeSession(requests.Timeout("private details")))
        self.clock.sleep(3600)
        data = self.data()
        data["Agent"]["2610.00002"] = row("2610.00002")
        atomic_write_json(self.config["json_readme_path"], data)
        session = FakeSession(Response())
        self.run_enrichment(session, max_papers=1)
        self.assertIn("2610.00002", session.calls[0][1]["params"]["q"])

    def test_transient_errors_have_short_ttl_no_immediate_retries_or_secrets(self):
        self.seed()
        session = FakeSession(requests.Timeout(f"headers contain {TOKEN}"))
        with self.assertLogs(self.module.LOG, level="INFO") as logs:
            self.assertEqual(self.run_enrichment(session), 0)
        self.assertEqual(len(session.calls), 1)
        self.assertNotIn(TOKEN, "\n".join(logs.output))
        self.assertNotIn(TOKEN, json.dumps(self.state()))
        entry = self.state()["papers"][PAPER_ID]
        self.assertEqual(entry["status"], "transient_error")
        self.assertGreater(entry["retry_after"], entry["last_attempt"])
        self.assertLessEqual(entry["retry_after"] - entry["last_attempt"], 3600)
        self.run_enrichment(FakeSession())
        self.clock.sleep(3601)
        self.run_enrichment(FakeSession(Response()))

    def test_403_and_429_stop_run_and_persist_global_cooldown(self):
        for status in (403, 429):
            with self.subTest(status=status):
                Path(self.config["enrichment_state_path"]).unlink(missing_ok=True)
                self.seed({"Agent": {PAPER_ID: row(), "2610.00002": row("2610.00002")}})
                now = self.clock.time()
                session = FakeSession(Response(status=status, headers={
                    "retry-after": "120", "x-ratelimit-reset": str(now + 300),
                }))
                self.assertEqual(self.run_enrichment(session), 0)
                state = self.state()
                self.assertEqual(len(session.calls), 1)
                self.assertGreaterEqual(state["retry_after"], now + 300)
                self.assertEqual(state["last_run"]["status"], "rate_limited")
                self.assertEqual(state["papers"][PAPER_ID]["status"], "rate_limited")
                self.run_enrichment(FakeSession())

    def test_retry_after_http_date_and_no_header_fallback(self):
        for headers in ({"Retry-After": formatdate(self.clock.time() + 400, usegmt=True)}, {}):
            with self.subTest(headers=headers):
                Path(self.config["enrichment_state_path"]).unlink(missing_ok=True)
                self.seed()
                now = self.clock.time()
                self.run_enrichment(FakeSession(Response(status=429, headers=headers)))
                self.assertGreaterEqual(self.state()["retry_after"], now + (400 if headers else 60))

    def test_success_with_exhausted_quota_stops_before_another_request(self):
        self.seed({"Agent": {PAPER_ID: row(), "2610.00002": row("2610.00002")}})
        session = FakeSession(Response([repository()], headers={
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(self.clock.time() + 100),
        }))
        self.assertEqual(self.run_enrichment(session), 0)
        self.assertEqual(len(session.calls), 1)
        self.assertIn(REPO, self.data()["Agent"][PAPER_ID])
        self.assertGreaterEqual(self.state()["retry_after"], self.clock.time() + 100)

    def test_invalid_token_stops_without_trying_every_paper(self):
        self.seed({"Agent": {PAPER_ID: row(), "2610.00002": row("2610.00002")}})
        session = FakeSession(Response(status=401))
        self.assertEqual(self.run_enrichment(session), 0)
        self.assertEqual(len(session.calls), 1)
        self.assertEqual(self.state()["last_run"]["status"], "unauthorized")

    def test_no_token_skips_without_loading_catalogue_or_network(self):
        os.environ.pop("GITHUB_TOKEN")
        summary = self.root / "summary.md"
        os.environ["GITHUB_STEP_SUMMARY"] = str(summary)
        with patch.object(self.module, "load_catalog", side_effect=AssertionError("loaded")), \
                self.assertLogs(self.module.LOG, level="INFO") as logs:
            self.assertEqual(self.run_enrichment(FakeSession()), 0)
        self.assertIn("GITHUB_TOKEN", summary.read_text())
        self.assertIn("skip", "\n".join(logs.output).lower())
        self.assertFalse(Path(self.config["enrichment_state_path"]).exists())

    def test_gh_token_fallback_and_authenticated_id_query_without_star_sort(self):
        os.environ.update({"GITHUB_TOKEN": " ", "GH_TOKEN": TOKEN})
        self.seed()
        session = FakeSession(Response())
        self.run_enrichment(session)
        url, options = session.calls[0]
        self.assertEqual(url, "https://api.github.com/search/repositories")
        self.assertIn(f'"{PAPER_ID}"', options["params"]["q"])
        self.assertIn("in:readme,description", options["params"]["q"])
        self.assertNotIn("sort", options["params"])
        self.assertNotIn(TITLE, options["params"]["q"])
        self.assertEqual(options["headers"]["Authorization"], f"Bearer {TOKEN}")
        self.assertIs(options["allow_redirects"], False)

    def test_requests_are_paced_and_timeout_fits_remaining_budget(self):
        self.seed({"Agent": {PAPER_ID: row(), "2610.00002": row("2610.00002")}})
        session = FakeSession(Response(), Response())
        self.run_enrichment(session, max_seconds=20)
        self.assertEqual(len(session.calls), 2)
        self.assertGreaterEqual(sum(self.clock.sleeps), 2)
        for _, options in session.calls:
            self.assertGreater(options["timeout"].total, 0)
            self.assertLessEqual(options["timeout"].total, 20)

    def test_deadline_does_not_sleep_past_budget_and_publishes_progress(self):
        self.seed({"Agent": {PAPER_ID: row(), "2610.00002": row("2610.00002")}})
        session = FakeSession(Response([repository()]))
        self.assertEqual(self.run_enrichment(session, max_seconds=1), 0)
        self.assertLessEqual(self.clock.elapsed, 1)
        self.assertEqual(self.state()["last_run"]["status"], "deadline")
        self.assertEqual(len(self.state()["papers"]), 1)
        self.assertIn(REPO, self.data()["Agent"][PAPER_ID])
        self.assertLessEqual(session.calls[0][1]["timeout"].total, 1)

    def test_keyboard_interrupt_checkpoints_completed_and_inflight_ids(self):
        self.seed({"Agent": {PAPER_ID: row(), "2610.00002": row("2610.00002")}})

        def interrupt():
            saved = self.state()["papers"]
            self.assertEqual(saved[PAPER_ID]["result"], REPO)
            self.assertEqual(saved["2610.00002"]["status"], "in_progress")
            raise KeyboardInterrupt

        self.assertEqual(self.run_enrichment(FakeSession(Response([repository()]), interrupt)), 0)
        self.assertIn(REPO, self.data()["Agent"][PAPER_ID])
        self.assertEqual(self.state()["last_run"]["status"], "interrupted")
        self.run_enrichment(FakeSession())

    def test_positive_cache_recovers_publication_failure_without_requery(self):
        self.seed()
        with patch.object(self.module, "publish_catalog", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.module.enrich_links(
                    self.config, session=FakeSession(Response([repository()])),
                    clock=self.clock,
                )
        self.assertEqual(self.state()["papers"][PAPER_ID]["result"], REPO)
        self.assertEqual(self.state()["last_run"]["status"], "publication_error")
        self.run_enrichment(FakeSession())
        self.assertIn(REPO, self.data()["Agent"][PAPER_ID])

    def test_truncated_search_page_is_deferred_instead_of_choosing_one_match(self):
        self.seed()
        self.run_enrichment(FakeSession(Response([repository()], total_count=20)))
        entry = self.state()["papers"][PAPER_ID]
        self.assertEqual(entry["status"], "transient_error")
        self.assertNotIn(REPO, self.data()["Agent"][PAPER_ID])

    def test_legacy_id_salvage_preserves_model_column_without_treating_it_as_code(self):
        paper_id = "cs/9901001"
        model = "**[link](https://huggingface.co/research/memory)**"
        self.seed({"Agent": {paper_id: row(paper_id, model=model)}})
        session = FakeSession(Response([repository()]))
        self.run_enrichment(session)
        self.assertEqual(len(session.calls), 1)
        self.assertIn(f"**[link]({REPO})**|{model}|", self.data()["Agent"][paper_id])

    def test_salvaged_existing_unbolded_code_is_never_overwritten(self):
        paper_id = PAPER_ID + "v2"
        original = "[source](https://example.org/project)"
        data = self.seed({"Agent": {paper_id: row(paper_id, code=original)}})
        self.run_enrichment(FakeSession())
        self.assertEqual(self.data(), data)

    def test_sigterm_during_publication_finishes_atomic_views(self):
        self.seed()
        config_path = self.root / "config.yaml"
        config_path.write_text(json.dumps(self.config), encoding="utf-8")

        def publish_then_signal(config, data):
            publish_catalog(config, data)
            os.kill(os.getpid(), signal.SIGTERM)

        original = signal.getsignal(signal.SIGTERM)
        with patch.object(self.module, "make_session", return_value=FakeSession(Response())), \
                patch.object(self.module, "publish_catalog", side_effect=publish_then_signal):
            try:
                result = self.module.main(["--config-path", str(config_path)])
            except KeyboardInterrupt:
                self.fail("SIGTERM interrupted final checkpoint/publication")
            self.assertEqual(result, 0)
        self.assertEqual(signal.getsignal(signal.SIGTERM), original)

    def test_no_network_or_restart_when_cache_schema_is_invalid(self):
        self.seed()
        path = Path(self.config["enrichment_state_path"])
        invalid = [
            {"version": 2, "papers": {}},
            {"version": 1, "papers": {}, "retry_after": "tomorrow"},
            {"version": 1, "papers": {PAPER_ID: {
                "status": "not_found", "last_attempt": self.clock.time(),
                "retry_after": self.clock.time() + 30 * 86400, "result": REPO,
            }}},
        ]
        for state in invalid:
            with self.subTest(state=state):
                atomic_write_json(path, state)
                with self.assertRaises(ValueError):
                    self.run_enrichment(FakeSession())
                self.assertEqual(read_json(path, {}), state)

    def test_incomplete_malformed_and_error_responses_are_not_negative_cached(self):
        responses = [
            Response(incomplete_results=True), Response(),
            Response(status=502), Response(status=302),
        ]
        responses[1].payload = {"message": "unexpected payload"}
        for response in responses:
            with self.subTest(response=response.payload, status=response.status_code):
                Path(self.config["enrichment_state_path"]).unlink(missing_ok=True)
                self.seed()
                self.run_enrichment(FakeSession(response))
                entry = self.state()["papers"][PAPER_ID]
                self.assertEqual(entry["status"], "transient_error")
                self.assertLessEqual(entry["retry_after"] - entry["last_attempt"], 3600)

    def test_corrupt_cache_fails_closed_without_overwriting_it(self):
        self.seed()
        path = Path(self.config["enrichment_state_path"])
        path.parent.mkdir()
        path.write_text("{corrupt", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.run_enrichment(FakeSession())
        self.assertEqual(path.read_text(), "{corrupt")

    def test_version_suffix_is_deduplicated_and_invalid_rows_preserved(self):
        original = self.seed({
            "Agent": {PAPER_ID: row(), "invalid-key": "not a markdown row"},
            "Other": {PAPER_ID + "v2": row()},
        })
        session = FakeSession(Response([repository()]))
        self.run_enrichment(session)
        self.assertEqual(len(session.calls), 1)
        self.assertIn(REPO, self.data()["Other"][PAPER_ID + "v2"])
        self.assertEqual(self.data()["Agent"]["invalid-key"], original["Agent"]["invalid-key"])

    def test_progress_and_summary_include_counts_but_not_credentials(self):
        self.seed()
        path = self.root / "summary.md"
        os.environ["GITHUB_STEP_SUMMARY"] = str(path)
        with self.assertLogs(self.module.LOG, level="INFO") as logs:
            self.run_enrichment(FakeSession(Response([repository()])))
        self.assertIn(PAPER_ID, "\n".join(logs.output))
        text = path.read_text()
        self.assertIn("attempted=1", text)
        self.assertIn("updated_rows=1", text)
        self.assertNotIn(TOKEN, text)

    def test_session_has_no_transport_retries(self):
        with self.module.make_session() as session:
            self.assertEqual(session.get_adapter("https://").max_retries.total, 0)

    def test_cli_forwards_budgets_and_config(self):
        config_path = self.root / "config.yaml"
        config_path.write_text("publish_readme: false\nkeywords: {}\n", encoding="utf-8")
        with patch.object(self.module, "enrich_links", return_value=0) as enrich:
            self.assertEqual(self.module.main([
                "--config-path", str(config_path),
                "--max-papers", "40", "--max-seconds", "600",
            ]), 0)
        self.assertEqual(enrich.call_args.args[0]["publish_readme"], False)
        self.assertEqual(enrich.call_args.kwargs["max_papers"], 40)
        self.assertEqual(enrich.call_args.kwargs["max_seconds"], 600)

    def test_cli_rejects_nonpositive_and_nonfinite_budgets(self):
        for args in (["--max-papers", "0"], ["--max-seconds", "-1"],
                     ["--max-seconds", "nan"], ["--max-seconds", "inf"]):
            with self.subTest(args=args), patch("sys.stderr", new_callable=StringIO):
                with self.assertRaises(SystemExit) as result:
                    self.module.main(args)
                self.assertEqual(result.exception.code, 2)

    def test_sigterm_uses_safe_interruption_and_restores_handler(self):
        self.seed()
        original = signal.getsignal(signal.SIGTERM)
        session = FakeSession(lambda: os.kill(os.getpid(), signal.SIGTERM))
        config_path = self.root / "config.yaml"
        config_path.write_text(json.dumps(self.config), encoding="utf-8")
        with patch.object(self.module, "make_session", return_value=session):
            self.assertEqual(self.module.main(["--config-path", str(config_path)]), 0)
        self.assertEqual(self.state()["last_run"]["status"], "interrupted")
        self.assertEqual(signal.getsignal(signal.SIGTERM), original)

    def test_matching_rejects_popular_unrelated_and_collection_repositories(self):
        popular = repository(
            "https://github.com/community/awesome-agents",
            f"Reading list including {TITLE}", stargazers_count=100_000,
        )
        unrelated = repository(
            "https://github.com/vendor/agent-framework",
            "General purpose application framework", stargazers_count=50_000,
        )
        candidates = [popular, unrelated, repository()]
        self.assertEqual(self.module.select_repository(candidates, TITLE), REPO)
        self.assertIsNone(self.module.select_repository(candidates[:2], TITLE))

    def test_matching_rejects_ambiguous_forks_unsafe_urls_and_generic_overlap(self):
        cases = [
            [repository(), repository("https://github.com/another/MemoryCraft")],
            [repository(fork=True)],
            [repository(archived=True)],
            [repository("https://evil.example/research/MemoryCraft")],
            [repository("https://github.com/research/MemoryCraft/issues")],
            [repository("https://github.com/vendor/planning", "Planning agents")],
        ]
        for candidates in cases:
            with self.subTest(candidates=candidates):
                self.assertIsNone(self.module.select_repository(candidates, TITLE))


if __name__ == "__main__":
    unittest.main()
