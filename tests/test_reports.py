import datetime as dt
import json
import logging
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import requests

from scripts import stats_panel, weekly_report


ROOT = Path(__file__).resolve().parents[1]
UTC = dt.timezone.utc
NOW = dt.datetime(2026, 10, 19, 9, tzinfo=UTC)
CATEGORIES = [
    "LLM Agents (Core)",
    "Tool Use & Function Calling",
    "RAG / Memory / Long Context for Agents",
    "Multi-Agent / Collaboration",
    "Web / Search / Browsing Agents",
    "GUI / Computer Use Agents",
    "Multimodal Agents (VLM)",
    "Evaluation / Benchmarks (Agents)",
]


class FrozenDateTime(dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW.astimezone(tz) if tz is not None else NOW.replace(tzinfo=None)


class FrozenDate(dt.date):
    @classmethod
    def today(cls):
        # Deliberately differ from UTC, so local-calendar regressions fail.
        return dt.date(2026, 10, 20)


def paper_row(date="2026-10-18", title="Collected paper"):
    return (
        f"|**{date}**|**{title}**|A et.al.|"
        "[2610.00001](https://arxiv.org/abs/2610.00001)|null|null|\n"
    )


class ReportTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        old_cwd = Path.cwd()
        os.chdir(directory.name)
        self.addCleanup(os.chdir, old_cwd)
        self.enter_patch(patch.dict(os.environ, {}, clear=True))
        log_level = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, log_level)
        self.enter_patch(patch.object(dt, "datetime", FrozenDateTime))
        self.enter_patch(patch.object(dt, "date", FrozenDate))
        self.network = self.enter_patch(
            patch.object(
                requests.sessions.Session,
                "request",
                side_effect=AssertionError("Unexpected outbound network request"),
            )
        )
        Path(".cache").mkdir()
        self.config = {
            "json_readme_path": "papers.json",
            "md_readme_path": "README.md",
            "harvest_state_path": ".cache/harvest.json",
            "weekly_categories": CATEGORIES,
            "freshness_max_days": 3,
            "keywords": {topic: {"filters": [topic]} for topic in CATEGORIES},
            "user_name": "example",
            "repo_name": "papers",
        }
        self.config["keywords"]["LLM (General)"] = {"filters": ["LLM"]}
        self.save_config()
        self.save_json("papers.json", {CATEGORIES[0]: {"2610.00001": paper_row()}})
        stamp = NOW.isoformat()
        self.state = {
            "last_success_at": stamp,
            "last_run": {
                "status": "success",
                "new_papers": 4,
                "finished_at": stamp,
                "failed_topics": [],
            },
            "topics": {
                topic: {"last_success_at": stamp}
                for topic in self.config["keywords"]
            },
        }
        self.save_state()
        Path("README.md").write_text("# Papers\n\n## Updated on old\n", encoding="utf-8")

    def enter_patch(self, patcher):
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def save_json(self, path, value):
        Path(path).write_text(json.dumps(value), encoding="utf-8")

    def save_config(self):
        # JSON is also valid YAML.
        self.save_json("config.yaml", self.config)

    def save_state(self):
        self.save_json(".cache/harvest.json", self.state)

    def run_cli(self, module, *args):
        with patch.object(sys, "argv", [module, *args]):
            try:
                runpy.run_path(str(ROOT / "scripts" / f"{module}.py"), run_name="__main__")
            except SystemExit as exc:
                return exc.code
        return 0

    def test_weekly_uses_configured_categories_and_deduplicates(self):
        row = paper_row(title="Shared paper")
        self.save_json(
            "papers.json",
            {
                CATEGORIES[0]: {"shared": row},
                CATEGORIES[1]: {"shared": row, "tool": paper_row(title="Tool paper")},
                "LLM (General)": {"general": paper_row(title="Not an agent")},
            },
        )
        weekly_report.run_weekly("papers.json")
        report = Path("docs/weekly/index.md").read_text(encoding="utf-8")
        self.assertEqual(report.count("**Shared paper**"), 1)
        self.assertIn("**Tool paper**", report)
        self.assertNotIn("Not an agent", report)

    def test_weekly_deduplicates_before_applying_top_five_limit(self):
        self.save_json(
            "papers.json",
            {
                category: {
                    str(number): paper_row(title=f"Paper {number}")
                    for number in range(7)
                }
                for category in CATEGORIES[:2]
            },
        )
        weekly_report.run_weekly("papers.json")
        report = Path("docs/weekly/index.md").read_text(encoding="utf-8")
        self.assertEqual(report.count("- ["), 5)
        for number in range(2, 7):
            self.assertEqual(report.count(f"**Paper {number}**"), 1)
        self.assertNotIn("**Paper 0**", report)
        self.assertNotIn("**Paper 1**", report)

    def test_weekly_covers_previous_seven_complete_utc_days(self):
        self.save_json(
            "papers.json",
            {
                CATEGORIES[0]: {
                    "before": paper_row("2026-10-11", "Before window"),
                    "start": paper_row("2026-10-12", "Monday included"),
                    "end": paper_row("2026-10-18", "Sunday included"),
                    "today": paper_row("2026-10-19", "Incomplete day excluded"),
                }
            },
        )
        weekly_report.run_weekly("papers.json")
        report = Path("docs/weekly/index.md").read_text(encoding="utf-8")
        self.assertIn("2026-10-12 ~ 2026-10-18", report)
        self.assertIn("Monday included", report)
        self.assertIn("Sunday included", report)
        self.assertNotIn("Before window", report)
        self.assertNotIn("Incomplete day excluded", report)
        self.assertEqual(
            Path("reports/weekly-2026-10-18.md").read_text(encoding="utf-8"), report
        )

    def test_weekly_cli_date_is_inclusive_end_date(self):
        self.assertEqual(
            self.run_cli("weekly_report", "--mode", "weekly", "--date", "2026-10-17"),
            0,
        )
        report = Path("reports/weekly-2026-10-17.md").read_text(encoding="utf-8")
        self.assertIn("2026-10-11 ~ 2026-10-17", report)
        self.assertNotIn("Collected paper", report)

    def test_every_configured_weekly_category_can_contribute(self):
        for category in CATEGORIES:
            with self.subTest(category=category):
                self.save_json("papers.json", {category: {"one": paper_row()}})
                weekly_report.run_weekly("papers.json")
                self.assertIn(
                    "Collected paper",
                    Path("docs/weekly/index.md").read_text(encoding="utf-8"),
                )

    def test_weekly_rejects_stale_harvest_without_overwriting_report(self):
        Path("docs/weekly").mkdir(parents=True)
        Path("docs/weekly/index.md").write_text("previous good report", encoding="utf-8")
        self.state["last_success_at"] = (NOW - dt.timedelta(days=4)).isoformat()
        self.save_state()
        with self.assertRaises(ValueError):
            weekly_report.run_weekly("papers.json")
        self.assertEqual(
            Path("docs/weekly/index.md").read_text(encoding="utf-8"),
            "previous good report",
        )

    def test_date_override_does_not_bypass_freshness(self):
        self.state["last_success_at"] = "2026-09-01T09:00:00Z"
        self.save_state()
        self.assertNotEqual(
            self.run_cli("weekly_report", "--mode", "weekly", "--date", "2026-09-01"),
            0,
        )
        self.assertFalse(Path("docs/weekly/index.md").exists())

    def test_weekly_accepts_exact_freshness_boundary(self):
        self.state["last_success_at"] = (NOW - dt.timedelta(days=3)).isoformat()
        self.save_state()
        weekly_report.run_weekly("papers.json")
        self.assertIn(
            "Collected paper", Path("docs/weekly/index.md").read_text(encoding="utf-8")
        )

    def test_weekly_rejects_partial_or_failed_latest_run(self):
        for status in ("partial", "failed"):
            with self.subTest(status=status):
                self.state["last_run"]["status"] = status
                self.state["last_run"]["failed_topics"] = ["LLM (General)"]
                self.save_state()
                with self.assertRaises(ValueError):
                    weekly_report.run_weekly("papers.json")

    def test_weekly_requires_success_for_all_topics_not_only_digest_topics(self):
        del self.state["topics"]["LLM (General)"]
        self.save_state()
        with self.assertRaises(ValueError):
            weekly_report.run_weekly("papers.json")

    def test_weekly_rejects_stale_topic_success(self):
        self.state["topics"][CATEGORIES[0]]["last_success_at"] = (
            NOW - dt.timedelta(days=4)
        ).isoformat()
        self.save_state()
        with self.assertRaises(ValueError):
            weekly_report.run_weekly("papers.json")

    def test_weekly_rejects_invalid_or_future_success_timestamp(self):
        for value in ("bad-date", "2026-10-19T09:00:00", "2026-10-20T00:00:00Z"):
            with self.subTest(value=value):
                self.state["last_success_at"] = value
                self.save_state()
                with self.assertRaises(ValueError):
                    weekly_report.run_weekly("papers.json")

    def test_weekly_honors_configured_freshness_limit(self):
        self.config["freshness_max_days"] = 1
        self.save_config()
        self.state["last_success_at"] = (NOW - dt.timedelta(days=2)).isoformat()
        self.save_state()
        with self.assertRaises(ValueError):
            weekly_report.run_weekly("papers.json")

    def test_weekly_rejects_unknown_configured_category(self):
        self.config["weekly_categories"] = ["Typo in topic"]
        self.save_config()
        with self.assertRaises(ValueError):
            weekly_report.run_weekly("papers.json")

    def test_json_readers_reject_corrupt_missing_or_non_object_catalogue(self):
        for module in (weekly_report, stats_panel):
            for content in ("{broken", "", "[]", "null", None):
                with self.subTest(module=module.__name__, content=content):
                    path = Path("bad.json")
                    if content is None:
                        path.unlink(missing_ok=True)
                    else:
                        path.write_text(content, encoding="utf-8")
                    with self.assertRaises((ValueError, OSError)):
                        module.load_json_data(str(path))

    def test_config_readers_do_not_swallow_file_or_yaml_errors(self):
        for module in (weekly_report, stats_panel):
            for content in ("keywords: [", "[]", None):
                with self.subTest(module=module.__name__, content=content):
                    path = Path("bad.yaml")
                    if content is None:
                        path.unlink(missing_ok=True)
                    else:
                        path.write_text(content, encoding="utf-8")
                    with self.assertRaises(Exception):
                        module.load_config(str(path))

    def test_invalid_catalogue_rows_are_not_silently_skipped(self):
        for value in ([], {"one": 42}, {"one": "|bad|row|"}, {"one": paper_row("bad-date")}):
            with self.subTest(value=value):
                data = {CATEGORIES[0]: value}
                with self.assertRaises(ValueError):
                    weekly_report.build_papers(data)
                self.save_json("papers.json", data)
                with self.assertRaises(ValueError):
                    stats_panel.main()

    def test_daily_write_errors_propagate(self):
        Path("reports").mkdir()
        for date in ("2026-10-19", "2026-10-20"):
            Path(f"reports/daily-{date}.md").mkdir()
        with self.assertRaises(OSError):
            weekly_report.run_daily("papers.json")

    def test_weekly_index_write_errors_propagate(self):
        Path("docs/weekly/index.md").mkdir(parents=True)
        with self.assertRaises(OSError):
            weekly_report.run_weekly("papers.json")

    def test_cli_core_catalogue_errors_exit_nonzero(self):
        Path("papers.json").write_text("{broken", encoding="utf-8")
        for module, args in (
            ("weekly_report", ["--mode", "weekly"]),
            ("stats_panel", []),
        ):
            with self.subTest(module=module):
                self.assertNotEqual(self.run_cli(module, *args), 0)

    def test_cli_missing_harvest_exits_nonzero(self):
        Path(".cache/harvest.json").unlink()
        self.assertNotEqual(self.run_cli("weekly_report", "--mode", "weekly"), 0)

    def test_cli_report_write_error_exits_nonzero(self):
        Path("docs/weekly/index.md").mkdir(parents=True)
        self.assertNotEqual(self.run_cli("weekly_report", "--mode", "weekly"), 0)

    def test_cli_stats_write_error_exits_nonzero(self):
        Path("README.md").unlink()
        Path("README.md").mkdir()
        with patch.object(stats_panel.requests, "get", side_effect=requests.Timeout):
            self.assertNotEqual(self.run_cli("stats_panel"), 0)

    def test_direct_script_entry_points_work_outside_repo_cwd(self):
        for module in ("weekly_report", "stats_panel"):
            with self.subTest(module=module):
                if module == "stats_panel":
                    # Keep subprocess verification entirely offline.
                    self.config["user_name"] = None
                    self.config["repo_name"] = None
                    self.save_config()
                result = subprocess.run(
                    [sys.executable, str(ROOT / "scripts" / f"{module}.py")],
                    text=True,
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("ModuleNotFoundError", result.stderr)

    def test_stats_today_uses_harvest_new_count_and_unique_total(self):
        self.save_json(
            "papers.json",
            {
                CATEGORIES[0]: {"old": paper_row("2025-01-01")},
                CATEGORIES[1]: {"old": paper_row("2025-01-01")},
            },
        )
        with patch.object(stats_panel, "fetch_stars", return_value="12"):
            stats_panel.main()
        panel = Path("README.md").read_text(encoding="utf-8")
        self.assertIn("+4 papers", panel)
        self.assertIn("**Total**: 1", panel)

    def test_stats_previous_run_is_zero_even_when_paper_date_is_today(self):
        self.save_json(
            "papers.json", {CATEGORIES[0]: {"one": paper_row("2026-10-19")}}
        )
        self.state["last_run"]["finished_at"] = "2026-10-18T23:59:59Z"
        self.save_state()
        with patch.object(stats_panel, "fetch_stars", return_value="12"):
            stats_panel.main()
        self.assertIn("+0 papers", Path("README.md").read_text(encoding="utf-8"))

    def test_stats_partial_run_can_count_papers_actually_collected(self):
        self.state["last_run"]["status"] = "partial"
        self.state["last_run"]["failed_topics"] = [CATEGORIES[1]]
        self.state["last_run"]["new_papers"] = 2
        self.save_state()
        with patch.object(stats_panel, "fetch_stars", return_value="12"):
            stats_panel.main()
        self.assertIn("+2 papers", Path("README.md").read_text(encoding="utf-8"))

    def test_stats_run_date_is_compared_in_utc(self):
        self.state["last_run"]["finished_at"] = "2026-10-20T01:00:00+08:00"
        self.save_state()
        with patch.object(stats_panel, "fetch_stars", return_value="12"):
            stats_panel.main()
        self.assertIn("+4 papers", Path("README.md").read_text(encoding="utf-8"))

    def test_stats_rejects_invalid_harvest_count(self):
        for count in (-1, True, "4", None):
            with self.subTest(count=count):
                self.state["last_run"]["new_papers"] = count
                self.save_state()
                with self.assertRaises(ValueError):
                    stats_panel.main()

    def test_stats_invalid_harvest_timestamp_fails_without_replacing_panel(self):
        original = Path("README.md").read_text(encoding="utf-8")
        self.state["last_run"]["finished_at"] = "not-an-iso-timestamp"
        self.save_state()
        with self.assertRaises(ValueError):
            stats_panel.main()
        self.assertEqual(Path("README.md").read_text(encoding="utf-8"), original)

    def test_stats_star_failure_falls_back_to_cache(self):
        self.save_json(".cache/stats.json", {"last_star_count": 123})
        with patch.object(stats_panel.requests, "get", side_effect=requests.Timeout):
            stats_panel.main()
        self.assertIn("**Stars**: 123", Path("README.md").read_text(encoding="utf-8"))
        self.assertEqual(json.loads(Path(".cache/stats.json").read_text())["last_star_count"], 123)

    def test_stats_successful_star_fetch_updates_cache(self):
        response = requests.Response()
        response.status_code = 200
        response._content = b'{"stargazers_count": 42}'
        with patch.object(stats_panel.requests, "get", return_value=response):
            stats_panel.main()
        self.assertEqual(json.loads(Path(".cache/stats.json").read_text()).get("last_star_count"), 42)
        self.assertIn("**Stars**: 42", Path("README.md").read_text(encoding="utf-8"))

    def test_stats_optional_api_failure_without_cache_is_nonfatal(self):
        with patch.object(stats_panel.requests, "get", side_effect=requests.Timeout):
            stats_panel.main()
        self.assertIn("**Stars**: -", Path("README.md").read_text(encoding="utf-8"))

    def test_stats_api_errors_and_invalid_counts_keep_cached_stars(self):
        for status, content in (
            (403, b'{"message": "rate limited"}'),
            (200, b"not-json"),
            (200, b'{"stargazers_count": -1}'),
            (200, b'{"stargazers_count": true}'),
        ):
            with self.subTest(status=status, content=content):
                self.save_json(".cache/stats.json", {"last_star_count": 5})
                response = requests.Response()
                response.status_code = status
                response._content = content
                with patch.object(stats_panel.requests, "get", return_value=response):
                    stats_panel.main()
                self.assertIn(
                    "**Stars**: 5", Path("README.md").read_text(encoding="utf-8")
                )

    def test_stats_cache_corruption_is_not_silently_discarded(self):
        Path(".cache/stats.json").write_text("{broken", encoding="utf-8")
        with self.assertRaises(ValueError):
            stats_panel.load_stats_cache()

    def test_stats_cache_write_error_propagates(self):
        Path(".cache/stats.json").mkdir()
        with self.assertRaises(OSError):
            stats_panel.save_stats_cache({"last_total": 4})

    def test_stats_cache_atomic_failure_preserves_milestone_guard(self):
        cache = {"last_star_count": 5, "last_milestone_total": 100}
        self.save_json(".cache/stats.json", cache)
        with patch("scripts.paper_store.os.replace", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                stats_panel.save_stats_cache({"last_milestone_total": 200})
        self.assertEqual(json.loads(Path(".cache/stats.json").read_text()), cache)
        self.assertEqual(sorted(path.name for path in Path(".cache").iterdir()), [
            "harvest.json", "stats.json",
        ])

    def test_readme_read_and_write_errors_propagate(self):
        with self.assertRaises(OSError):
            stats_panel.update_readme_panel("missing.md", 1, 2, "3", NOW.date())
        Path("directory.md").mkdir()
        with self.assertRaises(OSError):
            stats_panel.update_readme_panel("directory.md", 1, 2, "3", NOW.date())
        original_open = open

        def fail_writes(path, mode="r", *args, **kwargs):
            if str(path) == "README.md" and "w" in mode:
                raise PermissionError("read-only README")
            return original_open(path, mode, *args, **kwargs)

        with patch("builtins.open", side_effect=fail_writes):
            with self.assertRaises(OSError):
                stats_panel.update_readme_panel("README.md", 1, 2, "3", NOW.date())

    def test_milestone_disabled_even_with_token(self):
        with patch.dict(os.environ, {"GITHUB_TOKEN": "fake-token"}):
            with patch.object(stats_panel.requests, "post") as post:
                stats_panel.handle_milestone(125, NOW.date(), "owner", "repo", "fake-token")
        post.assert_not_called()

    def test_direct_milestone_creation_also_requires_explicit_opt_in(self):
        with patch.object(stats_panel.requests, "post") as post:
            self.assertFalse(
                stats_panel.create_milestone_issue(
                    125, NOW.date(), "owner", "repo", "fake-token"
                )
            )
        post.assert_not_called()

    def test_milestone_opt_in_uses_cache_to_avoid_duplicates(self):
        response = requests.Response()
        response.status_code = 201
        response._content = b'{"html_url": "https://example.invalid/issues/1"}'
        with patch.dict(os.environ, {"ENABLE_MILESTONE_ISSUES": "true"}):
            with patch.object(stats_panel.requests, "post", return_value=response) as post:
                stats_panel.handle_milestone(125, NOW.date(), "owner", "repo", "fake-token")
                stats_panel.handle_milestone(145, NOW.date(), "owner", "repo", "fake-token")
        self.assertEqual(post.call_count, 1)
        self.assertEqual(
            json.loads(Path(".cache/stats.json").read_text())["last_milestone_total"], 100
        )

    def test_readme_panel_replacement_is_idempotent(self):
        stats_panel.update_readme_panel("README.md", 1, 2, "3", NOW.date())
        stats_panel.update_readme_panel("README.md", 4, 5, "6", NOW.date())
        content = Path("README.md").read_text(encoding="utf-8")
        self.assertEqual(content.count(stats_panel.STATS_PANEL_START), 1)
        self.assertIn("+4 papers", content)
        self.assertIn("# Papers", content)


if __name__ == "__main__":
    unittest.main()
