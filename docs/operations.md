# Pipeline Operations

## Scheduled Jobs

All times below are UTC. GitHub schedule delivery can be delayed.

| Workflow | Schedule | Work |
| --- | --- | --- |
| LLM-Agent Arxiv Daily | 01:17, 13:17 daily | Recent metadata; no remote enrichment |
| Update Paper Links | 05:43 daily | At most 40 unique IDs / 10 minutes |
| Recover Historical Papers | 07:47, 19:47 daily | At most 1,000 records / 10 minutes |
| Weekly Agent Digest | Monday 10:23 | Previous seven complete UTC days |
| Publish Pages | After pipeline completion | Deploy current `main` explicitly |

Repository writers share `paper-catalog-main`, with cancellation disabled.
Checkout occurs after the job gets the lock. Publishing uses non-force git
pushes and stops on rebase conflicts rather than discarding edits.

## Data and Recovery

`docs/llm-agent-arxiv-daily.json` is the canonical category-to-paper catalogue.
Existing rows and links from the historical web copy are merged before both
JSON views are published. Writes replace files atomically. Invalid JSON is an
error, not an empty catalogue. Metadata is saved before a backfill cursor moves.

The current-data job covers a three-day overlap and extends it to the last
successful topic run after an outage. It enumerates results instead of silently
keeping only the newest 32 papers. It records successful and failed topics in
`.cache/harvest.json`; partial data is saved while the workflow still reports
failure. The stats panel counts newly collected IDs, not paper update dates.

Historical recovery covers **2026-05-15 through 2026-10-20**. The cursor in
`.cache/backfill.json` records the date, category and result offset. Each run
continues that cursor; a completed recovery is a cheap no-op. To run a different
range, configure a different `backfill_state_path` rather than overwriting the
saved cursor. Automatic scheduling can be disabled once `complete` is true.

Code enrichment shares results across categories and views. Positive matches
are cached; absent matches are retried after 30 days, transient failures after
15 minutes. A 403/429 stops the data source and saves its cooldown. No Papers
With Code redirect calls or repeated title-only searches are made. Existing
code/model links are not removed. Optional enrichment failure never blocks the
metadata job.

Weekly categories are explicit in `config.yaml`, with duplicate arXiv IDs
collapsed before selecting five recent papers. The weekly job rejects stale or
partial latest harvest state. It does not pretend that a failed source means
"no papers".

## Manual Runs

Use GitHub Actions **Run workflow** on the relevant workflow. No personal access
token secret is needed: the Python GitHub API calls receive the workflow's
short-lived `GITHUB_TOKEN`. Milestone issue creation is disabled by default;
enabling it also requires explicitly granting `issues: write`.

For local execution:

```bash
python -m pip install -r requirements.lock
python -m unittest discover -s tests -v
python -m scripts.harvest --max-seconds 900
python -m scripts.stats_panel
python -m scripts.enrich_links --max-papers 40 --max-seconds 600
python -m scripts.harvest --backfill --start-date 2026-05-15 --end-date 2026-10-20 --max-papers 1000 --max-seconds 600
python -m scripts.weekly_report --mode weekly
```

Without a GitHub token the local optional enrichment job explicitly skips.
Local commands update files but never commit or push automatically.

## Failures and Maintenance

Inspect the failed step, job summary and `pipeline-*` diagnostic artifact.
Artifacts include public catalogue data and checkpoints, not environment dumps
or tokens. Completed progress is committed even after a partial harvest failure.
If git push fails, recover the output and state from the same artifact together.
Configure GitHub Actions failure notifications for the repository owner.

Dependency versions are locked, including transitives. Update `requirements.txt`
and regenerate `requirements.lock` with:

```bash
uv pip compile --python-version 3.12 requirements.txt -o requirements.lock
```

Run the offline regression suite before merging dependency updates. Actions are
pinned to reviewed commit SHAs; Dependabot proposes updates rather than silently
changing the runtime. Ubuntu 24.04 and Python 3.12 are explicit.
