# Scheduled Pipeline Repair

## Design

Keep existing paper IDs, categories, JSON schemas and published URLs. Harvest
metadata independently of optional code/model enrichment. Use the README JSON as
the canonical catalogue, merging historical links from the web copy before
writing both views. Never replace a valid catalogue with an empty fallback.

The user approved implementation, push and remote workflow reconfiguration.
No credentials are written into the repository. Historical recovery is bounded
and resumable; completing its entire date range is separate from deploying it.

## Implementation

- [x] Add failing regression tests for removed arxiv API, optional lookup
  isolation, retained links, atomic writes, partial harvests and restart cursors.
- [x] Implement `scripts/paper_store.py` for strict loading, link preservation
  and atomic publication; implement `scripts/harvest.py` for current data and
  bounded historical backfill using one rate-limited arxiv Client.
- [x] Implement `scripts/enrich_links.py` with unique-ID caching, bounded batches,
  explicit token use and rate-limit circuit breaking.
- [x] Fix `scripts/weekly_report.py` category selection/freshness and
  `scripts/stats_panel.py` newly collected counts and optional API failure.
- [x] Lock Python dependencies, add offline CI, replace legacy commit Action,
  serialize repository writers and retain diagnostics/progress on failure.
- [x] Run unit tests, workflow validation and a small live arxiv smoke test.
- [x] Review the changes, commit with the repository's personal identity, push,
  dispatch workflows and check their actual conclusions.

## Verification

Run `python -m unittest discover -s tests -v` with the locked requirements.
Network tests use fake sessions except for an explicit one-paper smoke test.
Failures must preserve saved papers and report failed topics. A partial backfill
must restart from its persisted topic/date/offset. Weekly reports reject stale
harvest status instead of publishing misleading empty reports.

## Acceptance on 2026-10-20

- 89 offline regression tests passed; actionlint and shell syntax checks passed.
- Live metadata harvest recovered 1,404 unique current papers.
- Every historical paper ID and Code/Model link from both baseline catalogues
  was retained, including conflicting links.
- GitHub Tests run `39428777848`: success.
- Daily run `39428865939`: success for all ten topics. The first deployment
  smoke exposed arXiv read timeouts/429; bounded transport retry and cooldown,
  larger pages and failed-topic-only recovery resolved that run failure.
- Link enrichment run `39427497584`: success; 40 queries in 97 seconds, seven
  new repository matches.
- Weekly run `39429297592`: success with nonempty output.
- Historical recovery run `39429856717`: success; 1,000 records, persisted
  cursor `2026-05-15`, topic index `8`, offset `85`.
- Pages was switched to workflow publishing; successful deployments followed
  the automated data commits.

The complete historical recovery remains scheduled and is not yet finished.
