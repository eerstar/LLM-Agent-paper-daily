#!/usr/bin/env bash
set -euo pipefail

git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

for path in README.md docs/index.md docs/wechat.md \
    docs/llm-agent-arxiv-daily.json docs/llm-agent-arxiv-daily-web.json \
    docs/llm-agent-arxiv-daily-wechat.json docs/weekly reports \
    .cache/stats.json .cache/harvest.json .cache/backfill.json .cache/link-enrichment.json; do
  if [[ -e "$path" ]]; then
    git add -- "$path"
  fi
done
if git diff --cached --quiet; then
  echo "No generated changes."
  exit 0
fi
git commit -m "chore(papers): ${PIPELINE_MODE:-update} catalogue and progress"
for attempt in 1 2 3; do
  git fetch origin main
  # Never force-push or discard a concurrent human edit.
  git rebase origin/main
  if git push origin HEAD:main; then
    exit 0
  fi
  sleep "$((attempt * 2))"
done
echo "::error::Unable to push after three attempts; diagnostics contain saved progress."
exit 1
