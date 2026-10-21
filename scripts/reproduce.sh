#!/usr/bin/env bash
# Bring the stack up if needed, score the published splits against the live
# service, then run the official grader.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PORT="${PORT:-8000}"
BASE="http://127.0.0.1:${PORT}"

if ! python3 -c "import urllib.request; urllib.request.urlopen('${BASE}/health', timeout=2)" \
    >/dev/null 2>&1; then
  echo "starting stack on port ${PORT}"
  PORT="$PORT" docker compose up -d --build
fi

mkdir -p results

python3 scripts/score_live.py \
  --base "$BASE" \
  --eval data/dev.jsonl \
  --eval-noisy data/dev_noisy.jsonl \
  --eval-hostile data/hostile.jsonl \
  --pred-out results/predictions.jsonl

python3 scripts/grader.py \
  --pred results/predictions.jsonl \
  --gold data/dev.jsonl \
  --noisy data/dev_noisy.jsonl \
  --hostile data/hostile.jsonl \
  --schema data/schema.json \
  --report results/report.json
