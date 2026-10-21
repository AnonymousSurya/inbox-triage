# How to submit

Reply to the email that sent you this assignment with a **zip of your git repository**. Do not send a GitHub link, a private repo invite, or a public PR.

The zip must contain the `.git/` folder. We read the commit history. Commit as you work, in small honest commits. Do not squash to a single "final" commit, and do not delete `.git` and re-init before zipping.

```bash
# from inside your solution repo
git status          # working tree should be clean
zip -r ../inbox-triage-yourname.zip . -x '*.pyc' -x '*__pycache__*' -x '.venv/*' -x 'node_modules/*'
```

A zip that is only a snapshot of the files, with no `.git/`, is incomplete. Leave API keys and `.env` out of the archive; ship `.env.example` instead.

## What we expect to boot

The submission is a FastAPI application brought up with Docker Compose. Reviewers run:

```bash
docker compose up --build
./scripts/reproduce.sh
```

`reproduce.sh` talks to the already-running container on `PORT` (default 8000). It waits for `GET /health`, posts the published splits to `POST /triage`, writes `results/predictions.jsonl`, and runs `scripts/grader.py`. It must not start a second copy of the server that blocks the rest of evaluation.

The starter already does this. Day one, `app/inference/` is a dull lexical stand-in so the stack boots; the published numbers come from `scripts/baseline.py`, which the API does not import. Replace the classifier. Keep the service a codebase someone else could extend.

## What has to be in the zip

| Path | What it is |
|---|---|
| `docker-compose.yml`, `Dockerfile` | One command brings the stack up. |
| `app/` | A layered FastAPI service: HTTP, policy, inference, validation. Not a single module that does all four. |
| `data/` | Published splits, schemas and guidelines. Leave it in place. |
| `README.md` | How a reviewer starts Compose, which `PORT`, which env vars. |
| `scripts/reproduce.sh` | Scores the live container; see above. |
| `scripts/grader.py` | Official scorer. Leave it in place. |
| `scripts/baseline.py` | Published lexical baseline. Leave it in place. |
| `MEMO.md` | To the customer-support manager. See [PROBLEM.md](PROBLEM.md#memomd). |
| `ANALYSIS.md` | The required analysis in [PROBLEM.md](PROBLEM.md#required-analysis). |
| `results/predictions.jsonl` | One line per published development row — clean, noisy and hostile — `{id, intent, action, confidence, needs_human}`. |
| `results/manifest.json` | Your **development** measurements and how you produced them. See below. |
| `results/latency_bench.json` | Sample count, concurrency, request ids and externally measured durations. We rerun this. |
| `.env.example` | Names of secrets the service needs, not the values. |
| `.git/` | Intact history. |

We score the numbers **and** we read the code. Passing bars inside an unreadable or unscalable service does not pass. If Compose will not come up, we cannot score the numbers.

## `results/manifest.json`

Report development numbers only. Use JSON `null` for a metric you did not measure. Never invent a hidden-split score.

```json
{
  "claimed": {
    "intent_accuracy": 0.55,
    "macro_f1": 0.50,
    "action_accuracy": 0.55,
    "noisy_gap": 0.08,
    "schema_valid_pct": 100.0,
    "top70_error_ratio": 0.85,
    "cost_per_message_usd": null,
    "latency_p95_s": null
  },
  "spend": {
    "results_usd": 0,
    "development_usd": 0
  },
  "hardware": "CPU laptop; no GPU",
  "concurrency": 4,
  "runs": [
    {
      "run_id": "dev-001",
      "seed": 42,
      "model": "exact-snapshot-or-lexical",
      "status": "completed",
      "input_files": ["data/dev.jsonl", "data/dev_noisy.jsonl", "data/hostile.jsonl"],
      "raw_path": "results/predictions.jsonl"
    }
  ]
}
```

`claimed` must match what `scripts/grader.py` prints on the published development files. Record failed runs as well as successful ones. Artifact paths are relative to the solution root.

## What we run after you send it

Compose, then `reproduce.sh` against that process, then the same table as [PROBLEM.md](PROBLEM.md#how-this-is-assessed): official scorer, hostile pack on the live service, a check that confidence ranks errors, a load pass at the concurrency you stated, and a reading of `app/` for quality, readability and whether a second model would have somewhere to live.

To update a submission, reply again with a new zip and say it replaces the previous one.
