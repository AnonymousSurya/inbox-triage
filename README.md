# Inbox triage

This repository is the starter. `app/` is the service you extend. `data/` is the
published dataset and schemas. `scripts/baseline.py` and `scripts/grader.py` are
the official measurement tools — they are not part of the service.

`docker compose up --build` brings up the FastAPI service on `PORT` (default 8000).
`scripts/reproduce.sh` talks to that process: it waits for `GET /health`, posts
every published development row to `POST /triage`, writes `results/predictions.jsonl`,
and runs `scripts/grader.py`.

Day one, `app/inference/` ships a dull lexical stand-in so the stack boots. The
numbers we published were measured with `scripts/baseline.py`, not by importing
that script into the API. Replace the classifier. Keep the HTTP, policy and
validation layers as the place a second model or a new guideline entry would land.

```
app/                 FastAPI service
  api/               HTTP, request and response shapes
  services/          orchestration: validate → classify → policy hook
  inference/         the model; swap this without touching routes
  policy/            guidelines.json, loaded and queryable
  validation/        empty, oversize and hostile text
data/                splits, schemas, guidelines — see data/README.md
scripts/
  reproduce.sh       score the live container
  score_live.py      POST each published row
  baseline.py        published lexical baseline (stdlib)
  grader.py          official scorer (stdlib)
results/             predictions you ship
```

We do not provide API keys. If you call a model, stay in the budget class
named in [PROBLEM.md](PROBLEM.md) — GPT mini, Gemini Flash, or a Groq-hosted
open-weight model under 35B — and use a free tier (Groq and Gemini both have
one). Put the key in `.env`, not in the zip.

```bash
cp .env.example .env          # add a key only if your classifier needs one
docker compose up --build     # service on :8000
./scripts/reproduce.sh        # scores the live container
```

To reproduce the published baseline numbers without Docker:

```bash
python scripts/baseline.py --train data/train.jsonl.gz \
    --eval data/dev.jsonl --eval-noisy data/dev_noisy.jsonl \
    --eval-hostile data/hostile.jsonl --pred-out results/baseline_predictions.jsonl

python scripts/grader.py --pred results/baseline_predictions.jsonl \
    --gold data/dev.jsonl --noisy data/dev_noisy.jsonl \
    --hostile data/hostile.jsonl --schema data/schema.json
```

You should see the table in [PROBLEM.md](PROBLEM.md#deliverable): intent 0.3748,
action 0.3913, macro-F1 0.4508, noisy gap 0.0743, schema 100%, policy 0.1404
(gold 0.0883), confidence error ratio 1.0.

If the container is already healthy, `reproduce.sh` does not rebuild it.

## Service contract

`GET /health` returns 200 after the process has loaded.

`POST /triage` accepts [`data/input.schema.json`](data/input.schema.json) and
returns [`data/schema.json`](data/schema.json), echoing the input `id`. Extra
response keys are tolerated; `cost_usd` and `latency_ms`, if you send them, are
claims we remeasure. Gold fields (`intent`, `flow`, `action`,
`permitted_actions`) are not on the wire. Inputs may be empty, unusually
encoded, or long.

If the envelope is valid but the text cannot be handled, return the normal
shape with `intent: "unknown"`, `action: "none"`, `confidence: 0`,
`needs_human: true`. An extra `error: {"code": "invalid_input", "message": "..."}`
may explain the abstention. That still passes structural validation. It earns
no intent or action credit on a labelled row.

Malformed HTTP or JSON may return HTTP 400. Those transport probes are separate
from the published valid-JSON hostile rows. Bound request timeouts and body
limits so the published pack fits.

Read [PROBLEM.md](PROBLEM.md) before changing the model. Submit as described in
[SUBMISSION.md](SUBMISSION.md).
