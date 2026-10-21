# Dataset

Published splits, schemas and written procedures. Nothing in this folder is the
service — that is `../app/`. The official scorer and the published lexical
baseline are `../scripts/grader.py` and `../scripts/baseline.py`.

How to boot the stack and replay the baseline numbers is in the [root README](../README.md).
The measured baseline table, and what you are supposed to build, are in [PROBLEM.md](../PROBLEM.md#deliverable).

## Files

| File | What it is |
|---|---|
| `train.jsonl.gz` | 20,000 labelled rows from ABCD's train split. Fit on this. |
| `dev.jsonl` | 2,000 labelled rows from ABCD's dev split. Tune and report on this. |
| `dev_noisy.jsonl` | 300 rows from `dev.jsonl` with the last customer turn rewritten. Same labels. `noise_kind` is one of: `typo`, `code_mixed`, `two_intents`, `empty`, `very_long`, `screenshot`. |
| `hostile.jsonl` | 9 inputs that are not requests (injection, unicode, binary junk, empty). No labels. Scored only for staying up and not being talked into anything. |
| `input.schema.json` | What `POST /triage` receives: `id`, `context`, optional `turn_index`. See the [service contract](../README.md#service-contract). |
| `schema.json` | What it must return: `id`, `intent`, `action`, `confidence`, `needs_human`. |
| `guidelines.json` | Written procedures. You may index them. |
| `manifest.json` | Upstream URLs, SHA-256, seed, counts. |

The held-out split is built the same way from ABCD's test conversations and is
not published. Split boundaries are ABCD's: no conversation appears in two
splits. Empty text appears in both `dev_noisy.jsonl` (labelled) and
`hostile.jsonl` (unlabelled). Those are different jobs.

## A row

One row is **one action-taking turn**: the dialogue so far, and the action the
human agent took next. Context is the last 8 turns.

```json
{"id": "3129-8",
 "context": [{"speaker": "customer", "text": "i want to return an item"},
             {"speaker": "agent", "text": "sure, may i have your name?"}],
 "intent": "return_size", "flow": "product_defect",
 "action": "pull-up-account",
 "permitted_actions": ["ask-the-oracle", "enter-details", "..."],
 "turn_index": 8}
```

`intent`, `flow`, `action` and `permitted_actions` are gold. They are in the
training and development files so you can fit and debug. They are **not** in
`input.schema.json` and they will not be on the wire at inference.

Speakers are `customer`, `agent` or `action`. Action turns describe prior
observed actions, never the withheld next-action label. Keep them when you
build model input.

A one-message action label would be a coin toss: the first action of an ABCD
conversation is `pull-up-account` 67% of the time. The unit is a prefix because
that is where the label honestly lives.

## `permitted_actions`

This is what that flow's written procedure in `guidelines.json` authorises,
resolved into the action names the turn labels actually use. The guidelines and
the labels name some things differently, and the guidelines fold every
knowledge-base lookup into one step; that is resolved once, at build time, so
you can test membership directly.

About **9% of the gold actions sit outside that set** (0.0883 on the published
development files), because real agents deviate. This is not a bug and we have
not cleaned it. A perfect predictor of the gold label still scores about 9%
policy violations, so `scripts/grader.py` reports your rate next to the gold's
own rate rather than against zero. Follow the label, follow the policy, or flag
the conflict — and write down which.

## Licence

ABCD is MIT, © ASAPP Inc. `manifest.json` has the upstream URLs and SHA-256 so
you can check you have the same bytes we do.
