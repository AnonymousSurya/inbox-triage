from __future__ import annotations

import gzip
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from app.api.schemas import TriageRequest, TriageResponse
from app.paths import DATA

TOKEN = re.compile(r"[a-z0-9']+")
TAIL_TURNS = 4


def _tokens(row: dict) -> list[str]:
    text = " ".join(t.get("text", "") for t in (row.get("context") or [])[-TAIL_TURNS:])
    return TOKEN.findall(text.lower())


def _turn_bucket(row: dict) -> str:
    idx = row.get("turn_index") or 0
    return "early" if idx <= 6 else ("mid" if idx <= 14 else "late")


def _load_jsonl(path: str) -> list[dict]:
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            text = fh.read()
    else:
        text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


class LexicalClassifier:
    """Day-one stand-in, same idea as `scripts/baseline.py`. Replace it.

    The published numbers come from that script, not from this module. Do not
    import the script from here.
    """

    def __init__(self) -> None:
        self._centroids: dict[str, dict[str, float]] = {}
        self._action_by_intent_bucket: dict[tuple[str, str], str] = {}
        self._action_by_intent: dict[str, str] = {}
        self._fallback_action = "pull-up-account"
        self._fallback_intent = "manage"

    @classmethod
    def from_train(cls, train_path: str | None = None) -> LexicalClassifier:
        path = train_path or str(DATA / "train.jsonl.gz")
        model = cls()
        model.fit(_load_jsonl(path))
        return model

    def fit(self, rows: list[dict]) -> LexicalClassifier:
        per_intent: dict[str, Counter] = defaultdict(Counter)
        doc_freq: Counter = Counter()
        for row in rows:
            toks = _tokens(row)
            per_intent[row["intent"]].update(toks)
            doc_freq.update(set(toks))
        n_docs = max(1, len(rows))
        for intent, counts in per_intent.items():
            vec = {t: c * math.log(n_docs / (1 + doc_freq[t])) for t, c in counts.items()}
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            self._centroids[intent] = {t: v / norm for t, v in vec.items()}

        pair: dict[tuple[str, str], Counter] = defaultdict(Counter)
        solo: dict[str, Counter] = defaultdict(Counter)
        for row in rows:
            pair[(row["intent"], _turn_bucket(row))][row["action"]] += 1
            solo[row["intent"]][row["action"]] += 1
        self._action_by_intent_bucket = {k: c.most_common(1)[0][0] for k, c in pair.items()}
        self._action_by_intent = {k: c.most_common(1)[0][0] for k, c in solo.items()}
        self._fallback_action = Counter(r["action"] for r in rows).most_common(1)[0][0]
        self._fallback_intent = Counter(r["intent"] for r in rows).most_common(1)[0][0]
        return self

    def predict(self, request: TriageRequest) -> TriageResponse:
        row = {
            "id": request.id,
            "context": [turn.model_dump() for turn in request.context],
            "turn_index": request.turn_index or 0,
        }
        toks = Counter(_tokens(row))
        norm = math.sqrt(sum(v * v for v in toks.values())) or 1.0
        scored = [
            (sum(centroid.get(t, 0.0) * c for t, c in toks.items()) / norm, intent)
            for intent, centroid in self._centroids.items()
        ]
        scored.sort(reverse=True)

        if not scored or scored[0][0] <= 0:
            intent, confidence = self._fallback_intent, 0.05
        else:
            intent = scored[0][1]
            runner_up = scored[1][0] if len(scored) > 1 else 0.0
            confidence = round(min(0.99, max(0.05, (scored[0][0] - runner_up) * 4)), 4)

        action = (
            self._action_by_intent_bucket.get((intent, _turn_bucket(row)))
            or self._action_by_intent.get(intent)
            or self._fallback_action
        )
        return TriageResponse(
            id=str(row["id"]),
            intent=intent,
            action=action,
            confidence=confidence,
            needs_human=confidence < 0.20,
        )
