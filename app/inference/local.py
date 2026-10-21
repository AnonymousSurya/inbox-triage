from __future__ import annotations

import gzip
import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from app.api.schemas import TriageRequest, TriageResponse
from app.paths import DATA

log = logging.getLogger(__name__)

TAIL_TURNS = 8
CONFIDENCE_THRESHOLD = 0.45


def _load_jsonl(path: str) -> list[dict]:
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    else:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def _context_text(context: list[dict], tail: int = TAIL_TURNS) -> str:
    # Include speaker role prefix so the model can distinguish who said what.
    parts = []
    for t in context[-tail:]:
        speaker = t.get("speaker", "")
        text = t.get("text", "")
        parts.append(f"[{speaker}] {text}")
    return " ".join(parts)


def _bucket_token(turn_index: int) -> str:
    if turn_index <= 6:
        return "__early__"
    return "__mid__" if turn_index <= 14 else "__late__"


def _tfidf_union() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2),
            max_features=60_000, sublinear_tf=True,
            strip_accents="unicode", min_df=2,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4),
            max_features=80_000, sublinear_tf=True,
            strip_accents="unicode", min_df=3,
        )),
    ])


class LocalClassifier:
    """Calibrated LinearSVC for intent and action prediction.

    Character n-grams handle typos and code-mixed text.
    CalibratedClassifierCV gives proper probabilities for confidence scoring.
    """

    def __init__(self) -> None:
        self._intent_pipe: Pipeline | None = None
        self._action_pipe: Pipeline | None = None
        self._intent_enc = LabelEncoder()
        self._action_enc = LabelEncoder()
        self.intent_to_actions: dict[str, list[str]] = {}

    @classmethod
    def from_train(cls, train_path: str | None = None) -> LocalClassifier:
        path = train_path or str(DATA / "train.jsonl.gz")
        log.info("Loading training data from %s", path)
        rows = _load_jsonl(path)
        log.info("Loaded %d training rows", len(rows))
        clf = cls()
        clf.fit(rows)
        return clf

    def fit(self, rows: list[dict]) -> LocalClassifier:
        texts = [_context_text(r.get("context") or []) for r in rows]
        texts_act = [
            f"{_context_text(r.get('context') or [])} {_bucket_token(r.get('turn_index') or 0)}"
            for r in rows
        ]
        y_intent = self._intent_enc.fit_transform([r["intent"] for r in rows])
        y_action = self._action_enc.fit_transform([r["action"] for r in rows])

        log.info("Training intent classifier on %d samples ...", len(rows))
        self._intent_pipe = Pipeline([
            ("feat", _tfidf_union()),
            ("clf", CalibratedClassifierCV(
                LinearSVC(max_iter=3000, C=1.0, class_weight="balanced"), cv=3, method="sigmoid"
            )),
        ])
        self._intent_pipe.fit(texts, y_intent)

        log.info("Training action classifier on %d samples ...", len(rows))
        self._action_pipe = Pipeline([
            ("feat", _tfidf_union()),
            ("clf", CalibratedClassifierCV(
                LinearSVC(max_iter=3000, C=1.0, class_weight="balanced"), cv=3, method="sigmoid"
            )),
        ])
        self._action_pipe.fit(texts_act, y_action)

        # Build intent → permitted actions lookup from training data
        intent_actions: dict[str, Counter] = defaultdict(Counter)
        for r in rows:
            for a in (r.get("permitted_actions") or []):
                intent_actions[r["intent"]][a] += 1
        self.intent_to_actions = {
            intent: [a for a, _ in counter.most_common()]
            for intent, counter in intent_actions.items()
        }

        log.info("Classifiers ready. %d intents, %d actions",
                 len(self._intent_enc.classes_), len(self._action_enc.classes_))
        return self

    def predict(self, request: TriageRequest) -> TriageResponse:
        context = [t.model_dump() for t in request.context]
        turn_idx = request.turn_index or 0
        text = _context_text(context)
        text_act = f"{text} {_bucket_token(turn_idx)}"

        i_proba = self._intent_pipe.predict_proba([text])[0]
        a_proba = self._action_pipe.predict_proba([text_act])[0]

        top_i = int(i_proba.argmax())
        top_a = int(a_proba.argmax())

        intent_conf = float(i_proba[top_i])
        action_conf = float(a_proba[top_a])
        confidence = round(math.sqrt(intent_conf * action_conf), 4)

        intent = str(self._intent_enc.inverse_transform([top_i])[0])
        action = str(self._action_enc.inverse_transform([top_a])[0])

        return TriageResponse(
            id=request.id,
            intent=intent,
            action=action,
            confidence=confidence,
            needs_human=confidence < CONFIDENCE_THRESHOLD,
        )

    def top_intents(self, request: TriageRequest, n: int = 8) -> list[tuple[str, float]]:
        """Top-N (intent, probability) for LLM candidate list."""
        context = [t.model_dump() for t in request.context]
        text = _context_text(context)
        proba = self._intent_pipe.predict_proba([text])[0]
        indices = proba.argsort()[-n:][::-1]
        labels = self._intent_enc.inverse_transform(indices)
        return [(str(label), round(float(proba[i]), 4)) for label, i in zip(labels, indices)]
