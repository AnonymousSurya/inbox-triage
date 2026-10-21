from __future__ import annotations

import logging
import os

from app.api.schemas import TriageRequest, TriageResponse
from app.inference.classifier import Classifier
from app.inference.llm import LLMClassifier
from app.inference.local import LocalClassifier
from app.inference.router import RouterClassifier
from app.policy.guidelines import GuidelineCatalog
from app.validation.input import inspect

log = logging.getLogger(__name__)


class TriageService:
    def __init__(
        self,
        classifier: Classifier,
        catalog: GuidelineCatalog,
        intent_to_actions: dict[str, list[str]] | None = None,
    ) -> None:
        self._classifier = classifier
        self._catalog = catalog
        self._intent_to_actions = intent_to_actions or {}

    def apply_policy(self, prediction: TriageResponse) -> TriageResponse:
        permitted = self._intent_to_actions.get(prediction.intent)
        if permitted and prediction.action not in permitted:
            # Gold labels violate policy ~9% of the time — don't override the
            # predicted action, but flag for human review.
            return TriageResponse(
                id=prediction.id,
                intent=prediction.intent,
                action=prediction.action,
                confidence=prediction.confidence,
                needs_human=True,
            )
        return prediction

    def handle(self, request: TriageRequest) -> TriageResponse:
        blocked = inspect(request)
        if blocked is not None:
            return blocked
        return self.apply_policy(self._classifier.predict(request))


def build_triage_service() -> TriageService:
    local = LocalClassifier.from_train()
    catalog = GuidelineCatalog.load()

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    llm: LLMClassifier | None = None
    if api_key:
        llm = LLMClassifier(intent_to_actions=local.intent_to_actions, api_key=api_key)
        log.info("LLM fallback enabled (model=%s)", os.environ.get("GROQ_MODEL"))
    else:
        log.info("No GROQ_API_KEY — running local classifier only")

    classifier = RouterClassifier(local=local, llm=llm)
    return TriageService(classifier, catalog, intent_to_actions=local.intent_to_actions)
