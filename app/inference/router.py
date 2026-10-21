from __future__ import annotations

import logging
import os

from app.api.schemas import TriageRequest, TriageResponse
from app.inference.llm import LLMClassifier
from app.inference.local import LocalClassifier  # noqa: F401 (re-exported via type hints)

log = logging.getLogger(__name__)

_DEFAULT_THRESHOLD = float(os.environ.get("LLM_CONFIDENCE_THRESHOLD", "0.45"))


class RouterClassifier:
    """Sends low-confidence local predictions to the LLM for a second opinion.

    High confidence (>= threshold): local model answer returned as-is.
    Low confidence (< threshold):   LLM is called with the top candidate intents.
    LLM failure or unavailable:     local model answer used as fallback.
    """

    def __init__(
        self,
        local: LocalClassifier,
        llm: LLMClassifier | None = None,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> None:
        self._local = local
        self._llm = llm
        self._threshold = threshold

    def predict(self, request: TriageRequest) -> TriageResponse:
        result = self._local.predict(request)

        if self._llm and self._llm.available() and result.confidence < self._threshold:
            candidates = self._local.top_intents(request, n=8)
            llm_result = self._llm.predict(request, candidates)
            if llm_result is not None:
                log.debug("LLM intent override for %s (local conf=%.3f)", request.id, result.confidence)
                # Use LLM intent (better on rare/noisy cases) but keep local
                # model's action and confidence — the local classifier is
                # trained on 20k action examples and has calibrated probabilities.
                return TriageResponse(
                    id=result.id,
                    intent=llm_result.intent,
                    action=result.action,
                    confidence=result.confidence,
                    needs_human=result.needs_human,
                )

        return result
