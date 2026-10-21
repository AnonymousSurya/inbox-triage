from __future__ import annotations

import logging
import os

from app.api.schemas import TriageRequest, TriageResponse
from app.inference.llm import LLMClassifier
from app.inference.local import LocalClassifier

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
                log.debug("LLM override for %s (local conf=%.3f)", request.id, result.confidence)
                return llm_result

        return result
