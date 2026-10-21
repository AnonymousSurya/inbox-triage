from __future__ import annotations

from typing import Protocol

from app.api.schemas import TriageRequest, TriageResponse


class Classifier(Protocol):
    def predict(self, request: TriageRequest) -> TriageResponse:
        """Return intent, action, confidence and a human-routing flag."""
