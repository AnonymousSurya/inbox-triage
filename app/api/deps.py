from __future__ import annotations

from fastapi import Request

from app.services.triage import TriageService


def get_triage(request: Request) -> TriageService:
    return request.app.state.triage
