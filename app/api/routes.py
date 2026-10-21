from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_triage
from app.api.schemas import TriageRequest, TriageResponse
from app.services.triage import TriageService

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/triage", response_model=TriageResponse)
def triage(
    request: TriageRequest,
    service: TriageService = Depends(get_triage),
) -> TriageResponse:
    return service.handle(request)
