from __future__ import annotations

from app.api.schemas import ErrorBody, TriageRequest, TriageResponse

MAX_TURN_CHARS = 8_000


def abstain(request_id: str, message: str) -> TriageResponse:
    return TriageResponse(
        id=request_id,
        intent="unknown",
        action="none",
        confidence=0.0,
        needs_human=True,
        error=ErrorBody(code="invalid_input", message=message),
    )


def inspect(request: TriageRequest) -> TriageResponse | None:
    """Return an abstention for inputs that are not a request, else None."""
    if not request.context:
        return abstain(request.id, "empty context")

    texts = [turn.text for turn in request.context]
    if all(not text.strip() for text in texts):
        return abstain(request.id, "empty message")

    for turn in request.context:
        if len(turn.text) > MAX_TURN_CHARS:
            return abstain(request.id, "turn exceeds bounded body limit")
        if "\x00" in turn.text:
            return abstain(request.id, "binary junk in a text field")

    return None
