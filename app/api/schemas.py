from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Speaker(str, Enum):
    customer = "customer"
    agent = "agent"
    action = "action"


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    speaker: Speaker
    text: str


class TriageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    context: list[Turn] = Field(max_length=8)
    turn_index: int | None = Field(default=None, ge=0)


class ErrorBody(BaseModel):
    code: str
    message: str


class TriageResponse(BaseModel):
    id: str
    intent: str
    action: str
    confidence: float = Field(ge=0.0, le=1.0)
    needs_human: bool
    error: ErrorBody | None = None
