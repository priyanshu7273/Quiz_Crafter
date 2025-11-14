"""Pydantic models used by the public API and internal components."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, EmailStr, Field, HttpUrl, validator


class QuizPayload(BaseModel):
    """Incoming payload received from the evaluator."""

    email: EmailStr = Field(..., description="Registered student email")
    secret: str = Field(..., description="Shared secret to validate authenticity")
    url: HttpUrl = Field(..., description="Quiz starting URL")


class QuizSubmission(BaseModel):
    """Payload sent back to the evaluator when submitting answers."""

    email: EmailStr
    secret: str
    url: HttpUrl
    answer: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QuizProcessingResult(BaseModel):
    """Structured result returned by the quiz processor."""

    correct: bool
    answer: Any
    explanation: str
    submitted_at: datetime = Field(default_factory=datetime.utcnow)
    next_url: Optional[HttpUrl] = None
    raw_response: Optional[Dict[str, Any]] = None


class APIResponse(BaseModel):
    """Default API response wrapper."""

    status: str = Field(..., description="Status string (accepted/failed)")
    detail: str = Field(..., description="Human readable message")
    defensive_system_prompt: str
    offensive_user_prompt: str


class QuizProcessingError(Exception):
    """Custom exception for recoverable quiz processing errors."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class SubmissionResponse(BaseModel):
    """Response returned by quiz endpoints when answers are submitted."""

    correct: bool = False
    reason: Optional[str] = None
    url: Optional[HttpUrl] = None

    @validator("reason", pre=True)
    def _convert_empty_string(cls, value: Optional[str]) -> Optional[str]:
        if isinstance(value, str) and not value.strip():
            return None
        return value
