from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request body accepted by the chat endpoint."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2_000,
        description="FMCG business question",
        examples=[
            (
                "Did Sparkle Summer achieve its planned "
                "sales lift, and why?"
            )
        ],
    )

    session_id: str | None = Field(
        default=None,
        max_length=200,
        description=(
            "Existing conversation session. Omit this field "
            "to create a new session."
        ),
    )


class ChatResponse(BaseModel):
    """Conversation-aware agent response."""

    session_id: str
    original_question: str
    resolved_question: str
    context_applied: bool
    response: dict[str, Any]


class HistoryResponse(BaseModel):
    """Recent and compressed session history."""

    session_id: str
    recent_turns: list[dict[str, Any]]
    compressed_summary: dict[str, Any]


class ClearHistoryResponse(BaseModel):
    """Result of deleting a conversation session."""

    session_id: str
    cleared: bool


class HealthResponse(BaseModel):
    """Application health information."""

    status: str
    service: str
    version: str
    database_available: bool
    document_index_available: bool
    semantic_index_available: bool
    internet_search_configured: bool


class ErrorResponse(BaseModel):
    """Standard API error payload."""

    detail: str