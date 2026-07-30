from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi.testclient import TestClient

from src.main import create_app


@dataclass
class FakeConversationResponse:
    session_id: str
    original_question: str
    resolved_question: str
    context_applied: bool
    response: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "original_question": (
                self.original_question
            ),
            "resolved_question": (
                self.resolved_question
            ),
            "context_applied": (
                self.context_applied
            ),
            "response": self.response,
        }


class FakeConversationManager:
    def __init__(self) -> None:
        self.sessions: dict[
            str,
            list[dict[str, Any]],
        ] = {}

    def answer(
        self,
        question: str,
        session_id: str | None = None,
    ) -> FakeConversationResponse:
        resolved_session_id = (
            session_id
            or "session-test123"
        )

        turns = self.sessions.setdefault(
            resolved_session_id,
            [],
        )

        turn = {
            "user_question": question,
            "resolved_question": question,
            "route": "structured",
            "status": "success",
            "answer": "Test answer",
        }

        turns.append(turn)

        return FakeConversationResponse(
            session_id=resolved_session_id,
            original_question=question,
            resolved_question=question,
            context_applied=False,
            response={
                "status": "success",
                "route": "structured",
                "answer": "Test answer",
                "route_decision": {
                    "intent": "structured",
                    "confidence": 1.0,
                    "reasons": [
                        "Test route",
                    ],
                },
            },
        )

    def get_history(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "recent_turns": self.sessions.get(
                session_id,
                [],
            ),
            "compressed_summary": {},
        }

    def clear_history(
        self,
        session_id: str,
    ) -> bool:
        return (
            self.sessions.pop(
                session_id,
                None,
            )
            is not None
        )


def fake_metadata_provider() -> dict[str, Any]:
    return {
        "structured": {
            "datasets": [
                {
                    "dataset_name": "sales",
                }
            ],
            "kpis": [
                {
                    "kpi_name": "Net Revenue",
                }
            ],
        },
        "documents": [
            {
                "document_id": "DOC-TEST-001",
                "title": "Test Document",
            }
        ],
    }


def create_test_client() -> TestClient:
    app = create_app(
        conversation_manager=(
            FakeConversationManager()
        ),
        metadata_provider=(
            fake_metadata_provider
        ),
    )

    return TestClient(app)


def test_health_endpoint() -> None:
    client = create_test_client()

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "healthy"
    assert "database_available" in payload


def test_chat_endpoint_creates_session() -> None:
    client = create_test_client()

    response = client.post(
        "/chat",
        json={
            "question": (
                "Show net revenue by region"
            )
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert (
        payload["session_id"]
        == "session-test123"
    )

    assert (
        payload["response"]["status"]
        == "success"
    )


def test_chat_endpoint_accepts_existing_session() -> None:
    client = create_test_client()

    response = client.post(
        "/chat",
        json={
            "question": "Why?",
            "session_id": "session-existing",
        },
    )

    assert response.status_code == 200

    assert (
        response.json()["session_id"]
        == "session-existing"
    )


def test_chat_rejects_empty_question() -> None:
    client = create_test_client()

    response = client.post(
        "/chat",
        json={
            "question": "",
        },
    )

    assert response.status_code == 422


def test_history_endpoint() -> None:
    client = create_test_client()

    client.post(
        "/chat",
        json={
            "question": "Show revenue",
            "session_id": "session-history",
        },
    )

    response = client.get(
        "/history/session-history"
    )

    assert response.status_code == 200

    payload = response.json()

    assert (
        payload["session_id"]
        == "session-history"
    )

    assert len(
        payload["recent_turns"]
    ) == 1


def test_missing_history_returns_404() -> None:
    client = create_test_client()

    response = client.get(
        "/history/missing-session"
    )

    assert response.status_code == 404


def test_clear_history_endpoint() -> None:
    client = create_test_client()

    client.post(
        "/chat",
        json={
            "question": "Show revenue",
            "session_id": "session-delete",
        },
    )

    response = client.delete(
        "/history/session-delete"
    )

    assert response.status_code == 200

    assert response.json() == {
        "session_id": "session-delete",
        "cleared": True,
    }


def test_metadata_endpoint() -> None:
    client = create_test_client()

    response = client.get(
        "/metadata"
    )

    assert response.status_code == 200

    payload = response.json()

    assert "structured" in payload
    assert "documents" in payload


def test_chart_path_traversal_is_rejected() -> None:
    client = create_test_client()

    response = client.get(
        "/charts/..%2Fprivate.png"
    )

    assert response.status_code in {
        400,
        404,
    }