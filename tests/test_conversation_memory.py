from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.memory.conversation_manager import (
    ConversationManager,
)
from src.memory.conversation_store import (
    ConversationTurn,
    InMemoryConversationStore,
)


@dataclass
class FakeOrchestratorResponse:
    status: str
    route: str
    answer: str
    citations: list[str] = field(
        default_factory=list
    )
    assumptions: list[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "route": self.route,
            "answer": self.answer,
            "citations": self.citations,
            "assumptions": self.assumptions,
        }


class FakeOrchestrator:
    def __init__(self) -> None:
        self.received_questions: list[str] = []

    def answer(
        self,
        question: str,
    ) -> FakeOrchestratorResponse:
        self.received_questions.append(
            question
        )

        route = (
            "hybrid"
            if "Explain why" in question
            or "enterprise documents" in question
            else "structured"
        )

        return FakeOrchestratorResponse(
            status="success",
            route=route,
            answer=f"Answered: {question}",
        )


def test_creates_session() -> None:
    manager = ConversationManager(
        orchestrator=FakeOrchestrator()
    )

    response = manager.answer(
        "Show net revenue by region"
    )

    assert response.session_id.startswith(
        "session-"
    )

    assert response.context_applied is False


def test_resolves_why_follow_up() -> None:
    orchestrator = FakeOrchestrator()

    manager = ConversationManager(
        orchestrator=orchestrator
    )

    first = manager.answer(
        "Show net revenue by region for Q2 2025"
    )

    second = manager.answer(
        "Why?",
        session_id=first.session_id,
    )

    assert second.context_applied is True

    assert (
        "Show net revenue by region for Q2 2025"
        in second.resolved_question
    )

    assert (
        "Explain why using structured data"
        in second.resolved_question
    )


def test_resolves_inventory_follow_up() -> None:
    manager = ConversationManager(
        orchestrator=FakeOrchestrator()
    )

    first = manager.answer(
        "Which region performed worst?"
    )

    second = manager.answer(
        "Was inventory a factor there?",
        session_id=first.session_id,
    )

    assert second.context_applied is True

    assert (
        "inventory was a factor"
        in second.resolved_question
    )


def test_independent_question_is_not_rewritten() -> None:
    manager = ConversationManager(
        orchestrator=FakeOrchestrator()
    )

    first = manager.answer(
        "Show net revenue by region"
    )

    second = manager.answer(
        "What KPIs are available?",
        session_id=first.session_id,
    )

    assert second.context_applied is False

    assert (
        second.resolved_question
        == "What KPIs are available?"
    )


def test_sessions_are_isolated() -> None:
    manager = ConversationManager(
        orchestrator=FakeOrchestrator()
    )

    first_session = manager.answer(
        "Show net revenue by region"
    )

    second_session = manager.answer(
        "Show stockout days by state"
    )

    follow_up = manager.answer(
        "Why?",
        session_id=first_session.session_id,
    )

    assert (
        "Show net revenue by region"
        in follow_up.resolved_question
    )

    assert (
        "Show stockout days by state"
        not in follow_up.resolved_question
    )

    assert (
        first_session.session_id
        != second_session.session_id
    )


def test_stores_conversation_history() -> None:
    manager = ConversationManager(
        orchestrator=FakeOrchestrator()
    )

    first = manager.answer(
        "Show revenue by region"
    )

    manager.answer(
        "Why?",
        session_id=first.session_id,
    )

    history = manager.get_history(
        first.session_id
    )

    assert len(
        history["recent_turns"]
    ) == 2


def test_compacts_old_turns() -> None:
    store = InMemoryConversationStore(
        max_recent_turns=2
    )

    session_id = store.create_session(
        "test-session"
    )

    for number in range(4):
        store.add_turn(
            session_id=session_id,
            turn=ConversationTurn(
                turn_id=f"turn-{number}",
                user_question=f"Question {number}",
                resolved_question=(
                    f"Resolved question {number}"
                ),
                route="structured",
                status="success",
                answer=f"Answer {number}",
            ),
        )

    context = store.get_context(
        session_id
    )

    assert len(
        context["recent_turns"]
    ) == 2

    assert (
        context[
            "compressed_summary"
        ]["compressed_turn_count"]
        == 2
    )


def test_clears_history() -> None:
    manager = ConversationManager(
        orchestrator=FakeOrchestrator()
    )

    response = manager.answer(
        "Show revenue by region"
    )

    removed = manager.clear_history(
        response.session_id
    )

    assert removed is True

    history = manager.get_history(
        response.session_id
    )

    assert history["recent_turns"] == []