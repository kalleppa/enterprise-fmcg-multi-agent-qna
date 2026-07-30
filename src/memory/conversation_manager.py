from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from src.agents.orchestrator import (
    EnterpriseQnAOrchestrator,
)
from src.memory.conversation_store import (
    ConversationTurn,
    InMemoryConversationStore,
)


@dataclass
class ConversationResponse:
    """Conversation-aware response returned to the API layer."""

    session_id: str
    original_question: str
    resolved_question: str
    context_applied: bool
    response: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConversationManager:
    """Add session memory around the main orchestrator."""

    CONTEXTUAL_EXACT_PHRASES = {
        "why",
        "why?",
        "explain",
        "explain why",
        "how so",
        "what caused it",
        "what caused that",
        "what about this",
        "what about that",
        "and why",
        "tell me more",
        "show more",
        "continue",
    }

    CONTEXTUAL_PREFIXES = (
        "what about ",
        "how about ",
        "and ",
        "also ",
        "same for ",
        "do the same for ",
        "compare that with ",
        "compare it with ",
        "why was ",
        "why did ",
        "was inventory ",
        "did inventory ",
        "what caused ",
    )

    INDEPENDENT_PHRASES = (
        "what kpis",
        "what datasets",
        "what documents",
        "what dimensions",
        "what can you do",
        "search the internet",
        "show net revenue",
        "show revenue",
        "show units sold",
        "show stockout",
        "plot ",
        "calculate ",
    )

    def __init__(
        self,
        orchestrator: EnterpriseQnAOrchestrator
        | None = None,
        store: InMemoryConversationStore
        | None = None,
    ) -> None:
        self.orchestrator = (
            orchestrator
            or EnterpriseQnAOrchestrator()
        )

        self.store = (
            store
            or InMemoryConversationStore()
        )

    def answer(
        self,
        question: str,
        session_id: str | None = None,
    ) -> ConversationResponse:
        """Resolve context, execute the agent, and store the turn."""

        resolved_session_id = (
            self.store.create_session(
                session_id=session_id
            )
        )

        resolved_question, context_applied = (
            self._resolve_follow_up(
                question=question,
                session_id=resolved_session_id,
            )
        )

        orchestrator_response = (
            self.orchestrator.answer(
                resolved_question
            )
        )

        response_dict = (
            orchestrator_response.to_dict()
        )

        turn = ConversationTurn(
            turn_id=self._build_turn_id(
                session_id=resolved_session_id,
                turn_number=(
                    len(
                        self.store.get_recent_turns(
                            resolved_session_id
                        )
                    )
                    + 1
                ),
            ),
            user_question=question,
            resolved_question=resolved_question,
            route=orchestrator_response.route,
            status=orchestrator_response.status,
            answer=orchestrator_response.answer,
            citations=tuple(
                orchestrator_response.citations
            ),
            assumptions=tuple(
                orchestrator_response.assumptions
            ),
        )

        self.store.add_turn(
            session_id=resolved_session_id,
            turn=turn,
        )

        return ConversationResponse(
            session_id=resolved_session_id,
            original_question=question,
            resolved_question=resolved_question,
            context_applied=context_applied,
            response=response_dict,
        )

    def get_history(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        """Return recent and compressed conversation history."""

        return self.store.get_context(
            session_id
        )

    def clear_history(
        self,
        session_id: str,
    ) -> bool:
        """Clear one user's conversation history."""

        return self.store.clear_session(
            session_id
        )

    def _resolve_follow_up(
        self,
        question: str,
        session_id: str,
    ) -> tuple[str, bool]:
        """Resolve incomplete follow-ups using the previous turn."""

        cleaned_question = re.sub(
            r"\s+",
            " ",
            question.strip(),
        )

        if not cleaned_question:
            return cleaned_question, False

        previous_turn = (
            self.store.get_last_successful_turn(
                session_id
            )
        )

        if previous_turn is None:
            return cleaned_question, False

        if not self._is_contextual_follow_up(
            cleaned_question
        ):
            return cleaned_question, False

        normalized = cleaned_question.lower()

        if normalized in {
            "why",
            "why?",
            "explain",
            "explain why",
            "and why",
            "how so",
            "what caused it",
            "what caused that",
        }:
            resolved = (
                f"{previous_turn.resolved_question}. "
                "Explain why using structured data and "
                "enterprise documents."
            )

            return resolved, True

        if any(
            phrase in normalized
            for phrase in (
                "was inventory a factor",
                "did inventory affect",
                "inventory a factor",
            )
        ):
            resolved = (
                f"{previous_turn.resolved_question}. "
                "Determine whether inventory was a factor "
                "using inventory data and enterprise documents."
            )

            return resolved, True

        if normalized.startswith(
            (
                "same for ",
                "do the same for ",
            )
        ):
            new_target = re.sub(
                r"^(same for|do the same for)\s+",
                "",
                cleaned_question,
                flags=re.IGNORECASE,
            )

            resolved = (
                f"{previous_turn.resolved_question}. "
                f"Apply the same analysis to {new_target}."
            )

            return resolved, True

        resolved = (
            f"{previous_turn.resolved_question}. "
            f"Follow-up request: {cleaned_question}"
        )

        return resolved, True

    def _is_contextual_follow_up(
        self,
        question: str,
    ) -> bool:
        """Determine whether the request depends on prior context."""

        normalized = question.lower().strip()

        if any(
            phrase in normalized
            for phrase in self.INDEPENDENT_PHRASES
        ):
            return False

        if normalized in self.CONTEXTUAL_EXACT_PHRASES:
            return True

        if normalized.startswith(
            self.CONTEXTUAL_PREFIXES
        ):
            return True

        words = normalized.split()

        pronouns = {
            "it",
            "that",
            "this",
            "there",
            "those",
            "same",
            "previous",
            "above",
        }

        return (
            len(words) <= 10
            and bool(
                set(words)
                & pronouns
            )
        )

    @staticmethod
    def _build_turn_id(
        session_id: str,
        turn_number: int,
    ) -> str:
        """Build a deterministic turn identifier."""

        safe_session_id = re.sub(
            r"[^a-zA-Z0-9_-]",
            "",
            session_id,
        )

        return (
            f"{safe_session_id}-"
            f"turn-{turn_number:04d}"
        )


def main() -> None:
    """Run a multi-turn conversation demonstration."""

    manager = ConversationManager()

    first_response = manager.answer(
        question=(
            "Show net revenue by region for Q2 2025"
        )
    )

    session_id = first_response.session_id

    print("=" * 80)
    print("Session:", session_id)
    print("Question:", first_response.original_question)
    print("Resolved:", first_response.resolved_question)
    print(first_response.response["answer"])

    follow_up_response = manager.answer(
        question="Why?",
        session_id=session_id,
    )

    print("=" * 80)
    print("Question:", follow_up_response.original_question)
    print("Resolved:", follow_up_response.resolved_question)
    print(
        "Context applied:",
        follow_up_response.context_applied,
    )
    print(follow_up_response.response["answer"])

    third_response = manager.answer(
        question="Was inventory a factor there?",
        session_id=session_id,
    )

    print("=" * 80)
    print("Question:", third_response.original_question)
    print("Resolved:", third_response.resolved_question)
    print(
        "Context applied:",
        third_response.context_applied,
    )
    print(third_response.response["answer"])

    print("=" * 80)
    print("Conversation history")
    print(manager.get_history(session_id))


if __name__ == "__main__":
    main()