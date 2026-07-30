from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    """Return the current UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ConversationTurn:
    """One completed interaction in a conversation."""

    turn_id: str
    user_question: str
    resolved_question: str
    route: str
    status: str
    answer: str
    citations: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    created_at_utc: str = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConversationSession:
    """Conversation history and compressed context."""

    session_id: str
    turns: list[ConversationTurn] = field(
        default_factory=list
    )
    compressed_summary: dict[str, Any] = field(
        default_factory=dict
    )
    created_at_utc: str = field(
        default_factory=utc_now
    )
    updated_at_utc: str = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turns": [
                turn.to_dict()
                for turn in self.turns
            ],
            "compressed_summary": (
                self.compressed_summary
            ),
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
        }


class InMemoryConversationStore:
    """
    Thread-safe in-memory conversation store.

    Only recent turns are retained in full. Older turns are
    compressed into a lightweight summary to control memory growth.
    """

    def __init__(
        self,
        max_recent_turns: int = 8,
        max_sessions: int = 1_000,
    ) -> None:
        if max_recent_turns <= 0:
            raise ValueError(
                "max_recent_turns must be greater than zero."
            )

        if max_sessions <= 0:
            raise ValueError(
                "max_sessions must be greater than zero."
            )

        self.max_recent_turns = max_recent_turns
        self.max_sessions = max_sessions

        self._sessions: dict[
            str,
            ConversationSession,
        ] = {}

        self._lock = RLock()

    def create_session(
        self,
        session_id: str | None = None,
    ) -> str:
        """Create a session or return an existing session ID."""

        resolved_session_id = (
            session_id
            or f"session-{uuid4().hex[:16]}"
        )

        with self._lock:
            if resolved_session_id in self._sessions:
                return resolved_session_id

            self._evict_oldest_session_if_needed()

            self._sessions[
                resolved_session_id
            ] = ConversationSession(
                session_id=resolved_session_id
            )

        return resolved_session_id

    def get_session(
        self,
        session_id: str,
    ) -> ConversationSession | None:
        """Return a session when it exists."""

        with self._lock:
            return self._sessions.get(
                session_id
            )

    def add_turn(
        self,
        session_id: str,
        turn: ConversationTurn,
    ) -> None:
        """Add a turn and compact old conversation history."""

        with self._lock:
            if session_id not in self._sessions:
                self.create_session(session_id)

            session = self._sessions[
                session_id
            ]

            session.turns.append(turn)
            session.updated_at_utc = utc_now()

            while (
                len(session.turns)
                > self.max_recent_turns
            ):
                old_turn = session.turns.pop(0)

                self._merge_into_summary(
                    session=session,
                    turn=old_turn,
                )

    def get_recent_turns(
        self,
        session_id: str,
    ) -> list[ConversationTurn]:
        """Return recent turns for one session."""

        with self._lock:
            session = self._sessions.get(
                session_id
            )

            if session is None:
                return []

            return list(session.turns)

    def get_last_successful_turn(
        self,
        session_id: str,
    ) -> ConversationTurn | None:
        """Return the most recent successful interaction."""

        turns = self.get_recent_turns(
            session_id
        )

        for turn in reversed(turns):
            if turn.status == "success":
                return turn

        return None

    def get_context(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        """Return context for follow-up resolution."""

        with self._lock:
            session = self._sessions.get(
                session_id
            )

            if session is None:
                return {
                    "session_id": session_id,
                    "recent_turns": [],
                    "compressed_summary": {},
                }

            return {
                "session_id": session_id,
                "recent_turns": [
                    turn.to_dict()
                    for turn in session.turns
                ],
                "compressed_summary": dict(
                    session.compressed_summary
                ),
            }

    def clear_session(
        self,
        session_id: str,
    ) -> bool:
        """Delete one conversation session."""

        with self._lock:
            return (
                self._sessions.pop(
                    session_id,
                    None,
                )
                is not None
            )

    def session_count(self) -> int:
        """Return the current session count."""

        with self._lock:
            return len(self._sessions)

    def _merge_into_summary(
        self,
        session: ConversationSession,
        turn: ConversationTurn,
    ) -> None:
        """Compress an older turn into session-level memory."""

        summary = session.compressed_summary

        summary["compressed_turn_count"] = (
            int(
                summary.get(
                    "compressed_turn_count",
                    0,
                )
            )
            + 1
        )

        summary[
            "last_compressed_question"
        ] = turn.resolved_question

        summary[
            "last_compressed_route"
        ] = turn.route

        previous_questions = list(
            summary.get(
                "previous_questions",
                [],
            )
        )

        previous_questions.append(
            turn.resolved_question
        )

        summary["previous_questions"] = (
            previous_questions[-5:]
        )

        previous_citations = list(
            summary.get(
                "citations",
                [],
            )
        )

        previous_citations.extend(
            turn.citations
        )

        summary["citations"] = list(
            dict.fromkeys(
                previous_citations
            )
        )[-20:]

    def _evict_oldest_session_if_needed(
        self,
    ) -> None:
        """Remove the least recently updated session."""

        if len(self._sessions) < self.max_sessions:
            return

        oldest_session_id = min(
            self._sessions,
            key=lambda current_session_id: (
                self._sessions[
                    current_session_id
                ].updated_at_utc
            ),
        )

        del self._sessions[
            oldest_session_id
        ]