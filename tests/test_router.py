from __future__ import annotations

import pytest

from src.agents.router import IntentRouter


@pytest.fixture()
def router() -> IntentRouter:
    return IntentRouter()


@pytest.mark.parametrize(
    ("question", "expected_intent"),
    [
        ("Hello", "greeting"),
        (
            "What can you do?",
            "capability",
        ),
        (
            "What KPIs are available?",
            "metadata",
        ),
        (
            "Show net revenue by region",
            "structured",
        ),
        (
            "What does the pricing policy say?",
            "document",
        ),
        (
            (
                "Did Sparkle Summer achieve its "
                "target, and why?"
            ),
            "hybrid",
        ),
        (
            "Plot revenue by region",
            "coding",
        ),
        (
            "Search the internet for current trends",
            "internet",
        ),
    ],
)
def test_routes_supported_intents(
    router: IntentRouter,
    question: str,
    expected_intent: str,
) -> None:
    decision = router.route(question)

    assert decision.intent == expected_intent
    assert decision.confidence > 0


def test_routes_unknown_request_as_unsupported(
    router: IntentRouter,
) -> None:
    decision = router.route(
        "Book a flight for tomorrow"
    )

    assert decision.intent == "unsupported"


def test_routes_empty_request_as_unsupported(
    router: IntentRouter,
) -> None:
    decision = router.route("")

    assert decision.intent == "unsupported"