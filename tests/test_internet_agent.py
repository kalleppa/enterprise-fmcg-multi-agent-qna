from __future__ import annotations

from typing import Any

from src.agents.internet_agent import (
    InternetSearchAgent,
)


class FakeSearchClient:
    def __init__(self) -> None:
        self.last_query: str | None = None
        self.last_arguments: dict[
            str,
            Any,
        ] = {}

    def search(
        self,
        query: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.last_query = query
        self.last_arguments = kwargs

        return {
            "answer": (
                "India's FMCG market showed continued "
                "interest in digital channels."
            ),
            "response_time": 1.24,
            "results": [
                {
                    "title": "FMCG Market Report",
                    "url": (
                        "https://example.com/"
                        "fmcg-report"
                    ),
                    "content": (
                        "The report discusses digital "
                        "commerce and rural demand."
                    ),
                    "score": 0.91,
                    "published_date": "2026-07-20",
                },
                {
                    "title": "Retail Industry Update",
                    "url": (
                        "https://example.org/"
                        "retail-update"
                    ),
                    "content": (
                        "Retail channels continued to "
                        "change during the quarter."
                    ),
                    "score": 0.82,
                },
            ],
        }


class FailingSearchClient:
    def search(
        self,
        query: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        raise RuntimeError(
            "Provider unavailable"
        )


def test_returns_normalized_sources() -> None:
    client = FakeSearchClient()

    agent = InternetSearchAgent(
        client=client
    )

    response = agent.search(
        query="latest FMCG trends",
        topic="news",
        max_results=5,
        time_range="month",
    )

    assert response.status == "success"
    assert len(response.sources) == 2
    assert response.answer is not None

    assert response.sources[
        0
    ].domain == "example.com"

    assert response.sources[
        0
    ].citation.startswith(
        "[1]"
    )

    assert client.last_query == (
        "latest FMCG trends"
    )

    assert client.last_arguments[
        "topic"
    ] == "news"

    assert client.last_arguments[
        "time_range"
    ] == "month"



def test_rejects_empty_query() -> None:
    agent = InternetSearchAgent(
        client=FakeSearchClient()
    )

    response = agent.search("")

    assert response.status == "clarification"


def test_rejects_invalid_topic() -> None:
    agent = InternetSearchAgent(
        client=FakeSearchClient()
    )

    response = agent.search(
        query="market update",
        topic="sports",
    )

    assert response.status == "error"
    assert response.errors


def test_rejects_excessive_results() -> None:
    agent = InternetSearchAgent(
        client=FakeSearchClient()
    )

    response = agent.search(
        query="market update",
        max_results=50,
    )

    assert response.status == "error"


def test_handles_provider_failure() -> None:
    agent = InternetSearchAgent(
        client=FailingSearchClient()
    )

    response = agent.search(
        query="market update"
    )

    assert response.status == "error"

    assert "Provider unavailable" in (
        response.errors[0]
    )


def test_filters_invalid_urls() -> None:
    class InvalidURLClient:
        def search(
            self,
            query: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            return {
                "results": [
                    {
                        "title": "Invalid",
                        "url": "file:///tmp/private.txt",
                        "content": "Private data",
                    },
                    {
                        "title": "Valid",
                        "url": "https://example.com/report",
                        "content": "Public report",
                    },
                ]
            }

    agent = InternetSearchAgent(
        client=InvalidURLClient()
    )

    response = agent.search(
        query="test"
    )

    assert response.status == "success"
    assert len(response.sources) == 1
    assert (
        response.sources[0].title
        == "Valid"
    )