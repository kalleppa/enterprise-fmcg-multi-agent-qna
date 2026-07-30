from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Protocol
from urllib.parse import urlparse

from dotenv import load_dotenv


load_dotenv()


InternetAgentStatus = Literal[
    "success",
    "clarification",
    "unavailable",
    "error",
]


class SearchClientProtocol(Protocol):
    """Interface implemented by an external search provider."""

    def search(
        self,
        query: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class InternetSource:
    """One normalized internet-search result."""

    rank: int
    title: str
    url: str
    domain: str
    content: str
    score: float | None
    published_date: str | None
    citation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InternetSearchResponse:
    """Standard response returned by the internet agent."""

    status: InternetAgentStatus
    query: str
    message: str
    answer: str | None = None
    sources: list[InternetSource] = field(
        default_factory=list
    )
    citations: list[str] = field(
        default_factory=list
    )
    provider: str = "tavily"
    topic: str = "general"
    search_depth: str = "basic"
    retrieved_at_utc: str | None = None
    provider_response_time_seconds: float | None = None
    limitations: list[str] = field(
        default_factory=list
    )
    errors: list[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)

        result["sources"] = [
            source.to_dict()
            for source in self.sources
        ]

        return result


class InternetSearchAgent:
    """Retrieve current external evidence using Tavily."""

    ALLOWED_TOPICS = {
        "general",
        "news",
        "finance",
    }

    ALLOWED_DEPTHS = {
        "basic",
        "advanced",
    }

    ALLOWED_TIME_RANGES = {
        "day",
        "week",
        "month",
        "year",
        "d",
        "w",
        "m",
        "y",
    }

    def __init__(
        self,
        client: SearchClientProtocol | None = None,
        api_key: str | None = None,
    ) -> None:
        self.api_key = (
            api_key
            or os.getenv("TAVILY_API_KEY")
        )

        self.client = client

        if self.client is None and self.api_key:
            self.client = self._create_tavily_client(
                self.api_key
            )

    def search(
        self,
        query: str,
        topic: str = "general",
        max_results: int = 5,
        search_depth: str = "basic",
        time_range: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> InternetSearchResponse:
        """Run a safe and bounded internet search."""

        cleaned_query = query.strip()

        if not cleaned_query:
            return InternetSearchResponse(
                status="clarification",
                query=query,
                message=(
                    "Please provide a topic to search "
                    "on the internet."
                ),
            )

        normalized_topic = topic.strip().lower()
        normalized_depth = search_depth.strip().lower()

        if normalized_topic not in self.ALLOWED_TOPICS:
            return InternetSearchResponse(
                status="error",
                query=query,
                message="Unsupported internet-search topic.",
                errors=[
                    "topic must be general, news, or finance."
                ],
            )

        if normalized_depth not in self.ALLOWED_DEPTHS:
            return InternetSearchResponse(
                status="error",
                query=query,
                message="Unsupported search depth.",
                errors=[
                    "search_depth must be basic or advanced."
                ],
            )

        if not 1 <= max_results <= 10:
            return InternetSearchResponse(
                status="error",
                query=query,
                message="Invalid result limit.",
                errors=[
                    "max_results must be between 1 and 10 "
                    "for this prototype."
                ],
            )

        if (
            time_range is not None
            and time_range
            not in self.ALLOWED_TIME_RANGES
        ):
            return InternetSearchResponse(
                status="error",
                query=query,
                message="Invalid search time range.",
                errors=[
                    "time_range must be day, week, month, "
                    "year, d, w, m, or y."
                ],
            )

        if self.client is None:
            return InternetSearchResponse(
                status="unavailable",
                query=query,
                message=(
                    "Internet search is not configured. "
                    "Set TAVILY_API_KEY in the local .env file."
                ),
                limitations=[
                    "No external internet request was made."
                ],
            )

        search_arguments: dict[str, Any] = {
            "topic": normalized_topic,
            "search_depth": normalized_depth,
            "max_results": max_results,
            "include_answer": "basic",
            "include_raw_content": False,
            "include_images": False,
        }

        if time_range:
            search_arguments[
                "time_range"
            ] = time_range

        if start_date:
            search_arguments[
                "start_date"
            ] = start_date

        if end_date:
            search_arguments[
                "end_date"
            ] = end_date

        if include_domains:
            search_arguments[
                "include_domains"
            ] = include_domains

        if exclude_domains:
            search_arguments[
                "exclude_domains"
            ] = exclude_domains

        try:
            provider_response = self.client.search(
                query=cleaned_query,
                **search_arguments,
            )

        except Exception as error:
            return InternetSearchResponse(
                status="error",
                query=query,
                message=(
                    "The external internet-search provider "
                    "returned an error."
                ),
                errors=[str(error)],
            )

        sources = self._normalize_sources(
            provider_response.get(
                "results",
                [],
            )
        )

        if not sources:
            return InternetSearchResponse(
                status="success",
                query=query,
                message=(
                    "The internet search completed, but no "
                    "usable sources were returned."
                ),
                topic=normalized_topic,
                search_depth=normalized_depth,
                retrieved_at_utc=self._utc_now(),
                limitations=[
                    "No valid HTTP or HTTPS result URLs "
                    "were available."
                ],
            )

        provider_answer = provider_response.get(
            "answer"
        )

        answer = (
            str(provider_answer).strip()
            if provider_answer
            else None
        )

        response_time = self._to_optional_float(
            provider_response.get(
                "response_time"
            )
        )

        citations = [
            source.citation
            for source in sources
        ]

        return InternetSearchResponse(
            status="success",
            query=query,
            message=(
                f"Retrieved {len(sources)} external "
                "internet source(s)."
            ),
            answer=answer,
            sources=sources,
            citations=citations,
            provider="tavily",
            topic=normalized_topic,
            search_depth=normalized_depth,
            retrieved_at_utc=self._utc_now(),
            provider_response_time_seconds=response_time,
            limitations=[
                (
                    "External sources may change after the "
                    "retrieval timestamp."
                )
            ],
        )

    @staticmethod
    def _create_tavily_client(
        api_key: str,
    ) -> SearchClientProtocol:
        try:
            from tavily import TavilyClient
        except ImportError as error:
            raise RuntimeError(
                "tavily-python is not installed. Run "
                "`pip install -r requirements.txt`."
            ) from error

        return TavilyClient(
            api_key=api_key
        )

    @classmethod
    def _normalize_sources(
        cls,
        raw_results: Any,
    ) -> list[InternetSource]:
        """Normalize and deduplicate provider results."""

        if not isinstance(raw_results, list):
            return []

        normalized_sources: list[
            InternetSource
        ] = []

        seen_urls: set[str] = set()

        for raw_result in raw_results:
            if not isinstance(
                raw_result,
                dict,
            ):
                continue

            url = str(
                raw_result.get(
                    "url",
                    "",
                )
            ).strip()

            if not cls._is_safe_web_url(url):
                continue

            normalized_url = url.rstrip("/")

            if normalized_url in seen_urls:
                continue

            seen_urls.add(normalized_url)

            title = str(
                raw_result.get(
                    "title",
                    "Untitled source",
                )
            ).strip()

            content = str(
                raw_result.get(
                    "content",
                    "",
                )
            ).strip()

            domain = (
                urlparse(url).hostname
                or "unknown-domain"
            )

            rank = (
                len(normalized_sources)
                + 1
            )

            citation = (
                f"[{rank}] {title} ({domain})"
            )

            normalized_sources.append(
                InternetSource(
                    rank=rank,
                    title=title,
                    url=url,
                    domain=domain,
                    content=content,
                    score=cls._to_optional_float(
                        raw_result.get(
                            "score"
                        )
                    ),
                    published_date=cls._to_optional_string(
                        raw_result.get(
                            "published_date"
                        )
                    ),
                    citation=citation,
                )
            )

        return normalized_sources

    @staticmethod
    def _is_safe_web_url(
        url: str,
    ) -> bool:
        try:
            parsed = urlparse(url)
        except ValueError:
            return False

        return (
            parsed.scheme in {
                "http",
                "https",
            }
            and bool(parsed.netloc)
        )

    @staticmethod
    def _to_optional_float(
        value: Any,
    ) -> float | None:
        if value is None:
            return None

        try:
            return float(value)
        except (
            TypeError,
            ValueError,
        ):
            return None

    @staticmethod
    def _to_optional_string(
        value: Any,
    ) -> str | None:
        if value is None:
            return None

        cleaned = str(value).strip()

        return cleaned or None

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(
            timezone.utc
        ).isoformat()


def main() -> None:
    """Run a live search when an API key is configured."""

    agent = InternetSearchAgent()

    response = agent.search(
        query=(
            "latest FMCG market trends in India"
        ),
        topic="news",
        max_results=5,
        search_depth="basic",
        time_range="month",
    )

    print("Status:", response.status)
    print("Message:", response.message)
    print("Answer:", response.answer)
    print()

    for source in response.sources:
        print(source.citation)
        print(source.url)
        print(source.content[:250])
        print()


if __name__ == "__main__":
    main()