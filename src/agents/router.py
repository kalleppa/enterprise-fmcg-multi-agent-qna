from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Literal


Intent = Literal[
    "greeting",
    "capability",
    "metadata",
    "structured",
    "document",
    "hybrid",
    "coding",
    "internet",
    "unsupported",
]


@dataclass(frozen=True)
class RouteDecision:
    """Intent selected by the deterministic router."""

    intent: Intent
    confidence: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class IntentRouter:
    """Route user questions to specialist agents."""

    GREETING_PHRASES = {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "namaste",
    }

    CAPABILITY_PHRASES = (
        "what can you do",
        "how can you help",
        "what do you support",
        "your capabilities",
        "what are your capabilities",
    )

    METADATA_PHRASES = (
        "what kpis",
        "available kpis",
        "list kpis",
        "what metrics",
        "available metrics",
        "what datasets",
        "available datasets",
        "what tables",
        "available tables",
        "what dimensions",
        "available dimensions",
        "what documents are available",
        "list documents",
        "available periods",
        "date range",
        "data period",
    )

    INTERNET_PHRASES = (
        "search the internet",
        "search online",
        "search the web",
        "internet search",
        "web search",
        "latest news",
        "current market trend",
        "current market share",
        "today's market",
        "today market",
        "current commodity price",
        "latest commodity price",
        "public market information",
    )

    CODING_PHRASES = (
        "plot",
        "chart",
        "graph",
        "correlation",
        "percentage change",
        "summary statistics",
        "statistical summary",
    )

    HYBRID_PHRASES = (
        "and why",
        "explain why",
        "what affected",
        "what caused",
        "reason for",
        "reasons for",
        "did the campaign achieve",
        "did the campaign meet",
        "target and actual",
        "planned versus actual",
        "planned vs actual",
        "compare the target",
        "according to the data and document",
        "using data and documents",
    )

    DOCUMENT_PHRASES = (
        "document",
        "campaign brief",
        "brief say",
        "policy say",
        "pricing policy",
        "review say",
        "quarterly review",
        "category review",
        "distributor review",
        "inventory report",
        "exception report",
        "launch note",
        "discontinuation notice",
        "identified risks",
        "recommended actions",
        "root causes",
    )

    STRUCTURED_PHRASES = (
        "revenue",
        "sales",
        "units sold",
        "gross margin",
        "discount",
        "inventory",
        "stockout",
        "closing stock",
        "promotion spend",
        "sales lift",
        "highest",
        "lowest",
        "compare",
        "growth",
        "performance",
        "by region",
        "by state",
        "by brand",
        "by channel",
        "by month",
        "by quarter",
        "by year",
    )

    def route(self, question: str) -> RouteDecision:
        """Return the most appropriate intent."""

        normalized = self._normalize(question)

        if not normalized:
            return RouteDecision(
                intent="unsupported",
                confidence=1.0,
                reasons=("The request is empty.",),
            )

        if normalized in self.GREETING_PHRASES:
            return RouteDecision(
                intent="greeting",
                confidence=1.0,
                reasons=("The request is a greeting.",),
            )

        if self._contains_any(
            normalized,
            self.CAPABILITY_PHRASES,
        ):
            return RouteDecision(
                intent="capability",
                confidence=0.99,
                reasons=(
                    "The user asked about supported capabilities.",
                ),
            )

        if self._contains_any(
            normalized,
            self.METADATA_PHRASES,
        ):
            return RouteDecision(
                intent="metadata",
                confidence=0.98,
                reasons=(
                    "The request asks for available metadata.",
                ),
            )

        if self._contains_any(
            normalized,
            self.INTERNET_PHRASES,
        ):
            return RouteDecision(
                intent="internet",
                confidence=0.96,
                reasons=(
                    "The request explicitly requires current "
                    "or external information.",
                ),
            )

        coding_match = self._contains_any(
            normalized,
            self.CODING_PHRASES,
        )

        hybrid_match = self._contains_any(
            normalized,
            self.HYBRID_PHRASES,
        )

        document_match = self._contains_any(
            normalized,
            self.DOCUMENT_PHRASES,
        )

        structured_match = self._contains_any(
            normalized,
            self.STRUCTURED_PHRASES,
        )

        if coding_match:
            return RouteDecision(
                intent="coding",
                confidence=0.94,
                reasons=(
                    "The request requires a controlled "
                    "calculation or visualization.",
                ),
            )

        if hybrid_match or (
            document_match and structured_match
        ):
            return RouteDecision(
                intent="hybrid",
                confidence=0.93,
                reasons=(
                    "The request requires structured results "
                    "and document evidence.",
                ),
            )

        if document_match:
            return RouteDecision(
                intent="document",
                confidence=0.91,
                reasons=(
                    "The request asks for information from "
                    "enterprise documents.",
                ),
            )

        if structured_match:
            return RouteDecision(
                intent="structured",
                confidence=0.90,
                reasons=(
                    "The request contains a supported business "
                    "KPI, entity, dimension, or comparison.",
                ),
            )

        return RouteDecision(
            intent="unsupported",
            confidence=0.60,
            reasons=(
                "No supported enterprise intent was detected.",
            ),
        )

    @staticmethod
    def _contains_any(
        text: str,
        phrases: tuple[str, ...],
    ) -> bool:
        return any(
            phrase in text
            for phrase in phrases
        )

    @staticmethod
    def _normalize(value: str) -> str:
        normalized = value.lower().strip()
        normalized = normalized.replace("_", " ")

        normalized = re.sub(
            r"[^\w%\-']+",
            " ",
            normalized,
        )

        normalized = re.sub(
            r"\s+",
            " ",
            normalized,
        )

        return normalized.strip()