from __future__ import annotations

from types import SimpleNamespace

from src.agents.orchestrator import (
    EnterpriseQnAOrchestrator,
)


class FakeStructuredAgent:
    def answer(
        self,
        question: str,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            status="success",
            message="Structured result returned.",
            data={
                "rows": [
                    {
                        "region": "South Region",
                        "net_revenue_inr": 100000,
                    },
                    {
                        "region": "West Region",
                        "net_revenue_inr": 120000,
                    },
                ],
                "source": "fmcg.duckdb",
                "referenced_tables": [
                    "vw_sales_enriched"
                ],
            },
            query_id="sql-test123",
            assumptions=[],
            to_dict=lambda: {
                "status": "success",
                "message": (
                    "Structured result returned."
                ),
                "data": {
                    "rows": [
                        {
                            "region": "South Region",
                            "net_revenue_inr": 100000,
                        },
                        {
                            "region": "West Region",
                            "net_revenue_inr": 120000,
                        },
                    ],
                    "source": "fmcg.duckdb",
                    "referenced_tables": [
                        "vw_sales_enriched"
                    ],
                },
                "query_id": "sql-test123",
                "assumptions": [],
            },
        )


class FakeDocumentAgent:
    def answer(
        self,
        question: str,
        top_k: int = 5,
    ) -> SimpleNamespace:
        evidence = [
            {
                "chunk_id": "DOC-001-0001",
                "document_id": "DOC-001",
                "title": "Campaign Review",
                "document_type": (
                    "quarterly_business_review"
                ),
                "section": "Reasons",
                "content": (
                    "Inventory shortages reduced sales."
                ),
                "snippet": (
                    "Inventory shortages reduced sales."
                ),
                "citation": (
                    "Campaign Review — Reasons "
                    "(chunk 1)"
                ),
                "source_path": (
                    "data/documents/review.md"
                ),
                "score": 1.0,
                "retrieval_methods": (
                    "keyword",
                    "semantic",
                ),
                "metadata": {},
            }
        ]

        return SimpleNamespace(
            status="success",
            message="Document evidence returned.",
            evidence=evidence,
            citations=[
                (
                    "Campaign Review — Reasons "
                    "(chunk 1)"
                )
            ],
            limitations=[],
            to_dict=lambda: {
                "status": "success",
                "message": (
                    "Document evidence returned."
                ),
                "evidence": evidence,
                "citations": [
                    (
                        "Campaign Review — Reasons "
                        "(chunk 1)"
                    )
                ],
                "limitations": [],
            },
        )


class FakeCodingAgent:
    def analyze(
        self,
        **kwargs: object,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            status="success",
            message="Chart generated.",
            results={
                "chart_type": "bar",
                "plotted_points": 2,
            },
            chart_path=(
                "data/generated/charts/test.png"
            ),
            assumptions=[],
            to_dict=lambda: {
                "status": "success",
                "message": "Chart generated.",
                "results": {
                    "chart_type": "bar",
                    "plotted_points": 2,
                },
                "chart_path": (
                    "data/generated/charts/test.png"
                ),
                "assumptions": [],
            },
        )


def build_orchestrator() -> (
    EnterpriseQnAOrchestrator
):
    return EnterpriseQnAOrchestrator(
        structured_agent=FakeStructuredAgent(),
        document_agent=FakeDocumentAgent(),
        coding_agent=FakeCodingAgent(),
    )


def test_handles_greeting() -> None:
    orchestrator = build_orchestrator()

    response = orchestrator.answer("Hello")

    assert response.status == "success"
    assert response.route == "greeting"


def test_handles_structured_question() -> None:
    orchestrator = build_orchestrator()

    response = orchestrator.answer(
        "Show net revenue by region"
    )

    assert response.status == "success"
    assert response.route == "structured"
    assert "South Region" in response.answer
    assert response.structured_result is not None


def test_handles_document_question() -> None:
    orchestrator = build_orchestrator()

    response = orchestrator.answer(
        "What does the campaign review say?"
    )

    assert response.status == "success"
    assert response.route == "document"
    assert response.citations


def test_handles_hybrid_question() -> None:
    orchestrator = build_orchestrator()

    response = orchestrator.answer(
        (
            "Did Sparkle Summer achieve its "
            "target, and why?"
        )
    )

    assert response.status == "success"
    assert response.route == "hybrid"
    assert response.structured_result is not None
    assert response.document_result is not None
    assert response.citations


def test_handles_coding_question() -> None:
    orchestrator = build_orchestrator()

    response = orchestrator.answer(
        "Plot net revenue by region"
    )

    assert response.status == "success"
    assert response.route == "coding"
    assert response.analysis_result is not None
    assert "Chart generated" in response.answer


def test_reports_unconfigured_internet_agent() -> None:
    orchestrator = build_orchestrator()

    response = orchestrator.answer(
        "Search the internet for current FMCG trends"
    )

    assert response.status == "unsupported"
    assert response.route == "internet"
    assert response.limitations


def test_handles_out_of_scope_request() -> None:
    orchestrator = build_orchestrator()

    response = orchestrator.answer(
        "Book a flight for tomorrow"
    )

    assert response.status == "unsupported"
    assert response.route == "unsupported"