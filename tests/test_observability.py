from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from src.evaluation.observability import (
    ObservedOrchestrator,
)
from src.evaluation.response_evaluator import (
    ResponseEvaluator,
)


class FakeStructuredAgent:
    def answer(
        self,
        question: str,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            status="success",
            message=(
                "Structured result returned."
            ),
            data={
                "rows": [
                    {
                        "region": "South Region",
                        "net_revenue_inr": 100000,
                    }
                ],
                "source": "fmcg.duckdb",
                "referenced_tables": [
                    "vw_sales_enriched"
                ],
            },
            query_id="sql-observed-001",
            assumptions=[],
            to_dict=lambda: {
                "status": "success",
                "message": (
                    "Structured result returned."
                ),
                "data": {
                    "rows": [
                        {
                            "region": (
                                "South Region"
                            ),
                            "net_revenue_inr": (
                                100000
                            ),
                        }
                    ],
                    "source": "fmcg.duckdb",
                    "referenced_tables": [
                        "vw_sales_enriched"
                    ],
                },
                "query_id": (
                    "sql-observed-001"
                ),
                "assumptions": [],
            },
        )


class UnusedAgent:
    def answer(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> SimpleNamespace:
        raise AssertionError(
            "This specialist should not be called."
        )

    def analyze(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> SimpleNamespace:
        raise AssertionError(
            "This specialist should not be called."
        )

    def search(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> SimpleNamespace:
        raise AssertionError(
            "This specialist should not be called."
        )


def test_observed_structured_request() -> None:
    orchestrator = ObservedOrchestrator(
        structured_agent=(
            FakeStructuredAgent()
        ),
        document_agent=UnusedAgent(),
        coding_agent=UnusedAgent(),
        internet_agent=UnusedAgent(),
    )

    response = orchestrator.answer(
        "Show net revenue by region"
    )

    assert response.status == "success"
    assert response.route == "structured"

    assert response.observability is not None
    assert response.evaluation is not None

    trace = response.observability

    assert trace["trace_id"].startswith(
        "trace-"
    )

    assert trace["tool_call_count"] == 1

    assert (
        trace[
            "per_agent_call_count"
        ]["structured_agent"]
        == 1
    )

    assert (
        trace[
            "per_agent_latency_ms"
        ]["structured_agent"]
        >= 0
    )

    assert (
        trace[
            "model_usage"
        ]["model_call_count"]
        == 0
    )

    assert response.evaluation[
        "passed"
    ] is True


def test_trace_can_be_retrieved() -> None:
    orchestrator = ObservedOrchestrator(
        structured_agent=(
            FakeStructuredAgent()
        ),
        document_agent=UnusedAgent(),
        coding_agent=UnusedAgent(),
        internet_agent=UnusedAgent(),
    )

    response = orchestrator.answer(
        "Show net revenue by region"
    )

    trace_id = response.observability[
        "trace_id"
    ]

    stored_trace = orchestrator.get_trace(
        trace_id
    )

    assert stored_trace is not None
    assert (
        stored_trace["trace_id"]
        == trace_id
    )


def test_evaluator_detects_missing_citations() -> None:
    evaluator = ResponseEvaluator()

    evaluation = evaluator.evaluate(
        question="What does the policy say?",
        response={
            "status": "success",
            "route": "document",
            "answer": "The policy describes discounts.",
            "document_result": {
                "status": "success",
                "evidence": [
                    {
                        "title": "Pricing Policy",
                        "content": "Discount policy",
                    }
                ],
            },
            "citations": [],
            "limitations": [],
        },
    )

    assert evaluation.passed is False

    assert (
        "document_citations"
        in evaluation.failed_checks
    )


def test_evaluator_accepts_document_citations() -> None:
    evaluator = ResponseEvaluator()

    evaluation = evaluator.evaluate(
        question="What does the policy say?",
        response={
            "status": "success",
            "route": "document",
            "answer": "The policy describes discounts.",
            "document_result": {
                "status": "success",
                "evidence": [
                    {
                        "title": "Pricing Policy",
                        "citation": (
                            "Pricing Policy — Discounts "
                            "(chunk 1)"
                        ),
                    }
                ],
            },
            "citations": [
                (
                    "Pricing Policy — Discounts "
                    "(chunk 1)"
                )
            ],
            "limitations": [],
        },
    )

    assert evaluation.passed is True
    assert evaluation.score == 100.0


def test_evaluator_detects_missing_provenance() -> None:
    evaluator = ResponseEvaluator()

    evaluation = evaluator.evaluate(
        question="Show revenue by region",
        response={
            "status": "success",
            "route": "structured",
            "answer": "Revenue results",
            "structured_result": {
                "status": "success",
                "data": {
                    "rows": [],
                },
            },
            "limitations": [],
        },
    )

    assert evaluation.passed is False

    assert (
        "structured_provenance"
        in evaluation.failed_checks
    )
    