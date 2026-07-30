from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.evaluation.benchmark import (
    BenchmarkRunner,
    write_benchmark_report,
)


@dataclass
class FakeConversationResponse:
    session_id: str
    original_question: str
    resolved_question: str
    context_applied: bool
    response: dict[str, Any]


class FakeConversationManager:
    def answer(
        self,
        question: str,
        session_id: str | None = None,
    ) -> FakeConversationResponse:
        is_follow_up = (
            question.lower()
            == "why?"
        )

        response = {
            "status": "success",
            "route": (
                "hybrid"
                if is_follow_up
                else "structured"
            ),
            "answer": (
                "Structured and document evidence"
                if is_follow_up
                else "Structured result"
            ),
            "citations": (
                ["Test Document — Reasons"]
                if is_follow_up
                else []
            ),
            "structured_result": {
                "status": "success",
                "query_id": "sql-test",
                "data": {
                    "rows": [
                        {
                            "region": "South"
                        }
                    ],
                    "source": "test.duckdb",
                    "referenced_tables": [
                        "vw_sales_enriched"
                    ],
                },
            },
            "document_result": (
                {
                    "status": "success",
                    "evidence": [
                        {
                            "citation": (
                                "Test Document — Reasons"
                            )
                        }
                    ],
                }
                if is_follow_up
                else None
            ),
            "observability": {
                "duration_ms": 10.5
            },
            "evaluation": {
                "score": 100.0,
                "passed": True,
            },
        }

        return FakeConversationResponse(
            session_id=(
                session_id
                or "benchmark-session"
            ),
            original_question=question,
            resolved_question=(
                "Show revenue. Explain why."
                if is_follow_up
                else question
            ),
            context_applied=is_follow_up,
            response=response,
        )


def test_runs_single_turn_scenario() -> None:
    runner = BenchmarkRunner(
        conversation_manager=(
            FakeConversationManager()
        )
    )

    report = runner.run(
        [
            {
                "id": "structured-test",
                "category": "structured",
                "description": "Test",
                "turns": [
                    {
                        "question": (
                            "Show revenue by region"
                        ),
                        "expect": {
                            "route": "structured",
                            "status": "success",
                            "response_path_present": [
                                (
                                    "structured_result"
                                    ".data.rows"
                                )
                            ],
                        },
                    }
                ],
            }
        ]
    )

    assert (
        report["summary"][
            "scenario_pass_rate_pct"
        ]
        == 100.0
    )


def test_runs_multi_turn_scenario() -> None:
    runner = BenchmarkRunner(
        conversation_manager=(
            FakeConversationManager()
        )
    )

    report = runner.run(
        [
            {
                "id": "multi-turn-test",
                "category": "multi_turn",
                "description": "Test",
                "turns": [
                    {
                        "question": (
                            "Show revenue by region"
                        ),
                        "expect": {
                            "route": "structured",
                            "status": "success",
                        },
                    },
                    {
                        "question": "Why?",
                        "expect": {
                            "route": "hybrid",
                            "status": "success",
                            "context_applied": True,
                            "citations_required": True,
                        },
                    },
                ],
            }
        ]
    )

    assert (
        report["summary"][
            "scenario_pass_rate_pct"
        ]
        == 100.0
    )

    assert (
        report["summary"][
            "total_turns"
        ]
        == 2
    )


def test_detects_failed_expectation() -> None:
    runner = BenchmarkRunner(
        conversation_manager=(
            FakeConversationManager()
        )
    )

    report = runner.run(
        [
            {
                "id": "failure-test",
                "category": "routing",
                "description": "Test",
                "turns": [
                    {
                        "question": (
                            "Show revenue by region"
                        ),
                        "expect": {
                            "route": "document",
                            "status": "success",
                        },
                    }
                ],
            }
        ]
    )

    assert (
        report["summary"][
            "scenario_pass_rate_pct"
        ]
        == 0.0
    )


def test_writes_json_and_csv(
    tmp_path: Path,
) -> None:
    runner = BenchmarkRunner(
        conversation_manager=(
            FakeConversationManager()
        )
    )

    report = runner.run(
        [
            {
                "id": "output-test",
                "category": "structured",
                "description": "Test",
                "turns": [
                    {
                        "question": "Show revenue",
                        "expect": {
                            "route": "structured",
                            "status": "success",
                        },
                    }
                ],
            }
        ]
    )

    json_path, csv_path = (
        write_benchmark_report(
            report=report,
            output_directory=tmp_path,
        )
    )

    assert json_path.exists()
    assert csv_path.exists()