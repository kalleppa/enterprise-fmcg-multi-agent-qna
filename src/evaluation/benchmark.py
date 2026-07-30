from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from src.memory.conversation_manager import (
    ConversationManager,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_QUESTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "questions.json"
)

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "evaluation"
)


class ConversationManagerProtocol(Protocol):
    def answer(
        self,
        question: str,
        session_id: str | None = None,
    ) -> Any:
        ...


@dataclass(frozen=True)
class BenchmarkCheck:
    """Result of one benchmark assertion."""

    name: str
    passed: bool
    expected: Any
    actual: Any
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkTurnResult:
    """Evaluation result for one conversation turn."""

    scenario_id: str
    category: str
    turn_number: int
    question: str
    session_id: str | None
    resolved_question: str | None
    route: str | None
    status: str | None
    context_applied: bool | None
    latency_ms: float | None
    quality_score: float | None
    passed: bool
    score: float
    checks: list[BenchmarkCheck] = field(
        default_factory=list
    )
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "checks": [
                check.to_dict()
                for check in self.checks
            ],
        }


@dataclass
class BenchmarkScenarioResult:
    """Aggregated result for one scenario."""

    scenario_id: str
    category: str
    description: str
    passed: bool
    score: float
    turns: list[BenchmarkTurnResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "category": self.category,
            "description": self.description,
            "passed": self.passed,
            "score": self.score,
            "turns": [
                turn.to_dict()
                for turn in self.turns
            ],
        }


class BenchmarkRunner:
    """Run deterministic checks against conversation scenarios."""

    def __init__(
        self,
        conversation_manager:
        ConversationManagerProtocol | None = None,
    ) -> None:
        self.conversation_manager = (
            conversation_manager
            or ConversationManager()
        )

    def run(
        self,
        scenarios: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Run all scenarios and create an aggregate report."""

        scenario_results: list[
            BenchmarkScenarioResult
        ] = []

        for scenario in scenarios:
            scenario_results.append(
                self._run_scenario(
                    scenario
                )
            )

        return self._build_report(
            scenario_results
        )

    def _run_scenario(
        self,
        scenario: dict[str, Any],
    ) -> BenchmarkScenarioResult:
        scenario_id = str(
            scenario["id"]
        )

        category = str(
            scenario.get(
                "category",
                "uncategorized",
            )
        )

        description = str(
            scenario.get(
                "description",
                "",
            )
        )

        session_id = (
            f"benchmark-{scenario_id}"
        )

        turn_results: list[
            BenchmarkTurnResult
        ] = []

        for turn_number, turn in enumerate(
            scenario.get("turns", []),
            start=1,
        ):
            turn_results.append(
                self._run_turn(
                    scenario_id=scenario_id,
                    category=category,
                    turn_number=turn_number,
                    turn=turn,
                    session_id=session_id,
                )
            )

        scores = [
            result.score
            for result in turn_results
        ]

        return BenchmarkScenarioResult(
            scenario_id=scenario_id,
            category=category,
            description=description,
            passed=(
                bool(turn_results)
                and all(
                    result.passed
                    for result in turn_results
                )
            ),
            score=round(
                statistics.mean(scores),
                2,
            )
            if scores
            else 0.0,
            turns=turn_results,
        )

    def _run_turn(
        self,
        scenario_id: str,
        category: str,
        turn_number: int,
        turn: dict[str, Any],
        session_id: str,
    ) -> BenchmarkTurnResult:
        question = str(
            turn.get(
                "question",
                "",
            )
        )

        expectations = turn.get(
            "expect",
            {},
        )

        try:
            conversation_response = (
                self.conversation_manager.answer(
                    question=question,
                    session_id=session_id,
                )
            )

            response = dict(
                conversation_response.response
            )

            checks = self._evaluate_expectations(
                expectations=expectations,
                conversation_response=(
                    conversation_response
                ),
                response=response,
            )

            passed_checks = sum(
                check.passed
                for check in checks
            )

            score = (
                passed_checks
                / len(checks)
                * 100
                if checks
                else 100.0
            )

            observability = (
                response.get(
                    "observability"
                )
                or {}
            )

            evaluation = (
                response.get(
                    "evaluation"
                )
                or {}
            )

            return BenchmarkTurnResult(
                scenario_id=scenario_id,
                category=category,
                turn_number=turn_number,
                question=question,
                session_id=(
                    conversation_response.session_id
                ),
                resolved_question=(
                    conversation_response
                    .resolved_question
                ),
                route=response.get("route"),
                status=response.get("status"),
                context_applied=(
                    conversation_response
                    .context_applied
                ),
                latency_ms=self._optional_float(
                    observability.get(
                        "duration_ms"
                    )
                ),
                quality_score=self._optional_float(
                    evaluation.get(
                        "score"
                    )
                ),
                passed=all(
                    check.passed
                    for check in checks
                ),
                score=round(score, 2),
                checks=checks,
            )

        except Exception as error:
            return BenchmarkTurnResult(
                scenario_id=scenario_id,
                category=category,
                turn_number=turn_number,
                question=question,
                session_id=session_id,
                resolved_question=None,
                route=None,
                status="error",
                context_applied=None,
                latency_ms=None,
                quality_score=None,
                passed=False,
                score=0.0,
                checks=[],
                error=str(error),
            )

    def _evaluate_expectations(
        self,
        expectations: dict[str, Any],
        conversation_response: Any,
        response: dict[str, Any],
    ) -> list[BenchmarkCheck]:
        checks: list[BenchmarkCheck] = []

        if "route" in expectations:
            expected = expectations["route"]
            actual = response.get("route")

            checks.append(
                self._membership_check(
                    name="route",
                    expected=expected,
                    actual=actual,
                )
            )

        if "status" in expectations:
            expected = expectations["status"]
            actual = response.get("status")

            checks.append(
                self._membership_check(
                    name="status",
                    expected=expected,
                    actual=actual,
                )
            )

        if "context_applied" in expectations:
            expected = bool(
                expectations[
                    "context_applied"
                ]
            )

            actual = bool(
                conversation_response
                .context_applied
            )

            checks.append(
                BenchmarkCheck(
                    name="context_applied",
                    passed=actual == expected,
                    expected=expected,
                    actual=actual,
                )
            )

        if expectations.get(
            "citations_required",
            False,
        ):
            citations = response.get(
                "citations",
                [],
            )

            checks.append(
                BenchmarkCheck(
                    name="citations_required",
                    passed=bool(citations),
                    expected="one or more citations",
                    actual=len(citations),
                )
            )

        answer = str(
            response.get(
                "answer",
                "",
            )
        )

        for expected_text in expectations.get(
            "answer_contains",
            [],
        ):
            passed = (
                str(expected_text).lower()
                in answer.lower()
            )

            checks.append(
                BenchmarkCheck(
                    name=(
                        "answer_contains:"
                        f"{expected_text}"
                    ),
                    passed=passed,
                    expected=expected_text,
                    actual=answer[:500],
                )
            )

        for path in expectations.get(
            "response_path_present",
            [],
        ):
            value, found = self._get_path(
                response,
                str(path),
            )

            checks.append(
                BenchmarkCheck(
                    name=(
                        "response_path_present:"
                        f"{path}"
                    ),
                    passed=(
                        found
                        and value is not None
                        and value != []
                        and value != {}
                    ),
                    expected="present and non-empty",
                    actual=value,
                )
            )

        minimum_score = expectations.get(
            "minimum_evaluation_score"
        )

        if minimum_score is not None:
            evaluation = (
                response.get(
                    "evaluation"
                )
                or {}
            )

            actual_score = self._optional_float(
                evaluation.get("score")
            )

            passed = (
                actual_score is not None
                and actual_score
                >= float(minimum_score)
            )

            checks.append(
                BenchmarkCheck(
                    name=(
                        "minimum_evaluation_score"
                    ),
                    passed=passed,
                    expected=float(
                        minimum_score
                    ),
                    actual=actual_score,
                )
            )

        return checks

    @staticmethod
    def _membership_check(
        name: str,
        expected: Any,
        actual: Any,
    ) -> BenchmarkCheck:
        accepted_values = (
            list(expected)
            if isinstance(
                expected,
                (list, tuple, set),
            )
            else [expected]
        )

        return BenchmarkCheck(
            name=name,
            passed=actual in accepted_values,
            expected=accepted_values,
            actual=actual,
        )

    @staticmethod
    def _get_path(
        payload: dict[str, Any],
        path: str,
    ) -> tuple[Any, bool]:
        current: Any = payload

        for component in path.split("."):
            if (
                not isinstance(
                    current,
                    dict,
                )
                or component not in current
            ):
                return None, False

            current = current[component]

        return current, True

    def _build_report(
        self,
        scenario_results:
        list[BenchmarkScenarioResult],
    ) -> dict[str, Any]:
        turn_results = [
            turn
            for scenario in scenario_results
            for turn in scenario.turns
        ]

        passed_scenarios = sum(
            scenario.passed
            for scenario in scenario_results
        )

        category_summary: dict[
            str,
            dict[str, Any],
        ] = {}

        categories = sorted(
            {
                scenario.category
                for scenario in scenario_results
            }
        )

        for category in categories:
            matching = [
                scenario
                for scenario in scenario_results
                if scenario.category == category
            ]

            category_summary[category] = {
                "scenario_count": len(
                    matching
                ),
                "passed": sum(
                    scenario.passed
                    for scenario in matching
                ),
                "pass_rate_pct": round(
                    (
                        sum(
                            scenario.passed
                            for scenario in matching
                        )
                        / len(matching)
                        * 100
                    ),
                    2,
                ),
                "average_score": round(
                    statistics.mean(
                        scenario.score
                        for scenario in matching
                    ),
                    2,
                ),
            }

        latencies = [
            turn.latency_ms
            for turn in turn_results
            if turn.latency_ms is not None
        ]

        quality_scores = [
            turn.quality_score
            for turn in turn_results
            if turn.quality_score is not None
        ]

        total_scenarios = len(
            scenario_results
        )

        return {
            "summary": {
                "total_scenarios": (
                    total_scenarios
                ),
                "passed_scenarios": (
                    passed_scenarios
                ),
                "failed_scenarios": (
                    total_scenarios
                    - passed_scenarios
                ),
                "scenario_pass_rate_pct": round(
                    (
                        passed_scenarios
                        / total_scenarios
                        * 100
                    ),
                    2,
                )
                if total_scenarios
                else 0.0,
                "total_turns": len(
                    turn_results
                ),
                "passed_turns": sum(
                    turn.passed
                    for turn in turn_results
                ),
                "average_turn_score": round(
                    statistics.mean(
                        turn.score
                        for turn in turn_results
                    ),
                    2,
                )
                if turn_results
                else 0.0,
                "average_quality_score": round(
                    statistics.mean(
                        quality_scores
                    ),
                    2,
                )
                if quality_scores
                else None,
                "median_latency_ms": round(
                    statistics.median(
                        latencies
                    ),
                    3,
                )
                if latencies
                else None,
                "p95_latency_ms": (
                    self._percentile(
                        latencies,
                        95,
                    )
                    if latencies
                    else None
                ),
            },
            "categories": category_summary,
            "scenarios": [
                scenario.to_dict()
                for scenario in scenario_results
            ],
        }

    @staticmethod
    def _percentile(
        values: list[float],
        percentile: float,
    ) -> float:
        ordered = sorted(values)

        rank = math.ceil(
            percentile
            / 100
            * len(ordered)
        )

        index = max(
            0,
            min(
                rank - 1,
                len(ordered) - 1,
            ),
        )

        return round(
            ordered[index],
            3,
        )

    @staticmethod
    def _optional_float(
        value: Any,
    ) -> float | None:
        try:
            return (
                float(value)
                if value is not None
                else None
            )
        except (
            TypeError,
            ValueError,
        ):
            return None


def load_scenarios(
    path: Path = DEFAULT_QUESTIONS_PATH,
) -> list[dict[str, Any]]:
    """Load benchmark scenarios from JSON."""

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(payload, list):
        raise ValueError(
            "Benchmark file must contain a JSON list."
        )

    return payload


def write_benchmark_report(
    report: dict[str, Any],
    output_directory: Path = (
        DEFAULT_OUTPUT_DIRECTORY
    ),
) -> tuple[Path, Path]:
    """Write detailed JSON and turn-level CSV reports."""

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        output_directory
        / "benchmark_report.json"
    )

    csv_path = (
        output_directory
        / "benchmark_turns.csv"
    )

    json_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        fieldnames = [
            "scenario_id",
            "category",
            "turn_number",
            "question",
            "resolved_question",
            "route",
            "status",
            "context_applied",
            "passed",
            "score",
            "latency_ms",
            "quality_score",
            "failed_checks",
            "error",
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for scenario in report[
            "scenarios"
        ]:
            for turn in scenario["turns"]:
                failed_checks = [
                    check["name"]
                    for check in turn["checks"]
                    if not check["passed"]
                ]

                writer.writerow(
                    {
                        "scenario_id": (
                            scenario[
                                "scenario_id"
                            ]
                        ),
                        "category": (
                            scenario["category"]
                        ),
                        "turn_number": (
                            turn["turn_number"]
                        ),
                        "question": (
                            turn["question"]
                        ),
                        "resolved_question": (
                            turn[
                                "resolved_question"
                            ]
                        ),
                        "route": turn["route"],
                        "status": turn["status"],
                        "context_applied": (
                            turn[
                                "context_applied"
                            ]
                        ),
                        "passed": turn["passed"],
                        "score": turn["score"],
                        "latency_ms": (
                            turn["latency_ms"]
                        ),
                        "quality_score": (
                            turn[
                                "quality_score"
                            ]
                        ),
                        "failed_checks": (
                            ",".join(
                                failed_checks
                            )
                        ),
                        "error": turn["error"],
                    }
                )

    return json_path, csv_path