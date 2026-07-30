from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlparse


@dataclass(frozen=True)
class EvaluationCheck:
    """One deterministic response-quality check."""

    name: str
    passed: bool
    weight: float
    details: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResponseEvaluation:
    """Aggregated evaluation of an orchestrator response."""

    score: float
    passed: bool
    checks: tuple[EvaluationCheck, ...]
    failed_checks: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "passed": self.passed,
            "checks": [
                check.to_dict()
                for check in self.checks
            ],
            "failed_checks": list(
                self.failed_checks
            ),
        }


class ResponseEvaluator:
    """Run deterministic checks on multi-agent responses."""

    VALID_STATUSES = {
        "success",
        "clarification",
        "unsupported",
        "blocked",
        "error",
    }

    VALID_ROUTES = {
        "greeting",
        "capability",
        "metadata",
        "structured",
        "document",
        "hybrid",
        "coding",
        "internet",
        "unsupported",
    }

    def __init__(
        self,
        passing_score: float = 80.0,
    ) -> None:
        if not 0 <= passing_score <= 100:
            raise ValueError(
                "passing_score must be between 0 and 100."
            )

        self.passing_score = passing_score

    def evaluate(
        self,
        question: str,
        response: dict[str, Any],
    ) -> ResponseEvaluation:
        """Evaluate one serialized orchestrator response."""

        checks: list[EvaluationCheck] = []

        status = str(
            response.get(
                "status",
                "",
            )
        )

        route = str(
            response.get(
                "route",
                "",
            )
        )

        answer = str(
            response.get(
                "answer",
                "",
            )
        ).strip()

        checks.append(
            EvaluationCheck(
                name="valid_status",
                passed=(
                    status
                    in self.VALID_STATUSES
                ),
                weight=1.5,
                details=(
                    f"Status returned: {status or 'missing'}"
                ),
            )
        )

        checks.append(
            EvaluationCheck(
                name="valid_route",
                passed=(
                    route
                    in self.VALID_ROUTES
                ),
                weight=1.5,
                details=(
                    f"Route returned: {route or 'missing'}"
                ),
            )
        )

        checks.append(
            EvaluationCheck(
                name="answer_present",
                passed=bool(answer),
                weight=2.0,
                details=(
                    "A non-empty final answer is required."
                ),
            )
        )

        checks.append(
            EvaluationCheck(
                name="question_present",
                passed=bool(
                    question.strip()
                ),
                weight=0.5,
                details=(
                    "The evaluated user question must "
                    "not be empty."
                ),
            )
        )

        if route in {
            "structured",
            "hybrid",
            "coding",
        }:
            checks.append(
                self._check_structured_provenance(
                    response
                )
            )

        if route in {
            "document",
            "hybrid",
        }:
            checks.append(
                self._check_document_citations(
                    response
                )
            )

        if route == "coding":
            checks.append(
                self._check_analysis_result(
                    response
                )
            )

        if route == "internet":
            checks.append(
                self._check_internet_sources(
                    response
                )
            )

        if status in {
            "unsupported",
            "blocked",
            "error",
        }:
            limitations = response.get(
                "limitations",
                [],
            )

            checks.append(
                EvaluationCheck(
                    name="failure_transparency",
                    passed=bool(limitations),
                    weight=1.0,
                    details=(
                        "Unsupported, blocked, and failed "
                        "responses should explain limitations."
                    ),
                )
            )

        total_weight = sum(
            check.weight
            for check in checks
        )

        passed_weight = sum(
            check.weight
            for check in checks
            if check.passed
        )

        score = (
            passed_weight
            / total_weight
            * 100
            if total_weight
            else 0.0
        )

        failed_checks = tuple(
            check.name
            for check in checks
            if not check.passed
        )

        critical_checks = {
            "valid_status",
            "valid_route",
            "answer_present",
        }

        critical_failure = bool(
            critical_checks
            & set(failed_checks)
        )

        return ResponseEvaluation(
            score=round(score, 2),
            passed=(
                score >= self.passing_score
                and not critical_failure
            ),
            checks=tuple(checks),
            failed_checks=failed_checks,
        )

    @staticmethod
    def _check_structured_provenance(
        response: dict[str, Any],
    ) -> EvaluationCheck:
        structured_result = response.get(
            "structured_result"
        )

        if not isinstance(
            structured_result,
            dict,
        ):
            return EvaluationCheck(
                name="structured_provenance",
                passed=False,
                weight=2.0,
                details=(
                    "The structured result is missing."
                ),
            )

        if (
            structured_result.get("status")
            != "success"
        ):
            return EvaluationCheck(
                name="structured_provenance",
                passed=True,
                weight=2.0,
                details=(
                    "Structured retrieval did not claim "
                    "successful data retrieval."
                ),
            )

        data = structured_result.get(
            "data",
            {},
        )

        query_id = structured_result.get(
            "query_id"
        )

        source = (
            data.get("source")
            if isinstance(data, dict)
            else None
        )

        referenced_tables = (
            data.get(
                "referenced_tables",
                [],
            )
            if isinstance(data, dict)
            else []
        )

        passed = bool(
            query_id
            and source
            and referenced_tables
        )

        return EvaluationCheck(
            name="structured_provenance",
            passed=passed,
            weight=2.0,
            details=(
                "Successful structured retrieval should "
                "include query ID, source, and referenced tables."
            ),
        )

    @staticmethod
    def _check_document_citations(
        response: dict[str, Any],
    ) -> EvaluationCheck:
        document_result = response.get(
            "document_result"
        )

        if not isinstance(
            document_result,
            dict,
        ):
            return EvaluationCheck(
                name="document_citations",
                passed=False,
                weight=2.0,
                details=(
                    "The document result is missing."
                ),
            )

        if (
            document_result.get("status")
            != "success"
        ):
            return EvaluationCheck(
                name="document_citations",
                passed=True,
                weight=2.0,
                details=(
                    "Document retrieval did not claim "
                    "successful evidence retrieval."
                ),
            )

        evidence = document_result.get(
            "evidence",
            [],
        )

        citations = response.get(
            "citations",
            [],
        )

        if not evidence:
            return EvaluationCheck(
                name="document_citations",
                passed=True,
                weight=2.0,
                details=(
                    "No document evidence was returned, "
                    "so citations were not required."
                ),
            )

        evidence_has_citations = all(
            isinstance(item, dict)
            and bool(
                item.get("citation")
            )
            for item in evidence
        )

        return EvaluationCheck(
            name="document_citations",
            passed=(
                evidence_has_citations
                and bool(citations)
            ),
            weight=2.0,
            details=(
                "Every returned document chunk must include "
                "a citation, and final citations must be present."
            ),
        )

    @staticmethod
    def _check_analysis_result(
        response: dict[str, Any],
    ) -> EvaluationCheck:
        analysis_result = response.get(
            "analysis_result"
        )

        passed = (
            isinstance(
                analysis_result,
                dict,
            )
            and bool(
                analysis_result.get(
                    "operation"
                )
            )
            and bool(
                analysis_result.get(
                    "message"
                )
            )
        )

        return EvaluationCheck(
            name="analysis_result",
            passed=passed,
            weight=1.5,
            details=(
                "Coding routes must include an operation "
                "and analytical result message."
            ),
        )

    @classmethod
    def _check_internet_sources(
        cls,
        response: dict[str, Any],
    ) -> EvaluationCheck:
        internet_result = response.get(
            "internet_result"
        )

        if not isinstance(
            internet_result,
            dict,
        ):
            return EvaluationCheck(
                name="internet_sources",
                passed=False,
                weight=2.0,
                details=(
                    "The internet result is missing."
                ),
            )

        if (
            internet_result.get("status")
            != "success"
        ):
            return EvaluationCheck(
                name="internet_sources",
                passed=True,
                weight=2.0,
                details=(
                    "Internet retrieval did not claim "
                    "successful source retrieval."
                ),
            )

        sources = internet_result.get(
            "sources",
            [],
        )

        retrieved_at = internet_result.get(
            "retrieved_at_utc"
        )

        valid_sources = bool(sources) and all(
            isinstance(source, dict)
            and cls._is_safe_url(
                str(
                    source.get(
                        "url",
                        "",
                    )
                )
            )
            for source in sources
        )

        return EvaluationCheck(
            name="internet_sources",
            passed=(
                valid_sources
                and bool(retrieved_at)
            ),
            weight=2.0,
            details=(
                "Internet answers must include safe source URLs "
                "and an external retrieval timestamp."
            ),
        )

    @staticmethod
    def _is_safe_url(
        value: str,
    ) -> bool:
        try:
            parsed = urlparse(value)
        except ValueError:
            return False

        return (
            parsed.scheme
            in {
                "http",
                "https",
            }
            and bool(parsed.netloc)
        )