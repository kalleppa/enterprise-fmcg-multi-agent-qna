from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from src.agents.coding_agent import (
    CodingAnalysisAgent,
)
from src.agents.document_agent import (
    DocumentRetrievalAgent,
)
from src.agents.router import (
    IntentRouter,
    RouteDecision,
)
from src.agents.structured_agent import (
    StructuredDataAgent,
)


OrchestratorStatus = Literal[
    "success",
    "clarification",
    "unsupported",
    "blocked",
    "error",
]


@dataclass(frozen=True)
class AnalysisPlan:
    """Controlled analysis selected for a coding request."""

    operation: str
    x_column: str | None = None
    y_column: str | None = None
    second_y_column: str | None = None
    chart_type: str | None = None
    aggregation: str = "sum"
    title: str | None = None


@dataclass
class OrchestratorResponse:
    """Final standardized response from the main agent."""

    status: OrchestratorStatus
    route: str
    answer: str
    route_decision: dict[str, Any]
    structured_result: dict[str, Any] | None = None
    document_result: dict[str, Any] | None = None
    analysis_result: dict[str, Any] | None = None
    citations: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    follow_up_suggestions: list[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "net_revenue_inr": (
        "net revenue",
        "revenue",
        "sales value",
    ),
    "gross_revenue_inr": (
        "gross revenue",
    ),
    "units_sold": (
        "units sold",
        "unit sales",
        "sales volume",
        "volume",
    ),
    "gross_margin_inr": (
        "gross margin value",
        "margin value",
    ),
    "gross_margin_pct": (
        "gross margin percentage",
        "gross margin percent",
        "margin percentage",
        "margin percent",
        "margin %",
    ),
    "discount_inr": (
        "discount",
    ),
    "stockout_days": (
        "stockout days",
        "stockouts",
        "stockout",
    ),
    "closing_stock_units": (
        "closing stock",
        "closing inventory",
    ),
    "promotion_spend_inr": (
        "promotion spend",
        "campaign spend",
        "marketing spend",
    ),
    "planned_sales_lift_pct": (
        "planned sales lift",
        "planned lift",
        "target lift",
    ),
    "actual_sales_lift_pct": (
        "actual sales lift",
        "actual lift",
        "sales lift",
    ),
}


DIMENSION_ALIASES: dict[str, tuple[str, ...]] = {
    "month": (
        "month",
        "monthly",
        "over time",
        "trend",
    ),
    "quarter": (
        "quarter",
        "quarterly",
    ),
    "year": (
        "year",
        "yearly",
    ),
    "region": (
        "region",
        "regional",
    ),
    "state": (
        "state",
    ),
    "brand": (
        "brand",
    ),
    "product_name": (
        "product",
    ),
    "sku_id": (
        "sku",
    ),
    "channel": (
        "channel",
    ),
    "campaign_name": (
        "campaign",
    ),
}


class EnterpriseQnAOrchestrator:
    """Main agent coordinating all specialist agents."""

    def __init__(
        self,
        router: IntentRouter | None = None,
        structured_agent: StructuredDataAgent | None = None,
        document_agent: DocumentRetrievalAgent | None = None,
        coding_agent: CodingAnalysisAgent | None = None,
    ) -> None:
        self.router = router or IntentRouter()

        self.structured_agent = (
            structured_agent
            or StructuredDataAgent()
        )

        self.document_agent = (
            document_agent
            or DocumentRetrievalAgent()
        )

        self.coding_agent = (
            coding_agent
            or CodingAnalysisAgent()
        )

    def answer(
        self,
        question: str,
    ) -> OrchestratorResponse:
        """Route and execute one user request."""

        decision = self.router.route(question)

        if decision.intent == "greeting":
            return self._greeting_response(decision)

        if decision.intent == "capability":
            return self._capability_response(decision)

        if decision.intent == "metadata":
            return self._handle_metadata(
                question=question,
                decision=decision,
            )

        if decision.intent == "structured":
            return self._handle_structured(
                question=question,
                decision=decision,
            )

        if decision.intent == "document":
            return self._handle_document(
                question=question,
                decision=decision,
            )

        if decision.intent == "hybrid":
            return self._handle_hybrid(
                question=question,
                decision=decision,
            )

        if decision.intent == "coding":
            return self._handle_coding(
                question=question,
                decision=decision,
            )

        if decision.intent == "internet":
            return OrchestratorResponse(
                status="unsupported",
                route="internet",
                answer=(
                    "Internet search was requested, but the "
                    "internet-search provider has not yet been "
                    "configured in this prototype."
                ),
                route_decision=decision.to_dict(),
                limitations=[
                    "Only internal structured data, enterprise "
                    "documents, and controlled analytics are "
                    "currently available."
                ],
                follow_up_suggestions=[
                    "Ask the same question using internal data.",
                ],
            )

        return OrchestratorResponse(
            status="unsupported",
            route="unsupported",
            answer=(
                "I could not map this request to a supported "
                "FMCG business capability."
            ),
            route_decision=decision.to_dict(),
            limitations=[
                "The prototype supports FMCG structured data, "
                "documents, hybrid questions, and controlled "
                "analysis."
            ],
            follow_up_suggestions=[
                "Ask what KPIs or documents are available.",
            ],
        )

    def _handle_metadata(
        self,
        question: str,
        decision: RouteDecision,
    ) -> OrchestratorResponse:
        normalized = question.lower()

        if (
            "document" in normalized
            or "report" in normalized
        ):
            result = self.document_agent.answer(
                question
            )

            return OrchestratorResponse(
                status=self._map_status(
                    result.status
                ),
                route="metadata",
                answer=result.message,
                route_decision=decision.to_dict(),
                document_result=result.to_dict(),
                citations=result.citations,
                limitations=result.limitations,
            )

        result = self.structured_agent.answer(
            question
        )

        return OrchestratorResponse(
            status=self._map_status(
                result.status
            ),
            route="metadata",
            answer=result.message,
            route_decision=decision.to_dict(),
            structured_result=result.to_dict(),
            assumptions=result.assumptions,
        )

    def _handle_structured(
        self,
        question: str,
        decision: RouteDecision,
    ) -> OrchestratorResponse:
        result = self.structured_agent.answer(
            question
        )

        answer = result.message

        if result.status == "success":
            answer = self._format_structured_answer(
                result.to_dict()
            )

        return OrchestratorResponse(
            status=self._map_status(
                result.status
            ),
            route="structured",
            answer=answer,
            route_decision=decision.to_dict(),
            structured_result=result.to_dict(),
            assumptions=result.assumptions,
            follow_up_suggestions=(
                self._structured_follow_ups(
                    question
                )
                if result.status == "success"
                else []
            ),
        )

    def _handle_document(
        self,
        question: str,
        decision: RouteDecision,
    ) -> OrchestratorResponse:
        result = self.document_agent.answer(
            question=question,
            top_k=5,
        )

        answer = result.message

        if result.status == "success":
            answer = self._format_document_answer(
                result.to_dict()
            )

        return OrchestratorResponse(
            status=self._map_status(
                result.status
            ),
            route="document",
            answer=answer,
            route_decision=decision.to_dict(),
            document_result=result.to_dict(),
            citations=result.citations,
            limitations=result.limitations,
            follow_up_suggestions=(
                [
                    "Compare this document evidence with "
                    "structured performance data."
                ]
                if result.evidence
                else []
            ),
        )

    def _handle_hybrid(
        self,
        question: str,
        decision: RouteDecision,
    ) -> OrchestratorResponse:
        with ThreadPoolExecutor(
            max_workers=2
        ) as executor:
            structured_future = executor.submit(
                self.structured_agent.answer,
                question,
            )

            document_future = executor.submit(
                self.document_agent.answer,
                question,
                5,
            )

            structured_result = (
                structured_future.result()
            )

            document_result = (
                document_future.result()
            )

        successful_sources = sum(
            [
                structured_result.status
                == "success",
                document_result.status
                == "success",
            ]
        )

        if successful_sources == 0:
            return OrchestratorResponse(
                status="error",
                route="hybrid",
                answer=(
                    "Neither structured retrieval nor document "
                    "retrieval produced usable evidence."
                ),
                route_decision=decision.to_dict(),
                structured_result=(
                    structured_result.to_dict()
                ),
                document_result=(
                    document_result.to_dict()
                ),
                limitations=(
                    document_result.limitations
                ),
            )

        answer = self._format_hybrid_answer(
            structured_result=(
                structured_result.to_dict()
            ),
            document_result=(
                document_result.to_dict()
            ),
        )

        limitations = list(
            document_result.limitations
        )

        if structured_result.status != "success":
            limitations.append(
                "Structured retrieval did not produce "
                "a successful result."
            )

        if document_result.status != "success":
            limitations.append(
                "Document retrieval did not produce "
                "a successful result."
            )

        return OrchestratorResponse(
            status="success",
            route="hybrid",
            answer=answer,
            route_decision=decision.to_dict(),
            structured_result=(
                structured_result.to_dict()
            ),
            document_result=(
                document_result.to_dict()
            ),
            citations=document_result.citations,
            assumptions=structured_result.assumptions,
            limitations=limitations,
            follow_up_suggestions=[
                "Show the monthly trend for the same period.",
                "Compare the result with another region.",
            ],
        )

    def _handle_coding(
        self,
        question: str,
        decision: RouteDecision,
    ) -> OrchestratorResponse:
        plan = self._create_analysis_plan(
            question
        )

        if plan is None:
            return OrchestratorResponse(
                status="clarification",
                route="coding",
                answer=(
                    "Please specify the analytical operation "
                    "and columns, such as: plot net revenue by "
                    "region, or calculate correlation between "
                    "promotion spend and actual sales lift."
                ),
                route_decision=decision.to_dict(),
            )

        structured_question = (
            self._prepare_structured_question(
                question=question,
                plan=plan,
            )
        )

        structured_result = (
            self.structured_agent.answer(
                structured_question
            )
        )

        if structured_result.status != "success":
            return OrchestratorResponse(
                status=self._map_status(
                    structured_result.status
                ),
                route="coding",
                answer=structured_result.message,
                route_decision=decision.to_dict(),
                structured_result=(
                    structured_result.to_dict()
                ),
                assumptions=(
                    structured_result.assumptions
                ),
            )

        rows = structured_result.data.get(
            "rows",
            [],
        )

        analysis_result = self.coding_agent.analyze(
            rows=rows,
            operation=plan.operation,
            x_column=plan.x_column,
            y_column=plan.y_column,
            second_y_column=(
                plan.second_y_column
            ),
            chart_type=plan.chart_type,
            aggregation=plan.aggregation,
            title=plan.title,
        )

        answer = analysis_result.message

        if analysis_result.status == "success":
            answer = self._format_analysis_answer(
                analysis_result.to_dict()
            )

        return OrchestratorResponse(
            status=self._map_status(
                analysis_result.status
            ),
            route="coding",
            answer=answer,
            route_decision=decision.to_dict(),
            structured_result=(
                structured_result.to_dict()
            ),
            analysis_result=(
                analysis_result.to_dict()
            ),
            assumptions=(
                structured_result.assumptions
                + analysis_result.assumptions
            ),
            follow_up_suggestions=(
                [
                    "Break the analysis down by another "
                    "business dimension."
                ]
                if analysis_result.status == "success"
                else []
            ),
        )

    def _create_analysis_plan(
        self,
        question: str,
    ) -> AnalysisPlan | None:
        normalized = question.lower()

        metrics = self._detect_metrics(
            normalized
        )

        if "correlation" in normalized:
            if len(metrics) < 2:
                return None

            return AnalysisPlan(
                operation="correlation",
                y_column=metrics[0],
                second_y_column=metrics[1],
            )

        if (
            "percentage change" in normalized
            or "percent change" in normalized
        ):
            if not metrics:
                return None

            return AnalysisPlan(
                operation="percentage_change",
                x_column=self._detect_dimension(
                    normalized
                )
                or "month",
                y_column=metrics[0],
            )

        if (
            "summary statistics" in normalized
            or "statistical summary" in normalized
        ):
            return AnalysisPlan(
                operation="summary"
            )

        if any(
            term in normalized
            for term in (
                "plot",
                "chart",
                "graph",
            )
        ):
            if not metrics:
                return None

            x_column = (
                self._detect_dimension(
                    normalized
                )
            )

            if not x_column:
                return None

            chart_type = (
                "line"
                if any(
                    term in normalized
                    for term in (
                        "line",
                        "trend",
                        "over time",
                        "monthly",
                    )
                )
                else "bar"
            )

            return AnalysisPlan(
                operation="chart",
                x_column=x_column,
                y_column=metrics[0],
                chart_type=chart_type,
                aggregation="sum",
                title=(
                    f"{self._humanize(metrics[0])} "
                    f"by {self._humanize(x_column)}"
                ),
            )

        return None

    def _prepare_structured_question(
        self,
        question: str,
        plan: AnalysisPlan,
    ) -> str:
        additions: list[str] = []

        if plan.operation == "correlation":
            additions.append(
                "by campaign and state and channel"
            )

        elif plan.x_column:
            additions.append(
                f"by {self._humanize(plan.x_column)}"
            )

        return " ".join(
            [
                question,
                *additions,
            ]
        )

    @staticmethod
    def _detect_metrics(
        question: str,
    ) -> list[str]:
        matches: list[
            tuple[int, str]
        ] = []

        for metric, aliases in (
            METRIC_ALIASES.items()
        ):
            positions = [
                question.find(alias)
                for alias in aliases
                if question.find(alias) >= 0
            ]

            if positions:
                matches.append(
                    (
                        min(positions),
                        metric,
                    )
                )

        matches.sort(
            key=lambda item: item[0]
        )

        detected = [
            metric
            for _, metric in matches
        ]

        if (
            "gross_revenue_inr" in detected
            and "net_revenue_inr" in detected
            and "net revenue" not in question
        ):
            detected.remove(
                "net_revenue_inr"
            )

        return list(
            dict.fromkeys(detected)
        )

    @staticmethod
    def _detect_dimension(
        question: str,
    ) -> str | None:
        for dimension, aliases in (
            DIMENSION_ALIASES.items()
        ):
            if any(
                alias in question
                for alias in aliases
            ):
                return dimension

        return None

    @staticmethod
    def _format_structured_answer(
        result: dict[str, Any],
    ) -> str:
        data = result.get("data", {})
        rows = data.get("rows", [])

        if not rows:
            return result.get(
                "message",
                "No data was returned.",
            )

        headers = list(rows[0].keys())

        lines = [
            result.get(
                "message",
                "Structured result",
            ),
            "",
            "| "
            + " | ".join(headers)
            + " |",
            "| "
            + " | ".join(
                "---"
                for _ in headers
            )
            + " |",
        ]

        for row in rows[:10]:
            lines.append(
                "| "
                + " | ".join(
                    str(row.get(header, ""))
                    for header in headers
                )
                + " |"
            )

        lines.extend(
            [
                "",
                (
                    f"Source: {data.get('source')} · "
                    f"Query ID: {result.get('query_id')}"
                ),
            ]
        )

        return "\n".join(lines)

    @staticmethod
    def _format_document_answer(
        result: dict[str, Any],
    ) -> str:
        evidence = result.get(
            "evidence",
            [],
        )

        if not evidence:
            return result.get(
                "message",
                "No evidence was found.",
            )

        lines = [
            result.get(
                "message",
                "Document evidence",
            )
        ]

        for index, item in enumerate(
            evidence,
            start=1,
        ):
            lines.extend(
                [
                    "",
                    (
                        f"{index}. **{item['title']} — "
                        f"{item['section']}**"
                    ),
                    item["snippet"],
                    f"Citation: {item['citation']}",
                ]
            )

        return "\n".join(lines)

    @staticmethod
    def _format_hybrid_answer(
        structured_result: dict[str, Any],
        document_result: dict[str, Any],
    ) -> str:
        lines = [
            "## Structured evidence",
            "",
        ]

        structured_data = (
            structured_result.get(
                "data",
                {},
            )
        )

        rows = structured_data.get(
            "rows",
            [],
        )

        if rows:
            headers = list(
                rows[0].keys()
            )

            lines.extend(
                [
                    "| "
                    + " | ".join(headers)
                    + " |",
                    "| "
                    + " | ".join(
                        "---"
                        for _ in headers
                    )
                    + " |",
                ]
            )

            for row in rows[:10]:
                lines.append(
                    "| "
                    + " | ".join(
                        str(
                            row.get(
                                header,
                                "",
                            )
                        )
                        for header in headers
                    )
                    + " |"
                )

        else:
            lines.append(
                structured_result.get(
                    "message",
                    "No structured result.",
                )
            )

        lines.extend(
            [
                "",
                "## Document explanation",
            ]
        )

        evidence = document_result.get(
            "evidence",
            [],
        )

        for item in evidence[:5]:
            lines.extend(
                [
                    "",
                    (
                        f"- **{item['title']} — "
                        f"{item['section']}**: "
                        f"{item['snippet']}"
                    ),
                    f"  Citation: {item['citation']}",
                ]
            )

        return "\n".join(lines)

    @staticmethod
    def _format_analysis_answer(
        result: dict[str, Any],
    ) -> str:
        lines = [
            result.get(
                "message",
                "Analysis completed.",
            )
        ]

        analysis_results = result.get(
            "results",
            {},
        )

        if analysis_results:
            lines.append("")

            for key, value in (
                analysis_results.items()
            ):
                lines.append(
                    f"- {EnterpriseQnAOrchestrator._humanize(key)}: "
                    f"{value}"
                )

        chart_path = result.get(
            "chart_path"
        )

        if chart_path:
            lines.extend(
                [
                    "",
                    f"Chart: `{chart_path}`",
                ]
            )

        return "\n".join(lines)

    @staticmethod
    def _structured_follow_ups(
        question: str,
    ) -> list[str]:
        suggestions = [
            "Break this result down by channel.",
            "Compare it with the previous period.",
        ]

        if "region" in question.lower():
            suggestions[0] = (
                "Drill down from region to state."
            )

        return suggestions

    @staticmethod
    def _greeting_response(
        decision: RouteDecision,
    ) -> OrchestratorResponse:
        return OrchestratorResponse(
            status="success",
            route="greeting",
            answer=(
                "Hello! I can help with FMCG sales, revenue, "
                "margin, inventory, campaigns, enterprise "
                "documents, and controlled data analysis."
            ),
            route_decision=decision.to_dict(),
            follow_up_suggestions=[
                "Ask what KPIs and datasets are available.",
            ],
        )

    @staticmethod
    def _capability_response(
        decision: RouteDecision,
    ) -> OrchestratorResponse:
        return OrchestratorResponse(
            status="success",
            route="capability",
            answer=(
                "I support structured FMCG analytics, "
                "enterprise document retrieval with citations, "
                "hybrid data-and-document questions, metadata "
                "discovery, and controlled calculations and "
                "charts. Internet search will be added as a "
                "separate specialist agent."
            ),
            route_decision=decision.to_dict(),
            follow_up_suggestions=[
                "Ask what KPIs are available.",
                "Ask what documents are available.",
            ],
        )

    @staticmethod
    def _map_status(
        status: str,
    ) -> OrchestratorStatus:
        mapping: dict[str, OrchestratorStatus] = {
            "success": "success",
            "clarification": "clarification",
            "unsupported": "unsupported",
            "blocked": "blocked",
            "error": "error",
        }

        return mapping.get(
            status,
            "error",
        )

    @staticmethod
    def _humanize(value: str) -> str:
        return re.sub(
            r"\s+",
            " ",
            value.replace("_", " "),
        ).strip().title()


def main() -> None:
    """Run a local orchestrator demonstration."""

    orchestrator = EnterpriseQnAOrchestrator()

    questions = [
        "Hello",
        "What KPIs are available?",
        (
            "Show net revenue and gross margin percentage "
            "by region for Q2 2025"
        ),
        (
            "What risks were identified in the "
            "Sparkle Summer campaign brief?"
        ),
        (
            "Did Sparkle Summer achieve its planned "
            "sales lift, and why?"
        ),
        (
            "Plot actual sales lift by campaign"
        ),
        (
            "Calculate correlation between promotion spend "
            "and actual sales lift"
        ),
        (
            "Search the internet for current FMCG trends"
        ),
    ]

    for question in questions:
        response = orchestrator.answer(
            question
        )

        print("=" * 80)
        print("Question:", question)
        print("Route:", response.route)
        print("Status:", response.status)
        print()
        print(response.answer)
        print()


if __name__ == "__main__":
    main()