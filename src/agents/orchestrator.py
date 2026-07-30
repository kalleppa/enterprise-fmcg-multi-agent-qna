from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from src.agents.coding_agent import CodingAnalysisAgent
from src.agents.document_agent import DocumentRetrievalAgent
from src.agents.internet_agent import InternetSearchAgent
from src.agents.router import IntentRouter, RouteDecision
from src.agents.structured_agent import StructuredDataAgent


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
    internet_result: dict[str, Any] | None = None

    citations: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    follow_up_suggestions: list[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert the response to a serializable dictionary."""

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
        "gross margin inr",
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
        "discount value",
    ),
    "stockout_days": (
        "stockout days",
        "stockouts",
        "stockout",
        "out of stock",
    ),
    "closing_stock_units": (
        "closing stock",
        "closing inventory",
        "inventory units",
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
        "sales lift target",
    ),
    "actual_sales_lift_pct": (
        "actual sales lift",
        "actual lift",
        "sales lift",
    ),
    "lift_variance_pct_points": (
        "lift variance",
        "target variance",
    ),
}


DIMENSION_ALIASES: dict[str, tuple[str, ...]] = {
    "month": (
        "by month",
        "monthly",
        "over time",
        "trend",
        "month",
    ),
    "quarter": (
        "by quarter",
        "quarterly",
        "quarter",
    ),
    "year": (
        "by year",
        "yearly",
        "year",
    ),
    "region": (
        "by region",
        "regional",
        "region",
    ),
    "state": (
        "by state",
        "state",
    ),
    "brand": (
        "by brand",
        "brand",
    ),
    "product_name": (
        "by product",
        "product",
    ),
    "sku_id": (
        "by sku",
        "sku",
    ),
    "channel": (
        "by channel",
        "channel",
    ),
    "distributor_name": (
        "by distributor",
        "distributor",
    ),
    "campaign_name": (
        "by campaign",
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
        internet_agent: InternetSearchAgent | None = None,
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

        self.internet_agent = (
            internet_agent
            or InternetSearchAgent()
        )

    def answer(
        self,
        question: str,
    ) -> OrchestratorResponse:
        """
        Route and execute one user request.

        The orchestrator does not directly access databases,
        documents, external search providers, or analytical
        libraries. It delegates those responsibilities to
        specialist agents.
        """

        decision = self.router.route(question)

        if decision.intent == "greeting":
            return self._greeting_response(
                decision
            )

        if decision.intent == "capability":
            return self._capability_response(
                decision
            )

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

        if decision.intent == "coding":
            return self._handle_coding(
                question=question,
                decision=decision,
            )

        if decision.intent == "internet":
            return self._handle_internet(
                question=question,
                decision=decision,
            )

        return self._unsupported_response(
            decision=decision
        )

    def _handle_metadata(
        self,
        question: str,
        decision: RouteDecision,
    ) -> OrchestratorResponse:
        """Handle dataset, KPI, dimension, period, and document metadata."""

        normalized_question = question.lower()

        document_metadata_terms = (
            "document",
            "documents",
            "report",
            "reports",
        )

        if any(
            term in normalized_question
            for term in document_metadata_terms
        ):
            result = self.document_agent.answer(
                question
            )

            result_dict = self._to_dict(
                result
            )

            return OrchestratorResponse(
                status=self._map_status(
                    result.status
                ),
                route="metadata",
                answer=result.message,
                route_decision=decision.to_dict(),
                document_result=result_dict,
                citations=list(
                    getattr(
                        result,
                        "citations",
                        [],
                    )
                ),
                limitations=list(
                    getattr(
                        result,
                        "limitations",
                        [],
                    )
                ),
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
            structured_result=self._to_dict(
                result
            ),
            assumptions=list(
                getattr(
                    result,
                    "assumptions",
                    [],
                )
            ),
        )

    def _handle_structured(
        self,
        question: str,
        decision: RouteDecision,
    ) -> OrchestratorResponse:
        """Run structured-data retrieval."""

        result = self.structured_agent.answer(
            question
        )

        result_dict = self._to_dict(
            result
        )

        answer = result.message

        if result.status == "success":
            answer = self._format_structured_answer(
                result_dict
            )

        return OrchestratorResponse(
            status=self._map_status(
                result.status
            ),
            route="structured",
            answer=answer,
            route_decision=decision.to_dict(),
            structured_result=result_dict,
            assumptions=list(
                getattr(
                    result,
                    "assumptions",
                    [],
                )
            ),
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
        """Run document retrieval."""

        result = self.document_agent.answer(
            question=question,
            top_k=5,
        )

        result_dict = self._to_dict(
            result
        )

        answer = result.message

        if result.status == "success":
            answer = self._format_document_answer(
                result_dict
            )

        evidence = getattr(
            result,
            "evidence",
            [],
        )

        return OrchestratorResponse(
            status=self._map_status(
                result.status
            ),
            route="document",
            answer=answer,
            route_decision=decision.to_dict(),
            document_result=result_dict,
            citations=list(
                getattr(
                    result,
                    "citations",
                    [],
                )
            ),
            limitations=list(
                getattr(
                    result,
                    "limitations",
                    [],
                )
            ),
            follow_up_suggestions=(
                [
                    (
                        "Compare this document evidence with "
                        "structured performance data."
                    )
                ]
                if evidence
                else []
            ),
        )

    def _handle_hybrid(
        self,
        question: str,
        decision: RouteDecision,
    ) -> OrchestratorResponse:
        """
        Run structured and document retrieval concurrently.

        Independent retrieval operations are executed in parallel
        to reduce hybrid-request latency.
        """

        try:
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

        except Exception as error:
            return OrchestratorResponse(
                status="error",
                route="hybrid",
                answer=(
                    "The hybrid retrieval workflow failed "
                    "before all specialist agents completed."
                ),
                route_decision=decision.to_dict(),
                limitations=[
                    str(error)
                ],
            )

        structured_dict = self._to_dict(
            structured_result
        )

        document_dict = self._to_dict(
            document_result
        )

        structured_success = (
            structured_result.status
            == "success"
        )

        document_success = (
            document_result.status
            == "success"
        )

        if (
            not structured_success
            and not document_success
        ):
            return OrchestratorResponse(
                status=self._combined_failure_status(
                    structured_result.status,
                    document_result.status,
                ),
                route="hybrid",
                answer=(
                    "Neither structured retrieval nor document "
                    "retrieval produced usable evidence."
                ),
                route_decision=decision.to_dict(),
                structured_result=structured_dict,
                document_result=document_dict,
                limitations=list(
                    getattr(
                        document_result,
                        "limitations",
                        [],
                    )
                ),
            )

        answer = self._format_hybrid_answer(
            structured_result=structured_dict,
            document_result=document_dict,
        )

        limitations = list(
            getattr(
                document_result,
                "limitations",
                [],
            )
        )

        if not structured_success:
            limitations.append(
                "Structured retrieval did not produce "
                "a successful result."
            )

        if not document_success:
            limitations.append(
                "Document retrieval did not produce "
                "a successful result."
            )

        return OrchestratorResponse(
            status="success",
            route="hybrid",
            answer=answer,
            route_decision=decision.to_dict(),
            structured_result=structured_dict,
            document_result=document_dict,
            citations=list(
                getattr(
                    document_result,
                    "citations",
                    [],
                )
            ),
            assumptions=list(
                getattr(
                    structured_result,
                    "assumptions",
                    [],
                )
            ),
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
        """
        Retrieve approved structured data and pass it to
        the controlled coding agent.
        """

        plan = self._create_analysis_plan(
            question
        )

        if plan is None:
            return OrchestratorResponse(
                status="clarification",
                route="coding",
                answer=(
                    "Please specify the analytical operation "
                    "and columns. For example: plot net revenue "
                    "by region, or calculate correlation between "
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

        structured_dict = self._to_dict(
            structured_result
        )

        if structured_result.status != "success":
            return OrchestratorResponse(
                status=self._map_status(
                    structured_result.status
                ),
                route="coding",
                answer=structured_result.message,
                route_decision=decision.to_dict(),
                structured_result=structured_dict,
                assumptions=list(
                    getattr(
                        structured_result,
                        "assumptions",
                        [],
                    )
                ),
            )

        structured_data = getattr(
            structured_result,
            "data",
            {},
        )

        rows = structured_data.get(
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

        analysis_dict = self._to_dict(
            analysis_result
        )

        answer = analysis_result.message

        if analysis_result.status == "success":
            answer = self._format_analysis_answer(
                analysis_dict
            )

        structured_assumptions = list(
            getattr(
                structured_result,
                "assumptions",
                [],
            )
        )

        analysis_assumptions = list(
            getattr(
                analysis_result,
                "assumptions",
                [],
            )
        )

        return OrchestratorResponse(
            status=self._map_status(
                analysis_result.status
            ),
            route="coding",
            answer=answer,
            route_decision=decision.to_dict(),
            structured_result=structured_dict,
            analysis_result=analysis_dict,
            assumptions=(
                structured_assumptions
                + analysis_assumptions
            ),
            follow_up_suggestions=(
                [
                    (
                        "Break the analysis down by another "
                        "business dimension."
                    )
                ]
                if analysis_result.status
                == "success"
                else []
            ),
        )

    def _handle_internet(
        self,
        question: str,
        decision: RouteDecision,
    ) -> OrchestratorResponse:
        """Run current external-information retrieval."""

        normalized_question = (
            question.lower()
        )

        is_current_request = any(
            phrase in normalized_question
            for phrase in (
                "latest",
                "current",
                "today",
                "news",
                "recent",
            )
        )

        topic = (
            "news"
            if is_current_request
            else "general"
        )

        time_range = (
            "month"
            if is_current_request
            else None
        )

        result = self.internet_agent.search(
            query=question,
            topic=topic,
            max_results=5,
            search_depth="basic",
            time_range=time_range,
        )

        result_dict = self._to_dict(
            result
        )

        if result.status == "unavailable":
            return OrchestratorResponse(
                status="unsupported",
                route="internet",
                answer=result.message,
                route_decision=decision.to_dict(),
                internet_result=result_dict,
                limitations=list(
                    getattr(
                        result,
                        "limitations",
                        [],
                    )
                ),
            )

        if result.status != "success":
            return OrchestratorResponse(
                status=self._map_status(
                    result.status
                ),
                route="internet",
                answer=result.message,
                route_decision=decision.to_dict(),
                internet_result=result_dict,
                limitations=list(
                    getattr(
                        result,
                        "limitations",
                        [],
                    )
                ),
            )

        answer = self._format_internet_answer(
            result_dict
        )

        return OrchestratorResponse(
            status="success",
            route="internet",
            answer=answer,
            route_decision=decision.to_dict(),
            internet_result=result_dict,
            citations=list(
                getattr(
                    result,
                    "citations",
                    [],
                )
            ),
            limitations=list(
                getattr(
                    result,
                    "limitations",
                    [],
                )
            ),
            follow_up_suggestions=[
                (
                    "Compare the external findings with "
                    "internal FMCG performance."
                )
            ],
        )

    def _create_analysis_plan(
        self,
        question: str,
    ) -> AnalysisPlan | None:
        """Create an allowlisted analytical plan."""

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

        if any(
            phrase in normalized
            for phrase in (
                "percentage change",
                "percent change",
            )
        ):
            if not metrics:
                return None

            return AnalysisPlan(
                operation="percentage_change",
                x_column=(
                    self._detect_dimension(
                        normalized
                    )
                    or "month"
                ),
                y_column=metrics[0],
            )

        if any(
            phrase in normalized
            for phrase in (
                "summary statistics",
                "statistical summary",
                "descriptive statistics",
            )
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

            x_column = self._detect_dimension(
                normalized
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
        """
        Add dimensions needed to create the analysis dataset.

        The original user question is retained so that entity,
        period, and KPI filters remain available.
        """

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
        """Detect metrics in the order they appear."""

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

        # Prevent "revenue" inside "gross revenue"
        # from becoming both gross and net revenue.
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
        """Detect the requested analytical dimension."""

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
        """Format structured rows as a Markdown table."""

        data = result.get(
            "data",
            {},
        )

        rows = data.get(
            "rows",
            [],
        )

        if not rows:
            return result.get(
                "message",
                "No structured data was returned.",
            )

        headers = list(
            rows[0].keys()
        )

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

        if len(rows) > 10:
            lines.extend(
                [
                    "",
                    (
                        f"Showing 10 of "
                        f"{len(rows)} returned rows."
                    ),
                ]
            )

        source = data.get(
            "source"
        )

        query_id = result.get(
            "query_id"
        )

        if source or query_id:
            lines.extend(
                [
                    "",
                    (
                        f"Source: {source or 'unknown'} · "
                        f"Query ID: {query_id or 'unknown'}"
                    ),
                ]
            )

        return "\n".join(lines)

    @staticmethod
    def _format_document_answer(
        result: dict[str, Any],
    ) -> str:
        """Format document evidence with citations."""

        evidence = result.get(
            "evidence",
            [],
        )

        if not evidence:
            return result.get(
                "message",
                "No document evidence was found.",
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
                        f"{index}. **"
                        f"{item.get('title', 'Unknown document')}"
                        f" — "
                        f"{item.get('section', 'Unknown section')}"
                        f"**"
                    ),
                    item.get(
                        "snippet",
                        item.get(
                            "content",
                            "",
                        ),
                    ),
                    (
                        f"Citation: "
                        f"{item.get('citation', 'Unavailable')}"
                    ),
                ]
            )

        return "\n".join(lines)

    @staticmethod
    def _format_hybrid_answer(
        structured_result: dict[str, Any],
        document_result: dict[str, Any],
    ) -> str:
        """Combine structured evidence and document explanation."""

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
                    "No structured evidence was returned.",
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

        if not evidence:
            lines.extend(
                [
                    "",
                    document_result.get(
                        "message",
                        (
                            "No supporting document evidence "
                            "was returned."
                        ),
                    ),
                ]
            )

        for item in evidence[:5]:
            lines.extend(
                [
                    "",
                    (
                        f"- **"
                        f"{item.get('title', 'Unknown document')}"
                        f" — "
                        f"{item.get('section', 'Unknown section')}"
                        f"**: "
                        f"{item.get('snippet', '')}"
                    ),
                    (
                        f"  Citation: "
                        f"{item.get('citation', 'Unavailable')}"
                    ),
                ]
            )

        return "\n".join(lines)

    @staticmethod
    def _format_analysis_answer(
        result: dict[str, Any],
    ) -> str:
        """Format controlled analysis output."""

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
                    f"- "
                    f"{EnterpriseQnAOrchestrator._humanize(key)}"
                    f": {value}"
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
    def _format_internet_answer(
        result: dict[str, Any],
    ) -> str:
        """Format external search results with retrieval metadata."""

        provider_answer = result.get(
            "answer"
        )

        lines = [
            provider_answer
            or result.get(
                "message",
                "Internet search completed.",
            ),
            "",
            "## External sources",
        ]

        sources = result.get(
            "sources",
            [],
        )

        retrieved_at = result.get(
            "retrieved_at_utc",
            "unknown",
        )

        if not sources:
            lines.extend(
                [
                    "",
                    "No usable external sources were returned.",
                ]
            )

        for source in sources:
            lines.extend(
                [
                    "",
                    (
                        f"{source.get('rank', '')}. "
                        f"**{source.get('title', 'Untitled source')}**"
                    ),
                    source.get(
                        "content",
                        "",
                    ),
                    (
                        f"Source: {source.get('url', '')} · "
                        f"Retrieved: {retrieved_at}"
                    ),
                ]
            )

        return "\n".join(lines)

    @staticmethod
    def _structured_follow_ups(
        question: str,
    ) -> list[str]:
        """Return context-aware follow-up suggestions."""

        suggestions = [
            "Break this result down by channel.",
            "Compare it with the previous period.",
        ]

        if "region" in question.lower():
            suggestions[0] = (
                "Drill down from region to state."
            )

        if "campaign" in question.lower():
            suggestions = [
                (
                    "Compare planned and actual campaign "
                    "performance by state."
                ),
                (
                    "Retrieve the campaign documents that "
                    "explain the result."
                ),
            ]

        return suggestions

    @staticmethod
    def _greeting_response(
        decision: RouteDecision,
    ) -> OrchestratorResponse:
        """Return a deterministic greeting."""

        return OrchestratorResponse(
            status="success",
            route="greeting",
            answer=(
                "Hello! I can help with FMCG sales, revenue, "
                "margin, inventory, campaigns, enterprise "
                "documents, current internet information, "
                "and controlled data analysis."
            ),
            route_decision=decision.to_dict(),
            follow_up_suggestions=[
                (
                    "Ask what KPIs, datasets, and documents "
                    "are available."
                )
            ],
        )

    @staticmethod
    def _capability_response(
        decision: RouteDecision,
    ) -> OrchestratorResponse:
        """Return a deterministic capability introduction."""

        return OrchestratorResponse(
            status="success",
            route="capability",
            answer=(
                "I support structured FMCG analytics, "
                "enterprise document retrieval with citations, "
                "hybrid data-and-document questions, metadata "
                "discovery, current internet search, and "
                "controlled calculations and charts."
            ),
            route_decision=decision.to_dict(),
            follow_up_suggestions=[
                "Ask what KPIs are available.",
                "Ask what documents are available.",
            ],
        )

    @staticmethod
    def _unsupported_response(
        decision: RouteDecision,
    ) -> OrchestratorResponse:
        """Return a graceful out-of-scope response."""

        return OrchestratorResponse(
            status="unsupported",
            route="unsupported",
            answer=(
                "I could not map this request to a supported "
                "FMCG business capability."
            ),
            route_decision=decision.to_dict(),
            limitations=[
                (
                    "The prototype supports structured FMCG "
                    "data, enterprise documents, hybrid "
                    "questions, current internet search, "
                    "and controlled analysis."
                )
            ],
            follow_up_suggestions=[
                (
                    "Ask what KPIs, datasets, or documents "
                    "are available."
                )
            ],
        )

    @staticmethod
    def _combined_failure_status(
        structured_status: str,
        document_status: str,
    ) -> OrchestratorStatus:
        """Select a final status when both hybrid agents fail."""

        statuses = {
            structured_status,
            document_status,
        }

        if "blocked" in statuses:
            return "blocked"

        if "clarification" in statuses:
            return "clarification"

        if "unsupported" in statuses:
            return "unsupported"

        return "error"

    @staticmethod
    def _map_status(
        status: str,
    ) -> OrchestratorStatus:
        """Map specialist-agent statuses to orchestrator statuses."""

        mapping: dict[
            str,
            OrchestratorStatus,
        ] = {
            "success": "success",
            "clarification": "clarification",
            "unsupported": "unsupported",
            "unavailable": "unsupported",
            "blocked": "blocked",
            "error": "error",
        }

        return mapping.get(
            status,
            "error",
        )

    @staticmethod
    def _humanize(
        value: str,
    ) -> str:
        """Convert a technical column name into readable text."""

        return re.sub(
            r"\s+",
            " ",
            value.replace(
                "_",
                " ",
            ),
        ).strip().title()

    @staticmethod
    def _to_dict(
        value: Any,
    ) -> dict[str, Any]:
        """Convert a specialist response to a dictionary."""

        if hasattr(
            value,
            "to_dict",
        ):
            converted = value.to_dict()

            if isinstance(
                converted,
                dict,
            ):
                return converted

        if hasattr(
            value,
            "__dataclass_fields__",
        ):
            converted = asdict(
                value
            )

            if isinstance(
                converted,
                dict,
            ):
                return converted

        raise TypeError(
            "Specialist-agent response must provide "
            "to_dict() or be a dataclass."
        )


def main() -> None:
    """Run a local multi-agent demonstration."""

    orchestrator = EnterpriseQnAOrchestrator()

    questions = [
        "Hello",
        "What can you do?",
        "What KPIs are available?",
        "What documents are available?",
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
        "Plot actual sales lift by campaign",
        (
            "Calculate correlation between promotion spend "
            "and actual sales lift"
        ),
        (
            "Search the internet for the latest FMCG "
            "market trends in India"
        ),
        "Book a flight for tomorrow",
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

        if response.citations:
            print()
            print("Citations:")

            for citation in response.citations:
                print("-", citation)

        if response.assumptions:
            print()
            print("Assumptions:")

            for assumption in response.assumptions:
                print("-", assumption)

        if response.limitations:
            print()
            print("Limitations:")

            for limitation in response.limitations:
                print("-", limitation)

        print()


if __name__ == "__main__":
    main()