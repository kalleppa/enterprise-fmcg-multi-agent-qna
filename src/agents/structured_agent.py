from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from src.database.executor import (
    SQLExecutionError,
    UnsafeSQLQueryError,
    execute_safe_sql,
)
from src.database.metadata import (
    get_available_periods,
    list_datasets,
    list_dimensions,
    list_kpis,
)


AgentStatus = Literal[
    "success",
    "clarification",
    "blocked",
    "unsupported",
    "error",
]


@dataclass(frozen=True)
class PeriodFilter:
    """A period extracted from a user question."""

    year: int
    quarter: str | None = None
    month: int | None = None


@dataclass(frozen=True)
class StructuredQueryPlan:
    """Structured representation of an analytical request."""

    source: str
    metrics: tuple[str, ...]
    dimensions: tuple[str, ...]
    filters: dict[str, str]
    periods: tuple[PeriodFilter, ...]
    assumptions: tuple[str, ...] = ()


@dataclass
class StructuredAgentResponse:
    """Standard response produced by the structured-data agent."""

    status: AgentStatus
    question: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    sql: str | None = None
    query_id: str | None = None
    assumptions: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


METRICS: dict[str, dict[str, Any]] = {
    "net_revenue_inr": {
        "source": "sales",
        "phrases": (
            "net revenue",
            "revenue",
            "sales value",
        ),
        "expression": (
            "ROUND(SUM(net_revenue_inr), 2)"
        ),
        "alias": "net_revenue_inr",
    },
    "gross_revenue_inr": {
        "source": "sales",
        "phrases": (
            "gross revenue",
        ),
        "expression": (
            "ROUND(SUM(gross_revenue_inr), 2)"
        ),
        "alias": "gross_revenue_inr",
    },
    "units_sold": {
        "source": "sales",
        "phrases": (
            "units sold",
            "unit sales",
            "sales volume",
            "volume",
        ),
        "expression": "SUM(units_sold)",
        "alias": "units_sold",
    },
    "gross_margin_inr": {
        "source": "sales",
        "phrases": (
            "gross margin value",
            "margin value",
            "gross margin inr",
        ),
        "expression": (
            "ROUND(SUM(gross_margin_inr), 2)"
        ),
        "alias": "gross_margin_inr",
    },
    "gross_margin_pct": {
        "source": "sales",
        "phrases": (
            "gross margin percentage",
            "gross margin percent",
            "margin percentage",
            "margin percent",
            "margin %",
        ),
        "expression": """
            ROUND(
                SUM(gross_margin_inr)
                / NULLIF(SUM(net_revenue_inr), 0)
                * 100,
                2
            )
        """,
        "alias": "gross_margin_pct",
    },
    "discount_inr": {
        "source": "sales",
        "phrases": (
            "discount",
            "discount value",
        ),
        "expression": (
            "ROUND(SUM(discount_inr), 2)"
        ),
        "alias": "discount_inr",
    },
    "stockout_days": {
        "source": "inventory",
        "phrases": (
            "stockout days",
            "stockouts",
            "stockout",
            "out of stock",
        ),
        "expression": "SUM(stockout_days)",
        "alias": "stockout_days",
    },
    "closing_stock_units": {
        "source": "inventory",
        "phrases": (
            "closing stock",
            "closing inventory",
            "inventory units",
        ),
        "expression": "SUM(closing_stock_units)",
        "alias": "closing_stock_units",
    },
    "promotion_spend_inr": {
        "source": "promotion",
        "phrases": (
            "promotion spend",
            "campaign spend",
            "marketing spend",
        ),
        "expression": (
            "ROUND(SUM(promotion_spend_inr), 2)"
        ),
        "alias": "promotion_spend_inr",
    },
    "planned_sales_lift_pct": {
        "source": "promotion",
        "phrases": (
            "planned sales lift",
            "planned lift",
            "sales lift target",
            "target lift",
        ),
        "expression": (
            "ROUND(AVG(planned_sales_lift_pct), 2)"
        ),
        "alias": "planned_sales_lift_pct",
    },
    "actual_sales_lift_pct": {
        "source": "promotion",
        "phrases": (
            "actual sales lift",
            "actual lift",
            "sales lift",
        ),
        "expression": (
            "ROUND(AVG(actual_sales_lift_pct), 2)"
        ),
        "alias": "actual_sales_lift_pct",
    },
    "lift_variance_pct_points": {
        "source": "promotion",
        "phrases": (
            "lift variance",
            "target variance",
        ),
        "expression": (
            "ROUND(AVG(lift_variance_pct_points), 2)"
        ),
        "alias": "lift_variance_pct_points",
    },
}


DIMENSIONS: dict[str, tuple[str, ...]] = {
    "region": (
        "region",
        "regions",
        "region wise",
        "region-wise",
    ),
    "state": (
        "state",
        "states",
        "state wise",
        "state-wise",
    ),
    "city": (
        "city",
        "cities",
        "city wise",
        "city-wise",
    ),
    "brand": (
        "brand",
        "brands",
        "brand wise",
        "brand-wise",
    ),
    "product_name": (
        "product",
        "products",
        "product wise",
        "product-wise",
    ),
    "sku_id": (
        "sku",
        "skus",
        "sku wise",
        "sku-wise",
    ),
    "channel": (
        "channel",
        "channels",
        "channel wise",
        "channel-wise",
    ),
    "distributor_name": (
        "distributor",
        "distributors",
        "distributor wise",
        "distributor-wise",
    ),
    "campaign_name": (
        "campaign",
        "campaigns",
        "campaign wise",
        "campaign-wise",
    ),
    "month": (
        "month",
        "monthly",
        "by month",
    ),
    "quarter": (
        "quarter",
        "quarterly",
        "by quarter",
    ),
    "year": (
        "year",
        "yearly",
        "by year",
    ),
}


ENTITY_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "brand": {
        "FreshGlow": (
            "freshglow",
            "fresh glow",
        ),
        "PureHome": (
            "purehome",
            "pure home",
        ),
        "NutriBite": (
            "nutribite",
            "nutri bite",
        ),
    },
    "region": {
        "South Region": (
            "south region",
            "southern region",
        ),
        "West Region": (
            "west region",
            "western region",
        ),
        "North Region": (
            "north region",
            "northern region",
        ),
        "East Region": (
            "east region",
            "eastern region",
        ),
    },
    "state": {
        "Karnataka": (
            "karnataka",
            "ka",
        ),
        "Tamil Nadu": (
            "tamil nadu",
            "tn",
        ),
        "Maharashtra": (
            "maharashtra",
            "mh",
        ),
        "Gujarat": (
            "gujarat",
            "gj",
        ),
        "Delhi": (
            "delhi",
            "dl",
        ),
        "Uttar Pradesh": (
            "uttar pradesh",
            "up",
        ),
        "West Bengal": (
            "west bengal",
            "wb",
        ),
        "Odisha": (
            "odisha",
            "orissa",
            "od",
        ),
    },
    "channel": {
        "General Trade": (
            "general trade",
            "gt",
        ),
        "Modern Trade": (
            "modern trade",
            "mt",
        ),
        "E-commerce": (
            "e-commerce",
            "ecommerce",
            "online channel",
            "ec",
        ),
    },
    "campaign_name": {
        "Sparkle Summer 2025": (
            "sparkle summer 2025",
            "sparkle summer",
        ),
        "NutriBite Digital Boost": (
            "nutribite digital boost",
            "digital boost",
        ),
        "PureHome Hygiene Week": (
            "purehome hygiene week",
            "hygiene week",
        ),
    },
    "sku_id": {
        "FG-DW-LEM-500": (
            "fg-dw-lem-500",
            "fg dw lem 500",
        ),
        "FG-DW-LEM-1L": (
            "fg-dw-lem-1l",
            "fg dw lem 1l",
        ),
        "FG-SC-LAV-500": (
            "fg-sc-lav-500",
            "fg sc lav 500",
        ),
        "FG-SC-LAV-1L": (
            "fg-sc-lav-1l",
            "fg sc lav 1l",
        ),
        "PH-HW-ALO-250": (
            "ph-hw-alo-250",
            "ph hw alo 250",
        ),
        "PH-HW-ALO-500": (
            "ph-hw-alo-500",
            "ph hw alo 500",
        ),
        "PH-BW-NEE-500": (
            "ph-bw-nee-500",
            "ph bw nee 500",
        ),
        "PH-HW-ROS-250": (
            "ph-hw-ros-250",
            "ph hw ros 250",
        ),
        "NB-GR-CHO-250": (
            "nb-gr-cho-250",
            "nb gr cho 250",
        ),
        "NB-GR-HON-500": (
            "nb-gr-hon-500",
            "nb gr hon 500",
        ),
        "NB-PB-CHO-6P": (
            "nb-pb-cho-6p",
            "nb pb cho 6p",
        ),
    },
}


MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


SOURCE_VIEWS = {
    "sales": "vw_sales_enriched",
    "inventory": "vw_inventory_enriched",
    "promotion": "vw_promotions_enriched",
}


class StructuredDataAgent:
    """Rule-based first version of the structured-data agent."""

    def answer(
        self,
        question: str,
    ) -> StructuredAgentResponse:
        cleaned_question = question.strip()

        if not cleaned_question:
            return StructuredAgentResponse(
                status="clarification",
                question=question,
                message="Please enter a business question.",
            )

        if self._contains_sql_injection(cleaned_question):
            return StructuredAgentResponse(
                status="blocked",
                question=question,
                message=(
                    "The request contains SQL modification or "
                    "multi-statement patterns and was blocked."
                ),
            )

        metadata_response = self._handle_metadata_question(
            cleaned_question
        )

        if metadata_response is not None:
            return metadata_response

        plan_or_response = self._create_plan(
            cleaned_question
        )

        if isinstance(
            plan_or_response,
            StructuredAgentResponse,
        ):
            return plan_or_response

        plan = plan_or_response
        sql = self._build_sql(plan)

        try:
            result = execute_safe_sql(
                sql=sql,
                max_rows=500,
            )

        except UnsafeSQLQueryError as error:
            return StructuredAgentResponse(
                status="blocked",
                question=question,
                message="The generated query was rejected.",
                errors=[str(error)],
            )

        except SQLExecutionError as error:
            return StructuredAgentResponse(
                status="error",
                question=question,
                message=(
                    "The structured-data query could not "
                    "be executed."
                ),
                errors=[str(error)],
            )

        rows = list(result.rows)

        return StructuredAgentResponse(
            status="success",
            question=question,
            message=self._build_result_message(
                plan=plan,
                rows=rows,
            ),
            data={
                "rows": rows,
                "columns": list(result.columns),
                "row_count": result.row_count,
                "source": result.source,
                "referenced_tables": list(
                    result.referenced_tables
                ),
                "execution_time_ms": (
                    result.execution_time_ms
                ),
                "validation_time_ms": (
                    result.validation_time_ms
                ),
            },
            sql=result.normalized_sql,
            query_id=result.query_id,
            assumptions=list(plan.assumptions),
        )

    def _handle_metadata_question(
        self,
        question: str,
    ) -> StructuredAgentResponse | None:
        normalized = self._normalize_text(question)

        if any(
            phrase in normalized
            for phrase in (
                "what kpis",
                "available kpis",
                "list kpis",
                "supported kpis",
                "what metrics",
                "available metrics",
            )
        ):
            return StructuredAgentResponse(
                status="success",
                question=question,
                message="Retrieved the supported KPI catalog.",
                data={"kpis": list_kpis()},
            )

        if any(
            phrase in normalized
            for phrase in (
                "what datasets",
                "available datasets",
                "list datasets",
                "what tables",
                "available tables",
                "list tables",
            )
        ):
            return StructuredAgentResponse(
                status="success",
                question=question,
                message="Retrieved the available datasets.",
                data={"datasets": list_datasets()},
            )

        if any(
            phrase in normalized
            for phrase in (
                "what dimensions",
                "available dimensions",
                "list dimensions",
                "hierarchies",
            )
        ):
            return StructuredAgentResponse(
                status="success",
                question=question,
                message=(
                    "Retrieved the supported dimensions "
                    "and hierarchies."
                ),
                data={"dimensions": list_dimensions()},
            )

        if any(
            phrase in normalized
            for phrase in (
                "available periods",
                "date range",
                "data period",
                "latest period",
            )
        ):
            return StructuredAgentResponse(
                status="success",
                question=question,
                message="Retrieved the available data periods.",
                data={"periods": get_available_periods()},
            )

        return None

    def _create_plan(
        self,
        question: str,
    ) -> StructuredQueryPlan | StructuredAgentResponse:
        normalized = self._normalize_text(question)

        source = self._detect_source(normalized)
        metrics = self._detect_metrics(
            normalized,
            source=source,
        )
        dimensions = self._detect_dimensions(normalized)
        filters = self._detect_entity_filters(normalized)
        periods = self._detect_periods(normalized)

        assumptions: list[str] = []

        if (
            any(
                word in normalized
                for word in ("compare", "growth", "change")
            )
            and len(periods) < 2
            and not dimensions
        ):
            return StructuredAgentResponse(
                status="clarification",
                question=question,
                message=(
                    "Which two periods or business dimensions "
                    "should I compare?"
                ),
            )

        if source == "promotion" and any(
            phrase in normalized
            for phrase in (
                "achieve",
                "meet target",
                "target",
                "planned lift",
            )
        ):
            metrics = (
                "planned_sales_lift_pct",
                "actual_sales_lift_pct",
                "lift_variance_pct_points",
            )

            if not dimensions:
                dimensions = (
                    "state",
                    "channel",
                )

        if not metrics:
            if source == "sales" and any(
                word in normalized
                for word in (
                    "sales",
                    "performance",
                    "perform",
                )
            ):
                metrics = (
                    "net_revenue_inr",
                    "units_sold",
                    "gross_margin_pct",
                )
                assumptions.append(
                    "Sales performance was interpreted as "
                    "net revenue, units sold, and gross-margin "
                    "percentage."
                )

            elif source == "inventory":
                metrics = ("stockout_days",)
                assumptions.append(
                    "Inventory performance was interpreted "
                    "using stockout days."
                )

            elif source == "promotion":
                metrics = (
                    "planned_sales_lift_pct",
                    "actual_sales_lift_pct",
                )
                assumptions.append(
                    "Campaign performance was interpreted "
                    "using planned and actual sales lift."
                )

            else:
                return StructuredAgentResponse(
                    status="clarification",
                    question=question,
                    message=(
                        "Which KPI should I use, such as net "
                        "revenue, units sold, gross margin, or "
                        "stockout days?"
                    ),
                )

        metric_sources = {
            METRICS[metric]["source"]
            for metric in metrics
        }

        if len(metric_sources) > 1:
            return StructuredAgentResponse(
                status="unsupported",
                question=question,
                message=(
                    "This first structured-agent version cannot "
                    "combine sales, inventory, and promotion KPIs "
                    "in one SQL query. The orchestrator will later "
                    "handle this as a hybrid request."
                ),
            )

        source = next(iter(metric_sources))

        if len(periods) > 1:
            period_dimensions: list[str] = []

            if any(period.quarter for period in periods):
                period_dimensions.extend(
                    ["year", "quarter"]
                )
            else:
                period_dimensions.append("year")

            dimensions = tuple(
                dict.fromkeys(
                    period_dimensions + list(dimensions)
                )
            )

        return StructuredQueryPlan(
            source=source,
            metrics=metrics,
            dimensions=dimensions,
            filters=filters,
            periods=periods,
            assumptions=tuple(assumptions),
        )

    def _detect_source(
        self,
        question: str,
    ) -> str:
        if any(
            phrase in question
            for phrase in (
                "campaign",
                "promotion",
                "sales lift",
                "marketing spend",
            )
        ):
            return "promotion"

        if any(
            phrase in question
            for phrase in (
                "inventory",
                "stockout",
                "out of stock",
                "closing stock",
            )
        ):
            return "inventory"

        return "sales"

    def _detect_metrics(
        self,
        question: str,
        source: str,
    ) -> tuple[str, ...]:
        detected: list[str] = []

        ordered_metrics = sorted(
            METRICS.items(),
            key=lambda item: max(
                len(phrase)
                for phrase in item[1]["phrases"]
            ),
            reverse=True,
        )

        for metric_name, definition in ordered_metrics:
            if definition["source"] != source:
                continue

            if any(
                self._contains_phrase(question, phrase)
                for phrase in definition["phrases"]
            ):
                detected.append(metric_name)

        # Avoid treating "revenue" inside "gross revenue"
        # as both gross and net revenue.
        if (
            "gross_revenue_inr" in detected
            and "net_revenue_inr" in detected
            and "net revenue" not in question
        ):
            detected.remove("net_revenue_inr")

        return tuple(dict.fromkeys(detected))

    def _detect_dimensions(
        self,
        question: str,
    ) -> tuple[str, ...]:
        detected: list[str] = []

        for column, aliases in DIMENSIONS.items():
            for alias in aliases:
                patterns = (
                    f"by {alias}",
                    f"across {alias}",
                    f"per {alias}",
                    alias,
                )

                if any(
                    pattern in question
                    for pattern in patterns
                ):
                    if (
                        f"by {alias}" in question
                        or f"across {alias}" in question
                        or f"per {alias}" in question
                        or "wise" in alias
                        or alias in (
                            "monthly",
                            "quarterly",
                            "yearly",
                        )
                        or question.startswith(
                            f"which {alias}"
                        )
                    ):
                        detected.append(column)
                        break

        return tuple(dict.fromkeys(detected))

    def _detect_entity_filters(
        self,
        question: str,
    ) -> dict[str, str]:
        filters: dict[str, str] = {}

        for column, canonical_values in (
            ENTITY_ALIASES.items()
        ):
            for canonical, aliases in (
                canonical_values.items()
            ):
                ordered_aliases = sorted(
                    aliases,
                    key=len,
                    reverse=True,
                )

                if any(
                    self._contains_phrase(
                        question,
                        alias,
                    )
                    for alias in ordered_aliases
                ):
                    filters[column] = canonical
                    break

        return filters

    def _detect_periods(
        self,
        question: str,
    ) -> tuple[PeriodFilter, ...]:
        periods: list[PeriodFilter] = []

        quarter_matches = re.findall(
            r"\bq([1-4])\s*(20\d{2})\b"
            r"|\b(20\d{2})\s*q([1-4])\b",
            question,
        )

        for first_quarter, first_year, second_year, second_quarter in (
            quarter_matches
        ):
            quarter = first_quarter or second_quarter
            year = first_year or second_year

            periods.append(
                PeriodFilter(
                    year=int(year),
                    quarter=f"Q{quarter}",
                )
            )

        if "latest quarter" in question:
            available_periods = get_available_periods()
            latest_date = available_periods["sales_end"]

            latest_quarter = (
                (latest_date.month - 1) // 3
            ) + 1

            periods.append(
                PeriodFilter(
                    year=latest_date.year,
                    quarter=f"Q{latest_quarter}",
                )
            )

        if not periods:
            for month_name, month_number in MONTHS.items():
                match = re.search(
                    rf"\b{month_name}\s+(20\d{{2}})\b",
                    question,
                )

                if match:
                    periods.append(
                        PeriodFilter(
                            year=int(match.group(1)),
                            month=month_number,
                        )
                    )

        if not periods:
            years = re.findall(
                r"\b20\d{2}\b",
                question,
            )

            for year in dict.fromkeys(years):
                periods.append(
                    PeriodFilter(year=int(year))
                )

        return tuple(dict.fromkeys(periods))

    def _build_sql(
        self,
        plan: StructuredQueryPlan,
    ) -> str:
        table_name = SOURCE_VIEWS[plan.source]

        select_items: list[str] = list(
            plan.dimensions
        )

        for metric_name in plan.metrics:
            metric = METRICS[metric_name]

            expression = " ".join(
                metric["expression"].split()
            )

            select_items.append(
                f"{expression} AS {metric['alias']}"
            )

        where_conditions: list[str] = []

        for column, value in plan.filters.items():
            where_conditions.append(
                f"{column} = {self._sql_literal(value)}"
            )

        period_condition = self._build_period_condition(
            plan.periods,
            source=plan.source,
        )

        if period_condition:
            where_conditions.append(period_condition)

        sql_parts = [
            "SELECT",
            "    " + ",\n    ".join(select_items),
            f"FROM {table_name}",
        ]

        if where_conditions:
            sql_parts.extend(
                [
                    "WHERE",
                    "    " + "\n    AND ".join(
                        where_conditions
                    ),
                ]
            )

        if plan.dimensions:
            sql_parts.append(
                "GROUP BY "
                + ", ".join(plan.dimensions)
            )

            first_metric_alias = METRICS[
                plan.metrics[0]
            ]["alias"]

            sql_parts.append(
                f"ORDER BY {first_metric_alias} DESC"
            )

        return "\n".join(sql_parts)

    def _build_period_condition(
        self,
        periods: tuple[PeriodFilter, ...],
        source: str,
    ) -> str | None:
        if not periods:
            return None

        conditions: list[str] = []

        for period in periods:
            if source in {"sales", "inventory"}:
                parts = [f"year = {period.year}"]

                if period.quarter:
                    parts.append(
                        "quarter = "
                        + self._sql_literal(
                            period.quarter
                        )
                    )

                if period.month:
                    parts.append(
                        f"EXTRACT(MONTH FROM month) = "
                        f"{period.month}"
                    )

            else:
                parts = [
                    "EXTRACT(YEAR FROM start_date) = "
                    f"{period.year}"
                ]

                if period.quarter:
                    quarter_number = int(
                        period.quarter[-1]
                    )

                    parts.append(
                        "EXTRACT(QUARTER FROM start_date) = "
                        f"{quarter_number}"
                    )

                if period.month:
                    parts.append(
                        "EXTRACT(MONTH FROM start_date) = "
                        f"{period.month}"
                    )

            conditions.append(
                "(" + " AND ".join(parts) + ")"
            )

        if len(conditions) == 1:
            return conditions[0]

        return "(" + " OR ".join(conditions) + ")"

    def _build_result_message(
        self,
        plan: StructuredQueryPlan,
        rows: list[dict[str, Any]],
    ) -> str:
        if not rows:
            return (
                "No structured-data records matched the "
                "requested filters."
            )

        if {
            "planned_sales_lift_pct",
            "actual_sales_lift_pct",
        }.issubset(set(plan.metrics)):
            below_target = sum(
                1
                for row in rows
                if row.get(
                    "actual_sales_lift_pct",
                    0,
                )
                < row.get(
                    "planned_sales_lift_pct",
                    0,
                )
            )

            if below_target == len(rows):
                return (
                    "The campaign was below its planned "
                    f"sales-lift target in all {len(rows)} "
                    "returned market or channel records."
                )

            if below_target:
                return (
                    f"The campaign was below target in "
                    f"{below_target} of {len(rows)} returned "
                    "records."
                )

            return (
                "The campaign met or exceeded its planned "
                "sales-lift target in all returned records."
            )

        return (
            f"Retrieved {len(rows)} structured-data "
            "result row(s)."
        )

    @staticmethod
    def _normalize_text(value: str) -> str:
        normalized = value.lower()
        normalized = normalized.replace("_", " ")
        normalized = re.sub(
            r"[^\w%\-]+",
            " ",
            normalized,
        )
        normalized = re.sub(
            r"\s+",
            " ",
            normalized,
        )

        return normalized.strip()

    @staticmethod
    def _contains_phrase(
        text: str,
        phrase: str,
    ) -> bool:
        escaped = re.escape(phrase.lower())

        return bool(
            re.search(
                rf"(?<!\w){escaped}(?!\w)",
                text,
            )
        )

    @staticmethod
    def _sql_literal(value: str) -> str:
        escaped = value.replace("'", "''")
        return f"'{escaped}'"

    @staticmethod
    def _contains_sql_injection(
        question: str,
    ) -> bool:
        normalized = question.lower()

        blocked_patterns = (
            r";",
            r"\bdrop\s+table\b",
            r"\bdelete\s+from\b",
            r"\binsert\s+into\b",
            r"\bupdate\s+\w+\s+set\b",
            r"\balter\s+table\b",
            r"\btruncate\s+table\b",
            r"\bcreate\s+table\b",
        )

        return any(
            re.search(pattern, normalized)
            for pattern in blocked_patterns
        )


def main() -> None:
    """Run a few example structured questions."""

    agent = StructuredDataAgent()

    questions = [
        (
            "Show net revenue and gross margin percentage "
            "by region for Q2 2025"
        ),
        (
            "Did Sparkle Summer 2025 achieve its planned "
            "sales lift?"
        ),
        (
            "Show stockout days for FG-DW-LEM-500 by month "
            "in Q2 2025"
        ),
        (
            "Show net revenue for Fresh Glow in KA through "
            "GT in Q2 2025"
        ),
        "What KPIs are available?",
        "Compare FreshGlow sales",
    ]

    for question in questions:
        response = agent.answer(question)

        print("=" * 80)
        print("Question:", question)
        print("Status:", response.status)
        print("Message:", response.message)

        if response.sql:
            print("SQL:")
            print(response.sql)

        if response.data:
            print("Data:")
            print(response.data)

        if response.assumptions:
            print("Assumptions:", response.assumptions)


if __name__ == "__main__":
    main()