from __future__ import annotations

from pathlib import Path

import pytest

from src.agents.coding_agent import (
    CodingAnalysisAgent,
    PROJECT_ROOT,
)


@pytest.fixture()
def agent() -> CodingAnalysisAgent:
    return CodingAnalysisAgent()


@pytest.fixture()
def sample_rows() -> list[dict[str, object]]:
    return [
        {
            "month": "2025-01-01",
            "region": "South Region",
            "promotion_spend_inr": 100000,
            "actual_sales_lift_pct": 5.0,
        },
        {
            "month": "2025-02-01",
            "region": "West Region",
            "promotion_spend_inr": 200000,
            "actual_sales_lift_pct": 10.0,
        },
        {
            "month": "2025-03-01",
            "region": "North Region",
            "promotion_spend_inr": 300000,
            "actual_sales_lift_pct": 15.0,
        },
    ]


def test_generates_summary(
    agent: CodingAnalysisAgent,
    sample_rows: list[dict[str, object]],
) -> None:
    response = agent.analyze(
        rows=sample_rows,
        operation="summary",
    )

    assert response.status == "success"

    summaries = response.results[
        "numeric_summaries"
    ]

    assert "promotion_spend_inr" in summaries

    assert (
        summaries[
            "promotion_spend_inr"
        ]["sum"]
        == 600000
    )


def test_calculates_correlation(
    agent: CodingAnalysisAgent,
    sample_rows: list[dict[str, object]],
) -> None:
    response = agent.analyze(
        rows=sample_rows,
        operation="correlation",
        y_column="promotion_spend_inr",
        second_y_column="actual_sales_lift_pct",
    )

    assert response.status == "success"

    assert response.results[
        "correlation"
    ] == pytest.approx(1.0)


def test_calculates_percentage_change(
    agent: CodingAnalysisAgent,
    sample_rows: list[dict[str, object]],
) -> None:
    response = agent.analyze(
        rows=sample_rows,
        operation="percentage_change",
        x_column="month",
        y_column="actual_sales_lift_pct",
    )

    assert response.status == "success"

    assert response.results[
        "starting_value"
    ] == 5.0

    assert response.results[
        "ending_value"
    ] == 15.0

    assert response.results[
        "percentage_change"
    ] == 200.0


def test_creates_bar_chart(
    agent: CodingAnalysisAgent,
    sample_rows: list[dict[str, object]],
) -> None:
    response = agent.analyze(
        rows=sample_rows,
        operation="chart",
        x_column="region",
        y_column="actual_sales_lift_pct",
        chart_type="bar",
        title="Sales Lift Test",
    )

    assert response.status == "success"
    assert response.chart_path is not None

    chart_path = (
        PROJECT_ROOT
        / response.chart_path
    )

    assert chart_path.exists()
    assert chart_path.suffix == ".png"

    chart_path.unlink()


def test_supports_grouped_chart(
    agent: CodingAnalysisAgent,
    sample_rows: list[dict[str, object]],
) -> None:
    response = agent.analyze(
        rows=sample_rows,
        operation="chart",
        group_by="region",
        y_column="promotion_spend_inr",
        chart_type="bar",
        aggregation="sum",
    )

    assert response.status == "success"
    assert response.chart_path is not None

    chart_path = Path(
        PROJECT_ROOT
        / response.chart_path
    )

    assert chart_path.exists()
    chart_path.unlink()


def test_requests_columns_for_correlation(
    agent: CodingAnalysisAgent,
    sample_rows: list[dict[str, object]],
) -> None:
    response = agent.analyze(
        rows=sample_rows,
        operation="correlation",
    )

    assert response.status == "clarification"


def test_rejects_missing_column(
    agent: CodingAnalysisAgent,
    sample_rows: list[dict[str, object]],
) -> None:
    response = agent.analyze(
        rows=sample_rows,
        operation="percentage_change",
        x_column="month",
        y_column="unknown_metric",
    )

    assert response.status == "error"

    assert "Missing required column" in (
        response.errors[0]
    )


def test_rejects_unsupported_operation(
    agent: CodingAnalysisAgent,
    sample_rows: list[dict[str, object]],
) -> None:
    response = agent.analyze(
        rows=sample_rows,
        operation="execute_python",
    )

    assert response.status == "unsupported"


def test_rejects_empty_data(
    agent: CodingAnalysisAgent,
) -> None:
    response = agent.analyze(
        rows=[],
        operation="summary",
    )

    assert response.status == "clarification"


def test_rejects_unsupported_chart_type(
    agent: CodingAnalysisAgent,
    sample_rows: list[dict[str, object]],
) -> None:
    response = agent.analyze(
        rows=sample_rows,
        operation="chart",
        x_column="region",
        y_column="actual_sales_lift_pct",
        chart_type="pie",
    )

    assert response.status == "unsupported"