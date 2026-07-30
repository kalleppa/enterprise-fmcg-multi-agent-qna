from __future__ import annotations

import pytest

from src.agents.structured_agent import (
    StructuredDataAgent,
)
from src.database.metadata import DATABASE_PATH


@pytest.fixture(scope="module")
def agent() -> StructuredDataAgent:
    if not DATABASE_PATH.exists():
        pytest.fail(
            "DuckDB database is missing. Run "
            "`python scripts/build_database.py` first."
        )

    return StructuredDataAgent()


def test_returns_available_kpis(
    agent: StructuredDataAgent,
) -> None:
    response = agent.answer(
        "What KPIs are available?"
    )

    assert response.status == "success"
    assert "kpis" in response.data
    assert len(response.data["kpis"]) > 0


def test_retrieves_regional_q2_revenue(
    agent: StructuredDataAgent,
) -> None:
    response = agent.answer(
        "Show net revenue by region for Q2 2025"
    )

    assert response.status == "success"
    assert response.data["row_count"] == 4
    assert response.sql is not None
    assert "vw_sales_enriched" in response.sql
    assert "year = 2025" in response.sql
    assert "quarter = 'Q2'" in response.sql


def test_campaign_is_below_target(
    agent: StructuredDataAgent,
) -> None:
    response = agent.answer(
        "Did Sparkle Summer 2025 achieve its "
        "planned sales lift?"
    )

    assert response.status == "success"
    assert response.data["row_count"] == 2
    assert "below" in response.message.lower()

    for row in response.data["rows"]:
        assert (
            row["actual_sales_lift_pct"]
            < row["planned_sales_lift_pct"]
        )


def test_resolves_brand_state_and_channel_aliases(
    agent: StructuredDataAgent,
) -> None:
    response = agent.answer(
        "Show net revenue for Fresh Glow in KA "
        "through GT in Q2 2025"
    )

    assert response.status == "success"
    assert response.data["row_count"] == 1
    assert "FreshGlow" in response.sql
    assert "Karnataka" in response.sql
    assert "General Trade" in response.sql


def test_returns_stockout_results(
    agent: StructuredDataAgent,
) -> None:
    response = agent.answer(
        "Show stockout days for FG-DW-LEM-500 "
        "by month in Q2 2025"
    )

    assert response.status == "success"
    assert response.sql is not None
    assert "vw_inventory_enriched" in response.sql
    assert "stockout_days" in response.data["columns"]


def test_requests_clarification_for_incomplete_comparison(
    agent: StructuredDataAgent,
) -> None:
    response = agent.answer(
        "Compare FreshGlow sales"
    )

    assert response.status == "clarification"
    assert "which" in response.message.lower()


def test_blocks_sql_injection_pattern(
    agent: StructuredDataAgent,
) -> None:
    response = agent.answer(
        "Show sales; DROP TABLE sales"
    )

    assert response.status == "blocked"
    assert response.sql is None


def test_adds_provenance_and_timing(
    agent: StructuredDataAgent,
) -> None:
    response = agent.answer(
        "Show units sold by brand for 2025"
    )

    assert response.status == "success"
    assert response.query_id is not None
    assert response.data["source"] == "fmcg.duckdb"
    assert response.data["execution_time_ms"] >= 0
    assert response.data["referenced_tables"] == [
        "vw_sales_enriched"
    ]