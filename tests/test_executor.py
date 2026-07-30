from __future__ import annotations

import pytest

from src.database.executor import (
    SQLExecutionError,
    UnsafeSQLQueryError,
    execute_safe_sql,
)
from src.database.metadata import DATABASE_PATH


@pytest.fixture(scope="module", autouse=True)
def require_database() -> None:
    """Ensure the generated database exists before testing."""

    if not DATABASE_PATH.exists():
        pytest.fail(
            "DuckDB database is missing. Run "
            "`python scripts/build_database.py` first."
        )


def test_executes_approved_query() -> None:
    result = execute_safe_sql(
        """
        SELECT
            brand,
            COUNT(*) AS sku_count
        FROM products
        GROUP BY brand
        ORDER BY brand
        """,
        max_rows=100,
    )

    assert result.row_count == 3
    assert result.columns == (
        "brand",
        "sku_count",
    )
    assert result.referenced_tables == (
        "products",
    )
    assert "LIMIT 100" in result.normalized_sql
    assert result.execution_time_ms >= 0


def test_returns_expected_brands() -> None:
    result = execute_safe_sql(
        """
        SELECT DISTINCT brand
        FROM products
        ORDER BY brand
        """,
        max_rows=10,
    )

    brands = {
        row["brand"]
        for row in result.rows
    }

    assert brands == {
        "FreshGlow",
        "PureHome",
        "NutriBite",
    }


def test_enforces_maximum_rows() -> None:
    result = execute_safe_sql(
        """
        SELECT *
        FROM sales
        """,
        max_rows=5,
    )

    assert result.row_count == 5
    assert result.row_limit == 5
    assert result.limit_reached is True
    assert "LIMIT 5" in result.normalized_sql


def test_converts_database_values() -> None:
    result = execute_safe_sql(
        """
        SELECT
            DATE '2025-01-01' AS sample_date,
            CAST(1234.56 AS DECIMAL(12, 2)) AS amount
        """,
        max_rows=5,
    )

    assert result.rows[0]["sample_date"] == "2025-01-01"
    assert result.rows[0]["amount"] == pytest.approx(
        1234.56
    )


@pytest.mark.parametrize(
    "query",
    [
        "DROP TABLE sales",
        "DELETE FROM sales",
        "SELECT * FROM unknown_table",
        "SELECT * FROM read_csv_auto('/tmp/private.csv')",
        "SELECT * FROM sales; DELETE FROM sales",
    ],
)
def test_rejects_unsafe_queries(
    query: str,
) -> None:
    with pytest.raises(UnsafeSQLQueryError):
        execute_safe_sql(query)


def test_returns_query_metadata() -> None:
    result = execute_safe_sql(
        "SELECT COUNT(*) AS product_count FROM products"
    )

    assert result.query_id.startswith("sql-")
    assert result.source == "fmcg.duckdb"
    assert result.executed_at_utc
    assert result.validation_time_ms >= 0
    assert result.total_time_ms >= (
        result.execution_time_ms
    )


def test_reports_sql_execution_error() -> None:
    with pytest.raises(SQLExecutionError):
        execute_safe_sql(
            """
            SELECT
                SUM(nonexistent_column)
            FROM sales
            """
        )