from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timezone
from decimal import Decimal
from time import perf_counter
from typing import Any
from uuid import uuid4

import duckdb

from src.database.metadata import DATABASE_PATH, get_connection
from src.database.sql_safety import (
    DEFAULT_ALLOWED_TABLES,
    DEFAULT_MAX_ROWS,
    validate_sql,
)


class UnsafeSQLQueryError(ValueError):
    """Raised when SQL fails safety validation."""


class SQLExecutionError(RuntimeError):
    """Raised when safe SQL fails during database execution."""


@dataclass(frozen=True)
class SQLExecutionResult:
    """Standard response returned by the SQL executor."""

    query_id: str
    normalized_sql: str
    referenced_tables: tuple[str, ...]
    columns: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]
    row_count: int
    row_limit: int
    limit_reached: bool
    validation_time_ms: float
    execution_time_ms: float
    total_time_ms: float
    executed_at_utc: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        """Convert the execution result to a serializable dictionary."""

        return asdict(self)


def make_json_safe(value: Any) -> Any:
    """Convert DuckDB values into API-friendly Python values."""

    if value is None or isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    if isinstance(value, Decimal):
        return float(value)

    if isinstance(value, (date, datetime, time)):
        return value.isoformat()

    if isinstance(value, bytes):
        return value.hex()

    if isinstance(value, (list, tuple)):
        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    return str(value)


def execute_safe_sql(
    sql: str,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> SQLExecutionResult:
    """
    Validate and execute one read-only SQL query.

    The execution process is:

    1. Parse and validate SQL.
    2. Confirm that only approved tables are referenced.
    3. Add or validate the row limit.
    4. Open DuckDB in read-only mode.
    5. Execute the normalized SQL.
    6. Return standardized rows and execution metadata.
    """

    total_start = perf_counter()
    validation_start = perf_counter()

    validation = validate_sql(
        sql=sql,
        allowed_tables=DEFAULT_ALLOWED_TABLES,
        max_rows=max_rows,
    )

    validation_time_ms = round(
        (perf_counter() - validation_start) * 1000,
        3,
    )

    if (
        not validation.is_valid
        or not validation.normalized_sql
    ):
        error_message = "; ".join(validation.errors)

        raise UnsafeSQLQueryError(
            f"SQL query rejected: {error_message}"
        )

    connection = get_connection(read_only=True)

    execution_start = perf_counter()

    try:
        query_result = connection.execute(
            validation.normalized_sql
        )

        columns = tuple(
            description[0]
            for description in query_result.description
        )

        raw_rows = query_result.fetchall()

    except duckdb.Error as error:
        raise SQLExecutionError(
            "The validated SQL query could not be executed: "
            f"{error}"
        ) from error

    finally:
        connection.close()

    execution_time_ms = round(
        (perf_counter() - execution_start) * 1000,
        3,
    )

    rows = tuple(
        {
            column: make_json_safe(value)
            for column, value in zip(columns, row)
        }
        for row in raw_rows
    )

    total_time_ms = round(
        (perf_counter() - total_start) * 1000,
        3,
    )

    return SQLExecutionResult(
        query_id=f"sql-{uuid4().hex[:12]}",
        normalized_sql=validation.normalized_sql,
        referenced_tables=validation.referenced_tables,
        columns=columns,
        rows=rows,
        row_count=len(rows),
        row_limit=max_rows,
        limit_reached=len(rows) >= max_rows,
        validation_time_ms=validation_time_ms,
        execution_time_ms=execution_time_ms,
        total_time_ms=total_time_ms,
        executed_at_utc=datetime.now(
            timezone.utc
        ).isoformat(),
        source=DATABASE_PATH.name,
    )


def main() -> None:
    """Execute an example query for local verification."""

    query = """
        SELECT
            region,
            SUM(net_revenue_inr) AS net_revenue_inr,
            SUM(gross_margin_inr) AS gross_margin_inr,
            ROUND(
                SUM(gross_margin_inr)
                / NULLIF(SUM(net_revenue_inr), 0)
                * 100,
                2
            ) AS gross_margin_pct
        FROM vw_sales_enriched
        WHERE year = 2025
          AND quarter = 'Q2'
        GROUP BY region
        ORDER BY net_revenue_inr DESC
    """

    result = execute_safe_sql(
        sql=query,
        max_rows=100,
    )

    print("Query ID:", result.query_id)
    print("Tables:", result.referenced_tables)
    print("Columns:", result.columns)
    print("Rows:", result.row_count)
    print("Execution time:", result.execution_time_ms, "ms")
    print()

    for row in result.rows:
        print(row)


if __name__ == "__main__":
    main()