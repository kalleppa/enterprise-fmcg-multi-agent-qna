from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = PROJECT_ROOT / "data" / "generated" / "fmcg.duckdb"


def get_connection(
    read_only: bool = True,
) -> duckdb.DuckDBPyConnection:
    """Create an explicit DuckDB connection."""

    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {DATABASE_PATH}. "
            "Run python scripts/build_database.py first."
        )

    return duckdb.connect(
        database=str(DATABASE_PATH),
        read_only=read_only,
    )


def rows_to_dicts(
    connection: duckdb.DuckDBPyConnection,
) -> list[dict[str, Any]]:
    """Convert the most recent query result to dictionaries."""

    columns = [
        description[0]
        for description in connection.description
    ]

    return [
        dict(zip(columns, row))
        for row in connection.fetchall()
    ]


def list_datasets() -> list[dict[str, Any]]:
    """Return datasets that are approved for agent access."""

    connection = get_connection()

    try:
        result = connection.execute(
            """
            SELECT
                dataset_name,
                dataset_type,
                description,
                grain,
                time_column
            FROM dataset_catalog
            WHERE agent_access = TRUE
            ORDER BY dataset_type, dataset_name
            """
        )

        return rows_to_dicts(result)

    finally:
        connection.close()


def list_kpis() -> list[dict[str, Any]]:
    """Return supported KPIs and their calculation rules."""

    connection = get_connection()

    try:
        result = connection.execute(
            """
            SELECT
                kpi_name,
                source_dataset,
                source_column,
                recommended_aggregation,
                unit,
                description
            FROM kpi_catalog
            ORDER BY kpi_name
            """
        )

        return rows_to_dicts(result)

    finally:
        connection.close()


def list_dimensions() -> list[dict[str, Any]]:
    """Return supported dimensions and hierarchies."""

    connection = get_connection()

    try:
        result = connection.execute(
            """
            SELECT
                dimension_name,
                source_dataset,
                source_column,
                hierarchy_name,
                hierarchy_level,
                aliases,
                description
            FROM dimension_catalog
            ORDER BY
                hierarchy_name,
                hierarchy_level,
                dimension_name
            """
        )

        return rows_to_dicts(result)

    finally:
        connection.close()


def get_database_schema() -> list[dict[str, Any]]:
    """Read table and column information from information_schema."""

    connection = get_connection()

    try:
        result = connection.execute(
            """
            SELECT
                c.table_name,
                c.column_name,
                c.data_type,
                c.is_nullable,
                c.ordinal_position
            FROM information_schema.columns AS c
            INNER JOIN dataset_catalog AS d
                ON c.table_name = d.dataset_name
            WHERE c.table_schema = 'main'
              AND d.agent_access = TRUE
            ORDER BY
                c.table_name,
                c.ordinal_position
            """
        )

        return rows_to_dicts(result)

    finally:
        connection.close()


def get_available_periods() -> dict[str, Any]:
    """Return the available structured-data date ranges."""

    connection = get_connection()

    try:
        row = connection.execute(
            """
            SELECT
                MIN(month) AS sales_start,
                MAX(month) AS sales_end,
                (
                    SELECT MIN(month)
                    FROM inventory
                ) AS inventory_start,
                (
                    SELECT MAX(month)
                    FROM inventory
                ) AS inventory_end,
                (
                    SELECT MIN(start_date)
                    FROM promotions
                ) AS promotion_start,
                (
                    SELECT MAX(end_date)
                    FROM promotions
                ) AS promotion_end
            FROM sales
            """
        ).fetchone()

        columns = [
            description[0]
            for description in connection.description
        ]

        return dict(zip(columns, row))

    finally:
        connection.close()


def get_metadata_summary() -> dict[str, Any]:
    """Return all metadata needed by the structured-data agent."""

    return {
        "datasets": list_datasets(),
        "kpis": list_kpis(),
        "dimensions": list_dimensions(),
        "periods": get_available_periods(),
        "schema": get_database_schema(),
    }


def main() -> None:
    """Print a short metadata summary for local verification."""

    metadata = get_metadata_summary()

    print("Datasets:", len(metadata["datasets"]))
    print("KPIs:", len(metadata["kpis"]))
    print("Dimensions:", len(metadata["dimensions"]))
    print("Columns:", len(metadata["schema"]))
    print("Periods:", metadata["periods"])


if __name__ == "__main__":
    main()