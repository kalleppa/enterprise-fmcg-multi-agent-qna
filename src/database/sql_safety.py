from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError


DEFAULT_MAX_ROWS = 500

DEFAULT_ALLOWED_TABLES = frozenset(
    {
        "products",
        "geography",
        "distributors",
        "sales",
        "inventory",
        "promotions",
        "vw_sales_enriched",
        "vw_inventory_enriched",
        "vw_promotions_enriched",
        "dataset_catalog",
        "kpi_catalog",
        "dimension_catalog",
    }
)

ALLOWED_SCHEMAS = frozenset(
    {
        "",
        "main",
    }
)

# Node types that must never appear in generated SQL.
# Class names are used instead of importing every SQLGlot expression type,
# which keeps the validator compatible across SQLGlot releases.
BLOCKED_NODE_NAMES = frozenset(
    {
        "insert",
        "update",
        "delete",
        "drop",
        "create",
        "alter",
        "truncate",
        "truncatetable",
        "merge",
        "replace",
        "copy",
        "command",
        "attach",
        "detach",
        "install",
        "load",
        "grant",
        "revoke",
        "transaction",
        "commit",
        "rollback",
        "set",
        "use",
        "pragma",
        "vacuum",
        "analyze",
        "execute",
        "prepare",
        "export",
        "import",
    }
)

# DuckDB functions that could read files, URLs, external databases,
# secrets, extensions or environment-dependent resources.
BLOCKED_FUNCTION_NAMES = frozenset(
    {
        "read_csv",
        "read_csv_auto",
        "read_parquet",
        "parquet_scan",
        "csv_scan",
        "read_json",
        "read_json_auto",
        "read_ndjson",
        "read_text",
        "read_blob",
        "glob",
        "sqlite_scan",
        "sqlite_query",
        "postgres_scan",
        "postgres_query",
        "mysql_scan",
        "mysql_query",
        "delta_scan",
        "iceberg_scan",
        "httpfs",
        "query",
        "query_table",
        "current_setting",
        "getenv",
    }
)


@dataclass(frozen=True)
class SQLValidationResult:
    """Outcome returned by the SQL safety validator."""

    is_valid: bool
    normalized_sql: str | None
    errors: tuple[str, ...]
    referenced_tables: tuple[str, ...]


def normalize_name(value: str | None) -> str:
    """Normalize an SQL identifier for allowlist comparisons."""

    if not value:
        return ""

    return value.strip().strip('"').strip("`").lower()


def get_cte_names(expression: exp.Expression) -> set[str]:
    """Return aliases declared by common table expressions."""

    return {
        normalize_name(cte.alias_or_name)
        for cte in expression.find_all(exp.CTE)
        if cte.alias_or_name
    }


def get_referenced_tables(
    expression: exp.Expression,
) -> tuple[set[str], set[str]]:
    """
    Return physical table names and non-approved schema names.

    CTE aliases are excluded because they are temporary query-local names,
    not database tables.
    """

    cte_names = get_cte_names(expression)

    referenced_tables: set[str] = set()
    invalid_schemas: set[str] = set()

    for table in expression.find_all(exp.Table):
        table_name = normalize_name(table.name)
        schema_name = normalize_name(table.db)
        catalog_name = normalize_name(table.catalog)

        if catalog_name:
            invalid_schemas.add(catalog_name)

        if schema_name not in ALLOWED_SCHEMAS:
            invalid_schemas.add(schema_name)

        if table_name and table_name not in cte_names:
            referenced_tables.add(table_name)

    return referenced_tables, invalid_schemas


def get_function_name(function: exp.Func) -> str:
    """Return the actual name of a parsed SQL function."""

    if isinstance(function, exp.Anonymous):
        return normalize_name(function.name)

    try:
        return normalize_name(function.sql_name())
    except (AttributeError, TypeError):
        return normalize_name(type(function).__name__)


def find_blocked_nodes(
    expression: exp.Expression,
) -> set[str]:
    """Find prohibited SQL expression types."""

    blocked_nodes: set[str] = set()

    for node in expression.walk():
        node_name = type(node).__name__.lower()

        if node_name in BLOCKED_NODE_NAMES:
            blocked_nodes.add(node_name)

    return blocked_nodes


def find_blocked_functions(
    expression: exp.Expression,
) -> set[str]:
    """Find prohibited file, network or external-database functions."""

    blocked_functions: set[str] = set()

    for function in expression.find_all(exp.Func):
        function_name = get_function_name(function)

        if function_name in BLOCKED_FUNCTION_NAMES:
            blocked_functions.add(function_name)

    # Some table functions may be represented as specialized AST nodes.
    for node in expression.walk():
        node_name = type(node).__name__.lower()

        if node_name in BLOCKED_FUNCTION_NAMES:
            blocked_functions.add(node_name)

    return blocked_functions


def apply_row_limit(
    expression: exp.Query,
    max_rows: int,
) -> tuple[exp.Query, str | None]:
    """
    Add a maximum row limit.

    If the query already contains a larger or non-literal limit, reject it
    instead of silently changing the user's query.
    """

    limit_node = expression.args.get("limit")

    if limit_node is None:
        return expression.limit(max_rows), None

    limit_expression = limit_node.expression

    if not isinstance(limit_expression, exp.Literal):
        return expression, (
            "LIMIT must be a fixed integer value."
        )

    if not limit_expression.is_int:
        return expression, (
            "LIMIT must be a positive integer value."
        )

    requested_limit = int(limit_expression.this)

    if requested_limit <= 0:
        return expression, (
            "LIMIT must be greater than zero."
        )

    if requested_limit > max_rows:
        return expression, (
            f"Requested LIMIT {requested_limit} exceeds "
            f"the maximum allowed value of {max_rows}."
        )

    return expression, None


def validate_sql(
    sql: str,
    allowed_tables: Iterable[str] = DEFAULT_ALLOWED_TABLES,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> SQLValidationResult:
    """
    Parse and validate generated SQL before database execution.

    Safety rules:

    1. SQL must not be empty.
    2. Exactly one SQL statement is allowed.
    3. Only query expressions are allowed.
    4. Mutation, DDL and administrative nodes are rejected.
    5. File, network and external database functions are rejected.
    6. Only allowlisted tables and views can be queried.
    7. Only the DuckDB `main` schema is allowed.
    8. A maximum result-row limit is enforced.
    """

    errors: list[str] = []

    if not sql or not sql.strip():
        return SQLValidationResult(
            is_valid=False,
            normalized_sql=None,
            errors=("SQL query cannot be empty.",),
            referenced_tables=(),
        )

    if max_rows <= 0:
        return SQLValidationResult(
            is_valid=False,
            normalized_sql=None,
            errors=("max_rows must be greater than zero.",),
            referenced_tables=(),
        )

    normalized_allowed_tables = {
        normalize_name(table)
        for table in allowed_tables
    }

    try:
        statements = sqlglot.parse(
            sql,
            read="duckdb",
        )
    except ParseError as error:
        return SQLValidationResult(
            is_valid=False,
            normalized_sql=None,
            errors=(f"SQL parsing failed: {error}",),
            referenced_tables=(),
        )

    if len(statements) != 1:
        return SQLValidationResult(
            is_valid=False,
            normalized_sql=None,
            errors=(
                "Exactly one SQL statement is allowed.",
            ),
            referenced_tables=(),
        )

    expression = statements[0]

    if expression is None:
        return SQLValidationResult(
            is_valid=False,
            normalized_sql=None,
            errors=("SQL parser returned an empty statement.",),
            referenced_tables=(),
        )

    if not isinstance(expression, exp.Query):
        errors.append(
            "Only read-only SELECT queries are allowed."
        )

    blocked_nodes = find_blocked_nodes(expression)

    if blocked_nodes:
        errors.append(
            "Blocked SQL operations detected: "
            + ", ".join(sorted(blocked_nodes))
            + "."
        )

    blocked_functions = find_blocked_functions(expression)

    if blocked_functions:
        errors.append(
            "Blocked SQL functions detected: "
            + ", ".join(sorted(blocked_functions))
            + "."
        )

    referenced_tables, invalid_schemas = (
        get_referenced_tables(expression)
    )

    if invalid_schemas:
        errors.append(
            "Queries may access only the DuckDB main schema. "
            "Blocked schemas or catalogs: "
            + ", ".join(sorted(invalid_schemas))
            + "."
        )

    unknown_tables = (
        referenced_tables
        - normalized_allowed_tables
    )

    if unknown_tables:
        errors.append(
            "Query references non-approved tables or views: "
            + ", ".join(sorted(unknown_tables))
            + "."
        )

    normalized_sql: str | None = None

    if not errors and isinstance(expression, exp.Query):
        limited_expression, limit_error = apply_row_limit(
            expression,
            max_rows=max_rows,
        )

        if limit_error:
            errors.append(limit_error)
        else:
            normalized_sql = limited_expression.sql(
                dialect="duckdb",
                pretty=True,
            )

    return SQLValidationResult(
        is_valid=not errors,
        normalized_sql=normalized_sql,
        errors=tuple(errors),
        referenced_tables=tuple(
            sorted(referenced_tables)
        ),
    )


def require_safe_sql(
    sql: str,
    allowed_tables: Iterable[str] = DEFAULT_ALLOWED_TABLES,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> str:
    """
    Validate SQL and return normalized SQL.

    Raise ValueError when the query is unsafe. This helper will later be used
    immediately before database execution.
    """

    result = validate_sql(
        sql=sql,
        allowed_tables=allowed_tables,
        max_rows=max_rows,
    )

    if not result.is_valid or not result.normalized_sql:
        message = "; ".join(result.errors)

        raise ValueError(
            f"Unsafe SQL query rejected: {message}"
        )

    return result.normalized_sql


def main() -> None:
    """Run a small local demonstration."""

    sample_queries = [
        """
        SELECT
            region,
            SUM(net_revenue_inr) AS net_revenue_inr
        FROM vw_sales_enriched
        WHERE year = 2025
          AND quarter = 'Q2'
        GROUP BY region
        ORDER BY net_revenue_inr DESC
        """,
        "DROP TABLE sales",
        "SELECT * FROM read_csv_auto('/tmp/private.csv')",
        "SELECT * FROM unknown_table",
        "SELECT * FROM sales; DELETE FROM sales",
    ]

    for query in sample_queries:
        result = validate_sql(query)

        print("=" * 70)
        print("Query:")
        print(query.strip())
        print("\nValid:", result.is_valid)
        print("Tables:", result.referenced_tables)
        print("Errors:", result.errors)
        print("Normalized SQL:")
        print(result.normalized_sql)


if __name__ == "__main__":
    main()