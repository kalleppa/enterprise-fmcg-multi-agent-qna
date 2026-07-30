from __future__ import annotations

from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_DIRECTORY = PROJECT_ROOT / "data" / "structured"
DATABASE_DIRECTORY = PROJECT_ROOT / "data" / "generated"
DATABASE_PATH = DATABASE_DIRECTORY / "fmcg.duckdb"


TABLE_DEFINITIONS: dict[str, str] = {
    "products": """
        CREATE TABLE products (
            sku_id VARCHAR PRIMARY KEY,
            product_name VARCHAR NOT NULL,
            brand VARCHAR NOT NULL,
            category VARCHAR NOT NULL,
            subcategory VARCHAR NOT NULL,
            pack_size VARCHAR NOT NULL,
            launch_date DATE NOT NULL,
            discontinuation_date DATE,
            status VARCHAR NOT NULL,
            unit_price_inr DECIMAL(12, 2) NOT NULL,
            base_cogs_ratio DECIMAL(8, 4) NOT NULL
        )
    """,
    "geography": """
        CREATE TABLE geography (
            geography_id VARCHAR PRIMARY KEY,
            country VARCHAR NOT NULL,
            region VARCHAR NOT NULL,
            state VARCHAR NOT NULL,
            city VARCHAR NOT NULL
        )
    """,
    "distributors": """
        CREATE TABLE distributors (
            distributor_id VARCHAR PRIMARY KEY,
            distributor_name VARCHAR NOT NULL,
            region VARCHAR NOT NULL,
            state VARCHAR NOT NULL,
            channel VARCHAR NOT NULL,
            active_status BOOLEAN NOT NULL
        )
    """,
    "sales": """
        CREATE TABLE sales (
            month DATE NOT NULL,
            sku_id VARCHAR NOT NULL,
            geography_id VARCHAR NOT NULL,
            distributor_id VARCHAR NOT NULL,
            channel VARCHAR NOT NULL,
            units_sold INTEGER NOT NULL,
            gross_revenue_inr DECIMAL(18, 2) NOT NULL,
            discount_inr DECIMAL(18, 2) NOT NULL,
            net_revenue_inr DECIMAL(18, 2) NOT NULL,
            cogs_inr DECIMAL(18, 2) NOT NULL,
            gross_margin_inr DECIMAL(18, 2) NOT NULL,
            gross_margin_pct DECIMAL(8, 2) NOT NULL
        )
    """,
    "inventory": """
        CREATE TABLE inventory (
            month DATE NOT NULL,
            sku_id VARCHAR NOT NULL,
            geography_id VARCHAR NOT NULL,
            distributor_id VARCHAR NOT NULL,
            channel VARCHAR NOT NULL,
            opening_stock_units INTEGER NOT NULL,
            received_units INTEGER NOT NULL,
            units_sold INTEGER NOT NULL,
            closing_stock_units INTEGER NOT NULL,
            stockout_days INTEGER NOT NULL,
            inventory_status VARCHAR NOT NULL
        )
    """,
    "promotions": """
        CREATE TABLE promotions (
            campaign_id VARCHAR NOT NULL,
            campaign_name VARCHAR NOT NULL,
            sku_id VARCHAR NOT NULL,
            geography_id VARCHAR NOT NULL,
            channel VARCHAR NOT NULL,
            start_date DATE NOT NULL,
            end_date DATE NOT NULL,
            promotion_spend_inr DECIMAL(18, 2) NOT NULL,
            planned_sales_lift_pct DECIMAL(8, 2) NOT NULL,
            actual_sales_lift_pct DECIMAL(8, 2) NOT NULL,
            campaign_status VARCHAR NOT NULL
        )
    """,
}


CSV_FILES: dict[str, str] = {
    "products": "products.csv",
    "geography": "geography.csv",
    "distributors": "distributors.csv",
    "sales": "sales.csv",
    "inventory": "inventory.csv",
    "promotions": "promotions.csv",
}


def validate_input_files() -> None:
    """Confirm that all required CSV files exist."""

    missing_files = [
        filename
        for filename in CSV_FILES.values()
        if not (CSV_DIRECTORY / filename).exists()
    ]

    if missing_files:
        missing = ", ".join(missing_files)

        raise FileNotFoundError(
            "The following structured-data files are missing: "
            f"{missing}. Run scripts/generate_data.py first."
        )


def copy_csv_to_table(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    csv_path: Path,
) -> None:
    """Copy one CSV file into a previously created table."""

    safe_path = csv_path.resolve().as_posix().replace("'", "''")

    connection.execute(
        f"""
        COPY {table_name}
        FROM '{safe_path}'
        (
            HEADER,
            DELIMITER ',',
            NULLSTR ''
        )
        """
    )


def create_business_views(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create business-friendly views used by the SQL agent."""

    connection.execute(
        """
        CREATE VIEW vw_sales_enriched AS
        SELECT
            s.month,
            CAST(EXTRACT(YEAR FROM s.month) AS INTEGER) AS year,
            'Q' || CAST(
                CAST(EXTRACT(QUARTER FROM s.month) AS INTEGER)
                AS VARCHAR
            ) AS quarter,
            p.sku_id,
            p.product_name,
            p.brand,
            p.category,
            p.subcategory,
            p.pack_size,
            g.country,
            g.region,
            g.state,
            g.city,
            s.channel,
            d.distributor_id,
            d.distributor_name,
            s.units_sold,
            s.gross_revenue_inr,
            s.discount_inr,
            s.net_revenue_inr,
            s.cogs_inr,
            s.gross_margin_inr,
            s.gross_margin_pct
        FROM sales AS s
        INNER JOIN products AS p
            ON s.sku_id = p.sku_id
        INNER JOIN geography AS g
            ON s.geography_id = g.geography_id
        INNER JOIN distributors AS d
            ON s.distributor_id = d.distributor_id
        """
    )

    connection.execute(
        """
        CREATE VIEW vw_inventory_enriched AS
        SELECT
            i.month,
            CAST(EXTRACT(YEAR FROM i.month) AS INTEGER) AS year,
            'Q' || CAST(
                CAST(EXTRACT(QUARTER FROM i.month) AS INTEGER)
                AS VARCHAR
            ) AS quarter,
            p.sku_id,
            p.product_name,
            p.brand,
            p.category,
            g.country,
            g.region,
            g.state,
            g.city,
            i.channel,
            d.distributor_id,
            d.distributor_name,
            i.opening_stock_units,
            i.received_units,
            i.units_sold,
            i.closing_stock_units,
            i.stockout_days,
            i.inventory_status
        FROM inventory AS i
        INNER JOIN products AS p
            ON i.sku_id = p.sku_id
        INNER JOIN geography AS g
            ON i.geography_id = g.geography_id
        INNER JOIN distributors AS d
            ON i.distributor_id = d.distributor_id
        """
    )

    connection.execute(
        """
        CREATE VIEW vw_promotions_enriched AS
        SELECT
            pr.campaign_id,
            pr.campaign_name,
            pr.start_date,
            pr.end_date,
            p.sku_id,
            p.product_name,
            p.brand,
            p.category,
            g.country,
            g.region,
            g.state,
            g.city,
            pr.channel,
            pr.promotion_spend_inr,
            pr.planned_sales_lift_pct,
            pr.actual_sales_lift_pct,
            (
                pr.actual_sales_lift_pct
                - pr.planned_sales_lift_pct
            ) AS lift_variance_pct_points,
            pr.campaign_status
        FROM promotions AS pr
        INNER JOIN products AS p
            ON pr.sku_id = p.sku_id
        INNER JOIN geography AS g
            ON pr.geography_id = g.geography_id
        """
    )


def create_metadata_catalogs(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create metadata tables for datasets, KPIs and dimensions."""

    connection.execute(
        """
        CREATE TABLE dataset_catalog (
            dataset_name VARCHAR PRIMARY KEY,
            dataset_type VARCHAR NOT NULL,
            description VARCHAR NOT NULL,
            grain VARCHAR NOT NULL,
            time_column VARCHAR,
            agent_access BOOLEAN NOT NULL
        )
        """
    )

    connection.executemany(
        """
        INSERT INTO dataset_catalog
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "products",
                "dimension",
                "Product and SKU master data",
                "One row per SKU",
                None,
                True,
            ),
            (
                "geography",
                "dimension",
                "Country, region, state and city hierarchy",
                "One row per geography",
                None,
                True,
            ),
            (
                "distributors",
                "dimension",
                "Distributor and channel master data",
                "One row per distributor",
                None,
                True,
            ),
            (
                "sales",
                "fact",
                "Monthly FMCG sales and profitability data",
                (
                    "One row per month, SKU, geography, "
                    "distributor and channel"
                ),
                "month",
                True,
            ),
            (
                "inventory",
                "fact",
                "Monthly inventory and stockout information",
                (
                    "One row per month, SKU, geography, "
                    "distributor and channel"
                ),
                "month",
                True,
            ),
            (
                "promotions",
                "fact",
                "Campaign targets, spend and actual sales lift",
                (
                    "One row per campaign, SKU, geography "
                    "and channel"
                ),
                "start_date",
                True,
            ),
            (
                "vw_sales_enriched",
                "view",
                "Business-friendly sales view with dimensions",
                (
                    "One row per month, SKU, geography, "
                    "distributor and channel"
                ),
                "month",
                True,
            ),
            (
                "vw_inventory_enriched",
                "view",
                "Business-friendly inventory view with dimensions",
                (
                    "One row per month, SKU, geography, "
                    "distributor and channel"
                ),
                "month",
                True,
            ),
            (
                "vw_promotions_enriched",
                "view",
                "Business-friendly campaign performance view",
                (
                    "One row per campaign, SKU, geography "
                    "and channel"
                ),
                "start_date",
                True,
            ),
        ],
    )

    connection.execute(
        """
        CREATE TABLE kpi_catalog (
            kpi_name VARCHAR PRIMARY KEY,
            source_dataset VARCHAR NOT NULL,
            source_column VARCHAR NOT NULL,
            recommended_aggregation VARCHAR NOT NULL,
            unit VARCHAR NOT NULL,
            description VARCHAR NOT NULL
        )
        """
    )

    connection.executemany(
        """
        INSERT INTO kpi_catalog
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "Units Sold",
                "sales",
                "units_sold",
                "SUM",
                "units",
                "Number of product units sold",
            ),
            (
                "Gross Revenue",
                "sales",
                "gross_revenue_inr",
                "SUM",
                "INR",
                "Revenue before discounts",
            ),
            (
                "Discount",
                "sales",
                "discount_inr",
                "SUM",
                "INR",
                "Discount deducted from gross revenue",
            ),
            (
                "Net Revenue",
                "sales",
                "net_revenue_inr",
                "SUM",
                "INR",
                "Gross revenue minus discounts",
            ),
            (
                "Cost of Goods Sold",
                "sales",
                "cogs_inr",
                "SUM",
                "INR",
                "Product cost associated with sales",
            ),
            (
                "Gross Margin",
                "sales",
                "gross_margin_inr",
                "SUM",
                "INR",
                "Net revenue minus cost of goods sold",
            ),
            (
                "Gross Margin Percentage",
                "sales",
                "gross_margin_pct",
                "DERIVED",
                "percentage",
                (
                    "Calculate as SUM(gross_margin_inr) / "
                    "SUM(net_revenue_inr) * 100"
                ),
            ),
            (
                "Closing Stock",
                "inventory",
                "closing_stock_units",
                "SUM",
                "units",
                "Closing inventory balance",
            ),
            (
                "Stockout Days",
                "inventory",
                "stockout_days",
                "SUM",
                "days",
                "Number of days the product was unavailable",
            ),
            (
                "Promotion Spend",
                "promotions",
                "promotion_spend_inr",
                "SUM",
                "INR",
                "Amount invested in a campaign",
            ),
            (
                "Planned Sales Lift",
                "promotions",
                "planned_sales_lift_pct",
                "AVERAGE",
                "percentage",
                "Campaign sales-lift target",
            ),
            (
                "Actual Sales Lift",
                "promotions",
                "actual_sales_lift_pct",
                "AVERAGE",
                "percentage",
                "Measured campaign sales lift",
            ),
        ],
    )

    connection.execute(
        """
        CREATE TABLE dimension_catalog (
            dimension_name VARCHAR NOT NULL,
            source_dataset VARCHAR NOT NULL,
            source_column VARCHAR NOT NULL,
            hierarchy_name VARCHAR,
            hierarchy_level INTEGER,
            aliases VARCHAR,
            description VARCHAR NOT NULL
        )
        """
    )

    connection.executemany(
        """
        INSERT INTO dimension_catalog
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "Category",
                "products",
                "category",
                "Product",
                1,
                "product category",
                "Highest supported product classification",
            ),
            (
                "Brand",
                "products",
                "brand",
                "Product",
                2,
                "brand name",
                "FMCG brand",
            ),
            (
                "Product",
                "products",
                "product_name",
                "Product",
                3,
                "item, product name",
                "Commercial product name",
            ),
            (
                "SKU",
                "products",
                "sku_id",
                "Product",
                4,
                "sku code, item code",
                "Lowest supported product level",
            ),
            (
                "Country",
                "geography",
                "country",
                "Geography",
                1,
                "nation",
                "Country level",
            ),
            (
                "Region",
                "geography",
                "region",
                "Geography",
                2,
                "zone",
                "Sales region",
            ),
            (
                "State",
                "geography",
                "state",
                "Geography",
                3,
                "province",
                "State level",
            ),
            (
                "City",
                "geography",
                "city",
                "Geography",
                4,
                "market",
                "City level",
            ),
            (
                "Channel",
                "sales",
                "channel",
                "Sales",
                1,
                "GT, MT, EC, sales channel",
                "Sales channel",
            ),
            (
                "Distributor",
                "distributors",
                "distributor_name",
                "Sales",
                2,
                "distribution partner",
                "Distribution partner",
            ),
            (
                "Campaign",
                "promotions",
                "campaign_name",
                "Promotion",
                1,
                "promotion, initiative",
                "Marketing or trade campaign",
            ),
            (
                "Year",
                "vw_sales_enriched",
                "year",
                "Time",
                1,
                "financial year, calendar year",
                "Calendar year",
            ),
            (
                "Quarter",
                "vw_sales_enriched",
                "quarter",
                "Time",
                2,
                "Q1, Q2, Q3, Q4",
                "Calendar quarter",
            ),
            (
                "Month",
                "sales",
                "month",
                "Time",
                3,
                "period",
                "Calendar month",
            ),
        ],
    )


def validate_database(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Run database integrity and business-scenario checks."""

    table_counts = connection.execute(
        """
        SELECT 'products' AS table_name, COUNT(*) AS row_count
        FROM products

        UNION ALL

        SELECT 'geography', COUNT(*)
        FROM geography

        UNION ALL

        SELECT 'distributors', COUNT(*)
        FROM distributors

        UNION ALL

        SELECT 'sales', COUNT(*)
        FROM sales

        UNION ALL

        SELECT 'inventory', COUNT(*)
        FROM inventory

        UNION ALL

        SELECT 'promotions', COUNT(*)
        FROM promotions

        ORDER BY table_name
        """
    ).fetchall()

    for table_name, row_count in table_counts:
        if row_count == 0:
            raise ValueError(
                f"Validation failed: {table_name} is empty."
            )

    invalid_revenue_rows = connection.execute(
        """
        SELECT COUNT(*)
        FROM sales
        WHERE ABS(
            gross_revenue_inr
            - discount_inr
            - net_revenue_inr
        ) > 0.01
        """
    ).fetchone()[0]

    if invalid_revenue_rows:
        raise ValueError(
            "Validation failed: incorrect net-revenue values found."
        )

    invalid_margin_rows = connection.execute(
        """
        SELECT COUNT(*)
        FROM sales
        WHERE ABS(
            net_revenue_inr
            - cogs_inr
            - gross_margin_inr
        ) > 0.01
        """
    ).fetchone()[0]

    if invalid_margin_rows:
        raise ValueError(
            "Validation failed: incorrect gross-margin values found."
        )

    invalid_inventory_rows = connection.execute(
        """
        SELECT COUNT(*)
        FROM inventory
        WHERE closing_stock_units < 0
        """
    ).fetchone()[0]

    if invalid_inventory_rows:
        raise ValueError(
            "Validation failed: negative inventory values found."
        )

    campaign_count = connection.execute(
        """
        SELECT COUNT(*)
        FROM promotions
        WHERE campaign_name = 'Sparkle Summer 2025'
          AND actual_sales_lift_pct
              < planned_sales_lift_pct
        """
    ).fetchone()[0]

    if campaign_count == 0:
        raise ValueError(
            "Validation failed: Sparkle Summer scenario is missing."
        )

    critical_stockout_count = connection.execute(
        """
        SELECT COUNT(*)
        FROM vw_inventory_enriched
        WHERE sku_id = 'FG-DW-LEM-500'
          AND state = 'Karnataka'
          AND month BETWEEN DATE '2025-04-01'
                        AND DATE '2025-06-30'
          AND stockout_days >= 7
        """
    ).fetchone()[0]

    if critical_stockout_count == 0:
        raise ValueError(
            "Validation failed: campaign stockout scenario is missing."
        )


def print_database_summary(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Print useful database information after construction."""

    print("\nDatabase tables and views")
    print("-------------------------")

    objects = connection.execute(
        """
        SELECT
            table_name,
            table_type
        FROM information_schema.tables
        WHERE table_schema = 'main'
        ORDER BY table_type, table_name
        """
    ).fetchall()

    for object_name, object_type in objects:
        print(f"{object_type:<12} {object_name}")

    print("\nRow counts")
    print("----------")

    for table_name in CSV_FILES:
        row_count = connection.execute(
            f"SELECT COUNT(*) FROM {table_name}"
        ).fetchone()[0]

        print(f"{table_name:<15} {row_count:>8,}")

    minimum_month, maximum_month = connection.execute(
        """
        SELECT
            MIN(month),
            MAX(month)
        FROM sales
        """
    ).fetchone()

    print("\nAvailable sales period")
    print("----------------------")
    print(f"{minimum_month} to {maximum_month}")

    print("\nSparkle Summer 2025")
    print("-------------------")

    campaign_rows = connection.execute(
        """
        SELECT
            state,
            channel,
            planned_sales_lift_pct,
            actual_sales_lift_pct,
            lift_variance_pct_points
        FROM vw_promotions_enriched
        WHERE campaign_name = 'Sparkle Summer 2025'
        ORDER BY state
        """
    ).fetchall()

    for row in campaign_rows:
        print(row)


def main() -> None:
    validate_input_files()
    DATABASE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    if DATABASE_PATH.exists():
        DATABASE_PATH.unlink()

    connection = duckdb.connect(str(DATABASE_PATH))

    try:
        connection.execute("BEGIN TRANSACTION")

        for table_name, create_statement in TABLE_DEFINITIONS.items():
            connection.execute(create_statement)

            copy_csv_to_table(
                connection=connection,
                table_name=table_name,
                csv_path=CSV_DIRECTORY / CSV_FILES[table_name],
            )

        create_business_views(connection)
        create_metadata_catalogs(connection)
        validate_database(connection)

        connection.execute("COMMIT")
        print_database_summary(connection)

    except Exception:
        connection.execute("ROLLBACK")
        raise

    finally:
        connection.close()

    print(
        "\nDuckDB database created successfully at "
        f"{DATABASE_PATH.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()